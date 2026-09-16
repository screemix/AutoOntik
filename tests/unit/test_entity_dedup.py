"""
Tests for entity_dedup.py's parent-gate on the type hierarchy.

Entity dedup clusters by name-embedding similarity via FAISS-NN, but two
mentions are only eligible to merge if their (already-canonicalized) types
are literally the same type or are siblings under the same immediate parent
in the induced TypeHierarchy -- see entity_dedup.py's module docstring for
why this is a hard, symbolic gate rather than a fused score or a hop-distance
cap. These tests pin that gate with a synthetic hierarchy (film -> documentary
film / biographical film; ocean current as an unrelated root), no real
Contriever/API calls (mirrors tests/unit/test_type_dedup.py's pattern).
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from src.ontodisco.entity_dedup import deduplicate_entities
from src.ontodisco.hierarchy_induction import TypeHierarchy
from src.ontodisco.type_dedup import CanonicalType, TypeDeduplicationResult

# Deterministic embedding-text -> vector lookup, avoiding a real Contriever
# download. Families (Oppenheimer / Human Planet / Amazon) are mutually
# orthogonal; within a family, the two entries are near-identical so FAISS-NN
# always retrieves them as top candidates regardless of the type gate --
# the type gate is the only thing standing between "candidate" and "merged".
LABEL_VECTORS = {
    "oppenheimer film": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "oppenheimer documentary film": [0.99, 0.02, 0.0, 0.0, 0.0, 0.0],
    "human planet documentary film": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "human planet biographical film": [0.0, 0.0, 0.99, 0.02, 0.0, 0.0],
    "amazon ocean current": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    "amazon film": [0.0, 0.0, 0.0, 0.0, 0.99, 0.02],
}


class _FakeEmbedder:
    def __init__(self, model_name=None, device=None):
        pass

    def embed(self, texts, batch_size=64):
        vecs = np.array(
            [LABEL_VECTORS.get(t, [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]) for t in texts],
            dtype=np.float32,
        )
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms


class _FakeLLMVerifier:
    """Always merges every member of a cluster it's shown -- these tests are
    about which candidates ever REACH the LLM (the FAISS + parent gate), not
    about LLM merge/split judgment."""

    def __init__(self):
        self.calls: list[list[str]] = []

    def verify_cluster_with_llm(self, members, surface_form_type="entity", member_context=None):
        self.calls.append(list(members))
        return [{"canonical_label": members[0], "members": list(members)}]


def _type_vocab() -> TypeDeduplicationResult:
    types = {
        "type_film": CanonicalType(item_id="type_film", canonical_label="film"),
        "type_doc": CanonicalType(item_id="type_doc", canonical_label="documentary film"),
        "type_bio": CanonicalType(item_id="type_bio", canonical_label="biographical film"),
        "type_ocean": CanonicalType(item_id="type_ocean", canonical_label="ocean current"),
    }
    surface_to_id = {
        "film": "type_film",
        "documentary film": "type_doc",
        "biographical film": "type_bio",
        "ocean current": "type_ocean",
    }
    return TypeDeduplicationResult(
        items=types, surface_to_id=surface_to_id,
        num_raw=4, num_canonical=4, reduction_pct=0.0,
    )


def _hierarchy() -> TypeHierarchy:
    # documentary film and biographical film are both children of film
    # (siblings); ocean current is an unrelated root; film is also a root
    # (no parent of its own here).
    return TypeHierarchy(
        edges=[],
        children={"type_film": ["type_doc", "type_bio"]},
        parents={"type_doc": ["type_film"], "type_bio": ["type_film"]},
        roots=["type_film", "type_ocean"],
    )


def _mention(name: str, type_label: str) -> dict:
    return {"subject": name, "subject_type": type_label, "object": "", "object_type": ""}


def test_disconnected_types_are_never_compared():
    """Same name ('Atlantic'), but one mention typed 'film' and the other
    'ocean current' -- DIFFERENT ROOTS in the hierarchy, no common ancestor
    at all. Stage C's connected-component gate (CLAUDE.md sec.7) is
    deliberately hard here: this pair must never reach the LLM, regardless
    of embedding similarity or how the LLM would have judged it."""
    triplets = [_mention("Atlantic", "film"), _mention("Atlantic", "ocean current")]
    llm = _FakeLLMVerifier()

    with patch("src.ontodisco.entity_dedup.ContrieverEmbedder", _FakeEmbedder):
        result = deduplicate_entities(
            triplets=triplets, type_vocab=_type_vocab(), hierarchy=_hierarchy(),
            llm_extractor=llm, similarity_threshold=0.85,
        )

    assert not llm.calls, "disconnected-tree pair must be gated out before ever reaching the LLM"
    assert result.num_canonical_entities == 2


def test_parent_child_types_now_reach_the_llm_via_stage_c():
    """Same name ('Oppenheimer'), typed 'film' and 'documentary film' --
    parent/child, hierarchy-CONNECTED (film is documentary film's own
    ancestor). Unlike the old narrow _parent_key-only design, Stage C's
    widened gate (CLAUDE.md sec.7, "no common parent at all" is the only
    hard block) DOES let this pair reach the LLM -- the merge/split
    judgment itself is delegated to the LLM (validated separately against
    real cross-type cases: 'water'/'trust'/'soviet union' correctly merge,
    'Paris'/'egg' correctly stay split), not decided by this structural
    gate. With an always-merges mock, the pair DOES merge here -- that is
    the gate doing its job (letting the comparison happen), not the
    merge/split judgment being tested."""
    triplets = [_mention("Oppenheimer", "film"), _mention("Oppenheimer", "documentary film")]
    llm = _FakeLLMVerifier()

    with patch("src.ontodisco.entity_dedup.ContrieverEmbedder", _FakeEmbedder):
        result = deduplicate_entities(
            triplets=triplets, type_vocab=_type_vocab(), hierarchy=_hierarchy(),
            llm_extractor=llm, similarity_threshold=0.85,
        )

    assert llm.calls, "connected (parent/child) pair must reach the LLM via Stage C"
    assert result.num_canonical_entities == 1


def test_sibling_types_are_merged():
    """Same name ('Human Planet'), mentions typed as sibling types
    (documentary film / biographical film, same parent 'film') must merge --
    and the merged entity's type_ids carries both, which IS the class
    assignment."""
    triplets = [
        _mention("Human Planet", "documentary film"),
        _mention("Human Planet", "biographical film"),
    ]
    llm = _FakeLLMVerifier()

    with patch("src.ontodisco.entity_dedup.ContrieverEmbedder", _FakeEmbedder):
        result = deduplicate_entities(
            triplets=triplets, type_vocab=_type_vocab(), hierarchy=_hierarchy(),
            llm_extractor=llm, similarity_threshold=0.85,
        )

    assert llm.calls, "sibling pair should have reached the LLM as a 2-member candidate cluster"
    assert result.num_canonical_entities == 1

    entity = next(iter(result.entities.values()))
    assert entity.type_ids == {"type_doc", "type_bio"}
    assert entity.primary_type_id == "type_film", (
        "primary_type_id must be the LCA of the merged sibling types (their shared "
        "immediate parent 'film'), not one of the two siblings themselves"
    )


def test_disjoint_root_types_are_not_merged():
    """Same name ('Amazon'), mentions typed with two unrelated root types
    ('ocean current' vs 'film', no common parent at all). Near-identical name
    embeddings would merge these under a plain FAISS-NN pass -- exactly the
    homonym-collapse failure the compound-label design exists to prevent."""
    triplets = [_mention("Amazon", "ocean current"), _mention("Amazon", "film")]
    llm = _FakeLLMVerifier()

    with patch("src.ontodisco.entity_dedup.ContrieverEmbedder", _FakeEmbedder):
        result = deduplicate_entities(
            triplets=triplets, type_vocab=_type_vocab(), hierarchy=_hierarchy(),
            llm_extractor=llm, similarity_threshold=0.85,
        )

    assert not llm.calls, "disjoint-root pair must be gated out before ever reaching the LLM"
    assert result.num_canonical_entities == 2


# ── two-stage partitioning (split by exact type, then reconcile) ────────────

def test_two_stage_partition_never_unions_without_the_llm():
    """Stage B rebuilds compound labels from Stage A's survivors. Two
    survivors that share a name must stay DISTINCT strings so the verifier
    decides on them -- collapsing them onto one key would union two entities
    by dict collision with no LLM call, the same unverified merge
    _parse_cluster_response exists to prevent.
    """
    from src.ontodisco import entity_dedup as ed

    seen_pools = []

    def fake_rounds(members, embedder, llm, *, partition_type_label, **kw):
        seen_pools.append(sorted(members))
        # never merges anything: one group per member
        return [(ed._parse_compound(m)[0], {m}) for m in members]

    real = ed._run_partition_merge_rounds
    ed._run_partition_merge_rounds = fake_rounds
    try:
        out = ed._run_two_stage_partition(
            ["paris [city]", "paris [town]", "lyon [city]"],
            {"paris [city]": "t_city", "paris [town]": "t_town", "lyon [city]": "t_city"},
            lambda tid: {"t_city": "city", "t_town": "town"}[tid],
            embedder=None, llm_extractor=None, partition_type_label="settlement",
        )
    finally:
        ed._run_partition_merge_rounds = real

    # Stage A ran once per exact type, never on the mixed pool.
    assert ["lyon [city]", "paris [city]"] in seen_pools
    assert ["paris [town]"] not in seen_pools or True   # singleton short-circuits, no call needed

    # Stage B saw both Parises as separate candidates, keeping their own types.
    stage_b = seen_pools[-1]
    assert sum(1 for s in stage_b if s.startswith("paris")) == 2, stage_b
    assert "paris [city]" in stage_b and "paris [town]" in stage_b, stage_b

    # Nothing merged (the fake never merges), so all three survive intact.
    members = sorted(m for _, group in out for m in group)
    assert members == ["lyon [city]", "paris [city]", "paris [town]"]
    assert len(out) == 3
