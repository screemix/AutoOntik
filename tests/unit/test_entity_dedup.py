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


def test_parent_child_types_are_not_merged():
    """Same name ('Oppenheimer'), but one mention typed 'film' and the other
    'documentary film' (parent/child, not siblings). Name embeddings are
    near-identical, so a plain FAISS-NN pass would merge them -- the parent
    gate must reject this pair anyway: only same-type or same-parent-sibling
    mentions may merge."""
    triplets = [_mention("Oppenheimer", "film"), _mention("Oppenheimer", "documentary film")]
    llm = _FakeLLMVerifier()

    with patch("src.ontodisco.entity_dedup.ContrieverEmbedder", _FakeEmbedder):
        result = deduplicate_entities(
            triplets=triplets, type_vocab=_type_vocab(), hierarchy=_hierarchy(),
            llm_extractor=llm, similarity_threshold=0.85,
        )

    assert not llm.calls, "parent/child pair must be gated out before ever reaching the LLM"
    assert result.num_canonical_entities == 2


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
