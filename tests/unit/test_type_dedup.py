"""
Tests for type_dedup.py's hypernym/hyponym merge guard.

Real pipeline output (onto_artifacts/verified_groups.json) showed Step 3
silently merging general/specific pairs as if they were synonyms, e.g.
"journal" + "academic journal", "aircraft" + "military aircraft" -- the
specific/general distinction was destroyed before hierarchy induction ever
ran, and is unrecoverable once that happens. cluster_entity_types.txt was
fixed to apply an explicit "kind of" hyponymy test that takes priority over
context/relation overlap. These tests pin that behavior at the pipeline
level (via a scripted mock LLM verifier -- no real API calls) and guard the
prompt text itself against silent regression.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.ontodisco.type_dedup import deduplicate_types

# Deterministic label -> embedding lookup, avoiding a real Contriever
# download. Pairs within a family are near-identical (guaranteed to
# HAC-cluster together regardless of hac_threshold); families are mutually
# orthogonal (guaranteed not to cross-cluster).
LABEL_VECTORS = {
    "journal": [1.0, 0.0, 0.0, 0.0],
    "academic journal": [0.99, 0.02, 0.0, 0.0],
    "movie": [0.0, 1.0, 0.0, 0.0],
    "film": [0.0, 0.99, 0.02, 0.0],
}


class _FakeEmbedder:
    def __init__(self, model_name=None, device=None):
        pass

    def embed(self, texts, batch_size=64):
        vecs = np.array(
            [LABEL_VECTORS.get(t, [0.1, 0.1, 0.1, 0.1]) for t in texts],
            dtype=np.float32,
        )
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms


class _FakeLLMVerifier:
    """Returns a pre-scripted merge/split decision per cluster, keyed by its
    member set -- simulating exactly what the LLM is expected to decide under
    cluster_entity_types.txt's current rules, without a real API call."""

    def __init__(self, decisions: dict[frozenset, list[dict]]):
        self._decisions = decisions
        self.calls: list[list[str]] = []

    def verify_cluster_with_llm(self, members, surface_form_type="entity_type", member_context=None):
        self.calls.append(list(members))
        key = frozenset(m.lower() for m in members)
        return self._decisions.get(key, [{"canonical_label": members[0], "members": list(members)}])


def _canonical_label_by_surface_form(result) -> dict[str, str]:
    mapping = {}
    for ctype in result.types.values():
        for sf in ctype.surface_forms:
            mapping[sf] = ctype.canonical_label
    return mapping


def test_hypernym_types_are_not_merged():
    """'academic journal' and 'journal' get HAC-clustered together (they
    share almost all the same context relations), but are a general/specific
    pair, not synonyms. The LLM must split them per the "kind of" test, and
    deduplicate_types() must respect that split -- two distinct
    CanonicalTypes, not one merged type."""
    llm = _FakeLLMVerifier(decisions={
        frozenset({"journal", "academic journal"}): [
            {"canonical_label": "journal", "members": ["journal"]},
            {"canonical_label": "academic journal", "members": ["academic journal"]},
        ],
    })

    with patch("src.ontodisco.type_dedup.ContrieverEmbedder", _FakeEmbedder):
        result = deduplicate_types(
            raw_type_labels=["journal", "academic journal"],
            llm_extractor=llm,
            hac_threshold=0.8,
        )

    assert llm.calls, "the pair should have been HAC-clustered together and sent to the LLM"
    assert result.num_canonical_types == 2

    labels = _canonical_label_by_surface_form(result)
    assert labels["journal"] != labels["academic journal"], (
        "hypernym pair must remain two distinct types, not collapsed into one"
    )


def test_synonym_types_are_still_merged():
    """Contrasting regression: true synonyms ('movie' / 'film', no "kind of"
    relationship between them) must still merge into one CanonicalType --
    confirms the hyponymy guard didn't make Step 3 over-conservative."""
    llm = _FakeLLMVerifier(decisions={
        frozenset({"movie", "film"}): [
            {"canonical_label": "film", "members": ["movie", "film"]},
        ],
    })

    with patch("src.ontodisco.type_dedup.ContrieverEmbedder", _FakeEmbedder):
        result = deduplicate_types(
            raw_type_labels=["movie", "film"],
            llm_extractor=llm,
            hac_threshold=0.8,
        )

    assert llm.calls
    assert result.num_canonical_types == 1

    labels = _canonical_label_by_surface_form(result)
    assert labels["movie"] == labels["film"] == "film"


def test_hyponymy_guard_present_in_prompt():
    """Regression guard on the prompt text itself: cluster_entity_types.txt
    must keep the explicit "kind of" hyponymy test, so a future edit can't
    silently regress the fix without a test failure."""
    prompt_path = (
        Path(__file__).resolve().parents[2]
        / "src" / "ontodisco" / "utils" / "prompts" / "cluster_entity_types.txt"
    )
    text = prompt_path.read_text(encoding="utf-8").lower()

    assert "kind of" in text
    assert "documentary film" in text  # the worked example must survive edits
