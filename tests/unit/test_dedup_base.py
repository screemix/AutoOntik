"""
Tests for dedup_base.py's parallel LLM cluster verification and the
round-based convergence loop (deduplicate_with_rounds).

No real Contriever/API calls -- mirrors the existing test files' pattern of
small, deterministic fakes.
"""

from __future__ import annotations

import time

import numpy as np

from src.ontodisco.utils.dedup_base import (
    deduplicate_with_rounds,
    verify_clusters_with_llm,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  verify_clusters_with_llm: parallel execution
# ═══════════════════════════════════════════════════════════════════════════════

class _SlowLLMVerifier:
    """Merges every cluster it's shown, but sleeps a per-cluster-configured
    amount first -- used to force completion order to differ from submission
    order, so tests can check verified_groups' order doesn't depend on it."""

    def __init__(self, delays: dict[frozenset, float]):
        self._delays = delays
        self.calls: list[list[str]] = []

    def verify_cluster_with_llm(self, members, surface_form_type="entity_type", member_context=None):
        self.calls.append(list(members))
        delay = self._delays.get(frozenset(members), 0.0)
        time.sleep(delay)
        return [{"canonical_label": members[0], "members": list(members)}]


def test_verify_clusters_with_llm_output_order_is_deterministic():
    """Cluster 0 is the slowest, cluster 2 the fastest -- completion order is
    therefore the REVERSE of submission order. verified_groups must still
    come back in original cluster order, since downstream code assigns item
    IDs via enumerate(verified_groups)."""
    clusters = {
        0: ["a1", "a2"],
        1: ["b1", "b2"],
        2: ["c1", "c2"],
    }
    llm = _SlowLLMVerifier(delays={
        frozenset({"a1", "a2"}): 0.06,
        frozenset({"b1", "b2"}): 0.03,
        frozenset({"c1", "c2"}): 0.0,
    })

    verified_groups = verify_clusters_with_llm(clusters, llm, verbose=False, max_workers=4)

    assert [g[0] for g in verified_groups] == ["a1", "b1", "c1"], (
        "output order must follow original cluster order, not completion timing"
    )


class _FlakyLLMVerifier:
    """Raises for one specific cluster, merges every other cluster it sees."""

    def __init__(self, fails_on: frozenset):
        self._fails_on = fails_on
        self.calls: list[list[str]] = []

    def verify_cluster_with_llm(self, members, surface_form_type="entity_type", member_context=None):
        self.calls.append(list(members))
        if frozenset(members) == self._fails_on:
            raise RuntimeError("simulated LLM failure")
        return [{"canonical_label": members[0], "members": list(members)}]


def test_verify_clusters_with_llm_isolates_exceptions_per_cluster():
    """One cluster's LLM call failing must not affect any other cluster's
    result. The failing cluster falls back to SINGLETONS -- never a wholesale
    merge of its unverified HDBSCAN grouping: HDBSCAN is candidate generation,
    and only the LLM call decides identity, so a failed call means "unknown"
    and the safe direction is split (see _resolve_cluster_groups)."""
    clusters = {
        0: ["x1", "x2"],
        1: ["y1", "y2", "y3"],
    }
    llm = _FlakyLLMVerifier(fails_on=frozenset({"y1", "y2", "y3"}))

    verified_groups = verify_clusters_with_llm(clusters, llm, verbose=False, max_workers=4)

    by_canonical = {g[0]: g[1] for g in verified_groups}
    assert by_canonical["x1"] == ["x1", "x2"], "unaffected cluster must still verify normally"
    assert by_canonical["y1"] == ["y1"], (
        "a cluster whose LLM verification failed must fall back to singletons, "
        "not be merged wholesale on unverified embedding proximity"
    )
    assert by_canonical["y2"] == ["y2"]
    assert by_canonical["y3"] == ["y3"]
    assert {"y1", "y2", "y3"} <= set(by_canonical), (
        "no member of a failed cluster may be dropped -- each survives as its own entity"
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  verify_clusters_with_llm: max_cluster_size split + stitch
# ═══════════════════════════════════════════════════════════════════════════════

class _NoMergeLLMVerifier:
    """Never merges anything -- every member comes back as its own singleton
    group. Used to isolate the split/stitch bookkeeping from any actual
    merge decision."""

    def __init__(self):
        self.calls: list[list[str]] = []

    def verify_cluster_with_llm(self, members, surface_form_type="entity_type", member_context=None):
        self.calls.append(list(members))
        return [{"canonical_label": m, "members": [m]} for m in members]


def test_oversized_cluster_split_never_loses_a_trailing_singleton_chunk():
    """5 members with max_cluster_size=2 split into chunks of 2, 2, 1. The
    trailing chunk of 1 never gets its own LLM call (it's a singleton), but
    it must still surface in the final output -- not be silently dropped."""
    members = ["a", "b", "c", "d", "e"]
    llm = _NoMergeLLMVerifier()

    verified_groups = verify_clusters_with_llm(
        {0: members}, llm, verbose=False, max_cluster_size=2,
    )

    recovered = sorted(m for _, group in verified_groups for m in group)
    assert recovered == members, "no member may be lost when a trailing chunk is a singleton"


def test_oversized_cluster_split_exactly_one_full_chunk_plus_trailing_singleton():
    """41 members with max_cluster_size=40 split into chunks of 40 and 1 --
    only ONE chunk goes through verify_cluster_with_llm, so the naive 'was
    this cluster split?' check (num LLM-bearing work items > 1) would say
    no and skip stitching/folding-in entirely, dropping the last member."""
    members = [f"m{i}" for i in range(41)]
    llm = _NoMergeLLMVerifier()

    verified_groups = verify_clusters_with_llm(
        {0: members}, llm, verbose=False, max_cluster_size=40,
    )

    recovered = sorted(m for _, group in verified_groups for m in group)
    assert recovered == sorted(members)


def test_oversized_cluster_stitches_sub_batches_split_only_by_size_cap():
    """4 members, max_cluster_size=2 -> two sub-batches of 2. Each sub-batch
    merges internally; a stitching call must then be given the chance to
    reunite the two sub-batch canonical labels, since they were only ever
    split apart by the size cap."""
    members = ["alpha1", "alpha2", "alpha3", "alpha4"]

    class _StitchingVerifier:
        def __init__(self):
            self.calls: list[list[str]] = []

        def verify_cluster_with_llm(self, members, surface_form_type="entity_type", member_context=None):
            self.calls.append(list(members))
            if set(members) == {"alpha1", "alpha2"}:
                return [{"canonical_label": "alpha_a", "members": ["alpha1", "alpha2"]}]
            if set(members) == {"alpha3", "alpha4"}:
                return [{"canonical_label": "alpha_b", "members": ["alpha3", "alpha4"]}]
            # stitching call: reunite the two sub-batch canonical labels
            if set(members) == {"alpha_a", "alpha_b"}:
                return [{"canonical_label": "alpha_a", "members": ["alpha_a", "alpha_b"]}]
            raise AssertionError(f"unexpected call: {members}")

    llm = _StitchingVerifier()
    verified_groups = verify_clusters_with_llm(
        {0: members}, llm, verbose=False, max_cluster_size=2,
    )

    assert len(verified_groups) == 1, "stitching should have reunited both sub-batches into one group"
    canonical, group = verified_groups[0]
    assert sorted(group) == sorted(members)
    assert len(llm.calls) == 3, "two sub-batch calls plus one stitching call"


def test_llm_assigning_a_member_to_two_groups_does_not_duplicate_it():
    """A malformed response can list the same member under two different
    groups. The first group to claim it must win; the duplicate must be
    dropped, not counted twice -- otherwise a cluster's total output member
    count could exceed its own input size."""
    class _DuplicatingVerifier:
        def verify_cluster_with_llm(self, members, surface_form_type="entity_type", member_context=None):
            return [
                {"canonical_label": "g1", "members": ["x1", "x2"]},
                {"canonical_label": "g2", "members": ["x2", "x3"]},
            ]

    members = ["x1", "x2", "x3"]
    verified_groups = verify_clusters_with_llm(
        {0: members}, _DuplicatingVerifier(), verbose=False,
    )

    recovered = [m for _, group in verified_groups for m in group]
    assert sorted(recovered) == members, "no member may be counted twice across groups"
    assert recovered.count("x2") == 1


def test_duplicated_label_in_stitching_response_does_not_duplicate_a_whole_sub_batch():
    """The same duplicate-assignment bug is more consequential once it hits
    the stitching call: a duplicated sub-batch label there would duplicate
    every real member of that sub-batch, not just one label."""
    class _DuplicatingStitchVerifier:
        def verify_cluster_with_llm(self, members, surface_form_type="entity_type", member_context=None):
            if set(members) == {"a1", "a2"}:
                return [{"canonical_label": "batch_a", "members": ["a1", "a2"]}]
            if set(members) == {"b1", "b2"}:
                return [{"canonical_label": "batch_b", "members": ["b1", "b2"]}]
            # malformed stitch response: "batch_a" claimed by two groups
            return [
                {"canonical_label": "batch_a", "members": ["batch_a", "batch_b"]},
                {"canonical_label": "batch_a_dup", "members": ["batch_a"]},
            ]

    members = ["a1", "a2", "b1", "b2"]
    verified_groups = verify_clusters_with_llm(
        {0: members}, _DuplicatingStitchVerifier(), verbose=False, max_cluster_size=2,
    )

    recovered = [m for _, group in verified_groups for m in group]
    assert sorted(recovered) == sorted(members), (
        "a duplicated sub-batch label in the stitching response must not "
        "duplicate that whole sub-batch's real members"
    )


def test_stitch_pool_over_max_cluster_size_skips_stitching_instead_of_recursing():
    """A cluster oversized enough that its sub-batches still survive as more
    than max_cluster_size labels (i.e. the LLM barely merged anything within
    any sub-batch) must NOT attempt one giant stitch call, and must not
    recurse unboundedly either -- it should terminate immediately, keeping
    the sub-batch groups separate, with every member still recovered."""
    members = [f"m{i:04d}" for i in range(500)]
    llm = _NoMergeLLMVerifier()

    verified_groups = verify_clusters_with_llm(
        {1394: members}, llm, verbose=False, max_cluster_size=40,
    )

    recovered = sorted(m for _, group in verified_groups for m in group)
    assert recovered == sorted(members)
    # 13 sub-batch calls (ceil(500/40)), no stitching call on top
    assert len(llm.calls) == 13
    assert all(len(call) <= 40 for call in llm.calls), "no call may exceed max_cluster_size"


def test_cluster_at_or_under_max_cluster_size_makes_a_single_call_no_stitch():
    """A cluster that already fits within max_cluster_size must not be split
    at all -- one LLM call, no stitching call."""
    members = ["x1", "x2", "x3"]
    llm = _NoMergeLLMVerifier()

    verify_clusters_with_llm({0: members}, llm, verbose=False, max_cluster_size=40)

    assert llm.calls == [members], "no split should happen when the cluster already fits the cap"


# ═══════════════════════════════════════════════════════════════════════════════
#  deduplicate_with_rounds: convergence loop
# ═══════════════════════════════════════════════════════════════════════════════

# "a"/"b" are near-identical (round 1 merges them); "c" is orthogonal to both
# in round 1 (so it stays a singleton), but the LABEL the fake LLM picks for
# the a+b merge ("ab") is deliberately embedded identically to "c" -- so only
# round 2, re-embedding the SURVIVING canonical labels, can discover that
# merge. This is the same "merge in round N unlocks a merge in round N+1"
# shape as entity_dedup's own motivating example (Chris Nolan / C. Nolan /
# Christopher J. Nolan), at the plain-label level dedup_base operates on.
LABEL_VECTORS = {
    "a": [1.0, 0.0, 0.0, 0.0],
    "b": [0.99, 0.02, 0.0, 0.0],
    "ab": [0.0, 0.0, 1.0, 0.0],
    "c": [0.0, 0.0, 0.99, 0.02],
}


class _FakeEmbedder:
    def embed(self, texts, batch_size=64):
        vecs = np.array(
            [LABEL_VECTORS.get(t, [0.5, 0.5, 0.5, 0.5]) for t in texts],
            dtype=np.float32,
        )
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms


class _ScriptedLLMVerifier:
    def __init__(self, decisions: dict[frozenset, list[dict]]):
        self._decisions = decisions
        self.calls: list[list[str]] = []

    def verify_cluster_with_llm(self, members, surface_form_type="entity_type", member_context=None):
        self.calls.append(list(members))
        key = frozenset(m.lower() for m in members)
        return self._decisions.get(key, [{"canonical_label": members[0], "members": list(members)}])


def test_round_based_convergence_catches_merge_missed_in_round_one():
    """With max_rounds=1 (single pass, deduplicate()'s own behavior),
    "ab" and "c" never get compared -- they only look alike once "ab"
    exists as its own pool entry. max_rounds>=2 must catch it."""
    llm = _ScriptedLLMVerifier(decisions={
        frozenset({"a", "b"}): [{"canonical_label": "ab", "members": ["a", "b"]}],
        frozenset({"ab", "c"}): [{"canonical_label": "abc", "members": ["ab", "c"]}],
    })

    result_single_pass = deduplicate_with_rounds(
        all_labels=["a", "b", "c"],
        embedder=_FakeEmbedder(),
        llm_verifier=llm,
        hac_threshold=0.9,
        id_prefix="item",
        max_rounds=1,
    )
    assert result_single_pass.num_canonical == 2, (
        "single pass should only merge a+b, leaving c and 'ab' as two separate items"
    )

    llm2 = _ScriptedLLMVerifier(decisions={
        frozenset({"a", "b"}): [{"canonical_label": "ab", "members": ["a", "b"]}],
        frozenset({"ab", "c"}): [{"canonical_label": "abc", "members": ["ab", "c"]}],
    })
    result_rounds = deduplicate_with_rounds(
        all_labels=["a", "b", "c"],
        embedder=_FakeEmbedder(),
        llm_verifier=llm2,
        hac_threshold=0.9,
        id_prefix="item",
        max_rounds=5,
    )
    assert result_rounds.num_canonical == 1, (
        "round 2 must re-embed 'ab' and discover it belongs with 'c', converging to one item"
    )
    item = next(iter(result_rounds.items.values()))
    assert item.canonical_label == "abc"
    assert set(item.surface_forms) == {"a", "b", "c"}


def test_round_based_convergence_stops_without_extra_llm_calls_once_stable():
    """Once nothing merges further, no extra round should fire -- confirms
    the stop condition is a real pool-size check, not just "run max_rounds
    every time"."""
    llm = _ScriptedLLMVerifier(decisions={
        frozenset({"a", "b"}): [{"canonical_label": "ab", "members": ["a", "b"]}],
    })
    # "c" is orthogonal to "ab" too in this variant (reuse LABEL_VECTORS but
    # drop "ab"'s special placement by overriding it back to orthogonal).
    class _NoFurtherMergeEmbedder(_FakeEmbedder):
        def embed(self, texts, batch_size=64):
            overrides = dict(LABEL_VECTORS)
            overrides["ab"] = [0.0, 1.0, 0.0, 0.0]  # orthogonal to everything else now
            vecs = np.array(
                [overrides.get(t, [0.5, 0.5, 0.5, 0.5]) for t in texts], dtype=np.float32,
            )
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return vecs / norms

    result = deduplicate_with_rounds(
        all_labels=["a", "b", "c"],
        embedder=_NoFurtherMergeEmbedder(),
        llm_verifier=llm,
        hac_threshold=0.9,
        id_prefix="item",
        max_rounds=5,
    )
    assert result.num_canonical == 2
    # Only the a+b cluster should ever have reached the LLM -- round 2 (and
    # any later round) must never fire once "ab" vs "c" both come back as
    # all-singleton clusters.
    assert llm.calls == [["a", "b"]]
