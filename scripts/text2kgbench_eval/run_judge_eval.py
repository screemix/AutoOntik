"""
LLM-as-a-judge evaluation against Text2KGBench gold triples + ontology
============================================================================

Standalone (only needs pickle + the openai client via LLMTripletExtractor --
no torch/sklearn import required, unlike the ontology pipeline itself), so it
can be re-run against an already-built pipeline output any number of times,
e.g. to try a different --judge-model, without re-running the (expensive,
LLM-heavy) ontology pipeline.

The judge model is DELIBERATELY independent of whatever model built the KG
(--judge-model/--judge-base-url/--judge-api-key-env, not the pipeline's own
--config) -- comparing qwen-built vs gpt-oss-built KGs against a fixed
third-party judge (e.g. gpt-4o) avoids a model grading its own homework, and
keeps the judge constant across every run being compared. Mirrors
scripts/mine_benchmark/run_judge_eval.py's --judge-model/--judge-base-url/
--judge-api-key-env convention.

Reads:
  - triplets.jsonl written by extract_triplets.py (each row tagged with
    source_sentence_id AND source_ontology_id).
  - relation_dedup.pkl / type_dedup.pkl checkpoints written by
    run_pipeline_text2kgbench.py, to resolve every raw extracted triplet to
    its canonical relation_id / type labels (surface_to_id lookups -- same
    pattern constraints.py's own Stage 0 uses). Hierarchy/entity checkpoints
    aren't needed: this eval scores relation and type canonicalization, not
    hierarchy structure or entity resolution.
  - ground_truth.jsonl + ontology.ttl for ONE OR MORE domains (--ontology-ids)
    under --data-dir/<ontology-id>/. Text2KGBench's ground truth carries no
    per-triple types (see ontology.py's docstring), so domain/range for the
    type judge always come from the RIGHT domain's ontology file -- tracked
    per sentence via source_ontology_id, not assumed to be a single ontology
    for the whole run. This is what makes it possible to score a run of
    run_pipeline_text2kgbench.py over a MULTI-DOMAIN COMBINED corpus (one
    shared KG spanning several Text2KGBench domains, see
    run_multi_domain.sh's "combined" mode) as well as a single-domain one --
    passing one --ontology-ids value reduces to the single-domain case.

Reports two levels of aggregation: "overall" (every domain pooled together)
and "by_domain" (one summary per --ontology-ids value, computed by filtering
results to that domain's sentences). For a single-domain run these are
mostly redundant (by_domain has one entry matching overall); for a
multi-domain COMBINED run, by_domain is what lets you compare directly
against N separate single-domain runs' own summaries (run_multi_domain.sh's
compare_domain_reports.py does exactly that comparison).

Matching algorithm, per sentence (see CLAUDE.md's eval design discussion for
the full reasoning trail):
  1. Iterate from the GOLD side, not the generated side: for each gold
     triple, ask the LLM to pick its match (if any) from the REMAINING
     (not yet consumed) bucket of generated triples for that sentence, via
     LLMTripletExtractor.match_triple_with_llm(). Once a candidate is
     matched it is removed from the bucket (greedy consumption -- accepted
     as an approximation given how small each sentence's bucket is, not
     worth a real bipartite assignment solver).
  2. A gold triple with no match is a genuine recall miss. A generated
     triple left over in the bucket after all gold triples are processed is
     reported as an "extra", NOT scored as a false positive -- Text2KGBench's
     gold ontology covers a small curated relation vocabulary (~15-20
     relations per domain), so a schema-free pipeline can correctly extract
     true facts the gold ontology simply doesn't cover.
  3. For every matched pair, compare our resolved subject_type/object_type
     against the gold relation's domain/range (from the ontology file) via
     LLMTripletExtractor.compare_types_with_llm(), swapping which side
     compares to which when the match was "inverse".

Error-handling discipline: a judge call that raises (auth/rate-limit/network
failure) is NEVER folded into a "no match"/"not_related" judgment -- that's
exactly the failure mode scripts/mine_benchmark/run_judge_eval.py's docstring
documents from a real incident (a 402 Insufficient Credits error on every
call silently became a bogus 0.0% accuracy). Call failures are tracked
separately and excluded from recall, with a loud warning if they're a large
fraction of the total.

Usage:
    # Single domain, judged by gpt-4o via OpenRouter regardless of what built the KG
    python -m scripts.text2kgbench_eval.run_judge_eval \\
        --triplets data/text2kgbench/wikidata_tekgen/ont_2_music/triplets.jsonl \\
        --data-dir data/text2kgbench/wikidata_tekgen \\
        --ontology-ids ont_2_music \\
        --output-dir output/text2kgbench/wikidata_tekgen/ont_2_music \\
        --output output/text2kgbench/wikidata_tekgen/ont_2_music/judge_results.json \\
        --judge-model openai/gpt-4o --judge-base-url https://openrouter.ai/api/v1 --judge-api-key-env OPENROUTER_KEY

    # Multiple domains scored against one combined pipeline run
    python -m scripts.text2kgbench_eval.run_judge_eval \\
        --triplets data/text2kgbench/wikidata_tekgen/_combined/triplets.jsonl \\
        --data-dir data/text2kgbench/wikidata_tekgen \\
        --ontology-ids ont_2_music ont_1_movie \\
        --output-dir output/text2kgbench/wikidata_tekgen/_combined \\
        --output output/text2kgbench/wikidata_tekgen/_combined/judge_results.json \\
        --judge-model openai/gpt-4o --judge-base-url https://openrouter.ai/api/v1 --judge-api-key-env OPENROUTER_KEY
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.ontodisco.utils.dedup_base import normalize_label
from src.ontodisco.utils.openai_utils import LLMTripletExtractor

from scripts.text2kgbench_eval.ontology import Text2KGOntology, parse_ontology

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_RUN_DIR_RE = re.compile(r"^run_(\d+)$")
TYPE_JUDGE_LABELS = ("exact", "more_general", "more_narrow", "not_related")


# ═══════════════════════════════════════════════════════════════════════════════
#  Loading
# ═══════════════════════════════════════════════════════════════════════════════

def _latest_run_dir(output_dir: Path) -> Path:
    checkpoints_root = output_dir / "checkpoints"
    numbers = []
    if checkpoints_root.exists():
        for child in checkpoints_root.iterdir():
            m = _RUN_DIR_RE.match(child.name)
            if child.is_dir() and m:
                numbers.append(int(m.group(1)))
    if not numbers:
        raise FileNotFoundError(f"No run_<n> checkpoint directory found under {checkpoints_root}")
    return checkpoints_root / f"run_{max(numbers)}"


def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _resolve_type_label(raw_type: str, type_vocab) -> str | None:
    type_id = type_vocab.surface_to_id.get(normalize_label(raw_type)) or type_vocab.surface_to_id.get(raw_type)
    if not type_id:
        return None
    return type_vocab.items[type_id].canonical_label


def _resolve_relation_label(raw_relation: str, relation_vocab) -> tuple[str | None, str | None]:
    relation_id = relation_vocab.surface_to_id.get(raw_relation) or \
        relation_vocab.surface_to_id.get(normalize_label(raw_relation))
    if not relation_id:
        return None, None
    return relation_id, relation_vocab.items[relation_id].canonical_label


def build_sentence_buckets(triplets: list[dict], relation_vocab, type_vocab) -> dict[str, list[dict]]:
    """Resolve every raw extracted triplet to its canonical relation/type
    labels and group by source_sentence_id. Triplets missing that tag, or
    that never resolved to a canonical id (extraction noise that didn't
    survive canonicalization), are dropped and counted."""
    buckets: dict[str, list[dict]] = defaultdict(list)
    n_dropped = 0
    for t in triplets:
        sentence_id = t.get("source_sentence_id")
        raw_subject = t.get("subject", "").strip()
        raw_relation = t.get("relation", "").strip()
        raw_object = t.get("object", "").strip()
        raw_subject_type = t.get("subject_type", "").strip()
        raw_object_type = t.get("object_type", "").strip()
        if not (sentence_id and raw_subject and raw_relation and raw_object):
            n_dropped += 1
            continue

        relation_id, relation_label = _resolve_relation_label(raw_relation, relation_vocab)
        subject_type_label = _resolve_type_label(raw_subject_type, type_vocab) if raw_subject_type else None
        object_type_label = _resolve_type_label(raw_object_type, type_vocab) if raw_object_type else None

        buckets[sentence_id].append({
            "subject": raw_subject, "relation": raw_relation, "object": raw_object,
            "relation_id": relation_id, "relation_label": relation_label,
            "subject_type_label": subject_type_label, "object_type_label": object_type_label,
        })

    if n_dropped:
        logger.warning("%d/%d generated triplets dropped (missing sentence id, or empty subject/relation/object)",
                        n_dropped, len(triplets))
    return dict(buckets)


# ═══════════════════════════════════════════════════════════════════════════════
#  Per-sentence matching + type judging
# ═══════════════════════════════════════════════════════════════════════════════

def _judge_sentence(
    record: dict, bucket: list[dict], ontology: Text2KGOntology, extractor: LLMTripletExtractor,
) -> dict:
    """Match every gold triple for one sentence against the (mutable) bucket
    of generated triples for that sentence, consuming a candidate once
    matched, then type-judge each matched pair's domain/range against the
    gold ontology. See module docstring for the full algorithm and why
    iteration starts from the gold side."""
    remaining = list(bucket)
    gold_results = []
    call_failures = 0

    for gold in record.get("triples", []):
        if not remaining:
            gold_results.append({"gold": gold, "matched": False, "reason": "no_candidates_left"})
            continue

        candidates = [(e["subject"], e["relation"], e["object"]) for e in remaining]
        try:
            match_resp = extractor.match_triple_with_llm(gold["sub"], gold["rel"], gold["obj"], candidates)
        except Exception:
            logger.exception("match_triple_with_llm failed for sentence %s, gold triple %r", record["id"], gold)
            call_failures += 1
            gold_results.append({"gold": gold, "matched": False, "reason": "judge_call_failed"})
            continue

        idx = match_resp.get("match")
        if idx is None:
            gold_results.append({"gold": gold, "matched": False, "reason": "no_match"})
            continue

        direction = match_resp.get("direction", "same")
        matched_entry = remaining.pop(idx)

        # ── Type judge for this matched pair ──────────────────────────────
        gold_rel_info = ontology.relations.get(gold["rel"], {})
        gold_domain = gold_rel_info.get("domain")
        gold_range = gold_rel_info.get("range")

        if direction == "inverse":
            our_domain_side = matched_entry["object_type_label"]
            our_range_side = matched_entry["subject_type_label"]
        else:
            our_domain_side = matched_entry["subject_type_label"]
            our_range_side = matched_entry["object_type_label"]

        domain_pair = (our_domain_side, gold_domain) if (our_domain_side and gold_domain) else None
        range_pair = (our_range_side, gold_range) if (our_range_side and gold_range) else None

        type_judgment = {"domain": None, "range": None}
        if domain_pair or range_pair:
            try:
                type_judgment = extractor.compare_types_with_llm(domain_pair, range_pair)
            except Exception:
                logger.exception("compare_types_with_llm failed for sentence %s, gold triple %r", record["id"], gold)
                call_failures += 1
                type_judgment = {"domain": None, "range": None, "judge_call_failed": True}

        gold_results.append({
            "gold": gold, "matched": True, "direction": direction,
            "generated": {
                "subject": matched_entry["subject"], "relation": matched_entry["relation"],
                "object": matched_entry["object"],
                "subject_type": matched_entry["subject_type_label"],
                "object_type": matched_entry["object_type_label"],
            },
            "gold_domain": gold_domain, "gold_range": gold_range,
            "type_judgment": type_judgment,
        })

    return {
        "sentence_id": record["id"], "sent": record.get("sent", ""),
        "source_ontology_id": record.get("_source_ontology_id"),
        "gold_results": gold_results, "extras": remaining, "call_failures": call_failures,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Aggregation (reused for the overall summary AND each domain's own summary)
# ═══════════════════════════════════════════════════════════════════════════════

def summarize(results: list[dict], total_generated: int, *, ontology_id, judge_model: str) -> dict:
    """Compute the same recall / direction / extras / type-judgment-distribution
    summary from any subset of _judge_sentence results -- called once over
    every result (the "overall" summary) and once per domain (the
    "by_domain" breakdown), so a multi-domain combined run is directly
    comparable to N separate single-domain runs' own summaries."""
    total_gold = sum(len(r["gold_results"]) for r in results)
    total_call_failures = sum(r["call_failures"] for r in results)
    matched = [g for r in results for g in r["gold_results"] if g["matched"]]
    unmatched = [g for r in results for g in r["gold_results"] if not g["matched"]]
    failed_calls = [g for g in unmatched if g["reason"] == "judge_call_failed"]
    genuine_misses = [g for g in unmatched if g["reason"] != "judge_call_failed"]

    total_scored = total_gold - len(failed_calls)
    recall = len(matched) / total_scored if total_scored else None

    same_count = sum(1 for g in matched if g["direction"] == "same")
    inverse_count = sum(1 for g in matched if g["direction"] == "inverse")
    total_extras = sum(len(r["extras"]) for r in results)

    def _label_distribution(side: str) -> dict:
        dist = {label: 0 for label in TYPE_JUDGE_LABELS}
        dist["no_gold_type"] = 0
        dist["judge_call_failed"] = 0
        for g in matched:
            tj = g["type_judgment"]
            if tj.get("judge_call_failed"):
                dist["judge_call_failed"] += 1
                continue
            label = tj.get(side)
            if label in TYPE_JUDGE_LABELS:
                dist[label] += 1
            else:
                dist["no_gold_type"] += 1
        return dist

    return {
        "ontology_id": ontology_id,
        "judge_model": judge_model,
        "total_gold_triples": total_gold,
        "total_generated_triples": total_generated,
        "total_call_failures": total_call_failures,
        "recall_excl_call_failures": recall,
        "matched_same_direction": same_count,
        "matched_inverse_direction": inverse_count,
        "unmatched_genuine": len(genuine_misses),
        "unmatched_judge_call_failed": len(failed_calls),
        "extra_generated_triples": total_extras,
        "extra_rate": total_extras / total_generated if total_generated else None,
        "domain_type_judgment_distribution": _label_distribution("domain"),
        "range_type_judgment_distribution": _label_distribution("range"),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def _load_existing_results(output_path: Path) -> list[dict]:
    """Resume support: if --output already exists (a previous run was
    interrupted, e.g. by an API account running out of credits mid-domain),
    return its per_sentence results so already-judged sentences aren't
    re-paid-for. Returns [] if the file doesn't exist or can't be parsed
    (a truncated/corrupt partial write from a hard kill -- better to redo
    that one file's sentences than crash on it)."""
    if not output_path.exists():
        return []
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("per_sentence", [])
    except (json.JSONDecodeError, OSError):
        logger.warning("Could not parse existing %s, starting fresh", output_path)
        return []


def _write_output(
    results: list[dict], ground_truth: list[dict], buckets: dict, args, output_path: Path,
) -> dict:
    """Compute overall + by_domain summaries from `results` and write the
    full output file. Called periodically during judging (not just once at
    the end) so a mid-run interruption -- e.g. an API account running out
    of credits -- leaves a valid, already-resumable file on disk instead of
    losing every dollar spent so far."""
    overall_ontology_id = args.ontology_ids[0] if len(args.ontology_ids) == 1 else "+".join(args.ontology_ids)
    overall_summary = summarize(
        results, total_generated=sum(len(b) for b in buckets.values()),
        ontology_id=overall_ontology_id, judge_model=args.judge_model,
    )

    by_domain = {}
    for ontology_id in args.ontology_ids:
        domain_sentence_ids = {r["id"] for r in ground_truth if r["_source_ontology_id"] == ontology_id}
        domain_results = [r for r in results if r["sentence_id"] in domain_sentence_ids]
        domain_total_generated = sum(len(buckets.get(sid, [])) for sid in domain_sentence_ids)
        by_domain[ontology_id] = summarize(
            domain_results, total_generated=domain_total_generated,
            ontology_id=ontology_id, judge_model=args.judge_model,
        )

    output = {"overall": overall_summary, "by_domain": by_domain, "per_sentence": results}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    tmp_path.replace(output_path)  # atomic on POSIX -- never leaves a half-written judge_results.json
    return {"overall": overall_summary, "by_domain": by_domain}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triplets", required=True,
                         help="Combined triplets.jsonl -- one file regardless of how many "
                              "--ontology-ids are given (see extract_triplets.py's --ontology-id "
                              "tagging for how a multi-domain corpus stays one file).")
    parser.add_argument("--data-dir", required=True,
                         help="Parent dir containing <ontology-id>/{ground_truth.jsonl,ontology.ttl} "
                              "for each requested domain, e.g. data/text2kgbench/wikidata_tekgen.")
    parser.add_argument("--ontology-ids", nargs="+", required=True,
                         help="One or more ontology ids to evaluate, e.g. ont_2_music ont_1_movie. "
                              "Each sentence is scored against ITS OWN domain's ontology (tracked via "
                              "source_ontology_id), so this also works against a single combined "
                              "pipeline run spanning several domains, not just one domain at a time.")
    parser.add_argument("--output-dir", help="Pipeline output_dir -- the latest run_<n> checkpoint "
                                              "is used unless --checkpoints-dir is given explicitly.")
    parser.add_argument("--checkpoints-dir", help="Explicit output_dir/checkpoints/run_<n> dir "
                                                     "(overrides --output-dir's auto-latest lookup).")
    parser.add_argument("--output", required=True)
    parser.add_argument("--judge-model", default="openai/gpt-4o",
                         help="Chat-completions model name for the judge LLM -- deliberately "
                              "independent of whatever model built the KG (see module docstring). "
                              "Must match the exact casing/slug the endpoint expects.")
    parser.add_argument("--judge-base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--judge-api-key-env", default="OPENROUTER_KEY",
                         help="Env var holding the judge LLM's API key.")
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()

    if args.checkpoints_dir:
        run_dir = Path(args.checkpoints_dir)
    elif args.output_dir:
        run_dir = _latest_run_dir(Path(args.output_dir))
    else:
        raise SystemExit("Pass either --checkpoints-dir or --output-dir")

    logger.info("Loading canonical vocabularies from %s", run_dir)
    relation_vocab = _load_pickle(run_dir / "relation_dedup.pkl")
    type_vocab = _load_pickle(run_dir / "type_dedup.pkl")

    data_dir = Path(args.data_dir)
    ontologies: dict[str, Text2KGOntology] = {}
    ground_truth: list[dict] = []
    for ontology_id in args.ontology_ids:
        domain_dir = data_dir / ontology_id
        ontology = parse_ontology(domain_dir / "ontology.ttl")
        ontologies[ontology_id] = ontology
        records = _load_jsonl(str(domain_dir / "ground_truth.jsonl"))
        for r in records:
            r["_source_ontology_id"] = ontology_id
        ground_truth.extend(records)
        logger.info("Domain %s: %d classes, %d relations, %d ground-truth sentences",
                    ontology_id, len(ontology.class_label_by_iri), len(ontology.relations), len(records))

    triplets = _load_jsonl(args.triplets)
    buckets = build_sentence_buckets(triplets, relation_vocab, type_vocab)
    logger.info("Loaded %d generated triplets across %d sentences (%d domains)",
                len(triplets), len(buckets), len(args.ontology_ids))

    api_key = os.environ.get(args.judge_api_key_env)
    if not api_key:
        raise RuntimeError(f"Environment variable {args.judge_api_key_env!r} is not set")
    extractor = LLMTripletExtractor(api_key=api_key, model=args.judge_model, base_url=args.judge_base_url)

    output_path = Path(args.output)
    loaded = _load_existing_results(output_path)
    # A sentence that hit call failures (e.g. every remaining call failing
    # once an account runs out of credits) still gets a result dict back
    # from _judge_sentence -- it doesn't raise. Treating that as "done"
    # would permanently lock in a call-failure result instead of retrying
    # it once the underlying issue (credits, rate limit, ...) is fixed, so
    # only clean (zero call-failure) sentences count as already judged.
    results = [r for r in loaded if r.get("call_failures", 0) == 0]
    done_ids = {r["sentence_id"] for r in results}
    if loaded:
        logger.info("Resuming: %d/%d sentences already judged cleanly, %d had call failures and will "
                    "be retried, %d remaining", len(done_ids), len(ground_truth),
                    len(loaded) - len(done_ids), len(ground_truth) - len(done_ids))

    tasks = [
        (record, buckets.get(record["id"], []), ontologies[record["_source_ontology_id"]])
        for record in ground_truth if record["id"] not in done_ids
    ]

    # Checkpointed periodically (not just once at the end): an API account
    # running out of credits mid-run (a real, observed failure mode) must
    # not lose every dollar of judging already paid for -- see
    # _write_output's docstring and _load_existing_results above, which is
    # what makes re-running this exact command afterward a resume rather
    # than a full re-judge from scratch.
    if tasks:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(_judge_sentence, record, bucket, ontology, extractor): record["id"]
                for record, bucket, ontology in tasks
            }
            completed = 0
            for future in as_completed(futures):
                completed += 1
                results.append(future.result())
                if completed % 25 == 0 or completed == len(futures):
                    logger.info("Judged %d/%d sentences", completed, len(futures))
                    _write_output(results, ground_truth, buckets, args, output_path)

    summaries = _write_output(results, ground_truth, buckets, args, output_path)
    overall_summary, by_domain = summaries["overall"], summaries["by_domain"]

    total_call_failures = overall_summary["total_call_failures"]
    total_gold = overall_summary["total_gold_triples"]
    # Same discipline as scripts/mine_benchmark/run_judge_eval.py: a judge
    # call failure must never look like a "0" result in the summary above
    # (it's excluded from recall via total_scored inside summarize()) -- but
    # it also must not be silently invisible, so it's surfaced loudly here.
    if total_call_failures:
        logger.error(
            "%d/%d judge calls FAILED (excluded from recall, NOT counted as misses). "
            "If this is a large fraction (e.g. an account ran out of credits mid-run), fix the "
            "underlying cause and re-run this exact command -- already-judged sentences resume "
            "instead of being re-paid-for, but the sentences that hit call failures will be retried.",
            total_call_failures, total_gold,
        )

    recall = overall_summary["recall_excl_call_failures"]
    logger.info(
        "Overall recall (excl. call failures): %s. Same=%d Inverse=%d Extras=%d/%d. Results written to %s",
        f"{recall:.2%}" if recall is not None else "N/A",
        overall_summary["matched_same_direction"], overall_summary["matched_inverse_direction"],
        overall_summary["extra_generated_triples"], overall_summary["total_generated_triples"], output_path,
    )
    for ontology_id, s in by_domain.items():
        r = s["recall_excl_call_failures"]
        logger.info("  %s: recall=%s (%d/%d gold matched)",
                    ontology_id, f"{r:.2%}" if r is not None else "N/A",
                    s["matched_same_direction"] + s["matched_inverse_direction"], s["total_gold_triples"])


if __name__ == "__main__":
    main()
