"""
LLM-as-a-judge evaluation, one graph PER ESSAY (kg-gen-fidelity variant)
==================================================================================

Sibling of run_judge_eval.py, differing in exactly one respect: that script
builds ONE nx graph for the whole corpus and retrieves every essay's
queries from it; this script builds a FRESH nx graph (and fresh node
embeddings) per essay, from that essay's own Graph JSON (as produced by
scripts/mine_benchmark/ontology_linked_extraction.py), and only ever
retrieves an essay's queries from its own graph -- matching kg-gen's own
MINE evaluation methodology exactly (experiments/MINE/evaluation.py in the
kg-gen repo builds and evaluates one graph per essay, never a shared one).

Same environment requirement as run_judge_eval.py: Python >=3.10 with
kg-gen installed (this repo's own venv is 3.8).

Usage:
    python -m scripts.mine_benchmark.run_judge_eval_per_essay \\
        --graphs-dir output/mine/ontology_linked_graphs \\
        --queries data/mine/queries.json \\
        --output output/mine/judge_results_per_essay.json \\
        --judge-model Openai/Gpt-oss-120b \\
        --judge-base-url https://inference.airi.net:46783/v1 \\
        --judge-api-key-env AIRI_KEY
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from kg_gen.kg_gen import KGGen
from kg_gen.models import Graph

# Reuse run_judge_eval.py's own judge client / judge_one -- identical
# methodology, only the retrieval scope differs in this script.
from scripts.mine_benchmark.run_judge_eval import build_judge_client, judge_one

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_graph(graph_path: Path) -> Graph:
    with open(graph_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Graph(
        entities=set(data["entities"]),
        edges=set(data["edges"]),
        relations={tuple(r) for r in data["relations"]},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs-dir", default="output/mine/ontology_linked_graphs",
                         help="Directory holding one essay_<id>.json Graph file per essay "
                              "(as written by ontology_linked_extraction.py).")
    parser.add_argument("--queries", default="data/mine/queries.json")
    parser.add_argument("--output", default="output/mine/judge_results_per_essay.json")
    parser.add_argument("--retrieval-model", default="all-MiniLM-L6-v2",
                         help="sentence-transformers model for KGGen.retrieve() node embeddings "
                              "(matches kg-gen's own MINE eval default).")
    parser.add_argument("--top-k", type=int, default=8, help="Nodes retrieved per query (kg-gen default: 8).")
    parser.add_argument("--judge-model", default="Openai/Gpt-oss-120b")
    parser.add_argument("--judge-base-url", default=os.environ.get("AIRI_BASE_URL",
                         "https://inference.airi.net:46783/v1"))
    parser.add_argument("--judge-api-key-env", default="AIRI_KEY")
    parser.add_argument("--proxy-url-env", default="PROXY_URL")
    parser.add_argument("--max-workers", type=int, default=8,
                         help="Parallel QUERIES within one essay -- essays themselves are "
                              "processed sequentially, since each needs its own nx graph built "
                              "and embedded first.")
    parser.add_argument("--limit-essays", type=int, default=None,
                         help="Only evaluate the first N essays' queries (quick sanity check).")
    args = parser.parse_args()

    graphs_dir = Path(args.graphs_dir)
    with open(args.queries, "r", encoding="utf-8") as f:
        queries_by_essay = json.load(f)
    if args.limit_essays:
        queries_by_essay = queries_by_essay[: args.limit_essays]

    kggen = KGGen(retrieval_model=args.retrieval_model)
    judge_client = build_judge_client(args.judge_base_url, args.judge_api_key_env, args.proxy_url_env)

    results = []
    skipped_essays = []
    start = time.time()

    for i, essay in enumerate(queries_by_essay, start=1):
        essay_id, essay_topic = essay["id"], essay["topic"]
        graph_path = graphs_dir / f"essay_{essay_id}.json"
        if not graph_path.exists():
            logger.warning("[%d/%d] No graph file for essay %s (%s) -- skipping its %d queries",
                            i, len(queries_by_essay), essay_id, graph_path, len(essay["queries"]))
            skipped_essays.append(essay_id)
            continue

        graph = load_graph(graph_path)
        if not graph.entities:
            logger.warning("[%d/%d] Essay %s has an EMPTY graph (0 entities) -- every query will "
                            "retrieve no context; skipping rather than reporting a meaningless 0%%",
                            i, len(queries_by_essay), essay_id)
            skipped_essays.append(essay_id)
            continue

        nx_graph = kggen.to_nx(graph)
        node_embeddings, _ = kggen.generate_embeddings(nx_graph)
        logger.info("[%d/%d] Essay %s (%r): %d entities, %d relations",
                    i, len(queries_by_essay), essay_id, essay_topic,
                    len(graph.entities), len(graph.relations))

        def _run_one(query):
            _, _, context_text = kggen.retrieve(query, node_embeddings, nx_graph, k=args.top_k)
            evaluation, judge_error = judge_one(judge_client, args.judge_model, context_text, query)
            return {
                "essay_id": essay_id,
                "essay_topic": essay_topic,
                "correct_answer": query,
                "retrieved_context": context_text,
                "evaluation": evaluation,
                "judge_error": judge_error,
            }

        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [executor.submit(_run_one, q) for q in essay["queries"]]
            for future in as_completed(futures):
                results.append(future.result())

    elapsed = time.time() - start
    scored = [r for r in results if r["judge_error"] is None]
    failed = [r for r in results if r["judge_error"] is not None]
    overall_correct = sum(r["evaluation"] for r in scored)
    overall_accuracy = overall_correct / len(scored) if scored else None

    per_essay: dict[int, dict] = {}
    for r in results:
        bucket = per_essay.setdefault(r["essay_id"], {"topic": r["essay_topic"], "correct": 0, "total": 0, "failed": 0})
        if r["judge_error"] is not None:
            bucket["failed"] += 1
            continue
        bucket["total"] += 1
        bucket["correct"] += r["evaluation"]
    for essay_id, bucket in per_essay.items():
        bucket["accuracy"] = bucket["correct"] / bucket["total"] if bucket["total"] else None

    if failed:
        sample_errors = sorted({r["judge_error"] for r in failed})[:3]
        logger.error("%d/%d judge calls FAILED (excluded from accuracy). Sample errors: %s",
                     len(failed), len(results), sample_errors)

    output = {
        "graphs_dir": str(graphs_dir),
        "judge_model": args.judge_model,
        "retrieval_model": args.retrieval_model,
        "top_k": args.top_k,
        "num_essays_evaluated": len(queries_by_essay) - len(skipped_essays),
        "num_essays_skipped": len(skipped_essays),
        "skipped_essay_ids": skipped_essays,
        "num_queries": len(results),
        "num_scored": len(scored),
        "num_judge_failures": len(failed),
        "overall_accuracy": overall_accuracy,
        "elapsed_seconds": elapsed,
        "per_essay_accuracy": per_essay,
        "results": results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    if overall_accuracy is None:
        logger.error("Overall accuracy: N/A -- all %d judge calls failed. Results written to %s",
                     len(results), output_path)
    else:
        logger.info("Overall accuracy: %.2f%% (%d/%d scored, %d failed, %d essays skipped). "
                    "Results written to %s", overall_accuracy * 100, overall_correct, len(scored),
                    len(failed), len(skipped_essays), output_path)


if __name__ == "__main__":
    main()
