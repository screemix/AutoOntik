"""
LLM-as-a-judge evaluation of a constructed KG against MINE's generated queries
==================================================================================

Standalone / independent of build_kg_graph.py on purpose: this script only
reads the plain kg-gen-format graph JSON that script wrote (output/mine/
kg_graph.json by default) plus data/mine/queries.json -- it has NO
dependency on ontodisco's pickled dataclasses (no torch/transformers/sklearn
import needed), so it can be re-run any number of times against an
already-built graph, e.g. to try a different --judge-model once you have a
real API key, without re-running the (expensive, LLM-heavy) ontology
pipeline.

Must run in a Python >=3.10 environment with `kg-gen` installed (this repo's
own venv is Python 3.8, which kg-gen doesn't support) -- see the conda
env setup note in run_all.sh. Retrieval reuses kg-gen's own
KGGen.retrieve() (sentence-transformers node embeddings + 2-hop graph walk)
for fidelity with kg-gen's own MINE evaluation methodology
(https://github.com/stair-lab/kg-gen/blob/main/experiments/MINE/_1_evaluation.py).

Judge methodology mirrors kg-gen's own eval exactly: each MINE
`generated_queries` entry is BOTH the retrieval query and the "correct
answer" text to check for (MINE's queries are factual statements, not
questions) -- retrieve context for it from the graph, then ask an LLM
"Determine whether the context contains the information stated in the
correct answer. Respond with 1 if yes, 0 if no." kg-gen's own script hits
OpenAI's gpt-5 via dspy for this; here it's a plain chat-completion call so
the judge endpoint is swappable via --judge-base-url/--judge-api-key-env
without needing dspy's provider-string conventions.

Usage:
    python -m scripts.mine_benchmark.run_judge_eval \\
        --graph output/mine/kg_graph.json \\
        --queries data/mine/queries.json \\
        --output output/mine/judge_results.json \\
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
import openai
import httpx

from kg_gen.kg_gen import KGGen
from kg_gen.models import Graph

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = (
    "Determine whether the context contains the information stated in the correct answer. "
    "Respond with ONLY the single character 1 if the context supports/contains the correct "
    "answer, or 0 if it does not. No other text."
)


def load_graph(graph_path: str) -> Graph:
    with open(graph_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Graph(
        entities=set(data["entities"]),
        edges=set(data["edges"]),
        relations={tuple(r) for r in data["relations"]},
    )


def build_judge_client(base_url: str | None, api_key_env: str, proxy_key_env: str | None) -> openai.OpenAI:
    api_key = os.environ.get(api_key_env)
    if proxy_key_env:
        proxy = os.environ.get(proxy_key_env)
        http_client = httpx.Client(proxy=proxy)
        client = openai.OpenAI(
            api_key=api_key, http_client=http_client, base_url=base_url
        )
    else:
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
    if not api_key:
        raise RuntimeError(f"Environment variable {api_key_env!r} is not set")
    return client


def judge_one(client: openai.OpenAI, model: str, context_text: str, correct_answer: str,
              max_retries: int = 5) -> tuple[int | None, str | None]:
    """Returns (evaluation, error). evaluation is None iff error is not None --
    callers MUST NOT fold a failed call into a 0/"no" score (see the 402
    insufficient-credits incident: every one of 1500 calls raised the same
    APIStatusError, and because the old version of this function swallowed
    that into `return 0`, the run finished "successfully" with a silent
    0.0% accuracy that was indistinguishable from a real (very bad) result."""
    user_prompt = f"Context: {context_text}\n\nCorrect answer: {correct_answer}"
    last_err = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
            )
            content = response.choices[0].message.content.strip()
            for ch in content:
                if ch in "01":
                    return int(ch), None
            logger.warning("Unparseable judge response %r, treating as 0", content)
            return 0, None
        except openai.APIStatusError as e:
            # Non-retryable client/account errors (401 bad key, 402 no
            # credits, 404 unknown model) will fail identically every
            # attempt -- burning through max_retries with backoff just
            # delays reporting the same fatal problem, once per query.
            if e.status_code in (401, 402, 403, 404):
                logger.error("Non-retryable judge error (HTTP %d), giving up immediately: %s",
                             e.status_code, e)
                return None, f"HTTP {e.status_code}: {e}"
            last_err = e
            time.sleep(min(2 ** attempt, 30))
        except Exception as e:
            last_err = e
            time.sleep(min(2 ** attempt, 30))
    logger.error("Judge call failed after %d retries: %s", max_retries, last_err)
    return None, str(last_err)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", default="output/mine/kg_graph.json")
    parser.add_argument("--queries", default="data/mine/queries.json")
    parser.add_argument("--output", default="output/mine/judge_results.json")
    parser.add_argument("--retrieval-model", default="all-MiniLM-L6-v2",
                         help="sentence-transformers model for KGGen.retrieve() node embeddings "
                              "(matches kg-gen's own MINE eval default).")
    parser.add_argument("--top-k", type=int, default=8, help="Nodes retrieved per query (kg-gen default: 8).")
    parser.add_argument("--judge-model", default="Openai/Gpt-oss-120b",
                         help="Chat-completions model name for the judge LLM (must match the "
                              "exact casing the endpoint expects -- the AIRI gateway rejects "
                              "'openai/gpt-oss-120b', it wants 'Openai/Gpt-oss-120b', see "
                              "configs/gpt_oss.yaml).")
    parser.add_argument("--judge-base-url", default=os.environ.get("AIRI_BASE_URL",
                         "https://inference.airi.net:46783/v1"))
    parser.add_argument("--judge-api-key-env", default="AIRI_KEY",
                         help="Env var holding the judge LLM's API key. Swap this (and "
                              "--judge-model/--judge-base-url) to point the judge at a "
                              "different model later without touching the constructed graph.")
    parser.add_argument("--proxy-url-env", default="PROXY_URL", help="Optional proxy for LLM API")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--limit-essays", type=int, default=None,
                         help="Only evaluate the first N essays' queries (quick sanity check).")
    args = parser.parse_args()

    logger.info("Loading graph from %s", args.graph)
    graph = load_graph(args.graph)
    logger.info("MINE combined KG: %d entities, %d edge labels, %d relation triples",
                len(graph.entities), len(graph.edges), len(graph.relations))

    kggen = KGGen(retrieval_model=args.retrieval_model)
    nx_graph = kggen.to_nx(graph)
    logger.info("Computing node embeddings for %d nodes...", nx_graph.number_of_nodes())
    node_embeddings, _ = kggen.generate_embeddings(nx_graph)

    with open(args.queries, "r", encoding="utf-8") as f:
        queries_by_essay = json.load(f)
    if args.limit_essays:
        queries_by_essay = queries_by_essay[: args.limit_essays]

    judge_client = build_judge_client(args.judge_base_url, args.judge_api_key_env, args.proxy_url_env)

    # Flatten to (essay_id, essay_topic, query) tasks, run retrieval + judge per query.
    tasks = []
    for essay in queries_by_essay:
        for query in essay["queries"]:
            tasks.append((essay["id"], essay["topic"], query))
    logger.info("Evaluating %d queries across %d essays against the single combined KG "
                "(top_k=%d, max_workers=%d)...", len(tasks), len(queries_by_essay),
                args.top_k, args.max_workers)

    def _run_one(essay_id, essay_topic, query):
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

    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(_run_one, *task): task for task in tasks}
        completed = 0
        for future in as_completed(futures):
            completed += 1
            results.append(future.result())
            if completed % 25 == 0 or completed == len(tasks):
                logger.info("Judged %d/%d queries", completed, len(tasks))

    # Judge-call failures (bad model name, no credits, network errors, ...)
    # must NEVER be folded into the accuracy score as if they were a "0"
    # judgment -- that's exactly what silently turned a 402 Insufficient
    # Credits error, on every single call, into a bogus 0.0% accuracy in a
    # previous run. Failures are counted and reported separately instead.
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
        logger.error(
            "%d/%d judge calls FAILED (excluded from accuracy, not counted as wrong). "
            "Sample errors: %s", len(failed), len(results), sample_errors,
        )
        if len(failed) == len(results):
            logger.error(
                "EVERY judge call failed -- overall_accuracy is null, NOT 0.0%%. "
                "Check --judge-model/--judge-base-url/--judge-api-key-env and the "
                "account behind that key before trusting any number from this run."
            )

    output = {
        "graph_source": args.graph,
        "judge_model": args.judge_model,
        "retrieval_model": args.retrieval_model,
        "top_k": args.top_k,
        "num_queries": len(results),
        "num_scored": len(scored),
        "num_judge_failures": len(failed),
        "overall_accuracy": overall_accuracy,
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
        logger.info("Overall accuracy: %.2f%% (%d/%d scored, %d failed). Results written to %s",
                    overall_accuracy * 100, overall_correct, len(scored), len(failed), output_path)


if __name__ == "__main__":
    main()
