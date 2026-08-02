from __future__ import annotations

import openai

# import os
# from dotenv import load_dotenv, find_dotenv
from tenacity import (
    retry,
    wait_random_exponential,
    before_sleep_log,
    stop_after_attempt,
    stop_never
)
import logging
import sys
import json
import re
import threading
from pathlib import Path
from typing import Dict, List, Union, Optional
import tenacity
import httpx

from src.ontodisco.utils.logging_config import get_logger

# Configure logging
logger = get_logger("OpenAIUtils")

# _ = load_dotenv(find_dotenv())
# OpenAI
MAX_ATTEMPTS = 1


class LLMTripletExtractor:
    """A class for extracting and processing knowledge graph triplets using OpenAI's LLMs."""

    MODEL_PRICES = {
        "gpt-4o": {"input": 2.5, "output": 10},
        "gpt-4o-mini": {"input": 0.15, "output": 0.6},
        "gpt-4.1-mini": {"input": 0.4, "output": 1.6},
        "gpt-4.1": {"input": 2.0, "output": 8.0},
        "Meta-llama/Llama-3.3-70B-Instruct": {"input": 0.04, "output": 0.12},
        "qwen/qwen3-32b": {"input": 0.05, "output": 0.2},
        "Openai/Gpt-oss-120b": {"input": 0.05, "output": 0.2},
        "Qwen/Qwen3-32B": {"input": 0.05, "output": 0.2},
        "openai/gpt-oss-120b": {"input": 0.05, "output": 0.2},
        "Qwen/Qwen3-Next-80B-A3B-Instruct": {"input": 0.05, "output": 0.2},  # configs/qwen.yaml, same AIRI-gateway placeholder rate as the other AIRI-served models above
        "openai/gpt-4o": {"input": 2.5, "output": 10},  # OpenRouter slug for the judge model -- same real per-token rate as "gpt-4o" above
    }

    def __init__(
        self,
        api_key: str,
        prompt_folder_path: str = str(Path(__file__).parent / "prompts"),
        system_prompt_paths: Optional[Dict[str, str]] = None,
        model: str = "gpt-4o",
        max_attempts=MAX_ATTEMPTS,
        proxy: str = None,
        base_url: str = "https://api.openai.com/v1",
        save_messages: bool = False,
    ):
        """
        Initialize the LLMTripletExtractor.

        Args:
            prompt_folder_path: Path to folder containing prompt files
            system_prompt_paths: Dictionary mapping prompt types to file paths
            model: Name of the OpenAI model to use
        """
        if proxy:
            http_client = httpx.Client(proxy=proxy)
            self.client = openai.OpenAI(
                api_key=api_key, http_client=http_client, base_url=base_url
            )
        else:
            self.client = openai.OpenAI(api_key=api_key, base_url=base_url)

        if system_prompt_paths is None:
            system_prompt_paths = {
                "triplet_extraction": "prompt_1_with_types_and_qualifiers.txt",
                "cluster_entity_types": "cluster_entity_types.txt",
                "cluster_entity_names": "cluster_entity_names.txt",
                "cluster_relations": "cluster_relations.txt",
                "hierarchy_pairwise_action": "hierarchy_pairwise_action.txt",
                "relation_direction": "relation_direction.txt",
                "triple_match": "triple_match.txt",
                "type_compare": "type_compare.txt",
                "question_entity_extractor": "question_entity_extraction.txt",
                "question_entity_ranker": "question_entity_ranker.txt",
                "qa": "qa_answer.txt",
            }

        # Load all prompts (paths may include subfolders, e.g. triplet_extraction/foo.txt)
        prompt_folder = Path(prompt_folder_path)
        self.prompts = {}
        for prompt_type, filename in system_prompt_paths.items():
            prompt_path = prompt_folder / filename
            if prompt_path.is_file():
                self.prompts[prompt_type] = prompt_path.read_text(encoding="utf-8")
            else:
                logger.warning(f"Prompt file {filename} not found in {prompt_folder}")
                self.prompts[prompt_type] = ""

        self.model = model
        self.messages = []
        self.prompt_tokens_num = 0
        self.completion_tokens_num = 0
        self.current_cost = 0
        # get_completion() is called from multiple threads once dedup_base's
        # verify_clusters_with_llm / hierarchy_induction's band placement run
        # concurrently -- the plain `+=` below is a read-modify-write across
        # three attributes and isn't atomic under the GIL, so concurrent
        # callers can lose updates without this lock.
        self._token_lock = threading.Lock()
        self.save_messages = save_messages
        self._refine_attempt = 0
        self._prev_error = None  # store previous exception
        self.MAX_ATTEMPTS = max_attempts

        # Set pricing
        if model not in self.MODEL_PRICES:
            logger.error(f"Unknown model: {model}. Price will be set to 0.")
            self.input_price = 0.0
            self.output_price = 0.0
        else:
            self.input_price = self.MODEL_PRICES[model]["input"]
            self.output_price = self.MODEL_PRICES[model]["output"]
            logger.info(f"Model: {model}. Input price: {self.input_price}. Output price: {self.output_price}.")

    def extract_json(self, text: str) -> Union[dict, list, str]:
        """Extract JSON from text, handling both code blocks and inline JSON."""
        fenced_pattern = r"```json\s*(\{.*?\}|\[.*?\])\s*```"  # JSON in code blocks

        logger.log(logging.DEBUG, "LLM Output: %s", text)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        match = re.search(fenced_pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                logger.error("Failed to parse fenced JSON block: %s", text)

        # Inline JSON (no code fence): `(\{.*?\}|\[.*?\])` used to be tried
        # here too, but a non-greedy brace match has nothing forcing it to
        # backtrack past the FIRST closing brace it finds -- for nested
        # JSON (e.g. {"triplets": [{...}, {...}]}), that's the first
        # triplet's closing brace, not the outer object's, so the captured
        # substring was truncated and json.loads always failed on it.
        # json.JSONDecoder.raw_decode() parses however much valid JSON
        # starts at the given position and tells us where it ends, so it
        # isn't fooled by nested braces the way the regex was.
        start = next((i for i, ch in enumerate(text) if ch in "{["), None)
        if start is not None:
            try:
                obj, _ = json.JSONDecoder().raw_decode(text, start)
                return obj
            except json.JSONDecodeError:
                logger.error("Failed to parse inline JSON: %s", text)

        return text

    @retry(
        wait=wait_random_exponential(multiplier=1, max=60),
        before_sleep=before_sleep_log(logger, logging.ERROR),
        stop=stop_after_attempt(5),
    )
    # @tenacity.retry(stop=stop_after_attempt(5), reraise=True)
    def get_completion(
        self, system_prompt: str, user_prompt: str, transform_to_json: bool = True
    ) -> Union[dict, list, str]:
        """Get completion from OpenAI API with retry logic."""

        if self.model == "qwen/qwen3-32b" or self.model == "Qwen/Qwen3-32B":
            user_prompt = "/no_think \n" + user_prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if self.save_messages:
            self.messages.extend(messages)
            messages = self.messages

        response = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=0
        )
        with self._token_lock:
            self.completion_tokens_num += response.usage.completion_tokens
            self.prompt_tokens_num += response.usage.prompt_tokens
            self.current_cost += (
                response.usage.completion_tokens * self.output_price
                + response.usage.prompt_tokens * self.input_price
            )

        content = response.choices[0].message.content.strip()
        logger.debug("Output content: %s\n%s", str(content), "-" * 100)
        output = self.extract_json(content) if transform_to_json else content
                
        if self.save_messages:
            self.messages.append({"role": "assistant", "content": content})

        return output

    # @tenacity.retry(stop=tenacity.stop_after_attempt(MAX_ATTEMPTS), reraise=True)
    @retry(
        wait=wait_random_exponential(multiplier=1, max=60),
        before_sleep=before_sleep_log(logger, logging.ERROR),
        stop=stop_after_attempt(5),
    )
    def extract_triplets_from_text(self, text: str) -> dict:
        """Extract knowledge graph triplets from text."""

        self._refine_attempt += 1
        attempt = self._refine_attempt
        logger.log(
            logging.DEBUG,
            "Attempt of a function call extract_triplets_from_text: %s",
            attempt,
        )
        system_prompt = self.prompts["triplet_extraction"]
        if attempt > 1:
            prev_error = self._prev_error
            system_prompt += f"\n(Previous attempt #{attempt-1} failed with error: {prev_error}. Please adjust your answer!)"
            logger.log(logging.ERROR, "System prompt: %s", system_prompt)

        try:
            return self.get_completion(
                system_prompt=system_prompt, user_prompt=f'Text: "{text}"'
            )
        except Exception as e:
            self._prev_error = e
            # if json from output is broken after 3 attempts  - raise an exception
            logger.log(logging.ERROR, str(e))
            if attempt > self.MAX_ATTEMPTS:
                raise e
        
    def verify_cluster_with_llm(
        self, members: list[str], surface_form_type='entity_type',
        member_context: Optional[Dict[str, str]] = None,
    ) -> list[tuple[str, list[str]]]:
        """Verify clusters with LLM.

        member_context: optional label -> relational-context evidence string
        (e.g. "often appears as object of: directed, starred in"), rendered
        alongside each candidate so the LLM has concrete evidence for merge/
        split decisions beyond surface-form similarity.
        """
        if surface_form_type == 'entity_type':
            system_prompt = self.prompts["cluster_entity_types"]
        elif surface_form_type == 'entity':
            system_prompt = self.prompts["cluster_entity_names"]
        elif surface_form_type == 'relation':
            system_prompt = self.prompts["cluster_relations"]
        else:
            raise Exception("Unknown surface form type")

        if member_context:
            candidates_str = ", ".join(
                f"{m} (context: {member_context[m]})" if member_context.get(m) else m
                for m in members
            )
        else:
            candidates_str = ", ".join(members)

        response = self.get_completion(
            system_prompt=system_prompt, user_prompt=f'Candidates: {candidates_str}')

        logger.log(logging.DEBUG, f"Input: {members}")
        logger.log(logging.DEBUG, f"Response: {response}")

        if isinstance(response, str):
            logger.warning("verify_cluster_with_llm: LLM returned unparseable string")
            return []

        groups = response.get("groups", [])
        if not groups:
            # Fallback: maybe the LLM used "merged"/"split" keys (variant format)
            groups = response.get("merged", []) or response.get("split", [])

        return groups

    def resolve_hierarchy_relation(
        self, focal_label: str, candidate_labels: list[str],
        focal_context: str = "", candidate_context: Optional[Dict[str, str]] = None,
    ) -> dict:
        """Ask the LLM how one focal type relates to a batch of existing
        candidate types (see hierarchy_induction.py's
        _resolve_with_batched_escalation, which calls this once per
        similarity-ranked batch of candidates).

        Returns a dict with "relation" in
        {"parent", "child", "same_concept", "same_class", "none"}. For every
        relation except "child", the chosen candidate is under "candidate"
        (a single label string, omitted/ignored for "none"). For "child",
        the focal type may subsume MORE THAN ONE candidate at once (CLAUDE.md
        §16.2.2's "return a set, not a single child" requirement -- without
        this, a focal node that should retroactively absorb two existing
        siblings would only catch one per placement call, since nothing
        guarantees a later call ever re-examines the one left behind) --
        normalized here into "candidates" (always a list), regardless of
        whether the LLM used the singular "candidate" or plural "candidates"
        key in its raw JSON.
        Also includes "confidence", and -- only for "same_class" --
        "new_parent_label"/"new_parent_definition".
        Returns {"relation": "none"} on any unparseable response, never raises.
        """
        system_prompt = self.prompts["hierarchy_pairwise_action"]

        focal_str = focal_label
        if focal_context:
            focal_str = f"{focal_label} (context: {focal_context})"

        if candidate_context:
            candidates_str = ", ".join(
                f"{c} (context: {candidate_context[c]})" if candidate_context.get(c) else c
                for c in candidate_labels
            )
        else:
            candidates_str = ", ".join(candidate_labels)

        user_prompt = f"Focal type: {focal_str}\nExisting candidates: {candidates_str}"

        try:
            response = self.get_completion(system_prompt=system_prompt, user_prompt=user_prompt)
        except Exception:
            logger.exception("resolve_hierarchy_relation: LLM call failed")
            return {"relation": "none"}

        logger.log(logging.DEBUG, f"Input: focal={focal_label!r} candidates={candidate_labels!r}")
        logger.log(logging.DEBUG, f"Response: {response}")

        if not isinstance(response, dict):
            logger.warning("resolve_hierarchy_relation: LLM returned unparseable response")
            return {"relation": "none"}

        if response.get("relation") not in ("parent", "child", "same_concept", "same_class", "none"):
            logger.warning(
                "resolve_hierarchy_relation: LLM returned unknown relation %r; treating as none",
                response.get("relation"),
            )
            return {"relation": "none"}

        if response.get("relation") == "child":
            raw_candidates = response.get("candidates")
            if raw_candidates is None:
                single = response.get("candidate")
                raw_candidates = [single] if single else []
            elif not isinstance(raw_candidates, list):
                raw_candidates = [raw_candidates]
            response = dict(response)
            response["candidates"] = [str(c) for c in raw_candidates if c]

        return response

    def classify_relation_direction(
        self, canonical_label: str, surface_forms: list[str],
    ) -> Dict[str, str]:
        """
        Classify each surface form merged into one canonical relation as
        "forward" or "inverted" relative to canonical_label
        (see constraints.py's direction-resolution stage for why this is
        needed: relation canonicalization is role-blind, so a canonical
        relation can silently pool a predicate with its grammatical
        inverse).

        Only worth calling when a canonical relation has more than one
        surface form -- callers should treat a single-surface-form relation
        as trivially "forward" without invoking this.
        """
        system_prompt = self.prompts["relation_direction"]
        user_prompt = (
            f'Canonical relation: "{canonical_label}"\n'
            f"Surface forms: {json.dumps(surface_forms)}"
        )

        response = self.get_completion(system_prompt=system_prompt, user_prompt=user_prompt)

        logger.log(logging.DEBUG, f"Input: {canonical_label} / {surface_forms}")
        logger.log(logging.DEBUG, f"Response: {response}")

        if isinstance(response, str) or not isinstance(response, dict):
            logger.warning(
                "classify_relation_direction: LLM returned unparseable response for %r",
                canonical_label,
            )
            return {sf: "forward" for sf in surface_forms}

        labels = response.get("labels", {})
        # Defensively default any surface form the LLM omitted to "forward"
        # (no swap applied) rather than dropping it.
        return {sf: labels.get(sf, "forward") for sf in surface_forms}

    def match_triple_with_llm(
        self, gold_subject: str, gold_relation: str, gold_object: str,
        candidates: list[tuple[str, str, str]],
    ) -> Dict:
        """
        Match a reference (gold_subject, gold_relation, gold_object) triple
        against a list of candidate (subject, relation, object) triples
        extracted from the same sentence (see
        scripts/text2kgbench_eval/run_judge_eval.py's per-sentence matching
        loop, which shows this the REMAINING, not-yet-consumed candidate
        bucket for one sentence and removes whichever one gets matched).

        Returns {"match": <int index into candidates>, "direction": "same" |
        "inverse"} if exactly one candidate expresses the same real-world
        fact as the reference (possibly with subject/object swapped), or
        {"match": None} if none do (including on an unparseable response --
        that case is logged and defaulted, not raised). Does NOT catch
        exceptions from the underlying API call itself -- callers must
        handle those (see run_judge_eval.py's explicit call-failure
        tracking, kept separate from a genuine "no match" judgment so an API
        outage never gets silently counted as a real miss, mirroring the
        documented lesson in scripts/mine_benchmark/run_judge_eval.py).
        """
        system_prompt = self.prompts["triple_match"]
        candidates_str = "\n".join(
            f"{i}: ({s}) -[{r}]-> ({o})" for i, (s, r, o) in enumerate(candidates)
        )
        user_prompt = (
            f"Reference triple: ({gold_subject}) -[{gold_relation}]-> ({gold_object})\n"
            f"Candidate triples:\n{candidates_str}"
        )

        response = self.get_completion(system_prompt=system_prompt, user_prompt=user_prompt)

        logger.log(logging.DEBUG, f"Reference: {gold_subject}/{gold_relation}/{gold_object}")
        logger.log(logging.DEBUG, f"Response: {response}")

        if not isinstance(response, dict):
            logger.warning("match_triple_with_llm: LLM returned unparseable response")
            return {"match": None}

        match = response.get("match")
        if match is None:
            return {"match": None}
        if not isinstance(match, int) or not (0 <= match < len(candidates)):
            logger.warning("match_triple_with_llm: LLM returned out-of-range match index %r", match)
            return {"match": None}

        direction = response.get("direction", "same")
        if direction not in ("same", "inverse"):
            direction = "same"

        return {"match": match, "direction": direction}

    def compare_types_with_llm(
        self,
        domain_pair: Optional[tuple],
        range_pair: Optional[tuple],
    ) -> Dict[str, Optional[str]]:
        """
        Compare our induced type against a reference ontology's type, for
        the domain (subject) side and/or the range (object) side of a
        matched relation instance. Each pair is (our_type_label,
        reference_type_label); pass None for a side that has no reference
        type to compare against (some Text2KGBench relations only specify a
        domain, or only a range).

        Returns {"domain": label|None, "range": label|None} where each
        label (when the corresponding pair was given) is one of "exact" /
        "more_general" / "more_narrow" / "not_related". A side that wasn't
        given, or came back unparseable, is None -- never fabricated.
        Does NOT catch exceptions from the underlying API call (see
        match_triple_with_llm's docstring for why).
        """
        if domain_pair is None and range_pair is None:
            return {"domain": None, "range": None}

        system_prompt = self.prompts["type_compare"]
        lines = []
        if domain_pair:
            lines.append(f"Domain/subject side -- our type: {domain_pair[0]!r}, reference type: {domain_pair[1]!r}")
        if range_pair:
            lines.append(f"Range/object side -- our type: {range_pair[0]!r}, reference type: {range_pair[1]!r}")
        user_prompt = "\n".join(lines)

        response = self.get_completion(system_prompt=system_prompt, user_prompt=user_prompt)

        logger.log(logging.DEBUG, f"domain_pair={domain_pair} range_pair={range_pair}")
        logger.log(logging.DEBUG, f"Response: {response}")

        valid_labels = ("exact", "more_general", "more_narrow", "not_related")
        if not isinstance(response, dict):
            logger.warning("compare_types_with_llm: LLM returned unparseable response")
            return {"domain": None, "range": None}

        result = {"domain": None, "range": None}
        if domain_pair and response.get("domain") in valid_labels:
            result["domain"] = response["domain"]
        if range_pair and response.get("range") in valid_labels:
            result["range"] = response["range"]
        return result

    def extract_entities_from_question(self, question: str) -> list:
        """
        Extract entity mentions from a natural-language question (used by
        QA evaluation to seed retrieval into a constructed KG -- see
        scripts/musique_qa_eval/run_qa_eval.py). Mirrors Wikontic's
        identify_relevant_entities_from_question_with_llm's first step.
        Returns whatever the LLM produced (normally a list of strings);
        callers must defensively handle a non-list response.
        """
        return self.get_completion(
            system_prompt=self.prompts["question_entity_extractor"],
            user_prompt=f"Question: {question}",
        )

    def identify_relevant_entities(
        self, question: str, entity_list: List[Dict[str, str]]
    ) -> list:
        """
        Re-rank/filter a batch of {"entity", "entity_type"} candidates
        (typically an embedding-retrieval shortlist) for relevance to a
        question. Returns whatever the LLM produced (normally a list of
        {"entity", "entity_type"} dicts); callers must defensively handle a
        non-list response.
        """
        return self.get_completion(
            system_prompt=self.prompts["question_entity_ranker"],
            user_prompt=f"Question: {question}\nEntities: {entity_list}",
        )

    def answer_question(self, question: str, triplets: List[dict]) -> str:
        """Answer a question using a list of {subject, relation, object,
        qualifiers} dicts retrieved from a constructed KG."""
        return self.get_completion(
            system_prompt=self.prompts["qa"],
            user_prompt=f'Question: {question}\n\nTriplets: "{triplets}"',
            transform_to_json=False,
        )

    def calculate_cost(self) -> float:
        """Calculate the total cost of API usage."""
        return self.current_cost / 1e6

    def calculate_used_tokens(self) -> int:
        """Calculate the total # of used tokens for generation"""
        return self.prompt_tokens_num, self.completion_tokens_num

    def get_usage_snapshot(self) -> Dict[str, float]:
        """Atomic (single-lock) read of the three running counters, for
        callers that need a consistent point-in-time snapshot -- e.g.
        pipeline.py diffing before/after a step to get that step's own
        token usage, where reading the three attributes one at a time could
        otherwise race against a concurrent get_completion() call landing
        mid-read (see the lock added in get_completion() itself)."""
        with self._token_lock:
            prompt_tokens = self.prompt_tokens_num
            completion_tokens = self.completion_tokens_num
            cost = self.current_cost
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cost_usd": cost / 1e6,
        }

    def reset_tokens(self):
        """Reset the total # of used tokens for generation"""
        self.prompt_tokens_num = 0
        self.completion_tokens_num = 0

    def reset_messages(self):
        """Reset the messages"""
        self.messages = []

    def reset_error_state(self):
        self._prev_error = None
        self._refine_attempt = 0
