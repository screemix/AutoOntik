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
                "hierarchy_cluster_action": "hierarchy_cluster_action.txt",
                "relation_direction": "relation_direction.txt",
                "hierarchy_label_disambiguation": "hierarchy_label_disambiguation.txt",
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
        patterns = [
            r"```json\s*(\{.*?\}|\[.*?\])\s*```",  # JSON in code blocks
            r"(\{.*?\}|\[.*?\])",  # Inline JSON
        ]

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    logger.error("Failed to parse JSON: %s", text)

        return text

    # @retry(
    #     wait=wait_random_exponential(multiplier=1, max=60),
    #     before_sleep=before_sleep_log(logger, logging.ERROR),
    #     stop=stop_after_attempt(5),
    # )
    @tenacity.retry(stop=stop_after_attempt(5), reraise=True)
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

    def resolve_hierarchy_cluster(
        self, members: list[str],
        member_context: Optional[Dict[str, str]] = None,
    ) -> list[dict]:
        """Ask the LLM to organise one cluster of type labels into subClassOf
        groups, for hierarchy induction (see hierarchy_induction.py).

        members: candidate labels in this cluster. May include labels that are
        themselves an abstraction formed in an earlier round of the same
        recursive process, not just raw corpus surface forms.
        member_context: optional label -> relation-signature evidence string
        (e.g. "directed, produced, starred in"), shown alongside each
        candidate so the LLM has concrete distributional evidence.

        Returns the raw "groups" list from the LLM response (each a dict with
        "action" ["merge"|"no_parent"|"same_concept"], "members", and for
        merges "parent"/"parent_is_new"/"parent_definition"/"confidence").
        "same_concept" groups have no "parent" -- they mark members as the
        identical concept under different wording, to be collapsed rather
        than given a hierarchy edge. The caller is responsible for validating
        group contents and mapping labels back to type_ids.
        """
        system_prompt = self.prompts["hierarchy_cluster_action"]

        if member_context:
            candidates_str = ", ".join(
                f"{m} (context: {member_context[m]})" if member_context.get(m) else m
                for m in members
            )
        else:
            candidates_str = ", ".join(members)

        response = self.get_completion(
            system_prompt=system_prompt, user_prompt=f"Types: {candidates_str}"
        )

        logger.log(logging.DEBUG, f"Input: {members}")
        logger.log(logging.DEBUG, f"Response: {response}")

        if isinstance(response, str):
            logger.warning("resolve_hierarchy_cluster: LLM returned unparseable string")
            return []

        groups = response.get("groups", [])
        if not groups:
            logger.warning("resolve_hierarchy_cluster: LLM response had no 'groups' key")

        return groups

    def disambiguate_duplicate_labels(
        self, groups: list[Dict],
    ) -> Dict[str, list[list[str]]]:
        """
        Ask the LLM to partition candidate type nodes that share an identical
        label into same-concept clusters (see hierarchy_induction.py's
        _reconcile_duplicate_labels -- called only for the subset of
        duplicate-labeled groups whose relation-argument profiles couldn't
        already confirm or deny sameness on their own).

        groups: list of {"label": str, "candidates": [{"id", "definition",
        "relation_context"}, ...]}.

        Returns label -> list of clusters (each a list of candidate ids).
        Falls back to "every candidate is its own cluster" (i.e. no merge)
        for any label missing from the response, or any candidate id the
        response dropped -- an unconfirmed merge is a worse failure mode
        here than a leftover duplicate, which just gets re-tried next round.
        """
        system_prompt = self.prompts["hierarchy_label_disambiguation"]
        payload = [
            {
                "label": group["label"],
                "candidates": [
                    {"id": c["id"], "definition": c.get("definition", ""),
                     "relation_context": c.get("relation_context", "")}
                    for c in group["candidates"]
                ],
            }
            for group in groups
        ]
        response = self.get_completion(
            system_prompt=system_prompt, user_prompt=f"Groups: {json.dumps(payload)}"
        )

        logger.log(logging.DEBUG, f"Input: {payload}")
        logger.log(logging.DEBUG, f"Response: {response}")

        result: Dict[str, list[list[str]]] = {}
        parsed = response if isinstance(response, dict) else {}
        if not isinstance(response, dict):
            logger.warning("disambiguate_duplicate_labels: LLM returned unparseable response")

        for group in groups:
            label = group["label"]
            all_ids = [c["id"] for c in group["candidates"]]
            raw_clusters = parsed.get(label)

            if not isinstance(raw_clusters, list):
                result[label] = [[tid] for tid in all_ids]
                continue

            seen: set = set()
            clusters: list[list[str]] = []
            for raw_cluster in raw_clusters:
                if not isinstance(raw_cluster, list):
                    continue
                cluster = [tid for tid in raw_cluster if tid in all_ids and tid not in seen]
                if not cluster:
                    continue
                seen.update(cluster)
                clusters.append(cluster)

            # Any candidate id the response silently dropped stays its own
            # (unmerged) cluster rather than being lost.
            for tid in all_ids:
                if tid not in seen:
                    clusters.append([tid])

            result[label] = clusters

        return result

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

    def calculate_cost(self) -> float:
        """Calculate the total cost of API usage."""
        return self.current_cost / 1e6

    def calculate_used_tokens(self) -> int:
        """Calculate the total # of used tokens for generation"""
        return self.prompt_tokens_num, self.completion_tokens_num

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
