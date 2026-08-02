"""
Parse a Text2KGBench ontology .ttl file into plain dicts
============================================================

Text2KGBench (https://github.com/cenguix/Text2KGBench) ships one small OWL
ontology per domain (e.g. data/wikidata_tekgen/ontologies/owl/ont_2_music.ttl):
a handful of owl:Class blocks (each with an rdfs:label, and an optional
rdfs:subClassOf) and owl:ObjectProperty blocks (each with an rdfs:label and
an optional rdfs:domain / rdfs:range). The file is machine-generated and
very regular -- one blank-line-separated block per subject IRI -- so this is
a small hand-rolled block parser targeting exactly that shape, not a general
Turtle parser. No rdflib dependency needed.

This is only used by run_judge_eval.py, to get each gold relation's
domain/range class labels for the type-comparison judge (CLAUDE.md's eval
design notes: Text2KGBench ground-truth triples carry no per-triple types at
all -- only sub/rel/obj -- so domain/range have to come from here, once per
matched relation, not once per triple).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_SUBJECT_RE = re.compile(r"^<([^>]+)>\s+a\s+owl:(Class|ObjectProperty)\b")
_LABEL_RE = re.compile(r'rdfs:label\s+"([^"]*)"')
_SUBCLASS_RE = re.compile(r"rdfs:subClassOf\s+<([^>]+)>")
_DOMAIN_RE = re.compile(r"rdfs:domain\s+<([^>]+)>")
_RANGE_RE = re.compile(r"rdfs:range\s+<([^>]+)>")


@dataclass
class Text2KGOntology:
    class_label_by_iri: dict[str, str] = field(default_factory=dict)
    class_parent_iri: dict[str, Optional[str]] = field(default_factory=dict)   # class iri -> parent class iri, or None
    # relation label (exactly as it appears in ground_truth.jsonl's "rel" field) ->
    # {"domain": class label or None, "range": class label or None}
    relations: dict[str, dict] = field(default_factory=dict)

    def parent_label(self, class_label: str) -> Optional[str]:
        """Walk one step up class_label's rdfs:subClassOf chain, by label."""
        iri = next((i for i, lbl in self.class_label_by_iri.items() if lbl == class_label), None)
        if iri is None:
            return None
        parent_iri = self.class_parent_iri.get(iri)
        if parent_iri is None:
            return None
        return self.class_label_by_iri.get(parent_iri)


def parse_ontology(ttl_path: str | Path) -> Text2KGOntology:
    text = Path(ttl_path).read_text(encoding="utf-8")
    blocks = re.split(r"\n\s*\n", text)

    ontology = Text2KGOntology()
    relation_blocks: list[tuple[str, Optional[str], Optional[str]]] = []  # (label, domain_iri, range_iri)

    for block in blocks:
        m = _SUBJECT_RE.search(block)
        if not m:
            continue  # e.g. the leading owl:Ontology metadata block
        kind = m.group(2)
        label_m = _LABEL_RE.search(block)
        label = label_m.group(1) if label_m else None
        if label is None:
            continue

        if kind == "Class":
            iri = m.group(1)
            ontology.class_label_by_iri[iri] = label
            sub_m = _SUBCLASS_RE.search(block)
            ontology.class_parent_iri[iri] = sub_m.group(1) if sub_m else None
        else:  # ObjectProperty
            domain_m = _DOMAIN_RE.search(block)
            range_m = _RANGE_RE.search(block)
            relation_blocks.append((
                label,
                domain_m.group(1) if domain_m else None,
                range_m.group(1) if range_m else None,
            ))

    for label, domain_iri, range_iri in relation_blocks:
        ontology.relations[label] = {
            "domain": ontology.class_label_by_iri.get(domain_iri) if domain_iri else None,
            "range": ontology.class_label_by_iri.get(range_iri) if range_iri else None,
        }

    return ontology
