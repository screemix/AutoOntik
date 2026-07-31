"""
Tests for pipeline.load_seed_roots() -- loading an external seed-ontology
file (e.g. configs/seeds/dolce.yaml) into the shape
hierarchy_induction.HierarchyConfig.seed_roots expects. Pure YAML I/O, no
LLM/embedder involved.
"""

from __future__ import annotations

import pytest

from src.ontodisco.pipeline import load_seed_roots


def test_loads_wrapped_seed_roots_key(tmp_path):
    path = tmp_path / "seeds.yaml"
    path.write_text(
        "seed_roots:\n"
        "  - label: Endurant\n"
        "    children:\n"
        "      - label: Physical Endurant\n"
        "  - Perdurant\n"
    )

    seed_roots = load_seed_roots(path)

    assert seed_roots == [
        {"label": "Endurant", "children": [{"label": "Physical Endurant"}]},
        "Perdurant",
    ]


def test_loads_bare_top_level_list(tmp_path):
    path = tmp_path / "seeds.yaml"
    path.write_text("- Endurant\n- Perdurant\n")

    assert load_seed_roots(path) == ["Endurant", "Perdurant"]


def test_missing_seed_roots_key_returns_empty_list(tmp_path):
    path = tmp_path / "seeds.yaml"
    path.write_text("some_other_key: 1\n")

    assert load_seed_roots(path) == []


def test_invalid_top_level_shape_raises(tmp_path):
    path = tmp_path / "seeds.yaml"
    path.write_text("just a string, not a list or mapping\n")

    with pytest.raises(ValueError):
        load_seed_roots(path)
