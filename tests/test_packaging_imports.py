"""Packaging boundary tests for public import paths."""

import importlib.util

from diverse_guide import DiverseGuide


def test_public_import_paths():
    assert DiverseGuide.__name__ == "DiverseGuide"
    modules = ["metrics", "paths", "perplexity", "string_kernel", "string_kernel_py"]
    for module_name in modules:
        # Evaluation modules live in the package namespace only.
        assert importlib.util.find_spec(f"diverse_guide.evaluation.{module_name}")
        assert importlib.util.find_spec(module_name) is None
