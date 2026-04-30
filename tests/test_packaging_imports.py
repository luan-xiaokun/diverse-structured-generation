"""Packaging boundary tests for public import paths."""

import importlib.util

from diverse_guide import DiverseGuide


def test_public_diverseguide_api_is_exported():
    assert DiverseGuide.__name__ == "DiverseGuide"


def test_evaluation_modules_use_package_namespace():
    assert importlib.util.find_spec("diverse_guide.evaluation.metrics") is not None
    assert importlib.util.find_spec("diverse_guide.evaluation.paths") is not None
    assert importlib.util.find_spec("diverse_guide.evaluation.perplexity") is not None
    assert (
        importlib.util.find_spec("diverse_guide.evaluation.string_kernel") is not None
    )
    assert (
        importlib.util.find_spec("diverse_guide.evaluation.string_kernel_py")
        is not None
    )


def test_old_top_level_evaluation_modules_are_not_public():
    for module_name in [
        "metrics",
        "paths",
        "perplexity",
        "string_kernel",
        "string_kernel_py",
    ]:
        assert importlib.util.find_spec(module_name) is None
