import pytest

from experiments.semver_setss_2026.protocol import (
    SEMVER_COMPILER_REGEX,
    SEMVER_OUTLINES_REGEX,
    SEMVER_REGEX_SOURCE,
    build_semver_dfa,
    dfa_fullmatch,
)
from experiments.semver_setss_2026.runner import (
    classify_attempt,
    summarize_attempt_records,
)


@pytest.mark.parametrize(
    "value",
    [
        "0.0.0",
        "1.2.3",
        "1.0.0-alpha",
        "1.0.0-alpha.1",
        "1.0.0-0.3.7",
        "1.0.0-x.7.z.92",
        "1.0.0+20130313144700",
        "1.0.0+001",
        "1.0.0-beta+exp.sha.5114f85",
        "2.7.1-rc.1+build.123",
        "1.2.3-01a",
    ],
)
def test_generator_dfa_accepts_semver_examples(value):
    assert dfa_fullmatch(build_semver_dfa(), value)


@pytest.mark.parametrize(
    "value",
    [
        "",
        "v1.2.3",
        "1.2",
        "1.2.3.4",
        "01.2.3",
        "1.02.3",
        "1.2.03",
        "1.2.3-01",
        "1.2.3-",
        "1.2.3+",
        "1.2.3_alpha",
        "1.2.3-..",
        "1.2.3+abc..def",
    ],
)
def test_generator_dfa_rejects_non_semver_examples(value):
    assert not dfa_fullmatch(build_semver_dfa(), value)


def test_compiler_regex_uses_existing_fullmatch_normalization():
    assert SEMVER_COMPILER_REGEX.startswith("(?:")
    assert SEMVER_COMPILER_REGEX.endswith(")$")
    assert r"\d" in SEMVER_REGEX_SOURCE
    assert r"\d" in SEMVER_OUTLINES_REGEX
    assert r"\d" not in SEMVER_COMPILER_REGEX


def test_eos_consumes_budget_and_marks_valid_output_complete():
    result = classify_attempt([10, 99], eos_token_id=99, max_tokens=2, valid=True)

    assert result == {
        "completed": True,
        "token_budget_failure": False,
        "invalid_completed": False,
        "termination_reason": "eos",
    }


def test_historical_runner_accepts_a_valid_fullmatch_at_the_token_budget():
    result = classify_attempt([10, 11], eos_token_id=99, max_tokens=2, valid=True)

    assert result == {
        "completed": True,
        "token_budget_failure": False,
        "invalid_completed": False,
        "termination_reason": "max_tokens_valid_fullmatch",
    }


def test_invalid_prefix_at_the_token_budget_is_a_failure():
    result = classify_attempt([10, 11], eos_token_id=99, max_tokens=2, valid=False)

    assert result == {
        "completed": False,
        "token_budget_failure": True,
        "invalid_completed": False,
        "termination_reason": "max_tokens_incomplete",
    }


def test_eos_on_an_invalid_decode_is_counted_as_invalid_completed():
    result = classify_attempt([10, 99], eos_token_id=99, max_tokens=2, valid=False)

    assert result == {
        "completed": False,
        "token_budget_failure": False,
        "invalid_completed": True,
        "termination_reason": "eos_invalid",
    }


def test_attempt_summary_keeps_runs_and_completed_samples_distinct():
    records = [
        {
            "completed": True,
            "valid": True,
            "token_budget_failure": False,
            "invalid_completed": False,
            "termination_reason": "eos",
        },
        {
            "completed": False,
            "valid": False,
            "token_budget_failure": True,
            "invalid_completed": False,
            "termination_reason": "max_tokens_incomplete",
        },
        {
            "completed": True,
            "valid": True,
            "token_budget_failure": False,
            "invalid_completed": False,
            "termination_reason": "max_tokens_valid_fullmatch",
        },
    ]

    assert summarize_attempt_records(records) == {
        "attempts": 3,
        "completed": 2,
        "token_budget_failures": 1,
        "invalid_completed": 0,
        "completion_rate": 2 / 3,
        "eos_completed": 1,
        "valid_fullmatch_at_budget": 1,
    }
