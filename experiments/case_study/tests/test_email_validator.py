import itertools

import pytest

from tests.case_study_helpers import (
    get_selected_dataset_name,
    is_full_dataset_run,
    load_samples,
)
from tests.expected_metrics import EMAIL_EXPECTED_ERROR_COUNTS
from email_validator import EmailNotValidError, validate_email


EMAIL_VALIDATION_OPTIONS = tuple(
    itertools.product([True, False], repeat=3)
)


def _email_option_id(options):
    allow_domain_literal, allow_quoted_local, allow_empty_local = options
    return (
        "domain_literal="
        f"{allow_domain_literal},quoted_local={allow_quoted_local},"
        f"empty_local={allow_empty_local}"
    )


EMAIL_VALIDATION_OPTION_IDS = [
    _email_option_id(options) for options in EMAIL_VALIDATION_OPTIONS
]


@pytest.mark.parametrize(
    ("allow_domain_literal", "allow_quoted_local", "allow_empty_local"),
    EMAIL_VALIDATION_OPTIONS,
    ids=EMAIL_VALIDATION_OPTION_IDS,
)
def test_email_validation_regression(
    allow_domain_literal,
    allow_quoted_local,
    allow_empty_local,
):
    dataset_name = get_selected_dataset_name("baseline-email.json")
    samples = load_samples(default_filename="baseline-email.json")

    normalized_count = 0
    error_count = 0

    for email in samples:
        try:
            result = validate_email(
                email,
                check_deliverability=False,
                allow_domain_literal=allow_domain_literal,
                allow_quoted_local=allow_quoted_local,
                allow_empty_local=allow_empty_local,
            )
        except EmailNotValidError:
            error_count += 1
        else:
            normalized_count += 1
            assert result is not None
            assert result.normalized is not None

    assert normalized_count + error_count == len(samples)

    if is_full_dataset_run():
        expected_error_count = EMAIL_EXPECTED_ERROR_COUNTS[dataset_name][
            (
                allow_domain_literal,
                allow_quoted_local,
                allow_empty_local,
            )
        ]
        assert error_count == expected_error_count
