import pytest
import webcolors

from tests.case_study_helpers import (
    get_selected_dataset_name,
    is_full_dataset_run,
    load_samples,
)
from tests.expected_metrics import WEBCOLORS_EXPECTED_ERROR_COUNTS


CONVERSION_FUNCTIONS = (
    webcolors.html5_parse_legacy_color,
    webcolors.html5_parse_simple_color,
    webcolors.hex_to_name,
    webcolors.hex_to_rgb,
    webcolors.hex_to_rgb_percent,
    webcolors.name_to_hex,
    webcolors.name_to_rgb,
    webcolors.name_to_rgb_percent,
)


@pytest.mark.parametrize(
    "conversion_func",
    CONVERSION_FUNCTIONS,
    ids=lambda func: func.__name__,
)
def test_webcolors_regression(conversion_func):
    dataset_name = get_selected_dataset_name("baseline-css-color.json")
    samples = load_samples(default_filename="baseline-css-color.json")

    success_count = 0
    error_count = 0

    for css_color in samples:
        try:
            conversion_func(css_color)
        except Exception:
            error_count += 1
        else:
            success_count += 1

    assert success_count + error_count == len(samples)

    if is_full_dataset_run():
        expected_error_count = WEBCOLORS_EXPECTED_ERROR_COUNTS[dataset_name][
            conversion_func.__name__
        ]
        assert error_count == expected_error_count
