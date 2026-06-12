import pytest

from tests.case_study_helpers import load_samples

semver = pytest.importorskip("semver")

MATCH_EXPRESSIONS = (">=0.0.0", "<999999999999.0.0", "!=0.0.0")


def _exercise_parsed_version(version):
    assert version is not None
    str(version)
    repr(version)
    tuple(version)
    version.to_tuple()
    version.to_dict()
    version.finalize_version()
    version.bump_major()
    version.bump_minor()
    version.bump_patch()
    version.bump_prerelease()
    version.bump_build()
    version.replace(patch=version.patch)
    version.compare("0.0.0")
    version.is_compatible(version)

    for expression in MATCH_EXPRESSIONS:
        version.match(expression)


def test_semver_parser_regression():
    try:
        samples = load_samples(default_filename="baseline-semver.json")
    except FileNotFoundError:
        pytest.skip("semver case-study data not available")

    valid_count = 0
    invalid_count = 0

    for sample in samples:
        is_valid = semver.Version.is_valid(sample)
        try:
            version = semver.Version.parse(sample)
        except ValueError:
            invalid_count += 1
            assert not is_valid
        else:
            valid_count += 1
            assert is_valid
            _exercise_parsed_version(version)

    assert valid_count + invalid_count == len(samples)


@pytest.mark.parametrize("optional_minor_and_patch", [False, True])
def test_semver_parser_entrypoints(optional_minor_and_patch):
    try:
        samples = load_samples(default_filename="baseline-semver.json")
    except FileNotFoundError:
        pytest.skip("semver case-study data not available")

    for sample in samples:
        try:
            version = semver.Version.parse(
                sample,
                optional_minor_and_patch=optional_minor_and_patch,
            )
        except ValueError:
            continue
        _exercise_parsed_version(version)
