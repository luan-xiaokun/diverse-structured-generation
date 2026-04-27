import json

from scripts.repro_results import build_metadata, write_json


def test_build_metadata_includes_stable_keys():
    metadata = build_metadata()

    assert set(metadata) >= {"timestamp_utc", "git_commit", "python", "platform"}
    assert isinstance(metadata["timestamp_utc"], str)
    assert metadata["timestamp_utc"].endswith("Z")
    assert isinstance(metadata["python"], str)
    assert isinstance(metadata["platform"], str)


def test_build_metadata_merges_extra_values():
    metadata = build_metadata({"string_kernel_backend": "python"})

    assert metadata["string_kernel_backend"] == "python"


def test_write_json_creates_parent_directories(tmp_path):
    output_path = tmp_path / "nested" / "result.json"

    write_json(output_path, {"schema_version": 1, "value": 3})

    with output_path.open() as f:
        data = json.load(f)
    assert data == {"schema_version": 1, "value": 3}
