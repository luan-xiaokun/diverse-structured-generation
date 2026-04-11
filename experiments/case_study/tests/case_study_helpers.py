import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def get_selected_dataset_name(default_filename):
    return os.environ.get("TEST_DATA_FILENAME", default_filename)


def is_full_dataset_run():
    return os.environ.get("TEST_DATA_SAMPLE_SIZE") is None


def load_samples(default_filename):
    data_filename = get_selected_dataset_name(default_filename)
    sample_size = os.environ.get("TEST_DATA_SAMPLE_SIZE")
    file_path = DATA_DIR / data_filename

    with file_path.open(encoding="utf-8") as file_obj:
        payload = json.load(file_obj)

    samples = payload["samples"]
    if sample_size is not None:
        samples = samples[: int(sample_size)]

    return samples
