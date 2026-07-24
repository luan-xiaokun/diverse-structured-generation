import subprocess
import sys
from pathlib import Path


def test_nested_package_imports_from_project_directory():
    project_dir = Path(__file__).parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib.util, aiap; spec=importlib.util.find_spec('aiap'); "
            "assert spec and spec.origin and 'aiap/aiap/' in"
            " spec.origin.replace('\\\\', '/')",
        ],
        cwd=project_dir,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0


def test_catalog_is_deterministic_and_json_safe(dfa_multi):
    from aiap.obligations import ObligationCatalog

    catalog = ObligationCatalog.from_dfa(dfa_multi)
    assert [item.obligation_id for item in catalog] == [
        item.obligation_id for item in ObligationCatalog.from_dfa(dfa_multi)
    ]
    initial = dfa_multi.get_initial_state()
    assert catalog.by_id(f"state:{initial}").raw_meaning == initial
    transition = next(item for item in catalog if item.kind == "transition")
    assert isinstance(transition.raw_meaning, list)
    assert transition.obligation_id.startswith("transition:")
    assert all(item.kind in {"state", "transition", "path"} for item in catalog)
