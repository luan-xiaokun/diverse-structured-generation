import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

RUNS = (
    {
        "library": "email_validator",
        "setting": "baseline",
        "grammar": "email",
        "coverage_data": ".coverage.baseline.email",
        "html_dir": "htmlcov_baseline_email",
    },
    {
        "library": "email_validator",
        "setting": "diverse",
        "grammar": "email",
        "coverage_data": ".coverage.diverse.email",
        "html_dir": "htmlcov_diverse_email",
    },
    {
        "library": "webcolors",
        "setting": "baseline",
        "grammar": "css-color",
        "coverage_data": ".coverage.baseline.css-color",
        "html_dir": "htmlcov_baseline_css-color",
    },
    {
        "library": "webcolors",
        "setting": "diverse",
        "grammar": "css-color",
        "coverage_data": ".coverage.diverse.css-color",
        "html_dir": "htmlcov_diverse_css-color",
    },
)


def load_json_report(path):
    with path.open(encoding="utf-8") as file_obj:
        return json.load(file_obj)


def summarize_run(run):
    json_path = PROJECT_ROOT / f"{run['coverage_data']}.json"
    report = load_json_report(json_path)
    totals = report["totals"]
    total_coverage_percentage = round(totals["percent_covered"], 2)

    return {
        "library": run["library"],
        "setting": run["setting"],
        "grammar": run["grammar"],
        "coverage_data_file": run["coverage_data"],
        "coverage_json_file": json_path.name,
        "html_report_dir": run["html_dir"],
        "totals": {
            "covered_lines": totals["covered_lines"],
            "num_statements": totals["num_statements"],
            "covered_branches": totals["covered_branches"],
            "num_branches": totals["num_branches"],
            "total_coverage_percentage": total_coverage_percentage,
            "missing_lines": totals["missing_lines"],
            "excluded_lines": totals["excluded_lines"],
            "num_partial_branches": totals["num_partial_branches"],
        },
    }


def main():
    runs = [summarize_run(run) for run in RUNS]
    summary = {"runs": runs}

    output_path = PROJECT_ROOT / "case_study_summary.json"
    with output_path.open("w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2, sort_keys=True)
        file_obj.write("\n")

    print(f"Wrote summary to {output_path.name}")


if __name__ == "__main__":
    main()
