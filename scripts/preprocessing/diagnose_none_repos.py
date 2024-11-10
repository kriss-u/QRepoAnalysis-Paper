import json
from pathlib import Path


def diagnose_repos():
    project_root = Path(__file__).parent.parent.parent
    repos_file = project_root / "data" / "raw" / "repos_for_analysis_all.txt"
    repos_data_dir = project_root / "data" / "raw" / "repos_combined_all"

    issues = {
        "not_found": [],
        "empty_files": [],
        "invalid_json": [],
        "missing_fields": [],
    }

    with open(repos_file, "r") as f:
        repos = [line.strip() for line in f if line.strip()]

    print(f"Total repositories to check: {len(repos)}")

    for repo in repos:
        file_path = repos_data_dir / f"{repo.replace('/', '+')}.json"

        if not file_path.exists():
            issues["not_found"].append(repo)
            continue

        if file_path.stat().st_size == 0:
            issues["empty_files"].append(repo)
            continue

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()

            try:
                data = json.loads(content)

                required_fields = [
                    "owner",
                    "name",
                    "created_at",
                    "updated_at",
                    "language",
                ]
                missing = [
                    field
                    for field in required_fields
                    if field not in data or data.get(field) is None
                ]

                if missing:
                    issues["missing_fields"].append({"repo": repo, "missing": missing})

            except json.JSONDecodeError:
                issues["invalid_json"].append(repo)

        except Exception as e:
            print(f"Error processing {repo}: {str(e)}")

    print("\nDiagnostic Report")
    print("================")

    print(f"\nFiles not found ({len(issues['not_found'])}):")
    for repo in issues["not_found"]:
        print(f"  - {repo}")

    print(f"\nEmpty files ({len(issues['empty_files'])}):")
    for repo in issues["empty_files"]:
        print(f"  - {repo}")

    print(f"\nInvalid JSON ({len(issues['invalid_json'])}):")
    for repo in issues["invalid_json"]:
        print(f"  - {repo}")

    print(f"\nMissing fields ({len(issues['missing_fields'])}):")
    for item in issues["missing_fields"]:
        print(f"  - {item['repo']}")
        print(f"    Missing: {', '.join(item['missing'])}")

    report_path = project_root / "data" / "raw" / "repo_diagnostic_report.json"
    with open(report_path, "w") as f:
        json.dump(issues, f, indent=2)

    print(f"\nDetailed report saved to {report_path}")


if __name__ == "__main__":
    diagnose_repos()
