import argparse
import sys
from pathlib import Path
from scripts.utils.logger import setup_logger
from scripts.config.constants import PROCESSED_DATA_DIR, PROJECT_ROOT, RESULTS_DIR


def setup_preprocessing_parser(subparsers):
    prep_parser = subparsers.add_parser(
        "preprocess", help="Data preprocessing commands"
    )
    prep_subparsers = prep_parser.add_subparsers(dest="subcommand", required=True)

    clean_parser = prep_subparsers.add_parser(
        "clean-repos", help="Clean repository JSON data files"
    )
    clean_parser.add_argument("--input-dir", type=Path, help="Input directory override")
    clean_parser.add_argument(
        "--output-dir", type=Path, help="Output directory override"
    )


def setup_rq1_parser(subparsers):
    rq1_parser = subparsers.add_parser("rq1", help="Research Question 1 analysis")
    rq1_subparsers = rq1_parser.add_subparsers(dest="subcommand", required=True)

    create_parser = rq1_subparsers.add_parser(
        "create-dataset", help="Create analysis dataset"
    )
    create_parser.add_argument(
        "--input-dir", type=Path, help="Input directory override"
    )
    create_parser.add_argument(
        "--output-dir", type=Path, help="Output directory override"
    )

    # Analyze popularity command
    popularity_parser = rq1_subparsers.add_parser(
        "popularity", help="Analyze quantum computing popularity"
    )
    popularity_parser.add_argument(
        "--granularity",
        choices=["week", "month", "quarter", "year"],
        default="quarter",
        help="Time granularity for analysis",
    )

    popularity_with_updated_date_parser = rq1_subparsers.add_parser(
        "popularity-updated",
        help="Analyze quantum computing popularity including updated date of repo",
    )
    popularity_with_updated_date_parser.add_argument(
        "--granularity",
        choices=["week", "month", "quarter", "year"],
        default="quarter",
        help="Time granularity for analysis",
    )

    # Author analysis command
    author_parser = rq1_subparsers.add_parser(
        "analyze-authors", help="Analyze top authors from commit patterns"
    )
    author_parser.add_argument(
        "--commit-patterns-dir",
        type=Path,
        default=PROCESSED_DATA_DIR / "rq2" / "commit_patterns",
        help="Directory containing commit patterns data",
    )
    author_parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR / "rq1/author_analysis",
        help="Output directory override (default: data/processed/rq1/author_analysis)",
    )
    author_parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top authors to analyze (default: 20)",
    )

    # Repo star vs contributor count command
    repo_star_contributor_parser = rq1_subparsers.add_parser(
        "compare-repo-contributors",
        help="Analyze relationship between repository stars and contributor count",
    )
    repo_star_contributor_parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR / "rq1/repo_star_contributor",
        help="Output directory override (default: results/rq1/repo_star_contributor)",
    )

    # Programming language and frameworks trends command
    lang_framework_parser = rq1_subparsers.add_parser(
        "lang-framework-trends",
        help="Analyze programming language and framework trends",
    )

    lang_framework_parser.add_argument(
        "--granularity",
        choices=["week", "month", "quarter", "year"],
        default="quarter",
        help="Time granularity for analysis",
    )

    lang_framework_parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR / "rq1/lang_framework_trends",
        help="Output directory override (default: results/rq1/lang_framework_trends)",
    )

    # Repository classification trends command
    repo_classification_parser = rq1_subparsers.add_parser(
        "repo-classification-trends",
        help="Analyze repository classification trends",
    )

    repo_classification_parser.add_argument(
        "--granularity",
        choices=["week", "month", "quarter", "year"],
        default="quarter",
        help="Time granularity for analysis",
    )

    repo_classification_parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR / "rq1/repo_classification",
        help="Output directory override (default: results/rq1/repo_classification)",
    )


def setup_rq2_parser(subparsers):
    rq2_parser = subparsers.add_parser("rq2", help="Research Question 2 analysis")
    rq2_subparsers = rq2_parser.add_subparsers(dest="subcommand", required=True)

    create_parser = rq2_subparsers.add_parser(
        "create-dataset", help="Create commit patterns analysis dataset"
    )
    create_parser.add_argument(
        "--volume-path",
        type=Path,
        required=True,
        help="Path to volume containing originals and forks directories",
    )
    create_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory override (default: data/processed/rq2)",
    )
    create_parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of repositories to process in each batch",
    )

    create_parser = rq2_subparsers.add_parser(
        "create-dataset-classified",
        help="Create commit patterns analysis dataset with classification",
    )
    create_parser.add_argument(
        "--volume-path",
        type=Path,
        required=True,
        help="Path to volume containing originals and forks directories",
    )
    create_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory override (default: data/processed/rq2)",
    )
    create_parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of repositories to process in each batch",
    )

    # Temporal patterns command
    temporal_parser = rq2_subparsers.add_parser(
        "temporal-patterns", help="Analyze commit patterns over time"
    )
    temporal_parser.add_argument(
        "--granularity",
        choices=["day", "week", "month", "quarter", "year"],
        default="quarter",
        help="Time granularity for analysis",
    )
    temporal_parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results/rq2/temporal_patterns",
        help="Directory to save analysis results",
    )

    # Analyze commit sizes command
    commit_sizes_parser = rq2_subparsers.add_parser(
        "analyze-commit-sizes", help="Analyze commit sizes"
    )

    commit_sizes_parser.add_argument(
        "--granularity",
        choices=["day", "week", "month", "quarter", "year"],
        default="quarter",
        help="Time granularity for analysis",
    )

    commit_sizes_parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results/rq2/commit_sizes",
        help="Directory to save analysis results",
    )

    # Contributors growth command
    contributors_growth_parser = rq2_subparsers.add_parser(
        "contributors-growth", help="Analyze contributors growth"
    )
    contributors_growth_parser.add_argument(
        "--granularity",
        choices=["day", "week", "month", "quarter", "year"],
        default="quarter",
        help="Time granularity for analysis",
    )
    contributors_growth_parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results/rq2/contributors_growth",
        help="Directory to save analysis results",
    )

    # Commits classification command
    commits_classification_parser = rq2_subparsers.add_parser(
        "commits-classification", help="Analyze commit classification"
    )

    commits_classification_parser.add_argument(
        "--granularity",
        choices=["day", "week", "month", "quarter", "year"],
        default="quarter",
        help="Time granularity for analysis",
    )

    commits_classification_parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results/rq2/commits_classification",
        help="Directory to save analysis results",
    )


def main():
    parser = argparse.ArgumentParser(description="Quantum Computing Research Analysis")
    subparsers = parser.add_subparsers(dest="command", required=True)

    setup_preprocessing_parser(subparsers)
    setup_rq1_parser(subparsers)
    setup_rq2_parser(subparsers)

    args = parser.parse_args()

    logger = setup_logger(__name__, args.command, f"{args.command}_{args.subcommand}")
    logger.info(f"Starting {args.command} - {args.subcommand}")

    try:
        if args.command == "rq1":
            if args.subcommand == "create-dataset":
                from scripts.rq1.create_dataset import create_dataset

                create_dataset(args)
            elif args.subcommand == "popularity":
                from scripts.rq1.popularity import analyze_popularity

                analyze_popularity(args)

            elif args.subcommand == "popularity-updated":
                from scripts.rq1.popularity_updated import analyze_popularity_updated

                analyze_popularity_updated(args)
            elif args.subcommand == "analyze-authors":
                from scripts.rq1.author_analysis import analyze_authors

                analyze_authors(args.commit_patterns_dir, args.output_dir, args.top_n)

            elif args.subcommand == "lang-framework-trends":
                from scripts.rq1.language_trend import analyze_language_trends

                analyze_language_trends(args)

            elif args.subcommand == "repo-classification-trends":
                from scripts.rq1.repo_classification_trend import (
                    analyze_repo_classification_trends,
                )

                analyze_repo_classification_trends(args)

        elif args.command == "preprocess":
            if args.subcommand == "clean-repos":
                from scripts.preprocessing.clean_repos_data import clean_repo_files

                clean_repo_files(args)

        elif args.command == "rq2":
            if args.subcommand == "create-dataset":
                from scripts.rq2.create_dataset import create_dataset

                create_dataset(args)

            elif args.subcommand == "create-dataset-classified":
                from scripts.rq2.create_commits_classified import (
                    create_dataset as create_dataset_classified,
                )

                create_dataset_classified(args)

            elif args.subcommand == "temporal-patterns":
                from scripts.rq2.temporal_patterns import analyze_temporal_patterns

                analyze_temporal_patterns(args)

            elif args.subcommand == "analyze-commit-sizes":
                from scripts.rq2.commit_sizes import analyze_commit_sizes_with_log

                analyze_commit_sizes_with_log(args)

            elif args.subcommand == "contributors-growth":
                from scripts.rq2.contributors_growth import analyze_contributors_growth

                analyze_contributors_growth(args)

            elif args.subcommand == "commits-classification":
                from scripts.rq2.commits_classification import (
                    analyze_commits_classification,
                )

                analyze_commits_classification(args)

    except Exception as e:
        logger.error(f"Error during execution: {str(e)}", exc_info=True)
        sys.exit(1)

    logger.info("Analysis completed successfully")


if __name__ == "__main__":
    main()
