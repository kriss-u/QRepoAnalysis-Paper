"""
Dataset Loading Tutorial for QRepoAnalysis Research Project

This script demonstrates how to load the four main datasets using PyArrow:
1. data/processed/issues.parquet - Issue data
2. data/processed/rq1/repos.parquet - Repository metadata
3. data/processed/rq2/commit_patterns - Commits data
4. data/processed/rq2/commit_patterns_classified - Commits data with classifications

The script focuses on loading parquet directories and showing basic dataset information.
"""

import pyarrow.parquet as pq
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


def load_parquet_directory(dataset_path, dataset_name):
    """
    Load a parquet dataset from a directory containing multiple parquet files.

    Args:
        dataset_path (Path): Path to the directory containing parquet files
        dataset_name (str): Name of the dataset for display purposes

    Returns:
        pandas.DataFrame or None: The loaded dataframe or None if failed
    """
    try:
        print(f"\n📂 Loading {dataset_name} from: {dataset_path}")

        # Check if path exists and is a directory
        if not dataset_path.exists():
            print(f"   ❌ Path does not exist: {dataset_path}")
            return None

        if not dataset_path.is_dir():
            print(f"   ❌ Path is not a directory: {dataset_path}")
            return None

        # Load using PyArrow ParquetDataset for directories
        dataset = pq.ParquetDataset(dataset_path)
        table = dataset.read()
        df = table.to_pandas()

        print(
            f"   ✅ Successfully loaded {len(df):,} rows and {len(df.columns)} columns"
        )

        return df

    except Exception as e:
        print(f"   ❌ Error loading dataset: {str(e)}")
        return None


def load_parquet_file(dataset_path, dataset_name):
    """
    Load a single parquet file.

    Args:
        dataset_path (Path): Path to the parquet file
        dataset_name (str): Name of the dataset for display purposes

    Returns:
        pandas.DataFrame or None: The loaded dataframe or None if failed
    """
    try:
        print(f"\n📄 Loading {dataset_name} from: {dataset_path}")

        # Check if file exists
        if not dataset_path.exists():
            print(f"   ❌ File does not exist: {dataset_path}")
            return None

        # Load using PyArrow
        table = pq.read_table(dataset_path)
        df = table.to_pandas()

        print(
            f"   ✅ Successfully loaded {len(df):,} rows and {len(df.columns)} columns"
        )

        return df

    except Exception as e:
        print(f"   ❌ Error loading dataset: {str(e)}")
        return None


def show_dataset_preview(df, dataset_name):
    if df is None or df.empty:
        print(f"   ⚠️  Cannot show preview - dataset is empty or failed to load")
        return

    print(f"\n📋 First 3 rows of {dataset_name}:")
    print("-" * 80)

    preview = df.head(3)
    print(preview.to_string(max_cols=10, max_colwidth=30, index=False))

    if len(df.columns) > 10:
        print(f"\n   ... and {len(df.columns) - 10} more columns not shown")

    print("-" * 80)


def main():
    base_dir = Path(".")  # Current directory

    # ========================================================================
    # DATASET 1: Issues Dataset (Directory)
    # ========================================================================
    print("=" * 60)
    print("DATASET 1: ISSUES")
    print("=" * 60)

    issues_path = base_dir / "data/processed/issues.parquet"
    issues_df = load_parquet_directory(issues_path, "Issues Dataset")

    if issues_df is not None:
        print(f"\n📊 Total rows in Issues dataset: {len(issues_df):,}")
        show_dataset_preview(issues_df, "Issues Dataset")
    else:
        print("❌ Failed to load Issues dataset")

    # ========================================================================
    # DATASET 2: Repositories Dataset (Directory - RQ1)
    # ========================================================================
    print("\n" + "=" * 60)
    print("DATASET 2: REPOSITORIES (RQ1)")
    print("=" * 60)

    repos_path = base_dir / "data/processed/rq1/repos.parquet"
    repos_df = load_parquet_directory(repos_path, "Repositories Dataset")

    if repos_df is not None:
        print(f"\n📊 Total rows in Repositories dataset: {len(repos_df):,}")
        show_dataset_preview(repos_df, "Repositories Dataset")
    else:
        print("❌ Failed to load Repositories dataset")

    # ========================================================================
    # DATASET 3: Commit Patterns Dataset (Directory - RQ2)
    # ========================================================================
    print("\n" + "=" * 60)
    print("DATASET 3: COMMIT PATTERNS (RQ2)")
    print("=" * 60)

    commit_patterns_path = base_dir / "data/processed/rq2/commit_patterns"
    commit_patterns_df = load_parquet_directory(
        commit_patterns_path, "Commit Patterns Dataset"
    )

    if commit_patterns_df is not None:
        print(
            f"\n📊 Total rows in Commit Patterns dataset: {len(commit_patterns_df):,}"
        )
        show_dataset_preview(commit_patterns_df, "Commit Patterns Dataset")
    else:
        print("❌ Failed to load Commit Patterns dataset")

    # ========================================================================
    # DATASET 4: Classified Commit Patterns Dataset (Directory - RQ2)
    # ========================================================================
    print("\n" + "=" * 60)
    print("DATASET 4: CLASSIFIED COMMIT PATTERNS (RQ2)")
    print("=" * 60)

    classified_patterns_path = (
        base_dir / "data/processed/rq2/commit_patterns_classified"
    )
    classified_patterns_df = load_parquet_directory(
        classified_patterns_path, "Classified Commit Patterns Dataset"
    )

    if classified_patterns_df is not None:
        print(
            f"\n📊 Total rows in Classified Commit Patterns dataset: {len(classified_patterns_df):,}"
        )
        show_dataset_preview(
            classified_patterns_df, "Classified Commit Patterns Dataset"
        )
    else:
        print("❌ Failed to load Classified Commit Patterns dataset")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("DATASET LOADING SUMMARY")
    print("=" * 80)

    datasets_info = [
        ("Issues Dataset", issues_df),
        ("Repositories Dataset", repos_df),
        ("Commit Patterns Dataset", commit_patterns_df),
        ("Classified Commit Patterns Dataset", classified_patterns_df),
    ]

    total_rows = 0
    loaded_count = 0

    for name, df in datasets_info:
        if df is not None:
            rows = len(df)
            cols = len(df.columns)
            total_rows += rows
            loaded_count += 1
            print(f"✅ {name:<35}: {rows:>8,} rows × {cols:>3} columns")
        else:
            print(f"❌ {name:<35}: {'FAILED':>8}")

    print("-" * 80)
    print(f"📊 Successfully loaded: {loaded_count}/4 datasets")
    print(f"📊 Total rows across all datasets: {total_rows:,}")


def example_individual_loading():
    """
    Example showing how to load individual datasets for your own analysis.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE: HOW TO LOAD INDIVIDUAL DATASETS")
    print("=" * 80)

    print(
        """
# Example 1: Load Issues Dataset (Directory)
issues_path = Path("data/processed/issues.parquet")
issues_df = load_parquet_directory(issues_path, "Issues")

# Example 2: Load Repositories Dataset (Directory)
repos_path = Path("data/processed/rq1/repos.parquet") 
repos_df = load_parquet_directory(repos_path, "Repositories")

# Example 3: Load Commit Patterns Dataset (Directory)
patterns_path = Path("data/processed/rq2/commit_patterns")
patterns_df = load_parquet_directory(patterns_path, "Commit Patterns")

# Example 4: Load Classified Commit Patterns Dataset (Directory)
classified_path = Path("data/processed/rq2/commit_patterns_classified")
classified_df = load_parquet_directory(classified_path, "Classified Patterns")

# Each function returns None if loading fails, so always check:
if issues_df is not None:
    print(f"Issues dataset has {len(issues_df)} rows")
    # Your analysis code here...
"""
    )


if __name__ == "__main__":
    """
    Main execution block - demonstrates the complete dataset loading tutorial.
    """
    try:
        main()

        # Show example code
        example_individual_loading()

    except KeyboardInterrupt:
        print("\n\nTutorial interrupted by user.")
    except Exception as e:
        print(f"\n\nAn error occurred during the tutorial: {e}")
        print("Please check that the dataset files exist and are accessible.")
