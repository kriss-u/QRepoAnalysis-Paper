import dask.dataframe as dd
import pandas as pd
import pyarrow.parquet as pq
import numpy as np
from pathlib import Path
from scripts.config.constants import PROCESSED_DATA_DIR


def get_top_packages(n_top: int = 35) -> pd.DataFrame:
    """
    Get the top n most frequently used packages across repositories using numpy array methods.
    Args:
        n_top: Number of top packages to return (default: 35)
    Returns:
        DataFrame with package names and their usage counts, sorted by popularity
    """
    # Read the parquet file - ensure we properly handle list columns
    df = dd.read_parquet(PROCESSED_DATA_DIR / "common" / "repos.parquet")

    # Process and explode packages - no need for custom processing since packages is already a list
    print("Processing packages...")
    exploded = df.packages.explode()

    # Remove empty strings, None values, and '0' values efficiently
    mask = (exploded.notnull()) & (exploded != "") & (exploded != "0")
    exploded = exploded[mask]

    # Count package frequencies for ALL packages
    print("Counting package frequencies...")
    package_counts = exploded.value_counts().compute()

    # Filter out zero counts if any remain
    package_counts = package_counts[package_counts > 0]

    # Calculate total packages for statistics
    total_packages = len(package_counts)
    print(f"Total unique packages found: {total_packages}")

    # Convert full data to DataFrame first
    all_packages = (
        package_counts.reset_index()
        .rename(columns={"packages": "package", 0: "count"})
        .sort_values("count", ascending=False)
    )

    # Calculate statistics using the full dataset
    total_repos = len(df)
    all_packages["percentage"] = np.round(all_packages["count"] / total_repos * 100, 2)
    all_packages["rank"] = np.arange(1, len(all_packages) + 1)
    all_packages["cumulative_percentage"] = np.cumsum(all_packages["percentage"])
    all_packages["percentage_display"] = all_packages["percentage"].map(
        "{:.2f}%".format
    )

    # Now take the top N packages after all calculations are done
    top_packages = all_packages.head(n_top)

    # Display distribution analysis using full dataset
    all_counts = all_packages["count"].values
    percentiles = np.percentile(all_counts, [25, 50, 75])
    print("\nUsage distribution (across all packages):")
    print(f"- 25th percentile: {percentiles[0]:.0f} repositories")
    print(f"- Median: {percentiles[1]:.0f} repositories")
    print(f"- 75th percentile: {percentiles[2]:.0f} repositories")
    print(f"\nTotal number of packages found: {len(all_packages)}")
    print(f"Showing top {n_top} packages")

    return top_packages


def save_repos_with_top_packages(top_packages_df: pd.DataFrame) -> None:
    """
    Save repositories that use any of the top packages to a text file.

    Args:
        top_packages_df: DataFrame containing top packages analysis
    """
    print("\nFiltering repositories using top packages...")

    # Get set of top package names
    top_packages_set = set(top_packages_df["package"].str.lower())

    # Read full repository data using PyArrow
    table = pq.read_table(PROCESSED_DATA_DIR / "common" / "repos.parquet")
    repos_df = table.to_pandas()

    # Function to check if any top package is in repository packages
    def has_top_package(packages_array):
        if isinstance(packages_array, (list, np.ndarray)):
            return any(
                str(pkg).lower() in top_packages_set
                for pkg in packages_array
                if pd.notna(pkg)
            )
        return False

    # Filter repositories using vectorized operations
    mask = repos_df["packages"].apply(lambda x: has_top_package(x))
    filtered_repos = repos_df[mask]

    # Save to text file
    output_file = PROCESSED_DATA_DIR / "common" / "repos_with_top_packages.txt"
    filtered_repos["repo_id"].to_csv(output_file, index=False, header=False)

    print(f"Found {len(filtered_repos)} repositories using top packages")
    print(f"Results saved to: {output_file}")


def analyze_package_trends() -> pd.DataFrame:
    """
    Analyze package usage trends and produce detailed statistics.
    Returns:
        DataFrame with package statistics, sorted by popularity
    """
    top_packages = get_top_packages()

    # Add additional metrics (maintaining sort order)
    stats = top_packages.copy()
    stats["market_share"] = stats["percentage"].map("{:.1f}%".format)
    stats["cumulative_share"] = stats["cumulative_percentage"].map("{:.1f}%".format)

    return stats.sort_values("count", ascending=False)


def test_package_analysis():
    """
    Test the package analysis functions and display results
    """
    try:
        print("Starting package analysis...")

        # Get top packages
        results_df = get_top_packages()

        # Debug info
        print("\nDebug - Data structure:")
        print(f"Type: {type(results_df)}")
        print(f"Columns: {results_df.columns.tolist()}")
        print(f"Dtypes:\n{results_df.dtypes}")

        # Display basic statistics
        print(f"\nAnalyzed {len(results_df)} top packages")
        print("\nTop 50 packages (sorted by usage):")

        # Create a clean display format
        display_cols = [
            "package",
            "count",
            "percentage_display",
            "rank",
            "cumulative_percentage",
        ]
        print(results_df[display_cols].to_string(index=False))

        # Basic validation
        if len(results_df) > 0:
            print("\nValidation checks:")
            print(f"- Number of packages found: {len(results_df)}")
            print(f"- Maximum usage count: {results_df['count'].max()}")
            print(f"- Minimum usage count: {results_df['count'].min()}")
            print(f"- Total percentage covered: {results_df['percentage'].sum():.2f}%")

        # Save results to CSV (maintaining sort order)
        output_file = PROCESSED_DATA_DIR / "common" / "top_packages_analysis.csv"
        results_df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")

        # Save repositories using top packages
        save_repos_with_top_packages(results_df)

    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise


if __name__ == "__main__":
    test_package_analysis()
