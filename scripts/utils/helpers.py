import json
import logging
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd


logger = logging.getLogger(__name__)


def convert_categorical_to_boolean(df, column):
    try:
        if df[column].dtype.name == "category":
            unique_vals = {str(val).lower() for val in df[column].unique()}

            if unique_vals <= {"true", "false"}:
                df[column] = df[column].str.lower().map({"true": True, "false": False})
            elif unique_vals <= {True, False}:
                df[column] = df[column].astype(bool)
            elif unique_vals <= {"1", "0"} or unique_vals <= {1, 0}:
                df[column] = df[column].astype(int).astype(bool)
            else:
                raise ValueError(f"Unexpected values in {column}: {unique_vals}")

        return df

    except Exception as e:
        raise ValueError(f"Error converting {column} to boolean: {str(e)}")


def process_year_chunk(chunk_data):
    try:
        chunk_data = convert_categorical_to_boolean(chunk_data, "is_fork")
        chunk_data["author_date"] = pd.to_datetime(chunk_data["author_date"])
        return chunk_data
    except Exception as e:
        logger.error(f"Error processing chunk: {str(e)}")
        raise


def validate_commit_data(df):
    logger.info("Validating commit data...")
    fork_dist = df["is_fork"].value_counts()
    logger.info(f"Distribution of commits by is_fork value:\n{fork_dist}")

    date_range = df["author_date"].agg(["min", "max"])
    logger.info(f"Date range: {date_range['min']} to {date_range['max']}")

    if fork_dist.get(True, 0) > fork_dist.get(False, 0) * 10:
        logger.warning(
            "Unusual distribution: Fork commits significantly higher than original commits"
        )


def create_fork_comparison_plot(
    time_series, title="Original vs Fork Repository Commits", figsize=(12, 6)
):
    colors = sns.color_palette("Paired")
    fig, ax1 = plt.subplots(figsize=figsize)

    ax1.plot(
        time_series.index, time_series["original"], label="Original", color=colors[1]
    )
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Original Commits")
    ax1.tick_params(axis="y")

    ax2 = ax1.twinx()
    ax2.plot(time_series.index, time_series["fork"], label="Fork", color=colors[3])
    ax2.set_ylabel("Fork Commits")
    ax2.tick_params(axis="y")

    plt.title(title)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.tight_layout()
    return fig


def save_results(metrics, figures, output_dir):
    metrics_file = output_dir / "temporal_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    for name, fig in figures.items():
        fig.savefig(figures_dir / f"{name}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
