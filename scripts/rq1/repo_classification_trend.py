import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from datetime import datetime

from statsmodels.stats.proportion import proportions_ztest

from ..config.constants import (
    PROCESSED_DATA_DIR,
    REPOS_MAINTENANCE_INFO_FILE,
    RESULTS_DIR,
)
from ..utils.logger import setup_logger
from ..utils.plot_utils_ieee import (
    FONT_SIZES,
    FONT_WEIGHT_BOLD,
    GREY_COLORS_DARK,
    setup_plotting_style,
    MAIN_COLORS,
    PAIRED_COLORS,
    FIG_SIZE_SINGLE_COL,
    PLOT_LINE_WIDTH,
    MARKER_SIZE,
    setup_axis_ticks,
    setup_legend,
    save_plot,
    create_pie_chart,
    apply_grid_style,
    CATEGORICAL_COLORS,
)

logger = setup_logger(__name__, "rq1", "category_trends")


def log_uncategorized_originals(df):
    uncategorized = df[(df["category"].isna()) & (~df["is_fork"])].sort_values(
        "created_at"
    )

    logger.info("\nOriginal Repositories Without Categories:")
    logger.info("=======================================")
    logger.info(f"Total Count: {len(uncategorized)}")
    logger.info("\nRepository List:")
    for _, repo in uncategorized.iterrows():
        logger.info(
            f"- {repo['repo_id']} (Created: {repo['created_at'].strftime('%Y-%m-%d')})"
        )


def load_repository_data():
    try:
        dataset_path = PROCESSED_DATA_DIR / "rq1" / "repos.parquet"
        df = pd.read_parquet(dataset_path)

        df["created_at"] = pd.to_datetime(df["created_at"])

        log_uncategorized_originals(df)

        df = df[df["category"].notna()]
        df = df[df["created_at"] < pd.Timestamp("2024-10-01")]

        df = df.sort_values("created_at")

        logger.info(f"\nLoaded {len(df)} categorized repositories")
        return df

    except Exception as e:
        logger.error(f"Error loading repository data: {str(e)}")
        raise


def resample_category_data(df, granularity):
    freq_map = {"week": "W", "month": "ME", "quarter": "QE", "year": "YE"}
    freq = freq_map[granularity]

    period_dfs = []
    cumulative_dfs = []

    for category in df["category"].unique():
        category_df = df[df["category"] == category]

        period_count = category_df.resample(freq, on="created_at").size().reset_index()
        period_count.columns = ["period", "new_repos"]
        period_count["category"] = category
        period_dfs.append(period_count)

        cumulative_count = period_count.copy()
        cumulative_count["total_repos"] = cumulative_count["new_repos"].cumsum()
        cumulative_dfs.append(cumulative_count)

    period_df = pd.concat(period_dfs, ignore_index=True)
    cumulative_df = pd.concat(cumulative_dfs, ignore_index=True)

    return period_df, cumulative_df


def calculate_category_metrics(df):
    metrics = {
        "total_repositories": len(df),
        "total_categories": df["category"].nunique(),
        "category_distribution": df["category"].value_counts().to_dict(),
        "first_repository_date": df["created_at"].min(),
        "latest_repository_date": df["created_at"].max(),
        "years_active": (df["created_at"].max() - df["created_at"].min()).days / 365.25,
    }

    total = len(df)
    metrics["category_percentages"] = {
        category: (count / total * 100)
        for category, count in metrics["category_distribution"].items()
    }

    return metrics


def create_category_timeline(cumulative_df, granularity, plots_dir):
    fig, ax = plt.subplots(
        figsize=(FIG_SIZE_SINGLE_COL[0], FIG_SIZE_SINGLE_COL[1] * 1.28)
        # figsize=FIG_SIZE_SINGLE_COL,
    )

    final_counts = (
        cumulative_df.groupby("category")["total_repos"]
        .max()
        .sort_values(ascending=False)
    )
    categories = final_counts.index.tolist()

    colors = MAIN_COLORS

    for i, category in enumerate(categories):
        category_data = cumulative_df[cumulative_df["category"] == category]
        color_idx = i % len(colors)

        ax.plot(
            category_data["period"],
            category_data["total_repos"],
            label=category,
            color=colors[color_idx],
            linewidth=PLOT_LINE_WIDTH,
            alpha=0.8,
        )

    ax.set_xlabel("")
    ax.set_ylabel("Number of Repositories", fontsize=FONT_SIZES["axis_label"])

    dates = cumulative_df["period"].unique()
    setup_axis_ticks(ax, dates, granularity, n_ticks=8)

    ax.set_ylim(bottom=0)
    current_ticks = list(ax.get_yticks())
    if 1000 not in current_ticks:
        current_ticks.append(1000)
        current_ticks.sort()
        ax.set_yticks(current_ticks)

    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: f"{x/1000:.0f}K" if x >= 1000 else f"{int(x)}")
    )

    apply_grid_style(ax)

    legend_labels = [category.replace(" ", "\n") for category in categories]

    legend = ax.legend(
        legend_labels,
        title="Categories",
        loc="upper left",
        ncol=2,
        frameon=False,
        fontsize=FONT_SIZES["legend"],
        title_fontsize=FONT_SIZES["legend"],
        handlelength=0.5,
        handletextpad=0.25,
        columnspacing=0.25,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(0.9)
    legend.get_frame().set_edgecolor("#CCCCCC")

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.28)

    save_plot(fig, plots_dir, f"category_timeline_{granularity}")


def create_category_distribution_chart(df, plots_dir):
    category_counts = df["category"].value_counts()

    fig, ax = create_pie_chart(
        title="",
        data=category_counts.values,
        labels=category_counts.index,
        explode=[0.02] * len(category_counts),
    )

    save_plot(fig, plots_dir, "category_distribution_pie")


def save_results(period_df, cumulative_df, metrics, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    period_df.to_csv(output_dir / "category_period_activity.csv", index=False)
    cumulative_df.to_csv(output_dir / "category_cumulative_activity.csv", index=False)

    metrics_output = {
        k: str(v) if isinstance(v, datetime) else v for k, v in metrics.items()
    }
    pd.DataFrame([metrics_output]).to_json(
        output_dir / "category_metrics.json", orient="records", indent=2
    )


def create_visualizations(
    period_df, cumulative_df, df, granularity, metrics, output_dir
):
    setup_plotting_style()

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    create_category_timeline(cumulative_df, granularity, plots_dir)
    create_category_distribution_chart(df, plots_dir)

    with open(plots_dir / "category_analysis_summary.txt", "w") as f:
        f.write("Repository Category Analysis Summary\n")
        f.write("==================================\n\n")
        f.write(f"Total Repositories: {metrics['total_repositories']:,}\n")
        f.write(f"Number of Categories: {metrics['total_categories']}\n\n")
        f.write("Category Distribution:\n")
        for category, count in metrics["category_distribution"].items():
            percentage = metrics["category_percentages"][category]
            f.write(f"{category}: {count:,} ({percentage:.1f}%)\n")


def analyze_maintenance_patterns(df, output_dir):
    try:
        maintenance_data = pd.read_csv(REPOS_MAINTENANCE_INFO_FILE)
        df = pd.merge(
            df,
            maintenance_data[["repo_name", "maintenance_category"]],
            left_on="repo_id",
            right_on="repo_name",
            how="inner",
        )

        results = []
        for category in sorted(df["category"].unique()):
            high_repos = df[df["maintenance_category"] == "High maintenance"]
            low_repos = df[df["maintenance_category"] == "Low maintenance"]

            high_prop = (high_repos["category"] == category).mean()
            low_prop = (low_repos["category"] == category).mean()

            count = [
                (high_repos["category"] == category).sum(),
                (low_repos["category"] == category).sum(),
            ]
            nobs = [len(high_repos), len(low_repos)]
            _, p_value = proportions_ztest(count=count, nobs=nobs)

            results.append(
                {
                    "category": category,
                    "high_maintenance_prop": high_prop,
                    "low_maintenance_prop": low_prop,
                    "p_value": p_value,
                }
            )

        results_df = pd.DataFrame(results).sort_values(
            by="high_maintenance_prop", ascending=False
        )

        latex_lines = [
            "\\begin{tabular}{|l|c|c|c|}",
            "\\hline",
            "Category & Hi. Prop. & Lo. Prop. & p-value \\\\",
            "\\hline \\hline",
        ]

        for _, row in results_df.iterrows():
            p_val = (
                "\\textless{}0.001"
                if row["p_value"] < 0.001
                else f"{row['p_value']:.4f}"
            )
            latex_line = (
                f"{row['category']} & "
                f"{row['high_maintenance_prop']:.4f} & "
                f"{row['low_maintenance_prop']:.4f} & "
                f"{p_val} \\\\"
            )
            latex_lines.append(latex_line)

        latex_lines.extend(["\\hline", "\\end{tabular}"])

        maintenance_dir = output_dir / "maintenance"
        maintenance_dir.mkdir(exist_ok=True)

        with open(maintenance_dir / "maintenance_categorical.tex", "w") as f:
            f.write("\n".join(latex_lines))

    except Exception as e:
        logger.error(f"Error analyzing categorical patterns: {str(e)}")
        raise


def analyze_continuous_variables(df, output_dir):
    try:
        maintenance_data = pd.read_csv(REPOS_MAINTENANCE_INFO_FILE)
        df = pd.merge(
            df,
            maintenance_data[["repo_name", "maintenance_category"]],
            left_on="repo_id",
            right_on="repo_name",
            how="inner",
        )

        df["update_duration"] = (
            df["updated_at"] - df["created_at"]
        ).dt.total_seconds() / (24 * 3600)
        df["update_duration"] = df["update_duration"].clip(lower=0)

        variables = {
            "update_duration": "Updated Duration",
            "stargazers_count": "Stars",
            "forks_count": "Forks",
            "size": "Size",
            "contributor_count": "Contributors",
            "code_lines": "Code Lines",
        }

        results = []
        for var, label in variables.items():
            high_maint = df[df["maintenance_category"] == "High maintenance"][var]
            low_maint = df[df["maintenance_category"] == "Low maintenance"][var]

            threshold = pd.concat([high_maint, low_maint]).median()
            high_prop = (high_maint > threshold).mean()
            low_prop = (low_maint > threshold).mean()

            _, p_value = stats.mannwhitneyu(
                high_maint, low_maint, alternative="two-sided"
            )

            results.append(
                {
                    "variable": label,
                    "high_prop": high_prop,
                    "low_prop": low_prop,
                    "p_value": p_value,
                }
            )

        results_df = pd.DataFrame(results)
        results_df["diff"] = results_df["high_prop"] - results_df["low_prop"]
        results_df = results_df.sort_values("diff", ascending=False)

        latex_lines = [
            "\\begin{tabular}{|l|c|c|c|}",
            "\\hline",
            "Variable & Hi. Prop. & Lo. Prop. & p-value \\\\",
            "\\hline \\hline",
        ]

        for _, row in results_df.iterrows():
            p_val = (
                "\\textless{}0.001"
                if row["p_value"] < 0.001
                else f"{row['p_value']:.4f}"
            )
            latex_line = (
                f"{row['variable']} & "
                f"{row['high_prop']:.4f} & "
                f"{row['low_prop']:.4f} & "
                f"{p_val} \\\\"
            )
            latex_lines.append(latex_line)

        latex_lines.extend(["\\hline", "\\end{tabular}"])

        maintenance_dir = output_dir / "maintenance"
        maintenance_dir.mkdir(exist_ok=True)

        with open(maintenance_dir / "maintenance_continuous.tex", "w") as f:
            f.write("\n".join(latex_lines))

        logger.info("Maintenance analysis completed with LaTeX tables")

    except Exception as e:
        logger.error(f"Error analyzing continuous variables: {str(e)}")
        raise


def analyze_repo_classification_trends(args):
    logger.info(f"Analyzing category trends with {args.granularity} granularity")

    try:
        output_dir = RESULTS_DIR / "rq1" / "category_trends"
        output_dir.mkdir(parents=True, exist_ok=True)

        df = load_repository_data()
        metrics = calculate_category_metrics(df)

        logger.info(
            f"Analyzing {metrics['total_repositories']} repositories "
            f"across {metrics['total_categories']} categories "
            f"over {metrics['years_active']:.1f} years"
        )

        period_df, cumulative_df = resample_category_data(df, args.granularity)

        create_visualizations(
            period_df, cumulative_df, df, args.granularity, metrics, output_dir
        )

        save_results(period_df, cumulative_df, metrics, output_dir)

        analyze_maintenance_patterns(df, output_dir)

        analyze_continuous_variables(df, output_dir)

        logger.info("Category trend analysis completed successfully")

    except Exception as e:
        logger.error(f"Error analyzing category trends: {str(e)}")
        raise


if __name__ == "__main__":
    from argparse import Namespace

    analyze_repo_classification_trends(Namespace(granularity="quarter"))
