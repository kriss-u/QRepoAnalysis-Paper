import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..config.constants import PROCESSED_DATA_DIR, RESULTS_DIR
from ..utils.logger import setup_logger
from ..utils.plot_utils_ieee import (
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
    FONT_SIZES,
    CATEGORICAL_COLORS,
)
from scipy.signal import savgol_filter

logger = setup_logger(__name__, "rq1", "lang_framework_trends")

FRAMEWORK_ALIASES = {
    "qiskit-aer": "qiskit",
    "qiskit-terra": "qiskit",
    "cirq-core": "cirq",
}

LANGUAGE_ALIASES = {}

FRAMEWORK_DISPLAY_NAMES = {
    "qiskit": "Qiskit",
    "cirq": "Cirq",
    "projectq": "ProjectQ",
    "pennylane": "PennyLane",
    "qsharp": "Q#",
    "pyquil": "PyQuil",
    "qutip": "QuTiP",
    "quest": "QuEST",
    "qulacs": "Qulacs",
    "q#": "QDK",
    "openfermion": "OpenFermion",
    "amazon-braket-sdk": "Amazon Braket",
    "openqasm3": "OpenQASM 3",
}

LANGUAGE_DISPLAY_NAMES = {
    "python": "Python",
    "c++": "C++",
    "java": "Java",
    "c#": "C#",
    "f#": "F#",
    "rust": "Rust",
    "openqasm": "OpenQASM",
    "javascript": "JavaScript",
}


def get_normalized_framework_stats(df):
    packages_df = df.explode("packages")
    packages_df = packages_df.dropna(subset=["packages"])

    packages_df["normalized_framework"] = packages_df["packages"].map(
        lambda x: FRAMEWORK_ALIASES.get(x, x)
    )

    top_frameworks = (
        packages_df["normalized_framework"].value_counts().nlargest(10).index
    )

    stats_list = []
    for framework in top_frameworks:
        framework_repos = df[
            df["packages"].apply(
                lambda pkgs: any(FRAMEWORK_ALIASES.get(p, p) == framework for p in pkgs)
            )
        ]

        all_contributors = set()
        for contributors_list in framework_repos["contributors"].values:
            if (
                isinstance(contributors_list, (list, np.ndarray))
                and len(contributors_list) > 0
            ):
                all_contributors.update(contributors_list)

        stats = {
            "framework": get_display_name(framework, "Frameworks"),
            "total_repositories": len(framework_repos),
            "total_contributors": len(all_contributors),
        }
        stats_list.append(stats)

    return pd.DataFrame(stats_list)


def get_normalized_language_stats(df):
    df_norm = df.copy()
    df_norm["normalized_language"] = df_norm["language"].map(
        lambda x: LANGUAGE_ALIASES.get(x.lower(), x) if pd.notna(x) else x
    )

    top_languages = df_norm["normalized_language"].value_counts().nlargest(10).index

    stats_list = []
    for language in top_languages:
        language_repos = df_norm[df_norm["normalized_language"] == language]

        all_contributors = set()
        for contributors_list in language_repos["contributors"].values:
            if (
                isinstance(contributors_list, (list, np.ndarray))
                and len(contributors_list) > 0
            ):
                all_contributors.update(contributors_list)

        stats = {
            "language": get_display_name(language, "Programming Languages"),
            "total_repositories": len(language_repos),
            "total_contributors": len(all_contributors),
        }
        stats_list.append(stats)

    return pd.DataFrame(stats_list)


def save_repository_statistics(df, output_dir):
    try:
        framework_stats = get_normalized_framework_stats(df)
        language_stats = get_normalized_language_stats(df)

        framework_stats.to_csv(output_dir / "framework_statistics.csv", index=False)
        language_stats.to_csv(output_dir / "language_statistics.csv", index=False)

        stats_data = {
            "frameworks": {
                row["framework"]: {
                    "total_repositories": int(row["total_repositories"]),
                    "total_contributors": int(row["total_contributors"]),
                }
                for _, row in framework_stats.iterrows()
            },
            "languages": {
                row["language"]: {
                    "total_repositories": int(row["total_repositories"]),
                    "total_contributors": int(row["total_contributors"]),
                }
                for _, row in language_stats.iterrows()
            },
        }

        with open(output_dir / "repository_statistics.json", "w") as f:
            json.dump(stats_data, f, indent=2)

        logger.info("Statistics saved:")
        logger.info(f"- JSON: {output_dir}/repository_statistics.json")
        logger.info(f"- Framework CSV: {output_dir}/framework_statistics.csv")
        logger.info(f"- Language CSV: {output_dir}/language_statistics.csv")

        logger.info("\nTop frameworks after normalization:")
        for framework, stats in stats_data["frameworks"].items():
            logger.info(
                f"{framework}: {stats['total_repositories']} repositories, "
                f"{stats['total_contributors']} contributors"
            )

        logger.info("\nTop languages after normalization:")
        for language, stats in stats_data["languages"].items():
            logger.info(
                f"{language}: {stats['total_repositories']} repositories, "
                f"{stats['total_contributors']} contributors"
            )

    except Exception as e:
        logger.error(f"Error saving repository statistics: {str(e)}")
        raise


def get_display_name(name, plot_type):
    if plot_type == "Programming Languages":
        return LANGUAGE_DISPLAY_NAMES.get(name.lower(), name)
    else:
        return FRAMEWORK_DISPLAY_NAMES.get(name.lower(), name)


def normalize_framework_names(frameworks):
    if pd.isna(frameworks).all():
        return []

    if isinstance(frameworks, pd.Series):
        frameworks = frameworks.tolist()

    normalized = []
    seen = set()

    for framework in frameworks:
        if pd.isna(framework):
            continue
        norm_name = FRAMEWORK_ALIASES.get(framework, framework)
        if norm_name not in seen:
            normalized.append(norm_name)
            seen.add(norm_name)

    return normalized


def normalize_language_name(language):
    if pd.isna(language):
        return language
    return LANGUAGE_ALIASES.get(language.lower(), language.title())


def load_repository_data():
    try:
        dataset_path = PROCESSED_DATA_DIR / "rq1" / "repos.parquet"
        df = pd.read_parquet(dataset_path)

        logger.info("Sample packages data:")
        logger.info(df["packages"].head())
        logger.info(f"Packages dtype: {df['packages'].dtype}")

        df["created_at"] = pd.to_datetime(df["created_at"])

        df = df[df["created_at"] < pd.Timestamp("2024-10-01")]

        df = df.sort_values("created_at")

        df = df.dropna(subset=["language"])

        df["packages"] = df["packages"].apply(normalize_framework_names)
        df["language"] = df["language"].apply(normalize_language_name)

        logger.info(f"Loaded {len(df)} repositories")
        return df

    except Exception as e:
        logger.error(f"Error loading repository data: {str(e)}")
        raise


def format_time_label(date, granularity):
    if granularity == "week":
        return date.strftime("%Y-W%W")
    elif granularity == "month":
        return date.strftime("%Y-%m")
    elif granularity == "quarter":
        quarter = (date.month - 1) // 3 + 1
        return f"{date.year} Q{quarter}"
    else:
        return str(date.year)


def resample_language_data(df, granularity):
    freq_map = {"week": "W", "month": "ME", "quarter": "QE", "year": "YE"}
    freq = freq_map[granularity]

    grouped = df.groupby([pd.Grouper(key="created_at", freq=freq), "language"]).size()
    period_language = grouped.unstack(fill_value=0)

    top_languages = df["language"].value_counts().nlargest(10).index
    period_language = period_language[list(top_languages)]

    period_language = period_language.cumsum()

    return period_language.reset_index()


def resample_package_data(df, granularity):
    freq_map = {"week": "W", "month": "ME", "quarter": "QE", "year": "YE"}
    freq = freq_map[granularity]

    packages_df = df.explode("packages")
    packages_df = packages_df.dropna(subset=["packages"])

    grouped = packages_df.groupby(
        [pd.Grouper(key="created_at", freq=freq), "packages"]
    ).size()
    period_packages = grouped.unstack(fill_value=0)

    top_packages = packages_df["packages"].value_counts().nlargest(10).index
    period_packages = period_packages[list(top_packages)]

    period_packages = period_packages.cumsum()

    return period_packages.reset_index()


def create_trend_plot(
    df,
    granularity,
    plots_dir,
    plot_type,
    use_log=False,
):
    setup_plotting_style()

    if use_log:
        fig, ax = plt.subplots(figsize=(16, 9))
    else:
        fig, ax = plt.subplots(
            # figsize=(FIG_SIZE_SINGLE_COL[0], FIG_SIZE_SINGLE_COL[1] + 1.0)
            figsize=FIG_SIZE_SINGLE_COL
        )

    df_counts = df.set_index("created_at")
    dates = df_counts.index.to_pydatetime()

    colors = MAIN_COLORS
    line_styles = ["-"]

    for idx, column in enumerate(df_counts.columns):
        values = df_counts[column].values

        # Apply Savgol filter for smoothing
        if len(values) > 3:  # Only apply if we have enough points
            values = savgol_filter(values, window_length=3, polyorder=1)

        style_idx = idx % len(line_styles)
        color_idx = idx % len(colors)

        display_name = get_display_name(column, plot_type)

        ax.plot(
            dates,
            values,
            label=display_name,
            color=colors[color_idx],
            linestyle=line_styles[style_idx],
            linewidth=PLOT_LINE_WIDTH,
            alpha=0.8,
        )

    ncol = 3 if plot_type == "Programming Languages" else 2

    if use_log:
        ax.set_yscale("log")
        ax.set_ylim(bottom=1)
    else:
        ax.set_ylim(bottom=0)

    def format_thousands(x, p):
        if x >= 1000:
            return f"{x/1000:.0f}K"
        return f"{x:.0f}"

    legend = ax.legend(
        title=plot_type,
        loc="upper left",
        ncol=ncol,
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

    ax.set_xlabel("")
    ax.set_ylabel(
        ("Total Number of Repositories" if use_log else "Total Number of Repositories"),
        fontsize=FONT_SIZES["axis_label"],
    )

    apply_grid_style(ax)
    setup_axis_ticks(ax, dates, granularity, n_ticks=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_thousands))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.28)

    suffix = "_log" if use_log else ""
    save_plot(fig, plots_dir, f"{plot_type.lower()}_trends_{granularity}{suffix}")
    plt.close(fig)


def create_language_trend_plot(df, granularity, plots_dir):
    if df.empty:
        logger.warning("Empty DataFrame provided for language trends")
        return

    try:
        create_trend_plot(
            df=df,
            granularity=granularity,
            plots_dir=plots_dir,
            plot_type="Programming Languages",
            use_log=False,
        )

        create_trend_plot(
            df=df,
            granularity=granularity,
            plots_dir=plots_dir,
            plot_type="Programming Languages",
            use_log=True,
        )

        logger.info(
            f"Successfully created language trend plots with {granularity} granularity"
        )
    except Exception as e:
        logger.error(f"Error creating language trend plots: {str(e)}")
        raise


def create_package_trend_plot(df, granularity, plots_dir):
    if df.empty:
        logger.warning("Empty DataFrame provided for framework trends")
        return

    try:
        create_trend_plot(
            df=df,
            granularity=granularity,
            plots_dir=plots_dir,
            plot_type="Frameworks",
            use_log=False,
        )

        create_trend_plot(
            df=df,
            granularity=granularity,
            plots_dir=plots_dir,
            plot_type="Frameworks",
            use_log=True,
        )

        logger.info(
            f"Successfully created framework trend plots with {granularity} granularity"
        )
    except Exception as e:
        logger.error(f"Error creating framework trend plots: {str(e)}")
        raise


def analyze_language_trends(args):
    logger.info(
        f"Analyzing language and package trends using {args.granularity} granularity"
    )

    try:
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        setup_plotting_style()

        df = load_repository_data()

        language_trends = resample_language_data(df, args.granularity)
        package_trends = resample_package_data(df, args.granularity)

        create_language_trend_plot(language_trends, args.granularity, plots_dir)
        create_package_trend_plot(package_trends, args.granularity, plots_dir)

        language_trends.to_csv(output_dir / "language_trends.csv", index=False)
        package_trends.to_csv(output_dir / "package_trends.csv", index=False)

        save_repository_statistics(df, output_dir)

        logger.info("Language and package trend analysis completed successfully")

    except Exception as e:
        logger.error(f"Error analyzing language and package trends: {str(e)}")
        raise


if __name__ == "__main__":
    from argparse import Namespace

    analyze_language_trends(
        Namespace(
            granularity="quarter", output_dir=RESULTS_DIR / "rq1/lang_framework_trends"
        )
    )
