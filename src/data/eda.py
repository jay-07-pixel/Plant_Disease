"""Exploratory data analysis and visualization for dataset audits.

Generates read-only reports and charts from :class:`AuditSummary` results.
No dataset files are modified.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.data.dataset_audit import AuditSummary, DatasetAuditResult
from src.data.dataset_comparison import DatasetComparisonReport

logger = logging.getLogger(__name__)

# Chart styling
FIGURE_DPI = 150
TOP_N_CLASSES = 20
CHART_COLORS = "#2E86AB"


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------


def _ensure_parent(path: Path) -> None:
    """Create parent directories for an output file if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_class_distribution(
    images_per_class: dict[str, int],
    output_path: Path,
    *,
    title: str = "Class Distribution",
) -> None:
    """Generate a bar chart of image counts per class.

    Args:
        images_per_class: Mapping of class label to image count.
        output_path: Destination PNG path.
        title: Chart title.
    """
    _ensure_parent(output_path)

    if not images_per_class:
        logger.warning("No class data available for distribution chart.")
        _save_empty_chart(output_path, title, "No class data available.")
        return

    series = pd.Series(images_per_class).sort_values(ascending=False)
    fig_width = max(10, min(24, len(series) * 0.35))
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    series.plot(kind="bar", ax=ax, color=CHART_COLORS, edgecolor="white", linewidth=0.5)
    ax.set_title(title)
    ax.set_xlabel("Class")
    ax.set_ylabel("Image Count")
    ax.tick_params(axis="x", rotation=90, labelsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info("Saved class distribution chart to %s", output_path)


def plot_class_distribution_report(
    images_per_class: dict[str, int],
    output_path: Path,
    *,
    title: str = "Class Distribution Analysis",
    top_n: int = TOP_N_CLASSES,
) -> None:
    """Generate a combined class distribution report figure.

    The figure contains three panels: full class distribution bar chart,
    top-N largest classes, and top-N smallest classes.

    Args:
        images_per_class: Mapping of class label to image count.
        output_path: Destination PNG path.
        title: Overall figure title.
        top_n: Number of classes to show in the top/bottom panels.
    """
    _ensure_parent(output_path)

    if not images_per_class:
        logger.warning("No class data available for class distribution report.")
        _save_empty_chart(output_path, title, "No class data available.")
        return

    series = pd.Series(images_per_class).sort_values(ascending=False)
    largest = series.head(top_n)
    smallest = series.sort_values(ascending=True).head(top_n)

    fig, axes = plt.subplots(3, 1, figsize=(16, 14))

    series.plot(
        kind="bar",
        ax=axes[0],
        color=CHART_COLORS,
        edgecolor="white",
        linewidth=0.5,
    )
    axes[0].set_title("Class Distribution (All Classes)")
    axes[0].set_xlabel("Class")
    axes[0].set_ylabel("Image Count")
    axes[0].tick_params(axis="x", rotation=90, labelsize=7)
    axes[0].grid(axis="y", alpha=0.3)

    largest.plot(kind="barh", ax=axes[1], color="#2E86AB", edgecolor="white")
    axes[1].set_title(f"Top {len(largest)} Largest Classes")
    axes[1].set_xlabel("Image Count")
    axes[1].invert_yaxis()
    axes[1].grid(axis="x", alpha=0.3)

    smallest.plot(kind="barh", ax=axes[2], color="#A23B72", edgecolor="white")
    axes[2].set_title(f"Top {len(smallest)} Smallest Classes")
    axes[2].set_xlabel("Image Count")
    axes[2].invert_yaxis()
    axes[2].grid(axis="x", alpha=0.3)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info("Saved class distribution report to %s", output_path)


def plot_resolution_histogram(
    summary: AuditSummary,
    output_path: Path,
    *,
    title: str = "Image Resolution Distribution (Megapixels)",
) -> None:
    """Generate a histogram of image resolutions across all datasets.

    Args:
        summary: Aggregate audit summary containing per-image metadata.
        output_path: Destination PNG path.
        title: Chart title.
    """
    _ensure_parent(output_path)

    megapixels: list[float] = []
    for result in summary.datasets.values():
        for img in result.images:
            if img.is_corrupted or not img.width or not img.height:
                continue
            megapixels.append((img.width * img.height) / 1_000_000)

    if not megapixels:
        logger.warning("No resolution data available for histogram.")
        _save_empty_chart(output_path, title, "No resolution data available.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(megapixels, bins=30, color=CHART_COLORS, edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel("Megapixels (width × height / 1e6)")
    ax.set_ylabel("Frequency")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info("Saved resolution histogram to %s", output_path)


def plot_format_pie_chart(
    format_stats: dict[str, int],
    output_path: Path,
    *,
    title: str = "Image Format Distribution",
) -> None:
    """Generate a pie chart of image format counts.

    Args:
        format_stats: Mapping of format extension to image count.
        output_path: Destination PNG path.
        title: Chart title.
    """
    _ensure_parent(output_path)

    if not format_stats:
        logger.warning("No format data available for pie chart.")
        _save_empty_chart(output_path, title, "No format data available.")
        return

    labels = list(format_stats.keys())
    sizes = list(format_stats.values())

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        counterclock=False,
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info("Saved format pie chart to %s", output_path)


def _save_empty_chart(output_path: Path, title: str, message: str) -> None:
    """Save a placeholder chart when no data is available."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _format_resolution_stats(result: DatasetAuditResult) -> str:
    """Format resolution statistics as markdown bullet points."""
    stats = result.resolution_stats
    if stats is None:
        return "- No valid images available for resolution statistics.\n"

    return (
        f"- Images measured: {stats.count}\n"
        f"- Width (px): min={stats.width_min}, max={stats.width_max}, "
        f"mean={stats.width_mean:.1f}, median={stats.width_median:.1f}\n"
        f"- Height (px): min={stats.height_min}, max={stats.height_max}, "
        f"mean={stats.height_mean:.1f}, median={stats.height_median:.1f}\n"
        f"- Megapixels: min={stats.megapixels_min:.3f}, max={stats.megapixels_max:.3f}, "
        f"mean={stats.megapixels_mean:.3f}, median={stats.megapixels_median:.3f}\n"
    )


def _format_class_imbalance(result: DatasetAuditResult) -> str:
    """Format class imbalance metrics as markdown."""
    imbalance = result.class_imbalance
    if imbalance is None:
        return "- No classes detected.\n"

    return (
        f"- Majority class: `{imbalance.majority_class}` ({imbalance.majority_count} images)\n"
        f"- Minority class: `{imbalance.minority_class}` ({imbalance.minority_count} images)\n"
        f"- Imbalance ratio (max/min): {imbalance.imbalance_ratio:.2f}\n"
        f"- Coefficient of variation: {imbalance.coefficient_of_variation:.3f}\n"
    )


def _format_duplicate_section(result: DatasetAuditResult) -> str:
    """Format duplicate detection results as markdown."""
    lines = [
        f"- Perceptual duplicate groups: {len(result.duplicate_hash_groups)}",
        f"- Duplicate filename groups: {len(result.duplicate_filenames)}",
    ]

    if result.duplicate_hash_groups:
        lines.append("\n**Perceptual duplicate groups (sample, max 5):**\n")
        for group in result.duplicate_hash_groups[:5]:
            lines.append(f"- {len(group)} images: {', '.join(group[:3])}"
                         f"{'...' if len(group) > 3 else ''}")

    if result.duplicate_filenames:
        lines.append("\n**Duplicate filenames (sample, max 5):**\n")
        for filename, paths in list(result.duplicate_filenames.items())[:5]:
            lines.append(f"- `{filename}`: {len(paths)} occurrences")

    return "\n".join(lines) + "\n"


def generate_markdown_report(summary: AuditSummary, output_path: Path) -> None:
    """Write a human-readable markdown audit report.

    Args:
        summary: Completed audit summary.
        output_path: Destination markdown file path.
    """
    _ensure_parent(output_path)
    lines: list[str] = [
        "# Dataset Audit Report",
        "",
        f"**Generated:** {summary.generated_at}  ",
        f"**Source directory:** `{summary.source_dir}`  ",
        f"**Total images:** {summary.total_images}  ",
        f"**Total classes (aggregate):** {summary.total_classes}  ",
        "",
        "> This audit is read-only. No images were modified, renamed, or preprocessed.",
        "",
        "## Visualizations",
        "",
        "| Chart | File |",
        "|-------|------|",
        "| Class distribution (all, top 20, bottom 20) | `class_distribution.png` |",
        "| Resolution histogram | `image_resolution.png` |",
        "| Format distribution | `image_formats.png` |",
        "",
    ]

    if not summary.datasets:
        lines.extend(
            [
                "## No Datasets Found",
                "",
                "No dataset directories were discovered under the audited source directory.",
                "Place dataset folders inside `datasets/external/` and re-run the audit.",
                "",
            ]
        )
    else:
        lines.append("## Aggregate Summary\n")
        lines.append("### Format Distribution\n")
        for fmt, count in sorted(summary.aggregate_format_stats.items()):
            lines.append(f"- `{fmt}`: {count}")
        lines.append("")

        for name, result in sorted(summary.datasets.items()):
            lines.extend(
                [
                    f"## Dataset: `{name}`",
                    "",
                    f"**Path:** `{result.dataset_path}`  ",
                    f"**Total images:** {result.total_images}  ",
                    f"**Total classes:** {result.total_classes}  ",
                    f"**Max directory depth:** {result.max_directory_depth}  ",
                    "",
                    "### Class Distribution\n",
                    _format_class_imbalance(result),
                    "",
                    "### Class Names\n",
                ]
            )

            if result.class_names:
                for class_name in result.class_names:
                    count = result.images_per_class[class_name]
                    lines.append(f"- `{class_name}`: {count} images")
            else:
                lines.append("- No classes detected.")
            lines.append("")

            lines.extend(
                [
                    "### Resolution Statistics\n",
                    _format_resolution_stats(result),
                    "",
                    "### Format Statistics\n",
                ]
            )
            if result.format_stats:
                for fmt, count in result.format_stats.items():
                    lines.append(f"- `{fmt}`: {count}")
            else:
                lines.append("- No images found.")
            lines.append("")

            lines.extend(
                [
                    "### Data Quality",
                    "",
                    f"- Empty folders: {len(result.empty_folders)}",
                    f"- Non-image files: {len(result.non_image_files)}",
                    f"- Corrupted images: {len(result.corrupted_images)}",
                    "",
                ]
            )

            if result.empty_folders:
                lines.append("**Empty folders (sample, max 10):**\n")
                for folder in result.empty_folders[:10]:
                    lines.append(f"- `{folder}`")
                lines.append("")

            if result.non_image_files:
                lines.append("**Non-image files (sample, max 10):**\n")
                for file_path in result.non_image_files[:10]:
                    lines.append(f"- `{file_path}`")
                lines.append("")

            if result.corrupted_images:
                lines.append("**Corrupted images (sample, max 10):**\n")
                for file_path in result.corrupted_images[:10]:
                    lines.append(f"- `{file_path}`")
                lines.append("")

            lines.extend(
                [
                    "### Duplicates",
                    "",
                    _format_duplicate_section(result),
                    "---",
                    "",
                ]
            )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved markdown report to %s", output_path)


def generate_audit_outputs(summary: AuditSummary, reports_dir: Path) -> None:
    """Generate all EDA artifacts for a completed audit.

    Writes markdown report, and PNG visualizations under ``reports_dir``.

    Args:
        summary: Completed audit summary from :func:`audit_all_datasets`.
        reports_dir: Output directory for charts and markdown.
    """
    reports_dir.mkdir(parents=True, exist_ok=True)

    generate_markdown_report(summary, reports_dir / "dataset_audit.md")

    plot_class_distribution_report(
        summary.aggregate_images_per_class,
        reports_dir / "class_distribution.png",
        title="Class Distribution Analysis (All Datasets)",
    )

    plot_resolution_histogram(summary, reports_dir / "image_resolution.png")
    plot_format_pie_chart(
        summary.aggregate_format_stats,
        reports_dir / "image_formats.png",
    )


# ---------------------------------------------------------------------------
# Cross-dataset comparison outputs
# ---------------------------------------------------------------------------


def plot_dataset_sizes(
    report: DatasetComparisonReport,
    output_path: Path,
    *,
    title: str = "Dataset Size Comparison",
) -> None:
    """Generate a grouped bar chart comparing images and classes per dataset.

    Args:
        report: Cross-dataset comparison report.
        output_path: Destination PNG path.
        title: Chart title.
    """
    _ensure_parent(output_path)

    if not report.datasets:
        _save_empty_chart(output_path, title, "No datasets available.")
        return

    df = pd.DataFrame(
        [
            {
                "dataset": entry.dataset_name,
                "images": entry.num_images,
                "classes": entry.num_classes,
            }
            for entry in report.datasets.values()
        ]
    ).set_index("dataset")

    fig, ax1 = plt.subplots(figsize=(10, 6))
    x = range(len(df))
    width = 0.35

    ax1.bar(
        [index - width / 2 for index in x],
        df["images"],
        width=width,
        label="Images",
        color="#2E86AB",
        edgecolor="white",
    )
    ax1.set_ylabel("Image Count")
    ax1.set_xlabel("Dataset")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(df.index, rotation=15, ha="right")
    ax1.grid(axis="y", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.bar(
        [index + width / 2 for index in x],
        df["classes"],
        width=width,
        label="Classes",
        color="#A23B72",
        edgecolor="white",
    )
    ax2.set_ylabel("Class Count")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info("Saved dataset sizes chart to %s", output_path)


def plot_class_overlap(
    report: DatasetComparisonReport,
    output_path: Path,
    *,
    title: str = "Class Overlap Across Datasets",
) -> None:
    """Generate a chart showing common and unique class counts per dataset.

    Args:
        report: Cross-dataset comparison report.
        output_path: Destination PNG path.
        title: Chart title.
    """
    _ensure_parent(output_path)

    if not report.datasets:
        _save_empty_chart(output_path, title, "No datasets available.")
        return

    overlap = report.class_overlap
    rows = []
    for name, entry in sorted(report.datasets.items()):
        normalized_count = len(overlap.normalized_classes_by_dataset.get(name, []))
        unique_count = len(overlap.unique_classes.get(name, []))
        common_count = len(overlap.common_classes)
        shared_not_unique = max(normalized_count - unique_count - common_count, 0)
        rows.append(
            {
                "dataset": name,
                "common": common_count,
                "shared": shared_not_unique,
                "unique": unique_count,
            }
        )

    df = pd.DataFrame(rows).set_index("dataset")
    fig, ax = plt.subplots(figsize=(10, 6))
    df[["common", "shared", "unique"]].plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=["#2E86AB", "#7FB685", "#A23B72"],
        edgecolor="white",
    )
    ax.set_title(title)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Class Count (normalized labels)")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(["Common to all", "Shared with others", "Unique"])
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info("Saved class overlap chart to %s", output_path)


def plot_resolution_comparison(
    report: DatasetComparisonReport,
    output_path: Path,
    *,
    title: str = "Average Image Resolution by Dataset",
) -> None:
    """Generate a grouped bar chart of average width and height per dataset.

    Args:
        report: Cross-dataset comparison report.
        output_path: Destination PNG path.
        title: Chart title.
    """
    _ensure_parent(output_path)

    rows = [
        {
            "dataset": entry.dataset_name,
            "avg_width": entry.average_width,
            "avg_height": entry.average_height,
            "avg_megapixels": entry.average_megapixels,
        }
        for entry in report.datasets.values()
        if entry.average_width is not None and entry.average_height is not None
    ]

    if not rows:
        _save_empty_chart(output_path, title, "No resolution data available.")
        return

    df = pd.DataFrame(rows).set_index("dataset")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    df[["avg_width", "avg_height"]].plot(
        kind="bar",
        ax=axes[0],
        color=["#2E86AB", "#F18F01"],
        edgecolor="white",
    )
    axes[0].set_title("Average Width and Height (px)")
    axes[0].set_xlabel("Dataset")
    axes[0].set_ylabel("Pixels")
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend(["Avg Width", "Avg Height"])

    df["avg_megapixels"].plot(
        kind="bar",
        ax=axes[1],
        color="#A23B72",
        edgecolor="white",
    )
    axes[1].set_title("Average Megapixels")
    axes[1].set_xlabel("Dataset")
    axes[1].set_ylabel("Megapixels")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info("Saved resolution comparison chart to %s", output_path)


def _format_per_dataset_comparison(
    summary: AuditSummary,
    report: DatasetComparisonReport,
) -> list[str]:
    """Build markdown sections for each dataset in the comparison report."""
    lines: list[str] = []

    for name in sorted(report.datasets.keys()):
        entry = report.datasets[name]
        audit = summary.datasets[name]
        lines.extend(
            [
                f"### `{name}`",
                "",
                f"- **Images:** {entry.num_images}",
                f"- **Classes:** {entry.num_classes}",
                f"- **Average resolution:** "
                f"{entry.average_width:.1f} × {entry.average_height:.1f} px"
                if entry.average_width and entry.average_height
                else "- **Average resolution:** unavailable",
                f"- **Average megapixels:** {entry.average_megapixels:.3f}"
                if entry.average_megapixels is not None
                else "- **Average megapixels:** unavailable",
                f"- **Image formats:** {', '.join(f'`{k}` ({v})' for k, v in entry.image_formats.items()) or 'none'}",
                f"- **Corrupted images:** {entry.corrupted_images}",
                f"- **Empty folders:** {entry.empty_folders}",
                f"- **Duplicate filename groups:** {entry.duplicate_filename_groups}",
                f"- **Max directory depth:** {entry.max_directory_depth}",
                "",
                "**Class names:**",
            ]
        )
        if audit.class_names:
            for class_name in audit.class_names:
                count = audit.images_per_class[class_name]
                lines.append(f"- `{class_name}`: {count} images")
        else:
            lines.append("- none")
        lines.append("")

    return lines


def generate_comparison_markdown(
    summary: AuditSummary,
    report: DatasetComparisonReport,
    output_path: Path,
) -> None:
    """Write a cross-dataset comparison markdown report.

    Args:
        summary: Completed audit summary with per-dataset details.
        report: Cross-dataset comparison report.
        output_path: Destination markdown file path.
    """
    _ensure_parent(output_path)
    overlap = report.class_overlap

    lines = [
        "# Dataset Comparison Report",
        "",
        f"**Generated:** {report.generated_at}  ",
        f"**Source directory:** `{report.source_dir}`  ",
        f"**Datasets compared:** {len(report.datasets)}  ",
        "",
        "> Read-only audit. No images were modified, merged, or preprocessed.",
        "",
        "## Summary Table",
        "",
        "| Dataset | Images | Classes | Avg Width | Avg Height | Avg MP | Formats |",
        "|---------|--------|---------|-----------|------------|--------|---------|",
    ]

    for name in sorted(report.datasets.keys()):
        entry = report.datasets[name]
        formats = ", ".join(entry.image_formats.keys()) or "—"
        avg_w = f"{entry.average_width:.0f}" if entry.average_width else "—"
        avg_h = f"{entry.average_height:.0f}" if entry.average_height else "—"
        avg_mp = f"{entry.average_megapixels:.3f}" if entry.average_megapixels else "—"
        lines.append(
            f"| `{name}` | {entry.num_images} | {entry.num_classes} | "
            f"{avg_w} | {avg_h} | {avg_mp} | {formats} |"
        )

    lines.extend(
        [
            "",
            "## Class Overlap",
            "",
            f"- **Common classes (all datasets):** {len(overlap.common_classes)}",
        ]
    )

    if overlap.common_classes:
        sample = overlap.common_classes[:20]
        lines.append(
            "- Sample: "
            + ", ".join(f"`{label}`" for label in sample)
            + (" ..." if len(overlap.common_classes) > 20 else "")
        )

    lines.append("")
    lines.append("### Unique Classes per Dataset")
    lines.append("")
    for name in sorted(overlap.unique_classes.keys()):
        unique = overlap.unique_classes[name]
        lines.append(f"- **`{name}`:** {len(unique)} unique class(es)")
        if unique:
            sample = unique[:10]
            lines.append("  - " + ", ".join(f"`{label}`" for label in sample))
            if len(unique) > 10:
                lines.append(f"  - ... and {len(unique) - 10} more")

    lines.extend(["", "### Missing Classes Between Datasets", ""])
    if not overlap.missing_classes:
        lines.append("No pairwise differences detected.")
    else:
        for target_name in sorted(overlap.missing_classes.keys()):
            lines.append(f"#### Missing from `{target_name}`")
            lines.append("")
            for reference_name, missing in sorted(
                overlap.missing_classes[target_name].items()
            ):
                lines.append(
                    f"- Present in `{reference_name}` but not `{target_name}`: "
                    f"{len(missing)} class(es)"
                )
                if missing:
                    sample = missing[:10]
                    lines.append("  - " + ", ".join(f"`{label}`" for label in sample))
                    if len(missing) > 10:
                        lines.append(f"  - ... and {len(missing) - 10} more")
            lines.append("")

    lines.extend(
        [
            "## Per-Dataset Details",
            "",
            *_format_per_dataset_comparison(summary, report),
            "## Visualizations",
            "",
            "| Chart | File |",
            "|-------|------|",
            "| Dataset sizes | `dataset_sizes.png` |",
            "| Class overlap | `class_overlap.png` |",
            "| Resolution comparison | `resolution_comparison.png` |",
            "",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved comparison markdown report to %s", output_path)


def generate_comparison_outputs(
    summary: AuditSummary,
    report: DatasetComparisonReport,
    comparison_dir: Path,
) -> None:
    """Generate all cross-dataset comparison artifacts.

    Args:
        summary: Completed audit summary with per-image metadata.
        report: Cross-dataset comparison report.
        comparison_dir: Output directory for comparison artifacts.
    """
    comparison_dir.mkdir(parents=True, exist_ok=True)

    generate_comparison_markdown(
        summary,
        report,
        comparison_dir / "dataset_comparison.md",
    )
    plot_dataset_sizes(report, comparison_dir / "dataset_sizes.png")
    plot_class_overlap(report, comparison_dir / "class_overlap.png")
    plot_resolution_comparison(report, comparison_dir / "resolution_comparison.png")
