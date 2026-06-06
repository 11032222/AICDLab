from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import is_image_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize prepared animal image metadata.")
    parser.add_argument("--metadata-csv", type=Path, default=Path("Data") / "metadata_clean_binary.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--sample-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sample_dimensions(image_paths: list[Path], sample_size: int, seed: int) -> tuple[list[tuple[int, int]], int]:
    valid_paths = [path for path in image_paths if is_image_file(path)]
    rng = random.Random(seed)
    sampled_paths = valid_paths if len(valid_paths) <= sample_size else rng.sample(valid_paths, sample_size)
    dimensions = []
    for image_path in sampled_paths:
        with Image.open(image_path) as image:
            dimensions.append(image.size)
    return dimensions, len(sampled_paths)


def build_markdown(frame: pd.DataFrame, dimensions: list[tuple[int, int]], sampled_count: int) -> str:
    total = len(frame)
    class_counts = frame.groupby(["label", "class_name"]).size().reset_index(name="count").sort_values("label")
    fold_counts = (
        frame.groupby(["fold", "label", "class_name"])
        .size()
        .reset_index(name="count")
        .sort_values(["fold", "label"])
        if "fold" in frame.columns
        else pd.DataFrame()
    )
    unique_dimensions = sorted(set(dimensions))
    lines = [
        "# Dataset Analysis",
        "",
        "## Summary",
        "",
        f"- Total images: {total}",
        f"- Number of classes: {frame['label'].nunique()}",
        f"- Dimension samples checked: {sampled_count}",
        f"- Unique sampled image sizes: {unique_dimensions[:20]}",
        "",
        "## Class Distribution",
        "",
        "| Label | Class Name | Count | Share |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in class_counts.itertuples(index=False):
        lines.append(f"| {row.label} | {row.class_name} | {row.count} | {row.count / total:.2%} |")

    if not fold_counts.empty:
        lines.extend(["", "## Fold Distribution", "", "| Fold | Label | Class Name | Count |", "| ---: | ---: | --- | ---: |"])
        for row in fold_counts.itertuples(index=False):
            lines.append(f"| {row.fold} | {row.label} | {row.class_name} | {row.count} |")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Use balanced accuracy and macro F1 with accuracy, especially if the class counts are not even.",
            "- Keep the same validation fold when comparing EfficientNet-B7, EfficientNet-B0, and ResNet18 runs.",
        ]
    )
    return "\n".join(lines) + "\n"


def save_distribution_plot(frame: pd.DataFrame, output_path: Path) -> None:
    summary = frame.groupby(["label", "class_name"]).size().reset_index(name="count").sort_values("label")
    labels = [f"{row.label}: {row.class_name}" for row in summary.itertuples(index=False)]
    counts = summary["count"].tolist()
    plt.figure(figsize=(8, 4.5))
    plt.bar(labels, counts, color=["#2f80ed", "#27ae60"])
    plt.ylabel("Images")
    plt.title("Binary Animal Class Distribution")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.metadata_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = [Path(path) for path in frame["image_path"].tolist()]
    dimensions, sampled_count = sample_dimensions(image_paths, args.sample_size, args.seed)
    report = build_markdown(frame, dimensions, sampled_count)

    report_path = args.output_dir / "dataset_analysis.md"
    report_path.write_text(report, encoding="utf-8")
    plot_path = args.output_dir / "class_distribution.png"
    save_distribution_plot(frame, plot_path)
    print(f"wrote_report={report_path}")
    print(f"wrote_plot={plot_path}")


if __name__ == "__main__":
    main()
