from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import is_image_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create stratified CSV folds for a binary animal image dataset.")
    parser.add_argument("--data-dir", type=Path, default=Path("Data"), help="Kaggle dataset root after extraction.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Where CSV files are written. Defaults to data dir.")
    parser.add_argument("--classes", nargs=2, default=None, help="Two class folder names or aliases to keep.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-class", type=int, default=None, help="Optional cap for quick experiments.")
    return parser.parse_args()


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def discover_images(data_dir: Path) -> pd.DataFrame:
    rows = []
    for image_path in sorted(data_dir.rglob("*")):
        if not is_image_file(image_path):
            continue
        parent = image_path.parent.name
        if normalize_name(parent) in {"train", "test", "valid", "validation", "images"}:
            # For flat folders named "images", use the next parent as the class if possible.
            parent = image_path.parent.parent.name
        rows.append(
            {
                "image_path": str(image_path.resolve()),
                "class_name": parent,
                "source_split": infer_source_split(image_path, data_dir),
            }
        )
    if not rows:
        raise FileNotFoundError(f"No image files found under {data_dir}")
    return pd.DataFrame(rows)


def infer_source_split(image_path: Path, data_dir: Path) -> str:
    try:
        relative_parts = [normalize_name(part) for part in image_path.relative_to(data_dir).parts]
    except ValueError:
        relative_parts = [normalize_name(part) for part in image_path.parts]
    for split_name in ("train", "training", "trainingset"):
        if split_name in relative_parts:
            return "train"
    for split_name in ("valid", "validation", "val", "validationset"):
        if split_name in relative_parts:
            return "val"
    for split_name in ("test", "testset", "testing"):
        if split_name in relative_parts:
            return "test"
    return "unknown"


def choose_classes(frame: pd.DataFrame, requested: list[str] | None) -> list[str]:
    counts = frame["class_name"].value_counts()
    if requested is None:
        if len(counts) == 2:
            return sorted(counts.index.tolist(), key=normalize_name)

        common_binary_aliases = [
            ("cat", "dog"),
            ("cats", "dogs"),
            ("Cat", "Dog"),
            ("Cats", "Dogs"),
        ]
        names_by_norm = {normalize_name(name): name for name in counts.index}
        for left, right in common_binary_aliases:
            if normalize_name(left) in names_by_norm and normalize_name(right) in names_by_norm:
                return [names_by_norm[normalize_name(left)], names_by_norm[normalize_name(right)]]

        top_classes = counts.head(10).to_dict()
        raise ValueError(
            "Could not infer exactly two classes. Pass --classes CLASS_A CLASS_B. "
            f"Most common discovered classes: {top_classes}"
        )

    names_by_norm = {normalize_name(name): name for name in counts.index}
    selected = []
    for name in requested:
        normalized = normalize_name(name)
        if normalized not in names_by_norm:
            raise ValueError(f"Requested class {name!r} was not found. Available classes: {sorted(counts.index)}")
        selected.append(names_by_norm[normalized])
    return selected


def assign_labels(frame: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    label_map = {class_name: idx for idx, class_name in enumerate(classes)}
    filtered = frame[frame["class_name"].isin(classes)].copy()
    filtered["label"] = filtered["class_name"].map(label_map).astype(int)
    return filtered


def cap_per_class(frame: pd.DataFrame, max_per_class: int | None, seed: int) -> pd.DataFrame:
    if max_per_class is None:
        return frame
    rng = random.Random(seed)
    pieces = []
    for _, group in frame.groupby("label"):
        indices = group.index.tolist()
        rng.shuffle(indices)
        pieces.append(group.loc[indices[:max_per_class]])
    return pd.concat(pieces).reset_index(drop=True)


def assign_folds(frame: pd.DataFrame, n_folds: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    folded = frame.copy().reset_index(drop=True)
    folded["fold"] = -1
    for label, group in folded.groupby("label"):
        indices = list(group.index)
        rng.shuffle(indices)
        for position, index in enumerate(indices):
            folded.at[index, "fold"] = position % n_folds
    return folded


def write_outputs(
    metadata: pd.DataFrame,
    fold_frame: pd.DataFrame,
    output_dir: Path,
    classes: list[str],
    folds: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata_clean_binary.csv"
    metadata.to_csv(metadata_path, index=False)

    test_frame = metadata[metadata["source_split"] == "test"].copy()
    if not test_frame.empty:
        test_path = output_dir / "test.csv"
        test_frame.to_csv(test_path, index=False)

    class_map = {str(index): class_name for index, class_name in enumerate(classes)}
    with (output_dir / "label_num_to_class_map.json").open("w", encoding="utf-8") as handle:
        json.dump(class_map, handle, indent=2, ensure_ascii=False)

    fold_dir = output_dir / "folds"
    fold_dir.mkdir(parents=True, exist_ok=True)
    for fold_id in range(folds):
        train_df = fold_frame[fold_frame["fold"] != fold_id].copy()
        val_df = fold_frame[fold_frame["fold"] == fold_id].copy()
        train_df.to_csv(fold_dir / f"fold_{fold_id}_train.csv", index=False)
        val_df.to_csv(fold_dir / f"fold_{fold_id}_val.csv", index=False)
        print(f"saved_fold={fold_id} train={len(train_df)} val={len(val_df)}")

    summary = (
        fold_frame.groupby(["fold", "label", "class_name"])
        .size()
        .reset_index(name="count")
        .sort_values(["fold", "label"])
    )
    summary.to_csv(fold_dir / "fold_summary.csv", index=False)
    print(f"wrote_metadata={metadata_path}")
    if not test_frame.empty:
        print(f"wrote_test={test_path}")
    print(f"wrote_label_map={output_dir / 'label_num_to_class_map.json'}")
    print(f"wrote_fold_summary={fold_dir / 'fold_summary.csv'}")


def main() -> None:
    args = parse_args()
    if args.folds < 2:
        raise ValueError("--folds must be >= 2")

    data_dir = args.data_dir.resolve()
    output_dir = (args.output_dir or args.data_dir).resolve()
    discovered = discover_images(data_dir)
    classes = choose_classes(discovered, args.classes)
    metadata = assign_labels(discovered, classes).reset_index(drop=True)
    metadata = cap_per_class(metadata, args.max_per_class, args.seed)

    if (metadata["source_split"] == "train").any():
        fold_source = metadata[metadata["source_split"] == "train"].copy()
    else:
        fold_source = metadata[metadata["source_split"] != "test"].copy()
    if fold_source.empty:
        raise ValueError("No training images are available for fold creation.")
    folded = assign_folds(fold_source, args.folds, args.seed)

    print(f"data_dir={data_dir}")
    print(f"classes={classes}")
    print(f"all_class_counts={metadata['class_name'].value_counts().to_dict()}")
    print(f"fold_class_counts={folded['class_name'].value_counts().to_dict()}")
    print(f"source_split_counts={metadata['source_split'].value_counts().to_dict()}")
    write_outputs(metadata, folded, output_dir, classes, args.folds)


if __name__ == "__main__":
    main()
