from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import ImageCsvDataset, build_transforms
from src.metrics import (
    compute_balanced_accuracy,
    compute_class_f1s,
    compute_class_precisions,
    compute_class_recalls,
    find_best_binary_threshold,
)
from src.models import build_model, get_model_data_config
from train import forward_with_tta, infer_class_names, infer_mamba_architecture_from_checkpoint, infer_num_classes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved binary animal classifier checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path, default=None, help="Optional CSV used only for class names.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval-tta-flips", action="store_true")
    parser.add_argument("--full-metrics", action="store_true", help="Include per-class metrics and threshold search.")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


@torch.no_grad()
def evaluate_checkpoint(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_name = str(checkpoint["model_name"])
    checkpoint_args = checkpoint.get("args", {})
    if not isinstance(checkpoint_args, dict):
        checkpoint_args = {}

    image_size = int(checkpoint.get("image_size", checkpoint_args.get("image_size", 224)))
    num_classes = int(checkpoint.get("num_classes", infer_num_classes(args.val_csv, args.val_csv)))
    model_config = get_model_data_config(model_name, image_size)
    eval_resize_size = int(checkpoint.get("eval_resize_size", checkpoint_args.get("eval_resize_size", model_config.eval_resize_size)))
    interpolation_name = str(checkpoint.get("interpolation", checkpoint_args.get("interpolation", model_config.interpolation.name)))
    interpolation = InterpolationMode[interpolation_name]
    mamba_architecture = infer_mamba_architecture_from_checkpoint(checkpoint, "hybrid")

    model = build_model(
        model_name,
        num_classes=num_classes,
        image_size=image_size,
        pretrained=False,
        mamba_architecture=mamba_architecture,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = ImageCsvDataset(
        args.val_csv,
        transform=build_transforms(
            image_size=image_size,
            train=False,
            resize_size=eval_resize_size,
            interpolation=interpolation,
        ),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    criterion = nn.CrossEntropyLoss()
    use_amp = device.type == "cuda"

    running_loss = 0.0
    correct = 0
    total = 0
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    labels_cache: list[int] = []
    positive_probs: list[float] = []

    for batch_idx, (images, labels) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")
        logits = forward_with_tta(model, images, device, use_amp, args.eval_tta_flips)
        loss = criterion(logits, labels)
        predictions = logits.argmax(dim=1)

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        correct += (predictions == labels).sum().item()
        total += batch_size
        if args.full_metrics and num_classes == 2:
            positive_probs.extend(torch.softmax(logits, dim=1)[:, 1].detach().cpu().tolist())
            labels_cache.extend(labels.detach().cpu().tolist())
        for truth, pred in zip(labels.cpu(), predictions.cpu()):
            confusion[truth, pred] += 1

        if batch_idx % 20 == 0 or batch_idx == len(loader):
            print(f"[eval] batch={batch_idx:04d}/{len(loader):04d}", flush=True)

    confusion_list = confusion.tolist()
    class_names = checkpoint.get("class_names")
    if not isinstance(class_names, list):
        class_names = infer_class_names(args.train_csv or args.val_csv, args.val_csv, num_classes)

    result: dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "model_name": model_name,
        "mamba_architecture": mamba_architecture if model_name == "mamba" else None,
        "epoch": checkpoint.get("epoch"),
        "class_names": class_names,
        "val_loss": running_loss / total,
        "val_acc": correct / total,
        "target_acc": checkpoint.get("target_acc", 0.85),
        "target_met": correct / total >= float(checkpoint.get("target_acc", 0.85)),
        "confusion_matrix": confusion_list,
    }
    if args.full_metrics:
        class_recalls = compute_class_recalls(confusion_list)
        class_precisions = compute_class_precisions(confusion_list)
        class_f1s = compute_class_f1s(class_precisions, class_recalls)
        result.update(
            {
                "balanced_acc": compute_balanced_accuracy(confusion_list),
                "macro_f1": sum(class_f1s) / len(class_f1s) if class_f1s else 0.0,
                "class_recalls": class_recalls,
                "class_precisions": class_precisions,
                "class_f1s": class_f1s,
                "threshold_metrics": find_best_binary_threshold(labels_cache, positive_probs)
                if num_classes == 2
                else None,
            }
        )
    return result


def main() -> None:
    args = parse_args()
    result = evaluate_checkpoint(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output_json is not None:
        write_json(args.output_json, result)
        print(f"wrote_output_json={args.output_json}")


if __name__ == "__main__":
    main()
