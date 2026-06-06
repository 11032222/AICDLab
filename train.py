from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from src.data import ImageCsvDataset, build_transforms, build_weighted_sampler
from src.metrics import (
    compute_balanced_accuracy,
    compute_class_f1s,
    compute_class_precisions,
    compute_class_recalls,
    find_best_binary_threshold,
)
from src.models import build_model, count_parameters, get_model_data_config

TRAINING_OUTPUT_ROOT = Path("artifacts") / "training_outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an animal image binary classifier.")
    parser.add_argument("--train-csv", type=Path, default=None, help="CSV for training (auto-generated if --class1/--class2 given)")
    parser.add_argument("--val-csv", type=Path, default=None, help="CSV for validation (auto-generated if --class1/--class2 given)")
    parser.add_argument("--class1", type=str, default=None, help="First class name, e.g. Cat")
    parser.add_argument("--class2", type=str, default=None, help="Second class name, e.g. Dog")
    parser.add_argument("--train-dir", type=Path, default=Path("Training Data"), help="Training data root dir")
    parser.add_argument("--val-dir", type=Path, default=Path("Validation Data"), help="Validation data root dir")
    parser.add_argument("--model", choices=["vit", "mamba", "efficientnet_b7", "efficientnet_b0", "resnet18"], default="vit")
    parser.add_argument(
        "--mamba-architecture",
        choices=["hybrid", "patch"],
        default="hybrid",
        help="hybrid uses ImageNet EfficientNet features plus official Mamba blocks; patch is pure patch Mamba.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("animal_binary_mamba"))
    parser.add_argument("--image-size", type=int, default=None, help="Defaults to the model's native size.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--use-class-weights", action="store_true")
    parser.add_argument("--balanced-sampling", action="store_true")
    parser.add_argument("--use-randaugment", action="store_true")
    parser.add_argument("--use-random-erasing", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--warmup-start-factor", type=float, default=0.1)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=0)
    parser.add_argument("--backbone-lr-scale", type=float, default=0.2)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--eval-tta-flips", action="store_true")
    parser.add_argument("--full-metrics", action="store_true", help="Store per-class metrics and threshold search.")
    parser.add_argument("--target-acc", type=float, default=0.85)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-from", type=Path, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log_message(message: str) -> None:
    print(message, flush=True)


def write_json(path: Path, payload: dict[str, object] | list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_history_files(history_path: Path, metrics_jsonl_path: Path, history: list[dict[str, object]]) -> None:
    write_json(history_path, history)
    with metrics_jsonl_path.open("w", encoding="utf-8") as handle:
        for record in history:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def resolve_output_dir(output_dir: Path) -> Path:
    if output_dir.is_absolute():
        return output_dir
    if len(output_dir.parts) >= 2 and output_dir.parts[:2] == TRAINING_OUTPUT_ROOT.parts[:2]:
        return output_dir
    return TRAINING_OUTPUT_ROOT / output_dir


def resolve_image_size(args: argparse.Namespace) -> tuple[int, int, object]:
    requested = args.image_size if args.image_size is not None else 224
    config = get_model_data_config(args.model, requested)
    if args.image_size is not None and args.image_size != config.image_size:
        raise ValueError(
            f"{args.model} uses image_size={config.image_size}; received --image-size={args.image_size}."
        )
    return config.image_size, config.eval_resize_size, config.interpolation


def infer_num_classes(train_csv: Path, val_csv: Path) -> int:
    train_labels = pd.read_csv(train_csv)["label"]
    val_labels = pd.read_csv(val_csv)["label"]
    unique_labels = sorted(set(train_labels.tolist()) | set(val_labels.tolist()))
    return len(unique_labels)


def infer_class_names(train_csv: Path, val_csv: Path, num_classes: int) -> list[str]:
    label_to_name: dict[int, str] = {}
    for csv_path in (train_csv, val_csv):
        frame = pd.read_csv(csv_path)
        if "class_name" not in frame.columns:
            continue
        dedup = frame[["label", "class_name"]].drop_duplicates()
        for row in dedup.itertuples(index=False):
            label_to_name[int(row.label)] = str(row.class_name)
    return [label_to_name.get(index, f"class_{index}") for index in range(num_classes)]


def build_class_weights(csv_path: Path, num_classes: int) -> torch.Tensor:
    frame = pd.read_csv(csv_path)
    counts = frame["label"].value_counts().sort_index().reindex(range(num_classes), fill_value=1)
    frequencies = counts.to_numpy(dtype=np.float32)
    weights = 1.0 / np.sqrt(frequencies)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return (predictions == labels).float().mean().item()


def format_seconds(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def build_scheduler(optimizer: torch.optim.Optimizer, args: argparse.Namespace) -> torch.optim.lr_scheduler.LRScheduler:
    if args.warmup_epochs > 0 and args.warmup_epochs < args.epochs:
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=args.warmup_start_factor,
            end_factor=1.0,
            total_iters=args.warmup_epochs,
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.epochs - args.warmup_epochs,
            eta_min=args.min_lr,
        )
        return SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[args.warmup_epochs])
    return CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)


def maybe_set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    setter = getattr(model, "set_backbone_trainable", None)
    if callable(setter):
        setter(trainable)


def resolve_optimizer_params(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    backbone_lr_scale: float,
) -> list[dict[str, object]] | object:
    group_builder = getattr(model, "parameter_groups", None)
    if callable(group_builder):
        return group_builder(lr=lr, weight_decay=weight_decay, backbone_lr_scale=backbone_lr_scale)
    return model.parameters()


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def forward_with_tta(
    model: nn.Module,
    images: torch.Tensor,
    device: torch.device,
    use_amp: bool,
    eval_tta_flips: bool,
) -> torch.Tensor:
    views = [images]
    if eval_tta_flips:
        views.append(torch.flip(images, dims=[3]))

    logits_sum = None
    for view in views:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            view_logits = model(view)
        logits_sum = view_logits if logits_sum is None else logits_sum + view_logits
    return logits_sum / len(views)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_grad_norm: float,
    epoch: int,
    total_epochs: int,
    log_interval: int,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
    grad_accum_steps: int,
    progress_path: Path,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    seen_samples = 0
    epoch_start = time.time()
    total_batches = len(loader)
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, (images, labels) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
        loss_for_backward = loss / grad_accum_steps
        should_step = batch_idx % grad_accum_steps == 0 or batch_idx == total_batches

        if use_amp:
            scaler.scale(loss_for_backward).backward()
            if should_step:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            loss_for_backward.backward()
            if should_step:
                clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        batch_size = images.size(0)
        seen_samples += batch_size
        running_loss += loss.item() * batch_size
        running_acc += accuracy_from_logits(logits, labels) * batch_size

        if batch_idx % log_interval == 0 or batch_idx == total_batches:
            elapsed = time.time() - epoch_start
            eta = (elapsed / batch_idx) * (total_batches - batch_idx)
            current_loss = running_loss / seen_samples
            current_acc = running_acc / seen_samples
            write_json(
                progress_path,
                {
                    "phase": "train",
                    "epoch": epoch,
                    "total_epochs": total_epochs,
                    "batch": batch_idx,
                    "total_batches": total_batches,
                    "loss": current_loss,
                    "acc": current_acc,
                    "elapsed_seconds": elapsed,
                    "eta_seconds": eta,
                },
            )
            log_message(
                f"[train] epoch={epoch:02d}/{total_epochs:02d} "
                f"batch={batch_idx:04d}/{total_batches:04d} "
                f"loss={current_loss:.4f} acc={current_acc:.4f} "
                f"elapsed={format_seconds(elapsed)} eta={format_seconds(eta)}"
            )

    dataset_size = len(loader.dataset)
    return running_loss / dataset_size, running_acc / dataset_size


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    epoch: int,
    total_epochs: int,
    log_interval: int,
    use_amp: bool,
    eval_tta_flips: bool,
    full_metrics: bool,
    progress_path: Path,
) -> tuple[float, float, list[list[int]], dict[str, object] | None]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    eval_start = time.time()
    total_batches = len(loader)
    all_labels: list[int] = []
    all_positive_probs: list[float] = []

    for batch_idx, (images, labels) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")
        logits = forward_with_tta(model, images, device, use_amp, eval_tta_flips)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            loss = criterion(logits, labels)

        predictions = logits.argmax(dim=1)
        if full_metrics and num_classes == 2:
            positive_probs = torch.softmax(logits, dim=1)[:, 1]
            all_positive_probs.extend(positive_probs.detach().cpu().tolist())
            all_labels.extend(labels.detach().cpu().tolist())

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        correct += (predictions == labels).sum().item()
        total += batch_size

        for truth, pred in zip(labels.cpu(), predictions.cpu()):
            confusion[truth, pred] += 1

        if batch_idx % log_interval == 0 or batch_idx == total_batches:
            elapsed = time.time() - eval_start
            eta = (elapsed / batch_idx) * (total_batches - batch_idx)
            current_loss = running_loss / total
            current_acc = correct / total
            write_json(
                progress_path,
                {
                    "phase": "val",
                    "epoch": epoch,
                    "total_epochs": total_epochs,
                    "batch": batch_idx,
                    "total_batches": total_batches,
                    "loss": current_loss,
                    "acc": current_acc,
                    "elapsed_seconds": elapsed,
                    "eta_seconds": eta,
                },
            )
            log_message(
                f"[val]   epoch={epoch:02d}/{total_epochs:02d} "
                f"batch={batch_idx:04d}/{total_batches:04d} "
                f"loss={current_loss:.4f} acc={current_acc:.4f} "
                f"elapsed={format_seconds(elapsed)} eta={format_seconds(eta)}"
            )

    threshold_metrics = (
        find_best_binary_threshold(all_labels, all_positive_probs)
        if full_metrics and num_classes == 2
        else None
    )
    return running_loss / total, correct / total, confusion.tolist(), threshold_metrics


def build_epoch_record(
    epoch: int,
    train_loss: float,
    train_acc: float,
    val_loss: float,
    val_acc: float,
    confusion: list[list[int]],
    threshold_metrics: dict[str, object] | None,
    learning_rates: list[float],
    elapsed_seconds: float,
    target_acc: float,
    full_metrics: bool,
) -> dict[str, object]:
    record: dict[str, object] = {
        "epoch": epoch,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "target_acc": target_acc,
        "target_met": val_acc >= target_acc,
        "confusion_matrix": confusion,
        "lr": learning_rates,
        "elapsed_seconds": elapsed_seconds,
    }
    if not full_metrics:
        return record

    class_recalls = compute_class_recalls(confusion)
    class_precisions = compute_class_precisions(confusion)
    class_f1s = compute_class_f1s(class_precisions, class_recalls)
    record.update(
        {
            "balanced_acc": compute_balanced_accuracy(confusion),
            "macro_f1": sum(class_f1s) / len(class_f1s) if class_f1s else 0.0,
            "class_recalls": class_recalls,
            "class_precisions": class_precisions,
            "class_f1s": class_f1s,
            "threshold_metrics": threshold_metrics,
        }
    )
    return record


def save_checkpoint(path: Path, payload: dict[str, object]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp_path)
    temp_path.replace(path)


def resolve_resume_checkpoint(output_dir: Path, args: argparse.Namespace) -> Path | None:
    if args.resume_from is not None:
        return args.resume_from
    if args.resume:
        return output_dir / "last.pt"
    return None


def infer_mamba_architecture_from_checkpoint(checkpoint: dict[str, object], default: str) -> str:
    checkpoint_args = checkpoint.get("args", {})
    if isinstance(checkpoint_args, dict) and isinstance(checkpoint_args.get("mamba_architecture"), str):
        return checkpoint_args["mamba_architecture"]
    if isinstance(checkpoint.get("mamba_architecture"), str):
        return str(checkpoint["mamba_architecture"])

    state_dict = checkpoint.get("model_state_dict", {})
    if isinstance(state_dict, dict):
        keys = state_dict.keys()
        if any(key.startswith("patch_embed.") for key in keys):
            return "patch"
        if any(key.startswith("feature_extractor.") for key in keys):
            return "hybrid"
    return default


def main() -> None:
    args = parse_args()
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be >= 1.")

    set_seed(args.seed)
    output_dir = resolve_output_dir(args.output_dir) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_checkpoint_path = resolve_resume_checkpoint(output_dir, args)
    resume_checkpoint = None
    if resume_checkpoint_path is not None and resume_checkpoint_path.exists():
        resume_checkpoint = torch.load(resume_checkpoint_path, map_location="cpu", weights_only=False)
        if args.model == "mamba":
            args.mamba_architecture = infer_mamba_architecture_from_checkpoint(
                resume_checkpoint,
                args.mamba_architecture,
            )

    device = torch.device(args.device)
    use_amp = args.amp and device.type == "cuda"
    image_size, eval_resize_size, interpolation = resolve_image_size(args)
    args.image_size = image_size
    args.eval_resize_size = eval_resize_size
    args.interpolation = interpolation.name

    # Auto-generate CSVs from directory structure if --class1/--class2 given
    if args.train_csv is None and args.class1 and args.class2:
        os.makedirs('_csv', exist_ok=True)
        for split_name, dir_path in [('train', args.train_dir), ('val', args.val_dir)]:
            rows = []
            for label, cls in [(0, args.class1), (1, args.class2)]:
                cls_dir = dir_path / cls
                if cls_dir.exists():
                    for fname in sorted(os.listdir(cls_dir)):
                        if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                            rows.append({"image_path": str((cls_dir / fname).resolve()), "label": label, "class_name": cls})
            csv_path = Path(f'_csv/{split_name}.csv')
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            if split_name == 'train':
                args.train_csv = csv_path
            else:
                args.val_csv = csv_path
            log_message(f"Generated {csv_path} with {len(rows)} samples")

    if args.train_csv is None or args.val_csv is None:
        raise ValueError("Provide --train-csv/--val-csv OR --class1/--class2 for auto-CSV generation.")

    inferred_num_classes = infer_num_classes(args.train_csv, args.val_csv)
    num_classes = args.num_classes or inferred_num_classes
    if inferred_num_classes != num_classes:
        raise ValueError(f"CSV labels imply {inferred_num_classes} classes, but --num-classes={num_classes}.")
    class_names = infer_class_names(args.train_csv, args.val_csv, num_classes)

    train_dataset = ImageCsvDataset(
        args.train_csv,
        transform=build_transforms(
            image_size=image_size,
            train=True,
            use_randaugment=args.use_randaugment,
            use_random_erasing=args.use_random_erasing,
            interpolation=interpolation,
        ),
    )
    val_dataset = ImageCsvDataset(
        args.val_csv,
        transform=build_transforms(
            image_size=image_size,
            train=False,
            resize_size=eval_resize_size,
            interpolation=interpolation,
        ),
    )

    train_sampler = build_weighted_sampler(args.train_csv) if args.balanced_sampling else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )

    model = build_model(
        args.model,
        num_classes=num_classes,
        image_size=image_size,
        pretrained=not args.no_pretrained,
        mamba_architecture=args.mamba_architecture,
    ).to(device)
    log_message(f"model={args.model} parameters={count_parameters(model):,}")
    if args.model == "mamba":
        log_message(f"mamba_architecture={args.mamba_architecture}")
    log_message(f"device={device} amp={use_amp} pretrained={not args.no_pretrained}")
    log_message(f"class_names={class_names}")
    log_message(f"image_size={image_size} eval_resize_size={eval_resize_size} interpolation={interpolation.name}")
    log_message(f"resolved_output_dir={output_dir}")

    class_weights = None
    if args.use_class_weights:
        class_weights = build_class_weights(args.train_csv, num_classes).to(device)
        log_message(f"class_weights={class_weights.tolist()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    optimizer = AdamW(
        resolve_optimizer_params(model, args.lr, args.weight_decay, args.backbone_lr_scale),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = build_scheduler(optimizer, args)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history: list[dict[str, object]] = []
    best_val_acc = -1.0
    best_balanced_acc = -1.0
    best_threshold_balanced_acc = -1.0
    start_epoch = 1
    train_start = time.time()
    progress_path = output_dir / "progress.json"
    history_path = output_dir / "history.json"
    metrics_jsonl_path = output_dir / "metrics.jsonl"

    if resume_checkpoint_path is not None:
        checkpoint = resume_checkpoint or torch.load(resume_checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        move_optimizer_state_to_device(optimizer, device)
        if checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if checkpoint.get("scaler_state_dict") is not None and use_amp:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        history = list(checkpoint.get("history", []))
        best_val_acc = float(checkpoint.get("best_val_acc", best_val_acc))
        best_balanced_acc = float(checkpoint.get("best_balanced_acc", best_balanced_acc))
        best_threshold_balanced_acc = float(
            checkpoint.get("best_threshold_balanced_acc", best_threshold_balanced_acc)
        )
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        train_start = time.time() - float(checkpoint.get("total_elapsed_seconds", 0.0))
        write_history_files(history_path, metrics_jsonl_path, history)
        log_message(f"resumed_from={resume_checkpoint_path} next_epoch={start_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):
        backbone_trainable = epoch > args.freeze_backbone_epochs
        maybe_set_backbone_trainable(model, backbone_trainable)
        log_message(f"epoch={epoch:02d} backbone_trainable={backbone_trainable}")

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            max_grad_norm=args.max_grad_norm,
            epoch=epoch,
            total_epochs=args.epochs,
            log_interval=args.log_interval,
            scaler=scaler,
            use_amp=use_amp,
            grad_accum_steps=args.grad_accum_steps,
            progress_path=progress_path,
        )
        val_loss, val_acc, confusion, threshold_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
            epoch=epoch,
            total_epochs=args.epochs,
            log_interval=args.log_interval,
            use_amp=use_amp,
            eval_tta_flips=args.eval_tta_flips,
            full_metrics=args.full_metrics,
            progress_path=progress_path,
        )
        scheduler.step()

        elapsed = time.time() - train_start
        record = build_epoch_record(
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            confusion=confusion,
            threshold_metrics=threshold_metrics,
            learning_rates=scheduler.get_last_lr(),
            elapsed_seconds=elapsed,
            target_acc=args.target_acc,
            full_metrics=args.full_metrics,
        )
        history.append(record)
        write_history_files(history_path, metrics_jsonl_path, history)

        improved_val_acc = val_acc > best_val_acc
        balanced_acc = float(record.get("balanced_acc", val_acc))
        macro_f1 = record.get("macro_f1")
        improved_balanced_acc = args.full_metrics and balanced_acc > best_balanced_acc
        improved_threshold_balanced_acc = (
            args.full_metrics
            and threshold_metrics is not None
            and threshold_metrics["balanced_accuracy"] > best_threshold_balanced_acc
        )
        if improved_val_acc:
            best_val_acc = val_acc
        if improved_balanced_acc:
            best_balanced_acc = balanced_acc
        if improved_threshold_balanced_acc:
            best_threshold_balanced_acc = float(threshold_metrics["balanced_accuracy"])

        checkpoint_payload = {
            "epoch": epoch,
            "model_name": args.model,
            "mamba_architecture": args.mamba_architecture if args.model == "mamba" else None,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if use_amp else None,
            "args": vars(args),
            "train_csv": str(args.train_csv.resolve()),
            "val_csv": str(args.val_csv.resolve()),
            "num_classes": num_classes,
            "class_names": class_names,
            "image_size": image_size,
            "eval_resize_size": eval_resize_size,
            "history": history,
            "best_val_acc": best_val_acc,
            "best_balanced_acc": best_balanced_acc,
            "best_threshold_balanced_acc": best_threshold_balanced_acc,
            "target_acc": args.target_acc,
            "target_met": best_val_acc >= args.target_acc,
            "total_elapsed_seconds": elapsed,
        }
        save_checkpoint(output_dir / "last.pt", checkpoint_payload)

        if improved_val_acc:
            save_checkpoint(output_dir / "best_val_acc.pt", checkpoint_payload)
        if improved_balanced_acc:
            save_checkpoint(output_dir / "best_balanced_acc.pt", checkpoint_payload)
        if improved_threshold_balanced_acc:
            save_checkpoint(output_dir / "best_threshold_balanced_acc.pt", checkpoint_payload)

        metric_suffix = f" balanced_acc={balanced_acc:.4f} macro_f1={macro_f1:.4f}" if args.full_metrics else ""
        target_suffix = " target_met=yes" if val_acc >= args.target_acc else " target_met=no"
        log_message(
            f"[epoch] {epoch:02d}/{args.epochs:02d} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            f"{metric_suffix}{target_suffix}"
        )

    log_message(f"done output_dir={output_dir}")


if __name__ == "__main__":
    main()
