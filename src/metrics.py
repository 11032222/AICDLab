from __future__ import annotations


def compute_class_recalls(confusion: list[list[int]]) -> list[float]:
    recalls = []
    for idx, row in enumerate(confusion):
        total = sum(row)
        recalls.append((row[idx] / total) if total > 0 else 0.0)
    return recalls


def compute_class_precisions(confusion: list[list[int]]) -> list[float]:
    precisions = []
    for idx in range(len(confusion)):
        total = sum(row[idx] for row in confusion)
        precisions.append((confusion[idx][idx] / total) if total > 0 else 0.0)
    return precisions


def compute_class_f1s(precisions: list[float], recalls: list[float]) -> list[float]:
    f1s = []
    for precision, recall in zip(precisions, recalls, strict=True):
        denominator = precision + recall
        f1s.append((2 * precision * recall / denominator) if denominator > 0 else 0.0)
    return f1s


def compute_balanced_accuracy(confusion: list[list[int]]) -> float:
    recalls = compute_class_recalls(confusion)
    return sum(recalls) / len(recalls) if recalls else 0.0


def compute_macro_f1(confusion: list[list[int]]) -> float:
    precisions = compute_class_precisions(confusion)
    recalls = compute_class_recalls(confusion)
    f1s = compute_class_f1s(precisions, recalls)
    return sum(f1s) / len(f1s) if f1s else 0.0


def confusion_from_predictions(labels: list[int], predictions: list[int], num_classes: int) -> list[list[int]]:
    confusion = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for truth, pred in zip(labels, predictions):
        confusion[truth][pred] += 1
    return confusion


def find_best_binary_threshold(labels: list[int], positive_probs: list[float]) -> dict[str, object]:
    unique_probs = sorted(set(float(prob) for prob in positive_probs))
    candidate_thresholds = [0.0, *unique_probs, 1.0]
    best_record: dict[str, object] | None = None

    for threshold in candidate_thresholds:
        predictions = [1 if prob >= threshold else 0 for prob in positive_probs]
        confusion = confusion_from_predictions(labels, predictions, num_classes=2)
        accuracy = sum(int(truth == pred) for truth, pred in zip(labels, predictions)) / len(labels)
        balanced_accuracy = compute_balanced_accuracy(confusion)
        class_recalls = compute_class_recalls(confusion)
        class_precisions = compute_class_precisions(confusion)
        class_f1s = compute_class_f1s(class_precisions, class_recalls)
        record = {
            "threshold": threshold,
            "val_acc": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "class_recalls": class_recalls,
            "class_precisions": class_precisions,
            "class_f1s": class_f1s,
            "macro_f1": sum(class_f1s) / len(class_f1s),
            "confusion_matrix": confusion,
        }

        if best_record is None:
            best_record = record
            continue
        if balanced_accuracy > best_record["balanced_accuracy"]:
            best_record = record
        elif balanced_accuracy == best_record["balanced_accuracy"] and accuracy > best_record["val_acc"]:
            best_record = record

    if best_record is None:
        raise ValueError("Threshold search requires at least one validation sample.")
    return best_record
