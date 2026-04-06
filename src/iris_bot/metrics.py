from __future__ import annotations


CLASSES = (-1, 0, 1)


def confusion_matrix(y_true: list[int], y_pred: list[int], classes: tuple[int, ...] = CLASSES) -> list[list[int]]:
    index = {label: position for position, label in enumerate(classes)}
    matrix = [[0 for _ in classes] for _ in classes]
    for actual, predicted in zip(y_true, y_pred, strict=False):
        matrix[index[actual]][index[predicted]] += 1
    return matrix


def class_balance(labels: list[int], classes: tuple[int, ...] = CLASSES) -> dict[str, float]:
    total = len(labels)
    if total == 0:
        return {str(label): 0.0 for label in classes}
    return {str(label): labels.count(label) / total for label in classes}


def _precision_recall_f1_for_class(y_true: list[int], y_pred: list[int], positive_label: int) -> tuple[float, float, float]:
    tp = sum(1 for actual, predicted in zip(y_true, y_pred, strict=False) if actual == positive_label and predicted == positive_label)
    fp = sum(1 for actual, predicted in zip(y_true, y_pred, strict=False) if actual != positive_label and predicted == positive_label)
    fn = sum(1 for actual, predicted in zip(y_true, y_pred, strict=False) if actual == positive_label and predicted != positive_label)
    precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
    recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
    f1 = 0.0 if precision + recall == 0.0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def classification_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, object]:
    total = len(y_true)
    accuracy = 0.0 if total == 0 else sum(1 for actual, predicted in zip(y_true, y_pred, strict=False) if actual == predicted) / total
    per_class = {}
    recalls: list[float] = []
    macro_f1_values: list[float] = []
    for label in CLASSES:
        precision, recall, f1 = _precision_recall_f1_for_class(y_true, y_pred, label)
        per_class[str(label)] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        recalls.append(recall)
        macro_f1_values.append(f1)

    macro_f1 = sum(macro_f1_values) / len(macro_f1_values)
    balanced_accuracy = sum(recalls) / len(recalls)

    # Directional precision: fraction of non-neutral predictions that are correct.
    # The economically meaningful accuracy — neutral predictions cost nothing (no trade),
    # only wrong directional calls lose money.
    dir_tp = sum(
        1
        for actual, predicted in zip(y_true, y_pred, strict=False)
        if predicted != 0 and actual == predicted
    )
    dir_total = sum(1 for predicted in y_pred if predicted != 0)
    directional_precision = dir_tp / dir_total if dir_total > 0 else 0.0

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "directional_precision": directional_precision,
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "class_balance": class_balance(y_true),
        "per_class": per_class,
    }
