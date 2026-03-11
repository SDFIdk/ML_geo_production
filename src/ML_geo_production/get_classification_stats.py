"""
Compute classification statistics (IoU, accuracy, F1, precision, recall, confusion matrix)
from single-channel uint8 label and prediction GeoTIFF rasters.
Uses only numpy and rasterio (no scikit-learn).
"""

import argparse
import numpy as np
from typing import Tuple

try:
    import rasterio
except ImportError:
    rasterio = None


def get_classification_stats(
    label_im: np.ndarray, pred_im: np.ndarray
) -> Tuple[float, float, float, float, float, np.ndarray]:
    """
    Compute classification metrics from flattened label and prediction arrays.

    Expects single-channel uint8 arrays (e.g. from GeoTIFF band 1). Arrays are
    flattened; shapes must match.

    Parameters
    ----------
    label_im : np.ndarray
        Ground truth labels, shape (H, W), dtype uint8 or integer.
    pred_im : np.ndarray
        Predicted labels, same shape and type.

    Returns
    -------
    IoU : float
        Mean Intersection over Union (macro over classes present in label).
    Pixel_Accuracy : float
        Fraction of pixels where label == pred.
    f1_score : float
        Macro-averaged F1 over classes.
    precision : float
        Macro-averaged precision.
    recall : float
        Macro-averaged recall.
    confusion_matrix : np.ndarray
        2D array [n_classes, n_classes], rows=true class, cols=predicted.
        Class indices are sorted by the union of unique labels in label_im and pred_im.
    """
    label_flat = np.asarray(label_im, dtype=np.int64).ravel()
    pred_flat = np.asarray(pred_im, dtype=np.int64).ravel()

    if label_flat.shape != pred_flat.shape:
        raise ValueError(
            f"label_im and pred_im must have the same size; got {label_im.size} vs {pred_im.size}"
        )

    classes = np.unique(np.concatenate([label_flat, pred_flat]))
    n_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    # Build confusion matrix: rows = true, cols = pred
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for lt, pt in zip(label_flat, pred_flat):
        i = class_to_idx.get(lt, -1)
        j = class_to_idx.get(pt, -1)
        if i >= 0 and j >= 0:
            cm[i, j] += 1

    # Pixel accuracy
    correct = (label_flat == pred_flat).sum()
    total = label_flat.size
    Pixel_Accuracy = float(correct) / total if total else 0.0

    # Per-class TP, FP, FN from CM
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    # Only average over classes that appear in the label (avoid div by zero for pred-only classes)
    eps = 1e-12
    precision_per_class = np.where(tp + fp > 0, tp / (tp + fp + eps), 0.0)
    recall_per_class = np.where(tp + fn > 0, tp / (tp + fn + eps), 0.0)
    f1_per_class = np.where(
        precision_per_class + recall_per_class > 0,
        2 * precision_per_class * recall_per_class / (precision_per_class + recall_per_class + eps),
        0.0,
    )
    iou_per_class = np.where(
        tp + fp + fn > 0, tp / (tp + fp + fn + eps), 0.0
    )

    # Macro average over classes present in labels (or all if empty)
    n_valid = max(1, n_classes)
    precision = float(np.mean(precision_per_class))
    recall = float(np.mean(recall_per_class))
    f1_score = float(np.mean(f1_per_class))
    IoU = float(np.mean(iou_per_class))

    return IoU, Pixel_Accuracy, f1_score, precision, recall, cm


def _format_stats_markdown(
    IoU: float,
    Pixel_Accuracy: float,
    f1_score: float,
    precision: float,
    recall: float,
    confusion_matrix: np.ndarray,
) -> str:
    """Format the statistics as a Markdown string for --output_MD."""
    lines = [
        "# Classification statistics",
        "",
        "## Overall metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| IoU (mean) | {IoU:.4f} |",
        f"| Pixel accuracy | {Pixel_Accuracy:.4f} |",
        f"| F1 (macro) | {f1_score:.4f} |",
        f"| Precision (macro) | {precision:.4f} |",
        f"| Recall (macro) | {recall:.4f} |",
        "",
        "## Confusion matrix",
        "",
        "Rows = true class, columns = predicted class.",
        "",
    ]
    # Format CM as markdown table
    n = confusion_matrix.shape[0]
    header = "| | " + " | ".join(f"pred_{i}" for i in range(n)) + " |"
    sep = "|" + "---|" * (n + 1)
    lines.append(header)
    lines.append(sep)
    for i in range(n):
        row = "| " + f"true_{i}" + " | " + " | ".join(str(confusion_matrix[i, j]) for j in range(n)) + " |"
        lines.append(row)
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute classification stats from label and prediction GeoTIFFs; write Markdown to --output_MD."
    )
    parser.add_argument("--label_im", required=True, help="Path to single-band uint8 label GeoTIFF.")
    parser.add_argument("--prediction_im", required=True, help="Path to single-band uint8 prediction GeoTIFF.")
    parser.add_argument("--output_MD", required=True, help="Path to output Markdown file with statistics.")
    args = parser.parse_args()

    if rasterio is None:
        raise RuntimeError("rasterio is required for reading GeoTIFFs. Install it in the Docker image.")

    with rasterio.open(args.label_im) as src:
        label_im = src.read(1)
    with rasterio.open(args.prediction_im) as src:
        pred_im = src.read(1)

    if label_im.shape != pred_im.shape:
        raise ValueError(
            f"Raster shapes differ: label {label_im.shape} vs prediction {pred_im.shape}. "
            "Align rasters (same extent and resolution) first."
        )

    IoU, Pixel_Accuracy, f1_score, precision, recall, confusion_matrix = get_classification_stats(
        label_im, pred_im
    )

    md = _format_stats_markdown(
        IoU, Pixel_Accuracy, f1_score, precision, recall, confusion_matrix
    )
    with open(args.output_MD, "w") as f:
        f.write(md)
