from dataclasses import dataclass
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


@dataclass
class ClusteringScores:
    acc: float
    ari: float
    nmi: float

    def __str__(self) -> str:
        return f"ACC={self.acc:.4f}  ARI={self.ari:.4f}  NMI={self.nmi:.4f}"


def _hungarian_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return sum(w[r, c] for r, c in zip(row_ind, col_ind)) / y_pred.size


def labels_to_ids(labels: list[str]) -> np.ndarray:
    unique = list(set(labels))
    label_map = {l: i for i, l in enumerate(unique)}
    return np.array([label_map[l] for l in labels])


def compute_scores(y_true_labels: list[str], y_pred_labels: list[str]) -> ClusteringScores:
    y_true = labels_to_ids(y_true_labels)
    y_pred = labels_to_ids(y_pred_labels)
    return ClusteringScores(
        acc=_hungarian_accuracy(y_true, y_pred),
        ari=adjusted_rand_score(y_true, y_pred),
        nmi=normalized_mutual_info_score(y_true, y_pred),
    )
