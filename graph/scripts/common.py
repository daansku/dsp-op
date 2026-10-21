import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
SCORES = ROOT / "scores"
SEEDS = [1, 2, 3, 4, 5]


def best_threshold(y, score):
    precision, recall, thresholds = precision_recall_curve(y, score)
    p, r = precision[:-1], recall[:-1]
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=p + r > 0)
    return float(thresholds[np.flatnonzero(f1 == f1.max())[-1]])


def metrics(y, score, threshold):
    alert = score >= threshold
    tp = int((alert & (y == 1)).sum())
    return {"f1": 2 * tp / (alert.sum() + y.sum()), "precision": tp / max(alert.sum(), 1), "recall": tp / y.sum(),
            "alerts": int(alert.sum()), "ap": float(average_precision_score(y, score))}


def save_run(name, seed, val_y, val_score, test_ids, test_y, test_score, **info):
    threshold = best_threshold(val_y, val_score)
    run = {"run": name, "seed": seed, "threshold": threshold,
           "validation": metrics(val_y, val_score, threshold),
           "test": metrics(test_y, test_score, threshold),
           "test_at_0.5": metrics(test_y, test_score, 0.5), **info}
    (RESULTS / "runs").mkdir(parents=True, exist_ok=True)
    SCORES.mkdir(exist_ok=True)
    with open(RESULTS / "runs" / f"{name}_seed{seed}.json", "w") as f:
        json.dump(run, f, indent=2, default=float)
    pd.DataFrame({"edge_id": test_ids, "score": test_score}).to_parquet(SCORES / f"{name}_seed{seed}.parquet")
    print(f"{name} seed {seed}: threshold {threshold:.3f}, validation F1 {run['validation']['f1']:.3f}, "
          f"test F1 {run['test']['f1']:.3f}, test AP {run['test']['ap']:.3f}", flush=True)
    return run
