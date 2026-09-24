import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_score, recall_score, f1_score


def load_split(parquet_path: str | Path, split: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Load split subset from Parquet, guaranteeing EdgeID and Is Laundering are included."""
    read_cols = None
    if columns is not None:
        read_cols = list(set(columns) | {"EdgeID", "Is Laundering", "split"})
    df = pd.read_parquet(parquet_path, columns=read_cols, filters=[("split", "==", split)])
    if columns is not None and "split" not in columns:
        df = df.drop(columns=["split"])
    return df


def choose_threshold(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    """Select score threshold maximizing F1; on ties select the highest threshold."""
    precision, recall, thresholds = precision_recall_curve(y, score)
    p = precision[:-1]
    r = recall[:-1]
    denom = p + r
    f1 = np.zeros_like(denom, dtype=np.float64)
    valid = denom > 0
    f1[valid] = 2.0 * p[valid] * r[valid] / denom[valid]
    if len(thresholds) == 0:
        return 0.5, 0.0
    max_f1 = np.max(f1)
    # thresholds is in increasing order, so last index gives highest threshold
    tie_indices = np.where(f1 == max_f1)[0]
    best_idx = tie_indices[-1]
    return float(thresholds[best_idx]), float(f1[best_idx])


def evaluate(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float | int]:
    """Evaluate predictions at specified threshold."""
    preds = (score >= threshold).astype(int)
    ap = float(average_precision_score(y, score))
    prec = float(precision_score(y, preds, zero_division=0))
    rec = float(recall_score(y, preds, zero_division=0))
    f1 = float(f1_score(y, preds, zero_division=0))
    alerts = int(preds.sum())
    return {
        "average_precision": ap,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "alerts": alerts,
    }


def sweep_curve(y: np.ndarray, score: np.ndarray) -> pd.DataFrame:
    """Generate threshold, alerts, precision, recall, f1 operating curve."""
    precision, recall, thresholds = precision_recall_curve(y, score)
    p = precision[:-1]
    r = recall[:-1]
    denom = p + r
    f1 = np.zeros_like(denom, dtype=np.float64)
    valid = denom > 0
    f1[valid] = 2.0 * p[valid] * r[valid] / denom[valid]
    sorted_scores = np.sort(score)
    alerts = len(sorted_scores) - np.searchsorted(sorted_scores, thresholds, side="left")
    return pd.DataFrame({
        "threshold": thresholds,
        "alerts": alerts,
        "precision": p,
        "recall": r,
        "f1": f1,
    })


def save_run(
    out_dir: str | Path,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    metrics: dict,
    parquet_path: str | Path | None = None,
):
    """Save score parquets, operating curves, and metrics with strict integrity checks."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, df in [("validation", val_df), ("test", test_df)]:
        assert set(df.columns) >= {"EdgeID", "label", "score"}, f"{name}_df missing required columns"
        assert df["EdgeID"].nunique() == len(df), f"{name}_df EdgeIDs are not unique"
        scores = df["score"].to_numpy()
        assert np.all(np.isfinite(scores)), f"{name}_df contains non-finite scores"
        assert (scores >= 0.0).all() and (scores <= 1.0).all(), f"{name}_df scores out of [0, 1]"

    if parquet_path is not None:
        for split_name, df in [("val", val_df), ("test", test_df)]:
            expected = load_split(parquet_path, split_name, ["EdgeID", "Is Laundering"])
            assert len(df) == len(expected), f"{split_name} row count mismatch: {len(df)} vs {len(expected)}"
            # Verify exact ID coverage
            df_sorted = df.sort_values("EdgeID").reset_index(drop=True)
            exp_sorted = expected.sort_values("EdgeID").reset_index(drop=True)
            assert np.array_equal(df_sorted["EdgeID"].values, exp_sorted["EdgeID"].values), \
                f"{split_name} EdgeIDs do not exactly cover expected split"
            assert np.array_equal(df_sorted["label"].values, exp_sorted["Is Laundering"].values), \
                f"{split_name} labels do not agree with source Parquet"

    # Save scores
    val_df[["EdgeID", "label", "score"]].to_parquet(out_dir / "validation_scores.parquet", index=False)
    test_df[["EdgeID", "label", "score"]].to_parquet(out_dir / "test_scores.parquet", index=False)

    # Save curves
    val_curve = sweep_curve(val_df["label"].to_numpy(), val_df["score"].to_numpy())
    val_curve.to_csv(out_dir / "validation_curve.csv", index=False)

    test_curve = sweep_curve(test_df["label"].to_numpy(), test_df["score"].to_numpy())
    test_curve.to_csv(out_dir / "test_curve.csv", index=False)

    # Save metrics JSON
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
