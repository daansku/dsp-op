import argparse
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import DATA, SEEDS, best_threshold, metrics, save_run

BASIC = ["amount_received", "receiving_currency", "payment_format", "hour", "weekday", "sender_out_24h",
         "receiver_in_24h", "sender_out_per_day", "receiver_in_per_day"]
GRID = {"logreg": [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)],
        "rf": [{"max_depth": d, "min_samples_leaf": m} for d in (10, 20, None) for m in (1, 20)],
        "lightgbm": [{"num_leaves": n, "min_child_samples": m} for n in (15, 63) for m in (20, 200)]}
BEST = {"logreg": {"C": 0.1}, "rf": {"max_depth": 20, "min_samples_leaf": 1},
        "lightgbm": {"num_leaves": 63, "min_child_samples": 20}}


def make_model(name, params, seed):
    if name == "logreg":
        return make_pipeline(SimpleImputer(strategy="mean", add_indicator=True), StandardScaler(),
                             LogisticRegression(C=params["C"], max_iter=2000))
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, criterion="entropy", max_features="sqrt", n_jobs=-1,
                                      random_state=seed, **params)
    return lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.05, metric="average_precision", n_jobs=-1,
                              random_state=seed, verbose=-1, **params)


def sample(y, seed):
    # all laundering and 10% of the normal transactions, weights put the normal ones back to their real share
    rng = np.random.default_rng(seed)
    rows = np.flatnonzero((y == 1) | (rng.random(len(y)) < 0.10))
    weight = np.where(y[rows] == 1, 1.0, (y == 0).sum() / (y[rows] == 0).sum())
    return rows, weight


def fit_engineered(name, params, seed, X, y, sets):
    train, val = sets["train"], sets["val"]
    rows, _ = sample(y[train], seed)
    model = make_model(name, params, seed)
    if name == "lightgbm":
        stop, weight = sample(y[val], seed)
        model.fit(X[train][rows], y[train][rows], eval_X=X[val][stop], eval_y=y[val][stop],
                  eval_sample_weight=[weight], callbacks=[lgb.early_stopping(50, verbose=False)])
    else:
        model.fit(X[train][rows], y[train][rows])
    return model


def fit_basic(seed, X, y, sets):
    train, val = sets["train"], sets["val"]
    model = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05, num_leaves=31, metric="average_precision",
                               random_state=seed, verbose=-1)
    model.fit(X[train], y[train], eval_X=X[val], eval_y=y[val], callbacks=[lgb.early_stopping(50, verbose=False)])
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=list(GRID))
    parser.add_argument("--features", choices=["engineered", "basic"], default="engineered")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--tune", action="store_true", help="try every setting in GRID with the first seed")
    args = parser.parse_args()
    if args.features == "basic" and args.model != "lightgbm":
        parser.error("the basic columns are only used with lightgbm")

    df = pd.read_parquet(DATA / "transactions.parquet", columns=["edge_id", "is_laundering", "split"] + BASIC)
    y = df["is_laundering"].to_numpy()
    split = df["split"].to_numpy()
    sets = {s: slice(np.argmax(split == s), len(split) - np.argmax(split[::-1] == s)) for s in ("train", "val", "test")}
    if args.features == "basic":
        X = df[BASIC].copy()
        for c in ["receiving_currency", "payment_format"]:
            X[c] = X[c].cat.codes.astype("category")
    else:
        X = pd.read_parquet(DATA / "features.parquet").drop(columns="edge_id").to_numpy(np.float32)

    if args.tune:
        print(f"{'setting':40} {'val F1':>7} {'val AP':>7}")
        for params in GRID[args.model]:
            model = fit_engineered(args.model, params, args.seeds[0], X, y, sets)
            score = model.predict_proba(X[sets["val"]])[:, 1]
            m = metrics(y[sets["val"]], score, best_threshold(y[sets["val"]], score))
            print(f"{str(params):40} {m['f1']:7.3f} {m['ap']:7.3f}", flush=True)
        return

    for seed in args.seeds:
        start = time.time()
        if args.features == "basic":
            model = fit_basic(seed, X, y, sets)
        else:
            model = fit_engineered(args.model, BEST[args.model], seed, X, y, sets)
        val = model.predict_proba(X[sets["val"]])[:, 1]
        test = model.predict_proba(X[sets["test"]])[:, 1]
        info = {"params": BEST[args.model] if args.features == "engineered" else {}, "seconds": time.time() - start}
        if args.model == "lightgbm":
            info["trees"] = int(model.best_iteration_)
        save_run(f"{args.model}_{args.features}", seed, y[sets["val"]], val, df["edge_id"].to_numpy()[sets["test"]],
                 y[sets["test"]], test, **info)


if __name__ == "__main__":
    main()
