import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from common import DATA, RESULTS, SCORES

RUNS = ["lightgbm_basic", "gin_basic", "pna_basic",
        "logreg_engineered", "rf_engineered", "lightgbm_engineered", "gin_engineered", "pna_engineered"]
NAMES = {"logreg": "Logistic regression", "rf": "Random forest", "lightgbm": "LightGBM", "gin": "GIN", "pna": "PNA"}
COLORS = {"logreg": "#8a8a85", "rf": "#1baf7a", "lightgbm": "#eda100", "gin": "#eb6834", "pna": "#2a78d6"}
TOP = 1000
BUDGETS = np.unique(np.round(np.logspace(2, 4, 31)).astype(int))
DAY = 24 * 3600


def label(run):
    model, features = run.split("_")
    return f"{NAMES[model]} ({features})"


def schemes():
    """Scheme and attempt of every laundering transaction listed in the patterns file, matched by its exact line."""
    wanted, attempt, scheme = {}, -1, None
    with open(DATA / "HI-Small_Patterns.txt") as f:
        for line in f:
            line = line.strip()
            if line.startswith("BEGIN LAUNDERING ATTEMPT"):
                attempt += 1
                scheme = line.split("-", 1)[1].split(":")[0].strip()
            elif line.startswith("END LAUNDERING ATTEMPT"):
                scheme = None
            elif scheme and line.count(",") == 10:
                wanted.setdefault(line, []).append((attempt, scheme))
    rows = []
    with open(DATA / "HI-Small_Trans.csv") as f:
        next(f)
        for edge_id, line in enumerate(f):
            line = line.strip()
            if wanted.get(line):
                rows.append((edge_id, *wanted[line].pop(0)))
    return pd.DataFrame(rows, columns=["edge_id", "attempt", "scheme"])


def tail_start(day):
    # the last days only hold the end of laundering chains (under 1% of a normal day's transactions)
    counts = np.bincount(day)
    small = counts < 0.01 * np.median(counts[counts > 0])
    start = len(counts)
    while small[start - 1]:
        start -= 1
    return start


def evaluate(s, threshold, listed):
    y, score, edge_id = s["is_laundering"].to_numpy(), s["score"].to_numpy(), s["edge_id"].to_numpy()
    alert = score >= threshold
    tp = (alert & (y == 1)).sum()
    order = np.lexsort((edge_id, -score))
    found = np.cumsum(y[order])
    top = listed["edge_id"].isin(edge_id[order[:TOP]])
    return ({"f1": 2 * tp / (alert.sum() + y.sum()), "precision": tp / max(alert.sum(), 1), "recall": tp / y.sum(),
             "f1_at_0.5": 2 * (score[y == 1] >= 0.5).sum() / ((score >= 0.5).sum() + y.sum()),
             "ap": average_precision_score(y, score), f"recall_top_{TOP}": found[TOP - 1] / y.sum()},
            found[BUDGETS - 1] / y.sum(),
            listed.assign(found=top).groupby("scheme")["found"].mean())


def main():
    df = pd.read_parquet(DATA / "transactions.parquet", columns=["edge_id", "time", "is_laundering", "split"])
    day = df["time"].to_numpy() // DAY
    test = df[df["split"] == "test"].assign(normal=day[df["split"] == "test"] < tail_start(day))
    listed = schemes()
    views = {"normal": test[test["normal"]], "all": test}
    print(f"Test days: {len(test):,} transactions, {test['is_laundering'].sum():,} laundering. "
          f"Normal days only: {len(views['normal']):,} and {views['normal']['is_laundering'].sum():,}")

    rows, curves, found = [], [], []
    for run in RUNS:
        for path in sorted((RESULTS / "runs").glob(f"{run}_seed*.json")):
            info = json.loads(path.read_text())
            scores = pd.read_parquet(SCORES / f"{path.stem}.parquet")
            for view, rows_in_view in views.items():
                s = rows_in_view.merge(scores, on="edge_id")
                assert len(s) == len(rows_in_view)
                m, curve, by_scheme = evaluate(s, info["threshold"], listed[listed["edge_id"].isin(s["edge_id"])])
                rows.append({"run": run, "seed": info["seed"], "view": view, **m})
                if view == "normal":
                    curves.append(pd.DataFrame({"run": run, "seed": info["seed"], "budget": BUDGETS, "found": curve}))
                    found.append(by_scheme.reset_index().assign(run=run))
    per_run = pd.DataFrame(rows)
    summary = per_run.drop(columns="seed").groupby(["run", "view"], sort=False).agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary.insert(0, "seeds", per_run.groupby(["run", "view"], sort=False).size())
    lines = ["| Model | Features | F1, normal test days | F1, all test days | F1 at 0.5, all test days | "
             f"AP, normal test days | Laundering in top {TOP:,} |", "|---|---|---|---|---|---|---|"]
    for run in RUNS:
        n, a = summary.loc[(run, "normal")], summary.loc[(run, "all")]
        model, features = run.split("_")
        lines.append(f"| {NAMES[model]} | {features} | " + " | ".join(
            f"{100 * r[f'{k}_mean']:.1f} ± {100 * r[f'{k}_std']:.1f}"
            for r, k in [(n, "f1"), (a, "f1"), (a, "f1_at_0.5"), (n, "ap"), (n, f"recall_top_{TOP}")]) + " |")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))

    normal = views["normal"]
    n_pos = normal["is_laundering"].sum()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(BUDGETS, np.minimum(BUDGETS, n_pos) / n_pos, color="black", lw=1, ls=":", label="perfect ranking")
    for run, c in pd.concat(curves).groupby("run", sort=False):
        mean = c.groupby("budget")["found"].mean()
        model, features = run.split("_")
        ax.plot(mean.index, mean, color=COLORS[model], lw=2, ls="--" if features == "basic" else "-", label=label(run))
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Transactions reviewed, highest score first")
    ax.set_ylabel("Share of the laundering found")
    ax.set_title(f"Normal test days: {n_pos:,} laundering among {len(normal):,} transactions (mean of 5 seeds)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(RESULTS / "capture_curve.png", dpi=150)

    size = listed[listed["edge_id"].isin(normal["edge_id"])]["scheme"].value_counts()
    table = pd.concat(found).pivot_table(index="run", columns="scheme", values="found").loc[RUNS, size.index]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.imshow(table, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            v = table.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white" if v > 0.5 else "black")
    ax.set_xticks(range(table.shape[1]), [f"{s.title()}\n({size[s]})" for s in table.columns], fontsize=8)
    ax.set_yticks(range(table.shape[0]), [label(r) for r in table.index], fontsize=9)
    ax.set_title(f"Share of each scheme's laundering found in the top {TOP:,} (normal test days)")
    fig.tight_layout()
    fig.savefig(RESULTS / "scheme_recall.png", dpi=150)
    print(f"\nWrote summary.md, capture_curve.png and scheme_recall.png to {RESULTS}")


if __name__ == "__main__":
    main()
