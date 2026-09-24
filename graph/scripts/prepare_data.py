# Copyright 2024 The Multi-GNN Authors. Apache-2.0 License.
# Adapted for AML Graph Learning V2 baseline pipeline.

import argparse
import itertools
from pathlib import Path
import numpy as np
import pandas as pd


def read_data(csv_path: str | Path) -> pd.DataFrame:
    """Read raw Kaggle CSV, assert expected header, and rename Account columns."""
    df = pd.read_csv(csv_path, dtype=str)
    # Pandas renames duplicate 'Account' column to 'Account.1'
    expected_headers = [
        ["Timestamp", "From Bank", "Account", "To Bank", "Account",
         "Amount Received", "Receiving Currency", "Amount Paid",
         "Payment Currency", "Payment Format", "Is Laundering"],
        ["Timestamp", "From Bank", "Account", "To Bank", "Account.1",
         "Amount Received", "Receiving Currency", "Amount Paid",
         "Payment Currency", "Payment Format", "Is Laundering"]
    ]
    assert list(df.columns) in expected_headers, f"Unexpected CSV header: {list(df.columns)}"

    # Rename positional columns 2 and 4 to From Account and To Account
    cols = list(df.columns)
    cols[2] = "From Account"
    cols[4] = "To Account"
    df.columns = cols
    return df


def parse_timestamps_and_edge_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Add EdgeID, parse datetime to seconds since min, hour, and day_of_week."""
    df["EdgeID"] = np.arange(len(df), dtype=np.int64)
    ts = pd.to_datetime(df["Timestamp"], format="%Y/%m/%d %H:%M")
    # Comment: Multi-GNN reference subtracts an offset during formatting and the min in get_data;
    # subtracting ts.min() here directly produces the identical relative seconds.
    df["Timestamp"] = (ts - ts.min()).dt.total_seconds().astype(np.int64)
    df["hour"] = ts.dt.hour.astype(int)
    df["day_of_week"] = ts.dt.dayofweek.astype(int)
    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized categorical encoding mirroring get_dict_val first-appearance order."""
    # Shared currency dictionary (receiving before payment per row); only the received side is kept
    currency_uniques = pd.unique(
        np.column_stack([df["Receiving Currency"], df["Payment Currency"]]).ravel()
    )
    df["Received Currency"] = pd.Categorical(
        df["Receiving Currency"], categories=currency_uniques
    ).codes.astype(np.int64)

    # Payment format encoding
    fmt_uniques = pd.unique(df["Payment Format"])
    df["Payment Format"] = pd.Categorical(
        df["Payment Format"], categories=fmt_uniques
    ).codes.astype(np.int64)

    # Node IDs: bank + "_" + account
    from_key = df["From Bank"].astype(str) + "_" + df["From Account"].astype(str)
    to_key = df["To Bank"].astype(str) + "_" + df["To Account"].astype(str)
    node_uniques = pd.unique(np.column_stack([from_key, to_key]).ravel())
    df["from_id"] = pd.Categorical(from_key, categories=node_uniques).codes.astype(np.int64)
    df["to_id"] = pd.Categorical(to_key, categories=node_uniques).codes.astype(np.int64)

    # Numeric cast for amounts and label
    df["Amount Received"] = df["Amount Received"].astype(np.float64)
    df["Is Laundering"] = df["Is Laundering"].astype(np.int64)
    return df


def compute_split(df: pd.DataFrame) -> pd.DataFrame:
    """Chronological day-boundary split search from Multi-GNN data_loading.py lines 42-75."""
    timestamps = df["Timestamp"].to_numpy()
    n_days = int(timestamps.max() / (3600 * 24) + 1)
    daily_trans = []
    for day in range(n_days):
        l = day * 24 * 3600
        r = (day + 1) * 24 * 3600
        count = int(np.sum((timestamps >= l) & (timestamps < r)))
        daily_trans.append(count)

    split_per = [0.6, 0.2, 0.2]
    d_ts = np.array(daily_trans)
    I = list(range(len(d_ts)))
    split_scores = dict()
    for i, j in itertools.combinations(I, 2):
        split_totals = [d_ts[:i].sum(), d_ts[i:j].sum(), d_ts[j:].sum()]
        split_totals_sum = np.sum(split_totals)
        split_props = [v / split_totals_sum for v in split_totals]
        split_error = [abs(v - t) / t for v, t in zip(split_props, split_per)]
        score = max(split_error)
        split_scores[(i, j)] = score

    i, j = min(split_scores, key=split_scores.get)
    train_end_sec = i * 24 * 3600
    val_end_sec = j * 24 * 3600

    split_arr = np.where(
        timestamps < train_end_sec,
        "train",
        np.where(timestamps < val_end_sec, "val", "test")
    )
    df["split"] = split_arr

    print(f"Split days: train=[0, {i}), val=[{i}, {j}), test=[{j}, {n_days})")
    print(f"Split counts: train={(df['split'] == 'train').sum()}, "
          f"val={(df['split'] == 'val').sum()}, test={(df['split'] == 'test').sum()}")
    return df


def compute_rate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 24h count and lifetime rate features via vectorized np.searchsorted."""
    t = df["Timestamp"].to_numpy(dtype=np.int64)
    from_ids = df["from_id"].to_numpy(dtype=np.int64)
    to_ids = df["to_id"].to_numpy(dtype=np.int64)
    num_nodes = int(max(from_ids.max(), to_ids.max())) + 1
    BIG = 10_000_000
    assert t.max() < BIG, f"Timestamp max {t.max()} exceeds BIG {BIG}"

    # First seen timestamp across all interactions for each node
    first_seen = np.full(num_nodes, fill_value=np.iinfo(np.int64).max, dtype=np.int64)
    stacked_ids = np.concatenate([from_ids, to_ids])
    stacked_t = np.concatenate([t, t])
    np.minimum.at(first_seen, stacked_ids, stacked_t)

    def history_counts(account_ids: np.ndarray, timestamps: np.ndarray):
        keys = account_ids * BIG + timestamps
        sorted_keys = np.sort(keys)
        hi = np.searchsorted(sorted_keys, keys, side="left")
        lo24 = np.searchsorted(sorted_keys, account_ids * BIG + np.maximum(timestamps - 86400, 0), side="left")
        lo_all = np.searchsorted(sorted_keys, account_ids * BIG, side="left")
        count_24h = hi - lo24
        count_lifetime = hi - lo_all
        acc_first = first_seen[account_ids]
        rate_lifetime = count_lifetime / np.maximum((timestamps - acc_first) / 86400.0, 1.0)
        return count_24h, rate_lifetime

    sender_24h, sender_rate = history_counts(from_ids, t)
    receiver_24h, receiver_rate = history_counts(to_ids, t)

    df["sender_out_count_24h"] = sender_24h.astype(np.int64)
    df["receiver_in_count_24h"] = receiver_24h.astype(np.int64)
    df["sender_out_rate_lifetime"] = sender_rate.astype(np.float64)
    df["receiver_in_rate_lifetime"] = receiver_rate.astype(np.float64)
    return df


def prepare_data(input_csv: str | Path, output_dir: str | Path):
    """Pipeline entry point to load, format, split, featurize, and write Parquet."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = output_dir / "transactions.parquet"

    print(f"Reading raw data from {input_csv}...")
    df = read_data(input_csv)

    print("Parsing timestamps and EdgeIDs...")
    df = parse_timestamps_and_edge_ids(df)

    print("Encoding categorical attributes...")
    df = encode_categorical(df)

    # Release raw string columns before sorting and computing history features.
    df = df.drop(columns=[
        "From Bank", "From Account", "To Bank", "To Account",
        "Receiving Currency", "Payment Currency", "Amount Paid",
    ])

    print("Sorting stably by Timestamp then EdgeID...")
    df = df.sort_values(["Timestamp", "EdgeID"], kind="stable").reset_index(drop=True)

    print("Computing chronological day-boundary split...")
    df = compute_split(df)

    print("Computing vectorized rate features...")
    df = compute_rate_features(df)

    ordered_columns = [
        "EdgeID", "from_id", "to_id", "Timestamp",
        "Amount Received", "Received Currency",
        "Payment Format", "Is Laundering", "hour", "day_of_week", "split",
        "sender_out_count_24h", "receiver_in_count_24h",
        "sender_out_rate_lifetime", "receiver_in_rate_lifetime"
    ]
    df = df[ordered_columns]

    row_count = len(df)
    pos_count = int((df["Is Laundering"] == 1).sum())
    print(f"Total rows: {row_count}, Total positive (laundering): {pos_count}")

    if "HI-Small" in str(input_csv):
        assert row_count == 5_078_345, f"Expected 5,078,345 rows, got {row_count}"
        assert pos_count == 5_177, f"Expected 5,177 positive rows, got {pos_count}"

    print(f"Writing prepared Parquet to {out_parquet}...")
    df.to_parquet(out_parquet, index=False)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare AML transaction dataset")
    parser.add_argument("--input", default="Data/HI-Small_Trans.csv", help="Path to raw CSV")
    parser.add_argument("--output-dir", default="Data/processed", help="Output directory")
    args = parser.parse_args()
    prepare_data(args.input, args.output_dir)
