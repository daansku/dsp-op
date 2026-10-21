import itertools

import numpy as np
import pandas as pd

from common import DATA

COLUMNS = ["time", "from_bank", "from_account", "to_bank", "to_account", "amount_received", "receiving_currency",
           "amount_paid", "payment_currency", "payment_format", "is_laundering"]
SPLIT = np.array([0.6, 0.2, 0.2])
DAY = 24 * 3600


def first_seen(a, b):
    # a0, b0, a1, b1, ...: numbers are given in order of first appearance, row by row
    both = pd.concat([a, b], ignore_index=True)
    return pd.Index(pd.unique(both.iloc[np.arange(len(both)).reshape(2, -1).T.ravel()]))


def split_days(day):
    # whole days, closest to 60/20/20 (same search as the Multi-GNN code)
    counts = np.bincount(day)

    def error(cut):
        i, j = cut
        parts = np.array([counts[:i].sum(), counts[i:j].sum(), counts[j:].sum()]) / counts.sum()
        return np.max(np.abs(parts - SPLIT) / SPLIT)

    i, j = min(itertools.combinations(range(len(counts)), 2), key=error)
    return np.where(day < i, "train", np.where(day < j, "val", "test"))


def activity(account, t, first_time):
    big = 10_000_000
    keys = account * big + t
    sorted_keys = np.sort(keys)
    now = np.searchsorted(sorted_keys, keys)
    day_ago = np.searchsorted(sorted_keys, account * big + np.maximum(t - DAY, 0))
    ever = np.searchsorted(sorted_keys, account * big)
    per_day = (now - ever) / np.maximum((t - first_time[account]) / DAY, 1.0)
    return now - day_ago, per_day


def main():
    df = pd.read_csv(DATA / "HI-Small_Trans.csv", names=COLUMNS, header=0, engine="pyarrow",
                     dtype={c: str for c in ["from_bank", "from_account", "to_bank", "to_account"]})
    df.insert(0, "edge_id", np.arange(len(df)))
    t = pd.to_datetime(df["time"], format="%Y/%m/%d %H:%M")
    df["time"] = (t - t.min()).dt.total_seconds().astype(np.int64)
    df["hour"] = t.dt.hour
    df["weekday"] = t.dt.dayofweek

    currencies = first_seen(df["receiving_currency"], df["payment_currency"])
    for c in ["receiving_currency", "payment_currency"]:
        df[c] = pd.Categorical(df[c], categories=currencies)
    df["payment_format"] = pd.Categorical(df["payment_format"], categories=pd.unique(df["payment_format"]))

    sender_key = df["from_bank"] + "_" + df["from_account"]
    receiver_key = df["to_bank"] + "_" + df["to_account"]
    keys = first_seen(sender_key, receiver_key)
    df["sender"] = keys.get_indexer(sender_key)
    df["receiver"] = keys.get_indexer(receiver_key)

    df = df.sort_values("time", kind="stable").reset_index(drop=True)
    df["split"] = split_days(df["time"].to_numpy() // DAY)

    t = df["time"].to_numpy()
    sender, receiver = df["sender"].to_numpy(), df["receiver"].to_numpy()
    first_time = np.full(len(keys), t.max())
    np.minimum.at(first_time, np.concatenate([sender, receiver]), np.concatenate([t, t]))
    df["sender_out_24h"], df["sender_out_per_day"] = activity(sender, t, first_time)
    df["receiver_in_24h"], df["receiver_in_per_day"] = activity(receiver, t, first_time)

    df = df.drop(columns=["from_bank", "from_account", "to_bank", "to_account"])
    df.to_parquet(DATA / "transactions.parquet", index=False)

    # The accounts file writes banks without the leading zeros
    bank, number = keys.str.split("_").str[0].str.lstrip("0"), keys.str.split("_").str[1]
    owners = pd.read_csv(DATA / "HI-Small_accounts.csv", dtype=str)
    owners.index = owners["Bank ID"].str.lstrip("0") + "_" + owners["Account Number"]
    owners = owners[~owners.index.duplicated()].reindex(bank + "_" + number)
    accounts = pd.DataFrame({"bank": bank, "account": number, "owner": owners["Entity ID"].to_numpy(),
                             "owner_type": owners["Entity Name"].str.replace(r"\s*#\d+$", "", regex=True).to_numpy()})
    accounts.to_parquet(DATA / "accounts.parquet", index=False)

    print(f"{len(df):,} transactions, {df['is_laundering'].sum():,} laundering, {len(accounts):,} accounts")
    print(df.groupby("split", sort=False).agg(first_day=("time", lambda x: x.min() // DAY), transactions=("time", "size"),
                                              laundering=("is_laundering", "sum")))


if __name__ == "__main__":
    main()
