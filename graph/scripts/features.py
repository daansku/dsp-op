import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from common import DATA

DAY = 24 * 60
WINDOWS = {"1h": 60, "24h": 24 * 60, "72h": 72 * 60}
KEY = 10_000_000
CATEGORIES = ["payment_format", "payment_currency", "sender_type", "receiver_type"]
NOT_LOGGED = {"log_amount", "log_amount_sq", "is_round_100", "currency_mismatch", "self_transfer", "same_bank",
              "same_owner", "hour", "hour_sin", "hour_cos", "is_weekend", "sender_has_prior_out",
              "receiver_has_prior_in"}


def dollar_rates(df):
    paid, received = df["amount_paid"].to_numpy(), df["amount_received"].to_numpy()
    pay, recv = df["payment_currency"].astype(str).to_numpy(), df["receiving_currency"].astype(str).to_numpy()
    to_usd = (recv == "US Dollar") & (pay != "US Dollar")
    from_usd = (pay == "US Dollar") & (recv != "US Dollar")
    rates = pd.DataFrame({"currency": np.concatenate([pay[to_usd], recv[from_usd]]),
                          "rate": np.concatenate([received[to_usd] / paid[to_usd], paid[from_usd] / received[from_usd]])})
    rates = rates[np.isfinite(rates["rate"]) & (rates["rate"] > 0)].groupby("currency")["rate"].median()
    rates["US Dollar"] = 1.0
    return rates


def ratio(a, b, fill):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.divide(a, b, out=np.full(len(a), fill), where=b > 0)


def day_totals(account, partner, usd):
    g = pd.DataFrame({"account": account, "partner": partner, "usd": usd}).groupby("account")
    return pd.DataFrame({"count": g.size(), "usd": g["usd"].sum(), "partners": g["partner"].nunique()})


def look_up(table, account):
    found = table.reindex(account).fillna(0)
    return {c: found[c].to_numpy() for c in found.columns}


def events(account, minute, value=None):
    keys = account * KEY + minute
    order = np.argsort(keys, kind="stable")
    # running total per account, so a window sum is the difference of two running totals
    running = None if value is None else pd.Series(value[order]).groupby(account[order]).cumsum().to_numpy()
    return keys[order], running


def before(events, account, minute, window):
    """count and dollar sum of the account's transactions in [minute - window, minute)"""
    keys, running = events
    end = np.searchsorted(keys, account * KEY + minute)
    start = np.searchsorted(keys, account * KEY + np.maximum(minute - window, 0))
    if running is None:
        return end - start, None
    first = np.searchsorted(keys, account * KEY)
    upper = np.where(end > start, running[np.maximum(end - 1, 0)], 0.0)
    lower = np.where((end > start) & (start > first), running[np.maximum(start - 1, 0)], 0.0)
    return end - start, upper - lower


def since_last(events, account, minute, cap=WINDOWS["72h"]):
    keys, _ = events
    end = np.searchsorted(keys, account * KEY + minute)
    first = np.searchsorted(keys, account * KEY)
    gap = np.where(end > first, minute - keys[np.maximum(end - 1, 0)] % KEY, cap)
    return np.minimum(gap, cap)


def day_features(T, today, past):
    """Features of the transactions of one day. `past` is that day and the three days before it."""
    mine = slice(today.start - past.start, None)
    minute_p, sender_p, receiver_p, usd_p = (T[k][past] for k in ("minute", "sender", "receiver", "usd"))
    minute, sender, receiver, usd = minute_p[mine], sender_p[mine], receiver_p[mine], usd_p[mine]
    n = T["accounts"]
    pairs = pd.factorize(np.concatenate([sender_p * n + receiver_p, receiver * n + sender]))[0]
    pair_p, reverse_pair = pairs[:len(sender_p)], pairs[len(sender_p):]
    pair = pair_p[mine]

    F = {}
    F["log_amount"] = np.log1p(usd)
    F["log_amount_sq"] = F["log_amount"] ** 2
    F["is_round_100"] = T["round_100"][today]
    F["currency_mismatch"] = T["mismatch"][today]
    F["self_transfer"] = sender == receiver
    F["same_bank"] = T["bank"][sender] == T["bank"][receiver]
    owner = T["owner"]
    F["same_owner"] = (owner[sender] == owner[receiver]) & (owner[sender] >= 0) & (sender != receiver)
    hour = (minute % DAY) // 60
    F["hour"] = hour
    F["hour_sin"], F["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    F["is_weekend"] = T["weekday"][today] >= 5

    # same calendar day, the transaction itself included
    sent, received = day_totals(sender, receiver, usd), day_totals(receiver, sender, usd)
    s_out, s_in = look_up(sent, sender), look_up(received, sender)
    r_in, r_out = look_up(received, receiver), look_up(sent, receiver)
    F["sender_day_out_count"], F["sender_day_out_usd"] = s_out["count"], s_out["usd"]
    F["sender_day_distinct_receivers"] = s_out["partners"]
    F["sender_day_in_count"], F["sender_day_in_usd"] = s_in["count"], s_in["usd"]
    F["sender_amount_vs_day_avg"] = ratio(usd, s_out["usd"] / s_out["count"], 1.0)
    F["receiver_day_in_count"], F["receiver_day_in_usd"] = r_in["count"], r_in["usd"]
    F["receiver_day_distinct_senders"] = r_in["partners"]
    F["receiver_day_out_count"], F["receiver_day_out_usd"] = r_out["count"], r_out["usd"]
    pair_day = pd.Series(pair).value_counts()
    F["pair_day_count"] = pair_day.reindex(pair).to_numpy()
    F["reverse_pair_day_count"] = pair_day.reindex(reverse_pair).fillna(0).to_numpy()

    # look-back windows, strictly earlier minutes only
    L = {}
    sent_ev, received_ev = events(sender_p, minute_p, usd_p), events(receiver_p, minute_p, usd_p)
    for side, account in (("sender", sender), ("receiver", receiver)):
        for direction, ev in (("out", sent_ev), ("in", received_ev)):
            for w, window in WINDOWS.items():
                L[f"{side}_{direction}_count_{w}"], L[f"{side}_{direction}_usd_{w}"] = before(ev, account, minute, window)
    pair_ev = events(pair_p, minute_p)
    L["pair_count_72h"] = before(pair_ev, pair, minute, WINDOWS["72h"])[0]
    L["reverse_pair_count_72h"] = before(pair_ev, reverse_pair, minute, WINDOWS["72h"])[0]
    L["passthrough_ratio_24h"] = ratio(usd, L["sender_in_usd_24h"], 0.0)
    L["sender_amount_vs_avg_72h"] = ratio(usd, ratio(L["sender_out_usd_72h"], L["sender_out_count_72h"], 0.0), 1.0)
    L["sender_mins_since_last_out"] = since_last(sent_ev, sender, minute)
    L["receiver_mins_since_last_in"] = since_last(received_ev, receiver, minute)
    L["sender_has_prior_out"] = L["sender_out_count_72h"] > 0
    L["receiver_has_prior_in"] = L["receiver_in_count_72h"] > 0
    # a window that starts before the first day of data is incomplete: missing, not zero
    for name in L:
        window = WINDOWS.get(name.rsplit("_", 1)[-1], WINDOWS["72h"])
        L[name] = np.where(minute < window, np.nan, L[name])
    return F | L


def main():
    start = time.time()
    df = pd.read_parquet(DATA / "transactions.parquet")
    accounts = pd.read_parquet(DATA / "accounts.parquet")
    n = len(df)
    owner_type = accounts["owner_type"].fillna("Unknown").to_numpy()

    day = df["time"].to_numpy() // (DAY * 60)
    order = np.lexsort((df["edge_id"].to_numpy(), day))   # by day, then raw file order
    d = df.iloc[order]
    paid = d["amount_paid"].to_numpy()
    cents = np.round(paid * 100).astype(np.int64)
    T = {"minute": d["time"].to_numpy() // 60,
         "sender": d["sender"].to_numpy(), "receiver": d["receiver"].to_numpy(),
         "usd": paid * d["payment_currency"].astype(str).map(dollar_rates(df)).to_numpy(),
         "round_100": (cents > 0) & (cents % 10_000 == 0),
         "mismatch": (d["payment_currency"] != d["receiving_currency"]).to_numpy(),
         "weekday": d["weekday"].to_numpy(),
         "accounts": len(accounts),
         "bank": pd.factorize(accounts["bank"])[0],
         "owner": pd.factorize(accounts["owner"])[0]}
    del d

    columns = {}
    day_start = np.searchsorted(day[order], np.arange(day.max() + 2))
    for k in range(day.max() + 1):
        today = slice(day_start[k], day_start[k + 1])
        if today.start == today.stop:
            continue
        past = slice(day_start[max(k - 3, 0)], today.stop)
        for name, values in day_features(T, today, past).items():
            values = np.asarray(values, np.float32)
            if name not in NOT_LOGGED:
                values = np.log1p(values)
            columns.setdefault(name, np.empty(n, np.float32))[order[today]] = values
        print(f"day {k}: {today.stop - today.start:,} transactions", flush=True)

    category = {"payment_format": df["payment_format"].astype(str).to_numpy(),
                "payment_currency": df["payment_currency"].astype(str).to_numpy(),
                "sender_type": owner_type[df["sender"].to_numpy()],
                "receiver_type": owner_type[df["receiver"].to_numpy()]}
    for c in CATEGORIES:
        for value in sorted(set(category[c])):
            columns[f"{c}={value}"] = (category[c] == value).astype(np.float32)

    table = pa.table({"edge_id": df["edge_id"].to_numpy(), **columns})
    pq.write_table(table, DATA / "features.parquet", compression="zstd", row_group_size=500_000)
    print(f"{len(columns)} feature columns for {n:,} transactions in {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
