import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import BatchNorm, GINEConv, Linear, PNAConv
from torch_geometric.utils import degree

from common import DATA, MODELS, SEEDS, save_run

# Settings tuned for HI-Small in the Multi-GNN repository (Egressy et al.)
SETTINGS = {
    "gin": {"lr": 0.006213266113989207, "hidden": 66, "final_dropout": 0.10527690625126304,
            "class_weights": [1.0000182882773443, 6.275014431494497]},
    "pna": {"lr": 0.0006116418195373612, "hidden": 20, "final_dropout": 0.28812979737686323,
            "class_weights": [1.0003967674742307, 7.077633468006714]},
}
NEIGHBOURS = [100, 100]
BATCH = 8192
WINDOWS = [3600, 24 * 3600, 72 * 3600]
EVAL_SEED = 42


class GNN(nn.Module):
    def __init__(self, kind, edge_dim, hidden, final_dropout, deg):
        super().__init__()
        self.node_emb = nn.Linear(1, hidden)
        self.edge_emb = nn.Linear(edge_dim, hidden)
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(2):
            if kind == "gin":
                mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
                self.convs.append(GINEConv(mlp, edge_dim=hidden))
            else:
                self.convs.append(PNAConv(hidden, hidden, aggregators=["mean", "min", "max", "std"],
                                          scalers=["identity", "amplification", "attenuation"], deg=deg,
                                          edge_dim=hidden, towers=5, pre_layers=1, post_layers=1, divide_input=False))
            self.batch_norms.append(BatchNorm(hidden))
        self.mlp = nn.Sequential(Linear(3 * hidden, 50), nn.ReLU(), nn.Dropout(final_dropout), Linear(50, 25),
                                 nn.ReLU(), nn.Dropout(final_dropout), Linear(25, 2))

    def forward(self, x, edge_index, edge_attr):
        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)
        for conv, norm in zip(self.convs, self.batch_norms):
            x = (x + F.relu(norm(conv(x, edge_index, edge_attr)))) / 2
        # a transaction is classified from its sender, its receiver and its own features
        x = x[edge_index.T].reshape(-1, 2 * x.shape[1]).relu()
        return self.mlp(torch.cat([x, edge_attr], 1))


def z_norm(a):
    std = a.std(0)
    return (a - a.mean(0)) / torch.where(std == 0, 1.0, std)


def engineered_attr(n_train, t):
    X = pd.read_parquet(DATA / "features.parquet").drop(columns="edge_id").to_numpy(np.float32, copy=True)
    mean = np.nanmean(X[:n_train], 0, dtype=np.float64)
    std = np.nanstd(X[:n_train], 0, dtype=np.float64)
    X -= mean.astype(np.float32)
    X /= np.where(std > 0, std, 1).astype(np.float32)
    X[np.isnan(X)] = 0
    # one column per look-back window that starts before the data, instead of the missing values
    missing = [(t < w).astype(np.float32) for w in WINDOWS]
    return torch.from_numpy(np.column_stack([X] + missing))


def make_graphs(features):
    df = pd.read_parquet(DATA / "transactions.parquet")
    split = df["split"].to_numpy()
    n_train, n_val = (split == "train").sum(), (split == "val").sum()
    ends = {"train": n_train, "val": n_train + n_val, "test": len(df)}
    seeds = {"train": torch.arange(n_train), "val": torch.arange(n_train, n_train + n_val),
             "test": torch.arange(n_train + n_val, len(df))}
    edge_index = torch.from_numpy(np.vstack([df["sender"], df["receiver"]]))
    y = torch.tensor(df["is_laundering"].to_numpy())
    # every account gets the same input; Multi-GNN normalises a column of ones, which gives zeros
    x = torch.zeros(int(edge_index.max()) + 1, 1)
    if features == "basic":
        # normalised over each graph, as in Multi-GNN
        columns = np.column_stack([df["time"], df["amount_received"], df["receiving_currency"].cat.codes])
        columns = torch.from_numpy(columns.astype(np.float32))
        payment_format = torch.from_numpy(df["payment_format"].cat.codes.to_numpy(np.float32))[:, None]
        attr = {s: torch.cat([z_norm(columns[:end]), z_norm(payment_format[:end])], 1) for s, end in ends.items()}
    else:
        full = engineered_attr(n_train, df["time"].to_numpy())
        attr = {s: full[:end] for s, end in ends.items()}
    graphs = {s: Data(x=x, edge_index=edge_index[:, :end].contiguous(), edge_attr=attr[s], y=y[:end])
              for s, end in ends.items()}
    return graphs, seeds, df["edge_id"].to_numpy()


def make_loader(graph, seeds, shuffle):
    if shuffle:
        return LinkNeighborLoader(graph, num_neighbors=NEIGHBOURS, batch_size=BATCH, shuffle=True)
    return LinkNeighborLoader(graph, num_neighbors=NEIGHBOURS, edge_label_index=graph.edge_index[:, seeds],
                              edge_label=graph.y[seeds], batch_size=BATCH, shuffle=False)


def add_seed_edges(batch, seeds, graph):
    """The sampler can leave a seed transaction out of its own batch; put it back. Returns the seed mask."""
    seed_ids = seeds[batch.input_id]
    mask = torch.isin(batch.e_id, seed_ids)
    missing = seed_ids[~torch.isin(seed_ids, batch.e_id)]
    if len(missing):
        local = torch.full((graph.num_nodes,), -1)
        local[batch.n_id] = torch.arange(len(batch.n_id))
        batch.edge_index = torch.cat([batch.edge_index, local[graph.edge_index[:, missing]]], 1)
        batch.edge_attr = torch.cat([batch.edge_attr, graph.edge_attr[missing]])
        batch.y = torch.cat([batch.y, graph.y[missing]])
        batch.e_id = torch.cat([batch.e_id, missing])
        mask = torch.cat([mask, torch.ones(len(missing), dtype=torch.bool)])
    return mask


def train_epoch(model, loader, seeds, graph, optimizer, loss_fn, device):
    model.train()
    for batch in loader:
        optimizer.zero_grad()
        mask = add_seed_edges(batch, seeds, graph).to(device)
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.edge_attr)
        loss = loss_fn(out[mask], batch.y[mask])
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()


@torch.no_grad()
def predict(model, loader, seeds, graph, device):
    model.eval()
    ids, scores = [], []
    with torch.random.fork_rng():
        torch.manual_seed(EVAL_SEED)
        for batch in loader:
            mask = add_seed_edges(batch, seeds, graph)
            ids.append(batch.e_id[mask])
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr)
            scores.append(out.softmax(-1)[:, 1][mask.to(device)].cpu())
            torch.cuda.empty_cache()
    ids, scores = torch.cat(ids).numpy(), torch.cat(scores).numpy()
    order = np.argsort(ids)
    assert np.array_equal(ids[order], seeds.numpy())
    return scores[order]


def run(kind, features, seed, graphs, seeds, edge_id, train, epochs, patience):
    start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    loaders = {s: make_loader(graphs[s], seeds[s], shuffle=s == "train") for s in graphs}
    first = next(iter(loaders["train"]))
    deg = torch.bincount(degree(first.edge_index[1], dtype=torch.long))
    s = SETTINGS[kind]
    model = GNN(kind, graphs["train"].edge_attr.shape[1], s["hidden"], s["final_dropout"], deg).to(device)
    path = MODELS / f"{kind}_{features}_seed{seed}.pt"
    info = {"device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"}

    if path.exists() and not train:
        print(f"Loading {path.name}")
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        info["weights"] = "loaded"
    else:
        MODELS.mkdir(exist_ok=True)
        partial = path.with_suffix(".partial.pt")
        optimizer = torch.optim.Adam(model.parameters(), lr=s["lr"])
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(s["class_weights"], device=device))
        val_y = graphs["val"].y[seeds["val"]].numpy()
        best, best_epoch, stale = -1.0, 0, 0
        for epoch in range(epochs):
            t0 = time.time()
            train_epoch(model, loaders["train"], seeds["train"], graphs["train"], optimizer, loss_fn, device)
            val = predict(model, loaders["val"], seeds["val"], graphs["val"], device)
            f1 = f1_score(val_y, val > 0.5, zero_division=0)
            if f1 > best:
                best, best_epoch, stale = f1, epoch, 0
                torch.save(model.state_dict(), partial)
            elif best > 0:
                # the first epochs often flag nothing at all; they do not count towards the patience
                stale += 1
            print(f"epoch {epoch}: validation F1 at 0.5 = {f1:.3f} ({time.time() - t0:.0f}s)", flush=True)
            if stale >= patience:
                break
        partial.replace(path)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        info |= {"weights": "trained", "best_epoch": best_epoch, "epochs": epoch + 1}

    val = predict(model, loaders["val"], seeds["val"], graphs["val"], device)
    test = predict(model, loaders["test"], seeds["test"], graphs["test"], device)
    info["seconds"] = time.time() - start
    save_run(f"{kind}_{features}", seed, graphs["val"].y[seeds["val"]].numpy(), val, edge_id[seeds["test"].numpy()],
             graphs["test"].y[seeds["test"]].numpy(), test, **info)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=list(SETTINGS))
    parser.add_argument("--features", choices=["basic", "engineered"], default="basic")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--train", action="store_true", help="train even if models/ already has the weights")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()
    graphs, seeds, edge_id = make_graphs(args.features)
    for seed in args.seeds:
        run(args.model, args.features, seed, graphs, seeds, edge_id, args.train, args.epochs, args.patience)


if __name__ == "__main__":
    main()
