import json, torch, numpy as np
from pathlib import Path

OUT = Path("/gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro/outputs")
SEEDS = [0, 7, 13, 42, 1337]
DST = OUT / "r1_reannot_mlsplit_b_s42"

all_p, all_y = [], []
all_pc, all_yc = [], []   # keep 2D [N,7] for per-class
per_seed = {}
class_names = None
for s in SEEDS:
    rd = OUT / f"r1_reannot_mlsplit_b_s{s}"
    blob = torch.load(rd / "predictions.pt", map_location="cpu", weights_only=False)
    class_names = blob.get("class_names")
    test = blob["test"]
    logits = test["class_logits"].float()
    labels = test["labels"].float()
    P = torch.sigmoid(logits).numpy()      # [n,7]
    Y = labels.numpy()                     # [n,7]
    all_pc.append(P); all_yc.append(Y)
    p = P.ravel(); y = Y.ravel()
    all_p.append(p); all_y.append(y)
    per_seed[s] = {"n_cells": int(labels.shape[0]),
                   "brier": float(np.mean((p - y) ** 2))}
    print(f"seed {s}: cells={labels.shape[0]} entries={p.size} brier={per_seed[s]['brier']:.4f}")

Pc = np.concatenate(all_pc); Yc = np.concatenate(all_yc)   # [Ntot,7]
p = np.concatenate(all_p); y = np.concatenate(all_y)
N = p.size
brier = float(np.mean((p - y) ** 2))
print("POOLED entries", N, "positives", int(y.sum()), "brier", brier)


def reliability(nbins, equal_mass=False):
    if equal_mass:
        edges = np.quantile(p, np.linspace(0, 1, nbins + 1))
        edges[0], edges[-1] = 0.0, 1.0
        edges = np.unique(edges)
    else:
        edges = np.linspace(0.0, 1.0, nbins + 1)
    bins = []
    ece = 0.0
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi) if i < len(edges) - 2 else (p >= lo) & (p <= hi)
        c = int(m.sum())
        if c == 0:
            bins.append({"lo": float(lo), "hi": float(hi), "count": 0,
                         "mean_pred": None, "obs_freq": None})
            continue
        conf = float(p[m].mean()); acc = float(y[m].mean())
        ece += (c / N) * abs(acc - conf)
        bins.append({"lo": float(lo), "hi": float(hi), "count": c,
                     "mean_pred": conf, "obs_freq": acc})
    return bins, float(ece)

def ece_1d(pp, yy, nbins=15):
    edges = np.linspace(0, 1, nbins + 1); n = pp.size; e = 0.0
    for i in range(nbins):
        lo, hi = edges[i], edges[i + 1]
        m = (pp >= lo) & (pp < hi) if i < nbins - 1 else (pp >= lo) & (pp <= hi)
        c = int(m.sum())
        if c:
            e += (c / n) * abs(float(yy[m].mean()) - float(pp[m].mean()))
    return e

b15, e15 = reliability(15)
b10, e10 = reliability(10)
bm10, em10 = reliability(10, equal_mass=True)
# per-class ECE and macro mean (15-bin equal width)
per_class_ece = {}
for k in range(Pc.shape[1]):
    nm = class_names[k] if class_names else str(k)
    per_class_ece[nm] = ece_1d(Pc[:, k], Yc[:, k], 15)
macro_ece = float(np.mean(list(per_class_ece.values())))
per_class_brier = {}
for k in range(Pc.shape[1]):
    nm = class_names[k] if class_names else str(k)
    per_class_brier[nm] = float(np.mean((Pc[:, k] - Yc[:, k]) ** 2))
print("per_class_ece", {k: round(v, 3) for k, v in per_class_ece.items()})
print("macro_ece", macro_ece)
print("per_class_brier", {k: round(v, 3) for k, v in per_class_brier.items()})
out = {"seeds": SEEDS, "pooled_n_entries": int(N), "pooled_n_positives": int(y.sum()),
       "pooled_brier": brier, "per_seed": per_seed,
       "ece_15bin_equalwidth": e15, "ece_10bin_equalwidth": e10,
       "ece_10bin_equalmass": em10,
       "macro_ece_15bin": macro_ece, "per_class_ece_15bin": per_class_ece,
       "per_class_brier": per_class_brier, "class_names": class_names,
       "bins_15_equalwidth": b15, "bins_10_equalwidth": b10,
       "bins_10_equalmass": bm10}
(DST / "reliability_pooled_5seed.json").write_text(json.dumps(out, indent=2))
print("ece: 15ew", e15, "10ew", e10, "10em", em10)
print("WROTE", DST / "reliability_pooled_5seed.json")
