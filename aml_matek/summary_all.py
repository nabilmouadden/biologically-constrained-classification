"""Compact analysis over ALL completed output dirs."""
import json, math, sys
from pathlib import Path
import torch, numpy as np
sys.path.insert(0, "/gpfs/workdir/mouaddenn/thesis/aml_matek")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

CH3 = Path("/gpfs/workdir/mouaddenn/thesis/aml_matek")
cfg = json.loads((CH3 / "concept_config.json").read_text())
concepts = cfg["concepts"]
cidx = {c: i for i, c in enumerate(concepts)}
excl = [(cidx[p["concepts"][0]], cidx[p["concepts"][1]])
        for p in cfg["concept_constraint_matrix"]["mutually_exclusive_pairs"]]

all_dirs = sorted([d for d in (CH3 / "outputs").iterdir() if d.is_dir()])
print(f"Found {len(all_dirs)} output dirs.\n")

ROW_FMT = "{:<46s} {:>7s} {:>7s} {:>7s} {:>7s} {:>7s} {:>8s} {:>7s} {:>7s}"
print(ROW_FMT.format("config", "acc", "W-F1", "M-F1", "conF1", "AUROC", "viol", "probe", "cov05"))
print("-" * 120)

# Track per-concept F1 for later
per_concept_store = {}

for d in all_dirs:
    p = d / "predictions.pt"
    if not p.exists():
        print(ROW_FMT.format(d.name, "MISSING", "", "", "", "", "", "", ""))
        continue
    b = torch.load(p, map_location="cpu", weights_only=False)
    test = b["test"]
    y = test["labels"].numpy()
    y_pred = test["class_logits"].argmax(1).numpy()
    acc = accuracy_score(y, y_pred)
    w_f1 = f1_score(y, y_pred, average="weighted", zero_division=0)
    m_f1 = f1_score(y, y_pred, average="macro", zero_division=0)

    row = [f"{acc:.4f}", f"{w_f1:.4f}", f"{m_f1:.4f}"]
    if b.get("is_baseline"):
        row += ["-", "-", "-", "-", "-"]
    else:
        cp = test["concept_logits"].sigmoid().numpy()
        ct = (test["concept_targets"].numpy() >= 0.5).astype(int)
        f1s, aurocs = [], []
        per_cpt = {}
        for k in range(cp.shape[1]):
            sup = int(ct[:, k].sum())
            if 0 < sup < len(ct):
                f1_k = f1_score(ct[:, k], (cp[:, k] >= 0.5).astype(int), zero_division=0)
                auroc_k = roc_auc_score(ct[:, k], cp[:, k])
                f1s.append(f1_k); aurocs.append(auroc_k)
                per_cpt[concepts[k]] = (f1_k, sup)
        per_concept_store[d.name] = per_cpt
        c_f1 = float(np.mean(f1s))
        c_auroc = float(np.mean(aurocs))
        hard = cp >= 0.5
        n = hard.shape[0]
        viol_total = sum(int((hard[:, a] & hard[:, b]).sum()) for a, b in excl)
        viol = viol_total / (n * len(excl))
        cal = b["cal"]
        cal_cp = cal["concept_logits"].sigmoid().numpy()
        cal_y = cal["labels"].numpy()
        cal_ct = (cal["concept_targets"].numpy() >= 0.5).astype(int)
        clf = LogisticRegression(max_iter=2000, n_jobs=-1).fit(cal_cp, cal_y)
        probe = float(clf.score(cp, y))
        # conformal
        coverages = []
        alpha = 0.05
        for k in range(cp.shape[1]):
            s = np.where(cal_ct[:, k] == 1, 1 - cal_cp[:, k], cal_cp[:, k])
            m = len(s)
            q = np.quantile(s, min(1.0, math.ceil((1 - alpha) * (m + 1)) / m), method="higher")
            detected = cp[:, k] > 1 - q
            absent = cp[:, k] < q
            uncertain = ~(detected | absent)
            covered = ((ct[:, k] == 1) & detected) | ((ct[:, k] == 0) & absent) | uncertain
            coverages.append(float(covered.mean()))
        cov = float(np.mean(coverages))
        row += [f"{c_f1:.4f}", f"{c_auroc:.4f}", f"{viol:.5f}", f"{probe:.4f}", f"{cov:.4f}"]
    print(ROW_FMT.format(d.name, *row))

# Band-nucleus deep dive on the most relevant configs
print("\n=== BAND_NUCLEUS F1 across configs (support shown) ===")
for name, per in sorted(per_concept_store.items()):
    if "band_nucleus" in per:
        f1, sup = per["band_nucleus"]
        print(f"  {name:<50s}  F1={f1:.3f}  n_pos={sup}")
    else:
        print(f"  {name:<50s}  (not in support)")

# Rare concept summary for the pos_weight comparison
rare = ["band_nucleus", "azurophilic_granules", "basophilic_granules",
        "visible_nucleoli", "basophilic_cytoplasm"]
print("\n=== RARE-CONCEPT F1 — DinoBloom joint+const, default vs pos_weight ===")
for cfg_name in ["dinobloom_s_aml_matek_joint_const", "dinobloom_s_aml_matek_joint_const_posw"]:
    print(f"\n  {cfg_name}")
    per = per_concept_store.get(cfg_name, {})
    for c in rare:
        if c in per:
            f1, sup = per[c]
            print(f"    {c:<24s}  F1={f1:.3f}  n_pos={sup}")
