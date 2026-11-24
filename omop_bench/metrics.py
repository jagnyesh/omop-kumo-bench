import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def cluster_boot(y, preds, groups, n=1000, seed=0):
    """Patient-cluster bootstrap. preds: dict name -> scores. Returns 95% CIs for each AUROC/AUPRC and pairwise diffs vs the first."""
    rng = np.random.default_rng(seed); ug = np.unique(groups); idx = {g: np.where(groups == g)[0] for g in ug}
    names = list(preds); B = {}
    for _ in range(n):
        s = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])
        if y[s].min() == y[s].max():
            continue
        vals = {}
        for k in names:
            vals[f"{k}_auroc"] = roc_auc_score(y[s], preds[k][s]); vals[f"{k}_auprc"] = average_precision_score(y[s], preds[k][s])
        for k in names[1:]:
            vals[f"{k}_minus_{names[0]}_auroc"] = vals[f"{k}_auroc"] - vals[f"{names[0]}_auroc"]
            vals[f"{k}_minus_{names[0]}_auprc"] = vals[f"{k}_auprc"] - vals[f"{names[0]}_auprc"]
        for kk, v in vals.items():
            B.setdefault(kk, []).append(v)
    return {kk: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for kk, v in B.items()}
