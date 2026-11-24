"""Score Kumo's holdout predictions against LightGBM's on the same test events: paired patient-cluster bootstrap,
one event per patient, and the subset with no inpatient stay in the prior year."""
import json
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from .paths import task_dir, result_dir
from .metrics import cluster_boot

def scores(m, cols=("pred", "kumo")):
    return {f"{'lgbm' if c == 'pred' else c}_{k}": f(m.label, m[c]) for c in cols for k, f in (("auroc", roc_auc_score), ("auprc", average_precision_score))}

def run(cfg):
    name = cfg["name"]; out = result_dir(name)
    h = pd.read_parquet(out / "kumo_holdout.parquet")
    ent = [c for c in h.columns if c.upper().startswith("ENTITY")][0]
    ts = [c for c in h.columns if "TIME" in c.upper()][0]
    prob = ([c for c in h.columns if c.lower().startswith("true") and "prob" in c.lower()] or [c for c in h.columns if "PRED" in c.upper()])[0]
    tgt = [c for c in h.columns if c.upper().startswith("TARGET") and "PRED" not in c.upper()][0]
    k = h[[ent, ts, tgt, prob]].rename(columns={ent: "person_id", ts: "ts", tgt: "label_k", prob: "kumo"})
    k["ts"] = pd.to_datetime(k["ts"]).dt.normalize()
    k = k.groupby(["person_id", "ts"], as_index=False).agg(kumo=("kumo", "mean"), label_k=("label_k", "max"))
    l = pd.read_parquet(out / "lgbm_test_preds.parquet")
    l["ts"] = pd.to_datetime(l["anchor_date"]).dt.normalize()
    f = pd.read_parquet(task_dir(name) / "features.parquet", columns=["index_id", "ip_365"])
    m = l.merge(f, on="index_id").merge(k, on=["person_id", "ts"], how="left")
    matched = int(m.kumo.notna().sum())
    agree = float((m.label == m.label_k.astype(float)).mean())
    m = m[m.kumo.notna()]
    y, g = m.label.values, m.person_id.values
    res = dict(task=name, n=int(len(m)), patients=int(len(np.unique(g))), positives=int(y.sum()), matched=matched, label_agreement=agree, **scores(m))
    res["kumo_minus_lgbm_auroc"] = res["kumo_auroc"] - res["lgbm_auroc"]
    res["kumo_minus_lgbm_auprc"] = res["kumo_auprc"] - res["lgbm_auprc"]
    res["ci95_patient_bootstrap"] = cluster_boot(y, {"lgbm": m.pred.values, "kumo": m.kumo.values}, g)
    one = m.groupby("person_id").sample(1, random_state=0)
    res["one_event_per_patient"] = dict(n=int(len(one)), positives=int(one.label.sum()), **scores(one))
    sub = m[m.ip_365.fillna(0) == 0]
    if sub.label.nunique() == 2:
        res["no_inpatient_prior_365d"] = dict(n=int(len(sub)), positives=int(sub.label.sum()), **scores(sub),
            ci95_patient_bootstrap=cluster_boot(sub.label.values, {"lgbm": sub.pred.values, "kumo": sub.kumo.values}, sub.person_id.values, seed=1))
    json.dump(res, open(out / "comparison.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, dict)}, indent=1), flush=True)
    return res
