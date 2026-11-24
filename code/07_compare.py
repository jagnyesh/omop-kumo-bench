"""Score Kumo holdout predictions with the same metrics/bootstrap as LightGBM, plus paired AUROC difference."""
import sys, json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
kumo_holdout, lgbm_preds, out = sys.argv[1], sys.argv[2], sys.argv[3]
h = pd.read_parquet(kumo_holdout); print(h.columns.tolist(), h.shape)
ent = [c for c in h.columns if c.upper().startswith("ENTITY")][0]
ts = [c for c in h.columns if "TIME" in c.upper()][0]
prob = [c for c in h.columns if c.lower().startswith("true") and "prob" in c.lower()] or [c for c in h.columns if "PRED" in c.upper()]
prob = prob[0]
tgt = [c for c in h.columns if c.upper().startswith("TARGET") and "PRED" not in c.upper()][0]
k = h[[ent, ts, tgt, prob]].rename(columns={ent: "person_id", ts: "ts", tgt: "label_k", prob: "kumo"})
k["ts"] = pd.to_datetime(k["ts"]).dt.normalize()
k = k.groupby(["person_id", "ts"], as_index=False).agg(kumo=("kumo", "mean"), label_k=("label_k", "max"))
l = pd.read_parquet(lgbm_preds)
import duckdb
coh = duckdb.sql("select index_visit_id, discharge_date::timestamp ts, coalesce(ip_365,0) ip_365 from '/mnt/project-files/kumo-synthea/data/lgbm_features.parquet'").df()
l = l.merge(coh, on="index_visit_id")
m = l.merge(k, on=["person_id", "ts"], how="left")
print("matched", m.kumo.notna().sum(), "of", len(m), "label agreement", (m.label == m.label_k.astype(float)).mean())
m = m[m.kumo.notna()]
y, g = m.label.values, m.person_id.values
rng = np.random.default_rng(0); ug = np.unique(g); idx = {x: np.where(g == x)[0] for x in ug}
B = {"lgbm_auroc": [], "kumo_auroc": [], "lgbm_auprc": [], "kumo_auprc": [], "diff_auroc": [], "diff_auprc": []}
for _ in range(1000):
    s = np.concatenate([idx[x] for x in rng.choice(ug, len(ug))])
    a1, a2 = roc_auc_score(y[s], m.pred.values[s]), roc_auc_score(y[s], m.kumo.values[s])
    p1, p2 = average_precision_score(y[s], m.pred.values[s]), average_precision_score(y[s], m.kumo.values[s])
    for kk, v in zip(B, [a1, a2, p1, p2, a2 - a1, p2 - p1]): B[kk].append(v)
ci = {kk: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for kk, v in B.items()}
one = m.groupby("person_id").sample(1, random_state=0)
res = dict(n=int(len(m)), patients=int(len(ug)),
  lgbm_auroc=roc_auc_score(y, m.pred), kumo_auroc=roc_auc_score(y, m.kumo),
  lgbm_auprc=average_precision_score(y, m.pred), kumo_auprc=average_precision_score(y, m.kumo),
  kumo_minus_lgbm_auroc=roc_auc_score(y, m.kumo) - roc_auc_score(y, m.pred),
  kumo_minus_lgbm_auprc=average_precision_score(y, m.kumo) - average_precision_score(y, m.pred),
  ci95_patient_bootstrap=ci,
  one_stay_per_patient=dict(lgbm_auroc=roc_auc_score(one.label, one.pred), kumo_auroc=roc_auc_score(one.label, one.kumo),
                            lgbm_auprc=average_precision_score(one.label, one.pred), kumo_auprc=average_precision_score(one.label, one.kumo)))
sub = m[m.ip_365 == 0]
def sb(col, f):
    vals=[]; gg=sub.person_id.values; u=np.unique(gg); ix={x: np.where(gg==x)[0] for x in u}; r=np.random.default_rng(1)
    for _ in range(1000):
        s_=np.concatenate([ix[x] for x in r.choice(u,len(u))])
        if sub.label.values[s_].sum()==0: continue
        vals.append(f(sub.label.values[s_], sub[col].values[s_]))
    return [float(np.percentile(vals,2.5)), float(np.percentile(vals,97.5))]
res["no_inpatient_prior_365d_subset"] = dict(n=int(len(sub)), positives=int(sub.label.sum()),
    lgbm_auroc=roc_auc_score(sub.label, sub.pred), lgbm_auroc_ci=sb("pred", roc_auc_score),
    kumo_auroc=roc_auc_score(sub.label, sub.kumo), kumo_auroc_ci=sb("kumo", roc_auc_score),
    lgbm_auprc=average_precision_score(sub.label, sub.pred), kumo_auprc=average_precision_score(sub.label, sub.kumo))
json.dump(res, open(out, "w"), indent=2); print(json.dumps(res, indent=2))
