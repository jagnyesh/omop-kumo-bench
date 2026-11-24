"""Tuned LightGBM on hand-built features. Optuna on validation AUROC, then test metrics with patient-level bootstrap CIs."""
import sys, json, time, numpy as np, pandas as pd, lightgbm as lgb, optuna
from sklearn.metrics import roc_auc_score, average_precision_score
t0 = time.time()
df = pd.read_parquet(sys.argv[1]); out = sys.argv[2]
cat = ["gender", "race", "eth"]
for c in cat: df[c] = df[c].astype("category")
X = [c for c in df.columns if c not in ("index_visit_id", "person_id", "discharge_date", "label", "split", "a", "e") and not c.endswith(("_1","_2"))]
tr, va, te = (df[df.split == s] for s in (0, 1, 2))

def fit(params, rounds=2000):
    m = lgb.LGBMClassifier(n_estimators=rounds, verbose=-1, n_jobs=4, **params)
    m.fit(tr[X], tr.label, eval_set=[(va[X], va.label)], eval_metric="auc",
          callbacks=[lgb.early_stopping(100, verbose=False)])
    return m

def objective(trial):
    p = dict(learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
             num_leaves=trial.suggest_int("num_leaves", 8, 128, log=True),
             min_child_samples=trial.suggest_int("min_child_samples", 10, 300, log=True),
             subsample=trial.suggest_float("subsample", 0.5, 1.0), subsample_freq=1,
             colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
             reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True))
    m = fit(p)
    return roc_auc_score(va.label, m.predict_proba(va[X])[:, 1])

optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=7))
study.optimize(objective, n_trials=40)
best = dict(study.best_params, subsample_freq=1)
m = fit(best)
p = m.predict_proba(te[X])[:, 1]

def boot(y, p, groups, f, n=1000, seed=0):
    rng = np.random.default_rng(seed); ug = np.unique(groups); idx = {g: np.where(groups == g)[0] for g in ug}
    vals = []
    for _ in range(n):
        s = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])
        vals.append(f(y[s], p[s]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

y, g = te.label.values, te.person_id.values
# one random stay per test patient: guards against frequent-flyer patients dominating
one = te.assign(p=p).groupby("person_id").sample(1, random_state=0)
res = dict(model="LightGBM (tuned, hand-built features)", n_features=len(X), best_val_auroc=study.best_value, best_params=best,
           best_iteration=int(m.best_iteration_),
           test_auroc=roc_auc_score(y, p), test_auroc_ci=boot(y, p, g, roc_auc_score),
           test_auprc=average_precision_score(y, p), test_auprc_ci=boot(y, p, g, average_precision_score),
           test_prevalence=float(y.mean()), test_n=int(len(y)), test_patients=int(len(np.unique(g))),
           one_stay_per_patient_auroc=roc_auc_score(one.label, one.p), one_stay_per_patient_auprc=average_precision_score(one.label, one.p),
           compute_seconds=time.time() - t0)
imp = pd.Series(m.booster_.feature_importance("gain"), index=X).sort_values(ascending=False).head(15)
res["top_features_gain"] = {k: float(v) for k, v in imp.items()}
json.dump(res, open(out, "w"), indent=2)
pd.DataFrame({"index_visit_id": te.index_visit_id, "person_id": te.person_id, "label": y, "pred": p}).to_parquet(out.replace(".json", "_test_preds.parquet"))
print(json.dumps(res, indent=2))
