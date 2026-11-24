"""Tuned LightGBM on the hand-built features: Optuna on validation AUROC, test metrics with patient-cluster bootstrap CIs."""
import json, time
import numpy as np, pandas as pd, lightgbm as lgb, optuna
from sklearn.metrics import roc_auc_score, average_precision_score
from .paths import task_dir, result_dir
from .metrics import cluster_boot

ID_COLS = ("index_id", "person_id", "anchor_date", "label", "split")

def run(cfg):
    t0 = time.time(); name = cfg["name"]
    df = pd.read_parquet(task_dir(name) / "features.parquet")
    for col in ["gender", "race", "eth"]:
        df[col] = df[col].astype("category")
    X = [col for col in df.columns if col not in ID_COLS]
    tr, va, te = (df[df.split == s] for s in (0, 1, 2))

    def fit(params):
        m = lgb.LGBMClassifier(n_estimators=2000, verbose=-1, n_jobs=4, **params)
        m.fit(tr[X], tr.label, eval_set=[(va[X], va.label)], eval_metric="auc", callbacks=[lgb.early_stopping(100, verbose=False)])
        return m

    def objective(trial):
        p = dict(learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                 num_leaves=trial.suggest_int("num_leaves", 8, 128, log=True),
                 min_child_samples=trial.suggest_int("min_child_samples", 10, 300, log=True),
                 subsample=trial.suggest_float("subsample", 0.5, 1.0), subsample_freq=1,
                 colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
                 reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True))
        return roc_auc_score(va.label, fit(p).predict_proba(va[X])[:, 1])

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=7))
    study.optimize(objective, n_trials=int(cfg["lightgbm"]["trials"]))
    best = dict(study.best_params, subsample_freq=1)
    m = fit(best)
    p = m.predict_proba(te[X])[:, 1]
    y, g = te.label.values, te.person_id.values
    ci = cluster_boot(y, {"lgbm": p}, g)
    one = te.assign(p=p).groupby("person_id").sample(1, random_state=0)
    res = dict(task=name, model="LightGBM (tuned, hand-built features)", n_features=len(X), best_val_auroc=study.best_value,
               best_params=best, best_iteration=int(m.best_iteration_),
               test_auroc=roc_auc_score(y, p), test_auroc_ci=ci["lgbm_auroc"],
               test_auprc=average_precision_score(y, p), test_auprc_ci=ci["lgbm_auprc"],
               test_prevalence=float(y.mean()), test_n=int(len(y)), test_patients=int(len(np.unique(g))),
               one_event_per_patient_auroc=roc_auc_score(one.label, one.p), one_event_per_patient_auprc=average_precision_score(one.label, one.p),
               compute_seconds=time.time() - t0)
    imp = pd.Series(m.booster_.feature_importance("gain"), index=X).sort_values(ascending=False).head(15)
    res["top_features_gain"] = {k: float(v) for k, v in imp.items()}
    out = result_dir(name)
    json.dump(res, open(out / "lgbm.json", "w"), indent=2)
    pd.DataFrame({"index_id": te.index_id, "person_id": te.person_id, "anchor_date": te.anchor_date, "label": y, "pred": p}).to_parquet(out / "lgbm_test_preds.parquet")
    print(json.dumps({k: res[k] for k in ("test_auroc", "test_auprc", "best_val_auroc", "compute_seconds")}, indent=1), flush=True)
    return res
