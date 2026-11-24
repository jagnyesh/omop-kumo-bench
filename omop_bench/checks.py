"""Leakage and base-rate checks for a task cohort. Run before trusting a high score."""
import json
import duckdb, pandas as pd
from sklearn.metrics import roc_auc_score
from .paths import DB, task_dir, result_dir

EVENTS = [("condition_occurrence", "condition_start_date"), ("drug_exposure", "drug_exposure_start_date"),
          ("procedure_occurrence", "procedure_date"), ("measurement", "measurement_date")]

def run(cfg, con=None):
    c = con or duckdb.connect(str(DB), read_only=True)
    name = cfg["name"]; t = f"cohort_{name}"; r = {}
    one = lambda q: c.sql(q).fetchall()[0]
    r["patients_in_more_than_one_split"] = one(f"select count(*) from (select person_id from {t} group by 1 having count(distinct split) > 1)")[0]
    # records a model can see (dated <= anchor) that belong to a visit starting after the anchor: post-anchor information leaking backwards
    r["rows_dated_before_anchor_on_later_visits"] = {
        tab: one(f"""select count(*) from {t} k join {tab} e on e.person_id = k.person_id and e.{d} <= k.anchor_date
                     join visit_occurrence v on v.visit_occurrence_id = e.visit_occurrence_id and v.visit_start_date > k.anchor_date""")[0]
        for tab, d in EVENTS}
    r["rows_visible_total"] = sum(one(f"select count(*) from {t} k join {tab} e on e.person_id = k.person_id and e.{d} <= k.anchor_date")[0] for tab, d in EVENTS)
    n, rate = one(f"select count(*), avg(label) from {t}")
    r["index_events"], r["base_rate"] = n, float(rate)
    r["first_event_per_patient_rate"] = float(one(f"""select avg(label) from (select *, row_number() over (partition by person_id order by anchor_date, index_id) rn from {t}) where rn = 1""")[0])
    share, heavy = one(f"""with n as (select person_id, count(*) n from {t} group by 1)
        select sum(label) filter (where n >= 10) / sum(label), count(distinct person_id) filter (where n >= 10) from {t} join n using (person_id)""")
    r["share_of_positives_from_patients_with_10plus_events"], r["patients_with_10plus_events"] = float(share or 0), heavy
    f = pd.read_parquet(task_dir(name) / "features.parquet"); te = f[f.split == 2]
    single = {}
    for col in ["days_since_prev_ip", "days_since_prev_er", "ip_365", "er_365", "op_365", "ip_all", "age", "cond_n", "drug_n", "los"]:
        x = te[col].astype(float).fillna(1e5 if col.startswith("days_since") else -1)
        a = roc_auc_score(te.label, x); single[col] = round(max(a, 1 - a), 3)
    r["single_feature_test_auroc"] = single
    json.dump(r, open(result_dir(name) / "checks.json", "w"), indent=2, default=float)
    print(json.dumps(r, indent=1, default=float), flush=True)
    return r
