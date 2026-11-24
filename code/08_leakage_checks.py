"""Leakage and base-rate checks requested before trusting the ~0.96 AUROC."""
import duckdb, numpy as np, pandas as pd, sys
from sklearn.metrics import roc_auc_score
db = sys.argv[1]
c = duckdb.connect(db, read_only=True)
print("1. patients in more than one split:", c.sql("select count(*) from (select person_id from cohort group by 1 having count(distinct split)>1)").fetchall())
print("2. index stays whose label visit is the index visit itself (should be 0 by construction):",
      c.sql("""select count(*) from cohort k join ip b on b.person_id=k.person_id and b.s > k.discharge_date and b.s <= k.discharge_date+30
               where b.visit_occurrence_id = k.index_visit_id""").fetchall())
print("3. event rows dated <= discharge but attached to a later readmission visit (post-discharge info leaking backwards):")
for t, d in [("condition_occurrence","condition_start_date"),("drug_exposure","drug_exposure_start_date"),("procedure_occurrence","procedure_date"),("measurement","measurement_date")]:
    n = c.sql(f"""select count(*) from cohort k join {t} e on e.person_id=k.person_id and e.{d} <= k.discharge_date
                 join visit_c v on v.visit_occurrence_id=e.visit_occurrence_id and v.visit_start_date > k.discharge_date""").fetchall()[0][0]
    print("   ", t, n)
print("4. base rate:", c.sql("select avg(label), count(*) from cohort").fetchall())
print("   gap (days) from discharge to next inpatient admission, readmitted stays:")
print(c.sql("""select quantile_cont(g,[0.1,0.25,0.5,0.75,0.9]) q, count(*) from (
   select k.index_visit_id, min(b.s - k.discharge_date) g from cohort k join ip b on b.person_id=k.person_id and b.s>k.discharge_date
   where k.label=1 group by 1)""").fetchall())
print("   share of readmissions from patients with >=10 index stays:",
      c.sql("""with n as (select person_id, count(*) n from cohort group by 1)
               select sum(label) filter (where n>=10)/sum(label), count(distinct person_id) filter (where n>=10) from cohort join n using (person_id)""").fetchall())
print("   most common condition concepts in the readmission visits (top 8):")
print(c.sql("""select co.condition_concept_id, count(*) from cohort k join ip b on b.person_id=k.person_id and b.s>k.discharge_date and b.s<=k.discharge_date+30
    join condition_occurrence co on co.visit_occurrence_id=b.visit_occurrence_id group by 1 order by 2 desc limit 8""").fetchall())
f = pd.read_parquet('/mnt/project-files/kumo-synthea/data/lgbm_features.parquet'); te = f[f.split==2]
print("5. single-feature AUROC on test (no model):")
for col in ["ip_180","ip_365","ip_30","days_since_prev_ip","drug_n","ip_all","los","age"]:
    x = te[col].fillna(-1 if col!="days_since_prev_ip" else 1e5).astype(float)
    a = roc_auc_score(te.label, x); print(f"    {col}: {max(a,1-a):.3f}")
first = te.sort_values("discharge_date").groupby("person_id").head(1)
print("6. test, first index stay per patient only: n", len(first), "rate", round(first.label.mean(),3))
