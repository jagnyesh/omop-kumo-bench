"""Hand-built point-in-time features for LightGBM. Every feature uses only records dated <= discharge date."""
import duckdb, sys
db, out = sys.argv[1], sys.argv[2]
c = duckdb.connect(db)
c.sql("create or replace table k as select index_visit_id, person_id, admit_date a, discharge_date e, label, split from cohort")

# demographics + index stay
c.sql("""create or replace table f_base as
select k.index_visit_id, (year(k.a) - p.year_of_birth) age, p.gender_concept_id gender, p.race_concept_id race, p.ethnicity_concept_id eth,
       (k.e - k.a) los, month(k.e) disch_month
from k join person p using (person_id)""")

# prior utilization by visit type and window (visits starting <= discharge, excluding the index visit itself)
c.sql("""create or replace table f_util as
select k.index_visit_id,
  count(*) filter (where v.visit_concept_id in (9201,262)) ip_all,
  count(*) filter (where v.visit_concept_id in (9201,262) and v.visit_start_date > k.e - 30) ip_30,
  count(*) filter (where v.visit_concept_id in (9201,262) and v.visit_start_date > k.e - 180) ip_180,
  count(*) filter (where v.visit_concept_id in (9201,262) and v.visit_start_date > k.e - 365) ip_365,
  count(*) filter (where v.visit_concept_id = 9203 and v.visit_start_date > k.e - 365) er_365,
  count(*) filter (where v.visit_concept_id = 9203) er_all,
  count(*) filter (where v.visit_concept_id = 9202 and v.visit_start_date > k.e - 365) op_365,
  count(*) filter (where v.visit_concept_id = 9202) op_all,
  min(k.a - v.visit_end_date) filter (where v.visit_concept_id in (9201,262) and v.visit_end_date <= k.a) days_since_prev_ip,
  sum(v.visit_end_date - v.visit_start_date) filter (where v.visit_concept_id in (9201,262) and v.visit_start_date > k.e - 365) ip_days_365
from k left join visit_c v on v.person_id = k.person_id and v.visit_start_date <= k.e and v.visit_occurrence_id <> k.index_visit_id
group by 1""")

# top concepts by frequency among training patients
def top(table, col, date, n):
    return [r[0] for r in c.sql(f"""select {col} from {table} t join (select distinct person_id from k where split=0) using (person_id)
              group by 1 order by count(distinct person_id) desc limit {n}""").fetchall()]

def block(name, table, col, date, n, window=None):
    concepts = top(table, col, date, n)
    win = f"and t.{date} > k.e - {window}" if window else ""
    flags = ",\n".join(f"max(case when t.{col} = {cid} then 1 else 0 end) {name}_{cid}" for cid in concepts)
    c.sql(f"""create or replace table f_{name} as
      select k.index_visit_id, count(t.{col}) {name}_n, count(distinct t.{col}) {name}_ndistinct,
        count(t.{col}) filter (where t.{date} >= k.a) {name}_n_during_stay,
        {flags}
      from k left join {table} t on t.person_id = k.person_id and t.{date} <= k.e {win}
      group by 1""")
    print(name, len(concepts), flush=True)

block("cond", "condition_occurrence", "condition_concept_id", "condition_start_date", 100)
block("drug", "drug_exposure", "drug_concept_id", "drug_exposure_start_date", 50, window=365)
block("proc", "procedure_occurrence", "procedure_concept_id", "procedure_date", 50, window=365)

# measurement values are empty in this export, so use concept flags + counts only
block("meas", "measurement", "measurement_concept_id", "measurement_date", 30, window=365)

c.sql("""create or replace table features as
select k.index_visit_id, k.person_id, k.e discharge_date, k.label, k.split, * exclude (index_visit_id)
from k join f_base using (index_visit_id) join f_util using (index_visit_id) join f_cond using (index_visit_id)
 join f_drug using (index_visit_id) join f_proc using (index_visit_id) join f_meas using (index_visit_id)""")
c.sql(f"copy features to '{out}' (format parquet)")
print(c.sql("select count(*), count(distinct index_visit_id) from features").fetchall(), len(c.sql("select * from features limit 1").columns))
