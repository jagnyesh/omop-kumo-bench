"""Hand-built point-in-time features for the LightGBM baseline, for any task cohort.

Every feature uses only records dated on or before the index event's anchor date.
Concept vocabularies (top-N conditions, drugs, procedures, measurements) are picked on training patients only."""
import duckdb
from .paths import DB, task_dir

IP = "9201, 262"

def build(cfg, con=None):
    c = con or duckdb.connect(str(DB))
    name, lg = cfg["name"], cfg["lightgbm"]
    c.sql(f"create or replace temp table k as select index_id, person_id, start_date a, anchor_date e, label, split from cohort_{name}")
    c.sql("""create or replace temp table f_base as
      select k.index_id, (year(k.a) - p.year_of_birth) age, p.gender_concept_id gender, p.race_concept_id race,
             p.ethnicity_concept_id eth, (k.e - k.a) los, month(k.e) anchor_month
      from k join person p using (person_id)""")
    c.sql(f"""create or replace temp table f_util as
      select k.index_id,
        count(*) filter (where v.visit_concept_id in ({IP})) ip_all,
        count(*) filter (where v.visit_concept_id in ({IP}) and v.visit_start_date > k.e - 30) ip_30,
        count(*) filter (where v.visit_concept_id in ({IP}) and v.visit_start_date > k.e - 180) ip_180,
        count(*) filter (where v.visit_concept_id in ({IP}) and v.visit_start_date > k.e - 365) ip_365,
        count(*) filter (where v.visit_concept_id = 9203 and v.visit_start_date > k.e - 365) er_365,
        count(*) filter (where v.visit_concept_id = 9203) er_all,
        count(*) filter (where v.visit_concept_id = 9202 and v.visit_start_date > k.e - 365) op_365,
        count(*) filter (where v.visit_concept_id = 9202) op_all,
        min(k.a - v.visit_end_date) filter (where v.visit_concept_id in ({IP}) and v.visit_end_date <= k.a) days_since_prev_ip,
        min(k.a - v.visit_end_date) filter (where v.visit_concept_id = 9203 and v.visit_end_date <= k.a) days_since_prev_er,
        sum(v.visit_end_date - v.visit_start_date) filter (where v.visit_concept_id in ({IP}) and v.visit_start_date > k.e - 365) ip_days_365
      from k left join visit_occurrence v on v.person_id = k.person_id and v.visit_start_date <= k.e and v.visit_occurrence_id <> k.index_id
      group by 1""")

    def block(short, table, col, date, n, window=None):
        concepts = [r[0] for r in c.sql(f"""select {col} from {table} t join (select distinct person_id from k where split = 0) using (person_id)
                    group by 1 order by count(distinct person_id) desc, 1 limit {n}""").fetchall()]
        win = f"and t.{date} > k.e - {window}" if window else ""
        flags = "".join(f",\n max(case when t.{col} = {cid} then 1 else 0 end) {short}_{cid}" for cid in concepts)
        c.sql(f"""create or replace temp table f_{short} as
          select k.index_id, count(t.{col}) {short}_n, count(distinct t.{col}) {short}_ndistinct,
            count(t.{col}) filter (where t.{date} >= k.a) {short}_n_during_index {flags}
          from k left join {table} t on t.person_id = k.person_id and t.{date} <= k.e {win}
          group by 1""")

    block("cond", "condition_occurrence", "condition_concept_id", "condition_start_date", lg["top_conditions"])
    block("drug", "drug_exposure", "drug_concept_id", "drug_exposure_start_date", lg["top_drugs"], 365)
    block("proc", "procedure_occurrence", "procedure_concept_id", "procedure_date", lg["top_procedures"], 365)
    block("meas", "measurement", "measurement_concept_id", "measurement_date", lg["top_measurements"], 365)
    out = task_dir(name) / "features.parquet"
    c.sql(f"""copy (select k.index_id, k.person_id, k.e anchor_date, k.label, k.split,
                 f_base.* exclude (index_id), f_util.* exclude (index_id), f_cond.* exclude (index_id),
                 f_drug.* exclude (index_id), f_proc.* exclude (index_id), f_meas.* exclude (index_id)
               from k join f_base using (index_id) join f_util using (index_id) join f_cond using (index_id)
                 join f_drug using (index_id) join f_proc using (index_id) join f_meas using (index_id)) to '{out}' (format parquet)""")
    print("features", c.sql(f"select count(*) from '{out}'").fetchall()[0][0], "rows", len(c.sql(f"select * from '{out}' limit 1").columns) - 5, "features", flush=True)
    return out
