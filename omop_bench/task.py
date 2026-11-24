"""Task config -> cohort table of index events with labels and a patient-level split.

A task is "for each index event, does the outcome happen within N days after the anchor time?"
Every downstream stage only sees records dated on or before the anchor date."""
import argparse, yaml
import duckdb
from .paths import DB

DEFAULTS = {
    "index": {"anchor": "end", "from": None, "to": None, "per_person": "all", "min_age": None, "max_age": None},
    "outcome": {"new_onset": False},
    "population": {"max_persons": None},
    "split": {"seed": "kumo-synthea-seed-7", "train": 70, "val": 10},
    "lightgbm": {"top_conditions": 100, "top_drugs": 50, "top_procedures": 50, "top_measurements": 30, "trials": 40},
    "kumo": {"run_mode": "fast", "batch_size": [512], "channels": [64, 128]},
}

def load(path):
    cfg = yaml.safe_load(open(path))
    for k, v in DEFAULTS.items():
        cfg[k] = {**v, **cfg.get(k, {})}
    return cfg

def ids(xs):
    return ",".join(str(int(x)) for x in xs)

def build(cfg, con=None):
    c = con or duckdb.connect(str(DB))
    ix, oc, sp, pop = cfg["index"], cfg["outcome"], cfg["split"], cfg["population"]
    w = int(oc["window_days"])
    data_end = c.sql("select data_end from meta").fetchall()[0][0]
    anchor = "visit_end_date" if ix["anchor"] == "end" else "visit_start_date"
    lo = f"date '{ix['from']}'" if ix["from"] else "date '1900-01-01'"
    hi = f"date '{ix['to']}'" if ix["to"] else f"date '{data_end}' - {w}"
    age = []
    if ix["min_age"] is not None: age.append(f"year(v.visit_start_date) - p.year_of_birth >= {int(ix['min_age'])}")
    if ix["max_age"] is not None: age.append(f"year(v.visit_start_date) - p.year_of_birth <= {int(ix['max_age'])}")
    agesql = (" and " + " and ".join(age)) if age else ""
    c.sql(f"""create or replace temp table idx as
      select v.visit_occurrence_id index_id, v.person_id, v.visit_start_date start_date, v.{anchor} anchor_date
      from visit_occurrence v join person p using (person_id)
      where v.visit_concept_id in ({ids(ix['visit_concept_ids'])}) and v.{anchor} between {lo} and {hi} {agesql}""")
    # outcome events
    if oc["type"] == "visit":
        c.sql(f"""create or replace temp table ev as select person_id, visit_occurrence_id ev_id, visit_start_date d
                  from visit_occurrence where visit_concept_id in ({ids(oc['concept_ids'])})""")
    elif oc["type"] == "condition":
        c.sql(f"""create or replace temp table ev as select person_id, -1 ev_id, condition_start_date d
                  from condition_occurrence where condition_concept_id in ({ids(oc['concept_ids'])})""")
    else:
        raise ValueError("outcome.type must be visit or condition")
    if oc["new_onset"]:  # drop index events where the outcome already happened on or before the anchor
        c.sql("""delete from idx where exists (select 1 from ev where ev.person_id = idx.person_id and ev.d <= idx.anchor_date)""")
    if ix["per_person"] in ("first", "random"):
        order = "anchor_date" if ix["per_person"] == "first" else f"hash(index_id::varchar || '{sp['seed']}')"
        c.sql(f"create or replace temp table idx as select * exclude (rn) from (select *, row_number() over (partition by person_id order by {order}) rn from idx) where rn = 1")
    if pop["max_persons"]:
        c.sql(f"""create or replace temp table idx as select * from idx where person_id in (
                  select person_id from (select distinct person_id from idx) order by hash(person_id::varchar || '{sp['seed']}') limit {int(pop['max_persons'])})""")
    c.sql(f"""create or replace table cohort_{cfg['name']} as
      select i.*, (exists (select 1 from ev where ev.person_id = i.person_id and ev.ev_id <> i.index_id
                           and ev.d > i.anchor_date and ev.d <= i.anchor_date + {w}))::int label,
             case when h < {sp['train']} then 0 when h < {sp['train'] + sp['val']} then 1 else 2 end split
      from idx i join (select distinct person_id, hash(person_id::varchar || '{sp['seed']}') % 100 h from idx) s using (person_id)""")
    t = f"cohort_{cfg['name']}"
    print(c.sql(f"select split, count(*) rows, count(distinct person_id) patients, round(avg(label), 4) rate from {t} group by 1 order by 1"))
    return t

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("task"); a = ap.parse_args()
    build(load(a.task))

if __name__ == "__main__":
    main()
