"""Export the linked OMOP tables (task cohort patients only) as Parquet for Kumo, plus Kumo's custom training table
(same index events, labels and split as LightGBM)."""
import duckdb
from .paths import DB, task_dir

TABLES = {
 "person": "select person_id, gender_concept_id::varchar gender, year_of_birth, race_concept_id::varchar race, ethnicity_concept_id::varchar ethnicity from person",
 "visit": "select visit_occurrence_id, person_id, visit_concept_id::varchar visit_type, visit_start_date::timestamp visit_start, (visit_end_date - visit_start_date)::double length_of_stay_days from visit_occurrence",
 "condition": "select condition_row_id, person_id, visit_occurrence_id, condition_concept_id::varchar condition_concept, condition_start_date::timestamp condition_start from condition_occurrence",
 "drug": "select drug_row_id, person_id, visit_occurrence_id, drug_concept_id::varchar drug_concept, drug_exposure_start_date::timestamp drug_start from drug_exposure",
 "procedure": "select procedure_row_id, person_id, visit_occurrence_id, procedure_concept_id::varchar procedure_concept, procedure_date::timestamp procedure_time from procedure_occurrence",
 "measurement": "select measurement_row_id, person_id, visit_occurrence_id, measurement_concept_id::varchar measurement_concept, measurement_date::timestamp measurement_time from measurement",
}

def export(cfg, con=None):
    c = con or duckdb.connect(str(DB), read_only=True)
    name = cfg["name"]; d = task_dir(name) / "kumo"
    (d / "tables").mkdir(parents=True, exist_ok=True); (d / "train").mkdir(parents=True, exist_ok=True)
    keep = f"person_id in (select distinct person_id from cohort_{name})"
    for t, sql in TABLES.items():
        c.sql(f"copy (select * from ({sql}) where {keep}) to '{d}/tables/{t}.parquet' (format parquet)")
        print(t, c.sql(f"select count(*) from '{d}/tables/{t}.parquet'").fetchall()[0][0], flush=True)
    # Kumo labels each (entity, timestamp) row; anchor timestamp = anchor date, so only records dated <= anchor are visible.
    c.sql(f"""copy (select person_id::bigint ENTITY1, anchor_date::timestamp "TIMESTAMP", label::bool TARGET1, split::bigint SPLIT
             from cohort_{name} order by person_id, anchor_date) to '{d}/train/kumo_train.parquet' (format parquet)""")
    print(c.sql(f"select SPLIT, count(*), round(avg(TARGET1::int), 4) from '{d}/train/kumo_train.parquet' group by 1 order by 1").fetchall(), flush=True)
    return d
