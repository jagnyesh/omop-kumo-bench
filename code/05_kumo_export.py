"""Export the linked OMOP tables (cohort patients only) as Parquet for Kumo, plus the custom training table."""
import duckdb, sys
db, out = sys.argv[1], sys.argv[2]
c = duckdb.connect(db, read_only=True)
q = {
 "person": "select person_id, gender_concept_id::varchar gender, year_of_birth, race_concept_id::varchar race, ethnicity_concept_id::varchar ethnicity from person",
 "visit": "select visit_occurrence_id, person_id, visit_concept_id::varchar visit_type, visit_start_date::timestamp visit_start, (visit_end_date - visit_start_date)::double length_of_stay_days from visit_c",
 "condition": "select row_number() over () condition_row_id, person_id, visit_occurrence_id, condition_concept_id::varchar condition_concept, condition_start_date::timestamp condition_start from condition_occurrence",
 "drug": "select row_number() over () drug_row_id, person_id, visit_occurrence_id, drug_concept_id::varchar drug_concept, drug_exposure_start_date::timestamp drug_start from drug_exposure",
 "procedure": "select procedure_occurrence_id, person_id, visit_occurrence_id, procedure_concept_id::varchar procedure_concept, procedure_date::timestamp procedure_time from procedure_occurrence",
 "measurement": "select row_number() over () measurement_row_id, person_id, visit_occurrence_id, measurement_concept_id::varchar measurement_concept, measurement_date::timestamp measurement_time from measurement",
}
for name, sql in q.items():
    c.sql(f"copy ({sql}) to '{out}/tables/{name}.parquet' (format parquet)")
    print(name, c.sql(f"select count(*) from '{out}/tables/{name}.parquet'").fetchall())
# drug / procedure ids must be unique to serve as primary keys
for t, k in [("drug", "drug_row_id"), ("procedure", "procedure_occurrence_id"), ("visit", "visit_occurrence_id")]:
    print(t, "unique pk:", c.sql(f"select count(*) = count(distinct {k}) from '{out}/tables/{t}.parquet'").fetchall())
c.sql(f"""copy (select person_id::bigint ENTITY1, discharge_date::timestamp "TIMESTAMP", label::bool TARGET1, split::bigint SPLIT, index_visit_id
         from cohort order by person_id, discharge_date) to '{out}/train/training_table.parquet' (format parquet)""")
print(c.sql(f"select SPLIT, count(*), avg(TARGET1::int) from '{out}/train/training_table.parquet' group by 1 order by 1").fetchall())
