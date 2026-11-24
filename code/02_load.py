"""Decompress each Synthea 100k OMOP table, keep only cohort patients, store in DuckDB."""
import duckdb, os, sys, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unlzop import decode
d = sys.argv[1]
c = duckdb.connect(f"{d}/omop.duckdb")
c.sql("create or replace table persons_in as select distinct person_id from cohort")
c.sql(f"""create or replace table person as select person_id, gender_concept_id, year_of_birth, month_of_birth, race_concept_id, ethnicity_concept_id
          from read_csv('{d}/person.csv', header=true) join persons_in using (person_id)""")
c.sql("create or replace table visit_c as select v.* from visit v join persons_in using (person_id)")
specs = {
  "condition_occurrence": "condition_occurrence_id, person_id, condition_concept_id, condition_start_date, condition_end_date, visit_occurrence_id",
  "drug_exposure": "drug_exposure_id, person_id, drug_concept_id, drug_exposure_start_date, drug_exposure_end_date, visit_occurrence_id",
  "procedure_occurrence": "procedure_occurrence_id, person_id, procedure_concept_id, procedure_date, visit_occurrence_id",
  "measurement": "measurement_id, person_id, measurement_concept_id, measurement_date, value_as_number, visit_occurrence_id",
}
for t, cols in specs.items():
    csv = f"{d}/{t}.csv"
    if not os.path.exists(csv):
        print(t, "bytes", decode(f"{csv}.lzo", csv), flush=True)
    c.sql(f"create or replace table {t} as select {cols} from read_csv('{csv}', header=true) join persons_in using (person_id)")
    print(t, c.sql(f"select count(*) from {t}").fetchall(), flush=True)
    os.remove(csv)
for t in ["person", "visit_c"]:
    print(t, c.sql(f"select count(*) from {t}").fetchall())
