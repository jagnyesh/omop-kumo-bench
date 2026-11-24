"""Fetch the AWS Open Data Synthea OMOP release and load it into DuckDB, cleaned.

python -m omop_bench.data [--release synthea100k]

Cleaning (all found in this release): exact duplicate rows removed; value_as_number cast to double
(it is empty in synthea100k); condition and measurement ids are not unique, so surrogate row ids are added."""
import argparse, os, urllib.request
import duckdb
from .paths import RAW, DB
from .unlzop import decode

BUCKET = "https://synthea-omop.s3.amazonaws.com"
TABLES = {
    "person": "person_id, gender_concept_id, year_of_birth, month_of_birth, race_concept_id, ethnicity_concept_id",
    "visit_occurrence": "visit_occurrence_id, person_id, visit_concept_id, visit_start_date, visit_end_date",
    "condition_occurrence": "person_id, condition_concept_id, condition_start_date, condition_end_date, visit_occurrence_id",
    "drug_exposure": "person_id, drug_concept_id, drug_exposure_start_date, drug_exposure_end_date, visit_occurrence_id",
    "procedure_occurrence": "person_id, procedure_concept_id, procedure_date, visit_occurrence_id",
    "measurement": "person_id, measurement_concept_id, measurement_date, try_cast(value_as_number as double) value_as_number, visit_occurrence_id",
}
ROW_ID = {"condition_occurrence": "condition_row_id", "drug_exposure": "drug_row_id",
          "procedure_occurrence": "procedure_row_id", "measurement": "measurement_row_id"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", default="synthea100k", help="synthea1k (plain csv) or synthea100k (lzo)")
    a = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True); DB.parent.mkdir(parents=True, exist_ok=True)
    c = duckdb.connect(str(DB))
    lzo = a.release != "synthea1k"
    for t, cols in TABLES.items():
        csv = RAW / f"{t}.csv"
        if not csv.exists():
            src = RAW / (f"{t}.csv.lzo" if lzo else f"{t}.csv")
            if not src.exists():
                url = f"{BUCKET}/{a.release}/{src.name}"
                print("download", url, flush=True); urllib.request.urlretrieve(url, src)
            if lzo:
                print("decode", src.name, decode(src, csv), "bytes", flush=True)
        sel = f"select distinct {cols} from read_csv('{csv}', header=true)"
        if t in ROW_ID:
            sel = f"select row_number() over () {ROW_ID[t]}, * from ({sel})"
        c.sql(f"create or replace table {t} as {sel}")
        print(t, c.sql(f"select count(*) from {t}").fetchall()[0][0], "rows", flush=True)
        if lzo:
            os.remove(csv)  # keep the compressed copy only
    c.sql("create or replace table meta as select max(visit_end_date) data_end from visit_occurrence")
    print("data end", c.sql("select data_end from meta").fetchall()[0][0])

if __name__ == "__main__":
    main()
