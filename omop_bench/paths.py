import os
from pathlib import Path
# Large working files live outside the shared project folder; override with env vars.
WORK = Path(os.environ.get("OMOP_BENCH_WORK", "/tmp/omop_bench"))
RAW = Path(os.environ.get("OMOP_BENCH_RAW", WORK / "raw"))
DB = Path(os.environ.get("OMOP_BENCH_DB", WORK / "omop.duckdb"))
ROOT = Path(__file__).resolve().parent.parent
RESULTS = Path(os.environ.get("OMOP_BENCH_RESULTS", ROOT / "results"))

def task_dir(name):
    d = WORK / "tasks" / name
    d.mkdir(parents=True, exist_ok=True)
    return d

def result_dir(name):
    d = RESULTS / name
    d.mkdir(parents=True, exist_ok=True)
    return d
