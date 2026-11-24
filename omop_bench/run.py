"""Run a task end to end.

python -m omop_bench.run tasks/readmission_30d.yaml [--stages cohort,features,lgbm,kumo,compare,checks]

Kumo runs in a subprocess with KUMO_PYTHON (default /opt/kumo-venv/bin/python), since it needs its own torch build."""
import argparse, os, subprocess, sys, time, json
import duckdb
from . import task, features, lgbm, kumo_export, compare, checks
from .paths import DB, ROOT, WORK, result_dir

ALL = ["cohort", "features", "lgbm", "kumo", "compare", "checks"]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("task"); ap.add_argument("--stages", default=",".join(ALL))
    a = ap.parse_args(); cfg = task.load(a.task); stages = a.stages.split(",")
    timing = {}
    def stage(s, fn):
        if s in stages:
            t0 = time.time(); print(f"== {s}", flush=True); fn(); timing[s] = round(time.time() - t0, 1)
    with duckdb.connect(str(DB)) as c:
        stage("cohort", lambda: task.build(cfg, c))
        stage("features", lambda: features.build(cfg, c))
        stage("kumo", lambda: kumo_export.export(cfg, c))
    stage("lgbm", lambda: lgbm.run(cfg))
    py = os.environ.get("KUMO_PYTHON", "/opt/kumo-venv/bin/python")
    if "kumo" in stages:
        t0 = time.time(); print("== kumo train", flush=True)
        # run from the local work dir: torch calls getcwd() at import, which fails on some network mounts
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        subprocess.run([py, "-m", "omop_bench.kumo_run", os.path.abspath(a.task)], cwd=WORK, env=env, check=True)
        timing["kumo_train"] = round(time.time() - t0, 1)
    with duckdb.connect(str(DB), read_only=True) as c:
        stage("compare", lambda: compare.run(cfg))
        stage("checks", lambda: checks.run(cfg, c))
    p = result_dir(cfg["name"]) / "timing.json"
    old = json.load(open(p)) if p.exists() else {}
    json.dump({**old, **timing}, open(p, "w"), indent=2)
    print("timing", timing)

if __name__ == "__main__":
    main()
