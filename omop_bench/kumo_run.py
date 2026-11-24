"""Kumo Engine (local GNN, PQL) on the exported tables. Run with the Python that has kumo-engine installed:

/opt/kumo-venv/bin/python -m omop_bench.kumo_run tasks/<task>.yaml

The custom training table supplies the labels, so the PQL only fixes the target type (binary) and the graph."""
import argparse, json, time
from .task import load
from .paths import task_dir, result_dir

def run(cfg):
    t0 = time.time()
    from kumo.engine import Edge, FileConnector, Graph, PredictiveQuery, Table, Trainer, TrainingTable, configure
    from kumoapi.train import TrainingTableSpec
    from kumoapi.model_plan import RunMode
    name, kc = cfg["name"], cfg["kumo"]
    d = task_dir(name) / "kumo"
    configure(scratch_dir=str(task_dir(name) / "kumo_scratch"))
    files = FileConnector(root_dir=str(d / "tables"))
    T = {
     "person": Table.from_source_table(files["person"], primary_key="person_id"),
     "visit": Table.from_source_table(files["visit"], primary_key="visit_occurrence_id", time_column="visit_start"),
     "condition": Table.from_source_table(files["condition"], primary_key="condition_row_id", time_column="condition_start"),
     "drug": Table.from_source_table(files["drug"], primary_key="drug_row_id", time_column="drug_start"),
     "procedure": Table.from_source_table(files["procedure"], primary_key="procedure_row_id", time_column="procedure_time"),
     "measurement": Table.from_source_table(files["measurement"], primary_key="measurement_row_id", time_column="measurement_time"),
    }
    edges = [Edge(src_table="visit", fkey="person_id", dst_table="person")]
    for t in ["condition", "drug", "procedure", "measurement"]:
        edges += [Edge(src_table=t, fkey="person_id", dst_table="person"), Edge(src_table=t, fkey="visit_occurrence_id", dst_table="visit")]
    graph = Graph(tables=T, edges=edges)
    snap = graph.snapshot()
    pq = PredictiveQuery(graph, kc.get("pql", "PREDICT COUNT(visit.*, 0, 30, days) > 0 FOR EACH person.person_id"))
    plan = pq.suggest_model_plan(snap, run_mode=RunMode(kc["run_mode"]))
    plan.training_job.disable_explain = True
    # CPU container (~8 GB usable): Kumo's default (256 channels, batch 1024) was OOM-killed, so cap both.
    plan.optimization.batch_size = list(kc["batch_size"])
    plan.model_architecture.model[0].channels = list(kc["channels"])
    tt = TrainingTable.from_source_table(FileConnector(root_dir=str(d / "train"))["kumo_train"],
                                         TrainingTableSpec(split_col="SPLIT"), pq=pq, snapshot=snap)
    res = Trainer(plan).fit(tt, snapshot=snap)
    out = result_dir(name)
    res.holdout_df().to_parquet(out / "kumo_holdout.parquet")
    info = dict(task=name, run_mode=kc["run_mode"], batch_size=kc["batch_size"], channels=kc["channels"], compute_seconds=time.time() - t0)
    try:
        info["metrics"] = json.loads(json.dumps(res.metrics, default=str))
    except Exception as e:
        info["metrics"] = f"n/a: {e}"
    json.dump(info, open(out / "kumo_run.json", "w"), indent=2)
    print(json.dumps(info, indent=1, default=str), flush=True)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("task"); a = ap.parse_args()
    run(load(a.task))

if __name__ == "__main__":
    main()
