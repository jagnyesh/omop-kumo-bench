"""Kumo Engine (local GNN) on the linked OMOP tables, same rows/labels/split as LightGBM."""
import sys, time, json
t0 = time.time()
from kumo.engine import Edge, FileConnector, Graph, PredictiveQuery, Table, Trainer, TrainingTable, configure
from kumoapi.train import TrainingTableSpec
from kumoapi.model_plan import RunMode
root, scratch, out, mode = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
configure(scratch_dir=scratch)
files = FileConnector(root_dir=f"{root}/tables")
T = {
 "person": Table.from_source_table(files["person"], primary_key="person_id"),
 "visit": Table.from_source_table(files["visit"], primary_key="visit_occurrence_id", time_column="visit_start"),
 "condition": Table.from_source_table(files["condition"], primary_key="condition_row_id", time_column="condition_start"),
 "drug": Table.from_source_table(files["drug"], primary_key="drug_row_id", time_column="drug_start"),
 "procedure": Table.from_source_table(files["procedure"], primary_key="procedure_occurrence_id", time_column="procedure_time"),
 "measurement": Table.from_source_table(files["measurement"], primary_key="measurement_row_id", time_column="measurement_time"),
}
edges = [Edge(src_table="visit", fkey="person_id", dst_table="person")]
for t in ["condition", "drug", "procedure", "measurement"]:
    edges += [Edge(src_table=t, fkey="person_id", dst_table="person"), Edge(src_table=t, fkey="visit_occurrence_id", dst_table="visit")]
graph = Graph(tables=T, edges=edges)
for name, tab in T.items():
    print(name, tab, flush=True)
snap = graph.snapshot(); t_snap = time.time() - t0
print("snapshot s", t_snap, flush=True)
pq = PredictiveQuery(graph, "PREDICT COUNT(visit.*, 0, 30, days) > 0 FOR EACH person.person_id")
plan = pq.suggest_model_plan(snap, run_mode=RunMode(mode))
plan.training_job.disable_explain = True
# CPU container (~8 GB usable): first run was OOM-killed in trial 2 (256 channels, batch 1024).
plan.optimization.batch_size = [512]
plan.model_architecture.model[0].channels = [64, 128]
print(plan, flush=True)
tfiles = FileConnector(root_dir=f"{root}/train")
tt = TrainingTable.from_source_table(tfiles["kumo_train"], TrainingTableSpec(split_col="SPLIT"), pq=pq, snapshot=snap)
trainer = Trainer(plan)
res = trainer.fit(tt, snapshot=snap)
print("fit done s", time.time() - t0, flush=True)
h = res.holdout_df()
h.to_parquet(out)
print(h.head(), h.shape, flush=True)
try:
    print("metrics", res.metrics)
except Exception as e:
    print("metrics n/a", e)
print("total s", time.time() - t0)
