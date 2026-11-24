# OMOP prediction benchmark: Kumo Engine vs tuned LightGBM

A small, reusable harness that takes any "will X happen within N days of this visit?" question on an OMOP CDM
database and runs two approaches on exactly the same rows, labels and patient-level split:

- **LightGBM**: about 260 hand-built point-in-time features (demographics, prior use by visit type, top conditions,
  drugs, procedures and measurements), 40 Optuna trials on validation AUROC.
- **Kumo Engine**: a graph neural network trained straight on the linked OMOP tables, with no feature code.

Synthetic data only (Synthea). No real patient data is used or needed.

## Results so far (Synthea 100k, CPU only: 4 cores, about 8 GB usable RAM, no GPU)

| Task | Test events | Positive rate | LightGBM AUROC / AUPRC | Kumo AUROC / AUPRC |
|---|---|---|---|---|
| 30-day readmission after an inpatient stay | 10,068 | 16.7% | 0.957 / 0.813 | 0.952 / 0.795 |
| ED visit within 90 days of an outpatient visit | 5,025 | 5.5% | 0.814 / 0.616 | 0.809 / 0.591 |

Full numbers with 95% patient-cluster bootstrap intervals are in `results/<task>/comparison.json`, and leakage and
base-rate checks in `results/<task>/checks.json`.

What the two tasks show:
- **Readmission** is inflated by a Synthea artifact: 97.6% of readmissions come from 1,168 patients whom Synthea
  readmits every 21 to 30 days, and "days since previous inpatient stay" alone scores AUROC 0.93. A patient's first
  stay has a 2.2% readmission rate. The models are near a tie; an earlier run gave Kumo 0.956, so Kumo moves a few
  thousandths between runs.
- **ED visit** is a harder, less scheduled outcome, with no heavy repeat patients (one visit per patient). AUROC is
  a tie (difference -0.005, CI -0.028 to +0.017); LightGBM is ahead on AUPRC (-0.025, CI -0.046 to -0.005). The
  strongest signal is a cluster of outpatient follow-up procedures in the prior year (outpatient visits in the prior
  year alone score AUROC 0.77), which looks like Synthea's injury follow-up pattern rather than a timing leak.
- **Effort**: a new task is one YAML file for both models. Kumo needed no feature code; LightGBM reuses the generic
  feature builder. Kumo training took 8 minutes for the ED task and 84 minutes for readmission (55,046 rows, sharing
  the CPU); LightGBM took about 1 minute on readmission with the CPU to itself (its 55 minutes on the ED task were spent sharing the CPU with Kumo).

## Run it

```
pip install duckdb pandas pyyaml lightgbm optuna scikit-learn lzallright   # main environment
python -m venv /opt/kumo-venv && /opt/kumo-venv/bin/pip install kumo-engine duckdb pandas pyyaml scikit-learn
python -m omop_bench.data --release synthea100k        # download (about 1.5 GB) and load into DuckDB
python -m omop_bench.run tasks/readmission_30d.yaml    # cohort, features, LightGBM, Kumo, compare, checks
```

`--stages cohort,features,lgbm,kumo,compare,checks` runs a subset. Kumo runs in its own Python
(`KUMO_PYTHON`, default `/opt/kumo-venv/bin/python`) because it brings its own torch build. Large working files go
to `OMOP_BENCH_WORK` (default `/tmp/omop_bench`).

## Add a task

A task is a short YAML file in `tasks/`:

```yaml
name: ed_visit_90d
index:                      # the moment you predict at
  visit_concept_ids: [9202] # outpatient
  anchor: start             # start or end of the index visit
  from: 2010-01-01
  per_person: random        # all, first or random
population:
  max_persons: 25000        # keeps Kumo affordable on CPU
outcome:
  type: visit               # visit or condition
  concept_ids: [9203]       # emergency room visit
  window_days: 90
  new_onset: false          # true drops index events where the outcome already happened
```

Also supported: `index.min_age` / `max_age`, `index.to`, `split` (seed and percentages), `lightgbm` (top-N concept
counts, trials) and `kumo` (run mode, batch size, channels). The cohort builder guarantees that both models only see
records dated on or before the anchor date.

## Data notes (Synthea 100k from the AWS Open Data `synthea-omop` bucket, Apache-2.0)
- Ships with exact duplicate rows (visits x3, drugs and procedures x2); `omop_bench.data` removes them.
- `measurement.value_as_number` is empty, so labs are used as concept presence and counts only.
- Condition, drug and measurement ids are not unique, so surrogate row ids are added.
- Files are lzop-compressed; `omop_bench/unlzop.py` decodes them without the lzop binary.
- Not tested: NVIDIA's pretrained Kumo Relational model (weights are on Hugging Face, which this container could not reach).

## Layout
- `omop_bench/`: data, task (YAML to cohort), features, lgbm, kumo_export, kumo_run, compare, checks, run.
- `tasks/`: task configs. `results/<task>/`: metrics, predictions, timing.
- `code/` and the files directly in `results/`: the first readmission-only version (Nov 22), kept for reproducibility.
