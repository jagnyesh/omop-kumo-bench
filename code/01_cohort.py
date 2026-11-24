"""Build the 30-day readmission cohort from Synthea 100k OMOP visits.
Index: inpatient visits (9201, 262) discharged 2010-01-01 .. data_end-30d.
Label: another inpatient admission starting in (discharge, discharge+30d].
Split: patient-level, seeded random 70/10/20 train/val/test."""
import duckdb, sys
db = sys.argv[1]
c = duckdb.connect(db)
c.sql("""
create or replace table ip as
select visit_occurrence_id, person_id, visit_start_date s, visit_end_date e
from visit where visit_concept_id in (9201, 262)""")
data_end = c.sql("select max(visit_end_date) from visit").fetchall()[0][0]
print("data end", data_end)
c.sql(f"""
create or replace table cohort as
select a.visit_occurrence_id index_visit_id, a.person_id, a.s admit_date, a.e discharge_date,
       (exists(select 1 from ip b where b.person_id=a.person_id and b.s > a.e and b.s <= a.e + 30))::int label
from ip a
where a.e >= date '2010-01-01' and a.e <= date '{data_end}' - 30""")
c.sql("""
create or replace table person_split as
select person_id,
  case when h < 70 then 0 when h < 80 then 1 else 2 end split
from (select distinct person_id, hash(person_id::varchar || 'kumo-synthea-seed-7') % 100 h from cohort)""")
c.sql("create or replace table cohort as select c.*, s.split from cohort c join person_split s using (person_id)")
print(c.sql("select split, count(*) stays, count(distinct person_id) patients, avg(label) readmit_rate from cohort group by 1 order by 1"))
print(c.sql("select quantile_cont(n,[0.5,0.9,0.99]) , max(n) from (select person_id, count(*) n from cohort group by 1)"))
