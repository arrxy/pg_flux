# Resident CUDA joins through PostgreSQL

Extension version 0.1.1 adds a narrow, explicit SQL entry point for the
TPC-H `lineitem → orders → customer` pipeline. Loading a snapshot reads
the named tables under the caller's read-only repeatable-read transaction
and uploads the projected columns. A call to the resident operator runs
the two joins and final reduction on CUDA. Intermediate customer keys
stay on the device. The host receives a 24-byte aggregate result from
CUDA and returns it as a bigint array with PostgreSQL NULL semantics.

The snapshot is private to the backend, tied to the loading role, and
released on commit or rollback. Ordinary JOIN statements continue to use
PostgreSQL's planner. The SQL API currently runs one fixed aggregate shape
over preselected tables. Each warm call reruns CUDA work on the same
snapshot. Query predicates and transparent SQL planning are future work.

## PostgreSQL timings

The measurements are `EXPLAIN ANALYZE` execution times on PostgreSQL 18.6
with the A100. For each scale, three fresh PostgreSQL sessions timed the
explicit load, executed an untimed correctness query, and timed seven
warm calls. Here, cold refers to an unloaded GPU snapshot. The OS page
cache was left in its normal state. The tuned native figures are from
the earlier experiment on the same datasets, using four allowed parallel
workers and 256 MiB `work_mem`.

| Scale | Tuned native SQL | Resident load median | Resident warm median |
| --- | ---: | ---: | ---: |
| SF1 | 0.886 s | 1.702 s | 0.315 ms |
| SF5 | 4.957 s | 6.623 s | 0.640 ms |

The first SF5 result costs about 6.62 s including SQL scan, packing, upload,
and GPU lookup construction. Later calls reuse the snapshot and took
0.606–2.910 ms in these runs. The resident function is therefore an
experiment in reusing GPU input within a transaction. A single cold query
still favors native SQL. These timings cover the fixed aggregation, whose
output is three scalars. Wider host-returning joins have different costs.

Every session returned the same count and sums as the native SQL. The
disposable integration suite also tested repeated probe keys, NULL probe
keys, missing matches, empty dimensions, and snapshot cleanup after commit
and rollback. Existing CUDA aggregation and join integration suites passed.

## Reproduce

```sql
ALTER EXTENSION pg_gftr UPDATE TO '0.1.1';  -- for an existing installation
BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY;

EXPLAIN (ANALYZE, TIMING OFF)
SELECT pg_gftr_resident_load('lineitem', 'orders', 'customer');

SELECT pg_gftr_resident_run();

EXPLAIN (ANALYZE, TIMING OFF)
SELECT pg_gftr_resident_run();

COMMIT;
```

Native result check and comparison query:

```sql
SELECT ARRAY[count(*), sum(o.o_custkey), sum(c.c_nationkey)]
FROM lineitem l JOIN orders o ON l.l_orderkey = o.o_orderkey
JOIN customer c ON o.o_custkey = c.c_custkey;
```
