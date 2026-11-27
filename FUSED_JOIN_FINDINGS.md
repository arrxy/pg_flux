# Fused supplier join and grouped aggregation

The CPU experiment targets an inner integer equijoin with a proven unique
build key, an integer group column from the build side, and `COUNT(*)` with
optional `SUM(integer)` from the probe side. PostgreSQL child plans still
apply filters with normal snapshot semantics.

The executor maps each build key directly to a compact group index. It
streams projected probe slots through the lookup and updates count and sum in
place. Its output consists of group slots, so the supplier query on SF5 can
consume 29,999,795 line items and emit 25 rows. The path checks accumulator
overflow and builds under the existing host memory budget. It accounts for
nullable keys, nullable groups and all-NULL sums.

## Measurement protocol

The server is PostgreSQL 18.6 on the eight-vCPU AMD EPYC host used for the
original join experiments. All timings are median PostgreSQL
`EXPLAIN ANALYZE` execution time. Each scale uses a single repeatable-read
snapshot, an untimed result verification pass, one warmup, and three
randomized timed rounds. Reported worker counts are the workers observed in
the plans. Native preaggregation groups line items by supplier before
joining to the supplier table and keeps the same result types and values.

```bash
python3 bench/join_benchmark.py --dbname tpch_sf1 tpch_sf5 \
  --cases supplier_grouped --modes native native_preagg fused_cpu \
  --native-worker-sweep 0 2 4 --trials 3 --warmups 1 --output results/fused-main
```
