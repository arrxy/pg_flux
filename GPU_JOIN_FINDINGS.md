# CUDA join findings

## PostgreSQL integration

Branch `gpu-opt` starts from `main` at `6164581`. It adds the CUDA join
operator and selectivity-aware match compaction. The runtime join backend is
CUDA. The CPU join reference is used by standalone tests.

`pg_gftr.join_result_mode=auto` requests compaction below an estimated 10%
match rate. The GPU retains dense indices and stages at most one compact
pair per eight configured batch rows, within the host and device budgets.
A batch can use dense output when actual cardinality exceeds the compact
capacity or the compact transfer would be larger. Auto leaves the broad
query on the dense path in these measurements. The join operator stays
opt-in.

### Full query timings

Measured on PostgreSQL 18.6 and an A100 with CUDA 12.6. The benchmark used
one warmup and three randomized measured rounds per scale, 131,072 probe
rows per batch, and a repeatable-read snapshot. Times are median
`EXPLAIN ANALYZE` execution times. CUDA paths were forced for the A/B test.
Native tuned uses 256 MiB `work_mem`. All query result digests matched.

| Scale | Query | Native default | Native tuned | CUDA dense | CUDA auto |
| --- | --- | ---: | ---: | ---: | ---: |
| SF1 | January orders | 0.171 s | 0.178 s | 1.315 s | 1.057 s |
| SF1 | All orders | 1.029 s | 0.904 s | 1.350 s | 1.321 s |
| SF5 | January orders | 2.491 s | 2.468 s | 6.231 s | 5.942 s |
| SF5 | All orders | 5.137 s | 4.961 s | 6.994 s | 7.391 s |

Auto compacted all 46 selective batches on SF1 and all 229 on SF5. On
SF5, it reduced device-to-host output from 119,999,184 to 3,109,680 bytes
including control counts, a 97.4% reduction. The selective full-query
median fell by 4.6% on SF5. Three trials had overlapping wall times and
substantial scan-and-pack and pageable-transfer variation. The transfer
volume reduction is reproducible across the trials. The elapsed-time
difference needs more repetitions to isolate. Native PostgreSQL remains
faster than the forced CUDA plans.

Both SF5 broad CUDA modes used 229 dense batches and transferred the same
bytes. Their elapsed medians show run-to-run variation for the same kernel
and result representation. CUDA backend and PostgreSQL integration tests
covered sparse and dense keys, NULL probes, a partial warp, zero matches,
overflow of the compact capacity, and dense fallback.

Reproduce from `gpu-opt` with the extension installed on the A100 host.

```bash
python3 bench/join_benchmark.py --dbname tpch_sf1 tpch_sf5 \
  --cases orders_selective orders_broad \
  --modes native native_tuned cuda_dense cuda_auto \
  --trials 3 --warmups 1 --output results/gpu-opt
```

## Earlier standalone probe

The standalone benchmark projects the order keys from the actual generated
SF1 and SF5 `orders.tbl` and `lineitem.tbl` files. It runs a unique-build
integer equijoin for all orders and for orders dated January 1995. The CUDA
lookup uses the extension's dense or hash layout rule. Each case has one
warmup and seven interleaved measured trials on the A100. Every returned
match index, including the probe index of each compacted match, is checked
against a host reference. Native PostgreSQL independently confirmed the
match counts.

The probe measures prepacked keys and a prebuilt GPU lookup. The return
variants include one pinned host-to-device upload of the probe keys and
return the result to pinned host memory. The resident variants warm the
lookup and time one GPU probe with input and output on the device. Bench
timing excludes file parsing, lookup build and upload, host staging,
PostgreSQL tuple execution, and reference comparison. The standalone trial
uses one large CUDA launch, so its timings are distinct from the extension's
batched PostgreSQL operator.

## Output compaction

| Scale | Workload | Matches | Dense output | Compacted output | Dense wall | Compact wall |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| SF1 | All orders | 6,001,215 | 24.00 MB | 48.01 MB | 27.13 ms | 47.32 ms |
| SF1 | January orders | 77,914 | 24.00 MB | 0.62 MB | 23.54 ms | 4.92 ms |
| SF5 | All orders | 29,999,795 | 120.00 MB | 240.00 MB | 59.95 ms | 100.79 ms |
| SF5 | January orders | 388,595 | 120.00 MB | 3.11 MB | 57.57 ms | 14.66 ms |

The dense representation returns one 32-bit build index or miss sentinel
per probe. The GPU compactor uses one atomic reservation per warp with hits
and emits a pair of 32-bit probe and build indices for each match. The
selective SF5 return fell from 46.05 ms to 0.87 ms of measured D2H time.
Wall time for the transfer-inclusive probe stage fell by 3.93×. The
all-orders case is nearly all hits, so its compact representation doubles
return bytes. The 8-byte pair format breaks even on bytes near 50% hits.
The performance threshold needs more selectivity measurements because
atomic contention, synchronization and PostgreSQL output reconstruction
also matter.

## Resident probing and L2

| Scale | Workload | Normal probe kernel | Persisting L2 probe kernel |
| --- | --- | ---: | ---: |
| SF1 | All orders | 0.099 ms | 0.062 ms |
| SF1 | January orders | 0.084 ms | 0.084 ms |
| SF5 | All orders | 0.436 ms | 0.289 ms |
| SF5 | January orders | 0.408 ms | 0.408 ms |

These are CUDA event medians. The 120 MB SF5 dense lookup saw a small
absolute latency reduction on repeated resident probes. The 2 MiB SF5
filtered hash lookup showed no material change. L2 policy is a lower
priority than cutting materialization and transfer volume.

The standalone run motivated the compact CUDA result path. Device-resident
join chaining is a potential next experiment when an adjacent CUDA
operator can consume the result directly.

### Reproduce from `gpu-join-probes`

```bash
nvcc -O3 -std=c++17 -arch=sm_80 -Isrc bench/join_gpu_probe.cu -o /tmp/join-gpu-probe
/tmp/join-gpu-probe /home/ubuntu/tpch/data/sf1 > sf1.csv
/tmp/join-gpu-probe /home/ubuntu/tpch/data/sf5 > sf5.csv
```
