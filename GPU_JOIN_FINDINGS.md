# GPU join probes on TPC-H

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

## What to integrate first

1. Add a compact CUDA result path for low-match joins and retain dense
   results for high-match joins. Keep both under the existing GPU memory
   budget. Validate output multisets, NULL and duplicate probe keys, and
   performance across selectivities before setting a crossover.
2. Add GPU-to-GPU join chaining for eligible plans. Reuse the device
   lookup and pass needed columns or match indices directly into the next
   GPU join. Measure cold uploads and host-returning versus device-resident
   outputs separately.
3. Test bounded async staging if host input remains necessary. Measure it
   through PostgreSQL with the same SQL, then compare against the fastest
   native plan.

The earlier full SF5 selective SQL benchmark took 5.92 s with the CUDA
join and 2.59 s with the faster native baseline. A roughly 43 ms isolated
stage saving is useful, and the SQL baseline sets the larger target. Keep
GPU kernel microbenchmarks and query-level speedups separate. See
[JOIN_FINDINGS.md](JOIN_FINDINGS.md).

## Reproduce

```bash
nvcc -O3 -std=c++17 -arch=sm_80 -Isrc bench/join_gpu_probe.cu -o /tmp/join-gpu-probe
/tmp/join-gpu-probe /home/ubuntu/tpch/data/sf1 > sf1.csv
/tmp/join-gpu-probe /home/ubuntu/tpch/data/sf5 > sf5.csv
```
