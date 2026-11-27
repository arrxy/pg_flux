# GPU-resident TPC-H join experiments

The experiments ran on the NVIDIA A100 with CUDA 12.6 and actual SF1/SF5
TPC-H keys. CUDA timings are medians of seven interleaved measured trials
after one warmup per mode. Native PostgreSQL timings use three interleaved
measured trials after one warmup per configuration.

## Two CUDA joins

The pipeline joins `lineitem → orders → customer`. The first join emits
customer keys on the device. The second uses those keys to look up nation
keys and reduces its results on the device to three scalar values.
Every GPU trial agreed with the host reference, and native PostgreSQL
returned the same counts and sums.

| Scale | GPU resident | GPU with intermediate round trip | GPU with input upload and build |
| --- | ---: | ---: | ---: |
| SF1 | 1.13 ms | 32.96 ms | 18.40 ms |
| SF5 | 2.26 ms | 76.87 ms | 27.73 ms |

The round-trip control copies the intermediate customer keys into pinned
host memory and back before launching the second GPU join. At SF5 that
moves about 240 MB across PCIe. The resident pipeline transfers only the
24-byte final result per execution. The upload-and-build variant starts
with projected columns in host memory. It includes their initial upload,
GPU lookup construction, both joins, and reduction. CUDA context creation,
buffer allocations, input decoding, and PostgreSQL execution are outside
those timings. Reading and projecting all files for both experiments took
1.08 s at SF1 and 4.85 s at SF5 in this run.

### Native PostgreSQL reference

| Scale | Default PostgreSQL | 256 MiB work_mem and four-worker limit |
| --- | ---: | ---: |
| SF1 | 1.435 s | 0.886 s |
| SF5 | 8.350 s | 4.957 s |

The native measurements time the full SQL statement with PostgreSQL
table scans and joins. The resident GPU benchmark starts from an
explicitly loaded snapshot. A PostgreSQL custom path that reuses that
snapshot is the next experiment needed to compare SQL execution times.
For a first integrated path, keep the current dense lookups for the two
narrow joins and carry the intermediate keys across on the device.
Consider the transformed gather for wider outputs based on measured
partition-plus-gather cost.

## Partitioned gather

For `lineitem ⋈ part`, the direct CUDA baseline probes a dense lookup and
gathers payloads from original build rows. Both partitioned variants radix
partition the build and probe keys on eight low bits and rebuild the dense
index. The GFUR variant gathers original payloads by row ID. The GFTR-style
variant applies the build permutation to the payload columns before
gathering. The reported times charge partitioning and payload
transformation. All materialized values were checked in warmup. The
one-column case uses actual `p_size`. The eight-column case adds seven
deterministic integer payload columns derived from the part key.

| Scale | Payload columns | Direct lookup and gather | Partitioned GFUR | Partitioned GFTR-style |
| --- | ---: | ---: | ---: | ---: |
| SF1 | 1 | 0.44 ms | 1.04 ms | 1.28 ms |
| SF1 | 8 | 1.05 ms | 1.59 ms | 1.47 ms |
| SF5 | 1 | 1.12 ms | 2.37 ms | 2.33 ms |
| SF5 | 8 | 8.45 ms | 5.40 ms | 4.52 ms |

At SF5 with eight payloads, the transformed gather kernel took 2.80 ms
versus 4.07 ms for partitioned GFUR. Including all measured work, the
GFTR-style variant was 1.87× as fast as the direct lookup and 1.19× as fast
as partitioned GFUR. The direct lookup led in the other three cases.
This ablation uses a global dense lookup and a shared build permutation,
so it tests the paper's transformed-payload principle in the extension's
setting. The paper's complete partitioned hash join is a separate algorithm.
[GFTR paper, §§4 and 5.1](https://arxiv.org/html/2312.00720v2)

## Reproduce

```bash
nvcc -O3 -std=c++17 -arch=sm_80 bench/join_resident_pipeline.cu -o /tmp/join-resident-pipeline
/tmp/join-resident-pipeline /home/ubuntu/tpch/data/sf1 > sf1.csv
/tmp/join-resident-pipeline /home/ubuntu/tpch/data/sf5 > sf5.csv
```

The CSV has an `expected` row with the three scalar results. Pipeline rows
then give variant, trial, fact rows, input load, first join, intermediate
transfer, second join plus reduction, and wall milliseconds. Gather rows
give variant, trial, width, partition, transform, gather, and wall
milliseconds.

```sql
SELECT count(*), sum(o.o_custkey), sum(c.c_nationkey)
FROM lineitem l JOIN orders o ON l.l_orderkey = o.o_orderkey
JOIN customer c ON o.o_custkey = c.c_custkey;
```
