# GPU joins and grouped aggregations in PostgreSQL

**Aritro Roy and Kiran Ghumare** · Final project report · November 2026

GPU grouped aggregation beat serial PostgreSQL on the focal workloads measured so far. SF5 A1 was 3.25× faster in GFTR, with GFUR already providing 3.16×. PostgreSQL's parallel plans beat CUDA on SF1 A2 and A3 and on SF5 A2. For ordinary order joins, tuned PostgreSQL won both the selective and broad comparisons. GFTR improved the wide SF5 GPU join benchmark by 1.20× versus partitioned GFUR, and reusing a loaded GPU snapshot paid off on the second SF5 execution of the fixed two-join workload.

This checkpoint covers completed trials. Additional SQL cases and the SF1 crossover repetition are in progress.

## Research question and implementation

We tested whether GPU execution accelerates joins and grouped aggregations in PostgreSQL, and whether transforming payloads improves performance relative to gathering from their original locations. Wu, Koutsoukos, and Alonso describe gather from transformed relations (GFTR) for join payloads in §4.1 and evaluate it in §5.1. They also apply transformed gathering to group-by and study hash, partition and sort-based strategies in §§4.2–4.4 and §5.2. [Paper, version 2](https://arxiv.org/html/2312.00720v2) · [Published DOI](https://doi.org/10.1145/3709689)

- **Grouped aggregation.** The extension packs PostgreSQL tuples into integer columns. Both paths use radix ordering, run-length encoding, a prefix scan and grouped reductions. GFUR sorts keys with row IDs, then indexes the original payloads. GFTR sorts keys with each payload column, then reduces the reordered values directly. This is a sort-based aggregation adaptation. Dictionary encoding, optimized hash grouping and partition-based grouping remain untested in this extension. [CUDA aggregation](src/gftr_cuda.cu)
- **Ordinary SQL join.** The CUDA custom scan implements a bounded integer inner join against a unique build side. It batches rows and can return dense matches or compact matching pairs. This tests selective transfer reduction independently of GFTR payload transformation. [PostgreSQL join integration](src/pg_gftr_join.c) · [CUDA join](src/gftr_join_cuda.cu)
- **Transformed join and data residency.** The standalone join benchmark applies an eight-bit radix partition to both inputs, then compares original and reordered build payloads using a dense lookup. It tests the transformed-payload principle with an adaptation of the paper's partitioned join. The resident SQL API instead keeps columns and intermediate keys on the GPU for a fixed two-join reduction within a read-only repeatable-read transaction. [CUDA benchmark](bench/join_resident_pipeline.cu) · [Resident SQL](src/pg_gftr_resident.c)

## Environment and measurement

- NVIDIA A100-SXM4-80GB, AMD EPYC host, NVIDIA driver 610.43.02, CUDA 12.6 (nvcc 12.6.85), PostgreSQL 18.6, extension 0.1.1, sm_80. The tested extension implementation is the one on `gpu-resident-sql` at `34085f6`. The report branch also contains benchmark instrumentation and drivers.
- The A100 datasets were generated and loaded on the host. SF1 and SF5 contain 6,001,215 and 29,999,795 lineitem rows. Both have the eight TPC-H tables and their primary-key indexes. Earlier RTX A6000 values come from the October 24 progress report. Differences between hosts, builds and workload reconstruction limit a direct hardware comparison.
- Full SQL timings use PostgreSQL server execution time. This includes execution-time scans, packing, transfers and result consumption inside the plan. Planning and normal client result delivery are outside that boundary. Cache dropping was disabled. CUDA timings identify whether data were already resident and whether they include allocation, upload and transformation. Ratios use baseline divided by candidate.
- The A100 verified the standalone CPU cases, 72 aggregation fixtures in each CUDA mode (144 GPU runs), CUDA join backend, and PostgreSQL integration for GFTR, GFUR, CUDA joins and the resident SQL API. The benchmark harness smoke test matched its checksums. These are controlled TPC-H workload measurements. The paper used an A100 40 GB and CUDA 12.2.

## Grouped aggregation in PostgreSQL

A1 groups lineitem by part, A2 by supplier, A3 by supplier nation after a native join, and A4 groups orders by customer. Each reconstruction uses COUNT(*) and one integer SUM. Q13 uses default dbgen query substitutions. Its inner count by customer runs on CUDA, while the customer–orders join and outer aggregate execute on the CPU.

The A100 runs used JIT off and 256 MB work_mem. GPU selection was forced for supported operators and GPU joins were disabled. We compared exact sorted result records and PostgreSQL output descriptions before timing. Every custom case used one timed warmup and five measured rounds per mode. Standard queries used three measured rounds after one warmup. Mode order was interleaved by a fixed seed. In the four-worker configuration, the planner selected the actual number of workers.

<!-- AGG SQL -->
| Scale | Case | Serial native (s) | Four-worker limit (s) | GFUR (s) | GFTR (s) | Serial / GFTR | Four-worker limit / GFTR | Workers launched |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SF1 | A1 | 2.454 [2.170–2.568] | 2.440 [1.925–2.625] | 1.035 [0.897–1.117] | 1.083 [0.888–1.151] | 2.27× | 2.25× | 0 |
| SF1 | A2 | 1.582 [1.443–1.742] | 0.356 [0.322–0.362] | 0.978 [0.822–1.222] | 1.047 [0.904–1.251] | 1.51× | 0.34× | 4 |
| SF1 | A3 | 2.365 [2.213–2.400] | 0.463 [0.448–0.470] | 1.818 [1.607–1.863] | 1.622 [1.574–1.801] | 1.46× | 0.29× | 4 |
| SF1 | A4 | 0.467 [0.416–0.536] | 0.486 [0.415–0.513] | 0.257 [0.252–0.313] | 0.263 [0.253–0.338] | 1.78× | 1.85× | 0 |
| SF1 | Q13 | 1.230 [1.212–1.254] | 1.263 [1.202–1.304] | 0.823 [0.803–0.890] | 0.834 [0.824–0.884] | 1.48× | 1.52× | 0 |
| SF5 | A1 | 18.016 [16.361–18.749] | 18.512 [17.278–20.425] | 5.700 [5.588–5.856] | 5.538 [5.466–5.563] | 3.25× | 3.34× | 0 |
| SF5 | A2 | 9.028 [8.208–9.069] | 1.901 [1.798–1.950] | 5.581 [5.063–5.804] | 5.547 [5.160–5.900] | 1.63× | 0.34× | 4 |
<!-- END AGG SQL -->

Brackets show the measured range. Each focal case has its own sampled plans and result signatures in `aggregation-sql/sf*/timed/`.

### October RTX A6000 reference

The October 24 progress report supplied these medians in seconds. The A100 workloads use the documented grouping keys and operator shapes. The original custom SQL, standard substitutions and SF5 subset list were unavailable, so the two tables provide a historical comparison by workload shape. They do not isolate a GPU model effect.

| Scale | Case | Serial PostgreSQL | GFUR | GFTR |
| --- | --- | ---: | ---: | ---: |
| SF1 | A1 | 2.840 | 1.344 | 1.359 |
| SF1 | A2 | 1.816 | 1.307 | 1.277 |
| SF1 | A3 | 2.580 | 2.093 | 2.128 |
| SF1 | A4 | 0.597 | 0.388 | 0.381 |
| SF1 | Q13 | 1.406 | — | 1.056 |
| SF5 | A1 | 18.079 | 6.818 | 6.804 |
| SF5 | A2 | 11.283 | 6.581 | 6.561 |
| SF5 | A3 | 14.900 | 11.475 | 11.491 |
| SF5 | A4 | 4.303 | 1.747 | 1.751 |
| SF5 | Q13 | 9.307 | — | 6.191 |

The earlier SF5 A1 claim is a 2.66× speedup over serial PostgreSQL. GFUR already achieved 2.65× under the October settings. In the October SF5 A2 stage timings, GFTR brought group discovery and reduction from 15.363 to 1.176 ms, while sorting increased from 6.149 to 15.661 ms. Full SQL went from 6.581 to 6.561 s, with overlapping trial ranges.

### SF5 A2 execution phases

The median group-discovery and reduction time in the A100 SQL run fell by **2.83×** with GFTR. The median sort time rose by 2.6%. These directions match October, with smaller differences on the A100 measurements. Total query medians differed by 34 ms, with overlapping ranges. The paired median GFUR minus GFTR difference was −44.5 ms, with a range of −318.4 to +244.8 ms. The SQL trials leave the faster GPU mode unresolved for this query. The much larger scan and packing phase accounted for about 94% of the GFTR median server execution time.

| Phase (ms) | GFUR | GFTR |
| --- | ---: | ---: |
| CPU scan and pack | 5,228.712 | 5,221.348 |
| Backend total | 338.874 | 331.446 |
| GPU sort | 23.972 | 24.603 |
| GPU group discovery and reduction | 2.657 | 0.938 |
| Transfer | 236.513 | 229.245 |
| Complete SQL | 5,581 | 5,547 |

The backend total contains the GPU and transfer stages. All figures are independent medians. The report JSON and phase summarizer are in the results archive.

### Device-resident aggregation

The synthetic benchmark uses six million shuffled keys, an explicit NULL group and one or four integer payload columns with about one NULL in 23 values. All input columns were uploaded before timing. The benchmark charges row-ID generation, sorting or payload transformation, group discovery, offsets and reduction. It validates all output keys, counts and payload statistics after the timed region. Results are medians in ms, with ranges from seven paired rounds.

| Groups | Payload columns | GFUR total | GFTR total | GFUR reduction | GFTR reduction | Paired GFUR/GFTR total ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 25 | 1 | 1.845 [1.839–1.847] | 1.467 [1.456–1.482] | 0.979 | 0.517 | 1.259× |
| 25 | 4 | 4.797 [4.791–4.831] | 5.227 [5.216–5.243] | 3.911 | 2.110 | 0.917× |
| 200,000 | 1 | 1.772 [1.649–1.836] | 1.724 [1.643–1.845] | 0.726 | 0.605 | 1.031× |
| 200,000 | 4 | 3.974 [3.529–3.997] | 6.106 [5.439–6.322] | 2.903 | 2.432 | 0.637× |

GFTR reduced the reduction-kernel median in all four cells. The strongest net gain was 1.26× for one payload at 25 groups. Trials went both ways for one payload at 200,000 groups. At four payloads the cost of sorting each payload outweighed the reduction savings at both group counts. The synthetic fixture supplies its known group count to keep the discovery-to-scan step on the GPU. Profiling executables were unavailable, so device memory traffic and L2 counters were unmeasured. Raw evidence is in `aggregation-operator/trials.csv`.

## Ordinary SQL joins

The join study used 256 MB work_mem, JIT off, a 131,072-row GPU probe batch, one warmup and five measured rounds. Serial native and the forced CUDA modes disallowed parallel workers. Tuned native allowed four workers per gather. Dense, compact and auto are output modes of the same forced CUDA operator. The auto comparison tests output representation selection after the operator is chosen. Result checks compared query outputs independently of timings.

<!-- JOIN SQL -->
| Scale | Case | Serial native (s) | Tuned native (s) | CUDA dense (s) | CUDA compact (s) | CUDA auto (s) | Native join + GFTR (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SF1 | orders_broad | 2.140 [2.088–2.244] | 0.618 [0.613–0.621] | 1.364 [1.345–1.400] | 1.358 [1.347–1.392] | 1.363 [1.332–1.379] | — |
| SF1 | orders_selective | 1.410 [1.354–1.468] | 0.141 [0.135–0.147] | 1.387 [1.314–1.410] | 1.317 [1.283–1.326] | 1.330 [1.285–1.364] | — |
| SF5 | orders_broad | 11.910 [11.174–12.233] | 3.149 [3.117–3.158] | 6.805 [6.697–6.981] | 6.883 [6.787–7.244] | 6.768 [6.625–7.006] | — |
| SF5 | orders_selective | 7.723 [6.138–7.768] | 1.502 [1.437–1.533] | 7.041 [6.896–7.249] | 6.981 [6.551–8.046] | 7.539 [6.630–7.849] | — |
<!-- END JOIN SQL -->

Brackets show the measured range. All five SQL shapes preserve their output descriptions across modes, as checked by `join-sql/join-types.json`.

- **Selective order join.** The SF5 match rate was 1.30%. Auto compacted 229 of 229 batches and reduced device-to-host match metadata from 119,999,184 to 3,109,680 bytes (97.4%). The device-to-host phase fell from 44.37 to 10.65 ms. Dense, compact and auto SQL ranges overlap. Tuned PostgreSQL used a four-worker hash join at SF5 and a three-worker nested loop with index access at SF1.
- **Broad order join.** Auto used dense output in all batches. At SF5, forcing compaction still used the dense fallback on all 229 batches. Median probe time increased from 3.63 to 10.73 ms and match output grew by 916 bytes of control metadata. Auto beat serial native by 1.76× but took 2.15× as long as tuned native.
- **Host work.** The custom scan packs input tuples, sends batches, then constructs PostgreSQL output slots on the host. The device-to-host byte counts cover match metadata. The SF5 selective auto plan recorded separate build and probe scan/pack medians of 1.27 s and 5.49 s. Raw plans and per-mode transfer counters are in `join-sql/`.

## Transformed join payloads and device residency

### Join operator

The `lineitem ⋈ part` benchmark uses actual TPC-H keys and one actual payload (`p_size`). The wide case adds seven deterministic integer payloads. Each cell contains the median and measured range of seven interleaved trials, in ms. Both partitioned variants include partition and lookup rebuild costs. GFTR also includes payload reordering. The harness checked all materialized values before timing.

| Scale | Payload columns | Direct lookup | Partitioned GFUR | Partitioned GFTR-style |
| --- | ---: | ---: | ---: | ---: |
| SF1 | 1 | 0.424 [0.416–0.444] | 1.001 [0.992–1.206] | 1.216 [1.187–1.255] |
| SF1 | 8 | 1.029 [1.017–1.050] | 1.571 [1.503–1.812] | 1.388 [1.344–1.481] |
| SF5 | 1 | 1.083 [1.078–1.100] | 2.277 [2.261–2.797] | 2.242 [2.227–2.279] |
| SF5 | 8 | 8.485 [8.463–8.496] | 5.414 [5.401–5.897] | 4.530 [4.512–4.613] |

At SF5 with eight payload columns, GFTR won by **1.20× over partitioned GFUR** and **1.87× over direct lookup**. Direct lookup won the other three cells. The microbenchmark starts with device-resident data and measures materialization in each variant's own probe order. Returning rows through PostgreSQL or restoring input order would add work. Raw evidence is in `join-residency/sf1.csv` and `sf5.csv`.

### Reusing GPU data across joins

The standalone `lineitem ⋈ orders ⋈ customer` bench checks the count and two sums. Median of seven runs, in ms:

| Scale | Resident intermediates | Pinned host intermediate round trip | Prepacked input upload and GPU build |
| --- | ---: | ---: | ---: |
| SF1 | 1.103 | 29.252 | 17.313 |
| SF5 | 2.313 | 78.314 | 27.668 |

The round-trip control moves the intermediate customer keys to the host and back. At SF5 it moves about 240 MB. The fully resident variant returns 24 bytes. The upload variant starts with packed host columns. It includes uploads, lookup construction, both joins and the reduction. Input decoding, PostgreSQL, allocations and CUDA context creation are outside these measurements. The SF5 host round-trip varied from 69.44 to 252.30 ms across the seven trials. Raw evidence is in the same files as the gather study.

### Resident SQL API

The extension provides an explicit load and a fixed SQL function that returns three scalars. The native query computes the same count and sums directly from the three PostgreSQL tables. Both paths used 256 MB work_mem and JIT off. The tuned native plan launched four workers. Each native cell shows three measured queries after one warmup. Each resident cell uses three fresh sessions and seven warm calls per session. Brackets show the measured range.

| Scale | Serial native (s) | Tuned native (s) | Load + first GPU call (s) | First GPU call after load (ms) | Later GPU calls (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| SF1 | 3.300 [3.152–3.313] | 0.862 [0.857–0.866] | 1.691 [1.674–1.741] | 1.842 | 0.274 [0.254–0.304] |
| SF5 | 20.027 [19.898–20.607] | 4.910 [4.800–4.932] | 6.862 [6.686–6.932] | 5.249 | 0.580 [0.553–0.663] |

A cold GPU snapshot lost to tuned native in both scales. The median-based model predicts a crossover at two executions. SF5 confirmed it in the paired series. At one execution the resident path took 6.851 s versus 4.914 s tuned native. At two it took 6.939 s versus 9.889 s. The initial SF1 pair at two executions took 1.732 s resident versus 1.660 s tuned native, so small timing shifts can reverse the result near the modeled crossover.

<!-- SF1 CROSSOVER -->

All these sums use server execution time and include the resident loader. The scope is one explicit snapshot and one fixed query. PostgreSQL's ordinary planner uses the separate SQL join path above. Raw evidence is in `join-residency/*-summary.json` and its accompanying SQL and session logs.

## Interpretation and follow-up

GFTR changes memory access after ordering. Aggregation still pays for sorting each transformed payload and for PostgreSQL tuple conversion. For join outputs, widening the payload makes the extra transformation worthwhile in the observed SF5 device-resident case. Retaining inputs and intermediates on the GPU avoids much larger copies in a narrow chain. The first SQL execution includes scanning, packing, upload and lookup construction.

Earlier A100 analysis of the persisting L2 policy saw a broad resident join probe drop from 0.436 to 0.289 ms, while the filtered hash lookup stayed at 0.408 ms. These are kernel timings from an earlier run, with a smaller absolute effect than eliminating intermediate transfers in the current study. We did not measure CPU cache misses. CPU-side scan and packing time cannot by itself identify CPU cache behavior. See `GPU_JOIN_FINDINGS.md` at `79cd3fd`.

Separately, the earlier CPU-only fused supplier join and aggregate measured 1.078 s at SF5 against 2.489 s for its four-worker native join and aggregate and 1.990 s for its fact preaggregation rewrite. Those figures belong to the earlier `fused-joins` experiment at `629db5d`. They are reported as CPU results and use that experiment's own native controls.

These measurements support choosing a strategy based on plan shape, payload width, expected matches and whether data are already on the device. For PostgreSQL ingestion, reducing tuple conversion and host/device traffic has the clearest opportunity shown here. For selective SQL, compare against the index and parallel plans the optimizer can use. For the aggregation kernel, a wider transformed-payload workload should charge the additional sorts that it introduces.

## Reproduce the A100 experiments

Run from the repository root with PostgreSQL 18.6, CUDA 12.6, the local `tpch_sf1` and `tpch_sf5` databases, and dbgen's PostgreSQL `qgen` from revision `32f1c1b`. The validation job installs the CUDA build in the PostgreSQL installation selected by `PG_CONFIG`. Run timings serially on an otherwise idle host. Here are commands to reproduce one focal aggregation and one join.

```bash
source /home/ubuntu/pg-flux-setup/env.sh
bash scripts/verify_a100.sh

python3 bench/aggregation_tpch.py --qgen /tmp/flux-report-qgen/qgen \
  --scales 5 --cases A2 --output results/reproduce/aggregation

python3 bench/join_benchmark.py --dbname tpch_sf5 --cases orders_selective \
  --modes native native_tuned cuda_dense cuda_compact cuda_auto \
  --match-work-mem --tuned-work-mem-mb 256 --tuned-workers 4 --disable-jit \
  --warmups 1 --trials 5 --output results/reproduce/join
```

The raw-results archive contains an index and SHA-256 manifest. Its directory names appear in the report and contain the exact SQL, run metadata, complete plans and trial times. The scripts under `aggregation-operator/` and `join-residency/` reproduce the CUDA experiments.

| Archive directory | Evidence |
| --- | --- |
| `verification/`, `verification.log`, `environment.txt`, `sf*-load.log` | A100 verification, host settings, indexes and data load counts |
| `aggregation-sql/` | Workload provenance, standard plan screening, per-case exact checks, timings and SQL |
| `aggregation-operator/` | Synthetic resident aggregation ablation, measurement method and compile/run command |
| `join-sql/` | Matched-settings SQL plans, exact checks, batch statistics and transfer bytes |
| `join-residency/` | Resident CUDA trials, same-snapshot checks, SQL plans and crossover trials |
