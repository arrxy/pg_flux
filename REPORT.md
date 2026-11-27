# GPU joins and grouped aggregations in PostgreSQL

**Aritro Roy and Kiran Ghumare** · Final project report · November 2026

GPU grouped aggregation beat serial PostgreSQL in all ten focal comparisons. SF5 A1 was 3.25× faster in GFTR, with GFUR already providing 3.16×. PostgreSQL's parallel plans beat CUDA on A2 and A3 at both scales. For ordinary order joins, tuned PostgreSQL won both the selective and broad comparisons. GFTR improved the wide SF5 GPU join benchmark by 1.20× versus partitioned GFUR, and reusing a loaded GPU snapshot paid off on the second SF5 execution of the fixed two-join workload.

The crossover model selected tuned PostgreSQL in all eight held-out joins and estimated CUDA latency with 11.42% mean absolute relative error. PG-Strom selected GPU plans but failed during execution on this host. Its planning evidence and the exact-verified controls are included.

## Research question and implementation

We tested whether GPU execution accelerates joins and grouped aggregations in PostgreSQL, and whether transforming payloads improves performance relative to gathering from their original locations. Wu, Koutsoukos, and Alonso describe gather from transformed relations (GFTR) for join payloads in §4.1 and evaluate it in §5.1. They also apply transformed gathering to group-by and study hash, partition and sort-based strategies in §§4.2–4.4 and §5.2. [Paper, version 2](https://arxiv.org/html/2312.00720v2) · [Published DOI](https://doi.org/10.1145/3709689)

- **Grouped aggregation.** The extension packs PostgreSQL tuples into integer columns. Both paths use radix ordering, run-length encoding, a prefix scan and grouped reductions. GFUR sorts keys with row IDs, then indexes the original payloads. GFTR sorts keys with each payload column, then reduces the reordered values directly. This is a sort-based aggregation adaptation. Dictionary encoding, optimized hash grouping and partition-based grouping remain untested in this extension. [CUDA aggregation](src/gftr_cuda.cu)
- **Ordinary SQL join.** The CUDA custom scan implements a bounded integer inner join against a unique build side. It batches rows and can return dense matches or compact matching pairs. This tests selective transfer reduction independently of GFTR payload transformation. [PostgreSQL join integration](src/pg_gftr_join.c) · [CUDA join](src/gftr_join_cuda.cu)
- **Transformed join and data residency.** The standalone join benchmark applies an eight-bit radix partition to both inputs, then compares original and reordered build payloads using a dense lookup. It tests the transformed-payload principle with an adaptation of the paper's partitioned join. The resident SQL API instead keeps columns and intermediate keys on the GPU for a fixed two-join reduction within a read-only repeatable-read transaction. [CUDA benchmark](bench/join_resident_pipeline.cu) · [Resident SQL](src/pg_gftr_resident.c)

## Environment and measurement

- NVIDIA A100-SXM4-80GB, AMD EPYC host, NVIDIA driver 610.43.02, CUDA 12.6 (nvcc 12.6.85), PostgreSQL 18.6, extension 0.1.1, sm_80. The tested extension implementation is the one on `gpu-resident-sql` at `34085f6`. The report branch also contains benchmark drivers.
- The A100 datasets were generated and loaded on the host. SF1 and SF5 contain 6,001,215 and 29,999,795 lineitem rows. Both have the eight TPC-H tables and their primary-key indexes. Earlier RTX A6000 values come from the October 24 progress report. Differences between hosts, builds and workload reconstruction limit a direct hardware comparison.
- Full SQL timings use PostgreSQL server execution time. This includes execution-time scans, packing, transfers and result consumption inside the plan. Planning and normal client result delivery are outside that boundary. The OS and PostgreSQL caches were left warm between runs. The SF1 Q2 and SF5 part-join runs may overlap host-only compiler activity during PG-Strom setup. CUDA timings identify whether data were already resident and whether they include allocation, upload and transformation. Ratios use baseline divided by candidate.
- The A100 verified the standalone CPU cases, 72 aggregation fixtures in each CUDA mode (144 GPU runs), CUDA join backend, and PostgreSQL integration for GFTR, GFUR, CUDA joins and the resident SQL API. The benchmark harness smoke test matched its checksums. These are controlled TPC-H workload measurements. The paper used an A100 40 GB and CUDA 12.2.

## Grouped aggregation in PostgreSQL

A1 groups lineitem by part, A2 by supplier, A3 by supplier nation after a native join, and A4 groups orders by customer. Each reconstruction uses COUNT(*) and one integer SUM. Q13 uses default dbgen query substitutions. Its inner count by customer runs on CUDA, while the customer–orders join and outer aggregate execute on the CPU.

The A100 runs used JIT off and 256 MB work_mem. GPU selection was forced for supported operators and GPU joins were disabled. We compared exact sorted result records and PostgreSQL output descriptions before timing. Every custom case used one timed warmup and five measured rounds per mode. Standard queries used three measured rounds after one warmup. Mode order was interleaved by a fixed seed. In the four-worker configuration, the planner selected the actual number of workers.

| Scale | Case | Serial native (s) | Four-worker limit (s) | GFUR (s) | GFTR (s) | Serial / GFTR | Four-worker limit / GFTR | Workers launched |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SF1 | A1 | 2.454 [2.170–2.568] | 2.440 [1.925–2.625] | 1.035 [0.897–1.117] | 1.083 [0.888–1.151] | 2.27× | 2.25× | 0 |
| SF1 | A2 | 1.582 [1.443–1.742] | 0.356 [0.322–0.362] | 0.978 [0.822–1.222] | 1.047 [0.904–1.251] | 1.51× | 0.34× | 4 |
| SF1 | A3 | 2.365 [2.213–2.400] | 0.463 [0.448–0.470] | 1.818 [1.607–1.863] | 1.622 [1.574–1.801] | 1.46× | 0.29× | 4 |
| SF1 | A4 | 0.467 [0.416–0.536] | 0.486 [0.415–0.513] | 0.257 [0.252–0.313] | 0.263 [0.253–0.338] | 1.78× | 1.85× | 0 |
| SF1 | Q13 | 1.230 [1.212–1.254] | 1.263 [1.202–1.304] | 0.823 [0.803–0.890] | 0.834 [0.824–0.884] | 1.48× | 1.52× | 0 |
| SF5 | A1 | 18.016 [16.361–18.749] | 18.512 [17.278–20.425] | 5.700 [5.588–5.856] | 5.538 [5.466–5.563] | 3.25× | 3.34× | 0 |
| SF5 | A2 | 9.028 [8.208–9.069] | 1.901 [1.798–1.950] | 5.581 [5.063–5.804] | 5.547 [5.160–5.900] | 1.63× | 0.34× | 4 |
| SF5 | A3 | 12.017 [11.977–12.470] | 2.389 [2.352–2.437] | 8.138 [8.001–9.095] | 8.322 [8.189–9.054] | 1.44× | 0.29× | 4 |
| SF5 | A4 | 4.567 [3.818–4.754] | 4.465 [3.781–4.987] | 1.420 [1.362–1.466] | 1.416 [1.311–1.488] | 3.22× | 3.15× | 0 |
| SF5 | Q13 | 10.132 [10.068–10.517] | 10.378 [9.900–12.056] | 6.354 [6.225–7.829] | 7.422 [6.613–7.893] | 1.37× | 1.40× | 0 |

Brackets show the measured range. Each focal case has its own sampled plans and result signatures in `aggregation-sql/sf*/timed/`. GFTR beat serial native in ten of ten cases. A2 and A3 were the only cases that launched native workers, and those plans beat both GPU modes. The four-worker setting left A1, A4 and Q13 serial at both scales. The sampled native focal plans reported zero temporary blocks written.

### Standard query coverage

All 22 standard shapes were screened at each scale. Q13 was the only plan showing the CUDA aggregate, and its executed CUDA plans passed the exact check. The SF1 timings cover 20 standard queries. Q17 and Q20 hit the five-minute statement limit during native result validation. Their plans contain correlated sequential scans of lineitem with the loaded primary-key indexes. The failed SQL and error logs are included.

The October report does not identify its ten timed SF5 standard queries. We selected Q1, Q3, Q4, Q5, Q6, Q9, Q12, Q13, Q14 and Q19 for this study. All ten selected SF5 queries completed with matching results and output types. Timing differences in plans that stayed native are evidence about variation on the host and do not measure accelerator throughput.

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

| Scale | Case | Serial native (s) | Tuned native (s) | CUDA dense (s) | CUDA compact (s) | CUDA auto (s) | Native join + GFTR (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SF1 | orders_broad | 2.140 [2.088–2.244] | 0.618 [0.613–0.621] | 1.364 [1.345–1.400] | 1.358 [1.347–1.392] | 1.363 [1.332–1.379] | — |
| SF1 | orders_selective | 1.410 [1.354–1.468] | 0.141 [0.135–0.147] | 1.387 [1.314–1.410] | 1.317 [1.283–1.326] | 1.330 [1.285–1.364] | — |
| SF1 | part_payload | 2.252 [2.176–2.309] | 0.533 [0.516–0.545] | 1.960 [1.853–1.989] | 1.988 [1.856–2.041] | 1.945 [1.926–2.001] | — |
| SF1 | supplier_grouped | 2.265 [2.103–2.381] | 0.467 [0.448–0.478] | 2.192 [2.144–2.307] | 2.268 [2.063–2.363] | 2.221 [2.047–2.246] | 1.700 [1.622–1.796] |
| SF1 | supplier_projected | 0.038 [0.034–0.042] | 0.038 [0.033–0.042] | 0.027 [0.027–0.029] | 0.029 [0.028–0.031] | 0.027 [0.027–0.029] | — |
| SF5 | orders_broad | 11.910 [11.174–12.233] | 3.149 [3.117–3.158] | 6.805 [6.697–6.981] | 6.883 [6.787–7.244] | 6.768 [6.625–7.006] | — |
| SF5 | orders_selective | 7.723 [6.138–7.768] | 1.502 [1.437–1.533] | 7.041 [6.896–7.249] | 6.981 [6.551–8.046] | 7.539 [6.630–7.849] | — |
| SF5 | part_payload | 21.039 [18.854–22.716] | 4.708 [4.598–4.802] | 10.380 [9.760–12.622] | 11.341 [10.504–12.503] | 10.812 [10.336–12.413] | — |
| SF5 | supplier_grouped | 11.971 [10.945–12.100] | 2.461 [2.380–2.535] | 11.193 [10.878–11.728] | 11.425 [10.819–11.721] | 11.401 [10.927–11.644] | 9.563 [9.437–9.831] |
| SF5 | supplier_projected | 0.037 [0.035–0.038] | 0.036 [0.035–0.036] | 0.038 [0.037–0.038] | 0.040 [0.039–0.041] | 0.037 [0.037–0.038] | — |

Brackets show the measured range. All five SQL shapes preserve their output descriptions across modes, as checked by `join-sql/join-types.json`.

- **Selective order join.** The SF5 match rate was 1.30%. Auto compacted 229 of 229 batches and reduced device-to-host match metadata from 119,999,184 to 3,109,680 bytes (97.4%). The device-to-host phase fell from 44.37 to 10.65 ms. Dense, compact and auto SQL ranges overlap. Tuned PostgreSQL used a four-worker hash join at SF5 and a three-worker nested loop with index access at SF1.
- **Broad order join.** Auto used dense output in all batches. At SF5, forcing compaction still used the dense fallback on all 229 batches. Median probe time increased from 3.63 to 10.73 ms and match output grew by 916 bytes of control metadata. Auto beat serial native by 1.76× but took 2.15× as long as tuned native.
- **Bounded supplier projection at SF1.** CUDA dense took 27.30 ms versus 38.17 ms for tuned native (1.40×), with disjoint measured ranges. Both used the lineitem primary-key range scan. The CUDA plan built a lookup from 10,000 suppliers and matched all 40,278 probe rows. The native plan used a nested loop with memoized supplier index lookups and launched zero workers. At SF5, tuned native took 35.64 ms versus 37.35 ms for the fastest CUDA mode.
- **Joined aggregation.** At SF5 the supplier-nation workload took 9.56 s with a native join and GFTR aggregate, 11.19 s with the fastest forced CUDA join, and 2.46 s with the four-worker native plan. Native plans won nine of the ten comparisons against the fastest ordinary CUDA join mode.
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

The follow-up at SF1 favored the resident path in three more paired two-execution trials, by 8–101 ms. Across the four pairs, two executions were sensitive to run variation. At three executions, the resident path won all three follow-up pairs. It took 1.648–1.698 s versus 2.605–2.656 s tuned native. Both datasets also passed a direct equality check in the same repeatable-read transaction.

All these sums use server execution time and include the resident loader. The scope is one explicit snapshot and one fixed query. PostgreSQL's ordinary planner uses the separate SQL join path above. Raw evidence is in `join-residency/*-summary.json` and its accompanying SQL and session logs.

## When GPU offload pays

### Selectivity and width

We calibrated a phase model on eight SF1 lineitem–part joins, then tested it on eight different lineitem–supplier joins. The queries use the same lineitem order-key range and filter the dimension's integer primary key. Target match fractions are 1%, 5%, 50% and 100%, with two SQL projection widths. The wide query consumes additional lineitem columns. Both widths return a handful of scalar aggregates, so the results concern host tuple processing as well as GPU match metadata transfers. The separate resident join ablation above measures the cost of gathering actual multi-column device payloads.

The calibration matched about 2,000 to 200,000 probe rows. Between the narrow 1% and 100% cases, return traffic changed from 15,764 to 801,460 bytes while probe scan and pack stayed near 49–51 ms. CUDA auto took 64.66 and 93.04 ms versus 23.97 and 68.42 ms for tuned PostgreSQL. At 100% matches, broadening the projection increased the CUDA median from 93.04 to 115.28 ms with unchanged match metadata bytes. The measured host residual increased from 17.93 to 36.83 ms. This residual includes uninstrumented execution above and inside the join.

The model sums predicted scan and packing, GPU build and probe, host-to-device and device-to-host transfers, and a fitted host residual. It selects compaction below an estimated 10% match rate. Held-out features use the probe size from calibration, the dimension key range and a uniform match estimate. The measured tuned native time supplies the comparison threshold. We wrote the model output and input digests before launching the held-out CUDA tests. This measures transfer across query shapes with a known native runtime.

The frozen model predicted the native winner and the CUDA output mode in all eight held-out cases. Every result digest matched across execution modes. In all eight cases, probe scan and packing alone took 42.50–47.73 ms and exceeded the tuned native query time. Measured transfers totaled 0.65–0.95 ms. CUDA predictions were 3.75–10.73 ms high, with a mean absolute relative error of 11.42%. The measured ranges are based on three rounds after a warmup. All eight validation cells favored native, so this accuracy measures one outcome class. The ordinary SQL matrix above includes a separate GPU win outside this grid.

| Actual matches | Projected columns | Predicted CUDA (ms) | Measured CUDA (ms) | Tuned native (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 0.96% | 1 | 63.73 | 57.04 [53.07–59.71] | 20.78 [20.65–20.95] |
| 0.96% | 4 | 66.24 | 62.49 [56.18–64.12] | 20.75 [19.90–20.90] |
| 4.94% | 1 | 64.27 | 60.16 [55.92–60.99] | 21.02 [20.58–22.19] |
| 4.94% | 4 | 67.52 | 61.55 [60.03–63.68] | 21.78 [21.53–22.07] |
| 49.95% | 1 | 70.14 | 60.56 [58.63–65.98] | 24.17 [23.99–25.54] |
| 49.95% | 4 | 81.79 | 72.32 [71.44–73.02] | 25.65 [25.38–26.79] |
| 100.00% | 1 | 76.64 | 66.17 [64.42–66.19] | 27.41 [26.88–27.84] |
| 100.00% | 4 | 97.61 | 86.88 [84.87–90.81] | 29.21 [29.19–30.08] |

### Snapshot reuse

For the fixed resident API, the prediction compares the one-time load and first call plus one warm cost per subsequent call against repeated tuned native execution. The predicted threshold was two calls at both scales. The measured SF5 pair crossed at two. At SF1 two calls went both ways and three calls won all follow-up pairs. These are reuse tests of one query shape. The model and outputs are in `offload-crossover/`, including the supplementary SQL plans and the measured snapshot series.

## PG-Strom comparison on the same A100

We built PG-Strom 6.1.0 at `4d12ef415759d` against a separate copy of the same PostgreSQL 18.6 installation and loaded the same SF1 tables and primary-key indexes into a disposable cluster. The comparison harness runs identical SQL in tuned native PostgreSQL, `pg_flux` and PG-Strom. It retains the four-worker native baseline, validates result multisets and output descriptions, and saves JSON plans. The PG-Strom service loaded an A100 fatbin for compute capability 8.0. [PG-Strom installation documentation](https://heterodb.github.io/pg-strom/install/)

Default PG-Strom planning selected GPU nodes for all nine screened workloads, without adjusting its planner costs. A1, A2 and A4 selected `GpuPreAgg`. A3, the grouped supplier join, the two order joins and the part payload join fused join work into `GpuPreAgg`. The bounded supplier projection selected `GpuJoin`. The operators can therefore cover different portions of the same SQL query.

**The PG-Strom speed comparison is unavailable on this runtime.** When executing A1, the unmodified release repeatedly logged `recv(2): Bad address` in its GPU service and failed to return the result. The GPU-service probe did not reproduce the receive error in isolation. A diagnostic build staged socket receives in host memory and removed that error, but its A1 query timed out. With the service ready, the diagnostic build's bounded `GpuJoin` reported a `cuMemAdvise` policy failure and an out-of-memory message. The A100 runtime reported `concurrentManagedAccess=0`. The data establish an execution failure on this host. Its underlying cause remains undetermined.

The preserved PG-Strom plans are planning-only. An early smoke query completed on CPU fallback while the service was starting. It supplies no PG-Strom GPU latency. The verified ordinary native and `pg_flux` executions and plans are in the preceding SQL sections.

We ran four exact-verified controls in the same disposable cluster with PG-Strom disabled. Medians from three measured rounds after one warmup are in ms. The last column shows the release's planning-only GPU operator.

| SF1 query | Tuned native | pg_flux | PG-Strom planned operator |
| --- | ---: | ---: | --- |
| A1 | 2,336.20 | 1,137.77 | GpuPreAgg |
| A2 | 337.47 | 1,065.10 | GpuPreAgg |
| Broad order join | 662.81 | 1,700.75 | GpuPreAgg with join |
| Bounded supplier projection | 39.70 | 26.49 | GpuJoin |

The archive's `pg-strom/screen/` contains the selected plans and SQL. Actual controls and ranges are under `pg-strom/controls/`. The build, service and failure logs are in the same directory tree. The repository includes the disposable installation and comparison scripts. [Build recipe](scripts/prepare_pg_strom_comparison.sh) · [Comparison harness](bench/pg_strom_comparison.py)

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
| `offload-crossover/` | Calibration and held-out JSON plans, frozen predictions, model errors and snapshot reuse |
| `pg-strom/` | Matched SQL, planning JSON, release and diagnostic run evidence |
