# A100 final report implementation plan

Deliver a repository-root `REPORT.md` covering PostgreSQL aggregations and joins at TPC-H SF1 and SF5. Answer two questions for each workload. Does GPU execution shorten the full SQL query versus serial and tuned parallel PostgreSQL? What incremental benefit comes from transforming payloads (GFTR) versus gathering original payloads by row ID (GFUR)? Use the October A6000 report as a historical comparison and Wu, Koutsoukos and Alonso as the algorithm reference. A result from one machine, operator, or timing boundary cannot fill a missing cell for another.

## Dispatch and integration

Use `gpu-resident-sql` at `34085f6` as the initial code reference because it contains both aggregate modes, SQL joins and the resident SQL API. The coordinator creates `report/a100` from the revision containing this plan. Each numbered task below has a separate suggested branch off that integration branch. Give each writer an isolated checkout and set the repository commit email to `aritroroy1999@gmail.com`. Commit harness changes before collecting final measurements. Record the benchmarked SHA separately from the report SHA.

| Wave | Agents and dependencies | Coordinator handoff |
| --- | --- | --- |
| Preparation | A inventories hardware and verifies the build. B recovers October workloads. C, D and E can prepare their harnesses from the same base as slots free up. | Integrate reviewed harness changes. Resolve SQL identity and environment differences before assigning timing slots. |
| Measurement | B runs aggregation SQL after A. C runs the operator study after A and B's workload inventory. D and E run join experiments after A. | Assign one timing slot at a time across SQL, CUDA and profiling on this host. Analysis of completed trials and work on another harness can proceed concurrently. |
| Synthesis | F starts the paper-to-code mapping during preparation, then audits results from A–E. | Coordinator integrates findings, reruns disputed cells, verifies links and packages evidence. Only F edits the final narrative. |

Every run writes to its own `results/final-report/<run-id>/<task>/` directory. Each handoff includes the commit, exact command, inputs, settings, raw trials, correctness status, plan or operator evidence, and a short list of unresolved cases. Do not rely on ignored results files being included in a Git fetch. Save an indexed raw-results archive and a Git bundle in `/mnt/data` when the report is complete.

## A. Freeze the environment and establish correctness — report/environment

- [ ] Source `/home/ubuntu/pg-flux-setup/env.sh`. Record GPU model, driver, clocks and load, CPU model and available cores, CUDA compiler, PostgreSQL build flags, extension version in both databases, commit, and relevant PostgreSQL settings. Check data and indexes for all eight tables. Record whether the A100 data were regenerated or copied, row counts, and any available hashes against the October inputs.
- [ ] Reserve the host for installation. Run `bash scripts/verify_a100.sh` and capture stdout, stderr and exit status. This script cleans, builds, installs and tests the extension. Confirm from logs that aggregation GFTR/GFUR, CUDA joins and resident SQL fixtures actually ran. Record fixture counts and failures.
- [ ] Smoke-test new sessions against both loaded databases. Record `work_mem`, JIT, worker limits, shared buffers, statistics state and cache policy. Keep the original data and indexes intact. Flag a correctness or build failure to the relevant owner before timing that path.
- **Done when** B–E can use one identified installation and both datasets, with a manifest that explains every known difference from October.

## B. Reproduce October aggregation SQL — report/aggregation-sql

- [ ] Recover exact A1–A4 SQL, the 22 standard query texts and substitutions, and the identity of the ten SF5 queries timed in October. Search archived case files and source history. The current `bench/benchmark.py` generates synthetic workloads. If original SQL is unavailable, preserve a reconstructed version and mark its results as a new comparison. Identify any missing October cases explicitly.
- [ ] Build a case manifest covering SF1 and SF5. A1 groups lineitem by part, A2 by supplier, A3 by supplier nation after a native join, and A4 groups orders by customer. Disable custom GPU joins for this study. Screen all 22 standard plans per scale. For Q13 identify which aggregate ran on CUDA and which operators stayed native.
- [ ] Match October settings first (serial native, JIT off, 256 MB work_mem, one warmup, five measured custom rounds and three standard rounds). Reproduce the October mode coverage, including native versus GFTR for Q13. Add GFUR for Q13 and a tuned parallel-native comparison on A1–A4 and Q13. Allow four workers, record actual workers launched, and keep other settings matched. Record whether native aggregation spilled.
- [ ] Compare types and exact result multisets including NULLs and duplicates outside the timed region. Verify plan providers. Capture server execution time using the same EXPLAIN options across modes. Save order, every trial and actual plan; interleave modes in reproducible blocks with a recorded seed. Label fallback timings.
- **Done when** every original timed cell has a reproduction or explicit gap, the five focal cases have both native baselines, and the SF5 A1 headline can be recalculated from raw times.

## C. Isolate the grouping mechanism — report/aggregation-operator

- [ ] From B's SF5 A2 plans, extract scan/pack, transfers, sort, group discovery/reduction, backend and SQL execution time. Check instrumentation in `src/gftr_cuda.cu` for overlapping stages. Recalculate the October claims about faster reduction and slower sorting using A100 trials.
- [ ] Inspect the CUDA interface for an existing resident aggregate benchmark. If needed, add a benchmark-only driver reusing the production sort/reduce logic. Hold row count, keys, NULL pattern and payloads constant within each paired comparison. Suggested first grid: six million rows, 25 and 200,000 groups, one and four int4 payload columns, fixed seed. Verify achieved group counts, input sizes and results for each cell.
- [ ] Warm input on device, charge per-execution ordering and transformation, time with CUDA events after synchronization, and log allocation/upload boundaries. Use one warmup and at least seven interleaved measured pairs. Collect memory traffic or L2 counters on separate profiling passes if available. Preserve counter definitions or the reason counters were unavailable.
- **Done when** SQL phase costs and resident operator timings have separate tables. A locality explanation must identify its counter evidence or be labeled an interpretation. For small SQL differences report paired variation and avoid asserting a winner from rounded medians.

## D. Validate ordinary SQL joins and compaction — report/join-sql

- [ ] Use `bench/join_benchmark.py` and its existing five cases. Start with `orders_selective` and `orders_broad`; add `part_payload`, `supplier_projected` and `supplier_grouped`. For the grouped case include `native_gpu_agg` to distinguish GPU aggregation above a native join. Cross-reference historical CPU-fused results with their branch and settings.
- [ ] Run SF1 and SF5, both native baselines, and cuda_dense, cuda_compact and cuda_auto. Start with one warmup and five measured trials per mode. The runner resets work_mem for several modes and sets 256 MB only for native_tuned. Add a matched-settings run or extend the harness to enforce the comparison explicitly. Pin JIT and parallel limits, inspect actual plans, and record any unselected GPU path.
- [ ] Save full SQL time, build/probe rows, actual selectivity, batch counts, host/device bytes and fallback counts where instrumented. Check exact SQL output against native separately from timing. Distinguish reduced transfer volume from reduced SQL elapsed time.
- **Done when** readers can tell whether auto selected compaction appropriately, whether it beat dense on selective joins, its cost on broad joins, and whether either path beat tuned PostgreSQL.

## E. Test transformed join payloads and snapshot reuse — report/join-residency

- [ ] Compile `bench/join_resident_pipeline.cu` using nvcc with `-O3 -std=c++17 -arch=sm_80`. Run against `/home/ubuntu/tpch/data/sf1` and `sf5`. Follow `GPU_RESIDENT_JOIN_FINDINGS.md`. Retain raw CSV and value checks.
- [ ] For lineitem joined to part, compare direct lookup, partitioned GFUR and partitioned GFTR-style at widths one and eight. Identify the seven synthetic columns at width eight. Include partition, lookup rebuild and payload transformation in total operator time. Verify exact payloads. For the two-join pipeline, compare resident intermediates, a forced host round trip, and prepacked upload plus build.
- [ ] Follow `GPU_RESIDENT_SQL_FINDINGS.md` for the fixed lineitem–orders–customer SQL API. Compare count and both sums against the equivalent native SQL in the same snapshot. In at least three fresh sessions per scale measure load and the actual first run separately and as a pair, then seven warm calls. State that cold means an unloaded device snapshot and record OS-cache policy. Retest equivalent serial and tuned native SQL.
- [ ] Calculate the smallest integer reuse count for which measured load + first run + subsequent warm runs beats the same number of tuned native executions. Validate the predicted crossover by timing the series just below and at that count, with matched timing boundaries. Label the fixed-query, same-transaction constraint.
- **Done when** the report can distinguish ordinary SQL joins, the specialized SQL API and standalone CUDA, and identify which width and scale benefited from GFTR-style materialization.

## F. Write and audit REPORT.md — report/findings

- [ ] Map the paper's transformed join payload gather and group-by algorithms to actual code and experiments. Distinguish adaptations, direct implementations and untested claims. Cite the paper by section and link the relevant source at the measured commit.
- [ ] Organize the report around the project question, methods, correctness, aggregation SQL, join SQL, operator experiments, limitations and reproducibility. Put the A100 end-to-end verdict first. Place A6000 historical results beside the matching reproduction. Identify data-generation, host, query, plan or settings differences affecting comparison.
- [ ] For each headline provide scale, query, timing boundary, native baseline, median, dispersion, measured rounds and source artifact. Compute speedup as baseline time divided by candidate time. For GFTR's incremental benefit use GFUR as baseline and also show the best direct join strategy. Report losses, ties, fallback cases and unanswered questions. Keep official TPC-H scores outside this prototype study.
- [ ] Recompute headline numbers from raw trials, check units and break-even rounding, and audit each GPU claim against an actual plan or CUDA variant. Link scripts, plans and archive manifest. Verify the bundle with Git and the archive against its checksum.
- **Done when** an independent reader can reproduce one aggregate and one join from the documented commands and trace every performance conclusion to a measured result with an explicit timing boundary.
