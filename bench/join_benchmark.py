#!/usr/bin/env python3
"""Benchmark native and custom join paths on loaded PostgreSQL TPC-H data."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import statistics
import subprocess

QUERIES = {
    "supplier_grouped": """
SELECT s.s_nationkey, count(*), sum(l.l_linenumber)
FROM lineitem l JOIN supplier s ON l.l_suppkey = s.s_suppkey
GROUP BY s.s_nationkey
ORDER BY s.s_nationkey""",
    "part_payload": """
SELECT count(*), sum(p.p_size), sum(p.p_partkey), sum(l.l_linenumber)
FROM lineitem l JOIN part p ON l.l_partkey = p.p_partkey""",
    "orders_broad": """
SELECT count(*), sum(o.o_custkey)
FROM lineitem l JOIN orders o ON l.l_orderkey = o.o_orderkey""",
    "orders_selective": """
SELECT count(*), sum(l.l_partkey)
FROM lineitem l JOIN orders o ON l.l_orderkey = o.o_orderkey
WHERE o.o_orderdate >= DATE '1995-01-01'
  AND o.o_orderdate < DATE '1995-02-01'""",
    "supplier_projected": """
SELECT l.l_orderkey, l.l_linenumber, l.l_suppkey, s.s_nationkey
FROM lineitem l JOIN supplier s ON l.l_suppkey = s.s_suppkey
WHERE l.l_orderkey BETWEEN 1 AND 40000
ORDER BY l.l_orderkey, l.l_linenumber""",
}
MODES = ("native", "native_tuned", "cpu", "cuda", "native_gpu_agg", "auto", "fused_cpu")
CUSTOM_JOIN_MODES = ("cpu", "cuda", "auto", "fused_cpu")
GROUPED_MODES = ("native_gpu_agg", "fused_cpu")
RECORD = re.compile(r"^GFTR_JOIN_RECORD (\w+)$", re.MULTILINE)
SETTINGS = (
    "server_version",
    "work_mem",
    "hash_mem_multiplier",
    "max_parallel_workers",
    "max_parallel_workers_per_gather",
    "jit",
    "shared_buffers",
    "effective_cache_size",
    "track_io_timing",
)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--psql", default="psql")
    parser.add_argument("--dbname", nargs="+", default=["tpch_sf1", "tpch_sf5"],
                        help="database names or libpq connection strings, in run order")
    parser.add_argument("--cases", choices=QUERIES, nargs="+", default=list(QUERIES))
    parser.add_argument("--modes", choices=MODES, nargs="+", default=list(MODES[:4]))
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--seed", type=int, default=714)
    parser.add_argument("--statement-timeout-ms", type=int, default=300_000)
    parser.add_argument("--serial", action="store_true", help="disable plan parallelism")
    parser.add_argument("--disable-jit", action="store_true")
    parser.add_argument("--tuned-work-mem-mb", type=int, default=256)
    parser.add_argument("--join-batch-rows", type=int, default=131_072)
    parser.add_argument("--join-max-host-mb", type=int, default=256)
    parser.add_argument("--join-max-device-mb", type=int, default=4096)
    parser.add_argument("--agg-max-input-mb", type=int, default=1024)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("results/join"))
    args = parser.parse_args()
    if (args.trials < 1 or args.warmups < 0 or args.device < 0
            or min(args.tuned_work_mem_mb, args.join_batch_rows, args.join_max_host_mb,
                   args.join_max_device_mb, args.agg_max_input_mb,
                   args.statement_timeout_ms) < 1):
        parser.error("invalid trial count, device, memory budget, batch size or timeout")
    if len(set(args.modes)) != len(args.modes) or len(set(args.cases)) != len(args.cases):
        parser.error("each mode and case may appear only once")
    if not any(case_modes(args, case) for case in args.cases):
        parser.error("native_gpu_agg and fused_cpu are applicable to supplier_grouped only")
    return args


def case_modes(args, case):
    return [mode for mode in args.modes
            if mode not in GROUPED_MODES or case == "supplier_grouped"]


def configure(args, mode):
    settings = [
        f"SET work_mem='{args.tuned_work_mem_mb}MB';"
        if mode == "native_tuned" else "RESET work_mem;",
        f"SET pg_gftr.enabled={'on' if mode == 'native_gpu_agg' else 'off'};",
    ]
    if mode in CUSTOM_JOIN_MODES:
        backend = "cpu" if mode in ("cpu", "fused_cpu") else "cuda"
        settings += [
            "SET pg_gftr.join_enabled=on;",
            f"SET pg_gftr.join_force={'off' if mode == 'auto' else 'on'};",
            f"SET pg_gftr.join_backend='{backend}';",
        ]
    else:
        # Also run native and aggregate-only modes against pre-join builds.
        settings.append(
            "SELECT 'SET pg_gftr.join_enabled=off;' "
            "WHERE current_setting('pg_gftr.join_enabled', true) IS NOT NULL \\gexec"
        )
    if mode == "fused_cpu":
        settings += ["SET pg_gftr.fused_enabled=on;", "SET pg_gftr.fused_force=on;"]
    else:
        # Older builds have no fused GUC. Native and existing modes still work.
        settings.append(
            "SELECT 'SET pg_gftr.fused_enabled=off;' "
            "WHERE current_setting('pg_gftr.fused_enabled', true) IS NOT NULL \\gexec"
        )
    return settings


def record(name):
    return f"\\echo GFTR_JOIN_RECORD {name}"


def workload(args, database_number):
    uses_extension = any(mode not in ("native", "native_tuned") for mode in args.modes)
    script = [
        f"SET statement_timeout={args.statement_timeout_ms};",
        "SET search_path=pg_catalog,public;",
    ]
    if uses_extension:
        script += ["LOAD 'pg_gftr';", f"SET pg_gftr.device={args.device};"]
    if any(mode in CUSTOM_JOIN_MODES for mode in args.modes):
        script += [
            f"SET pg_gftr.join_batch_rows={args.join_batch_rows};",
            f"SET pg_gftr.join_max_host_mb={args.join_max_host_mb};",
            f"SET pg_gftr.join_max_device_mb={args.join_max_device_mb};",
        ]
    if "native_gpu_agg" in args.modes:
        script += [
            "SET pg_gftr.backend='cuda';",
            "SET pg_gftr.mode='gftr';",
            "SET pg_gftr.force=on;",
            f"SET pg_gftr.max_input_mb={args.agg_max_input_mb};",
            f"SET pg_gftr.max_device_mb={args.join_max_device_mb};",
        ]
    if args.serial:
        script.append("SET max_parallel_workers_per_gather=0;")
    if args.disable_jit:
        script.append("SET jit=off;")
    keys = ", ".join(f"'{key}'" for key in SETTINGS)
    script += [
        "BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY;",
        record("environment"),
        "SELECT json_build_object('database', current_database(), 'postgres', version(), "
        f"'settings', (SELECT json_object_agg(name, current_setting(name)) "
        f"FROM pg_settings WHERE name IN ({keys})));",
    ]
    cases = {case: case_modes(args, case) for case in args.cases if case_modes(args, case)}
    verification = {}
    for case, modes in cases.items():
        verify_modes = list(dict.fromkeys(["native", *modes]))
        verification[case] = verify_modes
        for mode in verify_modes:
            script += configure(args, mode)
            script += [record(f"result_{case}_{mode}"), f"{QUERIES[case]};"]

    schedule = []
    rng = random.Random(args.seed + database_number)
    for trial in range(args.warmups + args.trials):
        order = [(case, mode) for case, modes in cases.items() for mode in modes]
        rng.shuffle(order)
        for case, mode in order:
            name = f"plan_{trial}_{case}_{mode}"
            schedule.append((trial, case, mode, name))
            script += configure(args, mode)
            script += [record(name),
                       "EXPLAIN (ANALYZE, BUFFERS, SETTINGS, TIMING OFF, FORMAT JSON)"
                       f" {QUERIES[case]};"]
    script.append("COMMIT;")
    return "\n".join(script) + "\n", verification, schedule


def walk_nodes(node):
    yield node
    for child in node.get("Plans", []):
        yield from walk_nodes(child)


def check_plan(plan, case, mode):
    nodes = list(walk_nodes(plan["Plan"]))
    joins = [node for node in nodes if node.get("Custom Plan Provider") == "GftrJoin"]
    aggregations = [node for node in nodes
                    if node.get("Custom Plan Provider") == "GftrAggregate"]
    fused = [node for node in nodes
             if node.get("Custom Plan Provider") == "GftrFusedJoinAggregate"]
    if len(fused) != int(mode == "fused_cpu"):
        raise ValueError(f"{case}/{mode} expected a different number of GftrFusedJoinAggregate nodes")
    if mode != "auto" and len(joins) != int(mode in ("cpu", "cuda", "fused_cpu")):
        raise ValueError(f"{case}/{mode} expected a different number of GftrJoin nodes")
    if len(aggregations) != int(mode == "native_gpu_agg"):
        raise ValueError(f"{case}/{mode} expected a different number of GftrAggregate nodes")
    native_joins = [node for node in nodes if "Join Type" in node and node not in joins]
    if not joins and not native_joins:
        raise ValueError(f"{case}/{mode} has no join node")
    for node in joins:
        backend = str(node.get("Backend", "")).lower()
        expected = "cpu" if mode in ("cpu", "fused_cpu") else "cuda"
        if expected not in backend:
            raise ValueError(f"{case}/{mode} expected the {expected} backend, got {backend!r}")
    return [{key: value for key, value in node.items() if key != "Plans"}
            for node in [*fused, *joins, *aggregations, *native_joins]]


def summarize(samples):
    return {
        "execution_ms": [sample["Execution Time"] for sample in samples],
        "median_ms": statistics.median(sample["Execution Time"] for sample in samples),
        "min_ms": min(sample["Execution Time"] for sample in samples),
        "max_ms": max(sample["Execution Time"] for sample in samples),
    }


def run_database(args, dbname, database_number, output):
    output.mkdir()
    script, verification, schedule = workload(args, database_number)
    sql_path = output / "workload.sql"
    sql_path.write_text(script)
    stdout_path = output / "stdout.txt"
    stderr_path = output / "stderr.txt"
    cmd = [args.psql, "-X", "-qAt", "-v", "ON_ERROR_STOP=1", "--dbname", dbname,
           "--file", str(sql_path)]
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        subprocess.run(cmd, stdout=stdout, stderr=stderr, check=True)
    chunks = RECORD.split(stdout_path.read_text())
    records = {name: value.strip("\n") for name, value in zip(chunks[1::2], chunks[2::2])}
    hashes = {
        case: {mode: hashlib.sha256(records[f"result_{case}_{mode}"].encode()).hexdigest()
               for mode in modes}
        for case, modes in verification.items()
    }
    samples = {case: {mode: [] for mode in case_modes(args, case)} for case in verification}
    plans = []
    for trial, case, mode, name in schedule:
        plan, = json.loads(records[name])
        operators = check_plan(plan, case, mode)
        warmup = trial < args.warmups
        if not warmup:
            samples[case][mode].append(plan)
        plans.append({"trial": trial + 1, "warmup": warmup, "case": case, "mode": mode,
                      "operators": operators, "explain": plan})
    for case, values in hashes.items():
        if len(set(values.values())) != 1:
            raise ValueError(f"{case} has different result checksums: {values}")
    report = {
        "environment": json.loads(records["environment"]),
        "configuration": {key: value for key, value in vars(args).items()
                          if key not in ("psql", "dbname", "output")},
        "queries": {case: QUERIES[case].strip() for case in verification},
        "checksums_sha256": hashes,
        "measurements": {case: {mode: summarize(values) for mode, values in modes.items()}
                         for case, modes in samples.items()},
        "plans": plans,
        "timing": ("EXPLAIN ANALYZE execution time in a single session and snapshot. "
                   "Includes SQL execution and accelerator transfers; excludes normal "
                   "result delivery. Result digests are collected outside measured trials."),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    args = arguments()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    root = args.output / stamp
    root.mkdir(parents=True)
    for number, dbname in enumerate(args.dbname):
        output = root / f"database-{number + 1}"
        print(f"Running database {number + 1}; artifacts in {output}", flush=True)
        try:
            report = run_database(args, dbname, number, output)
        except subprocess.CalledProcessError as error:
            raise SystemExit(f"psql exited with status {error.returncode}; see "
                             f"{output / 'stderr.txt'}") from None
        except (OSError, ValueError, KeyError) as error:
            raise SystemExit(f"Benchmark failed: {error}; see {output}") from error
        for case, modes in report["measurements"].items():
            summary = ", ".join(f"{mode} {result['median_ms']:.1f} ms"
                                for mode, result in modes.items())
            print(f"{report['environment']['database']} {case}: {summary}")
    print(f"Checksums agree. Reports in {root}")


if __name__ == "__main__":
    main()
