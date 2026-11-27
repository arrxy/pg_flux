import argparse
import json
import os
from pathlib import Path
import re
import statistics
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("output", type=Path)
parser.add_argument("scale", choices=("sf1", "sf5"))
args = parser.parse_args()
native = """SELECT ARRAY[count(*),sum(o.o_custkey),sum(c.c_nationkey)]
FROM lineitem l JOIN orders o ON l.l_orderkey=o.o_orderkey
JOIN customer c ON o.o_custkey=c.c_custkey"""
load = "SELECT pg_gftr_resident_load('lineitem','orders','customer')"
run = "SELECT pg_gftr_resident_run()"
explain = "EXPLAIN (ANALYZE, BUFFERS, TIMING OFF) "
settings = "SET jit=off; SET work_mem='256MB'; SET max_parallel_workers=4;"

def execute(name, sql):
    raw = subprocess.run(
        ["psql", "-XAtv", "ON_ERROR_STOP=1", "-d", "tpch_" + args.scale],
        input=sql, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    (args.output / f"{args.scale}-{name}.txt").write_text(sql + "\n")
    (args.output / f"{args.scale}-{name}.log").write_text(raw.stdout)
    raw.check_returncode()
    return ([float(sample) for sample in re.findall(r"Execution Time: ([\d.]+) ms", raw.stdout)],
            re.findall(r"^\{[^}]+\}$", raw.stdout, re.M))

def series(query, count):
    return (explain + query + ";\n") * count

args.output.mkdir(exist_ok=True, parents=True)
results = {}
results["native"] = {}
for label, workers in (("serial", 0), ("tuned", 4)):
    prefix = settings + f"SET max_parallel_workers_per_gather={workers};"
    times, values = execute(label + "-single", prefix + series(native, 4) + native + ";")
    assert len(times) == 4 and len(values) == 1
    results["native"][label] = {"median_ms": statistics.median(times[1:]), "trials_ms": times[1:], "value": values[0]}
expected = results["native"]["tuned"]["value"]
assert results["native"]["serial"]["value"] == expected

def resident(name, count):
    times, values = execute(name, settings + "BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY;\n"
                            + series(load, 1) + series(run, count) + run + ";\nCOMMIT;")
    assert len(times) == count + 1 and values == [expected]
    return times

sessions = [resident(f"resident-{i}", 8) for i in range(1, 4)]
load_ms = statistics.median(trial[0] for trial in sessions)
first_ms = statistics.median(trial[1] for trial in sessions)
cold_ms = statistics.median(trial[0] + trial[1] for trial in sessions)
warm_ms = statistics.median(sample for trial in sessions for sample in trial[2:])
results["resident"] = dict(sessions_ms=sessions, load_ms=load_ms, first_ms=first_ms, cold_ms=cold_ms, warm_ms=warm_ms)
tuned = results["native"]["tuned"]["median_ms"]
if tuned > warm_ms:
    crossover = max(1, int((cold_ms - warm_ms) // (tuned - warm_ms)) + 1)
    results["predicted_crossover"] = crossover
    results["validated_series"] = {}
    for count in sorted({max(1, crossover - 1), crossover}):
        resident_times = resident(f"resident-series-{count}", count)
        native_times, _ = execute(f"native-series-{count}", settings + "SET max_parallel_workers_per_gather=4;\n"
                                  + series(native, 1 + count))
        assert len(native_times) == 1 + count
        results["validated_series"][count] = dict(resident_ms=resident_times, native_ms=native_times[1:])
(args.output / f"{args.scale}-summary.json").write_text(json.dumps(results, indent=2) + "\n")
