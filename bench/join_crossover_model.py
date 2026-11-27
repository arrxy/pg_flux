#!/usr/bin/env python3
"""Freeze phase-based CUDA predictions and score them on held-out joins."""
import argparse
from hashlib import sha256
import json
import math
import re
from pathlib import Path
import statistics

PHASES = ("Build Scan and Pack", "Probe Scan and Pack", "Build", "H2D", "Probe", "D2H")
RESIDUAL = "Host residual"


def load_report(path):
    if path.is_dir():
        path, = sorted(path.glob("*/database-1/report.json"))
    return json.loads(path.read_text()), sha256(path.read_bytes()).hexdigest()


def gpu_observations(report):
    cases = {}
    for trial in report["plans"]:
        if trial["warmup"] or trial["mode"] != "cuda_auto":
            continue
        join, = (node for node in trial["operators"]
                 if node.get("Custom Plan Provider") == "GftrJoin")
        phases = {key: join[key] for key in PHASES}
        wall_ms = trial["explain"]["Execution Time"]
        phases[RESIDUAL] = max(0.0, wall_ms - sum(phases.values()))
        observation = {
            **phases,
            "probe_rows": join["Probe Rows"],
            "build_rows": join["Build Rows"],
            "matched_rows": join["Output Rows"],
            "batch_rows": join["Probe Batch Size"],
            "projected_columns": join["Projected Columns"],
            "h2d_bytes": join["H2D Bytes"],
            "d2h_bytes": join["D2H Bytes"],
            "compaction_requested": join["Compaction Requested"],
            "compact_batches": join["Compact Batches"],
            "dense_batches": join["Dense Batches"],
            "wall_ms": wall_ms,
        }
        cases.setdefault(trial["case"], []).append(observation)
    return {
        case: {key: statistics.median(row[key] for row in rows) for key in rows[0]}
        for case, rows in cases.items()
    }


def fit_affine(rows, feature, target):
    """Least-squares line with nonnegative intercept and coefficient."""
    x = [row[feature] for row in rows]
    y = [row[target] for row in rows]
    candidates = [(statistics.mean(y), 0.0)]
    denominator = sum(value * value for value in x)
    if denominator:
        candidates.append((0.0, sum(a * b for a, b in zip(x, y)) / denominator))
    if len(set(x)) > 1:
        slope, intercept = statistics.linear_regression(x, y)
        if slope >= 0 and intercept >= 0:
            candidates.append((intercept, slope))
    return min(candidates, key=lambda fit: sum((fit[0] + fit[1] * a - b) ** 2
                                                for a, b in zip(x, y)))


def predict(calibration, native):
    training = gpu_observations(calibration)
    rows = list(training.values())
    probe_rows = round(statistics.median(row["probe_rows"] for row in rows))
    batch_rows = round(statistics.median(row["batch_rows"] for row in rows))
    batches = math.ceil(probe_rows / batch_rows)
    predictions = {}
    for case, modes in native["measurements"].items():
        dimension, cutoff, width = case.split("_")
        # SF1 supplier keys are contiguous 1..10,000. Probe rows use the
        # same lineitem order-key predicate as the calibration cases.
        if dimension != "supplier":
            raise ValueError(f"unexpected held-out dimension {dimension}")
        build_rows = int(cutoff)
        fraction = build_rows / 10_000
        matched_rows = round(fraction * probe_rows)
        compact = fraction < 0.10
        same_width = [row for name, row in training.items() if name.endswith(f"_{width}")]
        same_mode = [row for row in rows if bool(row["compaction_requested"]) == compact]
        if not same_mode:
            same_mode = rows
        # Unique-key GPU lookup sends build int4 keys and int4 + validity
        # for each probe. Return bytes include one build check and, for
        # compaction, one count per batch and one 8-byte pair per match.
        features = {
            "probe_rows": probe_rows,
            "build_rows": build_rows,
            "matched_rows": matched_rows,
            "h2d_bytes": 4 * build_rows + 5 * probe_rows,
            "d2h_bytes": 4 + (4 * batches + 8 * matched_rows if compact else 4 * probe_rows),
        }
        fits = {
            "Build Scan and Pack": (rows, "build_rows"),
            "Build": (rows, "build_rows"),
            "H2D": (rows, "h2d_bytes"),
            "D2H": (same_mode, "d2h_bytes"),
            RESIDUAL: (same_width, "matched_rows"),
        }
        phases = {
            "Probe Scan and Pack": statistics.median(row["Probe Scan and Pack"]
                                                      for row in same_width),
            "Probe": statistics.median(row["Probe"] for row in same_mode),
        }
        coefficients = {}
        for phase, (fit_rows, feature_name) in fits.items():
            intercept, slope = fit_affine(fit_rows, feature_name, phase)
            coefficients[phase] = {"intercept_ms": intercept, "ms_per_unit": slope,
                                   "feature": feature_name}
            phases[phase] = intercept + slope * features[feature_name]
        gpu_ms = sum(phases.values())
        native_ms = modes["native_tuned"]["median_ms"]
        predictions[case] = {
            "width_class": width,
            "estimated_match_fraction": fraction,
            "predicted_compaction_requested": compact,
            "features": features,
            "coefficients": coefficients,
            "gpu_phase_ms": phases,
            "predicted_gpu_ms": gpu_ms,
            "tuned_native_ms": native_ms,
            "tuned_native_checksum": native["checksums_sha256"][case]["native_tuned"],
            "predicted_gpu_wins": gpu_ms < native_ms,
        }
    return {"calibration_medians": training, "predictions": predictions}


def score(frozen, validation):
    actual = gpu_observations(validation)
    rows = {}
    for case, prediction in frozen["predictions"].items():
        cuda_checksum = validation["checksums_sha256"][case]["cuda_auto"]
        if cuda_checksum != prediction["tuned_native_checksum"]:
            raise ValueError(f"held-out result checksum differs for {case}")
        observed = actual[case]
        measured_ms = observed["wall_ms"]
        wins = measured_ms < prediction["tuned_native_ms"]
        rows[case] = {
            "measured": observed,
            "actual_gpu_wins": wins,
            "winner_correct": wins == prediction["predicted_gpu_wins"],
            "compaction_selection_correct": bool(observed["compaction_requested"]) ==
            prediction["predicted_compaction_requested"],
            "signed_error_ms": prediction["predicted_gpu_ms"] - measured_ms,
            "absolute_relative_error": abs(prediction["predicted_gpu_ms"] / measured_ms - 1),
            "phase_signed_errors_ms": {phase: estimate - observed[phase] for phase, estimate in
                                       prediction["gpu_phase_ms"].items()},
        }
    return {
        "cases": rows,
        "winner_accuracy": statistics.mean(row["winner_correct"] for row in rows.values()),
        "mean_absolute_relative_error": statistics.mean(row["absolute_relative_error"]
                                                        for row in rows.values()),
    }


def snapshot_reuse(directory):
    output = {}
    for path in sorted(directory.glob("sf*-summary.json")):
        reference = json.loads(path.read_text())
        resident = reference["resident"]
        tuned_ms = reference["native"]["tuned"]["median_ms"]
        cold_ms = resident["cold_ms"]
        warm_ms = resident["warm_ms"]
        crossing = (max(1, math.floor((cold_ms - warm_ms) / (tuned_ms - warm_ms)) + 1)
                    if tuned_ms > warm_ms else (1 if cold_ms < tuned_ms else None))
        scale = path.name.removesuffix("-summary.json")
        sessions = {
            int(count): [(sum(series["resident_ms"]), sum(series["native_ms"]))]
            for count, series in reference["validated_series"].items()
        }
        for resident_log in sorted(directory.glob(f"{scale}-resident-series-*-repeat*.log")):
            tail = resident_log.name.removeprefix(f"{scale}-resident-series-")
            count = int(tail.split("-")[0])
            native_log = directory / f"{scale}-native-series-{tail}"
            pattern = r"Execution Time: ([0-9.]+) ms"
            resident_times = re.findall(pattern, resident_log.read_text())
            # Follow-up native scripts warm the query once before measuring.
            native_times = re.findall(pattern, native_log.read_text())[1:]
            if len(resident_times) != count + 1 or len(native_times) != count:
                raise ValueError(f"unexpected number of EXPLAIN timings in {tail}")
            sessions.setdefault(count, []).append((sum(map(float, resident_times)),
                                                   sum(map(float, native_times))))
        output[scale] = {
            "source_summary_sha256": sha256(path.read_bytes()).hexdigest(),
            "cold_ms": cold_ms,
            "warm_ms": warm_ms,
            "tuned_native_ms": tuned_ms,
            "predicted_crossover": crossing,
            "validation": {
                count: {
                    "predicted_resident_ms": cold_ms + (count - 1) * warm_ms,
                    "predicted_native_ms": count * tuned_ms,
                    "resident_total_ms": [resident_ms for resident_ms, _ in comparisons],
                    "native_total_ms": [native_ms for _, native_ms in comparisons],
                    "resident_wins": [resident_ms < native_ms
                                      for resident_ms, native_ms in comparisons],
                }
                for count, comparisons in sorted(sessions.items())
            },
        }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    freeze = commands.add_parser("predict")
    freeze.add_argument("calibration", type=Path)
    freeze.add_argument("tuned_native", type=Path)
    freeze.add_argument("output", type=Path)
    validate = commands.add_parser("score")
    validate.add_argument("prediction", type=Path)
    validate.add_argument("gpu_validation", type=Path)
    validate.add_argument("output", type=Path)
    snapshot = commands.add_parser("snapshot")
    snapshot.add_argument("resident_sql_results", type=Path)
    snapshot.add_argument("output", type=Path)
    args = parser.parse_args()

    if args.action == "predict":
        calibration, calibration_sha = load_report(args.calibration)
        native, baseline_sha = load_report(args.tuned_native)
        result = predict(calibration, native)
        result.update(calibration_sha256=calibration_sha, tuned_native_sha256=baseline_sha)
    elif args.action == "score":
        encoded = args.prediction.read_bytes()
        validation, validation_sha = load_report(args.gpu_validation)
        result = score(json.loads(encoded), validation)
        result.update(prediction_sha256=sha256(encoded).hexdigest(),
                      gpu_validation_sha256=validation_sha)
    else:
        result = snapshot_reuse(args.resident_sql_results)
    with args.output.open("x") as output:
        json.dump(result, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
