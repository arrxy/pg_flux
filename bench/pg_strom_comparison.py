#!/usr/bin/env python3
"""Compare PG-Strom, pg_flux, and PostgreSQL with identical SQL."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import statistics
import subprocess

from aggregation_tpch import FOCAL
from join_benchmark import QUERIES, walk_nodes

CASES = {**FOCAL, **QUERIES}
MODES = ('native_tuned', 'pg_flux', 'pg_strom')
MARKER = re.compile(r'^STROM_TRIAL (\w+)$', re.M)
SETTINGS = """
SELECT json_object_agg(name, setting ORDER BY name) FROM pg_settings
WHERE name IN ('shared_buffers','max_worker_processes','max_parallel_workers',
    'max_parallel_workers_per_gather','work_mem','jit','hash_mem_multiplier',
    'shared_preload_libraries','pg_strom.gpudirect_enabled','pg_strom.enabled',
    'pg_strom.gpu_setup_cost','pg_strom.gpu_tuple_cost',
    'pg_strom.gpu_operator_cost','pg_strom.cpu_fallback','pg_gftr.enabled',
    'pg_gftr.join_enabled','pg_gftr.mode','pg_gftr.join_result_mode');
"""


def set_mode(mode, case, prefer_gpu):
    workers = 0 if mode == 'pg_flux' else 4
    config = [
        f'SET max_parallel_workers_per_gather = {workers};',
        f"SET pg_strom.enabled = {'on' if mode == 'pg_strom' else 'off'};",
        f"SET pg_gftr.enabled = {'on' if mode == 'pg_flux' and case in FOCAL else 'off'};",
        f"SET pg_gftr.join_enabled = {'on' if mode == 'pg_flux' and case in QUERIES else 'off'};",
    ]
    if mode == 'pg_strom' and prefer_gpu:
        config += ['SET pg_strom.gpu_setup_cost = 0;', 'SET pg_strom.gpu_tuple_cost = 0;']
    else:
        config += ['RESET pg_strom.gpu_setup_cost;', 'RESET pg_strom.gpu_tuple_cost;']
    return config


def plan_details(explain):
    nodes = list(walk_nodes(explain['Plan']))
    return {
        'providers': [node['Custom Plan Provider'] for node in nodes
                      if 'Custom Plan Provider' in node],
        'workers_launched': sum(node.get('Workers Launched', 0) for node in nodes),
    }


def benchmark(args, case):
    query = CASES[case].strip()
    target = args.output / case
    target.mkdir(parents=True, exist_ok=True)
    setup = [
        "LOAD 'pg_gftr';",
        "SET statement_timeout='5min';",
        "SET search_path=pg_catalog,public;",
        "SET work_mem='256MB';",
        "SET jit=off;",
        "SET max_parallel_workers=8;",
        "SET pg_gftr.force=on;",
        "SET pg_gftr.backend='cuda';",
        "SET pg_gftr.mode='gftr';",
        "SET pg_gftr.max_input_mb=1024;",
        "SET pg_gftr.join_force=on;",
        "SET pg_gftr.join_result_mode='auto';",
        "BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY;",
    ]

    def record(label, lines):
        setup.extend([rf'\echo STROM_TRIAL {label}', *lines])

    record('versions', [
        "SELECT version();",
        "SELECT extname, extversion FROM pg_extension WHERE extname IN ('pg_strom','pg_gftr') ORDER BY 1;",
    ])
    for mode in MODES:
        setup.extend(set_mode(mode, case, args.prefer_gpu))
        record(f'settings_{mode}', [SETTINGS.strip()])
        if args.screen:
            record(f'plan_{mode}', [f'EXPLAIN (FORMAT JSON, SETTINGS) {query};'])
        else:
            record(f'types_{mode}', [query, r'\gdesc'])
            setup.extend([
                rf'\o rows_{mode}.txt',
                f'COPY ({query}) TO STDOUT;',
                r'\o',
            ])
    rounds = []
    if not args.screen:
        rng = random.Random(1831)
        for index in range(args.trials + 1):
            order = list(MODES)
            rng.shuffle(order)
            rounds.append(order)
            for mode in order:
                setup.extend(set_mode(mode, case, args.prefer_gpu))
                record(f'plan_{mode}_{index}', [
                    f'EXPLAIN (ANALYZE, BUFFERS, SETTINGS, TIMING OFF, FORMAT JSON) {query};'
                ])
    setup.append('COMMIT;')
    sql = target / 'workload.sql'
    sql.write_text('\n'.join(setup) + '\n')
    with (target / 'stdout.txt').open('w') as stdout, (target / 'stderr.txt').open('w') as stderr:
        subprocess.run([args.psql, '-XqAtv', 'ON_ERROR_STOP=1', '-d', args.dbname,
                        '-f', 'workload.sql'], cwd=target, stdout=stdout, stderr=stderr, check=True)

    chunks = MARKER.split((target / 'stdout.txt').read_text())
    records = dict(zip(chunks[1::2], (item.strip() for item in chunks[2::2])))
    report = {'case': case, 'sql': query, 'versions': records['versions'],
              'settings': {mode: json.loads(records[f'settings_{mode}']) for mode in MODES},
              'prefer_gpu': args.prefer_gpu, 'rounds': rounds, 'plans': [],
              'exact_checks': {}, 'measurements': {}}

    expected_rows = None
    for mode in MODES:
        if not args.screen:
            path = target / f'rows_{mode}.txt'
            rows = sorted(path.read_bytes().splitlines(keepends=True))
            types = records[f'types_{mode}']
            if expected_rows is None:
                expected_rows = rows
            if rows != expected_rows or types != records['types_native_tuned']:
                raise RuntimeError(f'{case}/{mode} output multiset or type mismatch')
            digest = hashlib.sha256(types.encode())
            for row in rows:
                digest.update(row)
            report['exact_checks'][mode] = {'rows': len(rows), 'columns': types,
                                            'sha256': digest.hexdigest()}
        times = []
        indexes = range(len(rounds)) if rounds else [None]
        for index in indexes:
            suffix = '' if index is None else f'_{index}'
            explain, = json.loads(records[f'plan_{mode}{suffix}'])
            details = plan_details(explain)
            report['plans'].append({'mode': mode, 'trial': index, 'warmup': index == 0,
                                    **details, 'explain': explain})
            if index:
                times.append(explain['Execution Time'])
        if times:
            report['measurements'][mode] = {
                'median_ms': statistics.median(times), 'min_ms': min(times), 'max_ms': max(times),
            }
    if not args.screen:
        for mode in MODES:
            (target / f'rows_{mode}.txt').unlink()
    (target / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'case': case, 'measurements': report['measurements'],
                      'providers': {mode: sorted({p for node in report['plans']
                                                 if node['mode'] == mode for p in node['providers']})
                                    for mode in MODES}}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--psql', default='psql')
    parser.add_argument('--dbname', default='tpch_sf1')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--cases', choices=CASES, nargs='+', default=['A1', 'A2', 'orders_broad'])
    parser.add_argument('--trials', type=int, default=5)
    parser.add_argument('--screen', action='store_true')
    parser.add_argument('--prefer-gpu', action='store_true',
                        help='test PG-Strom with zero GPU setup and transfer tuple cost')
    args = parser.parse_args()
    if args.trials < 1:
        parser.error('at least one timed trial is required')
    for case in args.cases:
        benchmark(args, case)


if __name__ == '__main__':
    main()
