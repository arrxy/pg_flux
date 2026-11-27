#!/usr/bin/env python3
"""Benchmark reconstructed October grouped SQL and generated TPC-H queries."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import statistics
import subprocess

FOCAL = {
    'A1': 'SELECT l_partkey, count(*), sum(l_linenumber) FROM lineitem GROUP BY l_partkey',
    'A2': 'SELECT l_suppkey, count(*), sum(l_linenumber) FROM lineitem GROUP BY l_suppkey',
    'A3': '''SELECT s_nationkey, count(*), sum(l_linenumber)
FROM lineitem JOIN supplier ON l_suppkey = s_suppkey GROUP BY s_nationkey''',
    'A4': 'SELECT o_custkey, count(*), sum(o_shippriority) FROM orders GROUP BY o_custkey',
}
MODES = ('native_serial', 'gfur', 'gftr', 'native_parallel')
PHASES = ('Scan and Pack', 'Backend Total', 'Sort', 'Group and Reduce', 'Transfer')
SEED = 231200720


def build_case(case, scale, qgen):
    if case in FOCAL:
        return '', FOCAL[case]
    env = dict(os.environ, DSS_QUERY=str(qgen.parent / 'queries'))
    output = subprocess.check_output(
        [str(qgen), '-d', '-s', str(scale), case[1:]], cwd=qgen.parent,
        env=env, text=True,
    )
    row_limit = re.search(r'^where rownum <= (-?\d+);$', output, flags=re.M)
    if not row_limit:
        raise ValueError(f'Missing qgen limit directive in {case}')
    sql = re.sub(r'^--.*$|^where rownum <= -?\d+;$', '', output, flags=re.M).strip()
    setup = ''
    if case == 'Q1':
        sql = sql.replace('day (3)', 'day')
    if case == 'Q15':
        view, sql, _drop, _empty = sql.split(';')
        setup = view.replace('create view', 'create temp view', 1) + ';'
    sql = sql.strip().removesuffix(';')
    if int(row_limit[1]) > 0:
        sql += f'\nLIMIT {row_limit[1]}'
    return setup, sql


def all_nodes(plan):
    yield plan
    for child in plan.get('Plans', []):
        yield from all_nodes(child)


def run_case(args, scale, case, screen):
    setup, query = build_case(case, scale, args.qgen)
    target = args.output / f'sf{scale}' / ('screen' if screen else 'timed') / case
    target.mkdir(parents=True, exist_ok=True)
    modes = ('native_serial', 'gftr') if screen or case not in FOCAL and case != 'Q13' else MODES
    script = [
        "LOAD 'pg_gftr';", 'SET pg_gftr.join_enabled = off;', 'SET pg_gftr.force = on;',
        "SET pg_gftr.backend = 'cuda';", 'SET pg_gftr.max_input_mb = 1024;',
        "SET jit = off;", "SET work_mem = '256MB';",
        'SET max_parallel_workers = 4;', "SET statement_timeout = '5min';", setup,
        r'\echo CASE_RECORD settings',
        "SELECT json_object_agg(name, setting) FROM pg_settings WHERE name IN "
        "('server_version','jit','work_mem','max_parallel_workers','shared_buffers',"
        "'hash_mem_multiplier','effective_cache_size','max_worker_processes');",
    ]

    def configure(mode):
        workers = 4 if mode == 'native_parallel' else 0
        enabled = 'on' if mode in ('gfur', 'gftr') else 'off'
        script.extend([f'SET max_parallel_workers_per_gather = {workers};',
                       f'SET pg_gftr.enabled = {enabled};',
                       f"SET pg_gftr.mode = '{mode if mode == 'gfur' else 'gftr'}';"])

    trials = args.standard_trials if case.startswith('Q') else args.custom_trials
    order_by_trial = {}
    if screen:
        for mode in modes:
            configure(mode)
            script.extend([f'\\echo CASE_RECORD {mode}', f'EXPLAIN (FORMAT JSON) {query};'])
    else:
        for mode in modes:
            configure(mode)
            script.extend([
                f'\\echo CASE_RECORD types_{mode}', query + '\n\\gdesc',
                f'\\o rows_{mode}.txt', f'COPY ({query}) TO STDOUT;', r'\o',
            ])
        rng = random.Random(SEED)
        for trial in range(trials + 1):
            order = list(modes)
            rng.shuffle(order)
            order_by_trial[trial] = order
            for mode in order:
                configure(mode)
                script.extend([
                    f'\\echo CASE_RECORD trial_{trial}_{mode}',
                    f'EXPLAIN (ANALYZE, BUFFERS, TIMING OFF, FORMAT JSON) {query};',
                ])
    sql = '\n'.join(script) + '\n'
    (target / 'workload.sql').write_text(sql)
    with (target / 'stdout.txt').open('w') as stdout, (target / 'stderr.txt').open('w') as stderr:
        subprocess.run([args.psql, '-X', '-qAt', '-v', 'ON_ERROR_STOP=1', '-d', f'tpch_sf{scale}'],
                       cwd=target, input=sql, text=True, stdout=stdout, stderr=stderr, check=True)
    records = {}
    for block in (target / 'stdout.txt').read_text().split('CASE_RECORD ')[1:]:
        name, value = block.split('\n', 1)
        records[name.strip()] = value.strip()
    report = {'case': case, 'scale': scale, 'settings': json.loads(records['settings']),
              'sql': query, 'setup': setup, 'plans': [], 'mode_order': order_by_trial, 'seed': SEED}
    if screen:
        for mode in modes:
            plan = json.loads(records[mode])[0]
            report['plans'].append({'mode': mode, 'explain': plan})
    else:
        native_rows = None
        signatures = {}
        for mode in modes:
            data_file = target / f'rows_{mode}.txt'
            rows = sorted(data_file.read_bytes().splitlines(keepends=True))
            description = records[f'types_{mode}']
            if native_rows is None:
                native_rows = rows
            elif native_rows != rows or records['types_native_serial'] != description:
                raise ValueError(f'{case} SF{scale}: result or type mismatch for {mode}')
            digest = hashlib.sha256()
            digest.update(description.encode())
            for row in rows:
                digest.update(row)
            signatures[mode] = {'rows': len(rows), 'sha256': digest.hexdigest(), 'types': description}
        for mode in modes:
            (target / f'rows_{mode}.txt').unlink()
        report['exact_checks'] = signatures
        for trial in range(trials + 1):
            for mode in order_by_trial[trial]:
                plan = json.loads(records[f'trial_{trial}_{mode}'])[0]
                report['plans'].append({'trial': trial, 'warmup': trial == 0, 'mode': mode, 'explain': plan})
    for run in report['plans']:
        nodes = list(all_nodes(run['explain']['Plan']))
        providers = [node for node in nodes if node.get('Custom Plan Provider')]
        actual_gpu = [node for node in providers if node.get('Custom Plan Provider') == 'GftrAggregate'
                      and node.get('Backend') == 'CUDA']
        mode = run['mode']
        if len(providers) != len(actual_gpu) or (mode.startswith('native') and providers):
            raise ValueError(f'{case} SF{scale}: unexpected custom plan for {mode}')
        if mode in ('gfur', 'gftr') and (case in FOCAL or case == 'Q13') and len(actual_gpu) != 1:
            raise ValueError(f'{case} SF{scale}: expected one CUDA aggregate for {mode}')
        if any(not node['Algorithm'].startswith(mode.upper()) for node in actual_gpu):
            raise ValueError(f'{case} SF{scale}: wrong CUDA algorithm for {mode}')
        run['gpu_nodes'] = len(actual_gpu)
        run['workers_launched'] = sum(node.get('Workers Launched', 0) for node in nodes)
        run['temp_written_blocks'] = run['explain']['Plan'].get('Temp Written Blocks', 0)
        run['phases_ms'] = [{name: node[name] for name in PHASES if name in node} for node in actual_gpu]
    (target / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    if screen:
        return sum(run['gpu_nodes'] for run in report['plans'])
    summary = {}
    for mode in modes:
        values = [run['explain']['Execution Time'] for run in report['plans']
                  if run['mode'] == mode and not run['warmup']]
        summary[mode] = {'median': statistics.median(values), 'min': min(values), 'max': max(values)}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qgen', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--psql', default='psql')
    parser.add_argument('--scales', nargs='+', type=int, choices=(1, 5), default=[1, 5])
    parser.add_argument('--cases', nargs='+', choices=(*FOCAL, *(f'Q{i}' for i in range(1, 23))),
                        default=[*FOCAL, 'Q13'])
    parser.add_argument('--screen', action='store_true')
    parser.add_argument('--custom-trials', type=int, default=5)
    parser.add_argument('--standard-trials', type=int, default=3)
    args = parser.parse_args()
    args.qgen = args.qgen.resolve()
    args.output = args.output.resolve()
    failures = []
    for scale in args.scales:
        for case in args.cases:
            try:
                result = run_case(args, scale, case, args.screen)
            except (ValueError, subprocess.SubprocessError) as error:
                failures.append((scale, case))
                result = {'error': str(error)}
            print(json.dumps({'scale': scale, 'case': case, 'result': result}), flush=True)
    if failures:
        raise SystemExit(f'Failed cases: {failures}')


if __name__ == '__main__':
    main()
