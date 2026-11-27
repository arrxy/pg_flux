#!/usr/bin/env python3
"""Measure join offload by build selectivity and SQL projection width."""
import os

import join_benchmark


def join_query(dimension, dimension_key, probe_key, limit, wide):
    # Count every non-null column used by the query without client-side output.
    # The second and subsequent sums keep their distinct projected columns live.
    aggregates = ["count(*)", f"sum(d.{dimension_key})"]
    if wide:
        aggregates.extend(f"sum(l.{name})" for name in
                          ("l_orderkey", "l_linenumber", "l_partkey", "l_suppkey")
                          if name != probe_key)
        aggregates.append("sum(l.l_linenumber * l.l_orderkey::bigint)")
    return (
        f"SELECT {', '.join(aggregates)} "
        f"FROM lineitem l JOIN {dimension} d ON l.{probe_key} = d.{dimension_key} "
        f"WHERE l.l_orderkey BETWEEN 1 AND 200000 AND d.{dimension_key} <= {limit}"
    )


def cases(study):
    if study == "calibration":
        dimension = ("part", "p_partkey", "l_partkey")
        limits = (2000, 10000, 100000, 200000)
    elif study == "validation":
        dimension = ("supplier", "s_suppkey", "l_suppkey")
        limits = (100, 500, 5000, 10000)
    else:
        raise ValueError(f"unexpected crossover study {study!r}")
    return {
        f"{dimension[0]}_{limit}_{'wide' if wide else 'narrow'}":
        join_query(*dimension, limit, wide)
        for limit in limits
        for wide in (False, True)
    }


if __name__ == "__main__":
    join_benchmark.QUERIES = cases(os.environ.get("PG_FLUX_CROSSOVER_STUDY", "calibration"))
    join_benchmark.main()
