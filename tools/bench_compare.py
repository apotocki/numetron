#!/usr/bin/env python3
# Numetron — Compile-time and runtime arbitrary-precision arithmetic
# (c) Alexander Pototskiy
# Licensed under the MIT License. See LICENSE file for details.

"""Compare two sets of numetron_bench_mul --csv outputs: "before" and "after".

Each set is one or more CSV files (repeated runs of the same build). Per point (section, shape,
column) the median over the runs of a set is taken, then the ratio after / before: above 1.00 is
slower. The GMP columns do not depend on numetron's build, so their ratios show how much the
machine itself drifted between the two sets -- the noise to read the numetron ratios against
(compare a set with itself, split into halves, for the noise of a single build).

    bench_compare.py --before base-*.csv --after new-*.csv [--threshold 0.03] [--normalize-gmp]

Output: the summary per section and column over size groups (median and max ratio), then every
point above the threshold.
"""

import argparse
import csv
import statistics
import sys
from collections import defaultdict

# size groups of the summary, by the larger operand dimension in limbs
GROUPS = [(1, 2), (3, 8), (9, 16), (17, 32), (33, 47), (48, None)]


def group_label(lo, hi):
    return f"{lo}+" if hi is None else (f"{lo}" if lo == hi else f"{lo}-{hi}")


def read_set(paths):
    """{(section, shape, column): [ns per run]}, and the configuration lines of the first file."""
    points = defaultdict(list)
    config = []
    for i, path in enumerate(paths):
        with open(path, newline="") as f:
            for row in csv.reader(f):
                if not row:
                    continue
                if row[0].startswith("#"):
                    if i == 0:
                        config.append(",".join(row).lstrip("# "))
                    continue
                section, shape, column, ns = row
                points[(section, shape, column)].append(float(ns))
    return {k: statistics.median(v) for k, v in points.items()}, config


def shape_size(shape):
    """The larger dimension of "n", "n*" or "unxvn"."""
    dims = [int(d) for d in shape.rstrip("*").split("x") if d]
    return max(dims) if dims else 0


def size_group(n):
    for lo, hi in GROUPS:
        if n >= lo and (hi is None or n <= hi):
            return (lo, hi)
    return None


def is_gmp(column):
    return column.startswith("gmp")


def gmp_column_for(column):
    """The GMP column timing the same operation as a numetron one, for --normalize-gmp."""
    return {"plain": "gmp", "reuse": "gmp", "sqr": "gmp_sqr", "mul": "gmp_mul",
            "add": "gmp_add", "sub": "gmp_sub"}.get(column)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--before", nargs="+", required=True, help="CSV files of the baseline build")
    ap.add_argument("--after", nargs="+", required=True, help="CSV files of the changed build")
    ap.add_argument("--threshold", type=float, default=0.03,
                    help="list the points slower than 1 + THRESHOLD (default 0.03)")
    ap.add_argument("--normalize-gmp", action="store_true",
                    help="divide each numetron ratio by the GMP ratio of the same point (machine drift)")
    args = ap.parse_args()

    before, before_config = read_set(args.before)
    after, after_config = read_set(args.after)

    print("before:", len(args.before), "run(s)")
    for line in before_config:
        print("  " + line)
    print("after:", len(args.after), "run(s)")
    for line in after_config:
        print("  " + line)
    if args.normalize_gmp:
        print("numetron ratios divided by the GMP ratio of the same point")
    print()

    common = sorted(set(before) & set(after))
    missing = sorted(set(before) ^ set(after))
    if not common:
        print("no common points", file=sys.stderr)
        return 1

    ratios = {}
    for key in common:
        section, shape, column = key
        r = after[key] / before[key]
        if args.normalize_gmp and not is_gmp(column):
            g = gmp_column_for(column)
            gk = (section, shape, g)
            if g and gk in before and gk in after:
                r /= after[gk] / before[gk]
        ratios[key] = r

    # summary: section / column x size group -> median, max
    table = defaultdict(lambda: defaultdict(list))
    for (section, shape, column), r in ratios.items():
        g = size_group(shape_size(shape))
        if g:
            table[(section, column)][g].append(r)

    used_groups = [g for g in GROUPS if any(g in cols for cols in table.values())]
    head = f"{'section':<18}{'column':<10}" + "".join(f"{group_label(*g):>16}" for g in used_groups) + f"{'all':>16}"
    print("after / before: median (max) per size group, limbs")
    print(head)
    print("-" * len(head))
    for (section, column) in sorted(table):
        cells = table[(section, column)]
        line = f"{section:<18}{column:<10}"
        for g in used_groups:
            v = cells.get(g)
            line += f"{statistics.median(v):>9.3f} ({max(v):.3f})" if v else f"{'':>16}"
        every = [r for v in cells.values() for r in v]
        line += f"{statistics.median(every):>9.3f} ({max(every):.3f})"
        print(line)

    limit = 1.0 + args.threshold
    slow = [(k, r) for k, r in ratios.items() if not is_gmp(k[2]) and r > limit]
    print()
    print(f"numetron points slower than {limit:.3f}: {len(slow)}")
    for (section, shape, column), r in sorted(slow, key=lambda x: -x[1]):
        b, a = before[(section, shape, column)], after[(section, shape, column)]
        print(f"  {section:<18}{shape:>10}  {column:<8}{b:>14.2f} -> {a:<14.2f}ns  {r:.3f}")

    if missing:
        print()
        print(f"{len(missing)} point(s) in only one of the sets (ignored), e.g. {missing[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
