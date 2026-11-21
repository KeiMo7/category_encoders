"""Before/after cost of the opt-in mapping integrity check.

Compares ``OrdinalEncoder`` fit and transform with ``mapping_integrity=False``
(the default, unchanged code path) and ``mapping_integrity=True``, and measures
chunked manifest save/load for a 50k-category column.

Usage: python3 benchmarks/repro-mapping-integrity.py [--rows 100000] [--cols 10]
       [--cats 20] [--repeat 3]

The manifest is built once per fit and verified once per distinct mapping
content; repeated transforms with an unchanged mapping and manifest are served
from a cache keyed on the mapping signature, so the steady-state transform cost
is the existing encode path only. No parallelism is introduced: the digest is a
single sequential SHA-256 over the sorted, canonicalized rows, so results do not
depend on thread count, and the fit timestamp is metadata that never enters the
digest.
"""

import argparse
import sys
import time
import tracemalloc
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import category_encoders as ce
from category_encoders.mapping_manifest import MappingManifest


def _frame(rows: int, cols: int, cats: int) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({f'c{i}': rng.integers(0, cats, rows).astype(str) for i in range(cols)})


def _measure(frame: pd.DataFrame, integrity: bool) -> dict:
    # Time first, then measure peak memory in a separate run: tracemalloc's
    # per-allocation bookkeeping makes the manifest's many small Python objects
    # look far more expensive than they are in wall-clock terms.
    started = time.perf_counter()
    encoder = ce.OrdinalEncoder(mapping_integrity=integrity).fit(frame)
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    encoder.transform(frame)
    transform_seconds = time.perf_counter() - started

    tracemalloc.start()
    memory_encoder = ce.OrdinalEncoder(mapping_integrity=integrity).fit(frame)
    memory_encoder.transform(frame)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {'fit_s': fit_seconds, 'transform_s': transform_seconds, 'peak_mb': peak / 1e6}


def main() -> None:
    """Print the before/after table and the high-cardinality manifest timings."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--rows', type=int, default=100_000)
    parser.add_argument('--cols', type=int, default=10)
    parser.add_argument('--cats', type=int, default=20)
    parser.add_argument('--repeat', type=int, default=3)
    args = parser.parse_args()

    warnings.simplefilter('ignore')
    frame = _frame(args.rows, args.cols, args.cats)

    # warm up pandas' string/arrow and index caches so the comparison is not
    # dominated by first-call overhead in whichever path runs first
    _measure(frame.iloc[:1000], False)
    _measure(frame.iloc[:1000], True)

    results = {False: [], True: []}
    for _ in range(args.repeat):
        for integrity in (False, True):
            results[integrity].append(_measure(frame, integrity))

    for integrity in (False, True):
        runs = results[integrity]
        label = 'integrity on ' if integrity else 'integrity off'
        fit = min(run['fit_s'] for run in runs)
        transform = min(run['transform_s'] for run in runs)
        peak = min(run['peak_mb'] for run in runs)
        print(
            f'{label}: fit {fit * 1000:7.1f} ms  '
            f'transform {transform * 1000:7.1f} ms  peak {peak:7.1f} MB'
        )

    off, on = results[False], results[True]

    def ratio(key: str) -> float:
        return min(run[key] for run in on) / min(run[key] for run in off) - 1

    print(
        'overhead: '
        f'fit {ratio("fit_s") * 100:5.1f}%  '
        f'transform {ratio("transform_s") * 100:5.1f}%  '
        f'peak {ratio("peak_mb") * 100:5.1f}%'
    )

    high_card = pd.DataFrame({'id': [f'id-{i}' for i in range(50_000)]})
    started = time.perf_counter()
    encoder = ce.OrdinalEncoder(mapping_integrity=True).fit(high_card)
    fit_seconds = time.perf_counter() - started
    path = Path('/tmp/mapping-integrity-highcard.jsonl')
    started = time.perf_counter()
    encoder.mapping_manifest_.save(path, chunk_size=4096)
    save_seconds = time.perf_counter() - started
    started = time.perf_counter()
    reloaded = MappingManifest.load(path)
    load_seconds = time.perf_counter() - started
    digest_equal = reloaded.content_digest == encoder.mapping_manifest_.content_digest
    print(
        f'high cardinality (50k): fit {fit_seconds * 1000:.0f} ms  '
        f'save {save_seconds * 1000:.0f} ms  load+verify {load_seconds * 1000:.0f} ms  '
        f'round-trip digest equal: {digest_equal}'
    )


if __name__ == '__main__':
    main()
