"""Generate the composite-key deliverables into output/.

Writes:
- output/composite_report.json     fitted composite key reports (hash prefixes only)
- output/separator_comparison.csv  naive join vs escaped composite key, side by side
- output/count_encoded.csv         CountEncoder output on separator-laden data
- output/onehot_encoded.csv        OneHotEncoder output (hash-prefixed dummy names)
- output/hashing_encoded.csv       HashingEncoder output
- output/unknown_marker.json       unknown-combination marker left at transform time
- output/test_summary.txt          pytest summary of the encoder test files
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import category_encoders as ce
import category_encoders.utils as util

OUT = Path(__file__).resolve().parent.parent / 'output'
OUT.mkdir(exist_ok=True)

rng = np.random.RandomState(7)
n = 240
X = pd.DataFrame(
    {
        'product': rng.choice(['phone', 'laptop', 'tablet'], n),
        'color': rng.choice(['red', 'green', 'blue', 'black'], n),
        'region': rng.choice(['north', 'south'], n),
    }
)

# --- separator collision comparison (done-criterion 1) -----------------------
sep = pd.DataFrame(
    {
        'product': ['a|b', 'a', 'a|b', 'x', 'back\\'],
        'color': ['c', 'b|c', 'c', 'r', 'slash'],
    }
)
naive = sep['product'] + '|' + sep['color']
escaped = ce.CountEncoder._join_composite(
    sep, ('product', 'color'), 'product|color', escape=True
)
comparison = sep.copy()
comparison['naive_join'] = naive
comparison['composite_key'] = escaped
comparison['naive_collides'] = naive.duplicated(keep=False)
comparison.to_csv(OUT / 'separator_comparison.csv', index=False)

# --- frequency / one-hot / hashing outputs on separator-laden data -----------
X_sep = pd.DataFrame(
    {
        'product': ['a|b', 'a', 'a|b', 'x', 'x', 'a', 'back\\', 'back\\'],
        'color': ['c', 'b|c', 'c', 'r', 'r', 'b|c', 'slash', 'slash'],
        'region': ['n', 'n', 's', 's', 'n', 's', 'n', 'n'],
    }
)

count = ce.CountEncoder(composite_cols=[('product', 'color')])
count.fit_transform(X_sep).to_csv(OUT / 'count_encoded.csv', index=False)

onehot = ce.OneHotEncoder(composite_cols=[('product', 'color')], use_cat_names=True)
onehot.fit_transform(X_sep).to_csv(OUT / 'onehot_encoded.csv', index=False)

hashing = ce.HashingEncoder(composite_cols=[('product', 'color')], n_components=8, max_process=1)
hashing.fit_transform(X_sep).to_csv(OUT / 'hashing_encoded.csv', index=False)

# --- budget merge on the 12-combo dataset ------------------------------------
lump = ce.CountEncoder(
    composite_cols=[('product', 'color')],
    max_composite_cardinality=4,
    composite_overflow='lump',
)
lump.fit(X)

report = {
    'count_separator_data': count.composite_report_,
    'onehot_separator_data': onehot.composite_report_,
    'hashing_separator_data': hashing.composite_report_,
    'count_lump_budget_4': lump.composite_report_,
}
with open(OUT / 'composite_report.json', 'w') as f:
    json.dump(report, f, indent=2, default=str)

# --- unknown-combination marker (req. 6) --------------------------------------
X_new = X.head(3).copy()
X_new['product'] = 'telegraph'
X_new['color'] = 'red'
count_unknown = ce.CountEncoder(composite_cols=[('product', 'color')]).fit(X)
count_unknown.transform(X_new)
with open(OUT / 'unknown_marker.json', 'w') as f:
    json.dump(count_unknown.unknown_composites_, f, indent=2)

# --- test run summary ----------------------------------------------------------
proc = subprocess.run(
    [
        sys.executable, '-m', 'pytest',
        'tests/test_count.py', 'tests/test_one_hot.py', 'tests/test_hashing.py',
        'tests/test_composite_cols.py', 'tests/test_composite_unsupervised.py',
        '-q', '--no-header',
    ],
    capture_output=True,
    text=True,
    cwd=OUT.parent,
)
summary = (
    'pytest tests/test_count.py tests/test_one_hot.py tests/test_hashing.py '
    'tests/test_composite_cols.py tests/test_composite_unsupervised.py\n'
    f'exit code: {proc.returncode}\n'
    + '\n'.join(proc.stdout.strip().splitlines()[-3:])
    + '\n'
)
(OUT / 'test_summary.txt').write_text(summary)
print(summary)
print('artifacts written to', OUT)
