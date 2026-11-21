"""Export a mapping manifest and demonstrate the mismatch interception.

Run with ``python examples/mapping_integrity/demo.py`` from the repository root.
Writes ``mapping.review.csv`` and ``mapping.manifest.jsonl`` next to this file
and prints one normal encoding, one table export and one intercepted mismatch.
"""

import json
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import category_encoders as ce  # noqa: E402
from category_encoders.mapping_manifest import MappingMismatchError  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent

TRAIN = pd.DataFrame(
    {
        'city': ['chicago', 'st louis', 'detroit', 'chicago', 'boston', 'st louis'],
        'tier': ['gold', 'silver', 'gold', 'bronze', 'silver', 'bronze'],
    }
)
INFER = pd.DataFrame(
    {'city': ['chicago', 'st louis', 'detroit'], 'tier': ['gold', 'silver', 'gold']}
)


def main() -> None:
    """Fit with integrity, export the table, then intercept a stale mapping."""
    encoder = ce.OrdinalEncoder(mapping_integrity=True, handle_unknown='value').fit(TRAIN)

    print('1. normal encoding (verified against the fitted manifest)')
    print(encoder.transform(INFER).to_string(index=False))

    print('\n2. content digest')
    print(encoder.mapping_manifest_.content_digest)

    print('\n3. export for manual review (category / code / sample count)')
    print(encoder.mapping_manifest_.export_csv(HERE / 'mapping.review.csv'))
    print((HERE / 'mapping.review.csv').read_text())

    print('4. saved manifest (chunked JSON Lines, chunk_size=2)')
    encoder.mapping_manifest_.save(HERE / 'mapping.manifest.jsonl', chunk_size=2)
    for line in (HERE / 'mapping.manifest.jsonl').read_text().splitlines()[:3]:
        print(line)

    print('\n5. regression: an inference-side copy still carries the old codes')
    stale = ce.OrdinalEncoder(mapping_integrity=True).fit(TRAIN)
    old_series = stale.mapping[0]['mapping'].copy(deep=True)
    # an older mapping assigned chicago a different code
    recoded = old_series.copy()
    recoded.loc['chicago'] = 9
    stale.mapping[0]['mapping'] = recoded
    stale.handle_unknown = 'error'
    try:
        stale.transform(INFER)
    except MappingMismatchError as error:
        print(error)
        print('\nstale =', error.stale)
        print('differences =', json.dumps(error.differences, indent=2, ensure_ascii=False))
    finally:
        stale.mapping[0]['mapping'] = old_series


if __name__ == '__main__':
    main()
