"""Tests for composite_cols on CountEncoder, OneHotEncoder and HashingEncoder.

Composite keys are a formal input of the frequency, one-hot and hashing paths:
the key is built from the member columns by the encoder itself (same code at
fit and at transform), escaping keeps member values containing the separator
unambiguous, and only hash prefixes of keys may appear in outputs and errors.
"""

import hashlib
from unittest import TestCase

import category_encoders as encoders
import category_encoders.utils as util
import numpy as np
import pandas as pd


def make_dataset() -> pd.DataFrame:
    """Small deterministic dataset with low-cardinality categorical columns."""
    rng = np.random.RandomState(7)
    n_rows = 240
    return pd.DataFrame(
        {
            'product': rng.choice(['phone', 'laptop', 'tablet'], n_rows),
            'color': rng.choice(['red', 'green', 'blue', 'black'], n_rows),
            'region': rng.choice(['north', 'south'], n_rows),
        }
    )


def separator_dataset() -> pd.DataFrame:
    """Member values deliberately containing the '|' separator (and the escape).

    ('a|b', 'c') and ('a', 'b|c') collapse to the same naive 'a|b|c' join;
    ('back\\', 'slash') checks the escape prefix itself.
    """
    return pd.DataFrame(
        {
            'product': ['a|b', 'a', 'a|b', 'x', 'x', 'a', 'back\\', 'back\\'],
            'color': ['c', 'b|c', 'c', 'r', 'r', 'b|c', 'slash', 'slash'],
            'region': ['n', 'n', 's', 's', 'n', 's', 'n', 'n'],
        }
    )


def manual_counts(X: pd.DataFrame, members=('product', 'color')) -> pd.Series:
    """Group counts per member tuple, the manual grouping the encoders must match."""
    return X.groupby(list(members), dropna=False).size()


def escaped_key(values) -> str:
    """The composite key the encoder must build for a member tuple."""
    return '|'.join(util.escape_composite_member(str(v)) for v in values)


class TestCompositeSeparatorSafety(TestCase):
    """Member values containing the separator must not collide (req. 4, done 1)."""

    def test_escaped_join_is_injective(self):
        X = separator_dataset()
        joined = encoders.CountEncoder._join_composite(
            X, ('product', 'color'), 'product|color', escape=True
        )
        naive = X['product'] + '|' + X['color']
        # the naive join collapses ('a|b','c') and ('a','b|c'); the escaped join does not
        self.assertEqual(naive[0], naive[1])
        self.assertNotEqual(joined[0], joined[1])
        self.assertEqual(joined.nunique(), len(manual_counts(X)))

    def test_count_keys_match_manual_groups(self):
        X = separator_dataset()
        enc = encoders.CountEncoder(composite_cols=[('product', 'color')])
        enc.fit(X)
        mapping = enc.mapping['product|color']
        expected = manual_counts(X)
        self.assertEqual(len(mapping), len(expected))
        for tup, count in expected.items():
            self.assertEqual(mapping[escaped_key(tup)], count)

    def test_onehot_columns_match_manual_groups(self):
        X = separator_dataset()
        enc = encoders.OneHotEncoder(composite_cols=[('product', 'color')], use_cat_names=True)
        result = enc.fit_transform(X)
        for tup in manual_counts(X).index:
            prefix = util.composite_key_hash_prefix(escaped_key(tup))
            col = f'product|color_k{prefix}'
            self.assertIn(col, result.columns)
            expected = ((X['product'] == tup[0]) & (X['color'] == tup[1])).astype(int)
            pd.testing.assert_series_equal(result[col], expected, check_names=False)

    def test_hashing_buckets_match_manual_hashes(self):
        X = separator_dataset()[['product', 'color']]
        enc = encoders.HashingEncoder(
            composite_cols=[('product', 'color')], n_components=16, max_process=1
        )
        result = enc.fit_transform(X)
        self.assertEqual(list(result.columns), [f'col_{i}' for i in range(16)])
        for i, tup in enumerate(zip(X['product'], X['color'])):
            digest = hashlib.md5(escaped_key(tup).encode('utf-8')).digest()
            bucket = int.from_bytes(digest, byteorder='big') % 16
            self.assertEqual(result.loc[i, f'col_{bucket}'], 1)
            self.assertEqual(result.loc[i].sum(), 1)


class TestCompositeStatsMatchManualGrouping(TestCase):
    """Frequency, one-hot and hashing statistics equal the manual grouping (done 2)."""

    def test_count_equals_manual_groupby(self):
        X = make_dataset()
        enc = encoders.CountEncoder(composite_cols=[('product', 'color')])
        result = enc.fit_transform(X)
        expected = X.groupby(['product', 'color'])['region'].transform('size')
        pd.testing.assert_series_equal(
            result['product|color'], expected, check_names=False, check_dtype=False
        )

    def test_count_normalize_equals_manual_groupby(self):
        X = make_dataset()
        enc = encoders.CountEncoder(composite_cols=[('product', 'color')], normalize=True)
        result = enc.fit_transform(X)
        expected = X.groupby(['product', 'color'])['region'].transform('size') / len(X)
        pd.testing.assert_series_equal(result['product|color'], expected, check_names=False)

    def test_onehot_equals_manual_indicators(self):
        X = make_dataset()
        enc = encoders.OneHotEncoder(composite_cols=[('product', 'color')])
        result = enc.fit_transform(X)
        groups = manual_counts(X)
        n_composite_cols = len([c for c in result.columns if c.startswith('product|color_')])
        self.assertEqual(n_composite_cols, len(groups))
        # each composite dummy row sums to 1 and rows of the same combo agree
        composite_cols = [c for c in result.columns if c.startswith('product|color_')]
        self.assertTrue((result[composite_cols].sum(axis=1) == 1).all())
        key = X['product'] + '|' + X['color']
        for combo in key.unique():
            rows = result.loc[key == combo, composite_cols]
            self.assertTrue((rows == rows.iloc[0]).all().all())

    def test_hashing_equals_manual_md5(self):
        X = make_dataset()[['product', 'color']]
        enc = encoders.HashingEncoder(
            composite_cols=[('product', 'color')], n_components=8, max_process=1
        )
        result = enc.fit_transform(X)
        for i, tup in enumerate(zip(X['product'], X['color'])):
            digest = hashlib.md5(escaped_key(tup).encode('utf-8')).digest()
            bucket = int.from_bytes(digest, byteorder='big') % 8
            self.assertEqual(result.loc[i, f'col_{bucket}'], 1)

    def test_composite_stats_do_not_read_target(self):
        """Fitting with and without y gives identical frequency/hash statistics."""
        X = make_dataset()
        y = pd.Series(np.random.RandomState(0).binomial(1, 0.5, len(X)))
        plain = encoders.CountEncoder(composite_cols=[('product', 'color')]).fit(X)
        with_y = encoders.CountEncoder(composite_cols=[('product', 'color')]).fit(X, y)
        for col in plain.mapping:
            pd.testing.assert_series_equal(plain.mapping[col], with_y.mapping[col])


class TestCompositeMissingUnknownBudget(TestCase):
    """Missing members, unknown combos and cardinality budget (reqs 5-8, done 3)."""

    def test_missing_member_counted_as_missing(self):
        X = make_dataset()
        X.loc[0, 'color'] = np.nan
        X.loc[1, 'product'] = np.nan
        enc = encoders.CountEncoder(composite_cols=[('product', 'color')])
        result = enc.fit_transform(X)
        # rows with any missing member share the missing composite key
        self.assertEqual(result.loc[0, 'product|color'], 2.0)
        self.assertEqual(result.loc[1, 'product|color'], 2.0)
        self.assertEqual(enc.composite_report_['product|color']['missing_rows'], 2)
        # and the missing group is counted separately from every normal combo
        normal = result.loc[2:, 'product|color']
        self.assertEqual(normal.max(), manual_counts(X.loc[2:]).max())

    def test_missing_member_onehot_indicator(self):
        X = make_dataset()
        X.loc[[0, 1], 'color'] = np.nan
        enc = encoders.OneHotEncoder(
            composite_cols=[('product', 'color')], handle_missing='indicator'
        )
        result = enc.fit_transform(X)
        composite_cols = [c for c in result.columns if c.startswith('product|color_')]
        self.assertTrue((result.loc[[0, 1], composite_cols].sum(axis=1) == 1).all())

    def test_unknown_combo_uses_unknown_strategy_and_leaves_marker(self):
        X = make_dataset()
        enc = encoders.CountEncoder(composite_cols=[('product', 'color')])
        enc.fit(X)
        X_new = X.head(3).copy()
        X_new['product'] = 'telegraph'  # unseen combination
        X_new['color'] = 'red'  # one unseen combo repeated three times
        result = enc.transform(X_new)
        # handle_unknown='value' fills unseen combos with 0
        self.assertTrue((result['product|color'] == 0).all())
        marker = enc.unknown_composites_['product|color']
        self.assertEqual(sum(marker.values()), 3)
        self.assertEqual(len(marker), 1)
        self.assertEqual(enc.composite_report_['product|color']['unknown_keys'], 1)
        # the marker is a hash prefix, never the raw business key
        self.assertNotIn('telegraph', str(marker))

    def test_unknown_combo_error_strategy_raises(self):
        X = make_dataset()
        enc = encoders.CountEncoder(composite_cols=[('product', 'color')], handle_unknown='error')
        enc.fit(X)
        X_new = X.head(3).copy()
        X_new['product'] = 'telegraph'
        with self.assertRaises(ValueError):
            enc.transform(X_new)

    def test_cardinality_budget_without_strategy_is_rejected(self):
        X = make_dataset()  # 3 x 4 = 12 combos
        enc = encoders.CountEncoder(
            composite_cols=[('product', 'color')], max_composite_cardinality=4
        )
        with self.assertRaises(ValueError) as ctx:
            enc.fit(X)
        self.assertIn('max_composite_cardinality', str(ctx.exception))

    def test_lump_overflow_merges_and_reports(self):
        X = make_dataset()
        enc = encoders.CountEncoder(
            composite_cols=[('product', 'color')],
            max_composite_cardinality=4,
            composite_overflow='lump',
        )
        result = enc.fit_transform(X)
        report = enc.composite_report_['product|color']
        self.assertEqual(report['cardinality'], 12)
        self.assertEqual(report['merged_keys'], 12 - 3)
        self.assertEqual(report['kept_keys'], 4)
        # the leftovers key carries the summed counts of the merged combos
        counts = manual_counts(X).sort_values(ascending=False)
        lumped_total = counts.iloc[3:].sum()
        self.assertIn(lumped_total, set(result['product|color']))
        # no full key leaks: the leftovers token is a fixed marker
        self.assertNotIn(util.COMPOSITE_LUMP_TOKEN, str(result.columns))

    def test_hash_overflow_bounds_onehot_columns(self):
        X = make_dataset()
        enc = encoders.OneHotEncoder(
            composite_cols=[('product', 'color')],
            max_composite_cardinality=4,
            composite_overflow='hash',
        )
        result = enc.fit_transform(X)
        composite_cols = [c for c in result.columns if c.startswith('product|color_')]
        report = enc.composite_report_['product|color']
        self.assertLessEqual(report['kept_keys'], 4)
        self.assertEqual(report['merged_keys'], 12 - report['kept_keys'])
        self.assertEqual(len(composite_cols), report['kept_keys'])
        self.assertTrue((result[composite_cols].sum(axis=1) == 1).all())

    def test_hashing_collision_report(self):
        X = make_dataset()
        enc = encoders.HashingEncoder(
            composite_cols=[('product', 'color')], n_components=2, max_process=1
        )
        enc.fit(X)
        report = enc.composite_report_['product|color']
        self.assertEqual(report['cardinality'], 12)
        self.assertLessEqual(report['hash_buckets_used'], 2)
        self.assertEqual(report['hash_collisions'], 12 - report['hash_buckets_used'])
        self.assertGreater(report['hash_collisions'], 0)


class TestCompositeValidationAndDeterminism(TestCase):
    """Fit-time validation and refit determinism (reqs 9, errors 1-2)."""

    def test_missing_member_column_raises(self):
        X = make_dataset()
        enc = encoders.CountEncoder(composite_cols=[('product', 'flavour')])
        with self.assertRaises(ValueError):
            enc.fit(X)

    def test_empty_member_name_raises(self):
        X = make_dataset()
        enc = encoders.CountEncoder(composite_cols=[('', 'color')])
        with self.assertRaises(ValueError):
            enc.fit(X)

    def test_name_collision_with_existing_column_raises(self):
        X = make_dataset()
        X['product|color'] = 'taken'
        enc = encoders.CountEncoder(composite_cols=[('product', 'color')])
        with self.assertRaises(ValueError):
            enc.fit(X)

    def test_invalid_overflow_strategy_raises(self):
        X = make_dataset()
        enc = encoders.CountEncoder(
            composite_cols=[('product', 'color')], composite_overflow='drop'
        )
        with self.assertRaises(ValueError):
            enc.fit(X)

    def test_residual_ambiguity_is_rejected_with_hash_prefixes(self):
        """Values that stringify identically (1 and '1') cannot be told apart."""
        X = pd.DataFrame(
            {
                'product': [1, '1'],
                'color': ['c', 'c'],
                'region': ['n', 's'],
            }
        )
        enc = encoders.CountEncoder(composite_cols=[('product', 'color')])
        with self.assertRaises(ValueError) as ctx:
            enc.fit(X)
        msg = str(ctx.exception)
        self.assertIn('ambiguous', msg)
        self.assertIn('member hashes', msg)
        # the raw conflicting values never appear, only hash prefixes
        self.assertNotIn("'1'", msg)

    def test_refit_is_deterministic(self):
        X = make_dataset()
        first = encoders.OneHotEncoder(
            composite_cols=[('product', 'color')],
            max_composite_cardinality=4,
            composite_overflow='lump',
        )
        first.fit(X)
        keys_first = [sorted(m) for m in first._composite_fit_raw_keys_.values()]
        names_first = first.get_feature_names_out().copy()
        out_first = first.transform(X)

        first.fit(X)  # refit the same instance
        self.assertEqual(keys_first, [sorted(m) for m in first._composite_fit_raw_keys_.values()])
        np.testing.assert_array_equal(names_first, first.get_feature_names_out())
        pd.testing.assert_frame_equal(out_first, first.transform(X))

        second = encoders.OneHotEncoder(
            composite_cols=[('product', 'color')],
            max_composite_cardinality=4,
            composite_overflow='lump',
        )
        pd.testing.assert_frame_equal(out_first, second.fit_transform(X))

    def test_composite_members_recorded(self):
        X = make_dataset()
        enc = encoders.CountEncoder(composite_cols=[('product', 'color')])
        enc.fit(X)
        self.assertEqual(enc.composite_members_, {'product|color': ('product', 'color')})

    def test_other_unsupervised_encoders_still_reject_composites(self):
        X = make_dataset()
        ordinal = encoders.OrdinalEncoder()
        ordinal.composite_cols = [('product', 'color')]
        with self.assertRaises(ValueError):
            ordinal.fit(X)
