"""Tests for the opt-in category coverage reporting (coverage_report=True)."""

import warnings

import category_encoders as ce
import numpy as np
import pandas as pd
import pytest
from category_encoders.coverage import CoverageWarning


@pytest.fixture()
def train():
    """Sample training frame."""
    return pd.DataFrame(
        {
            'a': ['x', 'y', 'z', 'x', 'y', 'x'],
            'b': ['p', 'q', 'p', 'p', 'q', 'p'],
            'n': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )


@pytest.fixture()
def target():
    """Sample target vector."""
    return pd.Series([0, 1, 0, 1, 0, 1])


class TestNormalCoverage:
    """Scenario: batches fully covered by the fit data."""

    def test_full_coverage_no_warning(self, train):
        """Full coverage no warning."""
        enc = ce.OrdinalEncoder(cols=['a', 'b'], coverage_report=True)
        enc.fit(train)
        with warnings.catch_warnings():
            warnings.simplefilter('error', CoverageWarning)
            enc.transform(train)
        summary = enc.coverage_summary()
        assert set(summary['status']) == {'ok'}
        assert (summary['coverage'] == 1.0).all()
        assert (summary['unseen_ratio'] == 0.0).all()

    def test_summary_available_right_after_fit(self, train):
        """Summary available right after fit."""
        # fit() itself transforms the training data, so a baseline report exists
        enc = ce.OrdinalEncoder(cols=['a', 'b'], coverage_report=True)
        enc.fit(train)
        summary = enc.coverage_summary()
        assert len(summary) == 2
        assert (summary['unseen_ratio'] == 0.0).all()

    def test_summary_table_schema(self, train):
        """Summary table schema."""
        enc = ce.OrdinalEncoder(cols=['a', 'b'], coverage_report=True)
        enc.fit(train)
        enc.transform(train)
        summary = enc.coverage_summary()
        assert list(summary.columns) == [
            'column',
            'n_fit_categories',
            'n_new_categories',
            'n_categories',
            'coverage',
            'unseen_rows',
            'n_rows',
            'unseen_ratio',
            'status',
            'lumped_categories',
            'lumped_row_fraction',
            'unseen_values',
            'error',
        ]
        assert set(summary['column']) == {'a', 'b'}

    def test_label_makes_summaries_comparable_across_datasets(self, train):
        """Label makes summaries comparable across datasets."""
        val = pd.DataFrame({'a': ['x', 'w'], 'b': ['p', 'p'], 'n': [7.0, 8.0]})
        enc = ce.OrdinalEncoder(cols=['a', 'b'], coverage_report=True)
        enc.fit(train)
        train_summary = enc.coverage_summary(label='train')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', CoverageWarning)
            enc.transform(val)
        val_summary = enc.coverage_summary(label='validation')
        combined = pd.concat([train_summary, val_summary])
        assert set(combined['dataset']) == {'train', 'validation'}
        # identical metric definitions: same columns in both slices
        assert train_summary.columns.equals(val_summary.columns)

    def test_summary_on_demand_column_subset(self, train):
        """Summary on demand column subset."""
        enc = ce.OrdinalEncoder(cols=['a', 'b'], coverage_report=True)
        enc.fit(train)
        enc.transform(train)
        summary = enc.coverage_summary(cols=['a'])
        assert list(summary['column']) == ['a']


class TestLowCoverageWarning:
    """Scenario: unseen categories trigger threshold warnings."""

    def test_warns_and_names_column_and_values(self, train):
        """Warns and names column and values."""
        val = pd.DataFrame({'a': ['x', 'w', 'w'], 'b': ['p', 'p', 'p'], 'n': [7, 8, 9]})
        enc = ce.OrdinalEncoder(cols=['a', 'b'], coverage_report=True)
        enc.fit(train)
        with pytest.warns(CoverageWarning, match=r"column 'a'.*unseen values: w"):
            enc.transform(val)
        summary = enc.coverage_summary()
        row = summary[summary['column'] == 'a'].iloc[0]
        assert row['status'] == 'low_coverage'
        assert row['unseen_ratio'] == pytest.approx(2 / 3)
        assert row['unseen_values'] == 'w'
        # the fully covered column stays quiet
        assert summary[summary['column'] == 'b'].iloc[0]['status'] == 'ok'

    def test_unseen_ratio_denominator_is_fit_union_batch(self, train):
        """Unseen ratio denominator is fit union batch."""
        # train has 3 categories {x, y, z}; the batch adds {w} -> denominator is 4,
        # not the 3 fit categories and not the 2 batch categories
        val = pd.DataFrame({'a': ['x', 'w'], 'b': ['p', 'p'], 'n': [7, 8]})
        enc = ce.OrdinalEncoder(cols=['a', 'b'], coverage_report=True)
        enc.fit(train)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', CoverageWarning)
            enc.transform(val)
        row = enc.coverage_summary().set_index('column').loc['a']
        assert row['n_categories'] == 4
        assert row['n_fit_categories'] == 3
        assert row['n_new_categories'] == 1
        assert row['coverage'] == pytest.approx(3 / 4)

    def test_threshold_suppresses_warning(self, train):
        """Threshold suppresses warning."""
        # 1/3 of rows unseen; threshold 0.5 -> no warning, status ok
        val = pd.DataFrame({'a': ['x', 'y', 'w'], 'b': ['p', 'p', 'p'], 'n': [7, 8, 9]})
        enc = ce.OrdinalEncoder(cols=['a', 'b'], coverage_report=True, coverage_threshold=0.5)
        enc.fit(train)
        with warnings.catch_warnings():
            warnings.simplefilter('error', CoverageWarning)
            enc.transform(val)
        row = enc.coverage_summary().set_index('column').loc['a']
        assert row['status'] == 'ok'
        assert row['unseen_ratio'] == pytest.approx(1 / 3)

    def test_invalid_threshold_rejected(self, train):
        """Invalid threshold rejected."""
        enc = ce.OrdinalEncoder(cols=['a'], coverage_report=True, coverage_threshold=1.5)
        with pytest.raises(ValueError, match='coverage_threshold'):
            enc.fit(train)


class TestMinGroupLumping:
    """Scenario: min_group_size lumping is described in the report."""

    def test_lumping_is_reported(self):
        """Lumping is reported."""
        # 'rare1'..'rare4' appear once each and are lumped below min_group_size=2
        values = ['common'] * 8 + ['rare1', 'rare2', 'rare3', 'rare4']
        X = pd.DataFrame({'a': values})
        enc = ce.OrdinalEncoder(cols=['a'], coverage_report=True, min_group_size=2)
        enc.fit(X)
        enc.transform(X)
        row = enc.coverage_summary().iloc[0]
        assert row['lumped_categories'] == 4
        assert row['lumped_row_fraction'] == pytest.approx(4 / 12)
        # after lumping the encoder knows 2 categories: 'common' + the leftovers group
        assert row['n_fit_categories'] == 2

    def test_lumping_is_reproducible(self):
        """Lumping is reproducible."""
        values = ['common'] * 8 + ['rare1', 'rare2', 'rare3', 'rare4']
        X = pd.DataFrame({'a': values})
        enc1 = ce.OrdinalEncoder(cols=['a'], coverage_report=True, min_group_size=2)
        enc2 = ce.OrdinalEncoder(cols=['a'], coverage_report=True, min_group_size=2)
        enc1.fit(X)
        enc2.fit(X)
        # same data + same parameters -> identical lumping maps and lumped names
        assert enc1.min_group_lumping_ == enc2.min_group_lumping_
        assert enc1.coverage_lumping_ == enc2.coverage_lumping_
        row1 = enc1.coverage_summary().drop(columns=[]).iloc[0]
        row2 = enc2.coverage_summary().iloc[0]
        assert row1['lumped_categories'] == row2['lumped_categories']
        assert row1['lumped_row_fraction'] == row2['lumped_row_fraction']

    def test_count_encoder_lumping_is_reported(self):
        """Count encoder lumping is reported."""
        values = ['common'] * 8 + ['rare1', 'rare2', 'rare3', 'rare4']
        X = pd.DataFrame({'a': values})
        enc = ce.CountEncoder(cols=['a'], coverage_report=True, min_group_size=2, normalize=False)
        enc.fit(X)
        enc.transform(X)
        row = enc.coverage_summary().iloc[0]
        assert row['lumped_categories'] == 4
        assert row['lumped_row_fraction'] == pytest.approx(4 / 12)


class TestAllNewColumn:
    """Scenario: a column whose values are all new."""

    def test_all_new_column_flagged_with_advice(self, train):
        """All new column flagged with advice."""
        val = pd.DataFrame({'a': ['new1', 'new2'], 'b': ['p', 'q'], 'n': [7, 8]})
        enc = ce.OrdinalEncoder(cols=['a', 'b'], coverage_report=True)
        enc.fit(train)
        with pytest.warns(CoverageWarning, match=r"every value of column 'a'.*refitting"):
            enc.transform(val)
        row = enc.coverage_summary().set_index('column').loc['a']
        assert row['status'] == 'all_new'
        assert row['coverage'] == pytest.approx(3 / 5)
        assert row['unseen_ratio'] == 1.0


class TestEmptyData:
    """Empty data is a hard error, never an empty report."""

    def test_empty_fit_data_raises(self):
        """Empty fit data raises."""
        enc = ce.OrdinalEncoder(cols=['a'], coverage_report=True)
        with pytest.raises(ValueError, match='non-empty fit data'):
            enc.fit(pd.DataFrame({'a': pd.Series([], dtype=object)}))

    def test_empty_transform_data_raises(self, train):
        """Empty transform data raises."""
        enc = ce.OrdinalEncoder(cols=['a', 'b'], coverage_report=True)
        enc.fit(train)
        with pytest.raises(ValueError, match='empty data'):
            enc.transform(train.iloc[0:0])


class TestDegradation:
    """Report generation failures degrade to warnings."""

    def test_report_failure_warns_but_encoding_still_works(self, train, monkeypatch):
        """Report failure warns but encoding still works."""
        import category_encoders.coverage as coverage

        def boom(*args, **kwargs):
            raise RuntimeError('simulated report failure')

        monkeypatch.setattr(coverage, 'build_report', boom)
        enc = ce.OrdinalEncoder(cols=['a', 'b'], coverage_report=True)
        with pytest.warns(CoverageWarning, match='report generation failed'):
            enc.fit(train)
        # encoding itself is unaffected
        val = pd.DataFrame({'a': ['x', 'w'], 'b': ['p', 'p'], 'n': [7, 8]})
        with pytest.warns(CoverageWarning, match='report generation failed'):
            out = enc.transform(val)
        assert len(out) == 2

    def test_missing_fit_statistics_warn_but_do_not_break(self, train):
        """Missing fit statistics warn but do not break."""
        # simulates a model pickled before coverage support existed
        enc = ce.OrdinalEncoder(cols=['a', 'b'], coverage_report=True)
        enc.fit(train)
        del enc.coverage_fit_categories_
        with pytest.warns(CoverageWarning, match='no fit-time category statistics'):
            enc.transform(train)

    def test_unhashable_values_degrade_to_column_error(self):
        """Unhashable values degrade to column error."""
        X = pd.DataFrame({'a': [['x'], ['y'], ['x']]})
        enc = ce.OrdinalEncoder.__new__(ce.OrdinalEncoder)
        # build the report directly: the unhashable column must not raise
        from category_encoders.coverage import build_report

        report = build_report(['a'], X, {}, {'a': 'unhashable type'}, {})
        cov = report.columns['a']
        assert cov.error == 'unhashable type'
        assert cov.status(0.0) == 'error'
        del enc  # not needed; construction path is covered by build_report


class TestDisabledBehaviourUnchanged:
    """coverage_report=False keeps the previous behaviour exactly."""

    def test_no_statistics_collected_when_disabled(self, train):
        """No statistics collected when disabled."""
        enc = ce.OrdinalEncoder(cols=['a', 'b'])
        enc.fit(train)
        with warnings.catch_warnings():
            warnings.simplefilter('error', CoverageWarning)
            enc.transform(train)
        assert not hasattr(enc, 'coverage_report_')
        assert not hasattr(enc, 'coverage_fit_categories_')

    def test_coverage_summary_requires_opt_in(self, train):
        """Coverage summary requires opt in."""
        enc = ce.OrdinalEncoder(cols=['a', 'b'])
        enc.fit(train)
        with pytest.raises(ValueError, match='coverage_report=True'):
            enc.coverage_summary()

    def test_encoding_identical_with_and_without_report(self, train):
        """Encoding identical with and without report."""
        val = pd.DataFrame({'a': ['x', 'w', None], 'b': ['p', 'p', 'p'], 'n': [7, 8, 9]})
        enc_off = ce.OrdinalEncoder(cols=['a', 'b']).fit(train)
        enc_on = ce.OrdinalEncoder(cols=['a', 'b'], coverage_report=True).fit(train)
        out_off = enc_off.transform(val)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', CoverageWarning)
            out_on = enc_on.transform(val)
        pd.testing.assert_frame_equal(out_off, out_on)


class TestOtherEncoders:
    """Coverage reporting works across encoder types."""

    def test_one_hot_encoder(self, train):
        """One hot encoder."""
        val = pd.DataFrame({'a': ['x', 'w'], 'b': ['p', 'p'], 'n': [7, 8]})
        enc = ce.OneHotEncoder(cols=['a', 'b'], coverage_report=True)
        enc.fit(train)
        with pytest.warns(CoverageWarning, match="column 'a'"):
            enc.transform(val)
        summary = enc.coverage_summary()
        assert len(summary) == 2

    def test_supervised_encoder(self, train, target):
        """Supervised encoder."""
        val = pd.DataFrame({'a': ['x', 'w'], 'b': ['p', 'p'], 'n': [7, 8]})
        enc = ce.TargetEncoder(cols=['a', 'b'], coverage_report=True)
        enc.fit(train, target)
        with pytest.warns(CoverageWarning, match="column 'a'"):
            enc.transform(val)
        row = enc.coverage_summary().set_index('column').loc['a']
        assert row['n_new_categories'] == 1

    def test_missing_values_count_as_one_category(self):
        """Missing values count as one category."""
        train = pd.DataFrame({'a': ['x', None, 'y']})
        val = pd.DataFrame({'a': ['x', None, np.nan]})
        enc = ce.OrdinalEncoder(cols=['a'], coverage_report=True)
        enc.fit(train)
        with warnings.catch_warnings():
            warnings.simplefilter('error', CoverageWarning)
            enc.transform(val)
        row = enc.coverage_summary().iloc[0]
        # None and nan are the same <missing> category, seen at fit time
        assert row['n_fit_categories'] == 3
        assert row['unseen_ratio'] == 0.0
