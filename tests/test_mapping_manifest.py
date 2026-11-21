"""Tests for the opt-in mapping manifest (versioned category mapping tables)."""

import json
from unittest import TestCase

import category_encoders as encoders
import numpy as np
import pandas as pd
import pytest
from category_encoders.mapping_manifest import (
    MappingDriftWarning,
    MappingManifest,
    MappingManifestCorruptionError,
    MappingManifestNotFoundError,
    MappingMismatchError,
    encode_strategy,
    strategy_values,
)

X_TRAIN = pd.DataFrame(
    {
        'city': ['chicago', 'st louis', 'detroit', 'chicago', 'boston'],
        'color': ['red', 'blue', 'red', 'green', 'blue'],
        'size': [1, 2, 3, 1, 2],
    }
)
X_INFER = pd.DataFrame(
    {
        'city': ['chicago', 'st louis', 'detroit'],
        'color': ['red', 'blue', 'red'],
        'size': [1, 2, 3],
    }
)


def fitted(**kwargs):
    """OrdinalEncoder with integrity enabled, fitted on the training fixture."""
    return encoders.OrdinalEncoder(
        cols=['city', 'color', 'size'], mapping_integrity=True, **kwargs
    ).fit(X_TRAIN)


class TestMappingContentUnchanged(TestCase):
    """The switch must not alter fit/transform output."""

    def test_fit_and_transform_outputs_identical_with_switch_on(self):
        """Encoding output and mapping are byte-for-byte unchanged with the switch on."""
        plain = encoders.OrdinalEncoder(cols=['city', 'color', 'size']).fit(X_TRAIN)
        verified = fitted()

        pd.testing.assert_frame_equal(plain.transform(X_INFER), verified.transform(X_INFER))
        for left, right in zip(plain.mapping, verified.mapping, strict=True):
            self.assertEqual(left['col'], right['col'])
            pd.testing.assert_series_equal(left['mapping'], right['mapping'])

    def test_switch_off_creates_no_manifest(self):
        """With the switch off, fit stores no manifest and transform skips verification."""
        enc = encoders.OrdinalEncoder(cols=['city']).fit(X_TRAIN)
        self.assertIsNone(enc.mapping_manifest_)
        self.assertFalse(hasattr(enc, 'mapping_integrity') and enc.mapping_integrity)
        # and transform applies no verification at all
        enc.transform(X_INFER[['city']].assign(color='red', size=1))

    def test_default_constructor_parameters_are_opt_in(self):
        """The integrity parameters default to off."""
        enc = encoders.OrdinalEncoder()
        self.assertFalse(enc.mapping_integrity)
        self.assertIsNone(enc.mapping_manifest)
        self.assertEqual(0.5, enc.mapping_drift_threshold)


class TestManifestShape(TestCase):
    """Every fitted table carries its identity, size and content digest."""

    def test_manifest_records_metadata(self):
        """The manifest records timestamp, sample size, category count and digest."""
        enc = fitted()
        manifest = enc.mapping_manifest_

        self.assertEqual('OrdinalEncoder', manifest.encoder)
        self.assertIsNotNone(manifest.fitted_at)
        self.assertEqual(len(X_TRAIN), manifest.n_samples)
        self.assertEqual(3, manifest.n_columns)
        # city: 5 categories + registered missing, color: 3 + registered missing,
        # size: 3 + registered missing
        self.assertEqual(13, manifest.n_categories)
        self.assertTrue(manifest.content_digest.startswith('sha256:'))
        self.assertIn('3 columns', manifest.summary())

    def test_unknown_policy_serialized_with_table(self):
        """The unknown-category policy is stored inside the manifest."""
        enc = encoders.OrdinalEncoder(
            cols=['city'], mapping_integrity=True, handle_unknown='error'
        ).fit(X_TRAIN)
        column = enc.mapping_manifest_.column('city')
        self.assertEqual({'kind': 'named', 'value': 'error'}, column.handle_unknown)

        # the policy is read back from the manifest, not from the code defaults
        self.assertEqual('error', strategy_values(enc.mapping_manifest_, 'handle_unknown'))

    def test_numeric_and_callable_policies_serialize(self):
        """Numeric and callable policies serialize without loss."""
        self.assertEqual({'kind': 'scalar', 'value': ['int', 42]}, encode_strategy(42))
        self.assertEqual({'kind': 'named', 'value': 'return_nan'}, encode_strategy('return_nan'))
        self.assertEqual('callable', encode_strategy(str.upper)['kind'])


class TestExport(TestCase):
    """The manifest exports aligned category / code / sample-count rows."""

    def test_export_csv_has_three_aligned_columns(self):
        """The CSV export aligns category, code and sample count."""
        enc = fitted()
        path = '/tmp/mapping_export_test.csv'
        enc.mapping_manifest_.export_csv(path)

        table = pd.read_csv(path)
        self.assertEqual(
            [
                'column',
                'category',
                'category_type',
                'encoded_value',
                'encoded_type',
                'sample_count',
            ],
            list(table.columns),
        )
        city = table[table['column'] == 'city'].set_index('category')
        self.assertEqual(2, city.loc['chicago', 'sample_count'])
        self.assertEqual(1, city.loc['chicago', 'encoded_value'])
        self.assertEqual(1, city.loc['boston', 'sample_count'])

    def test_export_records_include_sample_counts(self):
        """Export records carry the training count of each category."""
        enc = fitted()
        records = enc.mapping_manifest_.to_records()
        chicago = next(r for r in records if r['column'] == 'city' and r['category'] == 'chicago')
        self.assertEqual(2, chicago['sample_count'])
        self.assertEqual('1', chicago['encoded_value'])


class TestNormalEncoding(TestCase):
    """Verification passes on the mapping that produced the manifest."""

    def test_normal_transform_passes_and_matches_plain_output(self):
        """A transform with the fitted manifest passes and matches the plain encoder."""
        enc = fitted()
        out = enc.transform(X_INFER)
        plain = encoders.OrdinalEncoder(cols=['city', 'color', 'size']).fit(X_TRAIN)
        pd.testing.assert_frame_equal(plain.transform(X_INFER), out)

    def test_round_trip_through_file(self):
        """A saved manifest reloads with the same digest and no differences."""
        enc = fitted()
        path = '/tmp/mapping_manifest_roundtrip.jsonl'
        enc.mapping_manifest_.save(path, chunk_size=2)
        reloaded = MappingManifest.load(path)

        self.assertEqual(enc.mapping_manifest_.content_digest, reloaded.content_digest)
        self.assertEqual({}, reloaded.differences(enc.mapping_manifest_))
        reloaded.verify_against(enc.mapping_manifest_)


class TestMismatch(TestCase):
    """A mapping table that differs from the training-side one is rejected."""

    def test_content_change_with_identical_timestamp_is_mismatch(self):
        """A content change is a mismatch even when the timestamp is identical."""
        enc = fitted()
        expected = enc.mapping_manifest_

        # same timestamp, different content: the digest must still differ
        altered = MappingManifest.from_mapping(
            [
                {
                    'col': 'city',
                    'mapping': pd.Series(
                        {
                            'chicago': 9,
                            'st louis': 2,
                            'detroit': 3,
                            'boston': 4,
                            np.nan: -2,
                        }
                    ),
                    'data_type': 'object',
                }
            ],
            encoder='OrdinalEncoder',
            fitted_at=expected.fitted_at,
            n_samples=expected.n_samples,
        )
        self.assertNotEqual(expected.content_digest, altered.content_digest)
        with pytest.raises(MappingMismatchError) as excinfo:
            expected.verify_against(altered)
        self.assertFalse(excinfo.value.stale)
        self.assertEqual(
            {'chicago'},
            {
                item['category'][1]
                for item in excinfo.value.differences['columns']['city']['changed_codes']
            },
        )

    def test_transform_raises_on_replaced_mapping(self):
        """Transform refuses a mapping replaced after fit and names the differing categories."""
        enc = fitted()
        original = enc.mapping[0]['mapping'].copy(deep=True)
        try:
            enc.mapping[0]['mapping'] = original.rename({'chicago': 'chicago-renamed'})
            with pytest.raises(MappingMismatchError) as excinfo:
                enc.transform(X_INFER)
            detail = excinfo.value.differences['columns']['city']
            self.assertEqual(1, len(detail['missing_categories']))
            self.assertEqual(1, len(detail['extra_categories']))
        finally:
            enc.mapping[0]['mapping'] = original

    def test_stale_manifest_is_flagged_after_refit(self):
        """A newer training-side manifest marks the older mapping as stale."""
        # Training side refits later and exports a newer manifest; inference side
        # still runs the older mapping.
        old_encoder = fitted()
        old_manifest = old_encoder.mapping_manifest_
        refit = encoders.OrdinalEncoder(cols=['city', 'color', 'size'], mapping_integrity=True).fit(
            pd.DataFrame({'city': ['a', 'b', 'a'], 'color': ['x', 'y', 'x'], 'size': [1, 2, 1]})
        )
        new_manifest = refit.mapping_manifest_
        self.assertNotEqual(old_manifest.content_digest, new_manifest.content_digest)
        self.assertGreater(new_manifest.fitted_datetime, old_manifest.fitted_datetime)

        # The inference-side encoder still uses the old mapping but is given the
        # new (training-side) manifest: refused and flagged stale.
        old_encoder.mapping_manifest = new_manifest
        with pytest.raises(MappingMismatchError) as excinfo:
            old_encoder.transform(pd.DataFrame({'city': ['a'], 'color': ['x'], 'size': [1]}))
        self.assertTrue(excinfo.value.stale)
        self.assertIn('stale', str(excinfo.value))

    def test_old_manifest_not_silently_used_after_refit(self):
        """Refitting replaces the manifest instead of silently keeping the old one."""
        enc = fitted()
        stale = enc.mapping_manifest_
        enc.fit(pd.DataFrame({'city': ['x', 'y'], 'color': ['p', 'q'], 'size': [7, 8]}))
        # the encoder rebuilt its manifest; it must not be the stale object
        self.assertNotEqual(stale.content_digest, enc.mapping_manifest_.content_digest)
        self.assertIsNot(enc.mapping_manifest_, stale)
        enc.transform(pd.DataFrame({'city': ['x'], 'color': ['p'], 'size': [7]}))


class TestUnknownCategories(TestCase):
    """Unknown categories follow the serialized policy, read back from the file."""

    def test_unknown_policy_is_read_back_with_manifest(self):
        """The unknown-category policy is read back from the saved manifest."""
        enc = encoders.OrdinalEncoder(cols=['city'], mapping_integrity=True, handle_unknown=42).fit(
            X_TRAIN
        )
        path = '/tmp/mapping_manifest_unknown.jsonl'
        enc.mapping_manifest_.save(path)

        reloaded = MappingManifest.load(path)
        self.assertEqual(42, strategy_values(reloaded, 'handle_unknown'))

    def test_unknown_category_value_strategy(self):
        """The default value strategy encodes unseen categories as -1."""
        enc = fitted()
        out = enc.transform(pd.DataFrame({'city': ['never-seen'], 'color': ['red'], 'size': [1]}))
        self.assertEqual(-1, out['city'].iloc[0])

    def test_unknown_category_error_strategy(self):
        """The error strategy raises on unseen categories."""
        enc = encoders.OrdinalEncoder(
            cols=['city'], mapping_integrity=True, handle_unknown='error'
        ).fit(X_TRAIN)
        frame = X_INFER.assign(city='never-seen')
        with pytest.raises(ValueError, match='Unexpected categories'):
            enc.transform(frame)

    def test_unknown_category_return_nan_strategy(self):
        """The return_nan strategy encodes unseen categories as NaN."""
        enc = encoders.OrdinalEncoder(
            cols=['city'], mapping_integrity=True, handle_unknown='return_nan'
        ).fit(X_TRAIN)
        out = enc.transform(X_INFER.assign(city='never-seen'))
        self.assertTrue(pd.isna(out['city'].iloc[0]))

    def test_policy_change_is_reported_as_mismatch(self):
        """Changing the live policy after fit is reported as a mismatch."""
        enc = fitted()
        expected = enc.mapping_manifest_
        enc.handle_unknown = 'error'
        with pytest.raises(MappingMismatchError) as excinfo:
            enc.transform(X_INFER)
        policy = excinfo.value.differences['columns']['city']['changed_policies']
        self.assertEqual("'value'", policy['handle_unknown']['expected'])
        self.assertEqual("'error'", policy['handle_unknown']['actual'])
        self.assertEqual(expected.content_digest, excinfo.value.expected_digest)


class TestCorruption(TestCase):
    """A damaged manifest reports the offending segment and its impact."""

    def _saved(self):
        enc = fitted()
        path = '/tmp/mapping_manifest_corrupt.jsonl'
        enc.mapping_manifest_.save(path, chunk_size=2)
        return path

    def test_missing_manifest_raises_instead_of_defaulting(self):
        """A missing manifest is an error, never a silent default encoding."""
        enc = fitted()
        enc.mapping_manifest = '/tmp/definitely-not-here.jsonl'
        with pytest.raises(MappingManifestNotFoundError, match='Refusing to encode'):
            enc.transform(X_INFER)
        # the error must not be swallowed and fall back to a default encoding
        with pytest.raises(MappingManifestNotFoundError):
            MappingManifest.load('/tmp/definitely-not-here.jsonl')

    def test_corrupt_chunk_is_located_and_scoped(self):
        """A damaged chunk reports its line, column and category range."""
        path = self._saved()
        lines = open(path, encoding='utf-8').read().splitlines()
        chunk = json.loads(lines[1])
        chunk['rows'][0][1] = ['int', 999]  # change one encoded code
        lines[1] = json.dumps(chunk)
        open(path, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')

        with pytest.raises(MappingManifestCorruptionError) as excinfo:
            MappingManifest.load(path)
        message = str(excinfo.value)
        self.assertIn("column 'city'", message)
        self.assertIn('categories 0..1', message)
        self.assertIn('line 2', message)
        self.assertIn('rest of this column and the other columns are unaffected', message)

    def test_invalid_json_is_located(self):
        """Invalid JSON is reported with its line number."""
        path = self._saved()
        lines = open(path, encoding='utf-8').read().splitlines()
        lines[1] = lines[1][:20] + '!!not-json'
        open(path, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')

        with pytest.raises(MappingManifestCorruptionError, match='line 2'):
            MappingManifest.load(path)

    def test_missing_header_is_reported(self):
        """An empty manifest is reported as missing its header."""
        path = '/tmp/mapping_manifest_no_header.jsonl'
        open(path, 'w', encoding='utf-8').write('')
        with pytest.raises(MappingManifestCorruptionError, match='header record is missing'):
            MappingManifest.load(path)

    def test_transform_raises_on_corrupt_manifest(self):
        """Transform refuses to encode with a corrupt manifest."""
        path = self._saved()
        lines = open(path, encoding='utf-8').read().splitlines()
        lines[1] = lines[1][:10]
        open(path, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')

        enc = fitted()
        enc.mapping_manifest = path
        with pytest.raises(MappingManifestCorruptionError):
            enc.transform(X_INFER)


class TestCategoryDrift(TestCase):
    """Large training/inference category differences produce statistics and a refit hint."""

    def test_drift_statistics_and_warning(self):
        """Large category drift yields per-column statistics and a refit warning."""
        enc = fitted()
        drift = pd.DataFrame(
            {
                'city': ['x1', 'x2', 'x3'],
                'color': ['red', 'blue', 'red'],
                'size': [1, 2, 3],
            }
        )
        with pytest.warns(MappingDriftWarning, match='refit'):
            report = enc.mapping_manifest_.check_categories(drift)

        city = report['city']
        self.assertEqual(3, city['n_unknown_categories'])
        self.assertTrue(city['drift_too_large'])
        self.assertEqual(5, report['city']['n_absent_categories'])
        self.assertEqual(1.0, report['city']['unknown_fraction'])
        self.assertFalse(report['color']['drift_too_large'])

    def test_no_warning_on_identical_distribution(self):
        """An identical inference distribution emits no drift warning."""
        enc = fitted()
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            enc.transform(X_INFER)
        self.assertEqual([], [w for w in caught if isinstance(w.message, MappingDriftWarning)])


class TestAllMappingEncoders(TestCase):
    """The opt-in switch is available on every encoder that keeps a mapping."""

    def test_switch_supported_across_encoders(self):
        """Every encoder with a mapping exposes the opt-in switch."""
        y = pd.Series([1, 0, 1, 0, 1])
        X = X_TRAIN.copy()
        cases = [
            (encoders.OrdinalEncoder, False),
            (encoders.OneHotEncoder, False),
            (encoders.BinaryEncoder, False),
            (encoders.BaseNEncoder, False),
            (encoders.GrayEncoder, False),
            (encoders.HelmertEncoder, False),
            (encoders.SumEncoder, False),
            (encoders.CountEncoder, False),
            (encoders.MultiHotEncoder, False),
            (encoders.RankHotEncoder, False),
            (encoders.TargetEncoder, True),
            (encoders.WOEEncoder, True),
            (encoders.MEstimateEncoder, True),
            (encoders.JamesSteinEncoder, True),
            (encoders.CatBoostEncoder, True),
            (encoders.LeaveOneOutEncoder, True),
            (encoders.QuantileEncoder, True),
            (encoders.CountTargetEncoder, True),
        ]
        for cls, supervised in cases:
            with self.subTest(encoder=cls.__name__):
                enc = cls(mapping_integrity=True)
                enc.fit(X, y) if supervised else enc.fit(X)
                enc.transform(X, y) if supervised else enc.transform(X)
                self.assertIsNotNone(enc.mapping_manifest_)
                self.assertTrue(enc.mapping_manifest_.content_digest.startswith('sha256:'))

    def test_constructor_parameters_round_trip_through_clone(self):
        """The integrity parameters survive get_params / clone."""
        from sklearn.base import clone

        enc = encoders.OrdinalEncoder(
            mapping_integrity=True, mapping_drift_threshold=0.25, mapping_manifest='x.jsonl'
        )
        params = clone(enc).get_params()
        self.assertTrue(params['mapping_integrity'])
        self.assertEqual(0.25, params['mapping_drift_threshold'])
        self.assertEqual('x.jsonl', params['mapping_manifest'])


class TestDeterminismAndPersistence(TestCase):
    """The manifest is deterministic and survives pickling."""

    def test_same_input_same_digest_across_fits(self):
        """Two fits on identical data produce the same content digest."""
        first = fitted().mapping_manifest_
        second = fitted().mapping_manifest_
        self.assertEqual(first.content_digest, second.content_digest)

    def test_digest_independent_of_chunk_size(self):
        """The digest does not depend on how the file is chunked."""
        enc = fitted()
        enc.mapping_manifest_.save('/tmp/mmap_chunk2.jsonl', chunk_size=2)
        enc.mapping_manifest_.save('/tmp/mmap_chunk64.jsonl', chunk_size=64)
        left = MappingManifest.load('/tmp/mmap_chunk2.jsonl')
        right = MappingManifest.load('/tmp/mmap_chunk64.jsonl')
        self.assertEqual(left.content_digest, right.content_digest)
        self.assertEqual(enc.mapping_manifest_.content_digest, left.content_digest)

    def test_manifest_survives_pickle(self):
        """A fitted encoder (with its manifest) can be pickled and still verify."""
        import pickle

        enc = fitted()
        restored = pickle.loads(pickle.dumps(enc))
        restored.transform(X_INFER)
        self.assertEqual(
            enc.mapping_manifest_.content_digest, restored.mapping_manifest_.content_digest
        )

    def test_refit_does_not_reuse_old_manifest(self):
        """Refitting on the same data yields a fresh manifest object."""
        enc = fitted()
        first = enc.mapping_manifest_
        enc.fit(X_TRAIN)
        self.assertIsNot(first, enc.mapping_manifest_)
        self.assertEqual(first.content_digest, enc.mapping_manifest_.content_digest)
