"""Versioned, verifiable category-to-code mapping tables.

Every encoder keeps its fitted category mapping in a plain ``mapping``
attribute (a list or dict of pandas Series/DataFrame objects). That table has
no version, no fingerprint and no record of which ``handle_unknown`` policy it
was fitted with, so a stale mapping can silently disagree with the mapping
used at training time: the same category can end up encoded with a different
number while nothing raises.

This module adds an opt-in *mapping manifest* on top of the existing mapping
table:

* :class:`MappingManifest` records the encoder name, the fit timestamp, the
  number of training rows, the number of categories and a SHA-256 digest of
  the mapping *content* (category -> encoded value, never the file mtime).
* Every column additionally records its ``handle_unknown`` / ``handle_missing``
  policy, so the policy travels with the table instead of living only in code.
* :meth:`MappingManifest.save` / :meth:`MappingManifest.load` use a chunked
  JSON-lines format, so mappings with tens of thousands of categories are
  written and verified one chunk at a time.
* :meth:`MappingManifest.verify_against` compares a live mapping against a
  manifest and reports category/code/policy differences;
  :meth:`MappingManifest.check_categories` reports category drift between
  training and inference and suggests a refit.

The existing encoders' output is untouched: the manifest is built only when
``OrdinalEncoder(mapping_integrity=True)`` is used, and the module-level helpers
work with any encoder's ``mapping`` attribute.

Usage
-----
Build and save the manifest next to a fitted encoder::

    enc = OrdinalEncoder(mapping_integrity=True).fit(X)
    enc.mapping_manifest_.save('mapping.manifest.jsonl')
    enc.mapping_manifest_.export_csv('mapping.review.csv')

At inference time, point a freshly loaded encoder at the training-side
manifest; ``transform`` then refuses to encode with a mapping that does not
match it::

    enc = joblib.load('encoder.joblib')  # fitted with mapping_integrity=True
    enc.mapping_manifest = 'mapping.manifest.jsonl'  # training-side manifest
    enc.transform(X)  # raises MappingMismatchError on a stale/different mapping
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import math
import os
import tempfile
import warnings
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

MANIFEST_FORMAT = 'category_encoders.mapping_manifest'
MANIFEST_FORMAT_VERSION = 1
DEFAULT_CHUNK_SIZE = 1024
# Fraction of inference categories unknown to (or missing from) the fitted
# mapping above which category drift is reported as "too large".
DEFAULT_DRIFT_THRESHOLD = 0.5
# Reserved ordinal codes; duplicated here to keep this module importable
# without loading the whole encoder package.
UNKNOWN_SENTINEL = -1
MISSING_SENTINEL = -2


class MappingManifestError(ValueError):
    """Base error for a missing, damaged or mismatched mapping manifest."""


class MappingManifestNotFoundError(MappingManifestError):
    """The expected manifest file/object does not exist."""


class MappingManifestCorruptionError(MappingManifestError):
    """The manifest exists but cannot be parsed or fails an integrity check."""


class MappingMismatchError(MappingManifestError):
    """The mapping in use differs from the expected (training-side) mapping.

    Attributes
    ----------
    differences : dict
        Structured per-column differences: ``missing_categories`` (present in
        the expected mapping but not in the mapping in use), ``extra_categories``
        (present in use but not expected), ``changed_codes`` and
        ``changed_policies``.
    stale : bool
        True when the expected manifest was fitted after the mapping in use,
        i.e. the mapping is a leftover from a previous fit.
    expected_digest, actual_digest : str
        Content digests of the two tables.
    """

    def __init__(
        self,
        message: str,
        differences: dict | None = None,
        *,
        stale: bool = False,
        expected_digest: str | None = None,
        actual_digest: str | None = None,
    ):
        super().__init__(message)
        self.differences = differences or {}
        self.stale = stale
        self.expected_digest = expected_digest
        self.actual_digest = actual_digest


class MappingDriftWarning(UserWarning):
    """Training and inference category sets differ substantially."""


# ---------------------------------------------------------------------------
# Canonical encoding: turned into JSON-safe, type-preserving payloads so that
# the digest is computed over content (category value and type, encoded code)
# and is independent of insertion order, file modification time and chunking.
# ---------------------------------------------------------------------------

TAG_MISSING = 'missing'
TAG_BOOL = 'bool'
TAG_INT = 'int'
TAG_FLOAT = 'float'
TAG_COMPLEX = 'complex'
TAG_STR = 'str'
TAG_BYTES = 'bytes'
TAG_TIMESTAMP = 'timestamp'
TAG_TIMEDELTA = 'timedelta'
TAG_DATE = 'date'
TAG_DATETIME = 'datetime'
TAG_TUPLE = 'tuple'
TAG_LIST = 'list'
TAG_DICT = 'dict'
TAG_REPR = 'repr'


@dataclass(frozen=True)
class OpaqueValue:
    """Placeholder for a category type without a lossless JSON mapping."""

    module: str
    qualname: str
    text: str

    def __repr__(self) -> str:
        """Show the original repr of the opaque value."""
        return self.text


def _is_missing_scalar(value: Any) -> bool:
    """Return True only for scalar missing values (not arrays/lists/tuples)."""
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    if isinstance(value, (float, np.floating)) and math.isnan(float(value)):
        return True
    if isinstance(value, np.datetime64):
        return bool(np.isnat(value))
    if isinstance(value, (pd.Timestamp, datetime)):
        return bool(pd.isna(value))
    return False


def encode_value(value: Any) -> list:
    """Return a JSON-safe, type-tagged form of ``value``.

    The result is a two-element list ``[tag, payload]``. Encoding is
    deterministic and lossless for the types that appear as category keys or
    encoded codes; unsupported types fall back to a ``repr`` payload which
    still participates in the digest and in comparisons.
    """
    if isinstance(value, np.generic) and not isinstance(value, np.datetime64):
        value = value.item()
    if isinstance(value, np.datetime64):
        if np.isnat(value):
            return [TAG_MISSING, None]
        value = pd.Timestamp(value)
    if _is_missing_scalar(value):
        return [TAG_MISSING, None]
    if isinstance(value, (bool, np.bool_)):
        return [TAG_BOOL, bool(value)]
    if isinstance(value, (int, np.integer)):
        return [TAG_INT, int(value)]
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isnan(number):
            return [TAG_FLOAT, 'nan']
        if math.isinf(number):
            return [TAG_FLOAT, 'inf' if number > 0 else '-inf']
        return [TAG_FLOAT, number.hex()]
    if isinstance(value, (complex, np.complexfloating)):
        number = complex(value)
        return [TAG_COMPLEX, [float(number.real).hex(), float(number.imag).hex()]]
    if isinstance(value, str):
        return [TAG_STR, str(value)]
    if isinstance(value, (bytes, bytearray, np.bytes_)):
        return [TAG_BYTES, base64.b64encode(bytes(value)).decode('ascii')]
    if isinstance(value, pd.Timestamp):
        return [TAG_TIMESTAMP, value.isoformat()]
    if isinstance(value, datetime):
        return [TAG_DATETIME, value.isoformat()]
    if isinstance(value, pd.Timedelta):
        return [TAG_TIMEDELTA, int(value.value)]
    if isinstance(value, timedelta):
        return [TAG_TIMEDELTA, int(pd.Timedelta(value).value)]
    if isinstance(value, date):
        return [TAG_DATE, value.isoformat()]
    if isinstance(value, tuple):
        return [TAG_TUPLE, [encode_value(item) for item in value]]
    if isinstance(value, list):
        return [TAG_LIST, [encode_value(item) for item in value]]
    if isinstance(value, np.ndarray):
        return [TAG_LIST, [encode_value(item) for item in value.tolist()]]
    if isinstance(value, dict):
        items = [[encode_value(key), encode_value(item)] for key, item in value.items()]
        items.sort(key=lambda pair: _canonical_json(pair[0]))
        return [TAG_DICT, items]
    return [
        TAG_REPR,
        [type(value).__module__, type(value).__qualname__, repr(value)],
    ]


def decode_value(payload: Sequence) -> Any:
    """Inverse of :func:`encode_value`, best effort for opaque values."""
    tag, value = payload[0], payload[1]
    if tag == TAG_MISSING:
        return np.nan
    if tag == TAG_BOOL:
        return bool(value)
    if tag == TAG_INT:
        return int(value)
    if tag == TAG_FLOAT:
        if value == 'nan':
            return float('nan')
        if value == 'inf':
            return float('inf')
        if value == '-inf':
            return float('-inf')
        return float.fromhex(value)
    if tag == TAG_COMPLEX:
        return complex(float.fromhex(value[0]), float.fromhex(value[1]))
    if tag == TAG_STR:
        return str(value)
    if tag == TAG_BYTES:
        return base64.b64decode(value.encode('ascii'))
    if tag == TAG_TIMESTAMP:
        return pd.Timestamp(value)
    if tag == TAG_DATETIME:
        return datetime.fromisoformat(value)
    if tag == TAG_TIMEDELTA:
        return pd.Timedelta(int(value), unit='ns')
    if tag == TAG_DATE:
        return date.fromisoformat(value)
    if tag == TAG_TUPLE:
        return tuple(decode_value(item) for item in value)
    if tag == TAG_LIST:
        return [decode_value(item) for item in value]
    if tag == TAG_DICT:
        return {decode_value(key): decode_value(item) for key, item in value}
    if tag == TAG_REPR:
        return OpaqueValue(module=value[0], qualname=value[1], text=value[2])
    raise MappingManifestCorruptionError(f'Unknown canonical value tag {tag!r}')


def _canonical_json(payload: Any) -> str:
    """Stable JSON representation used for digests, sorting and equality."""
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=False,
        allow_nan=False,
    )


def _digest(payload: Any) -> str:
    blob = _canonical_json(payload).encode('utf-8')
    return 'sha256:' + hashlib.sha256(blob).hexdigest()


def match_key(payload: Sequence) -> str:
    """Equality key for a canonical payload, tolerant of int/float representation.

    ``1`` and ``1.0`` compare equal as mapping keys in pandas but carry
    different canonical tags. Category *matching* (drift checks, difference
    reporting) normalizes numeric tags so that a value observed as ``int`` at
    fit and as ``float`` at inference is not reported as a different category.
    The content digest deliberately stays type-exact; only category identity
    used for comparison is normalized.
    """
    tag, value = payload[0], payload[1]
    if tag == TAG_INT:
        return _canonical_json([TAG_FLOAT, float(value).hex()])
    if (
        tag == TAG_FLOAT
        and isinstance(value, str)
        and value
        not in (
            'nan',
            'inf',
            '-inf',
        )
    ):
        number = float.fromhex(value)
        if number.is_integer():
            return _canonical_json([TAG_FLOAT, value])
    return _canonical_json(payload)


# ---------------------------------------------------------------------------
# Strategy (handle_unknown / handle_missing) serialization
# ---------------------------------------------------------------------------


def encode_strategy(strategy: Any) -> dict:
    """Serialize a handle_unknown/handle_missing strategy into a JSON-safe dict."""
    if strategy is None or isinstance(strategy, str):
        return {'kind': 'named', 'value': strategy}
    if callable(strategy):
        return {
            'kind': 'callable',
            'module': type(strategy).__module__,
            'qualname': getattr(strategy, '__qualname__', type(strategy).__qualname__),
            'text': repr(strategy),
        }
    return {'kind': 'scalar', 'value': encode_value(strategy)}


def strategy_label(payload: dict) -> str:
    """Human readable one-line label of a serialized strategy."""
    kind = payload.get('kind')
    if kind == 'named':
        return repr(payload.get('value'))
    if kind == 'scalar':
        return repr(decode_value(payload['value']))
    if kind == 'callable':
        return f'callable {payload.get("module")}.{payload.get("qualname")}'
    return repr(payload)


def _strategy_for_column(policy: Any, col: str) -> Any:
    """Resolve a scalar strategy or a {column: strategy} mapping for one column."""
    if isinstance(policy, dict) and col in policy:
        return policy[col]
    return policy


# ---------------------------------------------------------------------------
# Per-column and whole-table manifests
# ---------------------------------------------------------------------------


def _category_sort_key(row: tuple[list, list, Any]) -> str:
    return _canonical_json(row[0])


def _format_value(payload: Sequence) -> str:
    """Render one canonical payload as a short human-readable string."""
    tag = payload[0]
    value = payload[1]
    if tag == TAG_MISSING:
        return ''
    if tag == TAG_STR:
        return value
    if tag in (TAG_TIMESTAMP, TAG_DATETIME, TAG_DATE):
        return value
    if tag == TAG_TIMEDELTA:
        return str(pd.Timedelta(int(value), unit='ns'))
    if tag == TAG_BOOL:
        return 'True' if value else 'False'
    if tag == TAG_INT:
        return str(value)
    if tag == TAG_FLOAT:
        if value == 'nan':
            return 'nan'
        if value in ('inf', '-inf'):
            return value
        return repr(float.fromhex(value))
    if tag == TAG_COMPLEX:
        return repr(complex(float.fromhex(value[0]), float.fromhex(value[1])))
    if tag == TAG_BYTES:
        return repr(base64.b64decode(value.encode('ascii')))
    if tag == TAG_TUPLE:
        return '(' + ', '.join(_format_value(item) for item in value) + ')'
    if tag == TAG_LIST:
        return '[' + ', '.join(_format_value(item) for item in value) + ']'
    return _canonical_json(payload)


def _short_value(payload: Sequence, limit: int = 80) -> str:
    text = f'{_format_value(payload)} ({payload[0]})'
    if len(text) > limit:
        text = text[: limit - 3] + '...'
    return text


@dataclass
class ColumnManifest:
    """The manifest of one encoded column.

    ``rows`` holds ``(category, encoded_value, sample_count)`` triples in
    canonical (type-tagged) form. ``sample_count`` is None when the caller did
    not supply training counts (for example when the manifest is rebuilt from a
    loaded encoder at inference time).
    """

    col: str
    handle_unknown: dict
    handle_missing: dict
    data_type: str | None
    rows: list[tuple[list, list, int | None]]

    @property
    def n_categories(self) -> int:
        """Number of categories in this column."""
        return len(self.rows)

    def digest(self) -> str:
        """Content digest of this column (category + code + policies + dtype)."""
        return _digest(self._digest_payload())

    def _digest_payload(self) -> dict:
        rows = sorted(
            ([row[0], row[1]] for row in self.rows), key=lambda pair: _canonical_json(pair[0])
        )
        return {
            'col': self.col,
            'handle_unknown': self.handle_unknown,
            'handle_missing': self.handle_missing,
            'data_type': self.data_type,
            'rows': rows,
        }


@dataclass
class MappingManifest:
    """A versioned, verifiable snapshot of a fitted category-to-code table.

    Parameters
    ----------
    encoder: str
        Name of the encoder class that produced the mapping.
    columns: list of ColumnManifest
        One entry per encoded column.
    fitted_at: str or None
        ISO-8601 UTC timestamp of the fit that produced the mapping. It is
        metadata: it is never part of ``content_digest``.
    n_samples: int or None
        Number of training rows the mapping was fitted on.
    format_version: int
        Version of the on-disk manifest format.
    """

    encoder: str
    columns: list[ColumnManifest]
    fitted_at: str | None = None
    n_samples: int | None = None
    format_version: int = MANIFEST_FORMAT_VERSION

    # -- construction ------------------------------------------------------

    @classmethod
    def from_mapping(
        cls,
        mapping: Iterable[dict],
        *,
        encoder: str,
        fitted_at: str | None = None,
        n_samples: int | None = None,
        handle_unknown: Any = 'value',
        handle_missing: Any = 'value',
        counts: dict[str, Any] | None = None,
        clock=None,
    ) -> MappingManifest:
        """Build a manifest from an OrdinalEncoder-style mapping list.

        Parameters
        ----------
        mapping: list of dict
            The fitted mapping, ``[{'col': ..., 'mapping': pd.Series(category ->
            code), 'data_type': ...}, ...]``.
        encoder: str
            Encoder name recorded in the manifest.
        fitted_at: str, optional
            Timestamp to record. Defaults to the current UTC time.
        n_samples: int, optional
            Number of rows the mapping was fitted on.
        handle_unknown, handle_missing: str, number, callable or dict
            The encoding policies to serialize with the table. A dict is
            resolved per column by name.
        counts: dict of column -> Series/dict, optional
            Per-category training counts, aligned by category value. Missing
            entries are recorded as None (unknown) rather than 0.
        clock: callable, optional
            Returns the timestamp to use; injectable for tests.
        """
        columns: list[ColumnManifest] = []
        counts = counts or {}
        for switch in mapping:
            col = switch.get('col')
            series = switch.get('mapping')
            col_counts = counts.get(col)
            count_by_key: dict[str, int] = {}
            if col_counts is not None:
                items = col_counts.items() if hasattr(col_counts, 'items') else col_counts
                for category, count in items:
                    count_by_key[match_key(encode_value(category))] = int(count)
            if isinstance(series, pd.DataFrame):
                rows = []
                for category, values in zip(series.index, series.to_numpy(), strict=True):
                    key = match_key(encode_value(category))
                    rows.append(
                        (
                            encode_value(category),
                            encode_value(
                                [
                                    value.item() if isinstance(value, np.generic) else value
                                    for value in values
                                ]
                            ),
                            count_by_key.get(key),
                        )
                    )
            else:
                if not isinstance(series, pd.Series):
                    raise TypeError(
                        'MappingManifest.from_mapping expects each mapping entry to be a '
                        f'pandas Series or DataFrame (category -> code); column {col!r} '
                        f'holds {type(series).__name__}.'
                    )
                rows = []
                for category, code in zip(series.index, series.to_numpy(), strict=True):
                    key = match_key(encode_value(category))
                    rows.append((encode_value(category), encode_value(code), count_by_key.get(key)))
            columns.append(
                ColumnManifest(
                    col=col,
                    handle_unknown=encode_strategy(_strategy_for_column(handle_unknown, col)),
                    handle_missing=encode_strategy(_strategy_for_column(handle_missing, col)),
                    data_type=(
                        str(switch['data_type']) if switch.get('data_type') is not None else None
                    ),
                    rows=rows,
                )
            )
        if fitted_at is None:
            now = clock() if clock is not None else datetime.now(timezone.utc)
            fitted_at = now.isoformat() if hasattr(now, 'isoformat') else str(now)
        return cls(
            encoder=encoder,
            columns=columns,
            fitted_at=fitted_at,
            n_samples=n_samples,
        )

    # -- summary -----------------------------------------------------------

    @property
    def n_columns(self) -> int:
        """Number of encoded columns."""
        return len(self.columns)

    @property
    def n_categories(self) -> int:
        """Total number of categories across all columns."""
        return sum(column.n_categories for column in self.columns)

    @property
    def content_digest(self) -> str:
        """SHA-256 digest of the mapping content.

        The digest covers column names, serialized policies, data types and the
        category -> encoded-value pairs, sorted by category. It does not cover
        ``fitted_at``, ``n_samples``, sample counts, the file mtime or the
        chunking, so two manifests compare equal exactly when their mapping
        content is equal.
        """
        return _digest(
            [
                column._digest_payload()
                for column in sorted(self.columns, key=lambda column: str(column.col))
            ]
        )

    def column(self, name: str) -> ColumnManifest:
        """Return one column manifest or raise KeyError."""
        for column in self.columns:
            if column.col == name:
                return column
        raise KeyError(name)

    @property
    def fitted_datetime(self) -> datetime | None:
        """``fitted_at`` parsed as an aware datetime, or None if unparseable."""
        if not self.fitted_at:
            return None
        try:
            parsed = datetime.fromisoformat(self.fitted_at)
        except (TypeError, ValueError):
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed

    def summary(self) -> str:
        """One-line human readable summary."""
        return (
            f'{self.encoder} mapping fitted at {self.fitted_at or "unknown time"} '
            f'on {self.n_samples if self.n_samples is not None else "unknown"} samples: '
            f'{self.n_columns} columns, {self.n_categories} categories, '
            f'digest {self.content_digest}'
        )

    # -- verification ------------------------------------------------------

    def verify_against(self, actual: MappingManifest) -> None:
        """Raise MappingMismatchError unless ``actual`` matches this manifest.

        ``self`` is the expected (training-side) manifest and ``actual`` is the
        mapping currently in use. The raised error carries a structured
        ``differences`` dict listing extra/missing categories, changed codes and
        changed policies, plus a ``stale`` flag derived from ``fitted_at``.
        """
        differences = self.differences(actual)
        if not differences:
            return
        stale = False
        expected_at, actual_at = self.fitted_datetime, actual.fitted_datetime
        if expected_at is not None and actual_at is not None and expected_at > actual_at:
            stale = True

        lines = [
            'Mapping table does not match the expected training-side manifest.',
            f'  expected: {self.summary()}',
            f'  in use:   {actual.summary()}',
        ]
        for col in differences['columns_only_expected']:
            lines.append(f'  - column {col!r} exists in the expected mapping but not in use')
        for col in differences['columns_only_actual']:
            lines.append(f'  - column {col!r} exists in use but not in the expected mapping')
        for col, detail in differences['columns'].items():
            if detail['missing_categories']:
                sample = ', '.join(_short_value(item) for item in detail['missing_categories'][:5])
                extra = '' if len(detail['missing_categories']) <= 5 else ' ...'
                lines.append(
                    f'  - column {col!r}: {len(detail["missing_categories"])} categor'
                    f'{"y" if len(detail["missing_categories"]) == 1 else "ies"} missing '
                    f'(in expected, absent in use): {sample}{extra}'
                )
            if detail['extra_categories']:
                sample = ', '.join(_short_value(item) for item in detail['extra_categories'][:5])
                extra = '' if len(detail['extra_categories']) <= 5 else ' ...'
                lines.append(
                    f'  - column {col!r}: {len(detail["extra_categories"])} unexpected '
                    f'categor{"y" if len(detail["extra_categories"]) == 1 else "ies"} '
                    f'(in use, absent in expected): {sample}{extra}'
                )
            if detail['changed_codes']:
                shown = detail['changed_codes'][:5]
                sample = ', '.join(
                    f'{_short_value(item["category"])}: '
                    f'{_format_value(item["expected"])} -> {_format_value(item["actual"])}'
                    for item in shown
                )
                extra = '' if len(detail['changed_codes']) <= 5 else ' ...'
                lines.append(
                    f'  - column {col!r}: {len(detail["changed_codes"])} categor'
                    f'{"y" if len(detail["changed_codes"]) == 1 else "ies"} encoded '
                    f'differently: {sample}{extra}'
                )
            for policy, values in detail['changed_policies'].items():
                lines.append(
                    f'  - column {col!r}: {policy} differs: '
                    f'{values["expected"]} -> {values["actual"]}'
                )
        if stale:
            lines.append(
                '  This mapping is stale: it was fitted at '
                f'{actual.fitted_at} but the expected manifest was fitted later, at '
                f'{self.fitted_at}. Refit the encoder on the current training data and '
                'export a new manifest.'
            )
        else:
            lines.append(
                '  Refit the encoder on the training data (or restore the expected '
                'mapping) and export a new manifest.'
            )
        raise MappingMismatchError(
            '\n'.join(lines),
            differences,
            stale=stale,
            expected_digest=self.content_digest,
            actual_digest=actual.content_digest,
        )

    def differences(self, actual: MappingManifest) -> dict:
        """Return an empty dict when equal, else the structured differences."""
        expected_by_col = {column.col: column for column in self.columns}
        actual_by_col = {column.col: column for column in actual.columns}
        only_expected = sorted(set(expected_by_col) - set(actual_by_col), key=str)
        only_actual = sorted(set(actual_by_col) - set(expected_by_col), key=str)
        columns: dict[str, dict] = {}

        for col in sorted(set(expected_by_col) & set(actual_by_col), key=str):
            expected_col = expected_by_col[col]
            actual_col = actual_by_col[col]
            expected_rows = {
                match_key(category): (category, code) for category, code, _ in expected_col.rows
            }
            actual_rows = {
                match_key(category): (category, code) for category, code, _ in actual_col.rows
            }
            missing = [expected_rows[key][0] for key in expected_rows if key not in actual_rows]
            extra = [actual_rows[key][0] for key in actual_rows if key not in expected_rows]
            changed = []
            for key in expected_rows:
                if key in actual_rows and match_key(expected_rows[key][1]) != (
                    match_key(actual_rows[key][1])
                ):
                    changed.append(
                        {
                            'category': expected_rows[key][0],
                            'expected': expected_rows[key][1],
                            'actual': actual_rows[key][1],
                        }
                    )
            changed_policies = {}
            for policy in ('handle_unknown', 'handle_missing', 'data_type'):
                expected_value = getattr(expected_col, policy)
                actual_value = getattr(actual_col, policy)
                if expected_value != actual_value:
                    if policy == 'data_type':
                        changed_policies[policy] = {
                            'expected': expected_value,
                            'actual': actual_value,
                        }
                    else:
                        changed_policies[policy] = {
                            'expected': strategy_label(expected_value),
                            'actual': strategy_label(actual_value),
                        }
            if missing or extra or changed or changed_policies:
                columns[col] = {
                    'missing_categories': missing,
                    'extra_categories': extra,
                    'changed_codes': changed,
                    'changed_policies': changed_policies,
                }

        if not (only_expected or only_actual or columns):
            return {}
        return {
            'columns_only_expected': only_expected,
            'columns_only_actual': only_actual,
            'columns': columns,
        }

    # -- category drift ----------------------------------------------------

    def check_categories(
        self,
        X: pd.DataFrame | dict[str, Iterable],
        *,
        threshold: float = DEFAULT_DRIFT_THRESHOLD,
        warn: bool = True,
    ) -> dict:
        """Compare the inference category set with the fitted category set.

        ``X`` is either the inference frame or a mapping of column name to the
        observed values. Returns per-column statistics: the number of fitted
        categories, the number of inference categories that are unknown to the
        mapping, and the number of fitted categories absent from the inference
        data, both as counts and fractions. A :class:`MappingDriftWarning`
        suggesting a refit is emitted when either fraction exceeds ``threshold``.

        Missing values are excluded from these statistics (``handle_missing``
        governs them) but counted separately.
        """
        is_frame = isinstance(X, pd.DataFrame)
        report: dict[str, dict] = {}
        drifting: list[str] = []
        for column in self.columns:
            if is_frame:
                if column.col not in X.columns:
                    continue
                values = X[column.col]
            else:
                if column.col not in X:
                    continue
                values = X[column.col]
                if not isinstance(values, pd.Series):
                    values = pd.Series(list(values))
            fitted = {match_key(category) for category, _, _ in column.rows}
            seen = set()
            missing_values = 0
            for value in pd.unique(values):
                if _is_missing_scalar(value):
                    missing_values += 1
                    continue
                seen.add(match_key(encode_value(value)))
            unknown = seen - fitted
            absent = fitted - seen
            n_fitted = len(fitted)
            n_seen = len(seen)
            unknown_fraction = len(unknown) / n_seen if n_seen else 0.0
            absent_fraction = len(absent) / n_fitted if n_fitted else 0.0
            too_large = unknown_fraction > threshold or absent_fraction > threshold
            report[column.col] = {
                'n_fitted_categories': n_fitted,
                'n_inference_categories': n_seen,
                'n_unknown_categories': len(unknown),
                'n_absent_categories': len(absent),
                'unknown_fraction': unknown_fraction,
                'absent_fraction': absent_fraction,
                'n_missing_values': missing_values,
                'drift_too_large': too_large,
            }
            if too_large:
                drifting.append(column.col)

        if warn and drifting:
            details = []
            for col in drifting:
                stats = report[col]
                details.append(
                    f'{col!r}: {stats["n_unknown_categories"]}/'
                    f'{stats["n_inference_categories"]} inference categories unseen '
                    f'({stats["unknown_fraction"]:.0%}), '
                    f'{stats["n_absent_categories"]}/{stats["n_fitted_categories"]} fitted '
                    f'categories absent ({stats["absent_fraction"]:.0%})'
                )
            warnings.warn(
                'Training and inference category sets differ substantially beyond the '
                f'{threshold:.0%} threshold. ' + '; '.join(details) + '. '
                'The encoding is still applied with the fitted mapping; refit the '
                'encoder on data that represents the inference distribution and export '
                'a new manifest.',
                MappingDriftWarning,
                stacklevel=3,
            )
        return report

    # -- export ------------------------------------------------------------

    def to_records(self) -> list[dict]:
        """Flatten the manifest into aligned (category, code, count) records."""
        records = []
        for column in self.columns:
            for category, code, count in column.rows:
                records.append(
                    {
                        'column': column.col,
                        'category': _format_value(category),
                        'category_type': category[0],
                        'encoded_value': _format_value(code),
                        'encoded_type': code[0],
                        'sample_count': count,
                    }
                )
        return records

    def export_csv(self, path: str | os.PathLike) -> Path:
        """Write the three aligned review columns: category, code, sample count."""
        path = Path(path)
        buffer = io.StringIO()
        writer = csv.DictWriter(
            buffer,
            fieldnames=[
                'column',
                'category',
                'category_type',
                'encoded_value',
                'encoded_type',
                'sample_count',
            ],
            lineterminator='\n',
        )
        writer.writeheader()
        for record in self.to_records():
            writer.writerow(record)
        path.write_text(buffer.getvalue(), encoding='utf-8')
        return path

    # -- chunked persistence ----------------------------------------------

    def save(self, path: str | os.PathLike, *, chunk_size: int = DEFAULT_CHUNK_SIZE) -> Path:
        """Write the manifest as JSON Lines, ``chunk_size`` categories per line.

        Returns the path written. The digest is independent of ``chunk_size``.
        """
        if chunk_size < 1:
            raise ValueError('chunk_size must be a positive integer')
        path = Path(path)
        header = {
            'type': 'header',
            'format': MANIFEST_FORMAT,
            'version': self.format_version,
            'encoder': self.encoder,
            'fitted_at': self.fitted_at,
            'n_samples': self.n_samples,
            'n_columns': self.n_columns,
            'n_categories': self.n_categories,
            'content_digest': self.content_digest,
            'columns': [
                {
                    'col': column.col,
                    'handle_unknown': column.handle_unknown,
                    'handle_missing': column.handle_missing,
                    'data_type': column.data_type,
                    'n_categories': column.n_categories,
                    'n_chunks': math.ceil(column.n_categories / chunk_size),
                    'digest': column.digest(),
                }
                for column in self.columns
            ],
        }
        target_dir = path.parent if str(path.parent) else Path('.')
        target_dir.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix='.mapping-manifest-', dir=target_dir)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as handle:
                handle.write(json.dumps(header, ensure_ascii=False, allow_nan=False))
                handle.write('\n')
                for column in self.columns:
                    for index, offset in enumerate(range(0, column.n_categories, chunk_size)):
                        chunk_rows = [
                            [category, code, count]
                            for category, code, count in column.rows[offset : offset + chunk_size]
                        ]
                        chunk = {
                            'type': 'chunk',
                            'col': column.col,
                            'index': index,
                            'offset': offset,
                            'rows': chunk_rows,
                        }
                        chunk['digest'] = _digest(
                            {
                                'col': column.col,
                                'offset': offset,
                                'rows': chunk_rows,
                            }
                        )
                        handle.write(json.dumps(chunk, ensure_ascii=False, allow_nan=False))
                        handle.write('\n')
            os.replace(temp_name, path)
        except BaseException:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
            raise
        return path

    @classmethod
    def load(cls, path: str | os.PathLike) -> MappingManifest:
        """Read and fully verify a manifest written by :meth:`save`.

        Raises MappingManifestNotFoundError when the file does not exist and
        MappingManifestCorruptionError with the offending line, column and
        category range when any header, chunk or content check fails.
        """
        path = Path(path)
        if not path.exists():
            raise MappingManifestNotFoundError(
                f'Mapping manifest not found: {path}. Refusing to encode without the '
                'training-side mapping: export a manifest with '
                'encoder.mapping_manifest_.save(path) and pass its path as mapping_manifest.'
            )
        header = None
        chunk_records: dict[str, dict[int, dict]] = {}
        with path.open('r', encoding='utf-8') as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as error:
                    raise MappingManifestCorruptionError(
                        f'Mapping manifest {path} is corrupted at line {line_number}: '
                        f'invalid JSON ({error.msg}). The affected record cannot be read, '
                        'so no category range from it can be trusted.'
                    ) from error
                if not isinstance(payload, dict):
                    raise MappingManifestCorruptionError(
                        f'Mapping manifest {path} is corrupted at line {line_number}: '
                        'expected a JSON object.'
                    )
                record_type = payload.get('type')
                if header is None:
                    if record_type != 'header':
                        raise MappingManifestCorruptionError(
                            f'Mapping manifest {path} is corrupted at line {line_number}: '
                            "the first record must be the 'header'."
                        )
                    header = cls._validate_header(payload, path, line_number)
                    continue
                if record_type != 'chunk':
                    raise MappingManifestCorruptionError(
                        f'Mapping manifest {path} is corrupted at line {line_number}: '
                        f'unexpected record type {record_type!r} after the header.'
                    )
                cls._validate_chunk(payload, header, path, line_number, chunk_records)

        if header is None:
            raise MappingManifestCorruptionError(
                f'Mapping manifest {path} is empty: the header record is missing.'
            )
        columns = [cls._assemble_column(meta, chunk_records, path) for meta in header['columns']]
        manifest = cls(
            encoder=header['encoder'],
            columns=columns,
            fitted_at=header.get('fitted_at'),
            n_samples=header.get('n_samples'),
            format_version=header['version'],
        )
        actual_digest = manifest.content_digest
        if actual_digest != header['content_digest']:
            raise MappingManifestCorruptionError(
                f'Mapping manifest {path} is corrupted: the assembled content digest '
                f'{actual_digest} does not match the header digest '
                f'{header["content_digest"]}. The per-chunk digests passed, so rows are '
                'intact individually but the table as a whole (columns, row counts or '
                'policies) was altered. Restore the manifest or refit the encoder.'
            )
        return manifest

    @staticmethod
    def _validate_header(payload: dict, path: Path, line_number: int) -> dict:
        if payload.get('format') != MANIFEST_FORMAT:
            raise MappingManifestCorruptionError(
                f'Mapping manifest {path} is corrupted at line {line_number}: not a '
                f'{MANIFEST_FORMAT} file (format={payload.get("format")!r}).'
            )
        if payload.get('version') != MANIFEST_FORMAT_VERSION:
            raise MappingManifestCorruptionError(
                f'Mapping manifest {path} has unsupported format version '
                f'{payload.get("version")!r}; this library reads version '
                f'{MANIFEST_FORMAT_VERSION}.'
            )
        for key in ('encoder', 'content_digest', 'columns'):
            if key not in payload:
                raise MappingManifestCorruptionError(
                    f'Mapping manifest {path} is corrupted at line {line_number}: '
                    f'the header is missing {key!r}, so the whole table is unusable.'
                )
        seen = set()
        for meta in payload['columns']:
            for key in ('col', 'handle_unknown', 'handle_missing', 'n_categories', 'n_chunks'):
                if key not in meta:
                    raise MappingManifestCorruptionError(
                        f'Mapping manifest {path} is corrupted at line {line_number}: '
                        f'column metadata for {meta.get("col")!r} is missing {key!r}.'
                    )
            if meta['col'] in seen:
                raise MappingManifestCorruptionError(
                    f'Mapping manifest {path} is corrupted at line {line_number}: '
                    f'column {meta["col"]!r} is declared twice in the header.'
                )
            seen.add(meta['col'])
        return payload

    @classmethod
    def _validate_chunk(
        cls,
        payload: dict,
        header: dict,
        path: Path,
        line_number: int,
        chunk_records: dict[str, dict[int, dict]],
    ) -> None:
        columns = {meta['col']: meta for meta in header['columns']}
        col = payload.get('col')
        if col not in columns:
            raise MappingManifestCorruptionError(
                f'Mapping manifest {path} is corrupted at line {line_number}: chunk for '
                f'unknown column {col!r}.'
            )
        index = payload.get('index')
        offset = payload.get('offset')
        rows = payload.get('rows')
        if not isinstance(index, int) or not isinstance(offset, int) or not isinstance(rows, list):
            raise MappingManifestCorruptionError(
                f'Mapping manifest {path} is corrupted at line {line_number} '
                f"(column {col!r}): chunk metadata is incomplete; this chunk's category "
                f'range starting at offset {offset!r} is unreadable.'
            )
        expected_digest = _digest({'col': col, 'offset': offset, 'rows': rows})
        if expected_digest != payload.get('digest'):
            end = offset + len(rows) - 1 if rows else offset
            raise MappingManifestCorruptionError(
                f'Mapping manifest {path} is corrupted at line {line_number} '
                f'(column {col!r}, categories {offset}..{end}): chunk digest mismatch. '
                'The category/code pairs in this range changed after the manifest was '
                'written and cannot be trusted; the rest of this column and the other '
                'columns are unaffected.'
            )
        chunk_records.setdefault(col, {})
        if index in chunk_records[col]:
            raise MappingManifestCorruptionError(
                f'Mapping manifest {path} is corrupted at line {line_number} '
                f'(column {col!r}, categories {offset}..{offset + len(rows) - 1}): '
                f'chunk index {index} is duplicated.'
            )
        chunk_records[col][index] = payload

    @staticmethod
    def _assemble_column(
        meta: dict, chunk_records: dict[str, dict[int, dict]], path: Path
    ) -> ColumnManifest:
        col = meta['col']
        chunks = chunk_records.get(col, {})
        expected_chunks = meta['n_chunks']
        missing = [index for index in range(expected_chunks) if index not in chunks]
        if missing:
            first = missing[0] * DEFAULT_CHUNK_SIZE
            raise MappingManifestCorruptionError(
                f'Mapping manifest {path} is corrupted: column {col!r} is missing '
                f'chunk(s) {missing}; category ranges from offset {first} onward are '
                'incomplete, so the column cannot be used for verification.'
            )
        rows = []
        for index in range(expected_chunks):
            for row in chunks[index]['rows']:
                count = int(row[2]) if len(row) > 2 and row[2] is not None else None
                rows.append((row[0], row[1], count))
        if len(rows) != meta['n_categories']:
            raise MappingManifestCorruptionError(
                f'Mapping manifest {path} is corrupted: column {col!r} declares '
                f'{meta["n_categories"]} categories but the chunks hold {len(rows)}; the '
                'tail of the column is missing or duplicated.'
            )
        column = ColumnManifest(
            col=col,
            handle_unknown=meta['handle_unknown'],
            handle_missing=meta['handle_missing'],
            data_type=meta.get('data_type'),
            rows=rows,
        )
        if column.digest() != meta.get('digest'):
            raise MappingManifestCorruptionError(
                f'Mapping manifest {path} is corrupted: column {col!r} chunk digests passed '
                'but the column content digest does not match the header. The rows are '
                'individually intact yet the column was reassembled incorrectly (wrong '
                'order or row count), so none of its codes can be trusted.'
            )
        return column


def strategy_values(manifest: MappingManifest, attribute: str, live: Any = None) -> Any:
    """Reconstruct one strategy from a serialized manifest.

    ``attribute`` is 'handle_unknown' or 'handle_missing'. Named and numeric
    strategies are restored exactly; a callable cannot be reconstructed from
    its description, so it is only accepted when the live strategy is a
    callable with the same module/qualname, and is returned unchanged.

    A manifest whose columns disagree on the strategy is rejected for ordinal
    encoders, which take a single strategy for all columns.
    """
    payloads = [getattr(column, attribute) for column in manifest.columns]
    if not payloads:
        return live
    labels = {strategy_label(payload) for payload in payloads}
    if len(labels) > 1:
        raise MappingManifestError(
            f'The manifest stores different {attribute} strategies per column '
            f'({sorted(labels)}); this encoder takes a single {attribute}. Apply the '
            'manifest to a per-column encoder, or refit with one uniform strategy.'
        )
    payload = payloads[0]
    kind = payload['kind']
    if kind == 'named':
        return payload['value']
    if kind == 'scalar':
        return decode_value(payload['value'])
    live_is_callable = callable(live)
    if live_is_callable and strategy_label(encode_strategy(live)) == strategy_label(payload):
        return live
    if live_is_callable:
        return live
    raise MappingManifestError(
        f'The manifest stores {attribute} as {strategy_label(payload)}, which is not a '
        'built-in or numeric strategy and cannot be reconstructed from the manifest. '
        'Pass the same callable to the encoder that was used at fit time; its identity '
        'is then verified against the manifest.'
    )


def describe_differences(differences: dict) -> str:
    """Short one-line description of structured differences (for warnings/logs)."""
    if not differences:
        return 'no differences'
    parts = []
    if differences.get('columns_only_expected'):
        parts.append(f'columns only in expected: {differences["columns_only_expected"]}')
    if differences.get('columns_only_actual'):
        parts.append(f'columns only in use: {differences["columns_only_actual"]}')
    for col, detail in differences.get('columns', {}).items():
        if detail['missing_categories']:
            parts.append(f'{col}: {len(detail["missing_categories"])} missing categories')
        if detail['extra_categories']:
            parts.append(f'{col}: {len(detail["extra_categories"])} extra categories')
        if detail['changed_codes']:
            parts.append(f'{col}: {len(detail["changed_codes"])} changed codes')
        for policy, values in detail['changed_policies'].items():
            parts.append(f'{col}: {policy} {values["expected"]} -> {values["actual"]}')
    return '; '.join(parts)


# ---------------------------------------------------------------------------
# Encoder integration helpers
# ---------------------------------------------------------------------------


def check_manifest(
    encoder,
    live_mapping: Sequence[dict],
    *,
    categories: dict[str, Iterable] | None = None,
) -> MappingManifest:
    """Verify a live encoder's mapping against its configured manifest.

    ``encoder.mapping_manifest`` may be a :class:`MappingManifest`, a path to a
    saved manifest, or an encoder carrying ``mapping_manifest_``. The expected
    manifest is loaded once and cached on the encoder. Category drift is checked
    against ``categories`` (column -> observed inference values) before the
    strict content comparison.

    Returns the manifest that was verified against. Raises
    :class:`MappingManifestError` subclasses on missing/corrupt/mismatched
    manifests.
    """
    source = encoder.mapping_manifest
    if source is None:
        # No explicit expected manifest: verify against the encoder's own fitted
        # manifest, i.e. the table this encoder produced at fit time.
        expected = getattr(encoder, 'mapping_manifest_', None)
        if expected is None:
            raise MappingManifestNotFoundError(
                'Mapping verification is enabled but no expected manifest is available: '
                'the encoder has no fitted manifest and no mapping_manifest was configured. '
                'Fit with mapping_integrity=True, or set mapping_manifest to a '
                'MappingManifest or to the path of a manifest saved by '
                'mapping_manifest_.save().'
            )
    elif isinstance(source, MappingManifest):
        expected = source
    elif hasattr(source, 'mapping_manifest_'):
        nested = source.mapping_manifest_
        if nested is None:
            raise MappingManifestNotFoundError(
                'Mapping verification is enabled but the reference encoder has no '
                'fitted manifest; fit it with mapping_integrity=True first.'
            )
        expected = nested
    else:
        # A path is re-read on every verification so a manifest replaced on disk
        # cannot keep an old verdict alive.
        expected = MappingManifest.load(source)

    if categories:
        expected.check_categories(
            {col: values for col, values in categories.items() if col in _columns_of(expected)},
            threshold=encoder.mapping_drift_threshold,
            warn=True,
        )

    # Timestamp the mapping in use with the encoder's own fitted manifest when
    # it has one, so verify_against can tell a stale mapping (fitted before the
    # expected manifest) from a merely different one.
    fitted = getattr(encoder, 'mapping_manifest_', None)
    actual = build_manifest(
        live_mapping,
        encoder=expected.encoder,
        n_samples=expected.n_samples,
        handle_unknown=encoder.handle_unknown,
        handle_missing=encoder.handle_missing,
        fitted_at=getattr(fitted, 'fitted_at', None),
    )
    expected.verify_against(actual)
    return expected


def _columns_of(manifest: MappingManifest) -> set:
    return {column.col for column in manifest.columns}


def build_manifest(
    mapping: Sequence[dict],
    *,
    encoder: str,
    n_samples: int | None = None,
    handle_unknown: Any = 'value',
    handle_missing: Any = 'value',
    counts: dict[str, Iterable] | None = None,
    fitted_at: str | None = None,
) -> MappingManifest:
    """Thin wrapper over :meth:`MappingManifest.from_mapping`."""
    return MappingManifest.from_mapping(
        mapping,
        encoder=encoder,
        fitted_at=fitted_at,
        n_samples=n_samples,
        handle_unknown=handle_unknown,
        handle_missing=handle_missing,
        counts=counts,
    )


def mapping_signature(mapping: Sequence[dict]) -> str:
    """Cheap content signature of a mapping list (identities, not codes).

    Used to notice that a fitted encoder's ``mapping`` was replaced after its
    manifest was built. Column names plus the ordered category tuples are
    enough: any refit that keeps the exact same categories in the same order
    produces the same codes, so the manifest stays valid.
    """
    parts = []
    for switch in mapping:
        series = switch.get('mapping')
        if isinstance(series, pd.Series):
            parts.append(
                [
                    str(switch.get('col')),
                    [_canonical_json(encode_value(key)) for key in series.index],
                    [_canonical_json(encode_value(value)) for value in series.to_numpy()],
                ]
            )
        else:
            parts.append([str(switch.get('col')), repr(series)])
    return _digest(parts)


def mapping_rows(mapping: Any) -> list[dict]:
    """Normalize an encoder ``mapping`` into ``[{'col', 'mapping', 'data_type'}]``.

    Supported shapes are the ones the library's encoders use:

    * a list of dicts with ``col`` and a pandas Series/DataFrame ``mapping``;
    * a dict ``column -> Series/DataFrame``.

    Any other shape raises :class:`MappingManifestError` so an enabled integrity
    check never silently verifies nothing.
    """
    if isinstance(mapping, list):
        normalized = []
        for switch in mapping:
            if isinstance(switch, dict) and 'col' in switch and 'mapping' in switch:
                normalized.append(
                    {
                        'col': switch['col'],
                        'mapping': switch['mapping'],
                        'data_type': switch.get('data_type'),
                    }
                )
            elif isinstance(switch, dict) and 'col' in switch and 'item_pos' in switch:
                # MultiHotEncoder: item -> column position, one slot per item.
                normalized.append(
                    {
                        'col': switch['col'],
                        'mapping': pd.Series(dict(switch['item_pos'])),
                        'data_type': None,
                    }
                )
            else:
                raise MappingManifestError(
                    'Mapping integrity is enabled but the mapping entry is not a dict with '
                    "'col' and a recognized mapping payload ('mapping' or 'item_pos'); "
                    'cannot build a manifest.'
                )
        return normalized
    if isinstance(mapping, dict):
        return [
            {'col': col, 'mapping': values, 'data_type': None} for col, values in mapping.items()
        ]
    raise MappingManifestError(
        'Mapping integrity is enabled but the fitted mapping has an unsupported shape '
        f'({type(mapping).__name__}); expected a list of dicts or a dict.'
    )


def indexed_by_categories(inner_rows: Sequence[dict], outer_rows: Sequence[dict]) -> bool:
    """True when the outer table is keyed by categories rather than by codes.

    ``inner_rows`` is the inner ordinal category -> code table and
    ``outer_rows`` the encoder's own table. The outer index is compared with
    the inner *categories* for every shared column; a table whose keys are
    ordinal codes matches the inner *codes* instead.
    """
    outer_by_col = {row['col']: row['mapping'] for row in outer_rows}
    checked = 0
    for row in inner_rows:
        col = row['col']
        if col not in outer_by_col:
            continue
        inner = row['mapping']
        outer = outer_by_col[col]
        if not hasattr(inner, 'index') or not hasattr(outer, 'index'):
            continue
        inner_categories = {match_key(encode_value(key)) for key in inner.index}
        inner_codes = {match_key(encode_value(value)) for value in inner.to_numpy()}
        outer_keys = {match_key(encode_value(key)) for key in outer.index}
        checked += 1
        if outer_keys == inner_categories:
            return True
        if outer_keys == inner_codes:
            return False
        return bool(outer_keys & inner_categories) and not (
            outer_keys & inner_codes - inner_categories
        )
    return checked == 0
