"""Per-column category coverage reporting (opt-in).

This module implements the data structures behind the ``coverage_report``
parameter of the encoders. It answers the question: "how much of this
column's values did the encoder actually see at fit time?"

Definitions (identical for every dataset, so reports are comparable
across train / validation / inference batches):

- ``n_categories``: the number of distinct values of the column across
  **both** the fit data and the evaluated batch (fit ∪ batch). This is
  the single denominator of the coverage metric; it is deliberately
  *not* restricted to the values seen at fit time.
- ``coverage``: ``n_fit_categories / n_categories`` — the share of all
  distinct values the encoder knows.
- ``unseen_ratio``: ``unseen_rows / n_rows`` — the share of *rows* of the
  evaluated batch whose value was never seen at fit time.

All missing-value spellings (None, np.nan, pd.NA, ...) count as one
single ``<missing>`` category.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Hashable, Iterable

import pandas as pd

__author__ = 'KeiMo7'

#: how many unseen values are rendered in warning messages / the summary table
MAX_RENDERED_UNSEEN = 5


class CoverageWarning(UserWarning):
    """Issued when a transformed batch contains categories never seen at fit time."""


class _Missing:
    """Singleton token standing for the missing-value category in category sets.

    A real object (not a string) so it can never collide with actual data.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return '<missing>'

    def __str__(self):
        return '<missing>'


MISSING = _Missing()


def column_categories(series: pd.Series) -> set:
    """Distinct values of one column as a set; missing values become one MISSING token.

    Raises TypeError for unhashable values (e.g. lists/dicts in an object column);
    callers treat that as a per-column report error, not an encoding failure.
    """
    is_missing = series.isna()
    categories = set(series[~is_missing].unique())
    if is_missing.any():
        categories.add(MISSING)
    return categories


@dataclass
class ColumnCoverage:
    """Coverage statistics of one column for one evaluated batch."""

    column: Hashable
    n_rows: int = 0
    n_fit_categories: int = 0
    n_observed_categories: int = 0
    # distinct values across fit ∪ batch; the single coverage denominator
    n_categories: int = 0
    # n_fit_categories / n_categories
    coverage: float = 1.0
    n_unseen_rows: int = 0
    # n_unseen_rows / n_rows
    unseen_ratio: float = 0.0
    unseen_values: list = field(default_factory=list)
    # True when no observed value of the batch was seen at fit time
    all_new: bool = False
    # min_group_size lumping statistics (0 / None when no lumping happened)
    n_lumped: int = 0
    lumped_row_fraction: float | None = None
    lumped_name: str | None = None
    # set when the statistics of this column could not be computed
    error: str | None = None

    def status(self, threshold: float) -> str:
        """Alert state of the column: 'error', 'all_new', 'low_coverage' or 'ok'."""
        if self.error is not None:
            return 'error'
        if self.all_new:
            return 'all_new'
        if self.unseen_ratio > threshold:
            return 'low_coverage'
        return 'ok'


def compute_column_coverage(
    column: Hashable,
    series: pd.Series,
    fit_categories: set,
    lumping: dict | None = None,
) -> ColumnCoverage:
    """Compute the coverage statistics of one column for one batch.

    Parameters
    ----------
    column: name of the column.
    series: the column of the evaluated batch (after min-group lumping was applied,
        so both sides of the comparison use the same label space).
    fit_categories: set recorded by ``column_categories`` at fit time.
    lumping: optional lumping statistics dict of the column
        (keys 'n_lumped', 'lumped_row_fraction', 'lumped_name').
    """
    lumping = lumping or {}
    n_rows = len(series)
    is_missing = series.isna()
    observed = column_categories(series)
    new = observed - fit_categories
    union = observed | fit_categories

    non_missing = series[~is_missing]
    # keep the MISSING token out of isin: Arrow-backed dtypes cannot ingest the
    # token object, and non-missing values can never match it anyway
    fit_values = fit_categories - {MISSING}
    n_unseen_rows = int((~non_missing.isin(fit_values)).sum())
    if MISSING not in fit_categories:
        n_unseen_rows += int(is_missing.sum())

    return ColumnCoverage(
        column=column,
        n_rows=n_rows,
        n_fit_categories=len(fit_categories),
        n_observed_categories=len(observed),
        n_categories=len(union),
        coverage=len(fit_categories) / len(union) if union else 1.0,
        n_unseen_rows=n_unseen_rows,
        unseen_ratio=n_unseen_rows / n_rows if n_rows else 0.0,
        # sorted by repr so the report is deterministic for identical inputs,
        # even with mixed-type values
        unseen_values=sorted(new, key=repr),
        all_new=len(observed) > 0 and len(observed & fit_categories) == 0,
        n_lumped=lumping.get('n_lumped', 0),
        lumped_row_fraction=lumping.get('lumped_row_fraction'),
        lumped_name=lumping.get('lumped_name'),
    )


def _render_values(values: list, max_rendered: int = MAX_RENDERED_UNSEEN) -> str:
    """Render a truncated, deterministic list of values for messages and tables."""
    rendered = ', '.join(str(v) for v in values[:max_rendered])
    if len(values) > max_rendered:
        rendered += f', ... (+{len(values) - max_rendered} more)'
    return rendered


class CoverageReport:
    """Coverage statistics of all encoded columns for one evaluated batch.

    Built once per ``transform`` call (when ``coverage_report=True``) and stored
    on the encoder as ``coverage_report_``. The tabular summary is only
    materialized on demand via :meth:`to_frame` (or the encoder's
    ``coverage_summary`` method), so wide frames do not pay for a report
    nobody reads.
    """

    def __init__(self, columns: Iterable[ColumnCoverage]):
        self.columns: dict[Hashable, ColumnCoverage] = {c.column: c for c in columns}

    def to_frame(
        self,
        cols: Iterable[Hashable] | None = None,
        label: str | None = None,
        threshold: float = 0.0,
    ) -> pd.DataFrame:
        """Materialize the per-column summary as a DataFrame.

        Parameters
        ----------
        cols: optional subset of columns to include; None includes all.
        label: optional dataset label (e.g. 'train' / 'validation') added as a
            'dataset' column, so summaries of successive transforms can be
            concatenated and compared with identical metric definitions.
        threshold: unseen-ratio threshold used to derive the 'status' column.
        """
        records = []
        for cov in self.columns.values():
            if cols is not None and cov.column not in cols:
                continue
            records.append(
                {
                    'column': cov.column,
                    'n_fit_categories': cov.n_fit_categories,
                    'n_new_categories': len(cov.unseen_values),
                    'n_categories': cov.n_categories,
                    'coverage': cov.coverage,
                    'unseen_rows': cov.n_unseen_rows,
                    'n_rows': cov.n_rows,
                    'unseen_ratio': cov.unseen_ratio,
                    'status': cov.status(threshold),
                    'lumped_categories': cov.n_lumped,
                    'lumped_row_fraction': cov.lumped_row_fraction,
                    'unseen_values': _render_values(cov.unseen_values),
                    'error': cov.error,
                }
            )
        frame = pd.DataFrame(
            records,
            columns=[
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
            ],
        )
        if label is not None:
            frame.insert(0, 'dataset', label)
        return frame

    def warn(self, threshold: float) -> None:
        """Issue a CoverageWarning for every column that misses the threshold."""
        for cov in self.columns.values():
            if cov.error is not None:
                warnings.warn(
                    f'coverage statistics for column {cov.column!r} could not be '
                    f'computed: {cov.error}',
                    CoverageWarning,
                    stacklevel=4,
                )
            elif cov.all_new:
                warnings.warn(
                    f'coverage: every value of column {cov.column!r} in this batch is '
                    f'new — none of the {cov.n_observed_categories} observed '
                    'categories was seen at fit time, so the encoding of this column '
                    'carries no fitted information. Consider refitting with these '
                    "categories included, or setting an explicit 'handle_unknown' "
                    'strategy.',
                    CoverageWarning,
                    stacklevel=4,
                )
            elif cov.unseen_ratio > threshold:
                warnings.warn(
                    f'coverage: column {cov.column!r}: {cov.unseen_ratio:.1%} of rows '
                    f'({cov.n_unseen_rows}/{cov.n_rows}) have categories not seen at '
                    f'fit time (unseen values: {_render_values(cov.unseen_values)}). '
                    f'Category coverage of this column is {cov.coverage:.1%} '
                    f'({cov.n_fit_categories}/{cov.n_categories} categories).',
                    CoverageWarning,
                    stacklevel=4,
                )


def build_report(
    cols: list,
    X: pd.DataFrame,
    fit_categories: dict,
    fit_errors: dict,
    lumping: dict,
) -> CoverageReport:
    """Build a CoverageReport for the encoded columns of one batch.

    Per-column failures (e.g. unhashable values) degrade to a ColumnCoverage
    with ``error`` set; they never raise, so reporting cannot break encoding.
    """
    columns = []
    for col in cols:
        if col in fit_errors:
            columns.append(ColumnCoverage(column=col, n_rows=len(X), error=fit_errors[col]))
            continue
        try:
            columns.append(
                compute_column_coverage(col, X[col], fit_categories[col], lumping.get(col))
            )
        except Exception as exc:  # noqa: BLE001 - reporting must not break encoding
            columns.append(ColumnCoverage(column=col, n_rows=len(X), error=str(exc)))
    return CoverageReport(columns)
