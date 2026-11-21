.. _coverage:

Category Coverage Reporting
===========================

.. currentmodule:: category_encoders

When a model is validated or deployed, the data often contains categories the
encoder never saw at fit time. The row still gets encoded (according to
``handle_unknown``), but silently: a column whose validation values are all new
encodes to a single constant, and nothing tells you the model effectively never
learned that column.

All encoders can report how well they *cover* the data being transformed. The
feature is opt-in via two constructor parameters and is off by default; with
``coverage_report=False`` (the default) behaviour is exactly as before.

.. code-block:: python

    import category_encoders as ce

    enc = ce.OrdinalEncoder(cols=['city', 'color'],
                            coverage_report=True, coverage_threshold=0.0)
    enc.fit(X_train)
    enc.transform(X_valid)          # warns if a column misses the threshold
    print(enc.coverage_summary(label='validation'))

Parameters
----------

``coverage_report: bool = False``
    Enables the collection of per-column coverage statistics at fit time and at
    every ``transform`` call, plus warnings when a transformed batch contains
    categories never seen at fit time.

``coverage_threshold: float = 0.0``
    Fraction of rows with unseen categories above which a column triggers a
    :class:`~category_encoders.coverage.CoverageWarning` at transform time and
    is flagged ``low_coverage`` in the summary. Must be in ``[0, 1]``; the
    default ``0.0`` warns on any unseen row.

Metrics
-------

All definitions are identical for every evaluated batch, so summaries of
different datasets (train / validation / inference) are directly comparable.
All missing-value spellings (``None``, ``np.nan``, ``pd.NA``, ...) count as one
single ``<missing>`` category.

``n_categories``
    Distinct values of the column across **both** the fit data and the
    evaluated batch (fit ∪ batch). This is the single denominator of the
    coverage metric — deliberately *not* restricted to the values seen at fit
    time.
``coverage``
    ``n_fit_categories / n_categories``: the share of all distinct values the
    encoder knows.
``unseen_ratio``
    ``unseen_rows / n_rows``: the share of *rows* of the evaluated batch whose
    value was never seen at fit time.
``status``
    ``ok``, ``low_coverage`` (unseen_ratio above the threshold), ``all_new``
    (no observed value was seen at fit time) or ``error`` (statistics of this
    column could not be computed).

When ``min_group_size`` lumping merged rare categories at fit time, the summary
additionally reports ``lumped_categories`` (how many categories were merged)
and ``lumped_row_fraction`` (the share of rows the merged group covers). This
works both for the encoder-level ``min_group_size`` and for
``CountEncoder``'s own lumping; the lumping itself is unchanged.

The summary table
-----------------

``coverage_summary(cols=None, label=None)`` returns one row per encoded column
and is built **on demand**: ``transform`` only stores compact per-column
counters and value sets, the table is materialized when you ask for it. On wide
frames (thousands of columns) you can restrict it with ``cols=[...]``. Passing
``label='validation'`` stamps a ``dataset`` column, so summaries of successive
transforms can be concatenated and compared:

.. code-block:: python

    enc.transform(X_train)
    train_summary = enc.coverage_summary(label='train')
    enc.transform(X_valid)
    valid_summary = enc.coverage_summary(label='validation')
    report = pd.concat([train_summary, valid_summary])

After ``fit`` the summary describes the training data itself; every
``transform`` replaces the stored statistics with those of the new batch
(``enc.coverage_report_`` holds the raw report object).

Edge cases
----------

* **All-new column**: if no value of a column was seen at fit time, the column
  is flagged ``all_new`` and the warning suggests refitting with these
  categories included or setting an explicit ``handle_unknown`` strategy.
* **Empty data**: fitting or transforming 0 rows with ``coverage_report=True``
  raises a ``ValueError`` — a report on no data would be meaningless.
* **Report failure**: if the statistics of a column cannot be computed (e.g.
  unhashable values), a ``CoverageWarning`` is issued and the column is marked
  ``error`` — encoding itself is never affected.

Known limitations
-----------------

* The statistics are per ``transform`` batch; the report reflects the most
  recent batch only. Use ``coverage_summary(label=...)`` and concatenate to
  keep several.
* ``SummaryEncoder`` and the wrappers in ``category_encoders.wrapper`` do not
  expose the parameters (enable coverage on the wrapped encoder instead).
* Models pickled by older versions have no fit-time statistics: they behave as
  if coverage were disabled. To get reports, set ``coverage_report=True`` and
  refit. Data and models produced with the feature off are fully compatible —
  the encoded output is identical either way.
* For ``HashingEncoder`` coverage reflects the categories seen at fit time even
  though the hashing trick itself has no fixed category set; treat its report
  as informational.
