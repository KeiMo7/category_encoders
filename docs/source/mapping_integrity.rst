Mapping integrity
=================

Every encoder keeps its fitted category mapping in ``mapping``: for the ordinal
family it is a list of ``{'col', 'mapping': pd.Series(category -> code),
'data_type'}`` entries. That table survives pickling, but on its own it carries
no identity: nothing records which fit produced it, and nothing notices that the
training side refitted while the inference side kept using the old table. The
symptom is silent — a category that was seen during training is encoded with a
different number at inference time.

The opt-in ``mapping_integrity`` mode adds a *manifest* next to the mapping. It
does not change the mapping, the encoded values or any default behaviour.

Enabling it
-----------

.. code-block:: python

    import category_encoders as ce

    encoder = ce.OrdinalEncoder(mapping_integrity=True).fit(X_train)

    # the fitted manifest: identity, size, per-category counts and content digest
    print(encoder.mapping_manifest_.summary())
    encoder.mapping_manifest_.save('mapping.manifest.jsonl')   # chunked JSON Lines
    encoder.mapping_manifest_.export_csv('mapping.review.csv')  # manual review

The parameters are available on every encoder that stores a mapping:
``OrdinalEncoder``, ``OneHotEncoder``, ``BaseNEncoder``/``BinaryEncoder``/
``GrayEncoder``, the contrast encoders, ``CountEncoder``, ``MultiHotEncoder``,
``RankHotEncoder`` and the supervised encoders (``TargetEncoder``,
``WOEEncoder``, ``MEstimateEncoder``, ``JamesSteinEncoder``,
``CatBoostEncoder``, ``LeaveOneOutEncoder``, ``QuantileEncoder``,
``CountTargetEncoder``).

Parameters
~~~~~~~~~~

``mapping_integrity`` (bool, default ``False``)
    Fit records a manifest of the fitted category mapping; transform verifies
    the mapping actually in use against it.

``mapping_manifest`` (manifest, path-like or reference encoder, default ``None``)
    The expected *training-side* manifest. ``None`` verifies against the
    encoder's own ``mapping_manifest_``; a path is re-read on every verification
    and fails if the file is missing or damaged.

``mapping_drift_threshold`` (float, default ``0.5``)
    Fraction of categories (unknown to the mapping, or fitted but absent from
    the inference data) above which transform emits ``MappingDriftWarning``.

What the manifest records
-------------------------

For the whole table: encoder name, ``fitted_at`` (UTC), ``n_samples``,
``n_columns``, ``n_categories`` and ``content_digest``. Per column: the
``handle_unknown`` and ``handle_missing`` policies as serialized values, the
fitted dtype, one digest, and the ``(category, encoded value, sample count)``
rows.

The digest is computed from the mapping **content** — category type and value,
encoded value, column name, policies and dtype — sorted by category. It is
independent of insertion order, of the file modification time, of
``fitted_at`` and of the chunk size used when saving. Two tables with the same
content always share a digest; identical timestamps cannot hide a content
difference.

Fractions, infinities, timestamps, ``NaN`` and ``None`` are canonicalized with
type tags, so ``1`` (int) and ``1.0`` (float) compare equal as *category
identity* while the digest stays type-exact.

The ``handle_unknown`` / ``handle_missing`` policies travel with the table. They
can be read back with :func:`category_encoders.mapping_manifest.strategy_values`,
so the policy in force is the one stored in the manifest, not a code default:

.. code-block:: python

    from category_encoders.mapping_manifest import MappingManifest, strategy_values

    manifest = MappingManifest.load('mapping.manifest.jsonl')
    strategy_values(manifest, 'handle_unknown')   # e.g. -1, 'error' or 'return_nan'

Only built-in (string) and numeric strategies are fully reconstructable; a
callable is recorded by module/qualname/repr and accepted only when the live
encoder is passed the same callable.

Verification
------------

With ``mapping_integrity=True``, ``transform`` verifies the mapping before
encoding. A mismatch raises ``MappingMismatchError`` and lists, per column, the
categories present only in the expected table, only in the table in use, the
categories whose encoded value changed, and any changed policy. When the
expected manifest was fitted *after* the mapping in use, the error is marked
``stale``:

.. code-block:: text

    Mapping table does not match the expected training-side manifest.
      expected: OrdinalEncoder mapping fitted at 2026-11-21T08:13:27+00:00 on 8 samples: ...
      in use:   OrdinalEncoder mapping fitted at 2026-11-01T04:00:00+00:00 on 8 samples: ...
      - column 'city': 1 category encoded differently: chicago (str): 1 -> 9
      This mapping is stale: it was fitted at 2026-11-01T04:00:00+00:00 but the expected
      manifest was fitted later, at 2026-11-21T08:13:27+00:00. Refit the encoder ...
    MappingMismatchError.stale is True

Errors and degradation
~~~~~~~~~~~~~~~~~~~~~~

* ``MappingManifestNotFoundError`` — the expected manifest is absent. Transform
  raises; it never falls back to a default encoding.
* ``MappingManifestCorruptionError`` — the manifest is damaged. The message
  names the line, the column and the category range. Chunks are digested
  individually and the column digest covers the reassembled column, so a
  tampered code, a dropped chunk or a truncated file is located rather than
  silently accepted.
* ``MappingDriftWarning`` — the inference category set differs from the fitted
  one beyond ``mapping_drift_threshold``. Per-column statistics
  (``n_unknown_categories``, ``n_absent_categories``, fractions) are returned by
  ``mapping_manifest_.check_categories(X)``; encoding proceeds with the fitted
  mapping.

Unknown categories at inference still follow the encoder's usual
``handle_unknown`` behaviour (``-1`` for ``'value'``, ``NaN`` for
``'return_nan'``, a raised ``ValueError`` for ``'error'``); the manifest simply
records which policy is in force.

File format
-----------

JSON Lines: one header line followed by chunk lines. Each chunk holds
``chunk_size`` category rows (default 1024) and its own digest, so mappings with
tens of thousands of categories are written and verified chunk by chunk without
one monolithic document. Saving is atomic (temporary file plus rename).

Backwards compatibility
-----------------------

With ``mapping_integrity=False`` (the default) no manifest is built, no
verification runs and the fit/transform output is bit-for-bit unchanged. The
manifest is a separate object (``mapping_manifest_``); ``mapping`` and the
existing ``category_mapping`` property keep their shapes.

The existing ``mapping`` constructor parameter (a supplied category mapping) is
unchanged and orthogonal: with integrity enabled, the supplied table is
recorded in the manifest like a fitted one.
