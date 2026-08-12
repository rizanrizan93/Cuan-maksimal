# Emir v1.9.15 Hotfix 3 — Arrow-safe Dashboard

## Problem fixed

A completed scan could crash while rendering `Autonomous evidence` in Streamlit with `pyarrow.lib.ArrowInvalid`. Provider/evidence DataFrames may contain object columns with mixed Python scalars and nested `numpy.ndarray`, `list`, `tuple`, `set`, `Series`, or `dict` values. Streamlit serializes DataFrames through PyArrow, which requires a stable type per column.

## Fix

- Added `ui_dataframe.py` with a display-only Arrow-safe normalizer.
- Object/string/category/complex columns are normalized only for Streamlit rendering.
- Nested values are serialized to compact JSON strings.
- Numeric/datetime columns remain numeric/datetime where possible.
- CSV exports, ranking DataFrames, persistence payloads, scoring, and execution logic are not modified.
- All `st.dataframe()` calls in `app.py` are routed through the safe renderer, preventing the same failure in other tabs.
- Added a final all-string rendering fallback if PyArrow still rejects an unusual provider cell.

## Compatibility

Core methodology remains `1.9.14-future-fundamental-db-acceleration`; this hotfix is UI/runtime-only and does not invalidate existing scan jobs, caches, or persisted results. Free-tier storage safety from v1.9.15 remains unchanged.

## Regression coverage

`tests/test_ui_dataframe_arrow_safe.py` verifies mixed scalar/nested values can be converted to a PyArrow table and confirms the source DataFrame is not mutated.
