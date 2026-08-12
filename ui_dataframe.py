from __future__ import annotations

from typing import Any
import json

import numpy as np
import pandas as pd


def _is_missing_scalar(value: Any) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except Exception:
        return False
    return bool(missing) if isinstance(missing, (bool, np.bool_)) else False


def _jsonable(value: Any) -> Any:
    if _is_missing_scalar(value):
        return None
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, pd.Series):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return [_jsonable(item) for item in sorted(value, key=str)]
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        stamp = pd.Timestamp(value)
        return stamp.isoformat() if pd.notna(stamp) else None
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", errors="replace")
    return value


def _object_cell_for_display(value: Any) -> str | None:
    if _is_missing_scalar(value):
        return None
    if isinstance(value, (dict, list, tuple, set, np.ndarray, pd.Series)):
        try:
            return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, default=str)
        except Exception:
            return str(value)
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", errors="replace")
    return str(value)


def arrow_safe_dataframe(data: Any) -> pd.DataFrame:
    """Return a display-only DataFrame that PyArrow can serialize safely.

    Scanner/export DataFrames remain untouched. Only object/string/category columns
    are normalized for Streamlit rendering, preventing ArrowInvalid when a provider
    column contains mixed scalars and nested numpy/list/dict values.
    """
    if isinstance(data, pd.DataFrame):
        local = data.copy(deep=False).copy()
    elif data is None:
        local = pd.DataFrame()
    else:
        local = pd.DataFrame(data)

    local.columns = [str(column) for column in local.columns]
    for column in local.columns:
        series = local[column]
        dtype = series.dtype
        if (
            pd.api.types.is_object_dtype(dtype)
            or pd.api.types.is_string_dtype(dtype)
            or isinstance(dtype, pd.CategoricalDtype)
        ):
            local[column] = series.map(_object_cell_for_display).astype("string")
        elif pd.api.types.is_complex_dtype(dtype):
            local[column] = series.map(_object_cell_for_display).astype("string")
    return local


def streamlit_dataframe(data: Any, *args: Any, **kwargs: Any) -> Any:
    """Render a DataFrame without allowing PyArrow mixed-type failures to crash UI."""
    import streamlit as st

    prepared = arrow_safe_dataframe(data)
    try:
        return st.dataframe(prepared, *args, **kwargs)
    except Exception:
        # Final display-only fallback: stringify every column. This does not touch
        # the source DataFrame used for ranking, persistence, or CSV downloads.
        fallback = prepared.copy()
        for column in fallback.columns:
            fallback[column] = fallback[column].map(_object_cell_for_display).astype("string")
        return st.dataframe(fallback, *args, **kwargs)


__all__ = ["arrow_safe_dataframe", "streamlit_dataframe"]
