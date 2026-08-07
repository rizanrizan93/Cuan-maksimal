from __future__ import annotations

from datetime import date, datetime
from typing import Any
import math
import re
import unicodedata

import numpy as np
import pandas as pd

_MISSING_TEXT = {"", "nan", "none", "nat", "<na>", "null", "n/a", "na", "-"}
_EXPLICIT_FORMATS = (
    "%Y%m%d",
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d-%m-%Y",
    "%d/%m/%Y",
    "%d.%m.%Y",
    "%d %b %Y",
    "%d %B %Y",
    "%b %d, %Y",
    "%B %d, %Y",
)


def _normalise_date_text(value: Any) -> str:
    """Return a stable textual representation for public-provider date fields.

    KSEI/public tables may expose compact dates as numbers (for example
    ``20260806.0``), strings with non-breaking spaces, or compact dates followed
    by a timestamp. Normalising first prevents pandas from inferring ``%Y%m%d``
    while ``dayfirst=True`` and emitting one warning per corporate-action row.
    """
    if value is None:
        return ""
    if isinstance(value, (pd.Timestamp, datetime, date, np.datetime64)):
        try:
            return pd.Timestamp(value).isoformat()
        except Exception:
            return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            return ""
        if number.is_integer():
            return str(int(number))
        return format(number, "f").rstrip("0").rstrip(".")

    text = unicodedata.normalize("NFKC", str(value)).replace("\u00a0", " ").strip()
    text = re.sub(r"\s+", " ", text)
    if text.lower() in _MISSING_TEXT:
        return ""
    # CSV/HTML-to-frame conversions may turn YYYYMMDD into YYYYMMDD.0.
    text = re.sub(r"^(\d{8})\.0+$", r"\1", text)
    return text


def parse_public_date(value: Any) -> pd.Timestamp | pd.NaT:
    """Parse provider dates deterministically and without inference warnings."""
    if isinstance(value, pd.Timestamp):
        return value
    if isinstance(value, (datetime, date, np.datetime64)):
        try:
            return pd.Timestamp(value)
        except Exception:
            return pd.NaT

    text = _normalise_date_text(value)
    if not text:
        return pd.NaT

    # Compact YYYYMMDD, optionally followed by a time component.
    match = re.fullmatch(r"(\d{8})(?:[ T].*)?", text)
    if match:
        return pd.to_datetime(match.group(1), format="%Y%m%d", errors="coerce")

    # ISO date/timestamp: parse the date component explicitly.
    match = re.match(r"^(\d{4}-\d{2}-\d{2})(?:[ T].*)?$", text)
    if match:
        return pd.to_datetime(match.group(1), format="%Y-%m-%d", errors="coerce")

    for fmt in _EXPLICIT_FORMATS[2:]:
        parsed = pd.to_datetime(text, format=fmt, errors="coerce")
        if pd.notna(parsed):
            return parsed

    # ``format='mixed'`` disables deprecated format inference and therefore
    # avoids the repetitive pandas warning seen in the live Streamlit log.
    try:
        return pd.to_datetime(text, format="mixed", errors="coerce", dayfirst=True)
    except (TypeError, ValueError):
        return pd.NaT
