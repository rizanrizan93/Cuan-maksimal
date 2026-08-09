from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from autonomous_enrichment import _first_date
from data_providers import parse_ksei_price_history_html
from date_utils import parse_public_date


def test_public_date_variants_emit_no_warnings():
    variants = [
        "20260806",
        20260806,
        20260806.0,
        np.int64(20260806),
        np.float64(20260806.0),
        "\u00a020260806\u00a0",
        "20260806.0",
        "20260806 00:00:00",
        "2026-08-06",
        "2026-08-06T09:15:00+07:00",
        "2026/08/06",
        "06/08/2026",
        "18 Jun 2026",
        "June 18, 2026",
    ]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        parsed = [parse_public_date(value) for value in variants]
        wrapped = [_first_date(value) for value in variants]
    assert all(pd.notna(value) for value in parsed)
    assert all(pd.notna(value) for value in wrapped)
    assert not caught


def test_ksei_price_history_compact_numeric_dates_emit_no_warnings():
    html = """<table><tr><th>Date</th><th>Open</th><th>High</th><th>Low</th><th>Close</th><th>Volume</th></tr>
    <tr><td>20260806.0</td><td>100</td><td>110</td><td>95</td><td>108</td><td>1,250</td></tr>
    <tr><td>07/08/2026</td><td>108</td><td>112</td><td>105</td><td>110</td><td>1,500</td></tr></table>"""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        frame = parse_ksei_price_history_html(html)
    assert len(frame) == 2
    assert frame.iloc[0]["Volume"] == 125_000
    assert not caught
