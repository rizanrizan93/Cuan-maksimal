from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa

from ui_dataframe import arrow_safe_dataframe


def test_arrow_safe_dataframe_accepts_mixed_nested_object_cells():
    raw = pd.DataFrame(
        {
            "ticker": ["OMED.JK", "MARK.JK", "ESSA.JK", "DGWG.JK", "MERK.JK", "SSIA.JK"],
            "mixed": [
                1.25,
                "text",
                np.array([1.0, 2.0, 3.0]),
                {"score": 71.0, "flags": ["A", "B"]},
                [1, 2, 3],
                None,
            ],
            "nested": [
                np.array([[1, 2], [3, 4]]),
                {"a": np.float64(2.5)},
                ("x", 1),
                {"values": np.array([4, 5])},
                pd.Series([6, 7]),
                pd.NA,
            ],
            "score": [63.6, 66.9, 70.0, 61.9, 69.2, 64.8],
        }
    )

    safe = arrow_safe_dataframe(raw)
    table = pa.Table.from_pandas(safe, preserve_index=False)

    assert table.num_rows == len(raw)
    assert str(safe["mixed"].dtype) == "string"
    assert str(safe["nested"].dtype) == "string"
    assert safe.loc[2, "mixed"].startswith("[")
    assert '"score"' in safe.loc[3, "mixed"]
    assert np.isclose(safe.loc[0, "score"], 63.6)


def test_arrow_safe_dataframe_does_not_mutate_source_dataframe():
    nested = np.array([1, 2, 3])
    raw = pd.DataFrame({"payload": [nested], "value": [10.0]})
    safe = arrow_safe_dataframe(raw)

    assert isinstance(raw.loc[0, "payload"], np.ndarray)
    assert safe.loc[0, "payload"] == "[1, 2, 3]"
    assert float(safe.loc[0, "value"]) == 10.0
