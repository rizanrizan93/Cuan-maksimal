"""Deterministic IDX/Jakarta completed-session calendar."""
from __future__ import annotations

from datetime import time, timedelta
from enum import Enum
from typing import Any, Iterable

import pandas as pd

CALENDAR_VERSION = "2026.08.30-official-2025-2026-session-contract"
CALENDAR_SOURCE = "IDX_OFFICIAL_EXCHANGE_HOLIDAY_ANNOUNCEMENT"
JAKARTA_TIMEZONE = "Asia/Jakarta"
SESSION_COMPLETION_TIME = time(16, 20)


class CalendarState(str, Enum):
    TRADING_SESSION = "TRADING_SESSION"
    CLOSED = "CLOSED"
    UNKNOWN = "UNKNOWN"


class CalendarCoverageError(ValueError):
    pass


_CLOSED = {
    2025: {
        "2025-01-01", "2025-01-27", "2025-01-28", "2025-01-29",
        "2025-03-28", "2025-03-31", "2025-04-01", "2025-04-02", "2025-04-03",
        "2025-04-04", "2025-04-07", "2025-04-18", "2025-05-01", "2025-05-12",
        "2025-05-13", "2025-05-29", "2025-05-30", "2025-06-06", "2025-06-09",
        "2025-06-27", "2025-08-18", "2025-09-05", "2025-12-25", "2025-12-26",
        "2025-12-31",
    },
    2026: {
        "2026-01-01", "2026-01-16", "2026-02-16", "2026-02-17", "2026-03-18",
        "2026-03-19", "2026-03-20", "2026-03-23", "2026-03-24", "2026-04-03",
        "2026-05-01", "2026-05-14", "2026-05-15", "2026-05-27", "2026-05-28",
        "2026-06-01", "2026-06-16", "2026-08-17", "2026-08-25", "2026-12-24",
        "2026-12-25", "2026-12-31",
    },
}
SUPPORTED_CALENDAR_YEARS = frozenset(_CLOSED)
OFFICIAL_CLOSED_DATES = frozenset(
    pd.Timestamp(value).normalize() for values in _CLOSED.values() for value in values
)


def _jakarta_timestamp(value: Any) -> pd.Timestamp:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        raise ValueError(f"invalid calendar value: {value!r}") from None
    if pd.isna(stamp):
        raise ValueError(f"invalid calendar value: {value!r}")
    return stamp.tz_localize(JAKARTA_TIMEZONE) if stamp.tzinfo is None else stamp.tz_convert(JAKARTA_TIMEZONE)


def _dates(values: Iterable[Any] | None) -> set[pd.Timestamp]:
    result = set()
    for value in values or ():
        try:
            result.add(_jakarta_timestamp(value).tz_localize(None).normalize())
        except (TypeError, ValueError):
            continue
    return result


def calendar_state(value: Any, *, extra_open_dates: Iterable[Any] | None = None,
                   extra_closed_dates: Iterable[Any] | None = None) -> CalendarState:
    try:
        day = _jakarta_timestamp(value).tz_localize(None).normalize()
    except (TypeError, ValueError):
        return CalendarState.UNKNOWN
    if day.year not in SUPPORTED_CALENDAR_YEARS:
        return CalendarState.UNKNOWN
    if day in _dates(extra_open_dates):
        return CalendarState.TRADING_SESSION
    if day.weekday() >= 5 or day in OFFICIAL_CLOSED_DATES or day in _dates(extra_closed_dates):
        return CalendarState.CLOSED
    return CalendarState.TRADING_SESSION


def is_idx_session(value: Any, **kwargs: Any) -> bool:
    return calendar_state(value, **kwargs) is CalendarState.TRADING_SESSION


def previous_idx_session(value: Any, *, include_date: bool = True) -> pd.Timestamp:
    day = _jakarta_timestamp(value).tz_localize(None).normalize()
    if not include_date:
        day -= timedelta(days=1)
    for _ in range(740):
        state = calendar_state(day)
        if state is CalendarState.TRADING_SESSION:
            return day
        if state is CalendarState.UNKNOWN:
            raise CalendarCoverageError(f"IDX calendar has no coverage for {day.date().isoformat()}")
        day -= timedelta(days=1)
    raise RuntimeError("no previous IDX session in covered range")


def n_idx_sessions_ago(value: Any, sessions: int) -> pd.Timestamp:
    if int(sessions) < 0:
        raise ValueError("sessions must be >= 0")
    current = previous_idx_session(value)
    for _ in range(int(sessions)):
        current = previous_idx_session(current, include_date=False)
    return current


def trading_session_age(observed_at: Any, decision_at: Any) -> int | None:
    try:
        observed = _jakarta_timestamp(observed_at).tz_localize(None).normalize()
        decision = _jakarta_timestamp(decision_at).tz_localize(None).normalize()
    except (TypeError, ValueError):
        return None
    if calendar_state(observed) is not CalendarState.TRADING_SESSION or calendar_state(decision) is CalendarState.UNKNOWN:
        return None
    if decision < observed:
        return -1
    age = 0
    day = observed + timedelta(days=1)
    while day <= decision:
        state = calendar_state(day)
        if state is CalendarState.UNKNOWN:
            return None
        age += int(state is CalendarState.TRADING_SESSION)
        day += timedelta(days=1)
    return age


def latest_expected_completed_session(value: Any = None, *, completion_time: time = SESSION_COMPLETION_TIME) -> pd.Timestamp:
    local = _jakarta_timestamp(pd.Timestamp.now(tz=JAKARTA_TIMEZONE) if value is None else value)
    today = local.tz_localize(None).normalize()
    state = calendar_state(today)
    if state is CalendarState.UNKNOWN:
        raise CalendarCoverageError(f"IDX calendar has no coverage for {today.date().isoformat()}")
    include_today = state is CalendarState.TRADING_SESSION and local.time().replace(tzinfo=None) >= completion_time
    return previous_idx_session(today, include_date=include_today)


def idx_session_lag(last_date: Any, expected_date: Any) -> int:
    age = trading_session_age(last_date, expected_date)
    return 9999 if age is None else age


__all__ = ["CALENDAR_SOURCE", "CALENDAR_VERSION", "CalendarCoverageError", "CalendarState",
           "JAKARTA_TIMEZONE", "OFFICIAL_CLOSED_DATES", "SESSION_COMPLETION_TIME",
           "SUPPORTED_CALENDAR_YEARS", "calendar_state", "idx_session_lag", "is_idx_session",
           "latest_expected_completed_session", "n_idx_sessions_ago", "previous_idx_session",
           "trading_session_age"]
