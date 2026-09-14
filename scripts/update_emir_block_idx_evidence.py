from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
import json
import sys
import time

import pandas as pd
from zoneinfo import ZoneInfo

from emir_block_idx_eod import (
    BlockIdxClient,
    EmirBlockIdxProducer,
    PRODUCER_VERSION,
    SupabaseSink,
    weekdays,
)

WIB = ZoneInfo("Asia/Jakarta")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Populate the dedicated EMIR database from official block.idx.id")
    parser.add_argument("--mode", choices=("daily", "backfill"), default="daily")
    parser.add_argument("--from-date", type=date.fromisoformat)
    parser.add_argument("--to-date", type=date.fromisoformat)
    parser.add_argument("--months", type=int, default=6)
    parser.add_argument("--fundamental-limit", type=int, default=60)
    parser.add_argument("--skip-fundamentals", action="store_true")
    parser.add_argument("--skip-company-reference", action="store_true")
    return parser.parse_args()


def bounds(args: argparse.Namespace) -> tuple[date, date]:
    today = datetime.now(WIB).date()
    end = args.to_date or today
    if args.mode == "daily":
        return end, end
    if args.from_date:
        start = args.from_date
    else:
        # Exact calendar-month subtraction; for 2026-09-14 this starts
        # 2026-03-14. Empty non-session weekdays are never synthesized.
        start = (pd.Timestamp(end) - pd.DateOffset(months=max(1, int(args.months)))).date()
    if start > end:
        raise ValueError("from-date must be on or before to-date")
    if (end - start).days > 370:
        raise ValueError("backfill window is capped at 370 calendar days")
    return start, end


def main() -> int:
    args = parse_args()
    start, end = bounds(args)
    sink = SupabaseSink.from_env()
    producer = EmirBlockIdxProducer(BlockIdxClient(), sink)
    run_key = f"{PRODUCER_VERSION}:{args.mode}:{start.isoformat()}:{end.isoformat()}"
    sink.ingestion_run(
        run_key,
        mode=args.mode,
        requested_from=start.isoformat(),
        requested_to=end.isoformat(),
        status="RUNNING",
        producer_version=PRODUCER_VERSION,
    )

    attempted = loaded = failures = rows_persisted = 0
    detail: dict[str, object] = {"sessions": {}, "event_rows": {}, "company_rows": 0, "financial_index_rows": 0}
    try:
        for session in weekdays(start, end):
            attempted += 1
            try:
                counts = producer.collect_market_day(session)
                loaded += 1
                rows_persisted += sum(counts.values())
                detail["sessions"][session.isoformat()] = counts
            except Exception as exc:
                failures += 1
                detail["sessions"][session.isoformat()] = {
                    "state": "NO_DATA_OR_FAILED",
                    "error": f"{type(exc).__name__}: {exc}"[:600],
                }
            if args.mode == "backfill":
                time.sleep(0.25)

        event_start = start if args.mode == "backfill" else max(start - timedelta(days=45), end - timedelta(days=45))
        event_counts = producer.collect_event_window(event_start, end)
        detail["event_rows"] = event_counts
        rows_persisted += sum(event_counts.values())

        if not args.skip_company_reference and (args.mode == "backfill" or end.weekday() == 0):
            company_rows = producer.collect_company_reference(end)
            detail["company_rows"] = company_rows
            rows_persisted += company_rows

        financial_rows = producer.collect_financial_index(end)
        detail["financial_index_rows"] = financial_rows
        rows_persisted += financial_rows

        rank_state = producer.refresh_ranking(end)
        detail["pre_fundamental_rank"] = rank_state

        if not args.skip_fundamentals:
            latest = sink.select(
                "cak_idx_rank_daily",
                {
                    "select": "ticker,overall_rank",
                    "rank_date": f"eq.{rank_state.get('rank_date') if isinstance(rank_state, dict) else end.isoformat()}",
                    "order": "overall_rank.asc",
                    "limit": str(max(3, min(int(args.fundamental_limit), 100))),
                },
            )
            finalists = [str(row.get("ticker") or "") for row in latest]
            fundamental_rows = producer.collect_official_fundamentals(finalists, end)
            detail["official_fundamental_rows"] = fundamental_rows
            rows_persisted += fundamental_rows
            detail["final_rank"] = producer.refresh_ranking(end)

        status = "COMPLETED" if loaded > 0 and failures == 0 else "COMPLETED_PARTIAL"
        if loaded == 0:
            status = "SOURCE_NOT_READY"
        sink.ingestion_run(
            run_key,
            mode=args.mode,
            requested_from=start.isoformat(),
            requested_to=end.isoformat(),
            status=status,
            core_sessions_attempted=attempted,
            core_sessions_loaded=loaded,
            rows_persisted=rows_persisted,
            failures=failures,
            detail=detail,
            completed_at=datetime.now(WIB).isoformat(),
            producer_version=PRODUCER_VERSION,
        )
        print(json.dumps({
            "run_key": run_key, "status": status, "from": start.isoformat(), "to": end.isoformat(),
            "sessions_attempted": attempted, "sessions_loaded": loaded,
            "rows_persisted": rows_persisted, "failures": failures,
            "ranking": detail.get("final_rank") or detail.get("pre_fundamental_rank"),
        }, indent=2, default=str))
        return 0 if loaded > 0 else 2
    except Exception as exc:
        sink.ingestion_run(
            run_key,
            mode=args.mode,
            requested_from=start.isoformat(),
            requested_to=end.isoformat(),
            status="FAILED",
            core_sessions_attempted=attempted,
            core_sessions_loaded=loaded,
            rows_persisted=rows_persisted,
            failures=failures + 1,
            detail={**detail, "fatal": f"{type(exc).__name__}: {exc}"[:1000]},
            completed_at=datetime.now(WIB).isoformat(),
            producer_version=PRODUCER_VERSION,
        )
        raise


if __name__ == "__main__":
    sys.exit(main())
