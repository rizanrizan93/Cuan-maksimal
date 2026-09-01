from __future__ import annotations

from datetime import timedelta
from pathlib import Path
import gzip
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from idx_public_participant_provider import aggregate_trade_detail, download_trade_detail, trim_daily_top_flow
from shared_participant_evidence import SharedParticipantEvidence


CACHE_PATH = Path("data/public_broker_flow_30d.csv.gz")
OWNER = "EMIR"


def _load_existing() -> pd.DataFrame:
    try:
        with gzip.open(CACHE_PATH, "rb") as stream:
            return pd.read_csv(stream)
    except (FileNotFoundError, OSError, ValueError):
        return pd.DataFrame()


def main() -> int:
    existing = _load_existing()
    shared = SharedParticipantEvidence(OWNER)
    print({"owner": OWNER, "shared_evidence_hub": shared.status()})
    today = pd.Timestamp.now(tz="Asia/Jakarta").tz_localize(None).normalize()
    daily = pd.DataFrame()
    source_url = ""
    source_date = None
    for offset in range(7):
        candidate = (today - timedelta(days=offset)).date()
        if candidate.weekday() >= 5:
            continue
        if shared.ready:
            daily, meta = shared.get_day(candidate, download=download_trade_detail, aggregate=aggregate_trade_detail)
            for attempt in meta.get("diagnostics", []):
                print({"owner": OWNER, **attempt})
            print({"owner": OWNER, "trade_date": candidate.isoformat(), **{key: meta.get(key) for key in ("state", "provider_called", "request_avoided", "lease_state", "rows", "ticker_breadth")}})
            source_url = str(meta.get("source_url") or "SHARED_IDX_EVIDENCE_HUB")
            if str(meta.get("state")) == "REFRESH_LOCKED":
                break
        else:
            path = None
            try:
                path, source_url = download_trade_detail(candidate)
                daily = aggregate_trade_detail(path, candidate)
            except Exception as exc:
                print(f"candidate {candidate}: {type(exc).__name__}: {exc}")
            finally:
                if path is not None:
                    path.unlink(missing_ok=True)
        daily = trim_daily_top_flow(daily)
        if not daily.empty:
            source_date = candidate
            break
    if daily.empty:
        print("No public IDX Trade Detail file available; cache remains unchanged.")
        return 0
    combined = pd.concat([existing, daily], ignore_index=True) if not existing.empty else daily
    combined["trade_date"] = pd.to_datetime(combined["trade_date"], errors="coerce").dt.normalize()
    combined = combined.dropna(subset=["trade_date", "ticker", "broker_code", "side"])
    combined = combined.drop_duplicates(["trade_date", "ticker", "broker_code", "side"], keep="last")
    dates = sorted(combined["trade_date"].unique())[-30:]
    combined = combined[combined["trade_date"].isin(dates)].sort_values(["trade_date", "ticker", "side", "net_rank"], kind="stable")
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(CACHE_PATH, "wb") as stream:
        combined.to_csv(stream, index=False)
    print({"owner": OWNER, "state": "PARTICIPANT_CACHE_WRITTEN", "source_date": str(source_date), "rows": len(combined), "source": source_url, "shared_cache": shared.metrics()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
