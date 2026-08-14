from pathlib import Path

RM = Path('research_memory.py')
RS = Path('resumable_scan.py')

rm = RM.read_text(encoding='utf-8')
rs = RS.read_text(encoding='utf-8')

marker = '\n\n__all__ = ["build_research_memory_rows", "persist_verify_research_memory", "load_latest_research_memory"]\n'
helper = r'''

def load_replayable_narrative_events(
    config: DatabaseConfig,
    tickers: list[str],
    *,
    as_of: Any = None,
    limit_per_ticker: int = 6,
    max_age_days: int = 540,
) -> pd.DataFrame:
    """Replay bounded raw narrative evidence from durable research memory.

    Only raw event payloads are replayed. Derived score snapshots are never fed
    back into scoring, which prevents circular coverage inflation. Current-scan
    events retain precedence because the caller appends this frame last and
    deduplicates by ticker/title/url. Future-dated observations and very old
    memories are excluded; source verification is preserved, never promoted.
    """
    if not config.ready or not tickers:
        return pd.DataFrame()
    memory = load_latest_research_memory(
        config, tickers, "NARRATIVE_EVENT", limit_per_ticker=max(1, int(limit_per_ticker))
    )
    now = pd.Timestamp.now(tz="UTC") if as_of is None else pd.Timestamp(as_of)
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    rows: list[dict[str, Any]] = []
    for ticker in tickers:
        for item in memory.get(str(ticker), []):
            payload = item.get("payload")
            if not isinstance(payload, Mapping):
                continue
            row = dict(payload)
            observed = pd.to_datetime(
                row.get("published_at") or row.get("event_date") or item.get("observed_at"),
                errors="coerce", utc=True,
            )
            if pd.isna(observed):
                continue
            age_days = (now - observed).total_seconds() / 86400.0
            if age_days < -1.0 or age_days > max(1, int(max_age_days)):
                continue
            row["ticker"] = str(row.get("ticker") or ticker)
            row["published_at"] = observed
            row.setdefault("source_verified", bool(item.get("source_verified", False)))
            if not row.get("source_tier") and bool(item.get("official_source", False)):
                row["source_tier"] = "OFFICIAL"
            row["research_memory_replayed"] = True
            row["research_memory_content_sha256"] = str(item.get("content_sha256") or "")
            row["research_memory_original_provider"] = str(item.get("provider") or "")
            row["collection_provider"] = "PERSISTED_RESEARCH_MEMORY"
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    dedupe = [column for column in ("ticker", "title", "url") if column in frame.columns]
    if dedupe:
        frame = frame.drop_duplicates(dedupe, keep="first")
    return frame.reset_index(drop=True)
'''

if 'def load_replayable_narrative_events(' not in rm:
    if marker not in rm:
        raise SystemExit('research_memory __all__ marker not found')
    rm = rm.replace(marker, helper + '\n\n__all__ = ["build_research_memory_rows", "persist_verify_research_memory", "load_latest_research_memory", "load_replayable_narrative_events"]\n')

old_import = 'from research_memory import build_research_memory_rows, persist_verify_research_memory'
new_import = 'from research_memory import build_research_memory_rows, persist_verify_research_memory, load_replayable_narrative_events'
if new_import not in rs:
    if old_import not in rs:
        raise SystemExit('resumable_scan research_memory import marker not found')
    rs = rs.replace(old_import, new_import, 1)

load_marker = '    online_events, news_load_audit = load_cached_news(config, shortlist)\n'
load_block = load_marker + '    persisted_narrative_events = load_replayable_narrative_events(\n        config, shortlist, as_of=now, limit_per_ticker=6, max_age_days=540\n    ) if config.ready else pd.DataFrame()\n'
if 'persisted_narrative_events = load_replayable_narrative_events(' not in rs:
    if load_marker not in rs:
        raise SystemExit('online event load marker not found')
    rs = rs.replace(load_marker, load_block, 1)

old_events = 'event_frames = [frame for frame in (persisted_forward_events, manual_events, online_events, ksei_events, official_events_frame) if isinstance(frame, pd.DataFrame) and not frame.empty]'
new_events = 'event_frames = [frame for frame in (persisted_forward_events, manual_events, online_events, ksei_events, official_events_frame, persisted_narrative_events) if isinstance(frame, pd.DataFrame) and not frame.empty]'
if new_events not in rs:
    if old_events not in rs:
        raise SystemExit('event_frames marker not found')
    rs = rs.replace(old_events, new_events, 1)

RM.write_text(rm, encoding='utf-8')
RS.write_text(rs, encoding='utf-8')
print('research memory replay patch applied')
