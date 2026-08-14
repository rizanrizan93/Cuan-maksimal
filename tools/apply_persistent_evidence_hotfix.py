from pathlib import Path

path = Path("resumable_scan.py")
text = path.read_text(encoding="utf-8")

import_anchor = "from future_fundamental import calculate_future_fundamental, future_fundamental_evidence_frame\n"
import_line = "from persistent_direct_evidence import load_verified_direct_evidence\n"
if import_line not in text:
    if import_anchor not in text:
        raise SystemExit("import anchor missing")
    text = text.replace(import_anchor, import_anchor + import_line, 1)

manual_anchor = '    manual_events = normalize_manual_events(_frame(settings.get("manual_events")))\n'
manual_replacement = '''    persistent_direct = load_verified_direct_evidence(\n        config, universe["ticker"].tolist(), as_of=now\n    ) if config.ready else {}\n    manual_events = normalize_manual_events(_frame(settings.get("manual_events")))\n    persisted_forward_events = normalize_manual_events(\n        persistent_direct.get("official_forward_events", pd.DataFrame())\n    )\n'''
if manual_anchor in text:
    text = text.replace(manual_anchor, manual_replacement, 1)
elif "persisted_forward_events = normalize_manual_events" not in text:
    raise SystemExit("manual event anchor missing")

event_anchor = "    event_frames = [frame for frame in (manual_events, online_events, ksei_events, official_events_frame) if isinstance(frame, pd.DataFrame) and not frame.empty]\n"
event_replacement = "    event_frames = [frame for frame in (persisted_forward_events, manual_events, online_events, ksei_events, official_events_frame) if isinstance(frame, pd.DataFrame) and not frame.empty]\n"
if event_anchor in text:
    text = text.replace(event_anchor, event_replacement, 1)
elif "persisted_forward_events, manual_events" not in text:
    raise SystemExit("event frame anchor missing")

raw_block = '''    raw_broker = _frame(settings.get("manual_broker"))\n    raw_ownership = _frame(settings.get("manual_ownership"))\n    raw_orderbook = _frame(settings.get("manual_orderbook"))\n    raw_idx_integrity = _frame(settings.get("manual_idx_integrity"))\n'''
raw_replacement = '''    def _merge_persisted_manual(key: str, setting_key: str) -> pd.DataFrame:\n        persisted = persistent_direct.get(key, pd.DataFrame()) if isinstance(persistent_direct, Mapping) else pd.DataFrame()\n        manual = _frame(settings.get(setting_key))\n        frames = [frame for frame in (persisted, manual) if isinstance(frame, pd.DataFrame) and not frame.empty]\n        return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()\n\n    raw_broker = _merge_persisted_manual("broker", "manual_broker")\n    raw_ownership = _merge_persisted_manual("ownership", "manual_ownership")\n    raw_orderbook = _merge_persisted_manual("orderbook", "manual_orderbook")\n    raw_idx_integrity = _merge_persisted_manual("idx_integrity", "manual_idx_integrity")\n'''
if raw_block in text:
    text = text.replace(raw_block, raw_replacement, 1)
elif '_merge_persisted_manual("ownership"' not in text:
    raise SystemExit("raw evidence anchor missing")

audit_anchor = '''    audit_frames = [frame for frame in (\n        chunk_audit, ohlcv_load_audit, benchmark_load_audit, ksei_load_audit, news_load_audit, fundamental_load_audit, official_fundamental_load_audit,\n    ) if isinstance(frame, pd.DataFrame) and not frame.empty]\n'''
audit_replacement = '''    persistent_direct_audit = persistent_direct.get("audit", pd.DataFrame()) if isinstance(persistent_direct, Mapping) else pd.DataFrame()\n    audit_frames = [frame for frame in (\n        chunk_audit, ohlcv_load_audit, benchmark_load_audit, ksei_load_audit, news_load_audit, fundamental_load_audit, official_fundamental_load_audit, persistent_direct_audit,\n    ) if isinstance(frame, pd.DataFrame) and not frame.empty]\n'''
if audit_anchor in text:
    text = text.replace(audit_anchor, audit_replacement, 1)
elif "persistent_direct_audit" not in text:
    raise SystemExit("audit anchor missing")

path.write_text(text, encoding="utf-8")
