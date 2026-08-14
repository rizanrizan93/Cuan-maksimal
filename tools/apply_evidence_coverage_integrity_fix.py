from pathlib import Path

# 1) Prevent administrative/earnings rows and ticker-name collisions from being
# counted as project/contract/guidance evidence.
path = Path("future_fundamental.py")
text = path.read_text(encoding="utf-8")
text = text.replace('FUTURE_FUNDAMENTAL_VERSION = "1.0.1-evidence-confidence-penalty"', 'FUTURE_FUNDAMENTAL_VERSION = "1.0.2-forward-event-integrity"')
old = '''def _event_text(row: Mapping[str, Any]) -> str:\n    return " ".join(str(row.get(key) or "") for key in ("title", "summary", "category")).lower()\n\n\ndef _weighted_observed'''
new = '''def _event_text(row: Mapping[str, Any]) -> str:\n    return " ".join(str(row.get(key) or "") for key in ("title", "summary", "category")).lower()\n\n\ndef _event_matches_terms(row: Mapping[str, Any], terms: tuple[str, ...]) -> bool:\n    \"\"\"Match forward terms without turning ticker/company identity into a catalyst.\n\n    Administrative corporate actions and backward-looking earnings filings are\n    evidence for other scanner layers, not forward project/contract evidence.\n    The bare ticker is stripped before term matching so symbols such as MINE do\n    not become a mining-project event merely because the symbol appears in a\n    title or KSEI summary.\n    \"\"\"\n    category = str(row.get("category") or "").strip().upper()\n    role = str(row.get("event_role") or "").strip().upper()\n    if role == "ADMINISTRATIVE_CORPORATE_ACTION" or category in {"EARNINGS_CONVERSION", "CORPORATE_ACTION"}:\n        return False\n    text = _event_text(row)\n    ticker = str(row.get("ticker") or "").strip().lower()\n    if ticker.endswith(".jk"):\n        ticker = ticker[:-3]\n    if ticker:\n        text = re.sub(rf"(?<![a-z0-9]){re.escape(ticker)}(?![a-z0-9])", " ", text)\n    for raw_term in terms:\n        term = str(raw_term or "").strip().lower()\n        if not term:\n            continue\n        if re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text):\n            return True\n    return False\n\n\ndef _weighted_observed'''
if old not in text and 'def _event_matches_terms' not in text:
    raise SystemExit('future event helper anchor not found')
if old in text:
    text = text.replace(old, new, 1)
old_match = '''        text = _event_text(row)\n        if not any(term in text for term in terms):\n            continue\n'''
new_match = '''        text = _event_text(row)\n        if not _event_matches_terms(row, terms):\n            continue\n'''
count = text.count(old_match)
if count:
    text = text.replace(old_match, new_match)
elif text.count('_event_matches_terms(row, terms)') < 2:
    raise SystemExit('future event match anchors not found')
path.write_text(text, encoding="utf-8")

# 2) Merge direct ownership/integrity evidence field-by-field over autonomous
# public context instead of replacing the whole profile.
path = Path("resumable_scan.py")
text = path.read_text(encoding="utf-8")
anchor = '''def _truthy(value: Any) -> bool:\n    return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "verified"}\n\n\ndef normalize_manual_events'''
helper = '''def _truthy(value: Any) -> bool:\n    return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "verified"}\n\n\ndef _evidence_value_present(value: Any) -> bool:\n    if value is None:\n        return False\n    if isinstance(value, str):\n        return bool(value.strip())\n    try:\n        missing = pd.isna(value)\n        if isinstance(missing, (bool, np.bool_)):\n            return not bool(missing)\n    except Exception:\n        pass\n    return True\n\n\ndef _merge_evidence_profile_maps(\n    base_map: Mapping[str, Mapping[str, Any]] | None,\n    direct_map: Mapping[str, Mapping[str, Any]] | None,\n    *,\n    coverage_key: str,\n    provenance_key: str,\n    hard_block_key: str | None = None,\n    reason_key: str | None = None,\n) -> dict[str, dict[str, Any]]:\n    \"\"\"Non-destructive evidence merge for autonomous + direct profiles.\n\n    Verified direct fields win where they are actually observed. Missing fields\n    retain the autonomous public context. Coverage cannot decrease merely because\n    a narrower but higher-tier direct row arrived. Safety hard blocks are OR'ed,\n    never cleared by a partial direct disclosure.\n    \"\"\"\n    base_map = dict(base_map or {})\n    direct_map = dict(direct_map or {})\n    output: dict[str, dict[str, Any]] = {}\n    for ticker in sorted(set(base_map) | set(direct_map)):\n        base = dict(base_map.get(ticker) or {})\n        direct = dict(direct_map.get(ticker) or {})\n        merged = dict(base)\n        for key, value in direct.items():\n            if _evidence_value_present(value):\n                merged[key] = value\n        base_cov = _finite(base.get(coverage_key), 0.0)\n        direct_cov = _finite(direct.get(coverage_key), 0.0)\n        merged[coverage_key] = round(max(base_cov, direct_cov), 1)\n        base_prov = str(base.get(provenance_key) or "").strip()\n        direct_prov = str(direct.get(provenance_key) or "").strip()\n        if base_prov and direct_prov and base_prov != direct_prov:\n            merged[provenance_key] = f"{direct_prov}+{base_prov}"\n        elif direct_prov:\n            merged[provenance_key] = direct_prov\n        elif base_prov:\n            merged[provenance_key] = base_prov\n        if hard_block_key:\n            merged[hard_block_key] = bool(base.get(hard_block_key, False)) or bool(direct.get(hard_block_key, False))\n        if reason_key:\n            reasons: list[str] = []\n            for source in (base.get(reason_key), direct.get(reason_key)):\n                for token in str(source or "").split("|"):\n                    token = token.strip()\n                    if token and token.upper() != "NONE" and token not in reasons:\n                        reasons.append(token)\n            merged[reason_key] = " | ".join(reasons) if reasons else "NONE"\n        output[ticker] = merged\n    return output\n\n\ndef normalize_manual_events'''
if anchor in text:
    text = text.replace(anchor, helper, 1)
elif 'def _merge_evidence_profile_maps(' not in text:
    raise SystemExit('resumable merge helper anchor not found')
old_maps = '''    broker_map = {**broker_proxy_map, **aggregate_broker_summary(raw_broker)}\n    ownership_map = {**ownership_auto_map, **parse_ownership(raw_ownership)}\n    orderbook_map = {**orderbook_proxy_map, **parse_orderbook_evidence(raw_orderbook)}\n    idx_integrity_map = {**integrity_auto_map, **parse_idx_integrity(raw_idx_integrity, as_of=now)}\n'''
new_maps = '''    broker_map = {**broker_proxy_map, **aggregate_broker_summary(raw_broker)}\n    ownership_map = _merge_evidence_profile_maps(\n        ownership_auto_map, parse_ownership(raw_ownership),\n        coverage_key="ownership_coverage_pct", provenance_key="ownership_provenance_state",\n    )\n    orderbook_map = {**orderbook_proxy_map, **parse_orderbook_evidence(raw_orderbook)}\n    idx_integrity_map = _merge_evidence_profile_maps(\n        integrity_auto_map, parse_idx_integrity(raw_idx_integrity, as_of=now),\n        coverage_key="idx_integrity_coverage_pct", provenance_key="idx_integrity_provenance_state",\n        hard_block_key="idx_integrity_hard_block", reason_key="idx_integrity_block_reasons",\n    )\n'''
if old_maps in text:
    text = text.replace(old_maps, new_maps, 1)
elif '_merge_evidence_profile_maps(\n        ownership_auto_map' not in text:
    raise SystemExit('resumable evidence map anchor not found')
path.write_text(text, encoding="utf-8")
