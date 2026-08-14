from pathlib import Path

path = Path("resumable_scan.py")
text = path.read_text(encoding="utf-8")

# 1) Numeric-safe coverage parsing. Apply only when legacy _finite parsing remains.
legacy_numeric = '''        base_cov = _finite(base.get(coverage_key), 0.0)\n        direct_cov = _finite(direct.get(coverage_key), 0.0)\n        merged[coverage_key] = round(max(base_cov, direct_cov), 1)\n'''
numeric_safe = '''        base_cov = pd.to_numeric(pd.Series([base.get(coverage_key)]), errors="coerce").iloc[0]\n        direct_cov = pd.to_numeric(pd.Series([direct.get(coverage_key)]), errors="coerce").iloc[0]\n        base_cov = float(base_cov) if np.isfinite(base_cov) else 0.0\n        direct_cov = float(direct_cov) if np.isfinite(direct_cov) else 0.0\n        merged[coverage_key] = round(max(base_cov, direct_cov), 1)\n'''
if legacy_numeric in text:
    text = text.replace(legacy_numeric, numeric_safe, 1)

# 2) Add coverage_mode only once. If it already exists, do not touch its body.
if 'coverage_mode: str = "max_overlap"' not in text:
    old_sig = '''def _merge_evidence_profile_maps(\n    base_map: Mapping[str, Mapping[str, Any]] | None,\n    direct_map: Mapping[str, Mapping[str, Any]] | None,\n    *,\n    coverage_key: str,\n    provenance_key: str,\n    hard_block_key: str | None = None,\n    reason_key: str | None = None,\n) -> dict[str, dict[str, Any]]:\n'''
    new_sig = '''def _merge_evidence_profile_maps(\n    base_map: Mapping[str, Mapping[str, Any]] | None,\n    direct_map: Mapping[str, Mapping[str, Any]] | None,\n    *,\n    coverage_key: str,\n    provenance_key: str,\n    hard_block_key: str | None = None,\n    reason_key: str | None = None,\n    coverage_mode: str = "max_overlap",\n) -> dict[str, dict[str, Any]]:\n'''
    if old_sig not in text:
        raise SystemExit('merge signature anchor not found')
    text = text.replace(old_sig, new_sig, 1)

    old_cov = '        merged[coverage_key] = round(max(base_cov, direct_cov), 1)\n'
    new_cov = '''        if str(coverage_mode).strip().lower() == "union_disjoint":\n            merged[coverage_key] = round(min(100.0, base_cov + direct_cov), 1)\n        else:\n            merged[coverage_key] = round(max(base_cov, direct_cov), 1)\n'''
    if old_cov not in text:
        raise SystemExit('coverage merge mode anchor not found')
    text = text.replace(old_cov, new_cov, 1)

# 3) Request disjoint union only for ownership, once.
if 'coverage_mode="union_disjoint"' not in text:
    old_call = '''        ownership_auto_map, parse_ownership(raw_ownership),\n        coverage_key="ownership_coverage_pct", provenance_key="ownership_provenance_state",\n    )\n'''
    new_call = '''        ownership_auto_map, parse_ownership(raw_ownership),\n        coverage_key="ownership_coverage_pct", provenance_key="ownership_provenance_state",\n        coverage_mode="union_disjoint",\n    )\n'''
    if old_call not in text:
        raise SystemExit('ownership merge call anchor not found')
    text = text.replace(old_call, new_call, 1)

path.write_text(text, encoding="utf-8")
