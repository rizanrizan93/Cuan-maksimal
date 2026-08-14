from pathlib import Path

path = Path("resumable_scan.py")
text = path.read_text(encoding="utf-8")

# Numeric-safe coverage parsing for the non-destructive evidence merge.
old = '''        base_cov = _finite(base.get(coverage_key), 0.0)\n        direct_cov = _finite(direct.get(coverage_key), 0.0)\n        merged[coverage_key] = round(max(base_cov, direct_cov), 1)\n'''
new = '''        base_cov = pd.to_numeric(pd.Series([base.get(coverage_key)]), errors="coerce").iloc[0]\n        direct_cov = pd.to_numeric(pd.Series([direct.get(coverage_key)]), errors="coerce").iloc[0]\n        base_cov = float(base_cov) if np.isfinite(base_cov) else 0.0\n        direct_cov = float(direct_cov) if np.isfinite(direct_cov) else 0.0\n        merged[coverage_key] = round(max(base_cov, direct_cov), 1)\n'''
if old in text:
    text = text.replace(old, new, 1)
elif 'base_cov = pd.to_numeric(pd.Series([base.get(coverage_key)])' not in text:
    raise SystemExit('evidence merge numeric anchor not found')

# KSEI registration fields are a deliberately partial 45%-max ownership proxy,
# while direct free-float/alignment/concentration rows represent different evidence
# dimensions. Allow a disjoint union only when explicitly requested; integrity stays
# conservative max-overlap because its direct and autonomous fields can overlap.
old_sig = '''def _merge_evidence_profile_maps(\n    base_map: Mapping[str, Mapping[str, Any]] | None,\n    direct_map: Mapping[str, Mapping[str, Any]] | None,\n    *,\n    coverage_key: str,\n    provenance_key: str,\n    hard_block_key: str | None = None,\n    reason_key: str | None = None,\n) -> dict[str, dict[str, Any]]:\n'''
new_sig = '''def _merge_evidence_profile_maps(\n    base_map: Mapping[str, Mapping[str, Any]] | None,\n    direct_map: Mapping[str, Mapping[str, Any]] | None,\n    *,\n    coverage_key: str,\n    provenance_key: str,\n    hard_block_key: str | None = None,\n    reason_key: str | None = None,\n    coverage_mode: str = "max_overlap",\n) -> dict[str, dict[str, Any]]:\n'''
if old_sig in text:
    text = text.replace(old_sig, new_sig, 1)
elif 'coverage_mode: str = "max_overlap"' not in text:
    raise SystemExit('merge signature anchor not found')

old_cov = '''        merged[coverage_key] = round(max(base_cov, direct_cov), 1)\n'''
new_cov = '''        if str(coverage_mode).strip().lower() == "union_disjoint":\n            merged[coverage_key] = round(min(100.0, base_cov + direct_cov), 1)\n        else:\n            merged[coverage_key] = round(max(base_cov, direct_cov), 1)\n'''
if old_cov in text:
    text = text.replace(old_cov, new_cov, 1)
elif 'union_disjoint' not in text:
    raise SystemExit('coverage merge mode anchor not found')

old_call = '''        ownership_auto_map, parse_ownership(raw_ownership),\n        coverage_key="ownership_coverage_pct", provenance_key="ownership_provenance_state",\n    )\n'''
new_call = '''        ownership_auto_map, parse_ownership(raw_ownership),\n        coverage_key="ownership_coverage_pct", provenance_key="ownership_provenance_state",\n        coverage_mode="union_disjoint",\n    )\n'''
if old_call in text:
    text = text.replace(old_call, new_call, 1)
elif 'coverage_mode="union_disjoint"' not in text:
    raise SystemExit('ownership merge call anchor not found')

path.write_text(text, encoding="utf-8")
