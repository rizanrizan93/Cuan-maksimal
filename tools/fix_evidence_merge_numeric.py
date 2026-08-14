from pathlib import Path

path = Path("resumable_scan.py")
text = path.read_text(encoding="utf-8")
old = '''        base_cov = _finite(base.get(coverage_key), 0.0)\n        direct_cov = _finite(direct.get(coverage_key), 0.0)\n        merged[coverage_key] = round(max(base_cov, direct_cov), 1)\n'''
new = '''        base_cov = pd.to_numeric(pd.Series([base.get(coverage_key)]), errors="coerce").iloc[0]\n        direct_cov = pd.to_numeric(pd.Series([direct.get(coverage_key)]), errors="coerce").iloc[0]\n        base_cov = float(base_cov) if np.isfinite(base_cov) else 0.0\n        direct_cov = float(direct_cov) if np.isfinite(direct_cov) else 0.0\n        merged[coverage_key] = round(max(base_cov, direct_cov), 1)\n'''
if old in text:
    text = text.replace(old, new, 1)
elif 'base_cov = pd.to_numeric(pd.Series([base.get(coverage_key)])' not in text:
    raise SystemExit('evidence merge numeric anchor not found')
path.write_text(text, encoding="utf-8")
