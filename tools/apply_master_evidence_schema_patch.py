from pathlib import Path

path = Path("persistence.py")
text = path.read_text(encoding="utf-8")

text = text.replace('DATABASE_SCHEMA_VERSION = "emir_autonomous_schema_v8"', 'DATABASE_SCHEMA_VERSION = "emir_autonomous_schema_v9"')
anchor = '        "cak_direct_evidence": "evidence_id",\n'
insert = anchor + '        "cak_persistent_direct_evidence": "evidence_key",\n'
if '"cak_persistent_direct_evidence": "evidence_key"' not in text:
    if anchor not in text:
        raise SystemExit("database preflight anchor not found")
    text = text.replace(anchor, insert, 1)
text = text.replace('HEALTHY_EMIR_DATABASE_V8', 'HEALTHY_EMIR_DATABASE_V9')
text = text.replace('DATABASE_NOT_READY_V8', 'DATABASE_NOT_READY_V9')
path.write_text(text, encoding="utf-8")
