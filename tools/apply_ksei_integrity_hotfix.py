from pathlib import Path

path = Path("autonomous_enrichment.py")
text = path.read_text(encoding="utf-8")

old_profile = '''    profile["registered_amount"] = _number(_label_value(text, "Current Amount"))
    profile["total_shares"] = _number(_label_value(text, "Number of Securities"))
    profile["local_pct"] = _number(_label_value(text, "Local Percentage"))
    profile["foreign_pct"] = _number(_label_value(text, "Foreign Percentage"))
    scripless_match = re.search(r"As of\\s+[^\\n]+\\s*\\n\\s*([\\d.,]+)%\\s+Scripless", text, flags=re.IGNORECASE)
'''
new_profile = '''    profile["registered_amount"] = _number(_label_value(text, "Current Amount"))
    profile["total_shares"] = _number(_label_value(text, "Number of Securities"))
    profile["local_pct"] = _number(_label_value(text, "Local Percentage"))
    profile["foreign_pct"] = _number(_label_value(text, "Foreign Percentage"))

    # A non-empty KSEI HTML response is not enough to prove that /lc/{ticker}
    # resolved to the requested issuer. Generic shells can expose headings such
    # as "ISIN Code", zero shares and UNKNOWN status.
    placeholder_names = {"", "isin code", "security name", "issuer", "undefined", "null", "none"}
    company_identity = _clean_text(profile.get("company_name")).lower()
    security_status = _clean_text(profile.get("security_status")).upper()
    total_shares = _finite(profile.get("total_shares"), np.nan)
    profile_verified = bool(
        text
        and company_identity not in placeholder_names
        and np.isfinite(total_shares)
        and total_shares > 0
        and security_status
        and not security_status.startswith("UNKNOWN")
    )
    profile["ksei_source_verified"] = profile_verified
    if not profile_verified:
        profile["profile_integrity_state"] = "PLACEHOLDER_OR_UNRESOLVED_PROFILE"
        if company_identity in placeholder_names:
            profile["company_name"] = ""
        if not np.isfinite(total_shares) or total_shares <= 0:
            profile["total_shares"] = np.nan
        registered = _finite(profile.get("registered_amount"), np.nan)
        if not np.isfinite(registered) or registered <= 0:
            profile["registered_amount"] = np.nan

    scripless_match = re.search(r"As of\\s+[^\\n]+\\s*\\n\\s*([\\d.,]+)%\\s+Scripless", text, flags=re.IGNORECASE)
'''
if old_profile not in text:
    if "PLACEHOLDER_OR_UNRESOLVED_PROFILE" in text:
        raise SystemExit(0)
    raise SystemExit("KSEI profile patch anchor not found")
text = text.replace(old_profile, new_profile, 1)

old_ok = '''            profile, actions = parse_ksei_profile_html(symbol, response.text, source_url=url)
            ok = bool(profile.get("company_name") or profile.get("sector") or np.isfinite(_finite(profile.get("total_shares"))))
'''
new_ok = '''            profile, actions = parse_ksei_profile_html(symbol, response.text, source_url=url)
            ok = bool(profile.get("ksei_source_verified", False))
'''
if old_ok not in text:
    raise SystemExit("KSEI fetch patch anchor not found")
text = text.replace(old_ok, new_ok, 1)
path.write_text(text, encoding="utf-8")
