"""Catalyst NLP helpers for Indonesian IDX/IHSG news filtering.

This module is tuned for Indonesian equities and macro catalysts.

Design goals:
- Reject rumor, clickbait, and retail noise.
- Prefer primary / authoritative sources.
- Classify government policy, macro, sector, and company-level catalysts.
- Keep the API compatible with the Streamlit app.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from typing import Any, Iterable

# ---------------------------------------------------------------------
# Signals tuned for IDX / IHSG
# ---------------------------------------------------------------------

NEGATIVE_MARKERS = (
    "rumor",
    "rumour",
    "isu",
    "isue",
    "kabar burung",
    "bocoran",
    "leak",
    "spekulasi",
    "speculation",
    "katanya",
    "diduga",
    "disebut-sebut",
    "diperkirakan",
    "viral",
    "trending",
    "social media",
    "media sosial",
    "twitter",
    "x.com",
    "retweet",
    "heboh",
    "unconfirmed",
    "belum dikonfirmasi",
    "tak dikonfirmasi",
    "clickbait",
    "hype",
    "gorengan",
    "saham gorengan",
    "cuan cepat",
    "pump",
    "dump",
    "scalp",
    "retail frenzy",
    "like and subscribe",
)

INDONESIA_GOVERNMENT_BODIES = (
    "pemerintah",
    "presiden",
    "dpr",
    "dpr ri",
    "dpr-ri",
    "mpr",
    "dprd",
    "menkeu",
    "kemenkeu",
    "kementerian keuangan",
    "bi",
    "bank indonesia",
    "ojk",
    "bei",
    "idx",
    "bursa efek indonesia",
    "kemenperin",
    "kementerian perindustrian",
    "esdm",
    "kementerian esdm",
    "kemendag",
    "kementerian perdagangan",
    "kemenhub",
    "kementerian perhubungan",
    "kemenkominfo",
    "kominfo",
    "kementerian investasi",
    "bkpm",
    "bumn",
    "kementerian bumn",
    "sekretariat kabinet",
    "menko",
    "mahkamah agung",
    "mk",
    "bappebti",
    "kppu",
    "bea cukai",
)

POSITIVE_POLICY_MARKERS = (
    "peraturan",
    "regulation",
    "aturan",
    "kebijakan",
    "policy",
    "surat edaran",
    "keputusan",
    "perpres",
    "pmk",
    "pp",
    "uu",
    "revisi aturan",
    "insentif",
    "subsidi",
    "tarif",
    "bea masuk",
    "bea keluar",
    "pajak",
    "cukai",
    "royalti",
    "dhe",
    "tkdn",
    "hilirisasi",
    "kuota",
    "moratorium",
    "relaksasi",
    "pengampunan",
    "stimulus",
    "pelonggaran",
    "pengetatan",
    "deregulasi",
    "standardisasi",
    "izin",
    "perizinan",
)

POSITIVE_MACRO_MARKERS = (
    "bi rate",
    "suku bunga",
    "interest rate",
    "inflasi",
    "inflation",
    "nilai tukar",
    "rupiah",
    "cadangan devisa",
    "neraca perdagangan",
    "current account",
    "defisit fiskal",
    "apbn",
    "pdb",
    "gdp",
    "growth",
    "perlambatan ekonomi",
    "resesi",
    "likuiditas",
    "lcr",
    "gwm",
    "loan growth",
    "kredit",
    "permintaan domestik",
    "ekspor",
    "impor",
    "harga komoditas",
    "commodity",
    "batubara",
    "coal",
    "nickel",
    "nikel",
    "cpo",
    "palm oil",
    "emas",
    "gold",
    "minyak",
    "oil",
    "gas",
)

POSITIVE_COMPANY_MARKERS = (
    "guidance",
    "earnings",
    "financial results",
    "results",
    "laporan keuangan",
    "laba",
    "laba bersih",
    "pendapatan",
    "revenue",
    "profit",
    "net income",
    "margin",
    "dividen",
    "dividend",
    "contract",
    "kontrak",
    "tender",
    "order",
    "pesanan",
    "acquisition",
    "akuisisi",
    "merger",
    "divestment",
    "divestasi",
    "spin-off",
    "rights issue",
    "buyback",
    "buy back",
    "capex",
    "capital expenditure",
    "debt restructuring",
    "restrukturisasi utang",
    "profit warning",
    "guidance raise",
    "guidance cut",
    "capacity expansion",
    "ekspansi kapasitas",
    "pabrik",
    "plant",
    "smelter",
    "production start",
    "commercial operation",
    "operasi komersial",
    "approval",
    "persetujuan",
    "license",
    "licence",
    "izin",
    "ipo",
    "listing",
    "offer",
    "offtake",
    "mou",
    "memorandum of understanding",
    "project",
    "proyek",
)

POSITIVE_SECTOR_MARKERS = (
    "bank",
    "banking",
    "insurance",
    "asuransi",
    "property",
    "properti",
    "telecom",
    "telekomunikasi",
    "consumer",
    "consumer goods",
    "retail",
    "mining",
    "pertambangan",
    "coal",
    "batubara",
    "nickel",
    "nikel",
    "gold",
    "emas",
    "oil",
    "gas",
    "energy",
    "energi",
    "cpo",
    "palm",
    "smelter",
    "industrial",
    "industri",
    "healthcare",
    "farmasi",
    "technology",
    "teknologi",
    "shipping",
    "logistics",
    "transport",
    "infrastructure",
    "infrastruktur",
    "data center",
    "renewable",
    "geothermal",
)

PRIMARY_SOURCE_HINTS = (
    "official",
    "exchange",
    "regulator",
    "statement",
    "filing",
    "company",
    "annual report",
    "quarterly report",
    "reuters",
    "bloomberg",
    "ap",
    "wsj",
    "ft",
    "the jakarta post",
    "bank indonesia",
    "ojk",
    "idx",
    "bei",
    "bursa efek indonesia",
    "kementerian",
    "kemendag",
    "kemenkeu",
    "esdm",
    "bkpm",
    "bumn",
)

TRUSTED_SECONDARY_SOURCE_HINTS = (
    "antara",
    "bisnis indonesia",
    "bisnis.com",
    "kontan",
    "cnbc indonesia",
    "cnn indonesia",
    "tempo",
    "kompas",
    "detik",
)

SYSTEM_PROMPT = """You are CatalystNLP, a strict institutional-grade news filter for Indonesian equities.

Task:
Decide whether a news item is a STRUCTURAL catalyst worth passing to the trading system.

Rules:
- Reject rumor, speculation, clickbait, and retail sentiment noise.
- Pass only verifiable, material, structural news.
- Prefer primary sources: company filings, exchange notices, regulator statements, central bank statements, ministry statements, audited results, or reputable wire reports quoting primary documents.
- Downweight stale reruns; recent items matter more than old echoes.
- Distinguish direct ticker relevance from broader sector or macro relevance.
- Classify macro regime changes, policy actions, sector rotation catalysts, and company-level fundamentals that materially affect earnings, cash flow, valuation, or capital allocation.
- If the story is mainly about price movement, public excitement, or social chatter, reject it.
- For Indonesian equities, prioritize Bank Indonesia, OJK, BEI/IDX, Kementerian Keuangan, ESDM, BKPM, BUMN, DPR, and other policy-related institutions.
- When uncertain, return WATCH rather than PASS.

Return only JSON with keys:
decision, category, confidence, impact_horizon, reasons, tags, summary, red_flags, source_quality, materiality
"""


@dataclass(frozen=True)
class CatalystDecision:
    decision: str
    category: str
    confidence: int
    impact_horizon: str
    reasons: list[str]
    tags: list[str]
    summary: str
    red_flags: list[str]
    source_quality: str = "unknown"
    materiality: str = "medium"
    freshness_score: int = 0
    relevance_score: int = 0
    source_tier: str = "unknown"

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False)


def build_catalyst_system_prompt() -> str:
    return SYSTEM_PROMPT


def _normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip()).lower()


def _compile_term_pattern(term: str) -> re.Pattern[str]:
    term = str(term or "").strip().lower()
    if not term:
        return re.compile(r"a^")
    escaped = re.escape(term)
    if re.fullmatch(r"[a-z0-9]{1,4}", term):
        return re.compile(rf"\b{escaped}\b", re.IGNORECASE)
    return re.compile(rf"\b{escaped}\b", re.IGNORECASE)


def _contains_term(text: str, term: str) -> bool:
    if not text or not term:
        return False
    term_l = str(term).strip().lower()
    if not term_l:
        return False

    # Short acronyms need word boundaries to avoid false positives (e.g. "bi" in "biasa").
    if re.fullmatch(r"[a-z0-9]{1,4}", term_l):
        return re.search(rf"\b{re.escape(term_l)}\b", text) is not None

    # Generic multi-word or longer terms.
    return term_l in text


def _find_terms(text: str, phrases: Iterable[str]) -> list[str]:
    hits: list[str] = []
    for phrase in phrases:
        if _contains_term(text, phrase):
            hits.append(phrase)
    return hits


def _count_hits(text: str, phrases: Iterable[str]) -> int:
    return len(_find_terms(text, phrases))


def _is_indonesia_policy_source(source: str, text: str) -> bool:
    src = _normalize_text(source)
    return any(_contains_term(src, h) for h in INDONESIA_GOVERNMENT_BODIES) or any(
        _contains_term(text, h) for h in INDONESIA_GOVERNMENT_BODIES
    )


def _source_quality(source: str, text: str) -> str:
    src = _normalize_text(source)

    if any(_contains_term(src, h) for h in PRIMARY_SOURCE_HINTS) or any(
        _contains_term(text, h) for h in PRIMARY_SOURCE_HINTS
    ):
        return "primary"

    if any(_contains_term(src, h) for h in TRUSTED_SECONDARY_SOURCE_HINTS) or any(
        _contains_term(text, h) for h in TRUSTED_SECONDARY_SOURCE_HINTS
    ):
        return "trusted_secondary"

    if src:
        return "secondary"

    return "unknown"



def _parse_news_datetime(value: Any) -> datetime | None:
    if value is None:
        return None

    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, (int, float)):
        try:
            raw = float(value)
            if raw > 1_000_000_000_000:
                raw = raw / 1000.0
            dt = datetime.fromtimestamp(raw, tz=timezone.utc)
        except Exception:
            return None
    else:
        text = str(value).strip()
        if not text:
            return None

        normalized = text.replace("Z", "+00:00")
        dt = None
        try:
            dt = datetime.fromisoformat(normalized)
        except Exception:
            pass

        if dt is None:
            for fmt in (
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M",
                "%Y-%m-%d",
                "%a, %d %b %Y %H:%M:%S %z",
                "%a, %d %b %Y %H:%M:%S GMT",
                "%d %b %Y %H:%M:%S",
                "%d %b %Y",
            ):
                try:
                    dt = datetime.strptime(text, fmt)
                    break
                except Exception:
                    continue

        if dt is None:
            return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(timezone.utc)


def _item_datetime(item: dict) -> datetime | None:
    for key in (
        "published_at",
        "publishedAt",
        "published",
        "pubDate",
        "providerPublishTime",
        "created_at",
        "createdAt",
        "date",
        "datetime",
        "timestamp",
    ):
        if key in item and item.get(key) not in (None, ""):
            dt = _parse_news_datetime(item.get(key))
            if dt is not None:
                return dt
    content = item.get("content")
    if isinstance(content, dict):
        for key in ("published_at", "publishedAt", "pubDate", "providerPublishTime", "date", "datetime", "timestamp"):
            if key in content and content.get(key) not in (None, ""):
                dt = _parse_news_datetime(content.get(key))
                if dt is not None:
                    return dt
    return None


def _freshness_bucket(published_at: Any, now: datetime | None = None) -> tuple[int, str, list[str]]:
    now_dt = now or datetime.now(timezone.utc)
    dt = _parse_news_datetime(published_at)
    if dt is None:
        return 0, "unknown", ["no_timestamp"]

    age_hours = max(0.0, (now_dt - dt).total_seconds() / 3600.0)
    if age_hours <= 6:
        return 12, "intraday", ["fresh_news"]
    if age_hours <= 24:
        return 10, "fresh", ["fresh_news"]
    if age_hours <= 72:
        return 7, "fresh_3d", ["recent_news"]
    if age_hours <= 168:
        return 3, "week_old", ["somewhat_old"]
    if age_hours <= 336:
        return -4, "stale_2w", ["stale_news"]
    if age_hours <= 720:
        return -8, "stale_month", ["very_stale_news"]
    return -12, "very_stale", ["very_stale_news"]


def _extract_related_tickers(item: dict) -> list[str]:
    collected: list[str] = []
    for key in ("relatedTickers", "related_tickers", "tickers", "symbols", "ticker"):
        value = item.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            collected.append(value)
        elif isinstance(value, dict):
            collected.extend([str(v) for v in value.values() if v])
        elif isinstance(value, Iterable):
            for v in value:
                if isinstance(v, dict):
                    collected.extend([str(x) for x in v.values() if x])
                elif v is not None:
                    collected.append(str(v))
    out: list[str] = []
    for tick in collected:
        t = re.sub(r"[^a-z0-9]+", "", str(tick).strip().lower())
        if t and t not in out:
            out.append(t)
    return out


def _normalize_signature(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", _normalize_text(text)).strip()


def _ticker_relevance(
    symbol: str,
    title_text: str,
    summary_text: str,
    source_text: str,
    item: dict | None = None,
) -> tuple[int, list[str], str]:
    sym = re.sub(r"[^a-z0-9]+", "", str(symbol or "").strip().lower())
    if not sym:
        return 0, [], "unscored"

    text = " ".join([title_text, summary_text, source_text]).strip()
    reasons: list[str] = []
    score = 0
    mode = "unscored"

    related = _extract_related_tickers(item or {}) if item else []
    if sym in related:
        score += 12
        reasons.append("related_ticker_match")
        mode = "direct"
    elif related:
        score -= 4
        reasons.append("other_ticker_focused")
        mode = "off_ticker"

    if _contains_term(text, sym):
        score += 8
        reasons.append("symbol_mentioned")
        mode = "direct" if mode != "off_ticker" else mode

    if item:
        company_name = str(item.get("longName") or item.get("shortName") or item.get("companyName") or "").strip()
        if company_name:
            company_norm = _normalize_text(company_name)
            if company_norm and any(tok in text for tok in company_norm.split() if len(tok) > 3):
                score += 3
                reasons.append("company_name_match")
                if mode == "unscored":
                    mode = "company"
        if item.get("link") and sym in _normalize_text(item.get("link")):
            score += 2
            reasons.append("link_symbol_match")

    if score == 0:
        mode = "broad" if not related else mode

    return max(-10, min(20, score)), reasons, mode


def _news_duplicate_signature(title: str, summary: str, source: str, link: str = "", published_at: Any = None) -> str:
    parts = [
        _normalize_signature(title),
        _normalize_signature(summary[:180]),
        _normalize_signature(source),
        _normalize_signature(link),
    ]
    dt = _parse_news_datetime(published_at)
    if dt is not None:
        parts.append(dt.strftime("%Y-%m-%d %H"))
    return " | ".join([p for p in parts if p])


def _news_item_payload(item: dict) -> tuple[str, str, str, Any, str]:
    title = str(
        item.get("title")
        or item.get("headline")
        or item.get("name")
        or item.get("article_title")
        or ""
    )
    summary = str(
        item.get("summary")
        or item.get("description")
        or item.get("content")
        or item.get("snippet")
        or item.get("body")
        or ""
    )
    source = str(
        item.get("source")
        or item.get("publisher")
        or item.get("provider")
        or item.get("site")
        or ""
    )
    link = str(item.get("link") or item.get("url") or "")
    published_at = item.get("published_at")
    if published_at in (None, ""):
        published_at = item.get("publishedAt") or item.get("published") or item.get("pubDate") or item.get("providerPublishTime")
    return title, summary, source, published_at, link


def _news_has_company_focus(text: str, symbol: str = "") -> bool:
    if not text:
        return False
    company_terms = (
        "pt ",
        "tbk",
        "bk",
        "perseroan",
        "emiten",
        "issuer",
        "annual report",
        "quarterly report",
        "laba",
        "pendapatan",
        "revenue",
        "earnings",
        "dividen",
        "buyback",
        "rights issue",
        "akuisisi",
        "merger",
        "kontrak",
        "tender",
        "project",
        "proyek",
        "smelter",
        "pabrik",
    )
    if any(term in text for term in company_terms):
        return True
    sym = re.sub(r"[^a-z0-9]+", "", str(symbol or "").lower())
    return bool(sym and _contains_term(text, sym))


def _duplicate_penalty(duplicate: bool) -> tuple[int, list[str], list[str]]:
    if not duplicate:
        return 0, [], []
    return -28, ["duplicate_news_item"], ["duplicate"]


def _policy_tags(text: str) -> list[str]:
    tags: list[str] = []
    if any(k in text for k in ("bi rate", "suku bunga", "inflasi", "rupiah", "cadangan devisa")):
        tags.append("macro_policy")
    if any(k in text for k in ("ojk", "bank indonesia", "bi", "bea cukai", "idx", "bei")):
        tags.append("regulator")
    if any(k in text for k in ("kemenkeu", "menkeu", "apbn", "pajak", "cukai", "subsidi")):
        tags.append("fiscal_policy")
    if any(k in text for k in ("esdm", "royalti", "nikel", "batubara", "coal", "hilirisasi", "tkdn")):
        tags.append("industrial_policy")
    if any(k in text for k in ("dpr", "uu", "revisi", "peraturan", "perpres", "keputusan")):
        tags.append("policy_change")
    return tags


def _sector_tags(text: str) -> list[str]:
    tags: list[str] = []
    mapping = {
        "banking": ("bank", "banking", "kredit", "lcr", "gwm", "bi rate", "suku bunga"),
        "property": ("properti", "property", "mortgage", "kpr"),
        "consumer": ("consumer", "consumer goods", "ritel", "retail", "subsidi"),
        "telecom": ("telekomunikasi", "telecom", "digi", "data"),
        "mining": ("mining", "pertambangan", "batubara", "coal", "nikel", "nickel", "emas", "gold"),
        "energy": ("oil", "gas", "energy", "energi", "cpo", "palm", "minyak"),
        "healthcare": ("healthcare", "farmasi", "pharma", "obat"),
        "technology": ("technology", "teknologi", "digital", "software", "platform"),
        "industrial": ("industrial", "industri", "smelter", "factory", "pabrik", "manufacturing"),
        "infrastructure": ("infrastructure", "infrastruktur", "logistics", "shipping", "transport"),
    }
    for tag, kws in mapping.items():
        if any(k in text for k in kws):
            tags.append(tag)
    return tags


def _category_profile(text: str) -> dict[str, dict[str, Any]]:
    """Collect evidence by category without letting one bucket overwrite another."""
    return {
        "policy": {
            "hits": _find_terms(text, POSITIVE_POLICY_MARKERS),
            "priority": 4,
            "horizon": "weeks",
            "base_materiality": "high",
            "reason": "policy_or_regulatory_signal",
            "tags": _policy_tags(text),
        },
        "macro": {
            "hits": _find_terms(text, POSITIVE_MACRO_MARKERS),
            "priority": 3,
            "horizon": "weeks",
            "base_materiality": "high",
            "reason": "macro_regime_relevant",
            "tags": ["macro"],
        },
        "company_structural": {
            "hits": _find_terms(text, POSITIVE_COMPANY_MARKERS),
            "priority": 2,
            "horizon": "weeks",
            "base_materiality": "high",
            "reason": "company_structural_catalyst",
            "tags": ["fundamental"],
        },
        "sector": {
            "hits": _find_terms(text, POSITIVE_SECTOR_MARKERS),
            "priority": 1,
            "horizon": "days",
            "base_materiality": "medium",
            "reason": "sector_rotation_relevant",
            "tags": ["sector"],
        },
        "noise": {
            "hits": _find_terms(text, NEGATIVE_MARKERS),
            "priority": 0,
            "horizon": "days",
            "base_materiality": "low",
            "reason": "rumor_or_noise_language",
            "tags": ["noise"],
        },
    }


def _detect_category(text: str) -> tuple[str, str, list[str], list[str], str]:
    """Return dominant category, horizon, reasons, tags, and materiality.

    Priority order for IDX:
    policy > macro > company_structural > sector > noise
    """
    reasons: list[str] = []
    tags: list[str] = []
    materiality = "medium"
    impact_horizon = "days"
    category = "unknown"

    profile = _category_profile(text)
    signal_counts = {cat: len(info["hits"]) for cat, info in profile.items()}

    # Choose the highest-priority category that has at least one real hit.
    for cat in ("policy", "macro", "company_structural", "sector", "noise"):
        if signal_counts.get(cat, 0) > 0:
            category = cat
            impact_horizon = str(profile[cat]["horizon"])
            materiality = str(profile[cat]["base_materiality"])
            reasons.append(str(profile[cat]["reason"]))
            tags.extend(profile[cat]["tags"])
            break

    # Add contextual evidence from lower-priority buckets without changing category.
    if category != "unknown":
        if category != "policy" and signal_counts["policy"] > 0:
            reasons.append("secondary_policy_context")
            tags.extend(profile["policy"]["tags"])
        if category not in ("policy", "macro") and signal_counts["macro"] > 0:
            reasons.append("secondary_macro_context")
            tags.extend(profile["macro"]["tags"])
        if category not in ("policy", "macro", "company_structural") and signal_counts["company_structural"] > 0:
            reasons.append("secondary_company_context")
            tags.extend(profile["company_structural"]["tags"])
        if category == "noise":
            materiality = "low"

    # If no structural bucket hit but institutional references exist, elevate to policy.
    if category == "unknown" and any(
        _contains_term(text, h) for h in ("bank indonesia", "ojk", "idx", "bei", "kemenkeu", "esdm", "bkpm", "dpr")
    ):
        category = "policy"
        impact_horizon = "weeks"
        materiality = "high"
        reasons.append("institutional_reference")
        tags.extend(_policy_tags(text))

    # Deduplicate while preserving order.
    tags = list(dict.fromkeys(tags))
    reasons = list(dict.fromkeys(reasons))

    return category, impact_horizon, reasons, tags, materiality


def _decision_thresholds(score: int, category: str, red_flags: list[str]) -> str:
    if category == "noise" and score < 60:
        return "REJECT"
    if "rumor_or_noise_language" in red_flags and score < 70:
        return "REJECT"
    if score >= 82:
        return "PASS"
    if score >= 60:
        return "WATCH"
    return "REJECT"


def score_news_item(
    title: str,
    summary: str = "",
    source: str = "",
    published_at: Any = None,
    symbol: str = "",
    url: str = "",
    item: dict | None = None,
    duplicate: bool = False,
) -> CatalystDecision:
    title_text = _normalize_text(title)
    summary_text = _normalize_text(summary)
    source_text = _normalize_text(source)
    url_text = _normalize_text(url)
    text = " ".join([title_text, summary_text, source_text, url_text]).strip()

    reasons: list[str] = []
    red_flags: list[str] = []
    tags: list[str] = []

    # Start neutral, then move up/down.
    score = 45
    category = "unknown"
    impact_horizon = "days"
    source_quality = _source_quality(source, text)
    source_tier = source_quality
    materiality = "medium"
    freshness_score = 0
    freshness_label = "unknown"
    relevance_score = 0
    relevance_mode = "unscored"

    if not text:
        return CatalystDecision(
            decision="REJECT",
            category="unknown",
            confidence=0,
            impact_horizon="days",
            reasons=["empty_news_item"],
            tags=["news"],
            summary="No title",
            red_flags=["empty_input"],
            source_quality="unknown",
            materiality="low",
            freshness_score=0,
            relevance_score=0,
            source_tier="unknown",
        )

    # Noise / rumor penalty.
    negative_hits = _find_terms(text, NEGATIVE_MARKERS)
    if negative_hits:
        score -= 30
        red_flags.append("rumor_or_noise_language")
        category = "noise"
        source_quality = "rumor"
        source_tier = "rumor"
        materiality = "low"
        reasons.append("negative_language_detected")
        tags.append("noise")

    # Very short / vague headlines are less trustworthy.
    if len(title_text) < 24 and len(summary_text) < 40:
        score -= 6
        red_flags.append("thin_context")

    cat, horizon, cat_reasons, cat_tags, cat_materiality = _detect_category(text)
    if cat != "unknown":
        category = cat
        impact_horizon = horizon
        reasons.extend(cat_reasons)
        tags.extend(cat_tags)
        materiality = cat_materiality

        if cat == "policy":
            score += 28
        elif cat == "macro":
            score += 22
        elif cat == "company_structural":
            score += 20
        elif cat == "sector":
            score += 14
        elif cat == "noise":
            score -= 8

    # Primary / trusted source boost, but keep hierarchy explicit.
    if source_quality == "primary":
        score += 12
        tags.append("primary_source")
    elif source_quality == "trusted_secondary":
        score += 6
        tags.append("trusted_secondary_source")
    elif source_quality == "secondary":
        score += 2

    # Strong government / regulator references are especially valuable for IDX.
    if _is_indonesia_policy_source(source, text):
        score += 8
        tags.append("id_policy")

    # Freshness matters for catalysts.
    freshness_score, freshness_label, freshness_notes = _freshness_bucket(published_at)
    if freshness_score:
        score += freshness_score
        tags.append(f"freshness_{freshness_label}")
        reasons.extend(freshness_notes)
    else:
        red_flags.append("no_timestamp")

    # Direct ticker relevance.
    relevance_score, relevance_reasons, relevance_mode = _ticker_relevance(
        symbol=symbol,
        title_text=title_text,
        summary_text=summary_text,
        source_text=source_text,
        item=item,
    )
    if relevance_score:
        score += relevance_score
        reasons.extend(relevance_reasons)
        if relevance_mode == "direct":
            tags.append("direct_ticker_match")
        elif relevance_mode == "company":
            tags.append("company_match")
        elif relevance_mode == "off_ticker":
            red_flags.append("off_ticker_focus")
    elif symbol:
        if _news_has_company_focus(text, symbol=symbol):
            score -= 3
            red_flags.append("weak_ticker_link")

    # Duplicate suppression: keep alignment, but downgrade repeats aggressively.
    dup_penalty, dup_reasons, dup_tags = _duplicate_penalty(duplicate)
    if dup_penalty:
        score += dup_penalty
        red_flags.extend(dup_reasons)
        tags.extend(dup_tags)
        reasons.append("duplicate_suppressed")

    # Positive wording that usually indicates structural change.
    structural_hits = _count_hits(
        text,
        (
            "audited",
            "results",
            "earnings",
            "guidance",
            "buyback",
            "rights issue",
            "contract",
            "tender",
            "approval",
            "license",
            "licence",
            "capex",
            "capacity expansion",
            "operasi komersial",
            "ekspansi kapasitas",
            "investasi",
            "proyek",
        ),
    )
    if structural_hits:
        score += 6
        tags.append("corporate_event")

    # Retail hype / analyst chatter / price-target noise.
    if _count_hits(
        text,
        (
            "social media",
            "viral",
            "trending",
            "analyst says",
            "price target",
            "target price",
            "unconfirmed",
            "rumored",
            "rumour",
        ),
    ):
        score -= 20
        red_flags.append("retail_hype")
        tags.append("noise")

    # Price action only news should not be treated as structural.
    if _count_hits(text, ("shares jump", "stock rallies", "stock falls", "price surges", "price tumbles", "sentimen", "market reacts")):
        score -= 10
        red_flags.append("price_action_only")

    # If the item mentions a concrete policy instrument, increase conviction.
    if _count_hits(text, ("tarif", "subsidi", "pajak", "cukai", "royalti", "dhe", "tkdn", "insentif", "kuota", "relaksasi", "pengetatan")):
        score += 6
        tags.append("policy_instrument")

    # Higher-quality local official context.
    if _count_hits(text, ("bank indonesia", "ojk", "idx", "bei", "kemenkeu", "esdm", "bkpm", "bumn", "dpr")):
        score += 4
        tags.append("id_institution")

    # A stale catalyst should not get PASS just because it is structurally relevant.
    if freshness_label in {"stale_month", "very_stale"} and score >= 82:
        score -= 8
        red_flags.append("stale_catalyst")
    if freshness_label in {"stale_month", "very_stale"} and category in {"company_structural", "policy"}:
        materiality = "medium"

    # Ensure tags and reasons are populated.
    if not reasons:
        reasons.append("insufficient_structural_signal")
    if not tags:
        tags.append("news")

    # Cap score.
    score = int(max(0, min(100, score)))

    decision = _decision_thresholds(score, category, red_flags)

    return CatalystDecision(
        decision=decision,
        category=category,
        confidence=score,
        impact_horizon=impact_horizon,
        reasons=reasons[:6],
        tags=list(dict.fromkeys(tags))[:10],
        summary=title.strip()[:180] if title else "No title",
        red_flags=red_flags[:6],
        source_quality=source_quality,
        materiality=materiality,
        freshness_score=int(max(-20, min(20, freshness_score))),
        relevance_score=int(max(-10, min(20, relevance_score))),
        source_tier=source_tier,
    )


def filter_news_items(items: Iterable[dict], symbol: str = "") -> list[CatalystDecision]:
    out: list[CatalystDecision] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue

        title, summary, source, published_at, link = _news_item_payload(item)
        duplicate_signature = _news_duplicate_signature(title, summary, source, link=link, published_at=published_at)
        duplicate = duplicate_signature in seen
        seen.add(duplicate_signature)

        out.append(
            score_news_item(
                title=title,
                summary=summary,
                source=source,
                published_at=published_at,
                symbol=symbol,
                url=link,
                item=item,
                duplicate=duplicate,
            )
        )
    return out


def _strip_code_fences(payload: str) -> str:
    text = str(payload or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_catalyst_response(payload: str) -> CatalystDecision:
    try:
        data = json.loads(_strip_code_fences(payload))
    except Exception:
        return CatalystDecision(
            decision="WATCH",
            category="unknown",
            confidence=50,
            impact_horizon="days",
            reasons=["invalid_json_response"],
            tags=["news"],
            summary="Parser fallback",
            red_flags=["parse_error"],
            source_quality="unknown",
            materiality="medium",
            freshness_score=0,
            relevance_score=0,
            source_tier="unknown",
        )

    reasons = data.get("reasons", [])
    tags = data.get("tags", [])
    red_flags = data.get("red_flags", [])

    if not isinstance(reasons, list):
        reasons = [str(reasons)]
    if not isinstance(tags, list):
        tags = [str(tags)]
    if not isinstance(red_flags, list):
        red_flags = [str(red_flags)]

    return CatalystDecision(
        decision=str(data.get("decision", "WATCH")),
        category=str(data.get("category", "unknown")),
        confidence=int(data.get("confidence", 50)),
        impact_horizon=str(data.get("impact_horizon", "days")),
        reasons=[str(x) for x in reasons],
        tags=[str(x) for x in tags],
        summary=str(data.get("summary", "")),
        red_flags=[str(x) for x in red_flags],
        source_quality=str(data.get("source_quality", "unknown")),
        materiality=str(data.get("materiality", "medium")),
        freshness_score=int(data.get("freshness_score", 0) or 0),
        relevance_score=int(data.get("relevance_score", 0) or 0),
        source_tier=str(data.get("source_tier", data.get("source_quality", "unknown"))),
    )
