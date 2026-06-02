"""Catalyst NLP helpers for structural news filtering.

This module is intentionally conservative: it is designed to reject rumor,
clickbait, and retail hype while passing only verifiable catalysts with
structural market relevance.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterable

NEGATIVE_MARKERS = (
    "rumor",
    "rumour",
    "speculation",
    "leak",
    "unnamed source",
    "anonymous source",
    "clickbait",
    "viral",
    "hype",
    "trending",
    "social media",
    "twitter",
    "x.com",
)

POSITIVE_MARKERS = (
    "central bank",
    "bi rate",
    "interest rate",
    "inflation",
    "fiscal",
    "tariff",
    "subsidy",
    "regulation",
    "guidance",
    "earnings",
    "financial results",
    "contract",
    "tender",
    "acquisition",
    "divestment",
    "capacity expansion",
    "factory",
    "licence",
    "approval",
)

SYSTEM_PROMPT = """You are CatalystNLP, a strict institutional-grade news filter for Indonesian equities.

Task:
Decide whether a news item is a STRUCTURAL catalyst worth passing to the trading system.

Rules:
- Reject rumor, speculation, clickbait, and retail sentiment noise.
- Pass only verifiable, material, structural news.
- Prefer primary sources: company filings, exchange notices, regulator statements, central bank statements, ministry statements, audited results, or reputable wire reports quoting primary documents.
- Classify macro regime changes, policy actions, sector rotation catalysts, and company-level fundamentals that materially affect earnings, cash flow, valuation, or capital allocation.
- If the story is mainly about price movement, public excitement, or social chatter, reject it.
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

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False)


def build_catalyst_system_prompt() -> str:
    return SYSTEM_PROMPT


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip()).lower()


def score_news_item(title: str, summary: str = "", source: str = "") -> CatalystDecision:
    text = _normalize_text(" ".join([title, summary, source]))
    reasons: list[str] = []
    red_flags: list[str] = []
    tags: list[str] = []

    score = 50
    category = "unknown"
    impact_horizon = "days"
    source_quality = "unknown"
    materiality = "medium"

    if any(m in text for m in NEGATIVE_MARKERS):
        score -= 30
        red_flags.append("rumor_or_noise_language")
        category = "noise"
        source_quality = "rumor"

    if any(m in text for m in POSITIVE_MARKERS):
        score += 20
        tags.append("structural")
        materiality = "high"

    if any(k in text for k in ("central bank", "bi rate", "inflation", "fx", "exchange rate", "tariff", "tax", "subsidy")):
        category = "macro"
        impact_horizon = "weeks"
        score += 20
        reasons.append("macro_regime_relevant")
        tags.append("macro")

    if any(k in text for k in ("earnings", "revenue", "guidance", "contract", "tender", "factory", "capacity", "approval", "license", "licence", "acquisition", "divestment")):
        category = "company_structural"
        impact_horizon = "weeks"
        score += 15
        reasons.append("company_structural_catalyst")
        tags.append("fundamental")

    if any(k in text for k in ("social media", "trending", "viral", "analyst says", "price target", "rumor", "unconfirmed")):
        score -= 20
        red_flags.append("retail_hype")

    if "official" in text or "exchange" in text or "regulator" in text or "statement" in text:
        score += 10
        source_quality = "primary"

    score = int(max(0, min(100, score)))

    if score >= 75:
        decision = "PASS"
    elif score >= 50:
        decision = "WATCH"
    else:
        decision = "REJECT"

    if not reasons:
        reasons.append("insufficient_structural_signal")
    if not tags:
        tags.append("news")

    return CatalystDecision(
        decision=decision,
        category=category,
        confidence=score,
        impact_horizon=impact_horizon,
        reasons=reasons[:4],
        tags=tags[:5],
        summary=title.strip()[:180] if title else "No title",
        red_flags=red_flags[:4],
        source_quality=source_quality,
        materiality=materiality,
    )


def filter_news_items(items: Iterable[dict]) -> list[CatalystDecision]:
    out: list[CatalystDecision] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        out.append(
            score_news_item(
                title=str(item.get("title", "")),
                summary=str(item.get("summary", item.get("description", ""))),
                source=str(item.get("source", "")),
            )
        )
    return out


def parse_catalyst_response(payload: str) -> CatalystDecision:
    data = json.loads(payload)
    return CatalystDecision(
        decision=str(data.get("decision", "WATCH")),
        category=str(data.get("category", "unknown")),
        confidence=int(data.get("confidence", 50)),
        impact_horizon=str(data.get("impact_horizon", "days")),
        reasons=list(data.get("reasons", [])),
        tags=list(data.get("tags", [])),
        summary=str(data.get("summary", "")),
        red_flags=list(data.get("red_flags", [])),
        source_quality=str(data.get("source_quality", "unknown")),
        materiality=str(data.get("materiality", "medium")),
    )
