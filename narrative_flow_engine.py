from __future__ import annotations

from typing import Any, Mapping
import hashlib
import math
import re
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from release_contract import SCANNER_RELEASE_VERSION

from data_providers import normalize_ticker


ENGINE_VERSION = SCANNER_RELEASE_VERSION
MARKET_FEATURE_VERSION = "1.9.14-market-features-v1"
FRAMEWORK_DISCLAIMER = "PUBLIC_CLEAN_ROOM_RECONSTRUCTION_NOT_AFFILIATED_NOT_PROPRIETARY"

# The registry separates what Emir has stated publicly from our own quantitative proxy.
# None of the numeric weights below are claimed to be an official CAK formula.
PUBLIC_FORMULA_REGISTRY: tuple[dict[str, str], ...] = (
    {
        "formula_id": "EP01_THESIS_BEFORE_ENTRY",
        "provenance_class": "EXPLICIT_PUBLIC",
        "public_basis": "Map scenarios and build a thesis before entry; process and disciplined decisions matter more than signals.",
        "scanner_implementation": "Hard gate: no production action without deep review, thesis evidence, invalidation and next-proof fields.",
    },
    {
        "formula_id": "EP02_THREE_MARKET_STRUCTURES",
        "provenance_class": "EXPLICIT_PUBLIC",
        "public_basis": "Momentum is approached through reversal, continuation, or sideways structure.",
        "scanner_implementation": "Compute reversal_score, continuation_score and sideways_quality_score; choose the dominant structure only when its evidence is sufficient.",
    },
    {
        "formula_id": "EP03_REVERSAL",
        "provenance_class": "PUBLIC_SYNTHESIS",
        "public_basis": "For reversal, inspect seller exhaustion, cover-buy area and change from downtrend to uptrend.",
        "scanner_implementation": "0.30 seller_exhaustion + 0.25 absorption + 0.25 structure_change + 0.20 failed_breakdown_reclaim.",
    },
    {
        "formula_id": "EP04_CONTINUATION",
        "provenance_class": "PUBLIC_SYNTHESIS",
        "public_basis": "Continuation requires higher highs, break of the last high, continuation flow and a story with runway.",
        "scanner_implementation": "0.30 higher_high + 0.20 higher_low + 0.25 flow_persistence + 0.25 story_runway.",
    },
    {
        "formula_id": "EP05_SIDEWAYS",
        "provenance_class": "PUBLIC_SYNTHESIS",
        "public_basis": "In sideways markets, identify strong demand/supply and watch fakeouts; distinguish healthy base from distribution.",
        "scanner_implementation": "0.30 range_compression + 0.25 absorption + 0.25 fakeout_reclaim + 0.20 non_distribution.",
    },
    {
        "formula_id": "EP06_BROKER_INVENTORY",
        "provenance_class": "EXPLICIT_PUBLIC",
        "public_basis": "Read broker summary, who accumulated from the bottom, average inventory, shifts from distribution to collection, and levels large interests may defend.",
        "scanner_implementation": "Verified multi-period broker records form inventory_persistence, defended_level, jumbo_crossing and fund_like_flow scores. Broker codes never identify beneficial owners.",
    },
    {
        "formula_id": "EP07_RETAIL_EXIT_SMART_MONEY_ENTRY",
        "provenance_class": "EXPLICIT_PUBLIC",
        "public_basis": "Prefer situations where retail exits while smart money accumulates; avoid retail-to-retail cannibalisation.",
        "scanner_implementation": "Verified participant labels produce retail_exit_score and retail_cannibalisation_risk; missing labels remain missing.",
    },
    {
        "formula_id": "EP08_BID_OFFER_TRIGGER",
        "provenance_class": "EXPLICIT_PUBLIC",
        "public_basis": "Use bid-offer for precise entry/exit; a thick small-lot offer at a psychological level that is broken quickly can be a deciding trigger.",
        "scanner_implementation": "Manual/direct order-book evidence computes retail_offer_stack and offer_absorption_speed. Without verified order-book evidence, scanner can approve a thesis but not a precise trigger.",
    },
    {
        "formula_id": "EP09_TOP_DOWN_NARRATIVE",
        "provenance_class": "EXPLICIT_PUBLIC",
        "public_basis": "Find strong secular trends or catalysts top-down, translate them into industry fundamentals, then into share-price implications.",
        "scanner_implementation": "top_down_catalyst + industry_translation + financial_conversion + issuer_alignment; narrative must show a revenue/margin/earnings/cash-flow/rerating bridge.",
    },
    {
        "formula_id": "EP10_NARRATIVE_LIFECYCLE",
        "provenance_class": "EXPLICIT_PUBLIC",
        "public_basis": "Narrative progresses through accumulation, hype and distribution; enter before euphoria and exit before the story is over.",
        "scanner_implementation": "Lifecycle states distinguish silent accumulation, flow-leading-story, convergence, expansion, retail euphoria and distribution.",
    },
    {
        "formula_id": "EP11_SECTOR_ROTATION_RRG",
        "provenance_class": "PUBLIC_SYNTHESIS",
        "public_basis": "Use relative sector strength and momentum quadrants while retaining liquidity as a stock-selection constraint.",
        "scanner_implementation": "RRG_PROXY: preserve absolute sector RS versus IHSG, then centre RS/momentum across the active sector universe so rotation quadrants remain discriminative; this is not official JdK RRG.",
    },
    {
        "formula_id": "EP12_OWNERSHIP_FREE_FLOAT",
        "provenance_class": "PUBLIC_SYNTHESIS",
        "public_basis": "Map actual holders, affiliations, buyback inventory and effective free float rather than relying on headline public float.",
        "scanner_implementation": "effective_free_float = reported_free_float - affiliated_public_holdings; verified ownership evidence controls coverage and passive-flow risk.",
    },
    {
        "formula_id": "EP13_REACT_NOT_PREDICT",
        "provenance_class": "EXPLICIT_PUBLIC",
        "public_basis": "React to market data rather than acting as a forecaster; focus stock by stock and change course when the thesis is wrong.",
        "scanner_implementation": "No deterministic price prediction. Output is scenario, trigger, invalidation, next proof and action state.",
    },
    {
        "formula_id": "EP14_FAST_CUT_AND_PRE_BUY_RISK",
        "provenance_class": "PUBLIC_SYNTHESIS",
        "public_basis": "Risk management starts with selective stock picking and the willingness to cut quickly; a public discussion used a 5% maximum-loss example.",
        "scanner_implementation": "Long stop is the tighter of structure invalidation and 5% below entry midpoint. This is a conservative public-discussion proxy, not an official universal rule.",
    },
    {
        "formula_id": "EP15_TRIM_OVERSHOOT",
        "provenance_class": "PUBLIC_SYNTHESIS",
        "public_basis": "Selling is harder than buying; trim when price overshoots and use broker/inventory evidence to optimise exits.",
        "scanner_implementation": "trim_state activates on extreme extension/crowding or verified broker distribution; it is guidance, not an automated sell order.",
    },
    {
        "formula_id": "EP16_EMPIRICAL_CONVICTION_WEIGHTS",
        "provenance_class": "EMPIRICAL_PROXY",
        "public_basis": "Public materials describe the ingredients and sequence, but do not publish a complete official numeric weighting formula.",
        "scanner_implementation": "Independent fixed-denominator weights combine flow/inventory, structure, story runway, context, alignment/ownership, direct order-book evidence, trend and liquidity/effective float. Must be walk-forward tested.",
    },
    {
        "formula_id": "EP17_DIRECT_EVIDENCE_BOUNDARY",
        "provenance_class": "MANUAL_EVIDENCE_REQUIRED",
        "public_basis": "Broker inventory, actual holder relationships, effective free float and live bid-offer observations cannot be inferred reliably from OHLCV or broker code alone.",
        "scanner_implementation": "Production gates require verified uploaded evidence; missing or unverified records remain missing and never default to a neutral score.",
    },
    {
        "formula_id": "EP18_POINT_IN_TIME_EVIDENCE",
        "provenance_class": "EMPIRICAL_PROXY",
        "public_basis": "Narrative and flow decisions are highly path dependent; the public framework does not provide a reproducible point-in-time data contract.",
        "scanner_implementation": "Every production decision records as-of date, source freshness, source independence, direct-evidence state and database readback. Future information is never backfilled into an earlier decision.",
    },
    {
        "formula_id": "EP19_IDX_HSC_FREE_FLOAT_OVERLAY",
        "provenance_class": "EMPIRICAL_PROXY",
        "public_basis": "Indonesia now publishes ownership above 1%, HSC indicators and more granular investor classifications; these are material to liquidity and manipulation risk.",
        "scanner_implementation": "Verified HSC, effective free float, affiliation and ownership freshness form an IDX integrity gate. HSC is a conservative production block, not a bullish scarcity signal.",
    },
    {
        "formula_id": "EP20_IDX_BOARD_REGULATORY_GATE",
        "provenance_class": "EMPIRICAL_PROXY",
        "public_basis": "Special monitoring, full-call-auction, suspension, UMA and sanctions can dominate narrative/flow signals in IDX microcaps.",
        "scanner_implementation": "Direct regulatory evidence is required for production. Suspension, special monitoring/FCA or serious sanctions block entry; UMA is a caution flag.",
    },
    {
        "formula_id": "EP21_CORPORATE_ACTION_INTEGRITY",
        "provenance_class": "EMPIRICAL_PROXY",
        "public_basis": "Rights issues, splits, reverse splits, warrants and other corporate actions can distort OHLCV-derived accumulation and structure proxies.",
        "scanner_implementation": "Extreme-return anomalies trigger a corporate-action review gate. Verified event evidence must clear or explain the anomaly before production use.",
    },
    {
        "formula_id": "EP22_EXECUTION_CAPACITY_SLIPPAGE",
        "provenance_class": "EMPIRICAL_PROXY",
        "public_basis": "A thesis can be correct while the trade is not executable because of tick size, thin ADTV, gaps and order-book depth.",
        "scanner_implementation": "Position value is capped by capital, risk budget and a liquidity-bucket participation limit. Tick impact, participation and gap risk form a conservative slippage proxy.",
    },
    {
        "formula_id": "EP23_OUTCOME_MEMORY_WALK_FORWARD",
        "provenance_class": "EMPIRICAL_PROXY",
        "public_basis": "Public performance examples do not substitute for point-in-time, survivorship-controlled validation of each scanner state.",
        "scanner_implementation": "Verified outcomes are stored by lifecycle and structure. SHADOW_ONLY never changes ranking; GUARDED can block states only after a minimum sample and adverse empirical evidence.",
    },
    {
        "formula_id": "EP24_FAIL_CLOSED_IDX_PRODUCTION",
        "provenance_class": "EMPIRICAL_PROXY",
        "public_basis": "IDX-specific data gaps and manipulation risks make optimistic default values unsafe.",
        "scanner_implementation": "Missing HSC/board/corporate-action evidence, stale OHLCV, provider failures or unverified direct evidence cannot be silently converted into a neutral score or production-ready action.",
    },
    {
        "formula_id": "EP25_AUTONOMOUS_PUBLIC_INGESTION",
        "provenance_class": "EMPIRICAL_PROXY",
        "public_basis": "A repeatable scanner should collect public evidence without requiring the user to assemble multiple manual CSV files.",
        "scanner_implementation": "Ticker-only input triggers OHLCV, KSEI profile/corporate-action, public news and deep fundamental collection with provider audit and fail-closed states.",
    },
    {
        "formula_id": "EP26_BROKER_INVENTORY_BEHAVIOURAL_PROXY",
        "provenance_class": "EMPIRICAL_PROXY",
        "public_basis": "Direct broker inventory is valuable but is not consistently available from a free public feed.",
        "scanner_implementation": "An explicitly labelled OHLCV proxy combines accumulation persistence, absorption, close acceptance, CMF, OBV and pullback-volume contraction. It never identifies a broker or owner.",
    },
    {
        "formula_id": "EP27_EOD_BID_OFFER_PROXY",
        "provenance_class": "EMPIRICAL_PROXY",
        "public_basis": "Live depth is licensed data, but EOD structure can still support a lower-confidence trigger plan.",
        "scanner_implementation": "EOD acceptance, absorption, volume confirmation, breakout proximity, gap and friction form a transparent microstructure proxy. Direct depth remains a higher evidence tier.",
    },
    {
        "formula_id": "EP28_DUAL_READINESS_TIER",
        "provenance_class": "EMPIRICAL_PROXY",
        "public_basis": "A fully automatic EOD workflow and a direct-flow precise-entry workflow should not be conflated.",
        "scanner_implementation": "EMIR_AUTO_EOD_READY permits a capped scenario using public/proxy evidence; EMIR_READY_WITH_PRECISE_TRIGGER still requires directly verified order-book and IDX integrity evidence.",
    },
    {
        "formula_id": "EP29_TRI_STATE_REGULATORY_EVIDENCE",
        "provenance_class": "EMPIRICAL_PROXY",
        "public_basis": "Absence of a public-provider response is not proof that an issuer has no regulatory flag.",
        "scanner_implementation": "HSC/FCA/suspension/UMA/sanction fields use TRUE_VERIFIED, FALSE_VERIFIED, or UNKNOWN_NOT_VERIFIED. Provider failure never becomes suspension=true or a clean clearance.",
    },
    {
        "formula_id": "EP30_ADMINISTRATIVE_EVENT_SEPARATION",
        "provenance_class": "EMPIRICAL_PROXY",
        "public_basis": "Administrative corporate actions do not automatically create a growth, earnings, or project narrative.",
        "scanner_implementation": "Bare KSEI cash-dividend, proxy-voting and other administrative rows remain integrity evidence and are excluded from narrative scoring unless supported by a separate eligible disclosure/news thesis.",
    },
    {
        "formula_id": "EP31_PERSISTENT_CACHE_INCREMENTAL_REFRESH",
        "provenance_class": "EMPIRICAL_PROXY",
        "public_basis": "Repeated point-in-time scans should reuse unchanged evidence while preserving source freshness and auditability.",
        "scanner_implementation": "Supabase stores bounded OHLCV and public-source caches. Warm scans use valid cache rows; stale OHLCV fetches only a tail window, merges by date, and verifies cache keys plus SHA-256 before scan publication.",
    },
    {
        "formula_id": "EP32_MULTIYEAR_INVENTORY_CONTEXT",
        "provenance_class": "PUBLIC_SYNTHESIS",
        "public_basis": "Public discussions emphasize looking beyond a one-year broker snapshot to understand who accumulated from the bottom, gradual distribution, bottoming and renewed collection.",
        "scanner_implementation": "When direct broker evidence is absent, an explicitly labelled OHLCV fallback now measures 20/60/120/252/504/756-bar inventory behaviour and distinguishes persistent collection, inventory release and bottoming-to-collection. It never identifies a broker or owner.",
    },
    {
        "formula_id": "EP33_MACRO_FIRST_SECTOR_LEADER_EXCEPTION",
        "provenance_class": "PUBLIC_SYNTHESIS",
        "public_basis": "Macro and sector rotation are mapped first, but leading sectors can retain relative strength even while the broad index is weak.",
        "scanner_implementation": "RISK_OFF remains a penalty/gate, except a strict LEADING-sector relative-strength condition can keep the thesis eligible with position size capped at 5% rather than treating the whole market as an absolute blocker.",
    },
    {
        "formula_id": "EP34_GROWTH_BASIS_AND_SOLVENCY_SEMANTICS",
        "provenance_class": "EMPIRICAL_PROXY",
        "public_basis": "Future-fundamental conversion requires financial evidence that is comparable through time and not distorted by ambiguous ratio definitions.",
        "scanner_implementation": "Fundamentals separate QoQ from same-quarter YoY, prefer YoY for the backward-compatible growth fields, add TTM profitability/cash-flow metrics, and distinguish interest-bearing-debt/equity from total-liabilities/equity and net-debt/equity.",
    },
    {
        "formula_id": "EP35_DUAL_EXECUTION_GEOMETRY",
        "provenance_class": "EMPIRICAL_PROXY",
        "public_basis": "A pullback accumulation and a breakout/retest are different executions and should not share a misleading single risk/reward number.",
        "scanner_implementation": "The scanner retains an accumulation plan but reports RR at entry-low/high and builds a separate breakout entry, stop and targets. The dashboard no longer labels midpoint RR as if it applied to the whole entry zone.",
    },
    {
        "formula_id": "EP36_BUSINESS_FUNDAMENTAL_PILLAR",
        "provenance_class": "PUBLIC_SYNTHESIS",
        "public_basis": "A narrative/flow thesis is stronger when the underlying business converts the story into revenue, earnings, margins and cash flow; price action alone does not prove that conversion.",
        "scanner_implementation": "Fundamental quality is now an independent 16% conviction pillar and a production-readiness gate. Missing or weak business evidence receives explicit FUNDAMENTAL_EVIDENCE_PENDING / WAIT_FUNDAMENTAL_CONVERSION states rather than being hidden inside flow or narrative scores.",
    },
    {
        "formula_id": "EP37_OFFICIAL_FIRST_FINANCIAL_RECONCILIATION",
        "provenance_class": "EMPIRICAL_PROXY",
        "public_basis": "Financial conversion should be grounded in issuer/exchange filings before lower-authority market-data proxies when capital is at risk.",
        "scanner_implementation": "Deep-review shortlist attempts IDX official XBRL. Same/newer official period overrides conflicting Yahoo proxy values, retains cross-source diagnostics, and records official OCF/FCF evidence with durable provenance.",
    },
    {
        "formula_id": "EP38_BLENDED_MARKET_REGIME",
        "provenance_class": "EMPIRICAL_PROXY",
        "public_basis": "Broad-index structure and stock/sector participation should be read together; one weak metric should not mechanically classify every stock as risk-off.",
        "scanner_implementation": "Market regime blends direct IHSG context 60% with universe breadth 40%; both weak remains RISK_OFF, otherwise healthy breadth can move a soft benchmark to SELECTIVE.",
    },
    {
        "formula_id": "EP39_GUARDED_REAL_MONEY_AUTHORIZATION",
        "provenance_class": "EMPIRICAL_PROXY",
        "public_basis": "Position sizing and entry require evidence, invalidation and liquidity discipline; a research proxy is not equivalent to executable market evidence.",
        "scanner_implementation": "GUARDED_REAL_MONEY caps risk at 0.75%, requires official/current financial and cash-flow evidence plus execution-quality gates, and never lets an EOD order-book proxy authorize production. Direct IDX integrity and direct bid-offer are required for scanner-side READY.",
    },
    {
        "formula_id": "EP40_FUNDAMENTAL_FRESHNESS_YTD_CONSISTENCY",
        "provenance_class": "EMPIRICAL_PROXY",
        "public_basis": "A strong latest quarter is more reliable when the cumulative/YTD business trend confirms it; stale relative reporting periods should not rank like current evidence.",
        "scanner_implementation": "v1.9.10 separates observed quality from evidence confidence so partial coverage is gated once rather than double-discounted; sector-aware business-quality weights remain unchanged.",
    },
)

NARRATIVE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "PROJECT_CAPACITY": (
        "project", "proyek", "expansion", "ekspansi", "capacity", "kapasitas", "plant", "pabrik",
        "smelter", "mine", "tambang", "contract", "kontrak", "order book", "data center", "hilirisasi",
        "commissioning", "commercial operation", "joint venture", "akuisisi", "acquisition", "capex",
    ),
    "EARNINGS_CONVERSION": (
        "profit", "laba", "revenue", "pendapatan", "margin", "ebitda", "guidance", "turnaround",
        "record high", "rekor", "cash flow", "free cash flow", "utilization",
    ),
    "OWNERSHIP_ALIGNMENT": (
        "buyback", "insider buy", "director buy", "pemegang saham", "strategic investor", "tender offer",
        "private placement", "rights issue", "merger", "spin off", "controller", "pengendali",
    ),
    "POLICY_SECTOR": (
        "policy", "kebijakan", "quota", "kuota", "tariff", "tarif", "export ban", "larangan ekspor",
        "subsidy", "subsidi", "commodity", "komoditas", "government contract", "proyek pemerintah",
    ),
    "SECULAR_TOP_DOWN": (
        "secular", "structural growth", "megatrend", "energy transition", "digitalisation", "digitalisasi",
        "ai demand", "data centre demand", "downstreaming", "industrial policy", "supply deficit", "supercycle",
    ),
}
NEGATIVE_KEYWORDS = (
    "default", "gagal bayar", "lawsuit", "gugatan", "fraud", "penipuan", "suspension", "suspensi",
    "bankruptcy", "pailit", "delisting", "investigation", "penyelidikan", "loss widens", "rugi membesar",
    "dilution", "dilusi", "going concern", "restatement", "revisi laporan", "debt restructuring",
)
HYPE_KEYWORDS = (
    "multibagger", "to the moon", "auto reject atas", " ara ", "viral", "hot stock", "must buy",
    "buy now", "cuan besar", "saham gorengan", "target price dinaikkan", "fomo",
)
OFFICIAL_HINTS = (
    "idx.id", "idx.co.id", "ojk.go.id", "investor relation", "annual report", "financial statement",
    "keterbukaan informasi", "press release", "laporan keuangan", "company website",
)

_REGULATOR_DOMAINS = ("idx.id", "idx.co.id", "ojk.go.id", "ksei.co.id")


def _is_https_regulator_url(value: Any) -> bool:
    try:
        parsed = urlparse(str(value or "").strip())
    except ValueError:
        return False
    host = str(parsed.hostname or "").lower().rstrip(".")
    return bool(
        parsed.scheme.lower() == "https"
        and any(host == domain or host.endswith("." + domain) for domain in _REGULATOR_DOMAINS)
    )
CONVERSION_TERMS = {
    "REVENUE": ("revenue", "pendapatan", "sales", "penjualan", "order book", "kontrak"),
    "MARGIN": ("margin", "efisiensi", "utilization", "utilisasi", "cost reduction", "biaya"),
    "EARNINGS": ("profit", "laba", "ebitda", "earnings", "net income"),
    "CASH_FLOW": ("cash flow", "arus kas", "free cash flow", "fcf"),
    "RERATING": ("strategic investor", "buyback", "dividend", "dividen", "index inclusion", "re-rating", "rerating"),
}


def formula_registry_frame() -> pd.DataFrame:
    return pd.DataFrame(PUBLIC_FORMULA_REGISTRY)


def _finite(value: Any, default: float = np.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _clip(value: Any, low: float = 0.0, high: float = 100.0) -> float:
    number = _finite(value, low)
    return max(low, min(high, number))


def _safe_div(numerator: Any, denominator: Any, default: float = np.nan) -> float:
    a = _finite(numerator, np.nan)
    b = _finite(denominator, np.nan)
    if not np.isfinite(a) or not np.isfinite(b) or abs(b) < 1e-12:
        return default
    return a / b


def _weighted_fixed(components: list[tuple[float, float, float]]) -> tuple[float, float]:
    """Return quality and evidence coverage as separate quantities.

    Previous versions multiplied each component score by evidence coverage and
    then nested the result inside another coverage-weighted aggregate.  That
    double-discounted partially observed but otherwise strong evidence.

    Quality is now the weighted mean of observed component values.  Coverage
    keeps the original fixed-denominator contract, so missing/weak evidence
    still blocks readiness without being misclassified as poor quality.
    """
    total_weight = sum(weight for _, weight, _ in components if weight > 0)
    if total_weight <= 0:
        return np.nan, 0.0
    score_sum = 0.0
    observed_weight = 0.0
    coverage = 0.0
    for value, weight, evidence_pct in components:
        if weight <= 0:
            continue
        evidence = _clip(evidence_pct) / 100.0
        coverage += weight * evidence
        if np.isfinite(value):
            score_sum += weight * _clip(value)
            observed_weight += weight
    score = score_sum / observed_weight if observed_weight > 0 else np.nan
    return score, 100.0 * coverage / total_weight


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        result = pd.isna(value)
        if isinstance(result, (bool, np.bool_)):
            return bool(result)
    except Exception:
        pass
    return False


def _clean_text(value: Any) -> str:
    if _is_missing(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "nat", "<na>"} else text


def _truthy(value: Any) -> bool:
    return _clean_text(value).lower() in {"true", "1", "yes", "y", "verified"}


def _flag_with_state(value: Any, *, verified: bool) -> tuple[Any, str]:
    if not verified or _is_missing(value) or not _clean_text(value):
        return np.nan, "UNKNOWN_NOT_VERIFIED"
    normalized = _clean_text(value).lower()
    if normalized in {"true", "1", "yes", "y", "verified"}:
        return True, "TRUE_VERIFIED"
    if normalized in {"false", "0", "no", "n", "clear", "none"}:
        return False, "FALSE_VERIFIED"
    return np.nan, "UNKNOWN_NOT_VERIFIED"


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = gain.div(loss.replace(0, np.nan))
    return 100 - (100 / (1 + rs))


def _atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    previous = frame["Close"].shift(1)
    true_range = pd.concat([
        frame["High"] - frame["Low"],
        (frame["High"] - previous).abs(),
        (frame["Low"] - previous).abs(),
    ], axis=1).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def _cmf(frame: pd.DataFrame, period: int = 20) -> pd.Series:
    spread = (frame["High"] - frame["Low"]).replace(0, np.nan)
    multiplier = ((frame["Close"] - frame["Low"]) - (frame["High"] - frame["Close"])).div(spread)
    mfv = multiplier.fillna(0) * frame["Volume"]
    return mfv.rolling(period).sum().div(frame["Volume"].rolling(period).sum().replace(0, np.nan))


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    return (np.sign(close.diff()).fillna(0) * volume.fillna(0)).cumsum()


def idx_tick(price: float) -> float:
    if price < 200:
        return 1.0
    if price < 500:
        return 2.0
    if price < 2_000:
        return 5.0
    if price < 5_000:
        return 10.0
    return 25.0


def round_idx(price: float, mode: str = "nearest") -> float:
    if not np.isfinite(price) or price <= 0:
        return np.nan
    tick = idx_tick(price)
    scaled = price / tick
    if mode == "up":
        return float(math.ceil(scaled) * tick)
    if mode == "down":
        return float(math.floor(scaled) * tick)
    return float(round(scaled) * tick)


def _liquidity_score(adtv20: float) -> float:
    thresholds = [
        (100_000_000_000, 95.0), (25_000_000_000, 85.0), (7_500_000_000, 72.0),
        (2_000_000_000, 55.0), (500_000_000, 38.0), (1, 20.0),
    ]
    for threshold, score in thresholds:
        if adtv20 >= threshold:
            return score
    return 0.0


def calculate_market_features(frame: pd.DataFrame, benchmark: pd.DataFrame | None = None, as_of: Any = None) -> dict[str, Any]:
    if frame is None or frame.empty or len(frame) < 60:
        return {"feature_state": "INSUFFICIENT_HISTORY", "history_bars": 0 if frame is None else len(frame)}
    local = frame.copy().sort_index()
    close = pd.to_numeric(local["Close"], errors="coerce")
    high = pd.to_numeric(local["High"], errors="coerce")
    low = pd.to_numeric(local["Low"], errors="coerce")
    volume = pd.to_numeric(local["Volume"], errors="coerce").fillna(0)
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    rsi14 = _rsi(close)
    atr14 = _atr(local)
    cmf20 = _cmf(local)
    obv = _obv(close, volume)
    value = close * volume
    value_ma20 = value.rolling(20).mean()
    volume_ma20 = volume.rolling(20).mean()
    range_ = (high - low).replace(0, np.nan)
    clv = ((close - low) / range_).clip(0, 1)
    daily_return = close.pct_change()
    high_volume = volume > volume_ma20 * 1.2
    median_range = range_.rolling(20).median()
    accumulation = (daily_return > 0) & high_volume & (clv >= 0.60)
    distribution = (daily_return < 0) & high_volume & (clv <= 0.45)
    absorption = high_volume & (range_ <= median_range * 1.05) & (clv >= 0.62)
    failed_absorption = high_volume & (clv <= 0.35)

    adtv20 = _finite(value.tail(20).mean(), 0.0)
    latest_close = _finite(close.iloc[-1], np.nan)
    latest_ema20 = _finite(ema20.iloc[-1], np.nan)
    latest_ema50 = _finite(ema50.iloc[-1], np.nan)
    latest_ema200 = _finite(ema200.iloc[-1], np.nan)
    latest_atr = _finite(atr14.iloc[-1], np.nan)
    latest_rsi = _finite(rsi14.iloc[-1], np.nan)
    latest_cmf = _finite(cmf20.iloc[-1], np.nan)
    volume_ratio = _safe_div(volume.iloc[-1], volume_ma20.iloc[-1], 0.0)
    turnover_acceleration = _safe_div(value.tail(5).mean(), value_ma20.iloc[-1], 0.0)
    up_value = value.where(daily_return > 0, 0)
    down_value = value.where(daily_return < 0, 0)
    up_value_ratio = _safe_div(up_value.tail(20).sum(), up_value.tail(20).sum() + down_value.tail(20).sum(), 0.5)
    acceptance = _finite(clv.tail(20).mul(volume.tail(20)).sum() / max(volume.tail(20).sum(), 1.0), 0.5)
    obv_slope = _safe_div(obv.iloc[-1] - obv.iloc[-21], abs(obv.iloc[-21]) + volume.tail(20).sum(), 0.0) if len(obv) >= 21 else 0.0
    pullback_mask = daily_return < 0
    pullback_volume_sample = volume.tail(20)[pullback_mask.tail(20)]
    pullback_volume_mean = pullback_volume_sample.mean() if not pullback_volume_sample.empty else np.nan
    pullback_volume_ratio = _safe_div(pullback_volume_mean, volume_ma20.iloc[-1], np.nan)
    momentum20 = 100 * _safe_div(latest_close, close.iloc[-21], np.nan) - 100 if len(close) >= 21 else np.nan
    momentum60 = 100 * _safe_div(latest_close, close.iloc[-61], np.nan) - 100 if len(close) >= 61 else np.nan

    rs20 = np.nan
    rs60 = np.nan
    if benchmark is not None and not benchmark.empty:
        aligned = pd.concat([close.rename("stock"), benchmark["Close"].rename("bench")], axis=1).dropna()
        if len(aligned) >= 21:
            rs20 = 100 * ((aligned["stock"].iloc[-1] / aligned["stock"].iloc[-21] - 1) - (aligned["bench"].iloc[-1] / aligned["bench"].iloc[-21] - 1))
        if len(aligned) >= 61:
            rs60 = 100 * ((aligned["stock"].iloc[-1] / aligned["stock"].iloc[-61] - 1) - (aligned["bench"].iloc[-1] / aligned["bench"].iloc[-61] - 1))
    rs_momentum = _finite(rs20, 0) - _finite(rs60, 0)

    trend_score = np.mean([
        100.0 if latest_close > latest_ema20 else 20.0,
        100.0 if latest_ema20 > latest_ema50 else 25.0,
        100.0 if latest_ema50 > latest_ema200 else 30.0,
        _clip(50 + 1.2 * _finite(momentum20, 0)),
        _clip(50 + 1.0 * _finite(rs60, 0)),
    ])
    absorption_score = _clip(45 + 11 * int(absorption.tail(20).sum()) - 14 * int(failed_absorption.tail(20).sum()))
    pullback_volume_contraction_score = _clip(
        82.0 - 55.0 * max(0.0, _finite(pullback_volume_ratio, 1.0) - 0.70)
    ) if np.isfinite(pullback_volume_ratio) else np.nan
    obv_slope20_pct = 100.0 * obv_slope
    stealth_score = _clip(
        0.24 * _clip(100 * up_value_ratio)
        + 0.20 * _clip(50 + 220 * latest_cmf)
        + 0.16 * _clip(50 + 6 * obv_slope20_pct)
        + 0.18 * _clip(100 * acceptance)
        + 0.12 * absorption_score
        + 0.10 * _finite(pullback_volume_contraction_score, 42)
    )

    # Emir publicly emphasises multi-year inventory rather than a one-year snapshot.
    # Without direct broker history this is only a behavioural OHLCV proxy; it never
    # identifies who owns inventory. The score asks whether accumulation/acceptance
    # persists across several horizons and whether recent flow is improving or fading.
    def _horizon_inventory(window: int) -> tuple[float, float, float]:
        n = min(int(window), len(local))
        if n < 20:
            return np.nan, np.nan, 0.0
        up = float(up_value.tail(n).sum())
        down = float(down_value.tail(n).sum())
        up_ratio_h = _safe_div(up, up + down, 0.5)
        vol_sum = float(volume.tail(n).sum())
        accept_h = _finite(clv.tail(n).mul(volume.tail(n)).sum() / max(vol_sum, 1.0), 0.5)
        obv_change_h = _safe_div(obv.iloc[-1] - obv.iloc[-n], abs(obv.iloc[-n]) + vol_sum, 0.0) if len(obv) >= n else 0.0
        acc_count = int(accumulation.tail(n).sum())
        dist_count = int(distribution.tail(n).sum())
        event_balance = _clip(50 + 9 * (acc_count - dist_count) / max(1.0, math.sqrt(n / 20.0)))
        range_high = _finite(high.tail(n).max(), np.nan)
        range_low = _finite(low.tail(n).min(), np.nan)
        location = 100 * _safe_div(latest_close - range_low, range_high - range_low, 0.5) if np.isfinite(range_high) and np.isfinite(range_low) and range_high > range_low else 50.0
        score_h = _clip(
            0.28 * _clip(100 * up_ratio_h)
            + 0.24 * _clip(100 * accept_h)
            + 0.20 * _clip(50 + 600 * obv_change_h)
            + 0.18 * event_balance
            + 0.10 * _clip(35 + 0.55 * location)
        )
        coverage_h = min(100.0, 100.0 * n / window)
        return score_h, _clip(100 * up_ratio_h), coverage_h

    horizon_specs = (20, 60, 120, 252, 504, 756)
    horizon_scores: dict[int, float] = {}
    horizon_up_ratios: dict[int, float] = {}
    horizon_coverages: dict[int, float] = {}
    for _window in horizon_specs:
        _score_h, _up_h, _cov_h = _horizon_inventory(_window)
        horizon_scores[_window] = _score_h
        horizon_up_ratios[_window] = _up_h
        horizon_coverages[_window] = _cov_h
    inventory_cycle_score, inventory_cycle_coverage = _weighted_fixed([
        (horizon_scores[20], 0.10, horizon_coverages[20]),
        (horizon_scores[60], 0.14, horizon_coverages[60]),
        (horizon_scores[120], 0.16, horizon_coverages[120]),
        (horizon_scores[252], 0.22, horizon_coverages[252]),
        (horizon_scores[504], 0.20, horizon_coverages[504]),
        (horizon_scores[756], 0.18, horizon_coverages[756]),
    ])
    long_inventory = np.nan
    _long_valid = [horizon_scores[w] for w in (252, 504, 756) if np.isfinite(horizon_scores[w])]
    if _long_valid:
        long_inventory = float(np.mean(_long_valid))
    recent_inventory = _weighted_fixed([
        (horizon_scores[20], 0.45, horizon_coverages[20]),
        (horizon_scores[60], 0.35, horizon_coverages[60]),
        (horizon_scores[120], 0.20, horizon_coverages[120]),
    ])[0]
    inventory_shift = _finite(recent_inventory, 50) - _finite(long_inventory, 50)
    if _finite(inventory_cycle_score, 0) >= 65 and inventory_shift >= -5:
        inventory_cycle_phase = "MULTIYEAR_COLLECTION_PERSISTING_PROXY"
    elif np.isfinite(long_inventory) and long_inventory >= 62 and inventory_shift <= -12:
        inventory_cycle_phase = "MULTIYEAR_INVENTORY_RELEASE_PROXY"
    elif np.isfinite(long_inventory) and long_inventory < 52 and inventory_shift >= 10:
        inventory_cycle_phase = "BOTTOMING_TO_COLLECTION_PROXY"
    else:
        inventory_cycle_phase = "MIXED_INVENTORY_CYCLE_PROXY"
    recent_value_mean = _finite(value.tail(20).mean(), np.nan)
    long_value_mean = _finite(value.tail(min(252, len(value))).mean(), np.nan)
    turnover_dryness = _clip(78 - 28 * max(0.0, _safe_div(recent_value_mean, long_value_mean, 1.0) - 0.75)) if np.isfinite(recent_value_mean) and np.isfinite(long_value_mean) and long_value_mean > 0 else np.nan
    # Preliminary dryness proxy; refined after range compression becomes available.
    inventory_dryness_multiyear_score = _clip(
        0.65 * _finite(inventory_cycle_score, 50)
        + 0.35 * _finite(turnover_dryness, 50)
    )

    distribution_score = _clip(
        12 * int(distribution.tail(20).sum())
        + 15 * int(failed_absorption.tail(20).sum())
        + max(0.0, 55 - 100 * acceptance)
    )
    extension_atr = _safe_div(latest_close - latest_ema20, latest_atr, np.nan)
    crowding_score = _clip(
        28 + 10 * max(0.0, _finite(extension_atr, 0) - 1.5)
        + 0.9 * max(0.0, _finite(latest_rsi, 50) - 65)
        + 9 * max(0.0, _finite(volume_ratio, 1) - 2.0)
        + 7 * max(0.0, _finite(turnover_acceleration, 1) - 2.0)
    )

    high20 = _finite(high.tail(20).max(), np.nan)
    prior_high20 = _finite(high.shift(1).tail(20).max(), np.nan)
    high55 = _finite(high.tail(55).max(), np.nan)
    low20 = _finite(low.tail(20).min(), np.nan)
    prior_low20 = _finite(low.shift(1).tail(20).min(), np.nan)
    previous_low20 = _finite(low.iloc[-40:-20].min(), np.nan) if len(low) >= 40 else np.nan
    previous_high20 = _finite(high.iloc[-40:-20].max(), np.nan) if len(high) >= 40 else np.nan

    recent_range_pct = 100 * _safe_div(high.tail(20).max() - low.tail(20).min(), close.tail(20).median(), np.nan)
    prior_range_pct = 100 * _safe_div(high.iloc[-80:-20].max() - low.iloc[-80:-20].min(), close.iloc[-80:-20].median(), np.nan) if len(close) >= 80 else np.nan
    compression_ratio = _safe_div(recent_range_pct, prior_range_pct, np.nan)
    range_compression_score = _clip(100 - 70 * max(0.0, _finite(compression_ratio, 1.0) - 0.45))
    inventory_dryness_multiyear_score = _clip(
        0.45 * _finite(inventory_cycle_score, 50)
        + 0.30 * _finite(turnover_dryness, 50)
        + 0.25 * range_compression_score
    )

    # Public-logic proxies for three market structures.
    lower_low_recent = bool(np.isfinite(previous_low20) and low.tail(20).min() < previous_low20)
    reclaimed_prior_low = bool(np.isfinite(previous_low20) and latest_close > previous_low20)
    failed_breakdown_reclaim_score = 85.0 if lower_low_recent and reclaimed_prior_low else 55.0 if reclaimed_prior_low else 20.0
    sell_value_recent = down_value.tail(10).sum()
    sell_value_prior = down_value.iloc[-30:-10].sum() if len(down_value) >= 30 else np.nan
    sell_pressure_change = _safe_div(sell_value_recent, sell_value_prior, 1.0)
    seller_exhaustion_score = _clip(
        0.45 * _clip(100 - 70 * max(0.0, _finite(sell_pressure_change, 1.0) - 0.45))
        + 0.30 * absorption_score
        + 0.25 * failed_breakdown_reclaim_score
    )
    higher_high_score = 90.0 if np.isfinite(previous_high20) and high.tail(20).max() > previous_high20 else 55.0 if latest_close >= prior_high20 * 0.98 else 20.0
    higher_low_score = 90.0 if np.isfinite(previous_low20) and low20 > previous_low20 else 55.0 if np.isfinite(previous_low20) and latest_close > previous_low20 else 20.0
    ema20_slope_pct = 100 * _safe_div(ema20.iloc[-1] - ema20.iloc[-11], abs(ema20.iloc[-11]), 0.0) if len(ema20) >= 11 else 0.0
    structure_change_score = _clip(
        0.35 * (100.0 if latest_close > latest_ema20 else 20.0)
        + 0.30 * (100.0 if latest_ema20 > latest_ema50 else 25.0)
        + 0.20 * _clip(50 + 8 * ema20_slope_pct)
        + 0.15 * failed_breakdown_reclaim_score
    )
    flow_persistence_score = _clip(
        0.35 * stealth_score + 0.25 * absorption_score + 0.20 * _clip(100 * up_value_ratio)
        + 0.20 * _clip(60 + 20 * (_finite(turnover_acceleration, 1) - 1))
    )
    reversal_score = _clip(
        0.30 * seller_exhaustion_score + 0.25 * absorption_score
        + 0.25 * structure_change_score + 0.20 * failed_breakdown_reclaim_score
    )
    continuation_price_flow_score = _clip(
        0.35 * higher_high_score + 0.20 * higher_low_score + 0.25 * flow_persistence_score + 0.20 * trend_score
    )
    sideways_quality_score = _clip(
        0.30 * range_compression_score + 0.25 * absorption_score
        + 0.25 * failed_breakdown_reclaim_score + 0.20 * (100 - distribution_score)
    )
    structure_scores = {
        "REVERSAL_SETUP": reversal_score,
        "CONTINUATION_SETUP": continuation_price_flow_score,
        "SIDEWAYS_ACCUMULATION": sideways_quality_score,
    }
    structure_mode = max(structure_scores, key=structure_scores.get)
    market_structure_score = structure_scores[structure_mode]
    if market_structure_score < 52:
        structure_mode = "NO_CLEAR_STRUCTURE"

    if distribution_score >= 60:
        stage = "DISTRIBUTION"
    elif latest_close > latest_ema20 > latest_ema50 and _finite(momentum20, 0) > 3:
        stage = "MARKUP"
    elif latest_close >= latest_ema50 and stealth_score >= 58:
        stage = "SILENT_ACCUMULATION"
    elif latest_close < latest_ema50 and latest_ema20 < latest_ema50:
        stage = "MARKDOWN"
    else:
        stage = "BASE_TRANSITION"
    markup_quality = _clip(0.45 * trend_score + 0.25 * stealth_score + 0.20 * _clip(100 * acceptance) + 0.10 * (100 - distribution_score))
    trim_state = "TRIM_OVERSHOOT_OR_DISTRIBUTION" if crowding_score >= 72 or distribution_score >= 55 or _finite(extension_atr, 0) >= 3.0 else "HOLD_SCENARIO_UNTIL_INVALIDATED"

    # IDX microstructure/data-integrity proxies. They are guards, not alpha signals.
    overnight_gap = (pd.to_numeric(local["Open"], errors="coerce") / close.shift(1) - 1).abs()
    realized_vol20 = 100 * _finite(daily_return.tail(20).std(ddof=0), 0.0)
    gap_days20 = int((overnight_gap.tail(20) >= 0.08).sum())
    extreme_move_days60 = int((daily_return.tail(60).abs() >= 0.20).sum())
    corporate_action_anomaly_days60 = int((daily_return.tail(60).abs() >= 0.45).sum())
    zero_volume_days20 = int(volume.tail(20).le(0).sum())
    gap_risk_score = _clip(18 * gap_days20 + 220 * _finite(overnight_gap.tail(20).median(), 0.0))
    extreme_move_risk_score = _clip(22 * extreme_move_days60 + 2.0 * max(0.0, realized_vol20 - 3.0))
    zero_volume_risk_score = _clip(5 * zero_volume_days20)
    execution_friction_score = _clip(
        0.35 * gap_risk_score + 0.30 * extreme_move_risk_score
        + 0.20 * zero_volume_risk_score + 0.15 * _clip(8 * realized_vol20)
    )
    now = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now(tz="Asia/Jakarta")
    if now.tzinfo is None:
        now = now.tz_localize("Asia/Jakarta")
    else:
        now = now.tz_convert("Asia/Jakarta")
    last_session = pd.Timestamp(local.index[-1]).normalize()
    data_age_days = max(0, int((now.tz_localize(None).normalize() - last_session).days))
    history_quality = _clip(100 * len(local) / 220)
    freshness_quality = 100.0 if data_age_days <= 4 else 65.0 if data_age_days <= 7 else 20.0
    volume_quality = _clip(100 - 5 * zero_volume_days20)
    anomaly_quality = _clip(100 - 25 * corporate_action_anomaly_days60 - 5 * max(0, extreme_move_days60 - corporate_action_anomaly_days60))
    ohlcv_integrity_score = _clip(0.30 * history_quality + 0.25 * freshness_quality + 0.25 * volume_quality + 0.20 * anomaly_quality)
    corporate_action_anomaly = bool(corporate_action_anomaly_days60 > 0)
    if data_age_days > 7:
        ohlcv_integrity_state = "STALE_DATA_BLOCK"
    elif len(local) < 220:
        ohlcv_integrity_state = "INSUFFICIENT_HISTORY"
    elif zero_volume_days20 >= 5:
        ohlcv_integrity_state = "ILLIQUID_OR_SUSPENDED_PATTERN"
    elif corporate_action_anomaly:
        ohlcv_integrity_state = "CORPORATE_ACTION_REVIEW_REQUIRED"
    else:
        ohlcv_integrity_state = "VALID"

    return {
        "feature_state": "OK",
        "history_bars": len(local),
        "last_date": local.index[-1].date().isoformat(),
        "last_price": round(latest_close, 4),
        "price_change_1d_pct": round(100.0 * (latest_close / _finite(close.iloc[-2], np.nan) - 1.0), 3) if len(close) >= 2 and np.isfinite(_finite(close.iloc[-2], np.nan)) and _finite(close.iloc[-2], 0) != 0 else np.nan,
        "ema20": round(latest_ema20, 4), "ema50": round(latest_ema50, 4), "ema200": round(latest_ema200, 4),
        "atr14": round(latest_atr, 4), "rsi14": round(latest_rsi, 2),
        "cmf20": round(latest_cmf, 4) if np.isfinite(latest_cmf) else np.nan,
        "momentum20_pct": round(momentum20, 2) if np.isfinite(momentum20) else np.nan,
        "momentum60_pct": round(momentum60, 2) if np.isfinite(momentum60) else np.nan,
        "relative_strength20_pct": round(rs20, 2) if np.isfinite(rs20) else np.nan,
        "relative_strength60_pct": round(rs60, 2) if np.isfinite(rs60) else np.nan,
        "relative_strength_momentum_pct": round(rs_momentum, 2),
        "adtv20_idr": round(adtv20, 2), "liquidity_score": round(_liquidity_score(adtv20), 1),
        "volume_ratio20": round(volume_ratio, 3), "turnover_acceleration": round(turnover_acceleration, 3),
        "up_value_ratio20_pct": round(100 * up_value_ratio, 1), "close_acceptance20_pct": round(100 * acceptance, 1),
        "accumulation_days20": int(accumulation.tail(20).sum()), "distribution_days20": int(distribution.tail(20).sum()),
        "absorption_days20": int(absorption.tail(20).sum()), "failed_absorption_days20": int(failed_absorption.tail(20).sum()),
        "pullback_volume_ratio": round(pullback_volume_ratio, 3) if np.isfinite(pullback_volume_ratio) else np.nan,
        "pullback_volume_contraction_score": round(pullback_volume_contraction_score, 1) if np.isfinite(pullback_volume_contraction_score) else np.nan,
        "obv_slope20_pct": round(obv_slope20_pct, 3) if np.isfinite(obv_slope20_pct) else np.nan,
        "inventory_score_20": round(horizon_scores[20], 1) if np.isfinite(horizon_scores[20]) else np.nan,
        "inventory_score_60": round(horizon_scores[60], 1) if np.isfinite(horizon_scores[60]) else np.nan,
        "inventory_score_120": round(horizon_scores[120], 1) if np.isfinite(horizon_scores[120]) else np.nan,
        "inventory_score_252": round(horizon_scores[252], 1) if np.isfinite(horizon_scores[252]) else np.nan,
        "inventory_score_504": round(horizon_scores[504], 1) if np.isfinite(horizon_scores[504]) else np.nan,
        "inventory_score_756": round(horizon_scores[756], 1) if np.isfinite(horizon_scores[756]) else np.nan,
        "inventory_cycle_score": round(inventory_cycle_score, 1) if np.isfinite(inventory_cycle_score) else np.nan,
        "inventory_cycle_coverage_pct": round(inventory_cycle_coverage, 1),
        "inventory_cycle_phase": inventory_cycle_phase,
        "inventory_shift_recent_vs_long_score": round(inventory_shift, 1),
        "inventory_dryness_multiyear_score": round(inventory_dryness_multiyear_score, 1) if np.isfinite(inventory_dryness_multiyear_score) else np.nan,
        "inventory_proxy_provenance_state": "MULTIHORIZON_OHLCV_PROXY_NOT_BROKER_INVENTORY",
        "trend_score": round(_clip(trend_score), 1),
        "smart_money_score": round(stealth_score, 1),
        "smart_money_coverage_pct": 100.0 if len(local) >= 220 else round(100 * len(local) / 220, 1),
        "absorption_score": round(absorption_score, 1), "markup_quality_score": round(markup_quality, 1),
        "distribution_score": round(distribution_score, 1), "crowding_score": round(crowding_score, 1),
        "extension_atr": round(extension_atr, 2) if np.isfinite(extension_atr) else np.nan,
        "price_stage": stage, "high20": round(high20, 4), "high55": round(high55, 4), "low20": round(low20, 4),
        "market_structure_mode": structure_mode,
        "market_structure_score": round(market_structure_score, 1),
        "reversal_score": round(reversal_score, 1),
        "continuation_price_flow_score": round(continuation_price_flow_score, 1),
        "sideways_quality_score": round(sideways_quality_score, 1),
        "seller_exhaustion_score": round(seller_exhaustion_score, 1),
        "structure_change_score": round(structure_change_score, 1),
        "higher_high_score": round(higher_high_score, 1),
        "higher_low_score": round(higher_low_score, 1),
        "flow_persistence_score": round(flow_persistence_score, 1),
        "range_compression_score": round(range_compression_score, 1),
        "fakeout_reclaim_score": round(failed_breakdown_reclaim_score, 1),
        "realized_volatility20_pct": round(realized_vol20, 3),
        "gap_days20": gap_days20,
        "gap_risk_score": round(gap_risk_score, 1),
        "extreme_move_days60": extreme_move_days60,
        "corporate_action_anomaly_days60": corporate_action_anomaly_days60,
        "extreme_move_risk_score": round(extreme_move_risk_score, 1),
        "zero_volume_days20": zero_volume_days20,
        "execution_friction_score": round(execution_friction_score, 1),
        "data_age_days": data_age_days,
        "ohlcv_integrity_score": round(ohlcv_integrity_score, 1),
        "ohlcv_integrity_state": ohlcv_integrity_state,
        "corporate_action_anomaly_flag": corporate_action_anomaly,
        "trim_state": trim_state,
        "react_not_predict_state": "SCENARIO_BASED_NO_PRICE_FORECAST",
    }


def calculate_market_context(benchmark: pd.DataFrame | None) -> dict[str, Any]:
    if benchmark is None or benchmark.empty or len(benchmark) < 60:
        return {"market_regime": "MARKET_CONTEXT_UNAVAILABLE", "market_context_score": np.nan, "market_context_coverage_pct": 0.0}
    features = calculate_market_features(benchmark, None)
    trend = _finite(features.get("trend_score"), 0)
    distribution = _finite(features.get("distribution_score"), 100)
    momentum = _finite(features.get("momentum20_pct"), 0)
    score = _clip(0.55 * trend + 0.25 * (100 - distribution) + 0.20 * _clip(50 + 2 * momentum))
    if score >= 68:
        regime = "RISK_ON"
    elif score <= 40 or distribution >= 60:
        regime = "RISK_OFF"
    else:
        regime = "SELECTIVE"
    return {
        "market_regime": regime,
        "market_context_score": round(score, 1),
        "market_context_coverage_pct": 100.0,
        "market_trend_score": features.get("trend_score"),
        "market_distribution_score": features.get("distribution_score"),
        "market_index_structure_mode": features.get("market_structure_mode"),
    }


def calculate_market_context_from_universe(fast: pd.DataFrame | None) -> dict[str, Any]:
    """Fail-closed market proxy when ^JKSE is unavailable.

    Uses only valid cross-sectional ticker features. This is explicitly a universe-breadth
    proxy, not direct IHSG data, and receives reduced coverage.
    """
    if fast is None or fast.empty or "feature_state" not in fast.columns:
        return {
            "market_regime": "MARKET_CONTEXT_UNAVAILABLE",
            "market_context_score": np.nan,
            "market_context_coverage_pct": 0.0,
            "market_context_provenance_state": "NO_BENCHMARK_OR_UNIVERSE_PROXY",
        }
    valid = fast[fast["feature_state"].astype(str).eq("OK")].copy()
    if len(valid) < 20:
        return {
            "market_regime": "MARKET_CONTEXT_UNAVAILABLE",
            "market_context_score": np.nan,
            "market_context_coverage_pct": 0.0,
            "market_context_provenance_state": "INSUFFICIENT_UNIVERSE_BREADTH",
            "market_proxy_valid_tickers": int(len(valid)),
        }

    last_price = pd.to_numeric(valid.get("last_price"), errors="coerce")
    ema50 = pd.to_numeric(valid.get("ema50"), errors="coerce")
    breadth_mask = last_price.notna() & ema50.notna()
    breadth = 100.0 * last_price[breadth_mask].gt(ema50[breadth_mask]).mean() if breadth_mask.any() else np.nan
    trend = pd.to_numeric(valid.get("trend_score"), errors="coerce").median()
    distribution = pd.to_numeric(valid.get("distribution_score"), errors="coerce").median()
    flow = pd.to_numeric(valid.get("smart_money_score"), errors="coerce").median()
    momentum = pd.to_numeric(valid.get("momentum20_pct"), errors="coerce").median()

    values = [breadth, trend, distribution, flow, momentum]
    if sum(np.isfinite(_finite(value, np.nan)) for value in values) < 4:
        return {
            "market_regime": "MARKET_CONTEXT_UNAVAILABLE",
            "market_context_score": np.nan,
            "market_context_coverage_pct": 0.0,
            "market_context_provenance_state": "UNIVERSE_PROXY_FEATURES_INCOMPLETE",
            "market_proxy_valid_tickers": int(len(valid)),
        }

    score = _clip(
        0.30 * _finite(breadth, 50)
        + 0.25 * _finite(trend, 50)
        + 0.20 * (100 - _finite(distribution, 50))
        + 0.15 * _finite(flow, 50)
        + 0.10 * _clip(50 + 2 * _finite(momentum, 0))
    )
    if score >= 65 and _finite(breadth, 0) >= 55:
        regime = "RISK_ON"
    elif score <= 38 or _finite(distribution, 0) >= 65 or _finite(breadth, 100) < 25:
        regime = "RISK_OFF"
    else:
        regime = "SELECTIVE"
    coverage = min(85.0, 55.0 + 30.0 * min(1.0, len(valid) / 100.0))
    return {
        "market_regime": regime,
        "market_context_score": round(score, 1),
        "market_context_coverage_pct": round(coverage, 1),
        "market_context_provenance_state": "UNIVERSE_BREADTH_PROXY_NOT_DIRECT_IHSG",
        "market_proxy_valid_tickers": int(len(valid)),
        "market_breadth_above_ema50_pct": round(_finite(breadth, np.nan), 1),
        "market_trend_score": round(_finite(trend, np.nan), 1),
        "market_distribution_score": round(_finite(distribution, np.nan), 1),
        "market_flow_score": round(_finite(flow, np.nan), 1),
        "market_momentum20_median_pct": round(_finite(momentum, np.nan), 2),
    }



def blend_market_context(benchmark_context: Mapping[str, Any] | None, universe_context: Mapping[str, Any] | None) -> dict[str, Any]:
    """Blend direct IHSG structure with cross-sectional breadth.

    Direct benchmark remains primary, while breadth prevents a single distribution metric
    from forcing RISK_OFF when participation is still healthy. Both weak => fail-closed.
    """
    b=dict(benchmark_context or {}); u=dict(universe_context or {})
    bscore=_finite(b.get("market_context_score"),np.nan); uscore=_finite(u.get("market_context_score"),np.nan)
    if not np.isfinite(bscore): return u
    if not np.isfinite(uscore): return b
    breadth=_finite(u.get("market_breadth_above_ema50_pct"),np.nan)
    dist=_finite(b.get("market_distribution_score"),50)
    score=_clip(.60*bscore+.40*uscore)
    if score>=65 and _finite(breadth,0)>=55 and dist<60: regime="RISK_ON"
    elif score<=38 or _finite(breadth,100)<25 or (dist>=75 and _finite(breadth,100)<40): regime="RISK_OFF"
    else: regime="SELECTIVE"
    return {**u,**b,
        "market_regime":regime,"market_context_score":round(score,1),
        "market_context_coverage_pct":round(min(100,.60*_finite(b.get("market_context_coverage_pct"),0)+.40*_finite(u.get("market_context_coverage_pct"),0)),1),
        "market_context_provenance_state":"BLENDED_IHSG_AND_UNIVERSE_BREADTH",
        "market_benchmark_score":round(bscore,1),"market_universe_breadth_score":round(uscore,1),
        "market_breadth_above_ema50_pct":u.get("market_breadth_above_ema50_pct"),
        "market_flow_score":u.get("market_flow_score"),"market_momentum20_median_pct":u.get("market_momentum20_median_pct"),
    }

def calculate_sector_context(fast: pd.DataFrame, universe: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Cross-sectionally centred RRG-style sector proxy.

    Absolute RS/momentum are retained for audit, but quadrants are centred on the current
    sector cross-section. This prevents an uninformative all-WEAKENING result when every
    sector shares the same market beta while still preserving which sectors lead/lag peers.
    """
    if fast.empty or universe.empty or "sector" not in universe.columns:
        return {}
    merged = fast.merge(universe[["ticker", "sector"]], on="ticker", how="left")
    merged["sector"] = merged["sector"].fillna("").astype(str).str.strip()
    merged = merged[merged["sector"].ne("") & merged["feature_state"].eq("OK")]
    if merged.empty:
        return {}

    def safe_median(values: Any) -> float:
        series = pd.to_numeric(values, errors="coerce")
        if not isinstance(series, pd.Series):
            series = pd.Series(series, dtype=float)
        finite = series.replace([np.inf, -np.inf], np.nan).dropna()
        return float(finite.median()) if not finite.empty else np.nan

    sector_rows: list[dict[str, Any]] = []
    for sector, group in merged.groupby("sector"):
        prices = pd.to_numeric(group["last_price"], errors="coerce")
        ema50 = pd.to_numeric(group["ema50"], errors="coerce")
        valid_breadth = prices.notna() & ema50.notna()
        breadth = float(100 * prices[valid_breadth].gt(ema50[valid_breadth]).mean()) if valid_breadth.any() else np.nan
        strength = safe_median(group.get("relative_strength60_pct", pd.Series(dtype=float)))
        strength_momentum = safe_median(group.get("relative_strength_momentum_pct", pd.Series(dtype=float)))
        median_flow = safe_median(group.get("smart_money_score", pd.Series(dtype=float)))
        score = _clip(
            0.30 * _finite(breadth, 50) + 0.25 * _clip(50 + 1.5 * _finite(strength, 0))
            + 0.20 * _clip(50 + 2.0 * _finite(strength_momentum, 0)) + 0.25 * _finite(median_flow, 50)
        )
        if np.isfinite(strength) and np.isfinite(strength_momentum):
            absolute = "LEADING" if strength >= 0 and strength_momentum >= 0 else "IMPROVING" if strength < 0 <= strength_momentum else "WEAKENING" if strength >= 0 > strength_momentum else "LAGGING"
        else:
            absolute = "UNKNOWN"
        component_coverage = 100.0 * sum(np.isfinite(value) for value in (breadth, strength, strength_momentum, median_flow)) / 4.0
        coverage = _clip(min(100.0, 25.0 * len(group)) * component_coverage / 100.0)
        sector_rows.append({"sector": sector, "group": group, "breadth": breadth, "strength": strength, "momentum": strength_momentum, "flow": median_flow, "score": score, "absolute": absolute, "coverage": coverage})

    strengths = pd.Series([r["strength"] for r in sector_rows], dtype=float).replace([np.inf, -np.inf], np.nan)
    momenta = pd.Series([r["momentum"] for r in sector_rows], dtype=float).replace([np.inf, -np.inf], np.nan)
    finite_s, finite_m = strengths.dropna(), momenta.dropna()
    center_s = float(finite_s.median()) if not finite_s.empty else np.nan
    center_m = float(finite_m.median()) if not finite_m.empty else np.nan
    mad_s = float((finite_s - center_s).abs().median()) if not finite_s.empty else 0.0
    mad_m = float((finite_m - center_m).abs().median()) if not finite_m.empty else 0.0
    std_s = float(finite_s.std(ddof=0)) if len(finite_s) > 1 else 0.0
    std_m = float(finite_m.std(ddof=0)) if len(finite_m) > 1 else 0.0
    scale_s = max(mad_s, std_s, 0.25)
    scale_m = max(mad_m, std_m, 0.25)

    output: dict[str, dict[str, Any]] = {}
    for row in sector_rows:
        strength, momentum = row["strength"], row["momentum"]
        rs_index = _clip(100 + 8.0 * (_finite(strength, center_s) - _finite(center_s, 0)) / scale_s, 80, 120)
        mom_index = _clip(100 + 8.0 * (_finite(momentum, center_m) - _finite(center_m, 0)) / scale_m, 80, 120)
        if np.isfinite(strength) and np.isfinite(momentum):
            state = "LEADING" if rs_index >= 100 and mom_index >= 100 else "IMPROVING" if rs_index < 100 <= mom_index else "WEAKENING" if rs_index >= 100 > mom_index else "LAGGING"
        else:
            state = "UNKNOWN"
        for ticker in row["group"]["ticker"]:
            output[str(ticker)] = {
                "sector": row["sector"], "sector_leadership_score": round(row["score"], 1),
                "sector_state": state, "sector_rrg_state": state, "sector_rrg_state_absolute": row["absolute"],
                "sector_relative_strength_pct": round(_finite(strength, 0), 2),
                "sector_strength_momentum_pct": round(_finite(momentum, 0), 2),
                "sector_rrg_rs_index": round(rs_index, 2), "sector_rrg_momentum_index": round(mom_index, 2),
                "sector_context_method": "RRG_PROXY_CROSS_SECTIONAL_CENTERED_NOT_OFFICIAL_RRG",
                "sector_context_coverage_pct": round(row["coverage"], 1), "sector_member_count": int(len(row["group"])),
            }
    return output

def aggregate_broker_summary(frame: pd.DataFrame | None) -> dict[str, dict[str, Any]]:
    if frame is None or frame.empty:
        return {}
    local = frame.copy()
    local.columns = [str(column).strip().lower() for column in local.columns]
    if "ticker" not in local.columns:
        return {}
    local["ticker"] = local["ticker"].map(normalize_ticker)
    for column in (
        "buy_value", "sell_value", "buy_volume", "sell_volume", "avg_buy_price", "avg_sell_price",
        "crossing_value", "crossing_price", "lookback_years",
    ):
        if column in local.columns:
            local[column] = pd.to_numeric(local[column], errors="coerce")
    if "date" in local.columns:
        local["date"] = pd.to_datetime(local["date"], errors="coerce")
    output: dict[str, dict[str, Any]] = {}
    for ticker, group in local.groupby("ticker"):
        verified_mask = group.get("source_verified", pd.Series(False, index=group.index)).map(_truthy)
        verified_group = group[verified_mask].copy()
        if verified_group.empty:
            output[ticker] = {
                "broker_summary_score": np.nan,
                "broker_summary_coverage_pct": 0.0,
                "broker_inventory_score": np.nan,
                "broker_inventory_coverage_pct": 0.0,
                "broker_summary_provenance_state": "UNVERIFIED_NOT_USED_FOR_PRODUCTION",
                "beneficial_owner_inference_state": "NOT_INFERRED_FROM_BROKER_CODE",
            }
            continue
        buy_series = verified_group["buy_value"] if "buy_value" in verified_group else verified_group.get("buy_volume", pd.Series(0.0, index=verified_group.index))
        sell_series = verified_group["sell_value"] if "sell_value" in verified_group else verified_group.get("sell_volume", pd.Series(0.0, index=verified_group.index))
        buy = pd.to_numeric(buy_series, errors="coerce").fillna(0).sum()
        sell = pd.to_numeric(sell_series, errors="coerce").fillna(0).sum()
        net_ratio = _safe_div(buy - sell, buy + sell, np.nan)
        broker_score = _clip(50 + 250 * _finite(net_ratio, 0)) if np.isfinite(net_ratio) else np.nan

        if "date" in verified_group and verified_group["date"].notna().any():
            monthly = verified_group.assign(net=pd.to_numeric(buy_series, errors="coerce").fillna(0) - pd.to_numeric(sell_series, errors="coerce").fillna(0)).groupby(verified_group["date"].dt.to_period("M"))["net"].sum()
            positive_month_ratio = _safe_div((monthly > 0).sum(), len(monthly), 0.0)
            coverage_years = max(0.0, (verified_group["date"].max() - verified_group["date"].min()).days / 365.25)
        else:
            positive_month_ratio = _safe_div((pd.to_numeric(buy_series, errors="coerce").fillna(0) > pd.to_numeric(sell_series, errors="coerce").fillna(0)).sum(), len(verified_group), 0.0)
            coverage_years = _finite(verified_group.get("lookback_years", pd.Series(np.nan)).max(), 0.0)
        holder_persistence = _clip(100 * positive_month_ratio)

        participant = verified_group.get("participant_type", pd.Series("", index=verified_group.index)).astype(str).str.upper()
        retail_mask = participant.str.contains("RETAIL") | verified_group.get("retail_proxy_flag", pd.Series(False, index=verified_group.index)).map(_truthy)
        fund_mask = participant.str.contains("FUND|INSTITUTION|ASSET|FOREIGN_INSTITUTION", regex=True)
        net_values = pd.to_numeric(buy_series, errors="coerce").fillna(0) - pd.to_numeric(sell_series, errors="coerce").fillna(0)
        retail_net = net_values[retail_mask].sum() if retail_mask.any() else np.nan
        smart_net = net_values[~retail_mask].sum() if retail_mask.any() else np.nan
        retail_exit_score = 85.0 if np.isfinite(retail_net) and retail_net < 0 and _finite(smart_net, 0) > 0 else 50.0 if retail_mask.any() else np.nan
        cannibalisation_risk = 80.0 if retail_mask.any() and _finite(retail_net, 0) > 0 and _finite(smart_net, 0) <= 0 else 15.0 if retail_mask.any() else np.nan
        fund_like_score = _clip(50 + 250 * _safe_div(net_values[fund_mask].sum(), net_values.abs().sum(), 0.0)) if fund_mask.any() else np.nan

        crossing_value = _finite(verified_group.get("crossing_value", pd.Series(np.nan)).max(), np.nan)
        median_trade = _finite((pd.to_numeric(buy_series, errors="coerce").fillna(0) + pd.to_numeric(sell_series, errors="coerce").fillna(0)).median(), np.nan)
        crossing_ratio = _safe_div(crossing_value, median_trade, np.nan)
        jumbo_crossing_score = _clip(35 + 18 * math.log1p(max(0.0, _finite(crossing_ratio, 0)))) if np.isfinite(crossing_ratio) else np.nan
        crossing_rows = verified_group[pd.to_numeric(verified_group.get("crossing_value", pd.Series(np.nan, index=verified_group.index)), errors="coerce").notna()]
        defended_level = _finite(crossing_rows.sort_values("crossing_value", ascending=False).iloc[0].get("crossing_price"), np.nan) if not crossing_rows.empty else np.nan
        if not np.isfinite(defended_level) and "avg_buy_price" in verified_group:
            weights = pd.to_numeric(buy_series, errors="coerce").fillna(0)
            prices = pd.to_numeric(verified_group["avg_buy_price"], errors="coerce")
            valid = prices.notna() & weights.gt(0)
            defended_level = _finite(np.average(prices[valid], weights=weights[valid]), np.nan) if valid.any() else np.nan
        defended_level_score = _clip(45 + 0.35 * _finite(jumbo_crossing_score, 0) + 0.20 * holder_persistence) if np.isfinite(defended_level) else np.nan

        buyback_flag = verified_group.get("buyback_flag", pd.Series(False, index=verified_group.index)).map(_truthy).any()
        inventory_dryness = _clip(0.55 * holder_persistence + 0.25 * _finite(broker_score, 50) + 0.20 * (85 if buyback_flag else 45))
        inventory_score, inventory_coverage = _weighted_fixed([
            (broker_score, 0.28, 100),
            (holder_persistence, 0.24, min(100, 40 + 30 * coverage_years)),
            (retail_exit_score, 0.16, 100 if np.isfinite(retail_exit_score) else 0),
            (fund_like_score, 0.12, 100 if np.isfinite(fund_like_score) else 0),
            (defended_level_score, 0.12, 100 if np.isfinite(defended_level_score) else 0),
            (inventory_dryness, 0.08, 100),
        ])
        shift_state = "COLLECTION_PERSISTING" if holder_persistence >= 65 and _finite(net_ratio, 0) > 0 else "DISTRIBUTION_DOMINANT" if _finite(net_ratio, 0) < -0.10 else "MIXED_OR_BOTTOMING"
        output[ticker] = {
            "broker_summary_score": round(broker_score, 1) if np.isfinite(broker_score) else np.nan,
            "broker_summary_coverage_pct": round(min(100.0, 12.5 * len(verified_group)), 1),
            "broker_net_ratio": round(net_ratio, 4) if np.isfinite(net_ratio) else np.nan,
            "broker_inventory_score": round(inventory_score, 1) if np.isfinite(inventory_score) else np.nan,
            "broker_inventory_coverage_pct": round(inventory_coverage, 1),
            "inventory_coverage_years": round(coverage_years, 2),
            "holder_persistence_score": round(holder_persistence, 1),
            "inventory_dryness_score": round(inventory_dryness, 1),
            "retail_exit_score": round(retail_exit_score, 1) if np.isfinite(retail_exit_score) else np.nan,
            "retail_cannibalisation_risk": round(cannibalisation_risk, 1) if np.isfinite(cannibalisation_risk) else np.nan,
            "fund_like_flow_score": round(fund_like_score, 1) if np.isfinite(fund_like_score) else np.nan,
            "jumbo_crossing_score": round(jumbo_crossing_score, 1) if np.isfinite(jumbo_crossing_score) else np.nan,
            "defended_level": round(defended_level, 4) if np.isfinite(defended_level) else np.nan,
            "defended_level_score": round(defended_level_score, 1) if np.isfinite(defended_level_score) else np.nan,
            "broker_inventory_shift_state": shift_state,
            "broker_summary_provenance_state": "DIRECT_SOURCE_VERIFIED",
            "beneficial_owner_inference_state": "NOT_INFERRED_FROM_BROKER_CODE",
        }
    return output


def parse_orderbook_evidence(frame: pd.DataFrame | None) -> dict[str, dict[str, Any]]:
    if frame is None or frame.empty:
        return {}
    local = frame.copy()
    local.columns = [str(column).strip().lower() for column in local.columns]
    if "ticker" not in local.columns:
        return {}
    local["ticker"] = local["ticker"].map(normalize_ticker)
    output: dict[str, dict[str, Any]] = {}
    for ticker, group in local.groupby("ticker"):
        verified = group.get("source_verified", pd.Series(False, index=group.index)).map(_truthy)
        valid = group[verified].copy()
        if valid.empty:
            output[ticker] = {
                "orderbook_trigger_score": np.nan,
                "orderbook_coverage_pct": 0.0,
                "orderbook_provenance_state": "UNVERIFIED_NOT_USED_FOR_PRODUCTION",
            }
            continue
        for column in ("resistance_price", "offer_lot", "median_offer_lot", "small_lot_share_pct", "break_seconds", "break_value"):
            if column in valid.columns:
                valid[column] = pd.to_numeric(valid[column], errors="coerce")
        latest = valid.iloc[-1]
        offer_ratio = _safe_div(latest.get("offer_lot"), latest.get("median_offer_lot"), np.nan)
        small_share = _finite(latest.get("small_lot_share_pct"), np.nan)
        retail_stack_score = _clip(35 + 18 * math.log1p(max(0.0, _finite(offer_ratio, 0))) + 0.35 * _finite(small_share, 0)) if np.isfinite(offer_ratio) else np.nan
        break_seconds = _finite(latest.get("break_seconds"), np.nan)
        speed_score = _clip(100 - 1.6 * max(0.0, _finite(break_seconds, 60))) if np.isfinite(break_seconds) else np.nan
        break_value = _finite(latest.get("break_value"), np.nan)
        break_value_score = _clip(40 + 12 * math.log10(max(1.0, break_value / 1_000_000))) if np.isfinite(break_value) else np.nan
        trigger_score, coverage = _weighted_fixed([
            (retail_stack_score, 0.40, 100 if np.isfinite(retail_stack_score) else 0),
            (speed_score, 0.35, 100 if np.isfinite(speed_score) else 0),
            (break_value_score, 0.25, 100 if np.isfinite(break_value_score) else 0),
        ])
        resistance = _finite(latest.get("resistance_price"), np.nan)
        output[ticker] = {
            "orderbook_trigger_score": round(trigger_score, 1) if np.isfinite(trigger_score) else np.nan,
            "orderbook_coverage_pct": round(coverage, 1),
            "precise_trigger_price": round(resistance, 4) if np.isfinite(resistance) else np.nan,
            "retail_offer_stack_score": round(retail_stack_score, 1) if np.isfinite(retail_stack_score) else np.nan,
            "offer_absorption_speed_score": round(speed_score, 1) if np.isfinite(speed_score) else np.nan,
            "orderbook_provenance_state": "DIRECT_SOURCE_VERIFIED",
            "orderbook_observed_at": str(latest.get("observed_at") or ""),
            "orderbook_note": str(latest.get("note") or ""),
        }
    return output


def parse_ownership(frame: pd.DataFrame | None) -> dict[str, dict[str, Any]]:
    if frame is None or frame.empty:
        return {}
    local = frame.copy()
    local.columns = [str(column).strip().lower() for column in local.columns]
    if "ticker" not in local.columns:
        return {}
    output: dict[str, dict[str, Any]] = {}
    for _, row in local.iterrows():
        ticker = normalize_ticker(row.get("ticker"))
        verified = _truthy(row.get("source_verified", True))
        if not verified:
            output[ticker] = {"ownership_score": np.nan, "ownership_coverage_pct": 0.0, "ownership_provenance_state": "UNVERIFIED_NOT_USED_FOR_PRODUCTION"}
            continue
        reported_float = _finite(row.get("free_float_pct"), np.nan)
        affiliated_public = _finite(row.get("affiliated_public_holding_pct"), 0.0)
        effective_float = max(0.0, reported_float - affiliated_public) if np.isfinite(reported_float) else np.nan
        fake_float_gap = affiliated_public if np.isfinite(reported_float) else np.nan
        alignment = _finite(row.get("owner_alignment_score"), np.nan)
        concentration = _finite(row.get("concentration_score"), np.nan)
        relationship_confidence = _finite(row.get("holder_relationship_confidence_pct"), np.nan)
        buyback_inventory = _finite(row.get("buyback_inventory_pct"), np.nan)
        disclosure_over_1 = _truthy(row.get("over_1pct_disclosure_flag"))
        insider = _truthy(row.get("insider_buy_flag"))
        effective_float_score = 85 if np.isfinite(effective_float) and 10 <= effective_float <= 35 else 62 if np.isfinite(effective_float) and 5 <= effective_float <= 50 else 30 if np.isfinite(effective_float) else np.nan
        network_score, network_coverage = _weighted_fixed([
            (alignment, 0.35, 100 if np.isfinite(alignment) else 0),
            (relationship_confidence, 0.25, 100 if np.isfinite(relationship_confidence) else 0),
            (concentration, 0.15, 100 if np.isfinite(concentration) else 0),
            (effective_float_score, 0.15, 100 if np.isfinite(effective_float_score) else 0),
            (80 if disclosure_over_1 else np.nan, 0.05, 100 if disclosure_over_1 else 0),
            (80 if insider else np.nan, 0.05, 100 if insider else 0),
        ])
        passive_flow_risk = 75 if np.isfinite(effective_float) and effective_float < 10 else 55 if np.isfinite(effective_float) and effective_float < 15 else 25 if np.isfinite(effective_float) else np.nan
        output[ticker] = {
            "ownership_score": round(network_score, 1) if np.isfinite(network_score) else np.nan,
            "ownership_coverage_pct": round(network_coverage, 1),
            "reported_free_float_pct": round(reported_float, 2) if np.isfinite(reported_float) else np.nan,
            "effective_free_float_pct": round(effective_float, 2) if np.isfinite(effective_float) else np.nan,
            "fake_float_gap_pct": round(fake_float_gap, 2) if np.isfinite(fake_float_gap) else np.nan,
            "ownership_network_score": round(network_score, 1) if np.isfinite(network_score) else np.nan,
            "buyback_inventory_pct": round(buyback_inventory, 2) if np.isfinite(buyback_inventory) else np.nan,
            "passive_flow_risk_score": round(passive_flow_risk, 1) if np.isfinite(passive_flow_risk) else np.nan,
            "ownership_provenance_state": "DIRECT_SOURCE_VERIFIED",
            "ownership_note": str(row.get("ownership_note") or ""),
        }
    return output


def parse_idx_integrity(frame: pd.DataFrame | None, as_of: Any = None) -> dict[str, dict[str, Any]]:
    """Parse direct IDX evidence with per-row tri-state semantics."""
    if frame is None or frame.empty:
        return {}
    local = frame.copy()
    local.columns = [str(column).strip().lower() for column in local.columns]
    if "ticker" not in local.columns:
        return {}
    aliases = {
        "fca_flag": "full_call_auction_flag",
        "special_monitoring_board_flag": "special_monitoring_flag",
        "high_shareholding_concentration_flag": "hsc_flag",
        "free_float": "free_float_pct",
        "board": "listing_board",
        "date": "observed_at",
        "source": "source_url",
    }
    for source, target in aliases.items():
        if source in local.columns and target not in local.columns:
            local[target] = local[source]
    local["ticker"] = local["ticker"].map(normalize_ticker)
    now = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now(tz="Asia/Jakarta")
    now = now.tz_localize("Asia/Jakarta") if now.tzinfo is None else now.tz_convert("Asia/Jakarta")
    output: dict[str, dict[str, Any]] = {}
    for _, row in local.iterrows():
        ticker = normalize_ticker(row.get("ticker"))
        verified = _truthy(row.get("source_verified", False))
        observed = pd.to_datetime(row.get("observed_at"), errors="coerce", utc=True)
        age_days = max(0.0, (now.tz_convert("UTC") - observed).total_seconds() / 86400) if pd.notna(observed) else np.nan
        listing_board = _clean_text(row.get("listing_board")).upper()
        hsc, hsc_state = _flag_with_state(row.get("hsc_flag"), verified=verified)
        special, special_state = _flag_with_state(row.get("special_monitoring_flag"), verified=verified)
        fca, fca_state = _flag_with_state(row.get("full_call_auction_flag"), verified=verified)
        suspended, suspension_state = _flag_with_state(row.get("suspension_flag"), verified=verified)
        uma, uma_state = _flag_with_state(row.get("uma_flag"), verified=verified)
        sanctions, sanctions_state = _flag_with_state(row.get("sanctions_flag"), verified=verified)
        corporate_action, corporate_action_state = _flag_with_state(row.get("corporate_action_flag"), verified=verified)
        free_float = _finite(row.get("free_float_pct"), np.nan)
        over_1, over_1_state = _flag_with_state(row.get("over_1pct_disclosure_flag"), verified=verified)
        freshness_score = 100.0 if np.isfinite(age_days) and age_days <= 35 else 65.0 if np.isfinite(age_days) and age_days <= 60 else 20.0 if np.isfinite(age_days) else np.nan
        fields_present = [
            bool(listing_board), np.isfinite(free_float), hsc_state != "UNKNOWN_NOT_VERIFIED",
            special_state != "UNKNOWN_NOT_VERIFIED" or fca_state != "UNKNOWN_NOT_VERIFIED",
            suspension_state != "UNKNOWN_NOT_VERIFIED", pd.notna(observed), bool(_clean_text(row.get("source_url"))),
        ]
        coverage = 100.0 * sum(fields_present) / len(fields_present) if verified else 0.0
        score = 100.0
        score -= 45.0 if hsc is True else 0.0
        score -= 55.0 if special is True or fca is True else 0.0
        score -= 90.0 if suspended is True else 0.0
        score -= 18.0 if uma is True else 0.0
        score -= 40.0 if sanctions is True else 0.0
        if np.isfinite(free_float):
            score -= 45.0 if free_float < 7.5 else 22.0 if free_float < 15.0 else 0.0
        score -= 12.0 if corporate_action is True else 0.0
        if np.isfinite(freshness_score):
            score = 0.85 * score + 0.15 * freshness_score
        score = _clip(score)
        hard_block_reasons: list[str] = []
        if suspended is True: hard_block_reasons.append("SUSPENDED")
        if special is True or fca is True: hard_block_reasons.append("SPECIAL_MONITORING_OR_FCA")
        if hsc is True: hard_block_reasons.append("HIGH_SHAREHOLDING_CONCENTRATION")
        if sanctions is True: hard_block_reasons.append("REGULATORY_SANCTION")
        if np.isfinite(free_float) and free_float < 7.5: hard_block_reasons.append("EXTREME_LOW_FREE_FLOAT")
        stale = not np.isfinite(age_days) or age_days > 60
        if stale: hard_block_reasons.append("STALE_OR_MISSING_IDX_EVIDENCE")
        hard_block = bool(hard_block_reasons)
        caution: list[str] = []
        if uma is True: caution.append("UMA_CAUTION")
        if np.isfinite(free_float) and 7.5 <= free_float < 15: caution.append("FREE_FLOAT_BELOW_15PCT")
        if corporate_action is True: caution.append("CORPORATE_ACTION_ACTIVE_OR_RECENT")
        if over_1 is not True: caution.append("OWNERSHIP_ABOVE_1PCT_NOT_CONFIRMED")
        unknown_states = [hsc_state, special_state, fca_state, suspension_state, uma_state, sanctions_state]
        unknown_count = sum(state == "UNKNOWN_NOT_VERIFIED" for state in unknown_states)
        if unknown_count:
            caution.append("CRITICAL_IDX_FIELDS_UNKNOWN_NOT_VERIFIED")
        if not verified:
            state = "UNVERIFIED_NOT_USED_FOR_PRODUCTION"
        elif hard_block:
            state = "IDX_INTEGRITY_HARD_BLOCK"
        elif caution:
            state = "IDX_INTEGRITY_CAUTION"
        else:
            state = "IDX_INTEGRITY_CLEAR"
        output[ticker] = {
            "idx_integrity_score": round(score, 1) if verified else np.nan,
            "idx_integrity_coverage_pct": round(coverage, 1),
            "idx_integrity_state": state,
            "idx_integrity_hard_block": hard_block if verified else True,
            "idx_integrity_block_reasons": " | ".join(hard_block_reasons) or "NONE",
            "idx_integrity_caution_flags": " | ".join(dict.fromkeys(caution)) or "NONE",
            "idx_integrity_provenance_state": "DIRECT_SOURCE_VERIFIED" if verified else "UNVERIFIED_NOT_USED_FOR_PRODUCTION",
            "idx_integrity_observed_at": observed.isoformat() if pd.notna(observed) else "",
            "idx_integrity_age_days": round(age_days, 1) if np.isfinite(age_days) else np.nan,
            "listing_board": listing_board or "UNKNOWN",
            "listing_board_verification_state": "TRUE_VERIFIED" if verified and listing_board else "UNKNOWN_NOT_VERIFIED",
            "hsc_flag": hsc, "hsc_verification_state": hsc_state,
            "special_monitoring_flag": special, "special_monitoring_verification_state": special_state,
            "full_call_auction_flag": fca, "full_call_auction_verification_state": fca_state,
            "suspension_flag": suspended, "suspension_verification_state": suspension_state,
            "uma_flag": uma, "uma_verification_state": uma_state,
            "sanctions_flag": sanctions, "sanctions_verification_state": sanctions_state,
            "regulatory_free_float_pct": round(free_float, 2) if np.isfinite(free_float) else np.nan,
            "regulatory_free_float_verification_state": "TRUE_VERIFIED" if verified and np.isfinite(free_float) else "UNKNOWN_NOT_VERIFIED",
            "over_1pct_disclosure_flag": over_1,
            "over_1pct_disclosure_verification_state": over_1_state,
            "idx_integrity_unknown_critical_count": unknown_count + int(not listing_board) + int(not np.isfinite(free_float)),
            "corporate_action_flag": corporate_action,
            "corporate_action_verification_state": corporate_action_state,
            "corporate_action_type": _clean_text(row.get("corporate_action_type")).upper(),
            "corporate_action_effective_date": _clean_text(row.get("corporate_action_effective_date")),
            "corporate_action_review_cleared": bool(verified and corporate_action is False),
            "idx_integrity_source_url": _clean_text(row.get("source_url")),
            "idx_integrity_note": _clean_text(row.get("note") or row.get("integrity_note")),
        }
    return output


def build_outcome_calibration(frame: pd.DataFrame | None) -> dict[str, dict[str, Any]]:
    """Build point-in-time shadow calibration by lifecycle/structure.

    Scores are not probabilities. Outcome memory is therefore a reliability gate, not a
    probability calibration transform.
    """
    if frame is None or frame.empty:
        return {}
    local = frame.copy()
    local.columns = [str(column).strip().lower() for column in local.columns]
    aliases = {
        "lifecycle": "emir_lifecycle",
        "structure": "market_structure_mode",
        "return": "return_pct",
        "max_drawdown": "max_drawdown_pct",
        "verified": "outcome_verified",
    }
    for source, target in aliases.items():
        if source in local.columns and target not in local.columns:
            local[target] = local[source]
    if "outcome_verified" not in local.columns:
        return {}
    local = local[local["outcome_verified"].map(_truthy)].copy()
    if local.empty:
        return {}
    for column in ("return_pct", "max_drawdown_pct", "horizon_days"):
        if column not in local.columns:
            local[column] = np.nan
        local[column] = pd.to_numeric(local[column], errors="coerce")
    for column in ("emir_lifecycle", "market_structure_mode"):
        if column not in local.columns:
            local[column] = "UNKNOWN"
        local[column] = local[column].fillna("UNKNOWN").astype(str)

    def metrics(group: pd.DataFrame) -> dict[str, Any]:
        returns = group["return_pct"].dropna()
        drawdowns = group["max_drawdown_pct"].dropna()
        n = int(len(returns))
        win_rate = 100.0 * float((returns > 0).mean()) if n else np.nan
        median_return = _finite(returns.median(), np.nan) if n else np.nan
        median_drawdown = _finite(drawdowns.median(), np.nan) if len(drawdowns) else np.nan
        invalidated = group.get("thesis_invalidated", pd.Series(False, index=group.index)).map(_truthy)
        invalidation_rate = 100.0 * float(invalidated.mean()) if len(group) else np.nan
        if n < 30:
            state = "INSUFFICIENT_SAMPLE_SHADOW_ONLY"
        elif (np.isfinite(win_rate) and win_rate < 40) or (np.isfinite(median_return) and median_return <= 0) or (np.isfinite(median_drawdown) and median_drawdown <= -18):
            state = "EMPIRICAL_EDGE_REJECTED"
        elif np.isfinite(win_rate) and win_rate >= 52 and np.isfinite(median_return) and median_return > 0:
            state = "EMPIRICAL_EDGE_SUPPORTED"
        else:
            state = "EMPIRICAL_EDGE_MIXED"
        return {
            "outcome_sample_n": n,
            "outcome_win_rate_pct": round(win_rate, 1) if np.isfinite(win_rate) else np.nan,
            "outcome_median_return_pct": round(median_return, 2) if np.isfinite(median_return) else np.nan,
            "outcome_median_drawdown_pct": round(median_drawdown, 2) if np.isfinite(median_drawdown) else np.nan,
            "outcome_thesis_invalidation_rate_pct": round(invalidation_rate, 1) if np.isfinite(invalidation_rate) else np.nan,
            "outcome_calibration_state": state,
        }

    output: dict[str, dict[str, Any]] = {"GLOBAL": metrics(local)}
    for (lifecycle, structure), group in local.groupby(["emir_lifecycle", "market_structure_mode"], dropna=False):
        output[f"{lifecycle}|{structure}"] = metrics(group)
    return output


def select_outcome_calibration(
    calibration_map: Mapping[str, Mapping[str, Any]] | None,
    lifecycle: str,
    structure_mode: str,
) -> dict[str, Any]:
    if not calibration_map:
        return {
            "outcome_sample_n": 0,
            "outcome_win_rate_pct": np.nan,
            "outcome_median_return_pct": np.nan,
            "outcome_median_drawdown_pct": np.nan,
            "outcome_thesis_invalidation_rate_pct": np.nan,
            "outcome_calibration_state": "NO_OUTCOME_MEMORY",
        }
    key = f"{lifecycle}|{structure_mode}"
    return dict(calibration_map.get(key) or calibration_map.get("GLOBAL") or {})


def _event_text(row: Mapping[str, Any]) -> str:
    return " ".join(str(row.get(key) or "") for key in (
        "title", "summary", "category", "publisher", "source_tier", "url", "macro_theme", "secular_trend", "catalyst",
    )).lower()


def _categorize(text: str) -> tuple[str, int]:
    category, hits = "GENERAL_MARKET_NEWS", 0
    for candidate, keywords in NARRATIVE_KEYWORDS.items():
        candidate_hits = sum(1 for keyword in keywords if keyword in text)
        if candidate_hits > hits:
            category, hits = candidate, candidate_hits
    return category, hits


def _issuer_alignment(text: str, issuer_context: Mapping[str, Any] | None) -> float:
    if not issuer_context:
        return np.nan
    tokens: list[str] = []
    for key in ("company_name", "sector", "theme", "macro_theme", "secular_trend", "catalyst"):
        value = str(issuer_context.get(key) or "").lower()
        tokens.extend(token for token in re.findall(r"[a-z0-9]+", value) if len(token) >= 4)
    unique = set(tokens)
    if not unique:
        return np.nan
    hits = sum(1 for token in unique if token in text)
    return _clip(20 + 16 * hits)



def _issuer_relevance(text: str, ticker: str, issuer_context: Mapping[str, Any] | None, *, trusted_tag: bool = False) -> tuple[bool, float]:
    """Reject obvious cross-ticker search contamination before narrative scoring.

    Direct/official evidence is trusted because its ticker tag is explicit. Public search
    results must mention the ticker or enough distinctive company-name tokens.
    """
    if trusted_tag:
        return True, 100.0
    bare = str(ticker or "").upper().replace(".JK", "").strip().lower()
    lower = str(text or "").lower()
    referenced = set(re.findall(r"(?:idx:|\()([a-z]{4})(?:\)|\b)", lower))
    if referenced and bare and bare not in referenced:
        return False, 0.0
    ticker_hit = bool(bare and re.search(rf"(?<![a-z0-9]){re.escape(bare)}(?![a-z0-9])", lower))
    context = issuer_context or {}
    company = str(context.get("company_name") or "").lower()
    stop = {"pt", "tbk", "persero", "indonesia", "international", "internasional", "group", "holding", "the", "and"}
    tokens = [t for t in re.findall(r"[a-z0-9]+", company) if len(t) >= 4 and t not in stop]
    distinctive = list(dict.fromkeys(tokens))[:6]
    hits = sum(1 for token in distinctive if token in lower)
    required = 1 if len(distinctive) <= 1 else 2
    company_hit = bool(distinctive and hits >= required)
    thematic_tokens: list[str] = []
    for key in ("theme", "macro_theme", "secular_trend", "catalyst", "sector"):
        thematic_tokens.extend(t for t in re.findall(r"[a-z0-9]+", str(context.get(key) or "").lower()) if len(t) >= 5 and t not in stop)
    thematic = list(dict.fromkeys(thematic_tokens))[:10]
    thematic_hits = sum(1 for token in thematic if token in lower)
    thematic_hit = bool(thematic and thematic_hits >= 1)
    score = 100.0 if ticker_hit else min(90.0, 35.0 + 22.0 * hits) if company_hit else min(72.0, 30.0 + 14.0 * thematic_hits) if thematic_hit else 0.0
    return bool(ticker_hit or company_hit or thematic_hit), score

def _story_fingerprint(title: Any) -> str:
    text = str(title or "").lower().split(" - ")[0]
    tokens = re.findall(r"[a-z0-9]+", text)
    stop = {"pt", "tbk", "dan", "yang", "untuk", "dari", "the", "a", "an", "of", "in", "on", "saham", "stock"}
    material = sorted({token for token in tokens if len(token) >= 3 and token not in stop})
    return "|".join(material[:18])


def score_narrative_events(events: pd.DataFrame | None, as_of: Any = None, issuer_context: Mapping[str, Any] | None = None) -> dict[str, Any]:
    empty = {
        "narrative_score": np.nan, "narrative_coverage_pct": 0.0, "narrative_state": "NO_ACTIVE_PUBLIC_NARRATIVE",
        "narrative_event_count": 0, "narrative_category": "NONE", "narrative_latest_title": "",
        "narrative_materiality_score": np.nan, "financial_conversion_score": np.nan,
        "issuer_alignment_score": np.nan, "issuer_alignment_coverage_pct": 0.0,
        "top_down_catalyst_score": np.nan, "industry_translation_score": np.nan, "story_runway_score": np.nan,
        "retail_adoption_stage": "UNKNOWN", "narrative_risk_flags": "NO_SOURCED_EVENT",
        "narrative_independent_story_count": 0, "narrative_source_independence_score": np.nan,
        "narrative_syndication_ratio_pct": np.nan, "narrative_contradiction_score": np.nan,
        "narrative_verified_source_count": 0, "narrative_official_source_count": 0,
        "narrative_future_event_filtered_count": 0,
        "narrative_source_provenance_state": "NO_VERIFIED_SOURCE", "narrative_relevance_filtered_count": 0,
        "conversion_path": "NO_EVIDENCE", "thesis_statement": "Narrative thesis belum memiliki evidence publik yang cukup.",
    }
    if events is None or events.empty:
        return empty
    local = events.copy()
    if "narrative_eligible" in local.columns:
        eligible_mask = local["narrative_eligible"].map(
            lambda value: True if _is_missing(value) or _clean_text(value) == "" else _truthy(value)
        )
        local = local[eligible_mask].copy()
    if "event_role" in local.columns:
        administrative = local["event_role"].map(_clean_text).str.upper().isin({
            "ADMINISTRATIVE_CORPORATE_ACTION", "GOVERNANCE_ADMINISTRATIVE", "PROXY_VOTING_ADMINISTRATIVE"
        })
        local = local[~administrative].copy()
    if local.empty:
        result = dict(empty)
        result["narrative_risk_flags"] = "ONLY_ADMINISTRATIVE_OR_INELIGIBLE_EVENTS"
        result["thesis_statement"] = "Hanya terdapat event administratif; belum ada narrative thesis yang eligible."
        return result
    local["published_at"] = pd.to_datetime(local.get("published_at", local.get("event_date")), errors="coerce", utc=True)
    now = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now(tz="UTC")
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")

    # Point-in-time contract: evidence cannot exist before it was public.  Historical
    # replay may pass a table containing events collected after `as_of`; those rows
    # must be removed rather than receiving age_days=0 / maximum freshness.
    future_event_mask = local["published_at"].notna() & local["published_at"].gt(now)
    future_event_filtered_count = int(future_event_mask.sum())
    if future_event_filtered_count:
        local = local.loc[~future_event_mask].copy()
    if local.empty:
        result = dict(empty)
        result["narrative_future_event_filtered_count"] = future_event_filtered_count
        result["narrative_risk_flags"] = "NO_POINT_IN_TIME_ELIGIBLE_EVENT"
        result["thesis_statement"] = "Tidak ada narrative event yang sudah tersedia pada timestamp keputusan."
        return result

    rows: list[dict[str, Any]] = []
    conversion_hits: set[str] = set()
    relevance_filtered = 0
    event_ticker = str(local.get("ticker", pd.Series([""])).iloc[0] if "ticker" in local.columns and len(local) else "")
    for _, row in local.iterrows():
        text = _event_text(row)
        direct_tag = _truthy(row.get("source_verified", False)) or str(row.get("source_tier") or "").upper() in {"OFFICIAL", "ISSUER", "REGULATOR"}
        relevant, relevance_score = _issuer_relevance(text, str(row.get("ticker") or event_ticker), issuer_context, trusted_tag=direct_tag)
        if not relevant:
            relevance_filtered += 1
            continue
        category, hits = _categorize(text)
        negative_hits = sum(1 for keyword in NEGATIVE_KEYWORDS if keyword in text)
        hype_hits = sum(1 for keyword in HYPE_KEYWORDS if keyword in f" {text} ")
        local_conversion_hits = {path for path, keywords in CONVERSION_TERMS.items() if any(keyword in text for keyword in keywords)}
        conversion_hits.update(local_conversion_hits)
        published = row.get("published_at")
        age_days = max(0.0, (now - published).total_seconds() / 86400) if pd.notna(published) else 120.0
        freshness = 100 * math.exp(-age_days / 55.0)
        source_verified = _truthy(row.get("source_verified", False))
        regulator_domain = _is_https_regulator_url(row.get("url"))
        declared_official = str(row.get("source_tier") or "").upper() in {"OFFICIAL", "ISSUER", "REGULATOR"}
        official = bool(regulator_domain or (declared_official and source_verified))
        evidence_verified = bool(source_verified or regulator_domain)
        source_quality = 95.0 if official else 80.0 if evidence_verified else 68.0 if row.get("url") else 35.0
        materiality = _finite(row.get("materiality_score"), np.nan)
        if not np.isfinite(materiality):
            materiality = _clip(28 + 13 * hits + 15 * int(official) - 20 * negative_hits)
        bridge = _finite(row.get("financial_bridge_score"), np.nan)
        if not np.isfinite(bridge):
            bridge = _clip(20 + 14 * len(local_conversion_hits) + 10 * int(category in {"PROJECT_CAPACITY", "EARNINGS_CONVERSION"}) - 18 * negative_hits)
        alignment = _issuer_alignment(text, issuer_context)
        top_down = _finite(row.get("top_down_catalyst_score"), np.nan)
        if not np.isfinite(top_down):
            top_down = _clip(30 + 16 * int(category in {"POLICY_SECTOR", "SECULAR_TOP_DOWN"}) + 10 * hits + 8 * int(official))
        industry_translation = _finite(row.get("industry_translation_score"), np.nan)
        if not np.isfinite(industry_translation):
            industry_translation = _clip(25 + 0.40 * top_down + 0.35 * bridge + 0.25 * _finite(alignment, 40))
        event_score = _clip(
            0.13 * freshness + 0.19 * materiality + 0.20 * bridge + 0.14 * source_quality
            + 0.10 * _finite(alignment, 40) + 0.10 * top_down + 0.08 * industry_translation + 0.06 * relevance_score
            - 20 * negative_hits - 8 * hype_hits
        )
        runway = _clip(0.30 * freshness + 0.25 * materiality + 0.25 * bridge + 0.20 * top_down)
        rows.append({
            "event_score": event_score, "category": category, "negative_hits": negative_hits, "hype_hits": hype_hits,
            "published_at": published, "title": str(row.get("title") or ""), "publisher": str(row.get("publisher") or ""),
            "official": official, "source_verified": evidence_verified, "materiality": materiality, "bridge": bridge, "alignment": alignment,
            "top_down": top_down, "industry_translation": industry_translation, "runway": runway,
            "story_fingerprint": _story_fingerprint(row.get("title")), "issuer_relevance_score": relevance_score,
        })
    if not rows:
        result = dict(empty)
        result["narrative_relevance_filtered_count"] = relevance_filtered
        result["narrative_risk_flags"] = "NO_ISSUER_RELEVANT_PUBLIC_EVENT"
        result["thesis_statement"] = "Public search events tersedia, tetapi tidak lolos issuer-relevance gate."
        return result
    scored = pd.DataFrame(rows).sort_values("event_score", ascending=False)
    scored["story_fingerprint"] = scored["story_fingerprint"].where(
        scored["story_fingerprint"].astype(str).str.len().gt(0),
        scored.index.map(lambda value: f"UNIQUE_{value}"),
    )
    independent = scored.drop_duplicates("story_fingerprint", keep="first")
    top = independent.head(5)
    weighted = np.average(top["event_score"], weights=np.linspace(1.0, 0.55, len(top))) if len(top) else np.nan
    unique_publishers = independent["publisher"].replace("", np.nan).nunique(dropna=True)
    official_count = int(independent["official"].sum())
    verified_count = int(independent["source_verified"].sum())
    independent_count = int(len(independent))
    raw_count = int(len(scored))
    source_independence = _clip(100.0 * independent_count / max(raw_count, 1))
    syndication_ratio = _clip(100.0 * (1.0 - independent_count / max(raw_count, 1)))
    coverage = _clip(10 * min(5, independent_count) + 8 * min(3, unique_publishers) + 14 * min(2, official_count) + 8 * min(2, verified_count))
    narrative_score = _clip(weighted) if np.isfinite(weighted) else np.nan
    materiality_score = _finite(top["materiality"].mean(), np.nan)
    bridge_score = _finite(top["bridge"].mean(), np.nan)
    alignment_score = _finite(top["alignment"].mean(), np.nan)
    top_down_score = _finite(top["top_down"].mean(), np.nan)
    industry_translation_score = _finite(top["industry_translation"].mean(), np.nan)
    runway_score = _finite(top["runway"].mean(), np.nan)
    alignment_coverage = coverage if np.isfinite(alignment_score) else 0.0
    negative_total = int(independent["negative_hits"].sum())
    hype_total = int(independent["hype_hits"].sum())
    contradiction_score = _clip(100.0 * float((independent["negative_hits"] > 0).mean())) if len(independent) else np.nan
    if negative_total >= 2:
        state = "CONTRADICTED_OR_NEGATIVE"
    elif hype_total >= 2 and _finite(narrative_score, 0) >= 58:
        state = "PUBLIC_HYPE_CROWDED"
    elif _finite(narrative_score, 0) >= 70 and coverage >= 55 and official_count >= 1:
        state = "MATERIAL_THESIS_CONFIRMED"
    elif _finite(narrative_score, 0) >= 56:
        state = "THESIS_FORMING"
    else:
        state = "WEAK_OR_UNCONVERTED"
    retail_stage = "EUPHORIA" if hype_total >= 2 else "ADOPTION" if hype_total == 1 else "EARLY_AWARENESS" if len(scored) >= 2 else "PRE_RETAIL"
    top_category = str(top.iloc[0]["category"]) if len(top) else "NONE"
    conversion_path = " → ".join(path for path in ("REVENUE", "MARGIN", "EARNINGS", "CASH_FLOW", "RERATING") if path in conversion_hits) or "UNPROVEN_CONVERSION"
    risks: list[str] = []
    if negative_total:
        risks.append("NEGATIVE_EVENT_PRESENT")
    if hype_total:
        risks.append("HYPE_LANGUAGE_PRESENT")
    if official_count == 0:
        risks.append("NO_OFFICIAL_SOURCE")
    if verified_count == 0:
        risks.append("NO_DIRECTLY_VERIFIED_SOURCE")
    if not np.isfinite(alignment_score):
        risks.append("ISSUER_ALIGNMENT_UNPROVEN")
    latest_title = str(independent.sort_values("published_at", ascending=False, na_position="last").iloc[0]["title"]) if len(independent) else ""
    thesis_statement = f"{top_category}: {latest_title}" if latest_title else f"{top_category} narrative sedang dibentuk."
    return {
        "narrative_score": round(narrative_score, 1) if np.isfinite(narrative_score) else np.nan,
        "narrative_coverage_pct": round(coverage, 1), "narrative_state": state,
        "narrative_event_count": len(scored), "narrative_independent_story_count": independent_count,
        "narrative_source_independence_score": round(source_independence, 1),
        "narrative_syndication_ratio_pct": round(syndication_ratio, 1),
        "narrative_contradiction_score": round(contradiction_score, 1) if np.isfinite(contradiction_score) else np.nan,
        "narrative_verified_source_count": verified_count, "narrative_relevance_filtered_count": relevance_filtered,
        "narrative_future_event_filtered_count": future_event_filtered_count,
        "narrative_official_source_count": official_count,
        "narrative_source_provenance_state": "VERIFIED_OFFICIAL_SOURCE" if official_count else "VERIFIED_NON_OFFICIAL_SOURCE" if verified_count else "UNVERIFIED_PUBLIC_NEWS_ONLY",
        "narrative_category": top_category, "narrative_latest_title": latest_title,
        "narrative_materiality_score": round(materiality_score, 1) if np.isfinite(materiality_score) else np.nan,
        "financial_conversion_score": round(bridge_score, 1) if np.isfinite(bridge_score) else np.nan,
        "issuer_alignment_score": round(alignment_score, 1) if np.isfinite(alignment_score) else np.nan,
        "issuer_alignment_coverage_pct": round(alignment_coverage, 1),
        "top_down_catalyst_score": round(top_down_score, 1) if np.isfinite(top_down_score) else np.nan,
        "industry_translation_score": round(industry_translation_score, 1) if np.isfinite(industry_translation_score) else np.nan,
        "story_runway_score": round(runway_score, 1) if np.isfinite(runway_score) else np.nan,
        "retail_adoption_stage": retail_stage, "narrative_risk_flags": " | ".join(risks) or "NO_MAJOR_NARRATIVE_RISK",
        "conversion_path": conversion_path, "thesis_statement": thesis_statement,
    }


def classify_lifecycle(features: Mapping[str, Any], narrative: Mapping[str, Any], broker: Mapping[str, Any] | None = None) -> str:
    broker = broker or {}
    flow = _finite(features.get("smart_money_score"), np.nan)
    inventory = _finite(broker.get("broker_inventory_score"), np.nan)
    story = _finite(narrative.get("narrative_score"), np.nan)
    coverage = _finite(narrative.get("narrative_coverage_pct"), 0)
    distribution = max(_finite(features.get("distribution_score"), 0), 100 - _finite(inventory, 100) if np.isfinite(inventory) else 0)
    crowding = _finite(features.get("crowding_score"), 0)
    stage = str(features.get("price_stage") or "")
    narrative_state = str(narrative.get("narrative_state") or "")
    retail_stage = str(narrative.get("retail_adoption_stage") or "")
    inventory_cycle_phase = str(broker.get("inventory_cycle_phase") or "")
    broker_shift = str(broker.get("broker_inventory_shift_state") or "")

    if (
        narrative_state == "CONTRADICTED_OR_NEGATIVE"
        or distribution >= 65
        or "INVENTORY_RELEASE" in inventory_cycle_phase
        or "INVENTORY_RELEASE" in broker_shift
    ):
        return "SMART_MONEY_DISTRIBUTION"
    if crowding >= 68 or narrative_state == "PUBLIC_HYPE_CROWDED" or retail_stage == "EUPHORIA":
        return "RETAIL_EUPHORIA"
    if np.isfinite(story) and story >= 58 and (not np.isfinite(flow) or flow < 52):
        return "STORY_LEADS_FLOW"
    if np.isfinite(flow) and flow >= 62 and coverage < 30:
        return "FLOW_LEADS_NARRATIVE"
    if np.isfinite(flow) and flow >= 62 and np.isfinite(story) and story >= 60:
        return "MOMENTUM_TRIGGERED" if stage == "MARKUP" else "EARLY_CONVERGENCE"

    # Do not call an advanced markup 'silent accumulation' merely because recent OHLCV is firm.
    # Without narrative-flow convergence, require a pullback/base or a verified breakout-retest.
    if stage == "MARKUP" and (
        (np.isfinite(inventory) and inventory >= 58)
        or (np.isfinite(flow) and flow >= 58)
    ):
        return "MARKUP_REACCUMULATION_REQUIRED"

    collection_stage = stage in {"SILENT_ACCUMULATION", "BASE_TRANSITION"}
    bottoming_cycle = inventory_cycle_phase == "BOTTOMING_TO_COLLECTION_PROXY"
    if (np.isfinite(inventory) and inventory >= 62 and (collection_stage or bottoming_cycle)) or (
        np.isfinite(flow) and flow >= 56 and collection_stage
    ):
        return "INVENTORY_COLLECTION"
    return "NO_EDGE"


def build_execution_plan(
    features: Mapping[str, Any], ready: bool, lifecycle: str, orderbook: Mapping[str, Any] | None = None,
    *, auto_eod_ready: bool = False,
) -> dict[str, Any]:
    """Build two independent execution paths and one unambiguous selected path.

    Plan A is an accumulation/pullback plan. Plan B is a breakout/retest plan.  A target
    from Plan A is never evaluated against the breakout trigger from Plan B.  Legacy
    ``entry_low/entry_high/stop_loss/tp1/tp2`` fields are retained as Plan A for backward
    compatibility; all actionable UI/export code should use the ``execution_*`` fields.
    """
    orderbook = orderbook or {}
    close = _finite(features.get("last_price"), np.nan)
    atr = _finite(features.get("atr14"), np.nan)
    ema20 = _finite(features.get("ema20"), np.nan)
    high20 = _finite(features.get("high20"), np.nan)
    low20 = _finite(features.get("low20"), np.nan)
    if not all(np.isfinite(value) for value in (close, atr, ema20, high20, low20)) or atr <= 0:
        return {
            "execution_state": "NO_VALID_SCENARIO",
            "execution_geometry_valid": False,
            "execution_geometry_state": "MISSING_PRICE_OR_ATR_INPUT",
            "preferred_execution_path": "NONE",
        }

    precise_trigger = _finite(orderbook.get("precise_trigger_price"), np.nan)
    orderbook_verified = str(orderbook.get("orderbook_provenance_state") or "") == "DIRECT_SOURCE_VERIFIED"
    if lifecycle == "MOMENTUM_TRIGGERED":
        trigger = round_idx(precise_trigger if orderbook_verified and np.isfinite(precise_trigger) else max(close, high20), "up")
        entry_low = round_idx(max(ema20, trigger - 0.65 * atr), "down")
        entry_high = round_idx(trigger + 0.20 * atr, "up")
    else:
        trigger = round_idx(precise_trigger if orderbook_verified and np.isfinite(precise_trigger) else high20, "up")
        entry_low = round_idx(max(low20, ema20 - 0.45 * atr), "down")
        entry_high = round_idx(min(trigger, max(close, ema20 + 0.20 * atr)), "up")
    if entry_high < entry_low:
        entry_low, entry_high = entry_high, entry_low

    def _rr(entry: float, target: float, stop_value: float) -> float:
        entry_risk = entry - stop_value
        if not (np.isfinite(entry) and np.isfinite(target) and np.isfinite(stop_value)) or entry_risk <= 0:
            return np.nan
        return (target - entry) / entry_risk

    # Plan A — accumulation/pullback. The breakout trigger is NOT part of this geometry.
    entry_mid = (entry_low + entry_high) / 2
    structure_stop = round_idx(min(low20, entry_low - 1.10 * atr), "down")
    hard_stop_5pct = round_idx(entry_mid * 0.95, "down")
    stop = max(structure_stop, hard_stop_5pct)
    if stop >= entry_low:
        stop = round_idx(entry_low - idx_tick(entry_low), "down")
    risk = entry_mid - stop
    if risk <= 0:
        return {
            "execution_state": "NO_VALID_SCENARIO",
            "execution_geometry_valid": False,
            "execution_geometry_state": "ACCUMULATION_RISK_NON_POSITIVE",
            "preferred_execution_path": "NONE",
        }
    tp1 = round_idx(entry_mid + 1.8 * risk, "up")
    tp2 = round_idx(entry_mid + 3.0 * risk, "up")
    accumulation_valid = bool(stop < entry_low <= entry_high < tp1 < tp2)
    if not accumulation_valid:
        return {
            "execution_state": "NO_VALID_SCENARIO",
            "execution_geometry_valid": False,
            "execution_geometry_state": "ACCUMULATION_GEOMETRY_INVALID",
            "preferred_execution_path": "NONE",
        }

    accum_rr_tp1_entry_low = _rr(entry_low, tp1, stop)
    accum_rr_tp2_entry_low = _rr(entry_low, tp2, stop)
    accum_rr_tp1_entry_high = _rr(entry_high, tp1, stop)
    accum_rr_tp2_entry_high = _rr(entry_high, tp2, stop)

    # Plan B — breakout/retest, with its own stop and targets.
    breakout_entry = trigger
    tick = idx_tick(max(breakout_entry, 1.0))
    breakout_structure_stop = min(
        entry_high - tick,
        breakout_entry - max(1.25 * atr, 0.02 * breakout_entry),
    )
    breakout_stop = round_idx(max(breakout_structure_stop, breakout_entry * 0.94), "down")
    if breakout_stop >= breakout_entry:
        breakout_stop = round_idx(breakout_entry - tick, "down")
    breakout_risk = breakout_entry - breakout_stop
    breakout_tp1 = round_idx(breakout_entry + 1.8 * breakout_risk, "up")
    breakout_tp2 = round_idx(breakout_entry + 3.0 * breakout_risk, "up")
    breakout_valid = bool(breakout_risk > 0 and breakout_stop < breakout_entry < breakout_tp1 < breakout_tp2)

    accum_rr1 = _rr(entry_high, tp1, stop)
    accum_rr2 = _rr(entry_high, tp2, stop)
    breakout_rr1 = _rr(breakout_entry, breakout_tp1, breakout_stop) if breakout_valid else np.nan
    breakout_rr2 = _rr(breakout_entry, breakout_tp2, breakout_stop) if breakout_valid else np.nan
    accumulation_min_rr_pass = bool(accumulation_valid and np.isfinite(accum_rr1) and accum_rr1 >= 1.5)
    breakout_min_rr_pass = bool(breakout_valid and np.isfinite(breakout_rr1) and breakout_rr1 >= 1.8)

    # Select exactly one path for actionable presentation. Inventory collection prefers
    # accumulation; momentum/markup regimes prefer breakout/retest. If the preferred regime
    # path misses its RR floor, choose the alternate valid path rather than mixing fields.
    accumulation_lifecycles = {"INVENTORY_COLLECTION", "FLOW_LEADS_NARRATIVE", "STORY_LEADS_FLOW"}
    breakout_lifecycles = {"MOMENTUM_TRIGGERED", "MARKUP_REACCUMULATION_REQUIRED", "EARLY_CONVERGENCE"}
    if lifecycle in accumulation_lifecycles and accumulation_min_rr_pass:
        preferred = "ACCUMULATION_PULLBACK"
    elif lifecycle in breakout_lifecycles and breakout_min_rr_pass:
        preferred = "BREAKOUT_RETEST"
    elif breakout_min_rr_pass:
        preferred = "BREAKOUT_RETEST"
    elif accumulation_min_rr_pass:
        preferred = "ACCUMULATION_PULLBACK"
    elif breakout_valid:
        preferred = "BREAKOUT_RETEST"
    elif accumulation_valid:
        preferred = "ACCUMULATION_PULLBACK"
    else:
        preferred = "NONE"

    if preferred == "BREAKOUT_RETEST":
        execution_entry_low = execution_entry_high = breakout_entry
        execution_entry_reference = breakout_entry
        execution_trigger = breakout_entry
        execution_stop = breakout_stop
        execution_tp1 = breakout_tp1
        execution_tp2 = breakout_tp2
        execution_rr1 = breakout_rr1
        execution_rr2 = breakout_rr2
        selected_min_rr_pass = breakout_min_rr_pass
        selected_geometry_valid = breakout_valid
    elif preferred == "ACCUMULATION_PULLBACK":
        execution_entry_low, execution_entry_high = entry_low, entry_high
        execution_entry_reference = entry_high  # conservative RR reference
        execution_trigger = np.nan  # no breakout trigger belongs to Plan A
        execution_stop = stop
        execution_tp1 = tp1
        execution_tp2 = tp2
        execution_rr1 = accum_rr1
        execution_rr2 = accum_rr2
        selected_min_rr_pass = accumulation_min_rr_pass
        selected_geometry_valid = accumulation_valid
    else:
        execution_entry_low = execution_entry_high = execution_entry_reference = np.nan
        execution_trigger = execution_stop = execution_tp1 = execution_tp2 = np.nan
        execution_rr1 = execution_rr2 = np.nan
        selected_min_rr_pass = False
        selected_geometry_valid = False

    selected_geometry_valid = bool(
        selected_geometry_valid
        and np.isfinite(execution_entry_reference)
        and np.isfinite(execution_stop)
        and np.isfinite(execution_tp1)
        and np.isfinite(execution_tp2)
        and execution_stop < execution_entry_reference < execution_tp1 < execution_tp2
    )

    stop_distance_pct = 100 * risk / entry_mid
    if ready and orderbook_verified:
        execution_state = "EMIR_PRECISE_TRIGGER_READY"
    elif auto_eod_ready:
        execution_state = "AUTO_EOD_PROXY_TRIGGER_READY"
    elif ready:
        execution_state = "THESIS_READY_WAIT_DIRECT_BID_OFFER_TRIGGER"
    else:
        execution_state = "RESEARCH_SCENARIO_ONLY"

    return {
        "execution_state": execution_state,
        "execution_plan_semantics": "DUAL_PLAN_SEPARATED_V2",
        # Plan A (accumulation). Legacy aliases remain Plan A only.
        "entry_low": entry_low, "entry_high": entry_high, "stop_loss": stop, "tp1": tp1, "tp2": tp2,
        "accumulation_entry_low": entry_low, "accumulation_entry_high": entry_high,
        "accumulation_stop_loss": stop, "accumulation_tp1": tp1, "accumulation_tp2": tp2,
        "accumulation_geometry_valid": accumulation_valid,
        "accumulation_rr_tp1_at_entry_low": round(accum_rr_tp1_entry_low, 2) if np.isfinite(accum_rr_tp1_entry_low) else np.nan,
        "accumulation_rr_tp2_at_entry_low": round(accum_rr_tp2_entry_low, 2) if np.isfinite(accum_rr_tp2_entry_low) else np.nan,
        "accumulation_rr_tp1_at_entry_high": round(accum_rr_tp1_entry_high, 2) if np.isfinite(accum_rr_tp1_entry_high) else np.nan,
        "accumulation_rr_tp2_at_entry_high": round(accum_rr_tp2_entry_high, 2) if np.isfinite(accum_rr_tp2_entry_high) else np.nan,
        # Backward-compatible RR aliases.
        "rr_tp1": round(_rr(entry_mid, tp1, stop), 2), "rr_tp2": round(_rr(entry_mid, tp2, stop), 2),
        "rr_tp1_at_entry_low": round(accum_rr_tp1_entry_low, 2) if np.isfinite(accum_rr_tp1_entry_low) else np.nan,
        "rr_tp2_at_entry_low": round(accum_rr_tp2_entry_low, 2) if np.isfinite(accum_rr_tp2_entry_low) else np.nan,
        "rr_tp1_at_entry_high": round(accum_rr_tp1_entry_high, 2) if np.isfinite(accum_rr_tp1_entry_high) else np.nan,
        "rr_tp2_at_entry_high": round(accum_rr_tp2_entry_high, 2) if np.isfinite(accum_rr_tp2_entry_high) else np.nan,
        # Informational breakout trigger lives only with Plan B.
        "trigger": trigger,
        "breakout_entry": breakout_entry if breakout_valid else np.nan,
        "breakout_stop_loss": breakout_stop if breakout_valid else np.nan,
        "breakout_tp1": breakout_tp1 if breakout_valid else np.nan,
        "breakout_tp2": breakout_tp2 if breakout_valid else np.nan,
        "breakout_geometry_valid": breakout_valid,
        "breakout_rr_tp1": round(breakout_rr1, 2) if np.isfinite(breakout_rr1) else np.nan,
        "breakout_rr_tp2": round(breakout_rr2, 2) if np.isfinite(breakout_rr2) else np.nan,
        # Single selected path for Real Money/UI/export. These fields never mix plans.
        "preferred_execution_path": preferred,
        "execution_entry_low": execution_entry_low,
        "execution_entry_high": execution_entry_high,
        "execution_entry_reference": execution_entry_reference,
        "execution_trigger": execution_trigger,
        "execution_stop_loss": execution_stop,
        "execution_tp1": execution_tp1,
        "execution_tp2": execution_tp2,
        "execution_rr_tp1": round(execution_rr1, 2) if np.isfinite(execution_rr1) else np.nan,
        "execution_rr_tp2": round(execution_rr2, 2) if np.isfinite(execution_rr2) else np.nan,
        "execution_geometry_valid": selected_geometry_valid,
        "execution_min_rr_pass": bool(selected_min_rr_pass),
        "execution_geometry_state": "VALID_SELECTED_PATH" if selected_geometry_valid else "NO_VALID_SELECTED_PATH",
        "hard_stop_distance_pct": round(stop_distance_pct, 2),
        "risk_doctrine_state": "SEPARATE_ACCUMULATION_VS_BREAKOUT_RISK_GEOMETRY_V2",
        "trigger_provenance": "DIRECT_BID_OFFER_EVIDENCE" if orderbook_verified else "OHLCV_EOD_MICROSTRUCTURE_PROXY" if auto_eod_ready else "OHLCV_STRUCTURE_PROXY_WAIT_DIRECT_BID_OFFER",
    }

def build_emir_profile(
    *, ticker: str, features: Mapping[str, Any], narrative: Mapping[str, Any] | None = None,
    broker: Mapping[str, Any] | None = None, ownership: Mapping[str, Any] | None = None,
    orderbook: Mapping[str, Any] | None = None, market: Mapping[str, Any] | None = None,
    sector: Mapping[str, Any] | None = None, integrity: Mapping[str, Any] | None = None,
    fundamental: Mapping[str, Any] | None = None,
    future_fundamental: Mapping[str, Any] | None = None,
    outcome_calibration_map: Mapping[str, Mapping[str, Any]] | None = None,
    deep_reviewed: bool = True, max_position_cap_pct: float = 20.0,
    capital_idr: float = 5_000_000.0, risk_budget_pct: float = 1.0,
    calibration_mode: str = "SHADOW_ONLY",
    capital_mode: str = "RESEARCH",
) -> dict[str, Any]:
    narrative, broker, ownership, orderbook, market, sector, integrity, fundamental, future_fundamental = map(dict, (
        narrative or {}, broker or {}, ownership or {}, orderbook or {}, market or {}, sector or {}, integrity or {}, fundamental or {}, future_fundamental or {},
    ))
    ohlcv_flow = _finite(features.get("smart_money_score"), np.nan)
    ohlcv_flow_coverage = _finite(features.get("smart_money_coverage_pct"), 0)
    broker_inventory = _finite(broker.get("broker_inventory_score"), np.nan)
    broker_coverage = _finite(broker.get("broker_inventory_coverage_pct"), 0)
    flow_score, flow_coverage = _weighted_fixed([
        (ohlcv_flow, 0.60, ohlcv_flow_coverage),
        (broker_inventory, 0.40, broker_coverage),
    ])
    if broker_coverage == 0 and np.isfinite(ohlcv_flow):
        flow_score = ohlcv_flow
        flow_coverage = 60 * ohlcv_flow_coverage / 100

    narrative_score = _finite(narrative.get("narrative_score"), np.nan)
    narrative_coverage = _finite(narrative.get("narrative_coverage_pct"), 0)
    runway = _finite(narrative.get("story_runway_score"), np.nan)
    top_down = _finite(narrative.get("top_down_catalyst_score"), np.nan)
    industry_translation = _finite(narrative.get("industry_translation_score"), np.nan)
    narrative_runway_score, narrative_runway_coverage = _weighted_fixed([
        (narrative_score, 0.40, narrative_coverage),
        (runway, 0.25, narrative_coverage if np.isfinite(runway) else 0),
        (top_down, 0.20, narrative_coverage if np.isfinite(top_down) else 0),
        (industry_translation, 0.15, narrative_coverage if np.isfinite(industry_translation) else 0),
    ])
    narrative_conversion = _finite(narrative.get("financial_conversion_score"), np.nan)
    fundamental_conversion = _finite(fundamental.get("fundamental_conversion_score"), np.nan)
    fundamental_coverage = _finite(fundamental.get("fundamental_coverage_pct"), 0)
    fundamental_data_quality = _finite(fundamental.get("fundamental_data_quality_score"), 0)
    # Fundamental conversion already owns a 16% pillar below.  Older versions
    # reused 40% of it inside this 8% narrative-conversion pillar, silently
    # adding another 3.2 percentage points of the same evidence.  Keep this
    # pillar narrative-only; missing narrative conversion reduces coverage and
    # never causes the fundamental score to be counted twice.
    conversion = narrative_conversion if np.isfinite(narrative_conversion) else np.nan
    conversion_coverage = narrative_coverage if np.isfinite(narrative_conversion) else 0.0
    alignment = _finite(narrative.get("issuer_alignment_score"), np.nan)
    alignment_coverage = _finite(narrative.get("issuer_alignment_coverage_pct"), 0)
    ownership_score = _finite(ownership.get("ownership_score"), np.nan)
    ownership_coverage = _finite(ownership.get("ownership_coverage_pct"), 0)
    alignment_owner_score, alignment_owner_coverage = _weighted_fixed([
        (alignment, 0.55, alignment_coverage),
        (ownership_score, 0.45, ownership_coverage),
    ])
    structure_score = _finite(features.get("market_structure_score"), np.nan)
    structure_coverage = 100.0 if np.isfinite(structure_score) else 0.0
    sector_score = _finite(sector.get("sector_leadership_score"), np.nan)
    sector_coverage = _finite(sector.get("sector_context_coverage_pct"), 0)
    market_score = _finite(market.get("market_context_score"), np.nan)
    market_coverage = _finite(market.get("market_context_coverage_pct"), 0)
    sector_context_score, sector_context_coverage = _weighted_fixed([
        (sector_score, 0.65, sector_coverage),
        (market_score, 0.35, market_coverage),
    ])
    orderbook_score = _finite(orderbook.get("orderbook_trigger_score"), np.nan)
    orderbook_coverage = _finite(orderbook.get("orderbook_coverage_pct"), 0)
    trend = _finite(features.get("trend_score"), np.nan)
    liquidity = _finite(features.get("liquidity_score"), np.nan)
    effective_float = _finite(ownership.get("effective_free_float_pct"), np.nan)
    float_quality = 80 if np.isfinite(effective_float) and 10 <= effective_float <= 35 else 55 if np.isfinite(effective_float) and 5 <= effective_float <= 50 else np.nan
    liquidity_float_score, liquidity_float_coverage = _weighted_fixed([
        (liquidity, 0.65, 100 if np.isfinite(liquidity) else 0),
        (float_quality, 0.35, ownership_coverage if np.isfinite(float_quality) else 0),
    ])
    integrity_score = _finite(integrity.get("idx_integrity_score"), np.nan)
    integrity_coverage = _finite(integrity.get("idx_integrity_coverage_pct"), 0)

    # Independent fixed-denominator implementation; not an official CAK formula.
    # v1.7.0 methodology hardening gives business/future-fundamental quality its own
    # scoring pillar instead of letting it contribute only through narrative conversion.
    # This is intentionally closer to the public macro/sector -> business conversion ->
    # narrative/flow -> execution sequence while preserving flow as the largest single pillar.
    raw_score, evidence_coverage = _weighted_fixed([
        (flow_score, 0.18, flow_coverage),
        (fundamental_conversion, 0.16, fundamental_coverage),
        (narrative_runway_score, 0.14, narrative_runway_coverage),
        (structure_score, 0.13, structure_coverage),
        (sector_context_score, 0.10, sector_context_coverage),
        (conversion, 0.08, conversion_coverage),
        (alignment_owner_score, 0.07, alignment_owner_coverage),
        (integrity_score, 0.05, integrity_coverage),
        (orderbook_score, 0.04, orderbook_coverage),
        (trend, 0.03, 100 if np.isfinite(trend) else 0),
        (liquidity_float_score, 0.02, liquidity_float_coverage),
    ])
    distribution = _finite(features.get("distribution_score"), 0)
    broker_shift = str(broker.get("broker_inventory_shift_state") or "")
    if broker_shift == "DISTRIBUTION_DOMINANT" or "INVENTORY_RELEASE" in broker_shift:
        distribution = max(distribution, 70)
    crowding = _finite(features.get("crowding_score"), 0)
    cannibalisation = _finite(broker.get("retail_cannibalisation_risk"), 0)
    friction = _finite(features.get("execution_friction_score"), 0)
    negative = "NEGATIVE" in str(narrative.get("narrative_state") or "") or "CONTRADICTED" in str(narrative.get("narrative_state") or "")
    penalty = (
        0.18 * distribution + 0.13 * max(0.0, crowding - 55)
        + 0.08 * cannibalisation + 0.06 * max(0.0, friction - 45)
        + (12 if negative else 0)
    )
    conviction = _clip(raw_score - penalty) if np.isfinite(raw_score) else np.nan
    lifecycle = classify_lifecycle(features, narrative, broker)
    market_regime = str(market.get("market_regime") or "MARKET_CONTEXT_UNAVAILABLE")
    structure_mode = str(features.get("market_structure_mode") or "NO_CLEAR_STRUCTURE")
    sector_rrg_state = str(sector.get("sector_rrg_state") or sector.get("sector_state") or "UNKNOWN")
    sector_relative_strength = _finite(sector.get("sector_relative_strength_pct"), np.nan)
    sector_strength_momentum = _finite(sector.get("sector_strength_momentum_pct"), np.nan)
    risk_off_sector_leader_exception = bool(
        market_regime == "RISK_OFF"
        and sector_rrg_state == "LEADING"
        and _finite(sector_score, 0) >= 65
        and _finite(sector_relative_strength, -999) > 0
        and _finite(sector_strength_momentum, -999) >= 0
    )
    market_allows_thesis = bool(market_regime != "RISK_OFF" or risk_off_sector_leader_exception)
    orderbook_provenance = str(orderbook.get("orderbook_provenance_state") or "")
    orderbook_verified = orderbook_provenance == "DIRECT_SOURCE_VERIFIED"
    orderbook_proxy = orderbook_provenance == "OHLCV_EOD_MICROSTRUCTURE_PROXY_NOT_LIVE_DEPTH"
    integrity_provenance = str(integrity.get("idx_integrity_provenance_state") or "")
    integrity_verified = integrity_provenance == "DIRECT_SOURCE_VERIFIED"
    integrity_auto_public = integrity_provenance in {
        "AUTO_PUBLIC_KSEI_PROXY", "AUTO_PUBLIC_KSEI_PARTIAL_PROXY", "AUTO_PUBLIC_KSEI_AND_REGULATORY_NEWS"
    }
    integrity_hard_block = bool(integrity.get("idx_integrity_hard_block", True))
    integrity_ready = bool(integrity_verified and integrity_coverage >= 70 and not integrity_hard_block)
    integrity_unknown_count = int(_finite(integrity.get("idx_integrity_unknown_critical_count"), 99))
    integrity_auto_ready = bool(
        integrity_auto_public and integrity_coverage >= 70 and integrity_unknown_count == 0 and not integrity_hard_block
    )
    ohlcv_state = str(features.get("ohlcv_integrity_state") or "UNKNOWN")
    corporate_action_anomaly = bool(features.get("corporate_action_anomaly_flag", False))
    corporate_action_cleared = bool(integrity.get("corporate_action_review_cleared", False))
    data_integrity_block = bool(
        str(features.get("feature_state") or "") != "OK"
        or ohlcv_state in {"STALE_DATA_BLOCK", "INSUFFICIENT_HISTORY", "ILLIQUID_OR_SUSPENDED_PATTERN"}
        or (corporate_action_anomaly and not corporate_action_cleared)
    )

    calibration = select_outcome_calibration(outcome_calibration_map, lifecycle, structure_mode)
    calibration_state = str(calibration.get("outcome_calibration_state") or "NO_OUTCOME_MEMORY")
    guarded = str(calibration_mode or "SHADOW_ONLY").upper() == "GUARDED"
    calibration_block = bool(guarded and calibration_state == "EMPIRICAL_EDGE_REJECTED")

    core_thesis_ready = bool(
        deep_reviewed and np.isfinite(conviction) and conviction >= 62 and evidence_coverage >= 56
        and fundamental_coverage >= 45 and _finite(fundamental_conversion, 0) >= 55 and fundamental_data_quality >= 75
        and narrative_coverage >= 30 and _finite(conversion, 0) >= 45
        and (
            int(narrative.get("narrative_verified_source_count", 0) or 0) >= 1
            or int(narrative.get("narrative_independent_story_count", 0) or 0) >= 2
        )
        and flow_coverage >= 52 and _finite(structure_score, 0) >= 54
        and _finite(liquidity, 0) >= 45 and distribution < 52 and crowding < 70
        and friction < 72 and lifecycle in {"EARLY_CONVERGENCE", "MOMENTUM_TRIGGERED"}
        and structure_mode != "NO_CLEAR_STRUCTURE" and market_allows_thesis
        and not data_integrity_block and not calibration_block
    )
    auto_core_thesis_ready = bool(
        deep_reviewed and np.isfinite(conviction) and conviction >= 55 and evidence_coverage >= 50
        and fundamental_coverage >= 35 and _finite(fundamental_conversion, 0) >= 55 and fundamental_data_quality >= 75
        and narrative_coverage >= 30 and _finite(conversion, 0) >= 45
        and (
            int(narrative.get("narrative_verified_source_count", 0) or 0) >= 1
            or int(narrative.get("narrative_independent_story_count", 0) or 0) >= 2
        )
        and flow_coverage >= 45 and _finite(structure_score, 0) >= 54
        and _finite(liquidity, 0) >= 45 and distribution < 50 and crowding < 68
        and friction < 68 and lifecycle in {"EARLY_CONVERGENCE", "MOMENTUM_TRIGGERED"}
        and structure_mode != "NO_CLEAR_STRUCTURE" and market_allows_thesis
        and not data_integrity_block and not calibration_block
    )

    # Execution-capacity overlay. Small accounts should not be falsely blocked, while thin stocks remain constrained.
    adtv20 = _finite(features.get("adtv20_idr"), 0.0)
    capital = max(0.0, _finite(capital_idr, 0.0))
    requested_cap = max(0.0, min(_finite(max_position_cap_pct, 20.0), 30.0))
    capital_mode_pre = str(capital_mode or "RESEARCH").upper().strip()
    if capital_mode_pre == "GUARDED_REAL_MONEY":
        requested_cap = min(requested_cap, 5.0 if market_regime == "RISK_OFF" else 7.5)
    preliminary_cap = 20.0 if np.isfinite(conviction) and conviction >= 78 and liquidity >= 70 else 12.0 if np.isfinite(conviction) and conviction >= 70 else 7.0
    capacity_target_cap = min(requested_cap, preliminary_cap)
    requested_value = capital * capacity_target_cap / 100.0
    max_participation_pct = 2.5 if liquidity >= 85 else 1.5 if liquidity >= 70 else 1.0 if liquidity >= 55 else 0.5
    max_safe_position_value = adtv20 * max_participation_pct / 100.0 if adtv20 > 0 else 0.0
    participation_pct = 100.0 * requested_value / adtv20 if adtv20 > 0 else np.inf
    tick_bps = 10000.0 * idx_tick(_finite(features.get("last_price"), 0.0)) / max(_finite(features.get("last_price"), 0.0), 1.0)
    slippage_bps = _clip(0.25 * tick_bps + 10.0 * math.sqrt(max(0.0, _finite(participation_pct, 0.0))) + 0.35 * _finite(features.get("gap_risk_score"), 0), 0, 500)
    capacity_block = bool(adtv20 <= 0 or requested_value > max_safe_position_value or slippage_bps > 220)
    if capital <= 0:
        capacity_state = "CAPITAL_NOT_CONFIGURED"
    elif capacity_block:
        capacity_state = "EXECUTION_CAPACITY_BLOCK"
    elif participation_pct > 0.5 * max_participation_pct:
        capacity_state = "EXECUTION_CAPACITY_CAUTION"
    else:
        capacity_state = "EXECUTION_CAPACITY_OK"

    thesis_ready = bool(core_thesis_ready or auto_core_thesis_ready)
    auto_eod_ready = bool(
        auto_core_thesis_ready and integrity_auto_ready and orderbook_proxy and orderbook_coverage >= 45
        and _finite(orderbook_score, 0) >= 52 and broker_coverage >= 45
        and fundamental_coverage >= 35 and not capacity_block
    )
    precise_ready = bool(
        core_thesis_ready and integrity_ready and orderbook_verified and orderbook_coverage >= 60
        and _finite(orderbook_score, 0) >= 55 and not capacity_block
    )

    # Known integrity failures must never be hidden behind the radar-only state.
    if integrity_verified and integrity_hard_block:
        state, action = "EMIR_REJECT_IDX_INTEGRITY", "REJECT_OR_MANUAL_REGULATORY_REVIEW"
    elif data_integrity_block:
        state, action = "EMIR_DATA_INTEGRITY_BLOCK", "REFRESH_OR_CLEAR_CORPORATE_ACTION_DATA"
    elif not deep_reviewed:
        state, action = "EMIR_RADAR_ONLY_NOT_DEEP_REVIEWED", "DEEP_REVIEW_REQUIRED"
    elif calibration_block:
        state, action = "EMIR_CALIBRATION_REJECTED", "RESEARCH_ONLY_UNTIL_EDGE_RECOVERS"
    elif precise_ready:
        state, action = "EMIR_READY_WITH_PRECISE_TRIGGER", "PLAN_ENTRY_ON_VERIFIED_BID_OFFER_TRIGGER"
    elif auto_eod_ready:
        state, action = "EMIR_AUTO_EOD_READY", "PLAN_CAPPED_ENTRY_ON_EOD_PROXY_TRIGGER"
    elif thesis_ready and not (integrity_ready or integrity_auto_ready):
        state, action = "EMIR_CORE_THESIS_READY_WAIT_IDX_INTEGRITY", "VERIFY_HSC_BOARD_FREE_FLOAT_AND_CORPORATE_ACTION"
    elif thesis_ready:
        state, action = "EMIR_THESIS_READY_WAIT_BID_OFFER", "WAIT_DIRECT_BID_OFFER_TRIGGER"
    elif fundamental_coverage < 35:
        state, action = "EMIR_FUNDAMENTAL_EVIDENCE_PENDING", "COMPLETE_FUNDAMENTAL_EVIDENCE"
    elif _finite(fundamental_conversion, 0) < 55:
        state, action = "EMIR_WAIT_FUNDAMENTAL_CONVERSION", "WAIT_BUSINESS_AND_CASHFLOW_CONVERSION"
    elif fundamental_data_quality < 75:
        state, action = "EMIR_FUNDAMENTAL_QUALITY_PENDING", "VERIFY_OR_IMPROVE_FUNDAMENTAL_DATA_QUALITY"
    elif lifecycle == "INVENTORY_COLLECTION":
        state, action = "EMIR_WATCH_INVENTORY_COLLECTION", "WATCH_SMART_MONEY_COLLECTION"
    elif lifecycle == "MARKUP_REACCUMULATION_REQUIRED":
        state, action = "EMIR_WAIT_REACCUMULATION", "WAIT_PULLBACK_OR_BREAKOUT_RETEST_NOT_CHASE"
    elif lifecycle == "FLOW_LEADS_NARRATIVE":
        state, action = "EMIR_WAIT_NARRATIVE", "WAIT_STORY_CONFIRMATION"
    elif lifecycle == "STORY_LEADS_FLOW":
        state, action = "EMIR_WAIT_MONEY_FLOW", "WAIT_FLOW_CONFIRMATION"
    elif lifecycle == "RETAIL_EUPHORIA":
        state, action = "EMIR_AVOID_RETAIL_EUPHORIA", "DO_NOT_CHASE_OR_TRIM"
    elif lifecycle == "SMART_MONEY_DISTRIBUTION":
        state, action = "EMIR_REJECT_SMART_MONEY_DISTRIBUTION", "REJECT_OR_EXIT_WATCH"
    elif evidence_coverage < 56:
        state, action = "EMIR_EVIDENCE_PENDING", "COMPLETE_DIRECT_EVIDENCE"
    else:
        state, action = "EMIR_NO_EDGE_YET", "WAIT_FOR_EDGE"

    risks: list[str] = []
    if distribution >= 45:
        risks.append("DISTRIBUTION_RISK")
    if crowding >= 65:
        risks.append("RETAIL_CROWDING_RISK")
    if narrative_coverage < 35:
        risks.append("NARRATIVE_EVIDENCE_WEAK")
    if int(narrative.get("narrative_verified_source_count", 0) or 0) < 1:
        risks.append("NO_DIRECTLY_VERIFIED_NARRATIVE_SOURCE")
    if fundamental_coverage < 35:
        risks.append("FUNDAMENTAL_PUBLIC_DATA_WEAK_OR_MISSING")
    elif _finite(fundamental_conversion, 0) < 55:
        risks.append("BUSINESS_OR_FUTURE_FUNDAMENTAL_CONVERSION_WEAK")
    if alignment_coverage < 35:
        risks.append("ISSUER_ALIGNMENT_UNPROVEN")
    if _finite(conversion, 0) < 45:
        risks.append("FUTURE_FUNDAMENTAL_CONVERSION_UNPROVEN")
    if broker_coverage < 40:
        risks.append("MULTI_PERIOD_BROKER_INVENTORY_UNPROVEN")
    if not orderbook_verified:
        risks.append("DIRECT_BID_OFFER_TRIGGER_MISSING_PROXY_USED" if orderbook_proxy else "DIRECT_BID_OFFER_TRIGGER_MISSING")
    if integrity_unknown_count > 0:
        risks.append("IDX_CRITICAL_FIELDS_UNKNOWN_NOT_VERIFIED")
    if not integrity_verified:
        risks.append("IDX_DIRECT_INTEGRITY_MISSING_AUTO_PUBLIC_PROXY_USED" if integrity_auto_ready else "IDX_INTEGRITY_EVIDENCE_MISSING")
    elif integrity_hard_block:
        risks.append("IDX_INTEGRITY_HARD_BLOCK")
    if data_integrity_block:
        risks.append("OHLCV_OR_CORPORATE_ACTION_INTEGRITY_BLOCK")
    if market_regime == "RISK_OFF":
        risks.append("MARKET_RISK_OFF_SECTOR_LEADER_EXCEPTION_POSITION_CAPPED" if risk_off_sector_leader_exception else "MARKET_RISK_OFF")
    if _finite(liquidity, 0) < 45:
        risks.append("LIQUIDITY_RISK")
    if cannibalisation >= 60:
        risks.append("RETAIL_CANNIBALISATION_RISK")
    if capacity_block:
        risks.append("EXECUTION_CAPACITY_OR_SLIPPAGE_BLOCK")
    if calibration_state == "EMPIRICAL_EDGE_REJECTED":
        risks.append("OUTCOME_MEMORY_EDGE_REJECTED")

    base_cap = 0.0
    if precise_ready:
        base_cap = 20.0 if conviction >= 78 and liquidity >= 70 else 12.0 if conviction >= 70 else 7.0
    elif auto_eod_ready:
        base_cap = 8.0 if conviction >= 75 and liquidity >= 70 else 5.0
    liquidity_cap_pct = 100.0 * max_safe_position_value / capital if capital > 0 else 0.0
    position_cap = min(max(0.0, requested_cap), base_cap, max(0.0, liquidity_cap_pct)) if (precise_ready or auto_eod_ready) else 0.0
    if risk_off_sector_leader_exception and position_cap > 0:
        position_cap = min(position_cap, 5.0)

    why_now = (
        f"Lifecycle {lifecycle}; structure {structure_mode}; flow/inventory {round(_finite(flow_score, 0), 1)}; "
        f"story runway {round(_finite(narrative_runway_score, 0), 1)}; market {market_regime}; "
        f"sector {sector.get('sector_rrg_state', sector.get('sector_state', 'UNKNOWN'))}; "
        f"IDX integrity {integrity.get('idx_integrity_state', 'MISSING')}."
    )
    next_proof = {
        "EMIR_WAIT_NARRATIVE": "Butuh catalyst material, sumber resmi, dan jalur top-down → industri → revenue/margin/laba/cash flow/rerating.",
        "EMIR_WAIT_MONEY_FLOW": "Butuh akumulasi/absorption, inventory persistence, price acceptance, dan relative strength yang mengonfirmasi cerita.",
        "EMIR_WATCH_INVENTORY_COLLECTION": "Butuh narrative trigger sebelum retail adoption berubah menjadi euforia.",
        "EMIR_FUNDAMENTAL_EVIDENCE_PENDING": "Lengkapi laporan keuangan, growth YoY/TTM, profitabilitas, OCF/FCF dan solvabilitas sebelum flow dipromosikan menjadi thesis.",
        "EMIR_WAIT_FUNDAMENTAL_CONVERSION": "Money flow atau cerita belum cukup: tunggu revenue/earnings/margin/cash-flow conversion membuktikan business thesis.",
        "EMIR_WAIT_REACCUMULATION": "Harga sudah berada pada fase markup; jangan chase. Tunggu base/pullback sehat atau breakout-retest yang diterima market dengan flow tetap konstruktif.",
        "EMIR_CORE_THESIS_READY_WAIT_IDX_INTEGRITY": "Thesis lulus; verifikasi HSC, board/FCA, suspension/UMA, free float, dan corporate action sebelum precise trigger.",
        "EMIR_THESIS_READY_WAIT_BID_OFFER": "Thesis, flow, dan IDX integrity lulus; butuh direct bid-offer evidence untuk precise trigger.",
        "EMIR_AUTO_EOD_READY": "Gunakan trigger EOD proxy dengan position cap lebih rendah; upgrade ke precise trigger hanya setelah direct bid-offer dan IDX integrity terverifikasi.",
        "EMIR_EVIDENCE_PENDING": "Lengkapi narrative, multi-period broker inventory, ownership/free-float, IDX integrity, dan bid-offer evidence.",
        "EMIR_DATA_INTEGRITY_BLOCK": "Refresh OHLCV atau jelaskan corporate action/abnormal return dengan evidence resmi.",
    }.get(state, "Pertahankan convergence; bereaksi pada data dan invalidation, bukan memaksakan prediksi harga.")
    invalidation = "Thesis batal bila catalyst terkontradiksi, inventory berubah menjadi distribusi, IDX integrity memburuk, atau harga kehilangan struktur/stop."
    plan = build_execution_plan(features, precise_ready, lifecycle, orderbook, auto_eod_ready=auto_eod_ready)

    # Real-money hardening: EOD proxies may identify a candidate, but they never authorize
    # capital. Direct bid/offer + direct IDX integrity are required for scanner-side READY.
    capital_mode_state = str(capital_mode or "RESEARCH").upper()
    guarded_real_money = capital_mode_state == "GUARDED_REAL_MONEY"
    official_cov = _finite(fundamental.get("fundamental_official_source_coverage_pct"), 0)
    data_quality = fundamental_data_quality
    cashflow_state = str(fundamental.get("fundamental_cashflow_state") or "")
    cashflow_quality_state = str(fundamental.get("fundamental_cashflow_quality_state") or "").upper()
    # Backward-compatible classification for persisted v1.9.12 rows that predate the
    # explicit quality field. Availability must never be mistaken for quality.
    ocf_value = _finite(fundamental.get("operating_cash_flow_ttm"), np.nan)
    if not np.isfinite(ocf_value):
        ocf_value = _finite(fundamental.get("operating_cash_flow_latest"), np.nan)
    fcf_value = _finite(fundamental.get("free_cash_flow_proxy_ttm"), np.nan)
    if not np.isfinite(fcf_value):
        fcf_value = _finite(fundamental.get("free_cash_flow_proxy_latest"), np.nan)
    ocf_conversion_value = _finite(fundamental.get("ocf_conversion_ratio"), np.nan)
    if not cashflow_quality_state:
        if not np.isfinite(ocf_value):
            cashflow_quality_state = "CASHFLOW_NOT_AVAILABLE"
        elif ocf_value < 0:
            cashflow_quality_state = "OCF_AND_FCF_NEGATIVE" if np.isfinite(fcf_value) and fcf_value < 0 else "OCF_NEGATIVE"
        elif np.isfinite(fcf_value) and fcf_value < 0:
            cashflow_quality_state = "FCF_NEGATIVE_CAPEX_REVIEW"
        elif np.isfinite(ocf_conversion_value) and ocf_conversion_value < 0.50:
            cashflow_quality_state = "CASHFLOW_POSITIVE_WEAK_CONVERSION"
        elif np.isfinite(ocf_conversion_value) and ocf_conversion_value >= 0.70:
            cashflow_quality_state = "CASHFLOW_POSITIVE_CONVERTING"
        else:
            cashflow_quality_state = "CASHFLOW_POSITIVE_PARTIAL"
    leverage_state = str(fundamental.get("fundamental_leverage_risk_state") or "UNKNOWN")
    freshness_state = str(fundamental.get("fundamental_period_freshness_state") or "UNKNOWN")
    growth_consistency_state = str(fundamental.get("fundamental_growth_consistency_state") or "YTD_NOT_AVAILABLE")
    verified_story_count = int(narrative.get("narrative_verified_source_count", 0) or 0)
    official_story_count = int(narrative.get("narrative_official_source_count", 0) or 0)
    manual_verified_story = bool(verified_story_count >= 1 or official_story_count >= 1)
    # v1.9.4: a diversified public narrative proxy can support MANUAL confirmation.
    # It never upgrades the candidate to DIRECT_VERIFIED_READY by itself.
    public_story_proxy_ok = bool(
        _finite(narrative.get("narrative_score"), 0) >= 42
        and _finite(narrative.get("narrative_coverage_pct"), 0) >= 60
        and int(narrative.get("narrative_event_count", 0) or 0) >= 5
        and int(narrative.get("narrative_independent_story_count", 0) or 0) >= 2
        and _finite(narrative.get("narrative_source_independence_score"), 0) >= 45
        and _finite(narrative.get("narrative_materiality_score"), 0) >= 28
        and _finite(narrative.get("narrative_contradiction_score"), 0) < 45
    )
    if manual_verified_story:
        narrative_evidence_tier = "DIRECT_OR_OFFICIAL_VERIFIED"
    elif public_story_proxy_ok:
        narrative_evidence_tier = "PUBLIC_PROXY_ACCEPTED_MANUAL"
    else:
        narrative_evidence_tier = "PUBLIC_PROXY_INSUFFICIENT"
    blockers=[]
    manual_conditions=[]
    if not deep_reviewed: blockers.append("NOT_DEEP_REVIEWED")
    # v1.9.1: official IDX/issuer evidence is a confidence upgrade, not a prerequisite
    # for a proxy-backed candidate. Proxy-only candidates can reach MANUAL_CONFIRMATION_REQUIRED
    # but never DIRECT_VERIFIED_READY.
    proxy_only = official_cov < 50
    if proxy_only: manual_conditions.append("PROXY_FUNDAMENTAL_MANUAL_VERIFY")
    if data_quality < 75: blockers.append("FUNDAMENTAL_DATA_QUALITY_LT_75")
    cashflow_missing = ("MISSING" in cashflow_state or not cashflow_state or cashflow_quality_state == "CASHFLOW_NOT_AVAILABLE")
    if cashflow_missing:
        manual_conditions.append("CASHFLOW_PROXY_OR_MANUAL_VERIFY")
    elif cashflow_quality_state in {"OCF_NEGATIVE", "OCF_AND_FCF_NEGATIVE"}:
        blockers.append("OPERATING_CASH_FLOW_NEGATIVE")
    elif cashflow_quality_state == "FCF_NEGATIVE_CAPEX_REVIEW":
        manual_conditions.append("FCF_NEGATIVE_CAPEX_REVIEW")
    elif cashflow_quality_state == "CASHFLOW_POSITIVE_WEAK_CONVERSION":
        manual_conditions.append("CASHFLOW_WEAK_CONVERSION_REVIEW")
    if freshness_state in {"STALE_QUARTERLY_PERIOD","UNKNOWN_PERIOD","LAGGING_REPORTING_PERIOD","STALE_RELATIVE_TO_UNIVERSE"}:
        blockers.append("FUNDAMENTAL_PERIOD_NOT_CURRENT")
    elif freshness_state == "AGING_QUARTERLY_PERIOD":
        manual_conditions.append("AGING_FUNDAMENTAL_PERIOD_MANUAL_VERIFY")
    if growth_consistency_state in {"TURNAROUND_INFLECTION_UNCONFIRMED","LATEST_QUARTER_DECELERATION_YTD_POSITIVE","QUARTER_YTD_DIVERGENCE_REVIEW"}:
        manual_conditions.append(growth_consistency_state)
    elif growth_consistency_state == "QUARTER_AND_YTD_WEAK":
        blockers.append("QUARTER_AND_YTD_WEAK")
    elif growth_consistency_state == "YTD_NOT_AVAILABLE":
        manual_conditions.append("YTD_GROWTH_NOT_AVAILABLE")
    if leverage_state in {"HIGH_LEVERAGE","EXTREME_LEVERAGE"}: blockers.append(leverage_state)
    if _finite(fundamental_conversion,0) < 55: blockers.append("FUNDAMENTAL_CONVERSION_LT_55")
    if not manual_verified_story:
        if public_story_proxy_ok:
            manual_conditions.append("PUBLIC_NARRATIVE_MANUAL_VERIFY")
        else:
            blockers.append("NO_VERIFIED_NARRATIVE_OR_OFFICIAL_EVENT")
    if market_regime == "RISK_OFF" and not risk_off_sector_leader_exception: blockers.append("MARKET_RISK_OFF")
    if _finite(liquidity,0) < 45: blockers.append("LIQUIDITY_LT_45")
    if distribution >= 40: blockers.append("DISTRIBUTION_GE_40")
    if crowding >= 65: blockers.append("CROWDING_GE_65")
    if friction >= 60: blockers.append("EXECUTION_FRICTION_GE_60")
    if _finite(structure_score,0) < 60: blockers.append("STRUCTURE_LT_60")
    if capacity_block: blockers.append("EXECUTION_CAPACITY_BLOCK")
    execution_geometry_valid = bool(plan.get("execution_geometry_valid", False))
    execution_min_rr_pass = bool(plan.get("execution_min_rr_pass", False))
    if not execution_geometry_valid:
        blockers.append("EXECUTION_GEOMETRY_INVALID")
    elif not execution_min_rr_pass:
        blockers.append("NO_EXECUTION_PATH_WITH_MIN_RR")
    real_money_candidate = len(blockers) == 0
    # Candidate quality and entry timing are intentionally separate. A good business can
    # remain a real-money research candidate while WAIT_REACCUMULATION / NO_EDGE prevents
    # it from appearing in the actionable Real Money Candidate Top 3.
    actionable_real_money_states = {
        "EMIR_READY_WITH_PRECISE_TRIGGER",
        "EMIR_AUTO_EOD_READY",
        "EMIR_CORE_THESIS_READY_WAIT_IDX_INTEGRITY",
        "EMIR_THESIS_READY_WAIT_BID_OFFER",
        "EMIR_WATCH_INVENTORY_COLLECTION",
    }
    real_money_entry_candidate = bool(real_money_candidate and execution_geometry_valid and execution_min_rr_pass and state in actionable_real_money_states)
    # Scanner-side READY remains strict. Proxy-only / missing-cashflow / public-news evidence
    # may become a manual candidate, but can never self-authorize capital.
    direct_fundamental_ready = (
        official_cov >= 50
        and not cashflow_missing
        and cashflow_quality_state not in {"OCF_NEGATIVE", "OCF_AND_FCF_NEGATIVE"}
    )
    direct_story_ready = manual_verified_story
    # DIRECT_VERIFIED_READY means *all* scanner-observed conditions are resolved.
    # If growth/cash-flow/freshness still asks for human review, the candidate remains
    # MANUAL_CONFIRMATION_REQUIRED even when direct IDX/orderbook evidence is present.
    unresolved_pre_authorization_conditions = list(dict.fromkeys(manual_conditions))
    real_money_ready = bool(
        real_money_entry_candidate
        and direct_fundamental_ready
        and direct_story_ready
        and integrity_ready
        and orderbook_verified
        and precise_ready
        and not unresolved_pre_authorization_conditions
    )
    if real_money_entry_candidate and not real_money_ready:
        if not integrity_ready:
            manual_conditions.append("VERIFY_DIRECT_IDX_INTEGRITY")
        if not orderbook_verified:
            manual_conditions.append("CONFIRM_LIVE_BID_OFFER_TRIGGER")
        if not direct_fundamental_ready:
            manual_conditions.append("VERIFY_DIRECT_FUNDAMENTAL_AND_CASHFLOW")
        if not direct_story_ready:
            manual_conditions.append("VERIFY_DIRECT_NARRATIVE_OR_ISSUER_EVENT")
        manual_conditions.append("CONFIRM_NO_NEW_CORPORATE_ACTION_OR_MATERIAL_NEWS")
        # Stable de-duplication while preserving diagnostic order.
        manual_conditions = list(dict.fromkeys(manual_conditions))
    if not real_money_candidate:
        real_money_gate_state="REAL_MONEY_BLOCKED"
    elif not real_money_entry_candidate:
        real_money_gate_state="REAL_MONEY_WAIT_TIMING"
    elif real_money_ready:
        real_money_gate_state="REAL_MONEY_DIRECT_VERIFIED_READY"
    else:
        real_money_gate_state="REAL_MONEY_MANUAL_CONFIRMATION_REQUIRED"
    manual_checklist = "VERIFY_IDX_BOARD/HSC/FCA/UMA/SUSPENSION; CONFIRM_LIVE_BID_OFFER_TRIGGER; CONFIRM_NO_NEW_CORPORATE_ACTION/NEWS; USE_LIMIT_ORDER; DO_NOT_AVERAGE_DOWN_BELOW_INVALIDATION"
    effective_risk_budget = min(max(0.0,_finite(risk_budget_pct,0.5)),0.75) if guarded_real_money else max(0.0,_finite(risk_budget_pct,1.0))
    if guarded_real_money and (proxy_only or cashflow_missing):
        effective_risk_budget = min(effective_risk_budget, 0.50)
    # v1.9.6: a MANUAL real-money candidate must display the guarded manual risk budget
    # even when the surrounding scan was launched in RESEARCH mode. This avoids a 1.0%
    # research risk field leaking into the Real Money Candidate table.
    if real_money_entry_candidate and not real_money_ready:
        effective_risk_budget = min(effective_risk_budget, 0.50)
    elif real_money_ready:
        effective_risk_budget = min(effective_risk_budget, 0.75)
    manual_cap = min(max(0.0,requested_cap), 5.0 if market_regime=="RISK_OFF" else 7.5) if real_money_entry_candidate else 0.0
    if proxy_only or cashflow_missing or not manual_verified_story:
        manual_cap = min(manual_cap, 3.0)
    if growth_consistency_state in {"TURNAROUND_INFLECTION_UNCONFIRMED","LATEST_QUARTER_DECELERATION_YTD_POSITIVE","QUARTER_YTD_DIVERGENCE_REVIEW"}:
        manual_cap = min(manual_cap, 2.0)
    if cashflow_quality_state in {"FCF_NEGATIVE_CAPEX_REVIEW", "CASHFLOW_POSITIVE_WEAK_CONVERSION"}:
        manual_cap = min(manual_cap, 2.0)
    if real_money_ready:
        position_cap = min(position_cap if position_cap>0 else manual_cap, 10.0, manual_cap if manual_cap>0 else 10.0)
    elif guarded_real_money:
        position_cap = 0.0

    if thesis_ready and not (integrity_ready or integrity_auto_ready):
        plan["execution_state"] = "CORE_THESIS_READY_WAIT_IDX_INTEGRITY"
    elif auto_eod_ready:
        plan["execution_state"] = "AUTO_EOD_PROXY_TRIGGER_READY"
    elif thesis_ready and (integrity_ready or integrity_auto_ready) and not orderbook_verified:
        plan["execution_state"] = "THESIS_READY_WAIT_DIRECT_BID_OFFER_TRIGGER"
    trim_state = str(features.get("trim_state") or "HOLD_SCENARIO_UNTIL_INVALIDATED")
    if broker_shift == "DISTRIBUTION_DOMINANT" or "INVENTORY_RELEASE" in broker_shift:
        trim_state = "TRIM_OR_EXIT_ON_VERIFIED_BROKER_DISTRIBUTION"

    return {
        "ticker": normalize_ticker(ticker), "engine_version": ENGINE_VERSION, "framework_disclaimer": FRAMEWORK_DISCLAIMER,
        "formula_provenance_state": "PUBLIC_IDEAS_PLUS_AUTONOMOUS_EMPIRICAL_IDX_ENGINE_NOT_OFFICIAL_CAK_FORMULA",
        **dict(features), **market, **sector, **narrative, **broker, **ownership, **orderbook, **integrity, **fundamental, **calibration,
        "deep_review_state": "DEEP_REVIEWED" if deep_reviewed else "RADAR_ONLY",
        "emir_lifecycle": lifecycle,
        "emir_conviction_score": round(conviction, 1) if np.isfinite(conviction) else np.nan,
        "emir_evidence_coverage_pct": round(evidence_coverage, 1),
        "emir_scoring_lineage_state": "DISJOINT_EVIDENCE_PILLARS_V2",
        "emir_overlap_control_state": "FUNDAMENTAL_NOT_REUSED_IN_NARRATIVE_CONVERSION",
        "emir_decision_state": state,
        "core_thesis_ready": core_thesis_ready,
        "auto_core_thesis_ready": auto_core_thesis_ready,
        "thesis_ready": thesis_ready,
        "idx_integrity_ready": integrity_ready,
        "idx_integrity_auto_ready": integrity_auto_ready,
        "auto_eod_ready": auto_eod_ready,
        "risk_off_sector_leader_exception": risk_off_sector_leader_exception,
        "market_allows_thesis": market_allows_thesis,
        "capital_mode": capital_mode_state,
        "fundamental_cashflow_quality_state": cashflow_quality_state,
        "real_money_gate_state": real_money_gate_state,
        "real_money_candidate": real_money_candidate,
        "real_money_entry_candidate": real_money_entry_candidate,
        "real_money_ready": real_money_ready,
        "real_money_block_reasons": " | ".join(blockers) or "NONE",
        "real_money_manual_conditions": " | ".join(manual_conditions) or "NONE",
        "real_money_fundamental_evidence_tier": "OFFICIAL_VERIFIED" if official_cov >= 50 else "PUBLIC_PROXY_ACCEPTED_MANUAL",
        "real_money_narrative_evidence_tier": narrative_evidence_tier,
        "real_money_manual_checklist": manual_checklist,
        "guarded_position_cap_after_manual_confirmation_pct": round(manual_cap,2),
        "entry_authorization_state": "SCANNER_AUTHORIZED_DIRECT_VERIFIED" if real_money_ready else "MANUAL_CONFIRMATION_REQUIRED" if real_money_entry_candidate else "WAIT_TIMING_NO_ENTRY" if real_money_candidate else "NO_ENTRY_AUTHORIZATION",
        "pyramiding_state": "ADD_ONLY_AFTER_CONFIRMATION_NOT_AVERAGE_DOWN_BELOW_INVALIDATION",
        # Even in RESEARCH mode, a hard real-money blocker must suppress the generic
        # production_ready flag. Research timing may remain visible through action/state,
        # but it cannot be mistaken for executable production authorization.
        "production_ready": bool(real_money_ready if guarded_real_money else ((precise_ready or auto_eod_ready) and real_money_candidate)),
        "production_tier": ("GUARDED_DIRECT_VERIFIED" if real_money_ready else "MANUAL_CONFIRMATION_REQUIRED" if real_money_entry_candidate else "WAIT_TIMING" if real_money_candidate else "NOT_READY") if guarded_real_money else ("NOT_READY" if not real_money_candidate else "DIRECT_PRECISE" if precise_ready else "AUTO_EOD_PROXY" if auto_eod_ready else "NOT_READY"),
        "calibration_mode": str(calibration_mode or "SHADOW_ONLY").upper(),
        "action": action,
        "why_now": why_now,
        "what_must_happen_next": next_proof,
        "thesis_invalidation": invalidation,
        "risk_flags": " | ".join(risks) or "NO_MAJOR_EMIR_IDX_FRAMEWORK_RISK",
        "requested_position_value_idr": round(requested_value, 2),
        "max_safe_position_value_idr": round(max_safe_position_value, 2),
        "estimated_participation_rate_pct": round(participation_pct, 4) if np.isfinite(participation_pct) else np.nan,
        "max_participation_rate_pct": round(max_participation_pct, 2),
        "slippage_bps_proxy": round(slippage_bps, 1),
        "execution_capacity_state": capacity_state,
        "risk_budget_pct": round(effective_risk_budget, 2),
        "position_cap_pct": round(position_cap, 2),
        "trim_state": trim_state,
        "react_not_predict_state": "SCENARIO_BASED_NO_PRICE_FORECAST",
        **future_fundamental,
        **plan,
    }


# Compatibility alias for v1.0/v1.1 callers/tests.
def build_public_method_profile(**kwargs: Any) -> dict[str, Any]:
    return build_emir_profile(**kwargs)


def make_scan_id(as_of: Any, tickers: list[str]) -> str:
    payload = f"{pd.Timestamp(as_of).isoformat()}|{'|'.join(sorted(tickers))}|{ENGINE_VERSION}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


__all__ = [
    "ENGINE_VERSION", "MARKET_FEATURE_VERSION", "PUBLIC_FORMULA_REGISTRY", "aggregate_broker_summary", "build_emir_profile",
    "build_public_method_profile", "calculate_market_context", "calculate_market_context_from_universe", "blend_market_context", "calculate_market_features",
    "calculate_sector_context", "classify_lifecycle", "formula_registry_frame", "make_scan_id",
    "parse_idx_integrity", "parse_orderbook_evidence", "parse_ownership",
    "build_outcome_calibration", "select_outcome_calibration",
    "round_idx", "score_narrative_events",
]
