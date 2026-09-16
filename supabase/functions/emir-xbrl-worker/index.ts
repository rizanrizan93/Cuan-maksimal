import "jsr:@supabase/functions-js/edge-runtime.d.ts";
import { createClient } from "npm:@supabase/supabase-js@2.57.4";
import { unzipSync } from "npm:fflate@0.8.2";
import { XMLParser } from "npm:fast-xml-parser@5.2.5";

const PARSER_VERSION = "EMIR_IDX_XBRL_EDGE_V1";
const ATTACHMENT_BASE = "https://block.idx.id";
const MAX_ATTACHMENT_BYTES = 25_000_000;
const MAX_UNCOMPRESSED_BYTES = 60_000_000;
const REQUEST_TIMEOUT_MS = 25_000;

type Json = Record<string, unknown>;

type QueueJob = {
  rank_date: string;
  ticker: string;
  report_year: number;
  report_period: string;
  period_end: string;
  publication_date: string;
  source_file_id: string;
  source_file_name: string;
  source_file_path: string;
  source_file_size: number;
  attempts: number;
};

type Context = {
  id: string;
  start: string | null;
  end: string | null;
  instant: string | null;
  dimensioned: boolean;
};

type Fact = {
  name: string;
  value: number;
  contextRef: string;
  unitRef: string;
};

const NAMES = {
  revenue: ["SalesAndRevenue", "Revenues", "Revenue", "OperatingRevenues"],
  netIncome: ["ProfitLossAttributableToParentEntity", "ProfitLoss"],
  assets: ["Assets", "TotalAssets"],
  liabilities: ["Liabilities", "TotalLiabilities"],
  equity: ["EquityAttributableToEquityOwnersOfParentEntity", "Equity"],
  currentAssets: ["CurrentAssets"],
  currentLiabilities: ["CurrentLiabilities"],
  cash: ["CashAndCashEquivalents", "CashAndCashEquivalentsCashFlows"],
  ocf: ["NetCashFlowsReceivedFromUsedInOperatingActivities", "NetCashFlowsFromUsedInOperatingActivities"],
  debt: [
    "ShortTermBankLoans", "LongTermBankLoans", "CurrentMaturitiesOfBankLoans",
    "CurrentMaturitiesOfBondsPayable", "BondsPayable", "LongTermBondsPayable",
    "ShortTermBorrowings", "LongTermBorrowings", "FinanceLeaseLiabilities",
  ],
  capex: [
    "PaymentsForAcquisitionOfPropertyPlantAndEquipment",
    "PaymentsForAcquisitionOfPropertyAndEquipment",
    "PaymentsForAdvancesForPurchaseOfPropertyPlantAndEquipment",
    "PaymentsForAdvancesForPurchaseOfPropertyAndEquipment",
    "PaymentsForAcquisitionOfIntangibleAssets",
    "PaymentsForAcquisitionOfOilAndGasAssets",
    "PaymentsForAcquisitionOfMiningProperties",
    "PaymentsForAcquisitionOfInvestmentProperties",
    "PaymentsForAcquisitionOfExplorationAndEvaluationAssets",
    "PaymentsForAcquisitionOfIndustrialTimberPlantations",
    "PaymentsForAcquisitionOfPlantationAssets",
    "PaymentsForAcquisitionOfPlasmaPlantations",
    "PaymentsForAcquisitionOfLivestockProduction",
    "PaymentsForAcquisitionOfTollRoadConcessionRights",
    "PaymentsForAcquisitionOfOtherNonFinancialAssets",
  ],
};

function clean(value: unknown): string {
  return String(value ?? "").trim();
}

function localName(value: string): string {
  return value.includes(":") ? value.split(":").at(-1)! : value;
}

function attribute(node: unknown, name: string): string {
  if (!node || typeof node !== "object" || Array.isArray(node)) return "";
  const wanted = name.toLowerCase();
  for (const [key, value] of Object.entries(node as Json)) {
    if (key.startsWith("@_") && localName(key.slice(2)).toLowerCase() === wanted) return clean(value);
  }
  return "";
}

function textValue(node: unknown): string {
  if (typeof node === "string" || typeof node === "number") return clean(node);
  if (Array.isArray(node)) return node.map(textValue).join("");
  if (!node || typeof node !== "object") return "";
  const record = node as Json;
  if ("#text" in record) return clean(record["#text"]);
  return "";
}

function numberValue(value: unknown): number | null {
  let text = clean(value);
  if (!text || ["-", "—", "na", "n/a", "nan", "none"].includes(text.toLowerCase())) return null;
  const negative = text.startsWith("(") && text.endsWith(")");
  text = text.replace(/[()\s\u00a0]/g, "").replace(/^(Rp|IDR)/i, "");
  if (text.includes(",") && text.includes(".")) {
    text = text.lastIndexOf(",") > text.lastIndexOf(".")
      ? text.replaceAll(".", "").replace(",", ".")
      : text.replaceAll(",", "");
  } else if (text.includes(",")) {
    const parts = text.split(",");
    text = parts.slice(1).every((part) => part.length === 3) ? parts.join("") : text.replace(",", ".");
  } else if ((text.match(/\./g) ?? []).length > 1) {
    text = text.replaceAll(".", "");
  }
  const parsed = Number(text);
  if (!Number.isFinite(parsed)) return null;
  return negative ? -parsed : parsed;
}

function collectByLocalName(node: unknown, wanted: string, output: unknown[] = []): unknown[] {
  if (Array.isArray(node)) {
    for (const item of node) collectByLocalName(item, wanted, output);
  } else if (node && typeof node === "object") {
    for (const [key, value] of Object.entries(node as Json)) {
      if (localName(key).toLowerCase() === wanted.toLowerCase()) {
        if (Array.isArray(value)) output.push(...value); else output.push(value);
      }
      collectByLocalName(value, wanted, output);
    }
  }
  return output;
}

function firstLocalText(node: unknown, wanted: string): string | null {
  const match = collectByLocalName(node, wanted, [])[0];
  const value = textValue(match);
  return value || null;
}

function extractContexts(document: unknown): Map<string, Context> {
  const contexts = new Map<string, Context>();
  for (const raw of collectByLocalName(document, "context")) {
    const id = attribute(raw, "id");
    if (!id) continue;
    contexts.set(id, {
      id,
      start: firstLocalText(raw, "startDate"),
      end: firstLocalText(raw, "endDate"),
      instant: firstLocalText(raw, "instant"),
      dimensioned: collectByLocalName(raw, "explicitMember").length > 0 ||
        collectByLocalName(raw, "typedMember").length > 0,
    });
  }
  return contexts;
}

function extractUnits(document: unknown): Map<string, string> {
  const units = new Map<string, string>();
  for (const raw of collectByLocalName(document, "unit")) {
    const id = attribute(raw, "id");
    if (!id) continue;
    units.set(id, collectByLocalName(raw, "measure").map(textValue).join("|").toUpperCase());
  }
  return units;
}

function walkFacts(node: unknown, facts: Fact[]): void {
  if (Array.isArray(node)) {
    for (const item of node) walkFacts(item, facts);
    return;
  }
  if (!node || typeof node !== "object") return;
  const record = node as Json;
  const contextRef = attribute(record, "contextRef");
  const inlineName = attribute(record, "name");
  if (contextRef) {
    let value = numberValue(textValue(record));
    if (value !== null) {
      const scale = numberValue(attribute(record, "scale"));
      if (scale !== null) value *= 10 ** Math.trunc(scale);
      if (attribute(record, "sign") === "-") value = -Math.abs(value);
      facts.push({
        name: localName(inlineName || clean(record["__tagName"])),
        value,
        contextRef,
        unitRef: attribute(record, "unitRef"),
      });
    }
  }
  for (const [key, child] of Object.entries(record)) {
    if (key.startsWith("@_") || key === "#text") continue;
    const children = Array.isArray(child) ? child : [child];
    for (const item of children) {
      if (item && typeof item === "object" && !Array.isArray(item)) {
        (item as Json)["__tagName"] = key;
      }
      walkFacts(item, facts);
    }
  }
}

function parseDate(value: string | null): number | null {
  if (!value) return null;
  const timestamp = Date.parse(value.slice(0, 10) + "T00:00:00Z");
  return Number.isFinite(timestamp) ? timestamp : null;
}

function daysBetween(a: string | null, b: string | null): number | null {
  const left = parseDate(a);
  const right = parseDate(b);
  return left === null || right === null ? null : Math.round((right - left) / 86_400_000);
}

function shiftYear(value: string, delta: number): string {
  const date = new Date(value + "T00:00:00Z");
  date.setUTCFullYear(date.getUTCFullYear() + delta);
  return date.toISOString().slice(0, 10);
}

function closestDuration(contexts: Map<string, Context>, targetEnd: string): number | null {
  const durations = [...contexts.values()]
    .filter((context) => context.start && context.end && !context.dimensioned)
    .filter((context) => Math.abs(daysBetween(context.end, targetEnd) ?? 9999) <= 7)
    .map((context) => daysBetween(context.start, context.end))
    .filter((value): value is number => value !== null && value >= 20 && value <= 400);
  return durations.length ? Math.max(...durations) : null;
}

function selectFact(
  facts: Fact[], contexts: Map<string, Context>, units: Map<string, string>, names: string[],
  duration: boolean, targetEnd: string, targetDuration: number | null,
): number | null {
  const wanted = new Set(names.map((name) => name.toLowerCase()));
  const ranked = facts.flatMap((fact) => {
    if (!wanted.has(localName(fact.name).toLowerCase())) return [];
    const context = contexts.get(fact.contextRef);
    if (!context) return [];
    const actualEnd = duration ? context.end : context.instant;
    const endDistance = Math.abs(daysBetween(actualEnd, targetEnd) ?? 9999);
    if (endDistance > 45) return [];
    const unit = units.get(fact.unitRef) ?? fact.unitRef.toUpperCase();
    const monetary = unit.includes("IDR") || unit.includes("USD");
    let score = 200 - endDistance * 2 + (context.dimensioned ? 0 : 80) + (monetary ? 20 : 0);
    if (duration) {
      if (!context.start || !context.end) return [];
      const durationDays = daysBetween(context.start, context.end);
      if (targetDuration !== null && durationDays !== null) score -= Math.abs(durationDays - targetDuration);
    } else if (!context.instant) return [];
    return [{ value: fact.value, score }];
  }).sort((a, b) => b.score - a.score);
  return ranked[0]?.value ?? null;
}

function sumNamedFacts(
  facts: Fact[], contexts: Map<string, Context>, units: Map<string, string>, names: string[],
  duration: boolean, targetEnd: string, targetDuration: number | null,
): number | null {
  const values = names.map((name) => selectFact(facts, contexts, units, [name], duration, targetEnd, targetDuration))
    .filter((value): value is number => value !== null && Number.isFinite(value) && Math.abs(value) > 0);
  return values.length ? values.reduce((sum, value) => sum + Math.abs(value), 0) : null;
}

function ratio(numerator: number | null, denominator: number | null, scale = 1): number | null {
  return numerator !== null && denominator !== null && denominator !== 0
    ? numerator / denominator * scale : null;
}

function rounded(value: number | null, digits = 4): number | null {
  return value === null || !Number.isFinite(value) ? null : Number(value.toFixed(digits));
}

function xmlDocuments(payload: Uint8Array, filename: string): Array<[string, Uint8Array]> {
  const isZip = payload[0] === 0x50 && payload[1] === 0x4b && payload[2] === 0x03 && payload[3] === 0x04;
  if (!isZip) return [[filename || "filing.xbrl", payload]];
  const archive = unzipSync(payload);
  let total = 0;
  const documents: Array<[string, Uint8Array]> = [];
  for (const [name, bytes] of Object.entries(archive)) {
    if (!/\.(xbrl|xml|xhtml|html|htm)$/i.test(name)) continue;
    total += bytes.byteLength;
    if (total > MAX_UNCOMPRESSED_BYTES) throw new Error("UNCOMPRESSED_LIMIT_EXCEEDED");
    documents.push([name, bytes]);
  }
  if (!documents.length) throw new Error("XBRL_DOCUMENT_NOT_FOUND");
  return documents;
}

function parseXbrl(payload: Uint8Array, job: QueueJob, payloadHash: string): Json {
  const parser = new XMLParser({
    ignoreAttributes: false,
    attributeNamePrefix: "@_",
    parseTagValue: false,
    trimValues: true,
    allowBooleanAttributes: true,
  });
  const contexts = new Map<string, Context>();
  const units = new Map<string, string>();
  const facts: Fact[] = [];
  const decoder = new TextDecoder();
  for (const [, bytes] of xmlDocuments(payload, job.source_file_name)) {
    let document: unknown;
    try {
      document = parser.parse(decoder.decode(bytes));
    } catch {
      continue;
    }
    for (const [key, value] of extractContexts(document)) contexts.set(key, value);
    for (const [key, value] of extractUnits(document)) units.set(key, value);
    walkFacts(document, facts);
  }
  if (!contexts.size || !facts.length) throw new Error("XBRL_FACTS_EMPTY");

  const currentEnd = job.period_end;
  const priorEnd = shiftYear(currentEnd, -1);
  const currentDuration = closestDuration(contexts, currentEnd);
  const priorDuration = currentDuration;
  const duration = (names: string[], end = currentEnd) =>
    selectFact(facts, contexts, units, names, true, end, end === currentEnd ? currentDuration : priorDuration);
  const instant = (names: string[]) => selectFact(facts, contexts, units, names, false, currentEnd, null);

  const revenue = duration(NAMES.revenue);
  const netIncome = duration(NAMES.netIncome);
  const priorRevenue = duration(NAMES.revenue, priorEnd);
  const priorNetIncome = duration(NAMES.netIncome, priorEnd);
  const assets = instant(NAMES.assets);
  const liabilities = instant(NAMES.liabilities);
  const equity = instant(NAMES.equity);
  const currentAssets = instant(NAMES.currentAssets);
  const currentLiabilities = instant(NAMES.currentLiabilities);
  const cash = instant(NAMES.cash);
  const ocf = duration(NAMES.ocf);
  const debt = sumNamedFacts(facts, contexts, units, NAMES.debt, false, currentEnd, null);
  const capex = sumNamedFacts(facts, contexts, units, NAMES.capex, true, currentEnd, currentDuration);
  const fcf = ocf !== null && capex !== null ? ocf - capex : null;
  const coverageCore = [revenue, netIncome, ocf, assets, liabilities, equity];
  const coveragePct = 100 * coverageCore.filter((value) => value !== null).length / coverageCore.length;
  if (coveragePct < 50) throw new Error(`OFFICIAL_COVERAGE_BELOW_50:${coveragePct.toFixed(1)}`);

  const annualization = job.report_period.toUpperCase() === "TW1" ? 4
    : job.report_period.toUpperCase() === "TW2" ? 2
    : job.report_period.toUpperCase() === "TW3" ? 4 / 3 : 1;
  const sourceUrl = new URL(job.source_file_path, ATTACHMENT_BASE).toString();
  const periodType = job.report_period.toUpperCase() === "AUDIT" ? "FY" : job.report_period.toUpperCase().replace("TW", "Q");
  return {
    ticker: job.ticker,
    period_end: job.period_end,
    observed_on: job.publication_date,
    period_type: periodType,
    revenue,
    revenue_growth_yoy_pct: rounded(ratio(revenue !== null && priorRevenue !== null ? revenue - priorRevenue : null, priorRevenue, 100), 2),
    net_income: netIncome,
    earnings_growth_yoy_pct: rounded(ratio(netIncome !== null && priorNetIncome !== null ? netIncome - priorNetIncome : null, priorNetIncome, 100), 2),
    net_margin_pct: rounded(ratio(netIncome, revenue, 100), 2),
    roe_pct: rounded(ratio(netIncome === null ? null : netIncome * annualization, equity, 100), 2),
    roa_pct: rounded(ratio(netIncome === null ? null : netIncome * annualization, assets, 100), 2),
    ocf,
    fcf,
    cash,
    debt,
    debt_to_equity: rounded(ratio(debt, equity), 4),
    current_ratio: rounded(ratio(currentAssets, currentLiabilities), 4),
    cash_to_debt_ratio: rounded(ratio(cash, debt), 4),
    source_url: sourceUrl,
    coverage_pct: rounded(coveragePct, 2),
    payload_hash: payloadHash,
    source_verified: true,
    raw_payload: {
      parser_version: PARSER_VERSION,
      source_file_id: job.source_file_id,
      source_file_name: job.source_file_name,
      source_file_size: job.source_file_size,
      publication_date: job.publication_date,
      comparative_prior_end: priorEnd,
      prior_revenue: priorRevenue,
      prior_net_income: priorNetIncome,
      assets,
      liabilities,
      equity,
      current_assets: currentAssets,
      current_liabilities: currentLiabilities,
      capex,
      facts_seen: facts.length,
      contexts_seen: contexts.size,
    },
  };
}

async function sha256(payload: Uint8Array): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", payload);
  return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

async function download(job: QueueJob): Promise<Uint8Array> {
  if (!job.source_file_path.startsWith("/Portals/")) throw new Error("ATTACHMENT_PATH_REJECTED");
  const url = new URL(job.source_file_path, ATTACHMENT_BASE);
  if (url.protocol !== "https:" || url.hostname !== "block.idx.id") {
    throw new Error("ATTACHMENT_HOST_REJECTED");
  }
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    const response = await fetch(url, {
      headers: {
        "User-Agent": "EMIR-Official-XBRL-Worker/1.0",
        "Accept": "application/zip,application/octet-stream,*/*",
        "Referer": "https://block.idx.id/",
      },
      redirect: "follow",
      signal: controller.signal,
    });
    const finalUrl = new URL(response.url);
    if (!response.ok) throw new Error(`ATTACHMENT_HTTP_${response.status}`);
    if (finalUrl.protocol !== "https:" || finalUrl.hostname !== "block.idx.id") {
      throw new Error("ATTACHMENT_REDIRECT_REJECTED");
    }
    const declared = Number(response.headers.get("content-length") ?? "0");
    if (declared > MAX_ATTACHMENT_BYTES) throw new Error("ATTACHMENT_TOO_LARGE");
    const payload = new Uint8Array(await response.arrayBuffer());
    if (!payload.length || payload.length > MAX_ATTACHMENT_BYTES) throw new Error("ATTACHMENT_SIZE_INVALID");
    return payload;
  } finally {
    clearTimeout(timeout);
  }
}

async function processJob(supabase: ReturnType<typeof createClient>, job: QueueJob): Promise<Json> {
  try {
    const payload = await download(job);
    const payloadHash = await sha256(payload);
    const row = parseXbrl(payload, job, payloadHash);
    const { error: upsertError } = await supabase.from("cak_idx_fundamental_snapshot").upsert(row, {
      onConflict: "ticker,period_end,observed_on,payload_hash",
      ignoreDuplicates: false,
    });
    if (upsertError) throw new Error(`SNAPSHOT_UPSERT:${upsertError.code ?? upsertError.message}`);
    const { data: status, error: finishError } = await supabase.rpc("cak_idx_finish_fundamental_job_v1", {
      p_rank_date: job.rank_date,
      p_ticker: job.ticker,
      p_success: true,
      p_payload_hash: payloadHash,
      p_coverage_pct: row.coverage_pct,
      p_error_code: null,
    });
    if (finishError) throw new Error(`QUEUE_FINISH:${finishError.code ?? finishError.message}`);
    return { ticker: job.ticker, state: status, coverage_pct: row.coverage_pct };
  } catch (error) {
    const errorCode = clean(error instanceof Error ? error.message : error).slice(0, 160) || "UNKNOWN_ERROR";
    await supabase.rpc("cak_idx_finish_fundamental_job_v1", {
      p_rank_date: job.rank_date,
      p_ticker: job.ticker,
      p_success: false,
      p_payload_hash: null,
      p_coverage_pct: null,
      p_error_code: errorCode,
    });
    return { ticker: job.ticker, state: "FAILED", error_code: errorCode };
  }
}

async function processPool<T, R>(items: T[], concurrency: number, worker: (item: T) => Promise<R>): Promise<R[]> {
  const output: R[] = [];
  let cursor = 0;
  async function run(): Promise<void> {
    while (cursor < items.length) {
      const index = cursor++;
      output[index] = await worker(items[index]);
    }
  }
  await Promise.all(Array.from({ length: Math.min(concurrency, items.length) }, () => run()));
  return output;
}

Deno.serve(async (request) => {
  if (request.method !== "POST") return new Response("method not allowed", { status: 405 });
  const supabaseUrl = Deno.env.get("SUPABASE_URL") ?? "";
  const serviceKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ?? "";
  const anonKey = Deno.env.get("SUPABASE_ANON_KEY") ?? "";
  if (!supabaseUrl || !serviceKey || !anonKey) return new Response("runtime configuration missing", { status: 500 });
  const token = request.headers.get("x-emir-worker-token") ?? "";
  const supabase = createClient(supabaseUrl, serviceKey, { auth: { persistSession: false } });
  const { data: authorized, error: authError } = await supabase.rpc("cak_idx_validate_worker_token_v1", {
    p_worker_name: "emir-xbrl-worker",
    p_token: token,
  });
  if (authError || authorized !== true) return new Response("unauthorized", { status: 401 });

  let body: Json = {};
  try { body = await request.json(); } catch { body = {}; }
  const limit = Math.min(3, Math.max(1, Number(body.limit ?? 3)));
  const workerId = `edge-${crypto.randomUUID().slice(0, 12)}`;
  const { data, error } = await supabase.rpc("cak_idx_claim_fundamental_jobs_v1", {
    p_limit: limit,
    p_worker_id: workerId,
  });
  if (error) return Response.json({ state: "CLAIM_FAILED", error: error.code ?? error.message }, { status: 500 });
  const jobs = (Array.isArray(data) ? data : []) as QueueJob[];
  const results = await processPool(jobs, 1, (job) => processJob(supabase, job));
  const complete = results.filter((item) => item.state === "COMPLETE").length;
  const failed = results.length - complete;

  if (jobs.length === 0) {
    await supabase.rpc("cak_idx_finalize_fundamental_evidence_v1", { p_rank_date: null });
  } else if (body.chain === true) {
    const nextDepth = Number(body.depth ?? 0) + 1;
    if (nextDepth <= 300) {
      const chained = new Promise<void>((resolve) => setTimeout(resolve, 1000)).then(() =>
        fetch(`${supabaseUrl}/functions/v1/emir-xbrl-worker`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "apikey": anonKey,
            "Authorization": `Bearer ${anonKey}`,
            "x-emir-worker-token": token,
          },
          body: JSON.stringify({ chain: true, limit, depth: nextDepth }),
        })
      );
      EdgeRuntime.waitUntil(chained.then(() => undefined).catch(() => undefined));
    }
  }

  return Response.json({
    state: jobs.length ? "PROCESSED" : "QUEUE_EMPTY",
    worker_id: workerId,
    claimed: jobs.length,
    complete,
    failed,
    parser_version: PARSER_VERSION,
  });
});
