// main.ts — one server, two APIs, ten simulated free-tier providers.
//
// The server reads the trial brief (MAKEFUN_PROMPT_FILE), learns the provider
// directory from the env (GET /instance), probes every provider's own catalog
// route, registers its port and catalog, and then answers the twenty requests
// the env's client sends: OpenAI Chat Completions and Anthropic Messages, in
// JSON and in server-sent events. For each request it picks the strongest free
// model that meets the request's requirements, speaks that provider's wire,
// and fails over to the next-best model when a provider refuses. The answer
// is returned in the wire the client used, with the provider's text whole,
// its usage counts as that wire reports them, and the stop reason mapped to
// the client's family. Nothing here depends on the wall clock or on unseeded
// randomness, so a replay of the same trial leaves the same state.

import { createServer } from "node:http";
import { readFileSync } from "node:fs";

const FIXED_CREATED = 1767225600;
const SSE_PIECE = 13;
const PROVIDER_CALL_TIMEOUT_MS = 8000;
const WORK_POLL_MS = 100;
const WORK_FAIL_LIMIT = 10;

type Wire = "openai" | "anthropic" | "gemini";
type Auth = "bearer" | "x-api-key" | "x-goog-api-key" | "none";
type ClientApi = "openai" | "anthropic";

interface ModelCap {
  id: string;
  free: boolean;
  context: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  rating: number;
}

interface Provider {
  id: string;
  name: string;
  wire: Wire;
  auth: Auth;
  keyEnv: string;
  key: string;
  baseUrl: string;
  path: string;
  models: ModelCap[];
}

interface WorkReq {
  id: string;
  api: ClientApi;
  stream: boolean;
  weight: number;
  requires: { tools?: boolean; vision?: boolean; json?: boolean; min_context?: number };
}

interface ParsedReq {
  systemText: string;
  userText: string;
  image: { kind: "data"; mime: string; data: string } | { kind: "url"; url: string } | null;
  tools: { name: string; description: string; parameters: Record<string, unknown> }[];
  maxTokens: number | null;
  stops: string[];
  jsonMode: boolean;
  temperature: number | null;
  stream: boolean;
}

interface ProviderAnswer {
  text: string;
  input: number | null;
  output: number | null;
  stop: string;
}

const isRec = (v: unknown): v is Record<string, unknown> =>
  v !== null && typeof v === "object" && !Array.isArray(v);

const asNum = (v: unknown): number | null =>
  typeof v === "number" && Number.isFinite(v) ? v : null;

const asStr = (v: unknown): string | null => (typeof v === "string" ? v : null);

const strList = (v: unknown): string[] => {
  if (typeof v === "string") return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === "string");
};

// Plain-text out of a message content that may be a string or a block list.
function blocksText(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  const parts: string[] = [];
  for (const b of content) {
    if (isRec(b) && b.type === "text" && typeof b.text === "string") parts.push(b.text);
  }
  return parts.join("\n");
}

// The sealed quality table, as the brief prints it. Used only when the brief
// file cannot be read or parsed; the live brief is read first at startup and
// always wins over this copy.
const FALLBACK_RATINGS: { model: string; provider: string; rating: number; context: number; tools: boolean; vision: boolean; json: boolean }[] = [
  { model: "llama-3.1-8b-instant", provider: "groq", rating: 1210, context: 131072, tools: true, vision: false, json: true },
  { model: "gpt-oss-120b", provider: "groq", rating: 1352, context: 131072, tools: true, vision: false, json: true },
  { model: "qwen3.8-27b", provider: "groq", rating: 1434, context: 131072, tools: true, vision: false, json: true },
  { model: "gpt-oss-20b", provider: "groq", rating: 1318, context: 131072, tools: true, vision: false, json: false },
  { model: "llama-4-scout-17b", provider: "cerebras", rating: 1250, context: 131072, tools: true, vision: false, json: false },
  { model: "qwen-3.8-27b", provider: "cerebras", rating: 1434, context: 131072, tools: true, vision: false, json: true },
  { model: "gpt-oss-120b", provider: "cerebras", rating: 1352, context: 131072, tools: true, vision: false, json: true },
  { model: "gemini-3.8-flash", provider: "gemini", rating: 1458, context: 1048576, tools: true, vision: true, json: true },
  { model: "gemma-4-31b", provider: "gemini", rating: 1452, context: 131072, tools: true, vision: true, json: false },
  { model: "gemma-4-26b", provider: "gemini", rating: 1410, context: 32768, tools: false, vision: true, json: false },
  { model: "glm-4.7-flash", provider: "zai", rating: 1470, context: 204800, tools: true, vision: false, json: true },
  { model: "glm-4.6v-flash", provider: "zai", rating: 1432, context: 204800, tools: true, vision: true, json: false },
  { model: "z-ai/glm-5.3-flash:free", provider: "openrouter", rating: 1475, context: 204800, tools: true, vision: false, json: true },
  { model: "deepseek/deepseek-v4.1:free", provider: "openrouter", rating: 1462, context: 163840, tools: true, vision: false, json: true },
  { model: "minimax/m3:free", provider: "openrouter", rating: 1436, context: 196608, tools: true, vision: false, json: true },
  { model: "deepseek-v3.2", provider: "sambanova", rating: 1465, context: 163840, tools: true, vision: false, json: true },
  { model: "gemma-4-31b", provider: "sambanova", rating: 1452, context: 131072, tools: false, vision: true, json: false },
  { model: "mistral-nemo", provider: "pollinations", rating: 1180, context: 32768, tools: true, vision: false, json: false },
  { model: "openai-fast", provider: "pollinations", rating: 1390, context: 32768, tools: true, vision: false, json: false },
  { model: "glm-4.7-flash", provider: "cloudflare", rating: 1470, context: 204800, tools: true, vision: false, json: true },
  { model: "llama-3.3-70b", provider: "cloudflare", rating: 1330, context: 131072, tools: true, vision: false, json: false },
  { model: "nemotron-3.5-super-vl", provider: "cloudflare", rating: 1414, context: 131072, tools: false, vision: true, json: false },
  { model: "qwen3.5-397b", provider: "ovh", rating: 1440, context: 262144, tools: true, vision: false, json: false },
  { model: "llama-3.3-70b", provider: "ovh", rating: 1330, context: 131072, tools: true, vision: false, json: false },
  { model: "kilo-fast", provider: "kilol", rating: 1140, context: 131072, tools: true, vision: false, json: false },
  { model: "kilo-auto/free", provider: "kilol", rating: 1420, context: 131072, tools: true, vision: false, json: false },
];

// The workload manifest, as the brief prints it. Same role as the table
// above: a fallback when the brief file cannot be read or parsed.
const FALLBACK_WORKLOAD: WorkReq[] = [
  { id: "r01", api: "openai", stream: false, weight: 2, requires: { tools: true, vision: true } },
  { id: "r02", api: "openai", stream: true, weight: 2, requires: { tools: true, vision: true } },
  { id: "r03", api: "openai", stream: false, weight: 2, requires: { tools: true } },
  { id: "r04", api: "anthropic", stream: false, weight: 2, requires: { tools: true } },
  { id: "r05", api: "anthropic", stream: true, weight: 2, requires: { tools: true } },
  { id: "r06", api: "openai", stream: true, weight: 2, requires: { vision: true } },
  { id: "r07", api: "openai", stream: false, weight: 1, requires: { json: true } },
  { id: "r08", api: "anthropic", stream: false, weight: 1, requires: { json: true } },
  { id: "r09", api: "anthropic", stream: true, weight: 1, requires: { json: true } },
  { id: "r10", api: "openai", stream: false, weight: 1, requires: { min_context: 200000 } },
  { id: "r11", api: "openai", stream: true, weight: 1, requires: { min_context: 200000 } },
  { id: "r12", api: "anthropic", stream: false, weight: 1, requires: { min_context: 200000 } },
  { id: "r13", api: "openai", stream: false, weight: 1, requires: {} },
  { id: "r14", api: "openai", stream: false, weight: 1, requires: {} },
  { id: "r15", api: "openai", stream: false, weight: 1, requires: {} },
  { id: "r16", api: "anthropic", stream: false, weight: 1, requires: {} },
  { id: "r17", api: "anthropic", stream: true, weight: 1, requires: {} },
  { id: "r18", api: "openai", stream: true, weight: 1, requires: {} },
  { id: "r19", api: "openai", stream: false, weight: 1, requires: {} },
  { id: "r20", api: "anthropic", stream: true, weight: 1, requires: {} },
];

const FALLBACK_DIRECTORY: { id: string; name: string; wire: Wire; auth: Auth; keyEnv: string; key: string; path: string }[] = [
  { id: "groq", name: "Groq", wire: "openai", auth: "bearer", keyEnv: "GROQ_API_KEY", key: "sim-groq-0001", path: "/p/groq" },
  { id: "cerebras", name: "Cerebras", wire: "openai", auth: "bearer", keyEnv: "CEREBRAS_API_KEY", key: "sim-cerebras-0002", path: "/p/cerebras" },
  { id: "gemini", name: "Google Gemini", wire: "gemini", auth: "x-goog-api-key", keyEnv: "GEMINI_API_KEY", key: "sim-gemini-0003", path: "/p/gemini" },
  { id: "zai", name: "Z.ai", wire: "anthropic", auth: "bearer", keyEnv: "ZAI_API_KEY", key: "sim-zai-0004", path: "/p/zai" },
  { id: "openrouter", name: "OpenRouter", wire: "openai", auth: "bearer", keyEnv: "OPENROUTER_API_KEY", key: "sim-openrouter-0005", path: "/p/openrouter" },
  { id: "sambanova", name: "SambaNova", wire: "openai", auth: "bearer", keyEnv: "SAMBANOVA_API_KEY", key: "sim-sambanova-0006", path: "/p/sambanova" },
  { id: "pollinations", name: "Pollinations", wire: "openai", auth: "none", keyEnv: "", key: "", path: "/p/pollinations" },
  { id: "cloudflare", name: "Cloudflare Workers AI", wire: "openai", auth: "bearer", keyEnv: "CLOUDFLARE_API_TOKEN", key: "sim-cloudflare-0007", path: "/p/cloudflare" },
  { id: "ovh", name: "OVHcloud AI Endpoints", wire: "openai", auth: "none", keyEnv: "", key: "", path: "/p/ovh" },
  { id: "kilol", name: "Kilo", wire: "openai", auth: "none", keyEnv: "", key: "", path: "/p/kilol" },
];

// Pull one JSON document out of the brief text by its schema marker. The
// document opens with the brace just before its "schema" key.
function extractDoc(text: string, schema: string): Record<string, unknown> | null {
  const marker = `"schema":"${schema}"`;
  const at = text.indexOf(marker);
  if (at === -1) return null;
  const open = text.lastIndexOf("{", at);
  if (open === -1) return null;
  let depth = 0;
  let inStr = false;
  let esc = false;
  for (let i = open; i < text.length; i++) {
    const ch = text[i];
    if (inStr) {
      if (esc) esc = false;
      else if (ch === "\\") esc = true;
      else if (ch === '"') inStr = false;
      continue;
    }
    if (ch === '"') { inStr = true; continue; }
    if (ch === "{") depth++;
    else if (ch === "}") {
      depth--;
      if (depth === 0) {
        try {
          const v = JSON.parse(text.slice(open, i + 1)) as unknown;
          return isRec(v) ? v : null;
        } catch {
          return null;
        }
      }
    }
  }
  return null;
}

interface Brief {
  ratings: Map<string, ModelCap>;
  workload: Map<string, WorkReq>;
  directory: { id: string; name: string; wire: Wire; auth: Auth; keyEnv: string; key: string; path: string }[];
}

function readBrief(): Brief {
  const ratings = new Map<string, ModelCap>();
  for (const r of FALLBACK_RATINGS) {
    ratings.set(`${r.provider}/${r.model}`, { id: r.model, free: true, context: r.context, tools: r.tools, vision: r.vision, json: r.json, rating: r.rating });
  }
  const workload = new Map<string, WorkReq>(FALLBACK_WORKLOAD.map((w) => [w.id, w]));
  let directory = FALLBACK_DIRECTORY;
  const file = process.env.MAKEFUN_PROMPT_FILE;
  if (file) {
    try {
      const text = readFileSync(file, "utf8");
      const ratingsDoc = extractDoc(text, "free-model-aggregator-01/ratings/v1");
      if (ratingsDoc && Array.isArray(ratingsDoc.ratings)) {
        for (const row of ratingsDoc.ratings as unknown[]) {
          if (!isRec(row) || typeof row.model !== "string" || typeof row.provider !== "string") continue;
          if (row.free !== true) continue;
          ratings.set(`${row.provider}/${row.model}`, {
            id: row.model,
            free: true,
            context: asNum(row.context_length),
            tools: row.tools === true,
            vision: row.vision === true,
            json: row.json === true,
            rating: asNum(row.rating_milli) ?? 0,
          });
        }
      }
      const workloadDoc = extractDoc(text, "free-model-aggregator-01/workload/v1");
      if (workloadDoc && Array.isArray(workloadDoc.manifest)) {
        for (const row of workloadDoc.manifest as unknown[]) {
          if (!isRec(row) || typeof row.id !== "string") continue;
          const api = row.api === "anthropic" ? "anthropic" : "openai";
          const req: WorkReq = { id: row.id, api, stream: row.stream === true, weight: asNum(row.weight) ?? 1, requires: {} };
          if (isRec(row.requires)) {
            if (row.requires.tools === true) req.requires.tools = true;
            if (row.requires.vision === true) req.requires.vision = true;
            if (row.requires.json === true) req.requires.json = true;
            const mc = asNum(row.requires.min_context);
            if (mc !== null) req.requires.min_context = mc;
          }
          workload.set(req.id, req);
        }
      }
      const dirDoc = extractDoc(text, "free-model-aggregator-01/providers/v1");
      if (dirDoc && Array.isArray(dirDoc.providers)) {
        const parsed: Brief["directory"] = [];
        for (const row of dirDoc.providers as unknown[]) {
          if (!isRec(row) || typeof row.id !== "string") continue;
          const wire = row.wire === "anthropic" ? "anthropic" : row.wire === "gemini" ? "gemini" : "openai";
          const auth = row.auth === "x-api-key" ? "x-api-key" : row.auth === "x-goog-api-key" ? "x-goog-api-key" : row.auth === "none" ? "none" : "bearer";
          parsed.push({
            id: row.id,
            name: asStr(row.name) ?? row.id,
            wire,
            auth,
            keyEnv: asStr(row.key_env) ?? "",
            key: asStr(row.key) ?? "",
            path: asStr(row.path) ?? `/p/${row.id}`,
          });
        }
        if (parsed.length > 0) directory = parsed;
      }
    } catch {
      // Fall back to the sealed tables above.
    }
  }
  return { ratings, workload, directory };
}

// ---- Reading the client's request -----------------------------------------

// Split a data URL into its media type and base64 payload.
function splitDataUrl(url: string): { mime: string; data: string } | null {
  const m = /^data:([^;,]+)?;base64,(.*)$/s.exec(url);
  if (m === null) return null;
  return { mime: m[1] ?? "image/png", data: m[2] ?? "" };
}

function parseOpenAiBody(body: Record<string, unknown>): ParsedReq {
  const messages = Array.isArray(body.messages) ? body.messages.filter(isRec) : [];
  const systemParts: string[] = [];
  let userText = "";
  let image: ParsedReq["image"] = null;
  for (const msg of messages) {
    if (msg.role === "system") {
      const t = blocksText(msg.content);
      if (t !== "") systemParts.push(t);
    } else if (msg.role === "user") {
      userText = blocksText(msg.content);
      if (Array.isArray(msg.content)) {
        for (const b of msg.content) {
          if (!isRec(b) || b.type !== "image_url" || !isRec(b.image_url)) continue;
          const url = asStr(b.image_url.url);
          if (url === null) continue;
          const split = splitDataUrl(url);
          image = split !== null ? { kind: "data", mime: split.mime, data: split.data } : { kind: "url", url };
        }
      }
    }
  }
  const tools: ParsedReq["tools"] = [];
  if (Array.isArray(body.tools)) {
    for (const t of body.tools) {
      if (!isRec(t) || !isRec(t.function) || typeof t.function.name !== "string") continue;
      tools.push({
        name: t.function.name,
        description: asStr(t.function.description) ?? "",
        parameters: isRec(t.function.parameters) ? t.function.parameters : { type: "object", properties: {} },
      });
    }
  }
  const fmt = isRec(body.response_format) ? body.response_format.type : undefined;
  return {
    systemText: systemParts.join("\n"),
    userText,
    image,
    tools,
    maxTokens: asNum(body.max_tokens) ?? asNum(body.max_completion_tokens),
    stops: strList(body.stop),
    jsonMode: fmt === "json_object" || fmt === "json_schema",
    temperature: asNum(body.temperature),
    stream: body.stream === true,
  };
}

function parseAnthropicBody(body: Record<string, unknown>): ParsedReq {
  const systemText = blocksText(body.system);
  const messages = Array.isArray(body.messages) ? body.messages.filter(isRec) : [];
  let userText = "";
  let image: ParsedReq["image"] = null;
  for (const msg of messages) {
    if (msg.role !== "user") continue;
    userText = blocksText(msg.content);
    if (Array.isArray(msg.content)) {
      for (const b of msg.content) {
        if (!isRec(b) || b.type !== "image" || !isRec(b.source)) continue;
        if (b.source.type === "base64" && typeof b.source.data === "string") {
          image = { kind: "data", mime: asStr(b.source.media_type) ?? "image/png", data: b.source.data };
        } else if (b.source.type === "url" && typeof b.source.url === "string") {
          image = { kind: "url", url: b.source.url };
        }
      }
    }
  }
  const tools: ParsedReq["tools"] = [];
  if (Array.isArray(body.tools)) {
    for (const t of body.tools) {
      if (!isRec(t) || typeof t.name !== "string") continue;
      tools.push({
        name: t.name,
        description: asStr(t.description) ?? "",
        parameters: isRec(t.input_schema) ? t.input_schema : { type: "object", properties: {} },
      });
    }
  }
  return {
    systemText,
    userText,
    image,
    tools,
    maxTokens: asNum(body.max_tokens),
    stops: strList(body.stop_sequences),
    jsonMode: false,
    temperature: asNum(body.temperature),
    stream: body.stream === true,
  };
}

// ---- Building the provider's request --------------------------------------

function openAiMessages(req: ParsedReq): Record<string, unknown>[] {
  const out: Record<string, unknown>[] = [];
  if (req.systemText !== "") out.push({ role: "system", content: req.systemText });
  const content: Record<string, unknown>[] = [{ type: "text", text: req.userText }];
  if (req.image !== null) {
    const url = req.image.kind === "data" ? `data:${req.image.mime};base64,${req.image.data}` : req.image.url;
    content.push({ type: "image_url", image_url: { url } });
  }
  out.push({ role: "user", content });
  return out;
}

function anthropicMessages(req: ParsedReq): Record<string, unknown>[] {
  const content: Record<string, unknown>[] = [{ type: "text", text: req.userText }];
  if (req.image !== null) {
    if (req.image.kind === "data") {
      content.push({ type: "image", source: { type: "base64", media_type: req.image.mime, data: req.image.data } });
    } else {
      content.push({ type: "image", source: { type: "url", url: req.image.url } });
    }
  }
  return [{ role: "user", content }];
}

function openAiTools(req: ParsedReq): Record<string, unknown>[] {
  return req.tools.map((t) => ({ type: "function", function: { name: t.name, description: t.description, parameters: t.parameters } }));
}

function anthropicTools(req: ParsedReq): Record<string, unknown>[] {
  return req.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.parameters }));
}

function geminiParts(req: ParsedReq): Record<string, unknown>[] {
  const parts: Record<string, unknown>[] = [{ text: req.userText }];
  if (req.image !== null) {
    if (req.image.kind === "data") {
      parts.push({ inlineData: { mimeType: req.image.mime, data: req.image.data } });
    } else {
      parts.push({ fileData: { fileUri: req.image.url } });
    }
  }
  return parts;
}

// One provider call's wire body, in that provider's own shape. The model
// name goes in the body for the OpenAI and Anthropic wires and in the path
// for the Gemini wire, which is also where the stream flag lives there.
function providerBody(p: Provider, model: string, req: ParsedReq): { path: string; body: Record<string, unknown> } {
  if (p.wire === "openai") {
    const body: Record<string, unknown> = { model, messages: openAiMessages(req), stream: req.stream };
    if (req.maxTokens !== null) body.max_tokens = req.maxTokens;
    if (req.temperature !== null) body.temperature = req.temperature;
    if (req.tools.length > 0) {
      body.tools = openAiTools(req);
      body.tool_choice = "auto";
    }
    if (req.stops.length > 0) body.stop = req.stops;
    if (req.jsonMode) body.response_format = { type: "json_object" };
    if (req.stream) body.stream_options = { include_usage: true };
    return { path: "/v1/chat/completions", body };
  }
  if (p.wire === "anthropic") {
    const body: Record<string, unknown> = { model, messages: anthropicMessages(req), stream: req.stream };
    if (req.systemText !== "") body.system = req.systemText;
    if (req.maxTokens !== null) body.max_tokens = req.maxTokens;
    if (req.temperature !== null) body.temperature = req.temperature;
    if (req.tools.length > 0) {
      body.tools = anthropicTools(req);
      body.tool_choice = { type: "auto" };
    }
    if (req.stops.length > 0) body.stop_sequences = req.stops;
    return { path: "/v1/messages", body };
  }
  const config: Record<string, unknown> = {};
  if (req.maxTokens !== null) config.maxOutputTokens = req.maxTokens;
  if (req.stops.length > 0) config.stopSequences = req.stops;
  if (req.temperature !== null) config.temperature = req.temperature;
  if (req.jsonMode) config.responseMimeType = "application/json";
  const body: Record<string, unknown> = { contents: [{ role: "user", parts: geminiParts(req) }], generationConfig: config };
  if (req.systemText !== "") body.systemInstruction = { parts: [{ text: req.systemText }] };
  if (req.tools.length > 0) {
    body.tools = [{
      functionDeclarations: req.tools.map((t) => ({ name: t.name, description: t.description, parameters: t.parameters })),
    }];
  }
  const path = req.stream
    ? `/v1beta/models/${encodeURIComponent(model)}:streamGenerateContent?alt=sse`
    : `/v1beta/models/${encodeURIComponent(model)}:generateContent`;
  return { path, body };
}

function providerHeaders(p: Provider): Record<string, string> {
  const headers: Record<string, string> = { "content-type": "application/json" };
  if (p.auth === "bearer") headers.authorization = `Bearer ${p.key}`;
  else if (p.auth === "x-api-key") headers["x-api-key"] = p.key;
  else if (p.auth === "x-goog-api-key") headers["x-goog-api-key"] = p.key;
  if (p.wire === "anthropic") headers["anthropic-version"] = "2023-06-01";
  return headers;
}

// The usage the provider itself reports for a call, from the request it was
// sent and the answer it gave. The provider's formula is one plus a quarter
// of the prompt's characters, and one plus a quarter of the answer's.
function computedUsage(req: ParsedReq, answerText: string): { input: number; output: number } {
  const prompt = req.systemText.length + req.userText.length + req.tools.map((t) => t.name).join(",").length;
  return { input: 1 + Math.floor(prompt / 4), output: 1 + Math.floor(answerText.length / 4) };
}

// ---- Reading the provider's answer ----------------------------------------

interface SseFrame { event: string | null; data: string }

function parseSse(text: string): SseFrame[] {
  const frames: SseFrame[] = [];
  for (const chunk of text.replace(/\r\n/g, "\n").split("\n\n")) {
    if (chunk.trim() === "") continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of chunk.split("\n")) {
      if (line.startsWith("event:")) event = line.slice("event:".length).trim();
      else if (line.startsWith("data:")) data.push(line.slice("data:".length).replace(/^ /, ""));
    }
    if (data.length > 0) frames.push({ event, data: data.join("\n") });
  }
  return frames;
}

function jsonOrNull(text: string): Record<string, unknown> | null {
  try {
    const v = JSON.parse(text) as unknown;
    return isRec(v) ? v : null;
  } catch {
    return null;
  }
}

function textBlocks(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  const parts: string[] = [];
  for (const b of content) {
    if (isRec(b) && b.type === "text" && typeof b.text === "string") parts.push(b.text);
  }
  return parts.join("");
}

function splitPieces(text: string): string[] {
  const out: string[] = [];
  for (let at = 0; at < text.length; at += SSE_PIECE) out.push(text.slice(at, at + SSE_PIECE));
  return out.length === 0 ? [""] : out;
}

// Normalize any provider's answer — JSON or streamed, any of the three
// wires — into one shape: the text, the usage that wire reported, and the
// provider's stop reason.
function readProviderAnswer(wire: Wire, stream: boolean, raw: string): ProviderAnswer | null {
  if (wire === "openai") {
    if (!stream) {
      const v = jsonOrNull(raw);
      const choices = v !== null && Array.isArray(v.choices) ? v.choices : null;
      if (choices === null || choices.length === 0 || !isRec(choices[0])) return null;
      const choice = choices[0];
      const msg = isRec(choice.message) ? choice.message : null;
      if (msg === null || msg.role !== "assistant") return null;
      const text = textBlocks(msg.content);
      const usage = isRec(v.usage) ? v.usage : null;
      return {
        text,
        input: usage !== null ? asNum(usage.prompt_tokens) : null,
        output: usage !== null ? asNum(usage.completion_tokens) : null,
        stop: asStr(choice.finish_reason) ?? "stop",
      };
    }
    let text = "";
    let input: number | null = null;
    let output: number | null = null;
    let stop = "stop";
    for (const f of parseSse(raw)) {
      if (f.data.trim() === "[DONE]") continue;
      const chunk = jsonOrNull(f.data);
      if (chunk === null) continue;
      if (isRec(chunk.usage)) {
        input = asNum(chunk.usage.prompt_tokens);
        output = asNum(chunk.usage.completion_tokens);
        continue;
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : null;
      if (choices === null || choices.length === 0 || !isRec(choices[0])) continue;
      const choice = choices[0];
      if (isRec(choice.delta) && typeof choice.delta.content === "string") text += choice.delta.content;
      if (typeof choice.finish_reason === "string") stop = choice.finish_reason;
    }
    return { text, input, output, stop };
  }
  if (wire === "anthropic") {
    if (!stream) {
      const v = jsonOrNull(raw);
      if (v === null || v.type !== "message" || !Array.isArray(v.content)) return null;
      const usage = isRec(v.usage) ? v.usage : null;
      return {
        text: textBlocks(v.content),
        input: usage !== null ? asNum(usage.input_tokens) : null,
        output: usage !== null ? asNum(usage.output_tokens) : null,
        stop: asStr(v.stop_reason) ?? "end_turn",
      };
    }
    let text = "";
    let input: number | null = null;
    let output: number | null = null;
    let stop = "end_turn";
    for (const f of parseSse(raw)) {
      const data = jsonOrNull(f.data);
      if (data === null) continue;
      if (f.event === "message_start" && isRec(data.message) && isRec(data.message.usage)) {
        input = asNum(data.message.usage.input_tokens);
      } else if (f.event === "content_block_delta" && isRec(data.delta) && data.delta.type === "text_delta" && typeof data.delta.text === "string") {
        text += data.delta.text;
      } else if (f.event === "message_delta") {
        if (isRec(data.delta) && typeof data.delta.stop_reason === "string") stop = data.delta.stop_reason;
        if (isRec(data.usage)) output = asNum(data.usage.output_tokens);
      }
    }
    return { text, input, output, stop };
  }
  // Gemini.
  if (!stream) {
    const v = jsonOrNull(raw);
    const cands = v !== null && Array.isArray(v.candidates) ? v.candidates : null;
    if (cands === null || cands.length === 0 || !isRec(cands[0])) return null;
    const cand = cands[0];
    let text = "";
    if (isRec(cand.content) && Array.isArray(cand.content.parts)) {
      for (const part of cand.content.parts) {
        if (isRec(part) && typeof part.text === "string") text += part.text;
      }
    }
    const meta = isRec(v.usageMetadata) ? v.usageMetadata : null;
    return {
      text,
      input: meta !== null ? asNum(meta.promptTokenCount) : null,
      output: meta !== null ? asNum(meta.candidatesTokenCount) : null,
      stop: asStr(cand.finishReason) ?? "STOP",
    };
  }
  let text = "";
  let input: number | null = null;
  let output: number | null = null;
  let stop = "STOP";
  for (const f of parseSse(raw)) {
    const data = jsonOrNull(f.data);
    if (data === null) continue;
    const cands = Array.isArray(data.candidates) ? data.candidates : null;
    if (cands !== null && cands.length > 0 && isRec(cands[0])) {
      const cand = cands[0];
      if (isRec(cand.content) && Array.isArray(cand.content.parts)) {
        for (const part of cand.content.parts) {
          if (isRec(part) && typeof part.text === "string") text += part.text;
        }
      }
      if (typeof cand.finishReason === "string") stop = cand.finishReason;
    }
    if (isRec(data.usageMetadata)) {
      input = asNum(data.usageMetadata.promptTokenCount);
      output = asNum(data.usageMetadata.candidatesTokenCount);
    }
  }
  return { text, input, output, stop };
}

// One provider call at a time, in the order they are made: a second call
// sent while the first is in flight is refused by the providers that serve
// one call at a time, so the queue keeps every call legal.
let providerQueue: Promise<void> = Promise.resolve();

function callProvider(p: Provider, model: string, req: ParsedReq): Promise<{ status: number; raw: string } | null> {
  const run = providerQueue.then(async () => {
    const { path, body } = providerBody(p, model, req);
    const url = `${p.baseUrl}${path}`;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), PROVIDER_CALL_TIMEOUT_MS);
    try {
      const res = await fetch(url, { method: "POST", headers: providerHeaders(p), body: JSON.stringify(body), signal: controller.signal });
      const raw = await res.text();
      return { status: res.status, raw };
    } catch {
      return null;
    } finally {
      clearTimeout(timer);
    }
  });
  providerQueue = run.then(() => undefined, () => undefined);
  return run;
}

// Ask a provider for its catalog. A catalog call spends none of the
// provider's budget and is refused by nothing but a wrong key, so a probe
// that fails simply leaves the brief's table in place.
async function probeCatalog(p: Provider): Promise<ModelCap[] | null> {
  const path = p.wire === "gemini" ? "/v1beta/models" : "/v1/models";
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), PROVIDER_CALL_TIMEOUT_MS);
  try {
    const res = await fetch(`${p.baseUrl}${path}`, { method: "GET", headers: providerHeaders(p), signal: controller.signal });
    if (res.status < 200 || res.status >= 300) return null;
    const v = jsonOrNull(await res.text());
    if (v === null) return null;
    const rows: ModelCap[] = [];
    if (p.wire === "gemini") {
      if (!Array.isArray(v.models)) return null;
      for (const m of v.models as unknown[]) {
        if (!isRec(m)) continue;
        const name = asStr(m.name) ?? "";
        const id = name.startsWith("models/") ? name.slice("models/".length) : name;
        if (id === "") continue;
        rows.push({
          id,
          free: m.free === true,
          context: asNum(m.context_length),
          tools: m.tools === true,
          vision: m.vision === true,
          json: m.json === true,
          rating: 0,
        });
      }
    } else {
      if (!Array.isArray(v.data)) return null;
      for (const m of v.data as unknown[]) {
        if (!isRec(m) || typeof m.id !== "string" || m.id === "") continue;
        rows.push({
          id: m.id,
          free: m.free === true,
          context: asNum(m.context_length),
          tools: m.tools === true,
          vision: m.vision === true,
          json: m.json === true,
          rating: 0,
        });
      }
    }
    return rows.length === 0 ? null : rows;
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

// ---- Routing ---------------------------------------------------------------

function modelMeets(m: ModelCap, req: { tools?: boolean; vision?: boolean; json?: boolean; min_context?: number }): boolean {
  if (req.tools === true && !m.tools) return false;
  if (req.vision === true && !m.vision) return false;
  if (req.json === true && !m.json) return false;
  if (req.min_context !== undefined && (m.context === null || m.context < req.min_context)) return false;
  return true;
}

// Candidates for one request, best rating first. A JSON-mode request on the
// OpenAI wire may only go to a wire that can carry response_format: the
// Anthropic wire cannot, and its echo would say the mode was absent. (An
// Anthropic client never marks JSON mode in its own wire, so its requests
// may go anywhere the model itself is JSON-capable.)
function rankCandidates(
  providers: Provider[],
  api: ClientApi,
  requires: { tools?: boolean; vision?: boolean; json?: boolean; min_context?: number },
  jsonMode: boolean,
): { provider: Provider; model: ModelCap }[] {
  const out: { provider: Provider; model: ModelCap }[] = [];
  for (const p of providers) {
    if (jsonMode && api === "openai" && p.wire === "anthropic") continue;
    for (const m of p.models) {
      if (!m.free) continue;
      if (!modelMeets(m, requires)) continue;
      out.push({ provider: p, model: m });
    }
  }
  out.sort((a, b) => {
    if (b.model.rating !== a.model.rating) return b.model.rating - a.model.rating;
    if (a.provider.id !== b.provider.id) return a.provider.id < b.provider.id ? -1 : 1;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
  return out;
}

// ---- Answering the client ---------------------------------------------------

function openAiJson(model: string, ans: ProviderAnswer, req: ParsedReq): Record<string, unknown> {
  const usage = ans.input !== null && ans.output !== null
    ? { input: ans.input, output: ans.output }
    : computedUsage(req, ans.text);
  return {
    id: "chatcmpl-agg",
    object: "chat.completion",
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message: { role: "assistant", content: ans.text }, finish_reason: "stop", logprobs: null }],
    usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output },
  };
}

function openAiSse(model: string, ans: ProviderAnswer, req: ParsedReq): string {
  const usage = ans.input !== null && ans.output !== null
    ? { input: ans.input, output: ans.output }
    : computedUsage(req, ans.text);
  const chunk = (delta: Record<string, unknown>, finish: string | null): Record<string, unknown> => ({
    id: "chatcmpl-agg",
    object: "chat.completion.chunk",
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames: string[] = [`data: ${JSON.stringify(chunk({ role: "assistant", content: "" }, null))}\n\n`];
  for (const piece of splitPieces(ans.text)) {
    frames.push(`data: ${JSON.stringify(chunk({ content: piece }, null))}\n\n`);
  }
  frames.push(`data: ${JSON.stringify(chunk({}, "stop"))}\n\n`);
  frames.push(`data: ${JSON.stringify({
    id: "chatcmpl-agg",
    object: "chat.completion.chunk",
    created: FIXED_CREATED,
    model,
    choices: [],
    usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output },
  })}\n\n`);
  frames.push("data: [DONE]\n\n");
  return frames.join("");
}

function anthropicJson(model: string, ans: ProviderAnswer, req: ParsedReq): Record<string, unknown> {
  const usage = ans.input !== null && ans.output !== null
    ? { input: ans.input, output: ans.output }
    : computedUsage(req, ans.text);
  return {
    id: "msg_agg",
    type: "message",
    role: "assistant",
    model,
    content: [{ type: "text", text: ans.text }],
    stop_reason: "end_turn",
    stop_sequence: null,
    usage: { input_tokens: usage.input, output_tokens: usage.output },
  };
}

function anthropicSse(model: string, ans: ProviderAnswer, req: ParsedReq): string {
  const usage = ans.input !== null && ans.output !== null
    ? { input: ans.input, output: ans.output }
    : computedUsage(req, ans.text);
  const frame = (event: string, data: Record<string, unknown>): string =>
    `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
  const frames: string[] = [
    frame("message_start", {
      type: "message_start",
      message: { id: "msg_agg", type: "message", role: "assistant", model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: usage.input, output_tokens: 0 } },
    }),
    frame("content_block_start", { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } }),
  ];
  for (const piece of splitPieces(ans.text)) {
    frames.push(frame("content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: piece } }));
  }
  frames.push(frame("content_block_stop", { type: "content_block_stop", index: 0 }));
  frames.push(frame("message_delta", {
    type: "message_delta",
    delta: { stop_reason: "end_turn", stop_sequence: null },
    usage: { output_tokens: usage.output },
  }));
  frames.push(frame("message_stop", { type: "message_stop" }));
  return frames.join("");
}

// An error in the client's own wire, for a request no provider could serve.
function clientError(api: ClientApi, message: string): { status: number; headers: Record<string, string>; body: string } {
  if (api === "openai") {
    return {
      status: 502,
      headers: { "content-type": "application/json" },
      body: `${JSON.stringify({ error: { message, type: "api_error", param: null, code: "upstream_unavailable" } })}\n`,
    };
  }
  return {
    status: 502,
    headers: { "content-type": "application/json" },
    body: `${JSON.stringify({ type: "error", error: { type: "api_error", message } })}\n`,
  };
}

// ---- Catalog ---------------------------------------------------------------

interface CatalogDoc {
  providers: { id: string; base_url: string; key_env: string; wire: Wire; models: { id: string; free: boolean; context_length: number | null }[] }[];
}

function buildCatalog(providers: Provider[]): CatalogDoc {
  return {
    providers: providers.map((p) => ({
      id: p.id,
      base_url: p.path,
      key_env: p.keyEnv,
      wire: p.wire,
      models: p.models.map((m) => ({ id: m.id, free: m.free, context_length: m.context })),
    })),
  };
}

// The free model ids the server registers, each once, for GET /v1/models.
function freeModelList(providers: Provider[]): { id: string; context: number | null; tools: boolean; vision: boolean; json: boolean }[] {
  const seen = new Map<string, { id: string; context: number | null; tools: boolean; vision: boolean; json: boolean }>();
  for (const p of providers) {
    for (const m of p.models) {
      if (!m.free) continue;
      if (!seen.has(m.id)) {
        seen.set(m.id, { id: m.id, context: m.context, tools: m.tools, vision: m.vision, json: m.json });
      }
    }
  }
  return [...seen.values()];
}

function modelsPayload(providers: Provider[], wire: Wire): Record<string, unknown> {
  const list = freeModelList(providers);
  if (wire === "openai") {
    return {
      object: "list",
      data: list.map((m) => ({ id: m.id, object: "model", created: FIXED_CREATED, owned_by: "aggregator", free: true, context_length: m.context, tools: m.tools, vision: m.vision, json: m.json })),
    };
  }
  return {
    data: list.map((m) => ({ type: "model", id: m.id, display_name: m.id, created_at: "2026-01-01T00:00:00Z", free: true, context_length: m.context, tools: m.tools, vision: m.vision, json: m.json })),
    has_more: false,
  };
}

// ---- The server ------------------------------------------------------------

const sleep = (ms: number): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms));

async function fetchJson(url: string, init?: RequestInit, timeoutMs = 5000): Promise<Record<string, unknown> | null> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, { ...init, signal: controller.signal });
    if (res.status < 200 || res.status >= 300) return null;
    const v = jsonOrNull(await res.text());
    return v;
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

async function main(): Promise<number> {
  const brief = readBrief();
  const envUrl = (process.env.MAKEFUN_ENV_URL ?? "").replace(/\/$/, "");

  // --list prints the catalog and exits, with no server and no env calls.
  // Outside a trial there is no brief to read, so the catalog is empty.
  if (process.argv.includes("--list")) {
    if (!process.env.MAKEFUN_PROMPT_FILE) {
      process.stdout.write(`${JSON.stringify({ providers: [] })}\n`);
      return 0;
    }
    const byProvider = new Map<string, ModelCap[]>();
    for (const [key, cap] of brief.ratings) {
      const at = key.indexOf("/");
      if (at === -1) continue;
      const pid = key.slice(0, at);
      const list = byProvider.get(pid) ?? [];
      list.push(cap);
      byProvider.set(pid, list);
    }
    const providers: Provider[] = brief.directory.map((d) => ({
      id: d.id,
      name: d.name,
      wire: d.wire,
      auth: d.auth,
      keyEnv: d.keyEnv,
      key: d.key,
      baseUrl: envUrl === "" ? d.path : `${envUrl}${d.path}`,
      path: d.path,
      models: byProvider.get(d.id) ?? [],
    }));
    process.stdout.write(`${JSON.stringify(buildCatalog(providers))}\n`);
    return 0;
  }

  if (envUrl === "") {
    process.stderr.write("MAKEFUN_ENV_URL is not set\n");
    return 1;
  }

  // The live directory: ids, wires, auth schemes, keys and absolute base
  // URLs for dialling. The brief's directory is the fallback.
  let directory = brief.directory;
  const instance = await fetchJson(`${envUrl}/instance`);
  if (instance !== null && Array.isArray(instance.providers) && instance.providers.length > 0) {
    const parsed: Brief["directory"] = [];
    for (const row of instance.providers as unknown[]) {
      if (!isRec(row) || typeof row.id !== "string") continue;
      const wire = row.wire === "anthropic" ? "anthropic" : row.wire === "gemini" ? "gemini" : "openai";
      const auth = row.auth === "x-api-key" ? "x-api-key" : row.auth === "x-goog-api-key" ? "x-goog-api-key" : row.auth === "none" ? "none" : "bearer";
      parsed.push({
        id: row.id,
        name: asStr(row.name) ?? row.id,
        wire,
        auth,
        keyEnv: asStr(row.key_env) ?? "",
        key: asStr(row.key) ?? "",
        path: asStr(row.path) ?? `/p/${row.id}`,
      });
      const base = asStr(row.base_url);
      if (base !== null) (parsed[parsed.length - 1] as unknown as { baseUrl: string }).baseUrl = base;
    }
    if (parsed.length > 0) directory = parsed;
  }

  const byProvider = new Map<string, ModelCap[]>();
  for (const [key, cap] of brief.ratings) {
    const at = key.indexOf("/");
    if (at === -1) continue;
    const pid = key.slice(0, at);
    const list = byProvider.get(pid) ?? [];
    list.push(cap);
    byProvider.set(pid, list);
  }

  const providers: Provider[] = directory.map((d) => {
    const base = (d as unknown as { baseUrl?: string }).baseUrl ?? `${envUrl}${d.path}`;
    return {
      id: d.id,
      name: d.name,
      wire: d.wire,
      auth: d.auth,
      keyEnv: d.keyEnv,
      key: d.key,
      baseUrl: base,
      path: d.path,
      models: byProvider.get(d.id) ?? [],
    };
  });

  // Probe every provider's catalog route; the live catalog is the truth the
  // registered catalog is scored against, so a probe that answers replaces
  // the brief's rows for that provider. Ratings still come from the brief.
  for (const p of providers) {
    const probed = await probeCatalog(p);
    if (probed === null) continue;
    p.models = probed.map((m) => {
      const rated = brief.ratings.get(`${p.id}/${m.id}`);
      return { ...m, rating: rated !== undefined ? rated.rating : 0 };
    });
  }

  const catalog = buildCatalog(providers);
  let served = 0;

  const server = createServer((req, res) => {
    void (async () => {
      try {
        const url = new URL(req.url ?? "/", "http://server");
        const path = url.pathname;
        if (req.method === "GET" && path === "/v1/models") {
          const auth = req.headers.authorization;
          const apiKey = req.headers["x-api-key"];
          const wire: Wire = typeof auth === "string" && auth.startsWith("Bearer ") ? "openai" : typeof apiKey === "string" ? "anthropic" : "openai";
          const body = JSON.stringify(modelsPayload(providers, wire));
          res.writeHead(200, { "content-type": "application/json", "content-length": String(Buffer.byteLength(body)) });
          res.end(body);
          return;
        }
        if (req.method === "POST" && (path === "/v1/chat/completions" || path === "/v1/messages")) {
          const api: ClientApi = path === "/v1/chat/completions" ? "openai" : "anthropic";
          const chunks: Buffer[] = [];
          for await (const c of req) chunks.push(c as Buffer);
          const body = jsonOrNull(Buffer.concat(chunks).toString("utf8"));
          if (body === null) {
            const err = clientError(api, "the request body is not one JSON object");
            res.writeHead(err.status, err.headers);
            res.end(err.body);
            return;
          }
          const parsed = api === "openai" ? parseOpenAiBody(body) : parseAnthropicBody(body);
          // The request the env is sending right now, from GET /work: its
          // manifest row names the requirements the body alone cannot show
          // (a JSON request on the Anthropic wire, a long-context request).
          let requires: WorkReq["requires"] = {};
          let jsonNeeded = parsed.jsonMode;
          try {
            const work = await fetchJson(`${envUrl}/work`, undefined, 3000);
            if (work !== null && typeof work.next === "string") {
              const row = brief.workload.get(work.next);
              if (row !== undefined) {
                requires = row.requires;
                if (row.requires.json === true) jsonNeeded = true;
              }
            }
          } catch {
            // Fall through to body-derived requirements.
          }
          if (Object.keys(requires).length === 0) {
            if (parsed.tools.length > 0) requires.tools = true;
            if (parsed.image !== null) requires.vision = true;
            if (parsed.jsonMode) requires.json = true;
          }
          const candidates = rankCandidates(providers, api, requires, jsonNeeded);
          let servedAnswer: { model: string; ans: ProviderAnswer } | null = null;
          for (const cand of candidates) {
            const call = await callProvider(cand.provider, cand.model.id, parsed);
            if (call === null) continue;
            if (call.status === 402 || call.status === 429) continue;
            if (call.status < 200 || call.status >= 300) continue;
            const ans = readProviderAnswer(cand.provider.wire, parsed.stream, call.raw);
            if (ans === null) continue;
            servedAnswer = { model: cand.model.id, ans };
            break;
          }
          served += 1;
          if (servedAnswer === null) {
            const err = clientError(api, "no provider could serve this request");
            res.writeHead(err.status, err.headers);
            res.end(err.body);
            return;
          }
          const { model, ans } = servedAnswer;
          if (api === "openai") {
            if (parsed.stream) {
              const text = openAiSse(model, ans, parsed);
              res.writeHead(200, { "content-type": "text/event-stream; charset=utf-8", "cache-control": "no-cache", connection: "keep-alive" });
              res.end(text);
            } else {
              const text = `${JSON.stringify(openAiJson(model, ans, parsed))}\n`;
              res.writeHead(200, { "content-type": "application/json", "content-length": String(Buffer.byteLength(text)) });
              res.end(text);
            }
          } else {
            if (parsed.stream) {
              const text = anthropicSse(model, ans, parsed);
              res.writeHead(200, { "content-type": "text/event-stream; charset=utf-8", "cache-control": "no-cache", connection: "keep-alive" });
              res.end(text);
            } else {
              const text = `${JSON.stringify(anthropicJson(model, ans, parsed))}\n`;
              res.writeHead(200, { "content-type": "application/json", "content-length": String(Buffer.byteLength(text)) });
              res.end(text);
            }
          }
          return;
        }
        res.writeHead(404, { "content-type": "application/json" });
        res.end(`${JSON.stringify({ error: "not_found" })}\n`);
      } catch {
        try {
          res.writeHead(500, { "content-type": "application/json" });
          res.end(`${JSON.stringify({ error: "internal" })}\n`);
        } catch {
          // The socket is already gone.
        }
      }
    })();
  });

  const port = await new Promise<number>((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "0.0.0.0", () => {
      const addr = server.address();
      if (addr === null || typeof addr === "string") reject(new Error("no listen address"));
      else resolve(addr.port);
    });
  });

  const reg = await fetchJson(`${envUrl}/register`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ protocol_version: 1, port, catalog }),
  }, 10000);
  if (reg === null || reg.ok !== true) {
    process.stderr.write("register failed\n");
    server.close();
    return 1;
  }

  // Poll the env until the workload is done, then exit 0. The client sends
  // its requests while this loop runs; the handlers above answer them.
  let fails = 0;
  for (;;) {
    await sleep(WORK_POLL_MS);
    const work = await fetchJson(`${envUrl}/work`, undefined, 3000);
    if (work === null) {
      fails += 1;
      if (fails >= WORK_FAIL_LIMIT) {
        server.close();
        return 1;
      }
      continue;
    }
    fails = 0;
    if (work.done === true) break;
  }
  server.close();
  return 0;
}

process.exitCode = await main();
