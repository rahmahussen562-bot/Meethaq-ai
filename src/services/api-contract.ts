export const NO_ANSWER = "I could not find an answer to this question in the provided documents.";

export interface SourceItem {
  label: string;
  source: string;
  chunk_index: number;
  chunk_id?: string;
  citation_id?: string;
  rerank_score: number;
  cosine_distance?: number;
  cosine_similarity?: number;
  text: string;
  contract_category?: string;
}
export interface TelemetryResponse {
  collection?: string;
  indexed_chunks?: number;
  total_chunks?: number;
  embed_model?: string;
  rerank_model?: string;
  llm_model?: string;
  llm_status?: string;
  llm_invoked?: boolean;
  answer_mode?: string;
  calibration_status?: string;
  status?: string;
}
export interface AuditResponse {
  answer: string;
  sources: SourceItem[];
  stats: TelemetryResponse;
}
export class ApiError extends Error {
  readonly kind: "configuration" | "network" | "timeout" | "http" | "protocol";
  readonly status?: number;
  constructor(message: string, kind: ApiError["kind"], status?: number) {
    super(message);
    this.name = "ApiError";
    this.kind = kind;
    this.status = status;
  }
}

export function resolveApiBaseUrl(configured: string | undefined, origin: string): string {
  const page = new URL(origin);
  const local = ["localhost", "127.0.0.1", "[::1]"].includes(page.hostname);
  const setting = configured?.trim();
  if (!setting) return local ? "http://127.0.0.1:8000" : page.origin;
  let url: URL;
  try {
    if (!setting.startsWith("/") && !/^https?:\/\//i.test(setting)) throw new Error();
    url = new URL(setting, page.origin);
  } catch {
    throw new ApiError("VITE_API_URL must be an HTTP(S) backend URL or a path on this site's origin.", "configuration");
  }
  if (!["http:", "https:"].includes(url.protocol) || url.username || url.password || url.search || url.hash) {
    throw new ApiError("VITE_API_URL cannot contain credentials, query parameters, or a fragment.", "configuration");
  }
  if (!local && ["localhost", "127.0.0.1", "[::1]"].includes(url.hostname)) {
    throw new ApiError("This deployment points to a loopback API. Set VITE_API_URL to your HTTPS tunnel URL and rebuild.", "configuration");
  }
  if (page.protocol === "https:" && url.protocol === "http:" && !local) {
    throw new ApiError("This HTTPS deployment requires an HTTPS API URL.", "configuration");
  }
  return url.origin + url.pathname.replace(/\/+$/, "").replace(/\/api$/, "");
}
function object(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new ApiError("The API returned an invalid response.", "protocol");
  return value as Record<string, unknown>;
}
export function parseTelemetry(value: unknown): TelemetryResponse {
  const data = object(value);
  const count = data.indexed_chunks ?? data.total_chunks;
  if (typeof count !== "number" || !Number.isSafeInteger(count) || count < 0) throw new ApiError("The API did not return a valid indexed chunk count.", "protocol");
  if (data.indexed_chunks !== undefined && data.total_chunks !== undefined && data.indexed_chunks !== data.total_chunks) throw new ApiError("The API returned inconsistent chunk counts.", "protocol");
  const telemetry: TelemetryResponse = { indexed_chunks: count, total_chunks: count };
  for (const key of ["collection", "embed_model", "rerank_model", "llm_model", "llm_status", "status", "answer_mode", "calibration_status"] as const) {
    if (typeof data[key] === "string") telemetry[key] = data[key];
  }
  if (typeof data.llm_invoked === "boolean") telemetry.llm_invoked = data.llm_invoked;
  return telemetry;
}
export function parseAuditResponse(value: unknown): AuditResponse {
  const data = object(value);
  if (typeof data.answer !== "string" || !data.answer.trim() || !Array.isArray(data.sources)) throw new ApiError("The API returned an invalid audit response.", "protocol");
  const stats = parseTelemetry(data.stats);
  if (data.answer === NO_ANSWER) {
    if (data.sources.length) throw new ApiError("An abstained answer cannot include citations.", "protocol");
    if (stats.llm_invoked === true) throw new ApiError("An abstention cannot claim successful LLM synthesis.", "protocol");
    return { answer: data.answer, sources: [], stats };
  }
  const identities = new Set<string>();
  const sources = data.sources.map((item, index) => {
    const source = object(item);
    if (source.label !== "[Source " + (index + 1) + "]" || typeof source.source !== "string" || !source.source || typeof source.text !== "string" || !source.text.trim() || typeof source.chunk_index !== "number" || !Number.isSafeInteger(source.chunk_index) || source.chunk_index < 0 || typeof source.rerank_score !== "number" || !Number.isFinite(source.rerank_score)) throw new ApiError("The API returned an invalid source citation.", "protocol");
    const identity = typeof source.chunk_id === "string" && source.chunk_id ? source.chunk_id : JSON.stringify([source.source, source.chunk_index]);
    if (identities.has(identity)) throw new ApiError("The API returned duplicate source citations.", "protocol");
    identities.add(identity);
    return source as unknown as SourceItem;
  });
  if (!sources.length) throw new ApiError("The API returned an answer without sources.", "protocol");
  if (stats.llm_invoked !== true) throw new ApiError("A synthesized answer must report successful LLM invocation.", "protocol");
  const cited = new Set<string>();
  for (const paragraph of data.answer.trim().split(/\n+/).filter(Boolean)) {
    const match = paragraph.match(/^([\s\S]+?)\s*((?:\[Source [1-9]\d*\]\s*)+)$/);
    const labels = match?.[2].match(/\[Source [1-9]\d*\]/g) ?? [];
    if (!match || !match[1].trim() || /\[Source\b/i.test(match[1]) || !labels.length) throw new ApiError("Every synthesized assertion must end with a source citation.", "protocol");
    for (const label of labels) {
      if (!sources.some((item) => item.label === label)) throw new ApiError("The answer cites an unknown source.", "protocol");
      cited.add(label);
    }
  }
  if (cited.size !== sources.length) throw new ApiError("The API returned sources that are not cited in the answer.", "protocol");
  return { answer: data.answer, sources, stats };
}
export function sourceForCitation(sources: SourceItem[], citation: number): SourceItem | null {
  return sources.find((item) => item.label === "[Source " + citation + "]") ?? null;
}
