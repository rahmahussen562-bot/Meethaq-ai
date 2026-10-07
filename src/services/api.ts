import { ApiError, parseAuditResponse, parseTelemetry, resolveApiBaseUrl, type AuditResponse, type TelemetryResponse } from "./api-contract";
export { ApiError, sourceForCitation, type AuditResponse, type SourceItem, type TelemetryResponse } from "./api-contract";

export const DEFAULT_TELEMETRY: TelemetryResponse = { status: "Checking contract index…" };

async function request(path: string, init: RequestInit, timeoutMs: number): Promise<unknown> {
  const base = resolveApiBaseUrl(import.meta.env.VITE_API_URL, window.location.origin);
  const controller = new AbortController();
  const externalSignal = init.signal;
  const abort = () => controller.abort();
  externalSignal?.addEventListener("abort", abort, { once: true });
  if (externalSignal?.aborted) controller.abort();
  let timedOut = false;
  const timer = window.setTimeout(() => { timedOut = true; controller.abort(); }, timeoutMs);
  try {
    const response = await fetch(base + path, { ...init, signal: controller.signal, credentials: "omit" });
    if (!response.ok) {
      let detail = "";
      try { const payload = await response.json(); if (typeof payload?.detail === "string") detail = payload.detail; } catch { /* A proxy may return HTML. */ }
      throw new ApiError(detail || "The API returned HTTP " + response.status + ". Check the backend and /api proxy routing.", "http", response.status);
    }
    try { return await response.json(); } catch { throw new ApiError("The API returned non-JSON content. Configure VITE_API_URL or route /api to FastAPI.", "protocol"); }
  } catch (error) {
    if (externalSignal?.aborted) throw new DOMException("Request cancelled", "AbortError");
    if (error instanceof ApiError) throw error;
    if (timedOut) throw new ApiError("The backend request timed out. Check indexing activity and backend logs.", "timeout");
    throw new ApiError("Could not reach the configured API at " + base + ". Check the backend, tunnel, and CORS settings.", "network");
  } finally {
    window.clearTimeout(timer);
    externalSignal?.removeEventListener("abort", abort);
  }
}
export async function sendAuditQuery(query: string, expandQuery = false, signal?: AbortSignal): Promise<AuditResponse> {
  return parseAuditResponse(await request("/api/audit", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, expand_query: expandQuery }), signal,
  }, 180000));
}
export const runAuditQuery = sendAuditQuery;
export async function fetchTelemetry(signal?: AbortSignal): Promise<TelemetryResponse> {
  return parseTelemetry(await request("/api/telemetry", { method: "GET", signal, cache: "no-store" }, 10000));
}
