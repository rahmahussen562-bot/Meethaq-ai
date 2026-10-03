const API_BASE_URL =
  import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

export interface SourceItem {
  label: string;
  source: string;
  chunk_index: number;
  rerank_score: number;
  text?: string;
  contract_category?: string;
}

export interface AuditResponse {
  answer: string;
  sources: SourceItem[];
  stats: {
    collection: string;
    indexed_chunks: number;
    embed_model: string;
    rerank_model: string;
    llm_model: string;
  };
}

export interface TelemetryResponse {
  collection?: string;
  indexed_chunks?: number;
  total_chunks?: number;
  embed_model?: string;
  rerank_model?: string;
  llm_model?: string;
  status?: string;
}

export const DEFAULT_TELEMETRY: TelemetryResponse = {
  collection: "meethaq_contracts",
  indexed_chunks: 729,
  total_chunks: 729,
  embed_model: "ONNX-all-MiniLM-L6-v2",
  rerank_model: "HNSW Cosine Similarity",
  llm_model: "llama3.2:3b",
  status: "Air-Gapped Local Host",
};

export async function sendAuditQuery(
  query: string,
  expandQuery: boolean = false
): Promise<AuditResponse> {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 60000); // 60s timeout for local LLM

    const response = await fetch(`${API_BASE_URL}/api/audit`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: query || "", expand_query: Boolean(expandQuery) }),
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    if (!response.ok) {
      throw new Error(`API error: ${response.status} ${response.statusText}`);
    }

    const data = await response.json();
    return {
      answer:
        typeof data?.answer === "string"
          ? data.answer
          : "I could not find an answer to this question in the provided documents.",
      sources: Array.isArray(data?.sources) ? data.sources : [],
      stats: data?.stats || {
        collection: "meethaq_contracts",
        indexed_chunks: 729,
        embed_model: "ONNX-all-MiniLM-L6-v2",
        rerank_model: "HNSW Cosine Similarity",
        llm_model: "llama3.2:3b",
      },
    };
  } catch (error: unknown) {
    console.warn("[Meethaq API] sendAuditQuery error:", error);
    throw error;
  }
}

// Alias for flexibility
export const runAuditQuery = sendAuditQuery;

export async function fetchTelemetry(): Promise<TelemetryResponse> {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000); // 5s timeout

    const response = await fetch(`${API_BASE_URL}/api/telemetry`, {
      method: "GET",
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    if (!response.ok) {
      return DEFAULT_TELEMETRY;
    }

    const data = await response.json();
    const count = Number(data?.indexed_chunks ?? data?.total_chunks ?? 729);
    return {
      collection: data?.collection || "meethaq_contracts",
      indexed_chunks: count,
      total_chunks: count,
      embed_model: data?.embed_model || "ONNX-all-MiniLM-L6-v2",
      rerank_model: data?.rerank_model || "HNSW Cosine Similarity",
      llm_model: data?.llm_model || "llama3.2:3b",
      status: data?.status || "Air-Gapped Local Host",
    };
  } catch (error: unknown) {
    console.warn("[Meethaq API] fetchTelemetry fallback (using safe default state):", error);
    return DEFAULT_TELEMETRY;
  }
}
