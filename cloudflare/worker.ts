type WorkerEnv = {
  ASSETS: { fetch(request: Request): Promise<Response> }
  MEETHAQ_API_ORIGIN?: string
}

const API_METHODS = new Map([
  ["/api/audit", "POST"],
  ["/api/telemetry", "GET"],
  ["/api/health", "GET"],
])

function errorResponse(status: number, detail: string): Response {
  return Response.json({ detail }, { status, headers: { "Cache-Control": "no-store" } })
}

function tunnelOrigin(value: string | undefined): URL | null {
  if (!value?.trim()) return null
  try {
    const origin = new URL(value.trim())
    const hostname = origin.hostname.toLowerCase()
    if (
      origin.protocol !== "https:" ||
      origin.username || origin.password || origin.search || origin.hash ||
      (origin.pathname !== "/" && origin.pathname !== "") ||
      hostname === "localhost" || hostname.endsWith(".localhost") ||
      hostname.startsWith("[") || /^\d{1,3}(?:\.\d{1,3}){3}$/.test(hostname)
    ) return null
    return origin
  } catch {
    return null
  }
}

export default {
  async fetch(request: Request, env: WorkerEnv): Promise<Response> {
    const url = new URL(request.url)
    if (url.pathname !== "/api" && !url.pathname.startsWith("/api/")) {
      return env.ASSETS.fetch(request)
    }

    const method = API_METHODS.get(url.pathname)
    if (!method) return errorResponse(404, "Unknown API route.")
    if (request.method !== method) {
      return Response.json(
        { detail: "Method not allowed." },
        { status: 405, headers: { Allow: method, "Cache-Control": "no-store" } },
      )
    }

    const origin = tunnelOrigin(env.MEETHAQ_API_ORIGIN)
    if (!origin) {
      return errorResponse(503, "Set MEETHAQ_API_ORIGIN to a valid HTTPS tunnel origin.")
    }
    if (origin.origin === url.origin) {
      return errorResponse(503, "MEETHAQ_API_ORIGIN must point to the backend tunnel.")
    }

    const target = new URL(url.pathname + url.search, origin)
    const headers = new Headers(request.headers)
    headers.delete("Host")
    headers.delete("Cookie")
    headers.delete("Origin")
    try {
      const upstream = await fetch(target, {
        method: request.method,
        headers,
        body: request.method === "POST" ? request.body : undefined,
        redirect: "manual",
        signal: AbortSignal.timeout(180_000),
      })
      if (upstream.status >= 300 && upstream.status < 400) {
        await upstream.body?.cancel()
        return errorResponse(502, "The backend tunnel returned a redirect instead of an API response.")
      }
      if (!/\bapplication\/(?:[\w.+-]*\+)?json\b/i.test(upstream.headers.get("Content-Type") || "")) {
        await upstream.body?.cancel()
        return errorResponse(502, "The backend tunnel did not return an API JSON response.")
      }
      const responseHeaders = new Headers(upstream.headers)
      responseHeaders.set("Cache-Control", "no-store")
      responseHeaders.delete("Set-Cookie")
      return new Response(upstream.body, { status: upstream.status, headers: responseHeaders })
    } catch {
      return errorResponse(502, "The backend tunnel could not be reached or timed out.")
    }
  },
}
