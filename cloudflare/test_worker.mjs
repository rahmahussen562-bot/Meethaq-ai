import test from "node:test"
import assert from "node:assert/strict"
import { readFile } from "node:fs/promises"
import ts from "typescript"

const source = await readFile(new URL("./worker.ts", import.meta.url), "utf8")
const compiled = ts.transpileModule(source, {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext },
}).outputText
const { default: worker } = await import(`data:text/javascript;base64,${Buffer.from(compiled).toString("base64")}`)
const request = (path = "/api/telemetry", options) => new Request(`https://meethaq-ai.workers.dev${path}`, options)
const assets = { fetch: async () => new Response("app shell", { headers: { "Content-Type": "text/html" } }) }
const env = (origin) => ({ ASSETS: assets, MEETHAQ_API_ORIGIN: origin })

async function mockedFetch(mock, run) {
  const originalFetch = globalThis.fetch
  globalThis.fetch = mock
  try { await run() } finally { globalThis.fetch = originalFetch }
}

test("navigation uses static assets while API routes keep JSON errors", async () => {
  const page = await worker.fetch(request("/audit"), env())
  assert.equal(await page.text(), "app shell")
  for (const path of ["/api", "/api/unknown", "/api/audit/extra"]) {
    const response = await worker.fetch(request(path), env())
    assert.equal(response.status, 404)
    assert.equal((await response.json()).detail, "Unknown API route.")
  }
})

test("invalid, insecure, loopback and self-referencing origins fail without network calls", async () => {
  const invalid = [
    undefined, "", "not a url", "http://backend.example.com", "https://localhost", "https://app.localhost",
    "https://127.0.0.1", "https://127.1", "https://0.0.0.0", "https://[::1]", "https://[::ffff:127.0.0.1]",
    "https://user:secret@backend.example.com", "https://backend.example.com/api", "https://backend.example.com?x=1",
    "https://backend.example.com#fragment", "https://meethaq-ai.workers.dev",
  ]
  await mockedFetch(() => { throw new Error("Unexpected network request") }, async () => {
    for (const origin of invalid) {
      const response = await worker.fetch(request(), env(origin))
      assert.equal(response.status, 503, `origin ${origin}`)
      assert.equal(response.headers.get("Cache-Control"), "no-store")
      assert.equal(typeof (await response.json()).detail, "string")
    }
  })
})

test("unsupported methods are rejected with the route Allow header", async () => {
  for (const [path, method, expected] of [["/api/audit", "GET", "POST"], ["/api/telemetry", "POST", "GET"]]) {
    const response = await worker.fetch(request(path, { method }), env("https://backend.example.com"))
    assert.equal(response.status, 405)
    assert.equal(response.headers.get("Allow"), expected)
  }
})

test("audit proxy preserves body and auth, protects cookies, and disables caching", async () => {
  await mockedFetch(async (url, options) => {
    assert.equal(String(url), "https://contract-tunnel.trycloudflare.com/api/audit?trace=1")
    assert.equal(options.method, "POST")
    assert.equal(options.headers.get("Authorization"), "Bearer test")
    assert.equal(options.headers.get("Cookie"), null)
    assert.equal(options.headers.get("Origin"), null)
    assert.equal(options.redirect, "manual")
    assert.deepEqual(await new Response(options.body).json(), { query: "What is the notice period?" })
    return Response.json({ answer: "Retrieved clause. [Source 1]" }, {
      headers: { "Cache-Control": "public, max-age=3600", "Set-Cookie": "token=private" },
    })
  }, async () => {
    const response = await worker.fetch(request("/api/audit?trace=1", {
      method: "POST", body: JSON.stringify({ query: "What is the notice period?" }),
      headers: { "Content-Type": "application/json", Authorization: "Bearer test", Cookie: "session=local", Origin: "https://meethaq-ai.workers.dev" },
    }), env("https://contract-tunnel.trycloudflare.com/"))
    assert.equal(response.status, 200)
    assert.equal(response.headers.get("Cache-Control"), "no-store")
    assert.equal(response.headers.get("Set-Cookie"), null)
    assert.equal((await response.json()).answer, "Retrieved clause. [Source 1]")
  })
})

test("JSON backend errors retain their status for the client", async () => {
  await mockedFetch(async () => Response.json({ detail: "Index unavailable." }, { status: 503 }), async () => {
    const response = await worker.fetch(request(), env("https://backend.example.com"))
    assert.equal(response.status, 503)
    assert.equal((await response.json()).detail, "Index unavailable.")
  })
})

test("redirects, HTML tunnel errors, and connection failures become explicit JSON errors", async () => {
  for (const mock of [
    async () => Response.redirect("https://login.example.com", 302),
    async () => new Response("<html>Tunnel unavailable</html>", { status: 502, headers: { "Content-Type": "text/html" } }),
    async () => { throw new TypeError("Connection refused") },
  ]) {
    await mockedFetch(mock, async () => {
      const response = await worker.fetch(request(), env("https://backend.example.com"))
      assert.equal(response.status, 502)
      assert.equal(typeof (await response.json()).detail, "string")
    })
  }
})
