import test from "node:test";
import assert from "node:assert/strict";
import { resolveApiBaseUrl, parseTelemetry, parseAuditResponse, sourceForCitation, NO_ANSWER } from "../src/services/api-contract.ts";

const source = { label: "[Source 1]", source: "Vendor Agreement", chunk_index: 4, chunk_id: "vendor-4", rerank_score: 0.8, text: "The agreement renews annually. Notice must be given thirty days before renewal." };
const stats = { collection: "meethaq_contracts", indexed_chunks: 42 };

test("local URL default is isolated from remote deployments", () => {
  assert.equal(resolveApiBaseUrl(undefined, "http://localhost:8443"), "http://127.0.0.1:8000");
  assert.equal(resolveApiBaseUrl("  ", "http://127.0.0.1:8443"), "http://127.0.0.1:8000");
  for (const origin of ["https://legal.pages.dev", "https://legal.workers.dev", "https://legal.example"]) assert.equal(resolveApiBaseUrl(undefined, origin), origin);
});
test("tunnel and same-origin API bases normalize consistently", () => {
  assert.equal(resolveApiBaseUrl(" https://legal.trycloudflare.com/api/ ", "https://legal.pages.dev"), "https://legal.trycloudflare.com");
  assert.equal(resolveApiBaseUrl("/proxy/api", "https://legal.pages.dev"), "https://legal.pages.dev/proxy");
});
test("misconfigured URLs produce configuration errors before requests", () => {
  for (const config of ["127.0.0.1:8000", "http://127.0.0.1:8000", "http://localhost:8000", "http://backend.example", "https://user:password@backend.example", "https://backend.example?key=secret", "https://backend.example#api", "javascript:alert(1)"]) {
    assert.throws(() => resolveApiBaseUrl(config, "https://legal.pages.dev"), { kind: "configuration" });
  }
});
test("telemetry preserves real zero and rejects fabricated or mismatched counts", () => {
  assert.equal(parseTelemetry({ indexed_chunks: 0 }).indexed_chunks, 0);
  assert.equal(parseTelemetry({ total_chunks: 19 }).indexed_chunks, 19);
  for (const value of [{}, {indexed_chunks: -1}, {indexed_chunks: "729"}, {indexed_chunks: NaN}, {indexed_chunks: 12, total_chunks: 0}]) assert.throws(() => parseTelemetry(value), {kind: "protocol"});
});
test("every accepted excerpt is verbatim and maps to the exact source label", () => {
  const response = parseAuditResponse({ answer: "The agreement renews annually. [Source 1]\n\nNotice must be given thirty days before renewal. [Source 1]", sources: [source], stats });
  assert.equal(sourceForCitation(response.sources, 1)?.chunk_id, "vendor-4");
  assert.equal(sourceForCitation([{...source, label:"[Source 2]"}], 1), null);
});
test("uncited, invented, duplicate, unused and mismatched sources fail closed", () => {
  for (const response of [
    { answer:"The agreement renews monthly. [Source 1]", sources:[source], stats },
    { answer:"The agreement renews annually.", sources:[source], stats },
    { answer:"The agreement renews annually. [Source 2]", sources:[source], stats },
    { answer:"The agreement renews annually. [source 1]", sources:[source], stats },
    { answer:"The agreement renews annually. [Source 1]", sources:[source,{...source,label:"[Source 2]"}], stats },
    { answer:"The agreement renews annually. [Source 1]", sources:[source,{...source,label:"[Source 2]",chunk_id:"other"}], stats },
    { answer:"The agreement renews annually. [Source 1]", sources:[], stats },
  ]) assert.throws(() => parseAuditResponse(response), { kind:"protocol" });
});
test("abstention stays exact and has no misleading sources", () => {
  assert.deepEqual(parseAuditResponse({answer:NO_ANSWER,sources:[],stats}).sources, []);
  assert.throws(() => parseAuditResponse({answer:NO_ANSWER,sources:[source],stats}), {kind:"protocol"});
});

