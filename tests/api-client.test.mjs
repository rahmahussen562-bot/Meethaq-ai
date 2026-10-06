import test, { after } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import ts from "typescript";

const oldFetch = globalThis.fetch;
const oldWindow = globalThis.window;
const timers = new Set();
globalThis.window = {
  location: { origin:"https://legal.pages.dev" },
  setTimeout: (callback, delay) => { const timer=setTimeout(callback,delay); timers.add(timer); return timer; },
  clearTimeout: (timer) => { timers.delete(timer); clearTimeout(timer); },
};
const contractUrl = new URL("../src/services/api-contract.ts",import.meta.url).href;
const source = fs.readFileSync(new URL("../src/services/api.ts",import.meta.url),"utf8")
  .replaceAll('"./api-contract"', JSON.stringify(contractUrl))
  .replaceAll("import.meta.env.VITE_API_URL", "undefined");
const output = ts.transpileModule(source,{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.ESNext}}).outputText;
const api = await import("data:text/javascript;base64,"+Buffer.from(output).toString("base64"));
after(()=> { globalThis.fetch=oldFetch; globalThis.window=oldWindow; for(const timer of timers) clearTimeout(timer); });

test("telemetry pings active same-origin API without needless CORS preflight",async()=>{
  let called;
  globalThis.fetch = async (url,init)=> { called={url,init}; return Response.json({indexed_chunks:0,total_chunks:0}); };
  assert.equal((await api.fetchTelemetry()).indexed_chunks,0);
  assert.equal(called.url,"https://legal.pages.dev/api/telemetry");
  assert.equal(called.init.headers,undefined);
  assert.equal(called.init.cache,"no-store");
  assert.equal(called.init.credentials,"omit");
  assert.equal(timers.size,0);
});
test("HTTP failures retain backend detail and status, never become offline",async()=>{
  globalThis.fetch=async()=>Response.json({detail:"Index storage unavailable."},{status:503});
  await assert.rejects(api.fetchTelemetry(),{kind:"http",status:503,message:"Index storage unavailable."});
  assert.equal(timers.size,0);
});
test("failed requests always clear timeout handles",async()=>{
  globalThis.fetch=async()=>{throw new TypeError("fetch failed");};
  await assert.rejects(api.fetchTelemetry(),{kind:"network"});
  assert.equal(timers.size,0);
});
test("proxy HTML is reported as routing failure",async()=>{
  globalThis.fetch=async()=>new Response("<html>SPA</html>",{headers:{"Content-Type":"text/html"}});
  await assert.rejects(api.fetchTelemetry(),{kind:"protocol"});
  assert.equal(timers.size,0);
});
test("request cancellation is distinct and never fabricates an answer",async()=>{
  globalThis.fetch=async(_url,init)=>new Promise((_resolve,reject)=> {
    if(init.signal.aborted) reject(new DOMException("Cancelled","AbortError"));
    else init.signal.addEventListener("abort",()=>reject(new DOMException("Cancelled","AbortError")),{once:true});
  });
  const controller=new AbortController();
  const request=api.sendAuditQuery("What termination notice applies?",false,controller.signal);
  controller.abort();
  await assert.rejects(request,{name:"AbortError"});
  assert.equal(timers.size,0);
});
test("audit posts the real question and accepts only grounded response contracts",async()=>{
  let payload;
  globalThis.fetch=async(_url,init)=> { payload=JSON.parse(init.body); return Response.json({
    answer:"I could not find an answer to this question in the provided documents.",
    sources:[],stats:{indexed_chunks:0},
  }); };
  const result=await api.sendAuditQuery("What governing law applies?");
  assert.deepEqual(payload,{query:"What governing law applies?",expand_query:false});
  assert.deepEqual(result.sources,[]);
  assert.equal(timers.size,0);
});

