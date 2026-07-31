#!/usr/bin/env python3
"""Zero-dependency document-QA UI for the GLM-5.2 sglang server.

The whole document lives in the model's context, not in a retriever. It is sent
as one fixed system message so that sglang's radix cache matches it as a prefix:
the first question pays the full prefill, every question after it reuses that KV
and only prefills its own tokens.

Two things follow from that, and both are load-bearing:

  * the prefix must be byte-identical on every request, so the document is held
    server-side here and stitched in on the way through -- the browser never
    sends it, and cannot perturb it.
  * the conversation grows by appending only, so each turn extends the cached
    prefix instead of invalidating it.

Serves the page on :8081 and proxies to the model on :8000, so the browser talks
same-origin.

Run:  python3 doc_ui.py [ui_port] [document_path]
"""
import json
import os
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_BASE = os.environ.get("MODEL_BASE", "http://127.0.0.1:8000")

# This module is also imported (by bench/doc_context_bench.py) purely to reuse
# SYSTEM_PROMPT, so that the bench and the UI send a byte-identical prefix and
# share one radix-cache entry. Parse argv leniently so an importer's own flags
# don't blow up at import time.
_args = [a for a in sys.argv[1:] if not a.startswith("-")]
UI_PORT = int(_args[0]) if _args and _args[0].isdigit() else 8081
_doc_args = [a for a in _args if not a.isdigit()]
DOC_PATH = _doc_args[0] if _doc_args else "unlimited_ocr.txt"
# Resolve relative to this file, not the caller's cwd, so importers and
# launchers started from anywhere still find the document.
if not os.path.isabs(DOC_PATH):
    DOC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), DOC_PATH)

with open(DOC_PATH, encoding="utf-8") as fh:
    DOCUMENT = fh.read()

DOC_NAME = os.path.basename(DOC_PATH)

# The document is OCR of a page-numbered RFP and keeps its "===== PAGE n OF m ====="
# separators, so the model can cite where an answer came from. That matters more
# than usual here: at half a million tokens the user cannot eyeball the source.
SYSTEM_PROMPT = (
    "You are answering questions about a single document that is reproduced in "
    "full below. It is OCR text, so tables appear as HTML and there may be "
    "scanning errors; read through minor typos.\n\n"
    "Rules:\n"
    "1. Answer only from the document. If it does not say, reply that it does "
    "not say -- do not fill the gap from general knowledge.\n"
    "2. Cite the page number(s) you used, e.g. (p. 143). The document is split "
    "by lines of the form '===== PAGE n OF m ====='.\n"
    "3. Quote the exact wording when the question turns on specific language "
    "(eligibility thresholds, penalties, dates, amounts).\n"
    "4. If the document is inconsistent or says different things in different "
    "places, say so and cite each place.\n\n"
    # Deliberately no filename here. The prompt is the radix-cache key, so
    # anything that varies with the file on disk -- a rename, a move -- would
    # silently invalidate a ~26-minute prefill.
    "=== BEGIN DOCUMENT ===\n"
    f"{DOCUMENT}\n"
    "=== END DOCUMENT ==="
)

PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__DOCNAME__ — GLM-5.2</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
         background:#0d1117; color:#e6edf3; display:flex; flex-direction:column; height:100vh; }
  header { padding:10px 16px; border-bottom:1px solid #21262d; display:flex; align-items:center; gap:12px; flex-wrap:wrap; }
  header h1 { font-size:15px; margin:0; font-weight:600; }
  header .meta { font-size:12px; color:#7d8590; }
  header .spacer { margin-left:auto; }
  .pill { font-size:11px; padding:2px 8px; border-radius:999px; border:1px solid #30363d; color:#9aa4ad; }
  .pill.warm { border-color:#238636; color:#3fb950; }
  .pill.cold { border-color:#9e6a03; color:#d29922; }
  #log { flex:1; overflow-y:auto; padding:16px; display:flex; flex-direction:column; gap:14px; }
  .msg { max-width:900px; width:100%; margin:0 auto; }
  .role { font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:#7d8590; margin-bottom:4px; }
  .bubble { white-space:pre-wrap; line-height:1.55; padding:10px 14px; border-radius:8px; overflow-wrap:anywhere; }
  .user .bubble { background:#1f6feb22; border:1px solid #1f6feb55; }
  .assistant .bubble { background:#161b22; border:1px solid #21262d; }
  .system .bubble { background:#0b0f14; border:1px dashed #30363d; color:#9aa4ad; font-size:13px; }
  details.reasoning { margin:0 0 8px 0; border:1px dashed #30363d; border-radius:8px; background:#0b0f14; }
  details.reasoning > summary { cursor:pointer; padding:6px 12px; font-size:12px; color:#9aa4ad; }
  details.reasoning .rc { white-space:pre-wrap; padding:0 12px 10px; font-size:13px; color:#8b949e; line-height:1.45; }
  footer { border-top:1px solid #21262d; padding:10px 16px; }
  .row { max-width:900px; margin:0 auto; display:flex; gap:8px; align-items:flex-end; }
  textarea { flex:1; resize:none; background:#0d1117; color:#e6edf3; border:1px solid #30363d;
             border-radius:8px; padding:10px 12px; font:inherit; min-height:44px; max-height:200px; }
  button { background:#238636; color:#fff; border:0; border-radius:8px; padding:0 18px; height:44px;
           font:inherit; font-weight:600; cursor:pointer; }
  button.sec { background:#21262d; color:#c9d1d9; height:28px; padding:0 12px; font-size:12px; font-weight:500; }
  button.danger { background:#8b2c2c; }
  .note { font-size:12px; color:#d29922; margin-top:6px; }
  button:disabled { background:#30363d; color:#7d8590; cursor:not-allowed; }
  .ctrls { max-width:900px; margin:6px auto 0; display:flex; gap:14px; font-size:12px; color:#7d8590;
           align-items:center; flex-wrap:wrap; }
  .ctrls input[type=number] { width:70px; background:#0d1117; color:#e6edf3; border:1px solid #30363d; border-radius:6px; padding:3px 6px; }
  select { background:#0d1117; color:#e6edf3; border:1px solid #30363d; border-radius:6px; padding:3px 6px; font:inherit; font-size:12px; }
  .stat { color:#3fb950; margin-left:auto; font-variant-numeric:tabular-nums; }
  .err { color:#f85149; }
</style></head>
<body>
  <header>
    <h1 id="title">GLM-5.2</h1>
    <span class="meta" id="docmeta">__DOCNAME__</span>
    <span class="spacer"></span>
    <span class="pill" id="cache">cache: unknown</span>
    <button class="sec" id="warm">Warm cache</button>
    <button class="sec" id="clear">New chat</button>
  </header>
  <div id="log"></div>
  <footer>
    <div class="row">
      <textarea id="inp" placeholder="Ask about the document…  (Enter to send, Shift+Enter for newline)"></textarea>
      <button id="send">Ask</button>
      <button id="stop" class="danger" style="display:none">Stop</button>
    </div>
    <div class="ctrls">
      <!-- Reasoning is billed against this too, and on a 533k context the model
           thinks for a while before answering; 1200 truncated mid-thought. -->
      <label>max_tokens <input type="number" id="maxtok" value="2500" min="1" max="8000"></label>
      <label>temperature <input type="number" id="temp" value="0.2" min="0" max="2" step="0.1"></label>
      <label>intelligence
        <select id="tier" title="genuine top experts kept per token: higher = more faithful, lower = faster">
          <option value="8" selected>top-8 · exact routing</option>
          <option value="4">top-4 · balanced</option>
          <option value="2">top-2 · fast</option>
        </select>
      </label>
      <label title="Applies a mild frequency penalty and stops generation if the output starts repeating itself">
        <input type="checkbox" id="guard" checked> repetition guard
      </label>
      <span id="stat" class="stat"></span>
    </div>
  </footer>
<script>
const log = document.getElementById('log');
const inp = document.getElementById('inp');
const sendBtn = document.getElementById('send');
const stopBtn = document.getElementById('stop');
const cachePill = document.getElementById('cache');
const stat = document.getElementById('stat');
let history = [];

// In-flight request: the AbortController drops our end of the stream, and `rid`
// lets the server actually cancel the generation instead of leaving it running
// on the GPU with nobody reading it.
let inflight = null;

async function cancelInflight(reason){
  if(!inflight) return;
  const {rid, controller} = inflight;
  inflight = null;
  try { controller.abort(); } catch(e){}
  try {
    await fetch('/api/stop', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({rid: rid})
    });
  } catch(e){}
  if(reason) stat.textContent = reason;
}

// Degenerate output is a tail that is literally the same span over and over --
// "0,0,0,..." at greedy decoding, or a reasoning step restated verbatim. Look for
// a unit that repeats three times back to back at the very end of the stream.
// Requiring three exact consecutive repeats keeps legitimately repetitive text
// (page citations, table rows) from tripping it.
function detectLoop(s){
  const t = s.slice(-800);
  for(let u = 8; u <= 240; u++){
    if(t.length < u*3) break;
    const unit = t.slice(-u);
    if(unit.trim().length < 4) continue;   // whitespace runs are not loops
    if(t.endsWith(unit.repeat(3))) return unit;
  }
  return null;
}

fetch('/api/status').then(r=>r.json()).then(j=>{
  document.getElementById('title').textContent = j.model || 'GLM-5.2';
  document.getElementById('docmeta').textContent =
    j.doc + ' · ' + j.doc_chars.toLocaleString() + ' chars in context';
}).catch(()=>{});

function el(cls, html){ const d=document.createElement('div'); d.className=cls; if(html!==undefined) d.innerHTML=html; return d; }
function addMsg(role){
  const m = el('msg '+role);
  m.appendChild(el('role', role));
  if(role==='assistant'){
    const det=document.createElement('details'); det.className='reasoning';
    const sum=document.createElement('summary'); sum.textContent='💭 reasoning'; det.appendChild(sum);
    const rc=el('rc'); det.appendChild(rc); det.style.display='none';
    m.appendChild(det); m._det=det; m._rc=rc;
  }
  const b=el('bubble'); m.appendChild(b); m._b=b;
  log.appendChild(m); log.scrollTop=log.scrollHeight; return m;
}
function setCache(state, text){
  cachePill.className = 'pill ' + state;
  cachePill.textContent = 'cache: ' + text;
}

// Streams one turn. `warm` sends no question and caps generation at a single
// token -- it exists purely to pay the document prefill once, up front, so the
// first real question is not the one that waits.
async function run(text, warm){
  sendBtn.disabled = true;
  stopBtn.style.display = '';
  const am = warm ? null : addMsg('assistant');
  if(!warm) stat.textContent = 'thinking…';

  const t0 = performance.now();
  let content='', reasoning='', usage=null, firstTokT=null;
  let stoppedFor = null;

  const guard = document.getElementById('guard').checked;
  const rid = 'ui-' + (crypto.randomUUID ? crypto.randomUUID() : Date.now()+'-'+Math.random());
  const controller = new AbortController();
  inflight = {rid: rid, controller: controller};

  try{
    const resp = await fetch('/api/ask', {
      method:'POST', headers:{'Content-Type':'application/json'},
      signal: controller.signal,
      body: JSON.stringify({
        history: history,
        warm: !!warm,
        rid: rid,
        // Penalising tokens the model has already emitted discourages loops
        // before they start. sglang accumulates this over output tokens only
        // (BatchedFrequencyPenalizer._cumulate_output_tokens), so the 533k-token
        // prompt does not feed into it.
        frequency_penalty: guard ? 0.3 : 0.0,
        tier: document.getElementById('tier').value,
        temperature: +document.getElementById('temp').value,
        max_tokens: warm ? 1 : Math.max(1, Math.min(8000, +document.getElementById('maxtok').value || 1200))
      })
    });
    if(!resp.ok){
      const msg = 'HTTP ' + resp.status + ': ' + await resp.text();
      if(am) am._b.innerHTML = '<span class="err">'+msg+'</span>'; else stat.innerHTML='<span class="err">'+msg+'</span>';
      return;
    }

    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf='';
    while(true){
      const {value, done} = await reader.read();
      if(done) break;
      buf += dec.decode(value, {stream:true});
      let nl;
      while((nl = buf.indexOf('\n')) >= 0){
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl+1);
        if(!line.startsWith('data:')) continue;
        const data = line.slice(5).trim();
        if(data === '[DONE]') continue;
        let j; try{ j = JSON.parse(data); }catch(e){ continue; }
        if(j.usage) usage = j.usage;
        const d = ((j.choices && j.choices[0]) || {}).delta || {};
        if(d.reasoning_content){
          if(firstTokT===null) firstTokT = performance.now();
          reasoning += d.reasoning_content;
          if(am){ am._det.style.display=''; am._rc.textContent = reasoning; }
        }
        if(d.content){
          if(firstTokT===null) firstTokT = performance.now();
          content += d.content;
          if(am) am._b.textContent = content;
        }
        log.scrollTop = log.scrollHeight;
      }

      if(guard && !warm && !stoppedFor){
        // Check the stream the model is actually producing right now.
        const unit = detectLoop(content || reasoning);
        if(unit){
          stoppedFor = unit.trim().slice(0, 40);
          await cancelInflight(null);
          break;
        }
      }
    }

    const ttft = firstTokT ? (firstTokT - t0)/1000 : (performance.now()-t0)/1000;
    const gen  = usage ? usage.completion_tokens : 0;
    const rate = (gen && firstTokT) ? gen / ((performance.now()-firstTokT)/1000) : 0;
    const parts = ['TTFT ' + ttft.toFixed(1) + 's'];
    if(rate) parts.push(rate.toFixed(1) + ' tok/s');
    if(usage) parts.push(usage.prompt_tokens.toLocaleString() + ' prompt tok');
    stat.textContent = parts.join(' · ');

    // A warm prefix answers in seconds; a cold one takes minutes, so TTFT is a
    // reliable readout of whether the document KV survived.
    setCache(ttft < 60 ? 'warm' : 'cold', ttft < 60 ? 'warm' : 'cold (' + ttft.toFixed(0) + 's prefill)');

    if(stoppedFor && am){
      const n = el('note');
      n.textContent = '⚠ Stopped: the model began repeating "' + stoppedFor +
                      '…". Try a lower tier, a higher temperature, or rephrase.';
      am.appendChild(n);
    }

    if(!warm && content) history.push({role:'assistant', content: content});
  } catch(e){
    // An abort is a deliberate stop, not a failure.
    if(e && e.name === 'AbortError'){
      if(am && !content && !reasoning) am._b.textContent = '(stopped)';
      if(!stoppedFor) stat.textContent = 'stopped';
      if(!warm && content) history.push({role:'assistant', content: content});
    } else {
      const msg = String(e);
      if(am) am._b.innerHTML = '<span class="err">'+msg+'</span>';
      else stat.innerHTML = '<span class="err">'+msg+'</span>';
    }
  } finally {
    inflight = null;
    sendBtn.disabled = false;
    stopBtn.style.display = 'none';
    inp.focus();
  }
}

async function send(){
  const text = inp.value.trim();
  if(!text) return;
  inp.value='';
  addMsg('user')._b.textContent = text;
  history.push({role:'user', content:text});
  await run(text, false);
}

sendBtn.onclick = send;
stopBtn.onclick = ()=> cancelInflight('stopped');
inp.addEventListener('keydown', e=>{
  if(e.key==='Enter' && !e.shiftKey){ e.preventDefault(); send(); }
  if(e.key==='Escape') cancelInflight('stopped');
});
document.getElementById('warm').onclick = ()=>{
  setCache('cold','warming…');
  stat.textContent = 'prefilling the document — first time takes a while…';
  run('', true);
};
document.getElementById('clear').onclick = ()=>{
  // Only the turns are dropped. The document prefix is server-side and stays
  // cached, so a new chat is instant rather than another full prefill.
  history = []; log.innerHTML='';
  const m = addMsg('system');
  m._b.textContent = 'New chat. The document stays loaded in the model context.';
};
inp.focus();
</script>
</body></html>
"""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # quiet
        pass

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            page = PAGE.replace("__DOCNAME__", DOC_NAME)
            self._send(200, page, "text/html; charset=utf-8")
        elif self.path == "/api/status":
            model = "GLM-5.2"
            try:
                with urllib.request.urlopen(f"{MODEL_BASE}/v1/models", timeout=5) as r:
                    model = json.load(r)["data"][0]["id"]
            except Exception:
                pass
            self._send(200, json.dumps({
                "model": model, "doc": DOC_NAME, "doc_chars": len(DOCUMENT),
            }))
        else:
            self._send(404, b"not found", "text/plain")

    def _abort_upstream(self, rid):
        """Cancel a generation server-side. Dropping the socket is not enough --
        without this the model keeps decoding into a stream nobody is reading."""
        if not rid:
            return
        try:
            req = urllib.request.Request(
                f"{MODEL_BASE}/abort_request",
                data=json.dumps({"rid": rid}).encode(),
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=5).close()
        except Exception:
            pass

    def do_POST(self):
        if self.path not in ("/api/ask", "/api/stop"):
            self._send(404, b"not found", "text/plain")
            return

        try:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        except Exception as exc:
            self._send(400, json.dumps({"error": str(exc)}))
            return

        if self.path == "/api/stop":
            self._abort_upstream(body.get("rid"))
            self._send(200, json.dumps({"stopped": True}))
            return

        model = "GLM5.2"
        try:
            with urllib.request.urlopen(f"{MODEL_BASE}/v1/models", timeout=5) as r:
                model = json.load(r)["data"][0]["id"]
        except Exception:
            pass

        # The tier suffix picks the expert-routing tier live, per request.
        tier = str(body.get("tier", "8"))
        if tier and tier != "8":
            model = f"{model}-top{tier}"

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if body.get("warm"):
            # Something must follow the prefix for the server to generate at all;
            # keep it trivial so the prefill is the only real work.
            messages.append({"role": "user", "content": "Reply with OK."})
        else:
            messages.extend(body.get("history", []))

        # A caller-supplied rid is what makes /abort_request usable: sglang's own
        # chatcmpl- id is not the rid (serving_base._generate_request_id_base
        # returns None), so without this there is no handle to cancel by.
        rid = body.get("rid")

        payload = json.dumps({
            "model": model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": body.get("temperature", 0.2),
            "max_tokens": body.get("max_tokens", 1200),
            "frequency_penalty": body.get("frequency_penalty", 0.0),
            **({"rid": rid} if rid else {}),
        }).encode()

        req = urllib.request.Request(
            f"{MODEL_BASE}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        try:
            # No read timeout: a cold document prefill legitimately runs for many
            # minutes before the first byte comes back.
            upstream = urllib.request.urlopen(req)
        except urllib.error.HTTPError as exc:
            self._send(exc.code, exc.read(), "text/plain")
            return
        except Exception as exc:
            self._send(502, str(exc).encode(), "text/plain")
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        client_gone = False
        try:
            for chunk in upstream:
                self.wfile.write(chunk)
                self.wfile.flush()
        except Exception:
            # Browser hung up (Stop, tab closed, tunnel dropped).
            client_gone = True
        finally:
            upstream.close()
            if client_gone:
                self._abort_upstream(rid)


if __name__ == "__main__":
    print(f"Document QA UI on http://0.0.0.0:{UI_PORT}  (model -> {MODEL_BASE})")
    print(f"Document: {DOC_PATH}  ({len(DOCUMENT):,} chars)")
    ThreadingHTTPServer(("0.0.0.0", UI_PORT), Handler).serve_forever()
