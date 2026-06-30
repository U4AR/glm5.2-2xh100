#!/usr/bin/env python3
"""Minimal zero-dependency chat UI for the GLM-5.2 sglang server.

Talks to the OpenAI-compatible endpoint (/v1/chat/completions) with streaming,
so the server's chat template + glm45 reasoning parser do the work: the UI just
sends a `messages` array and renders `delta.content` (answer) separately from
`delta.reasoning_content` (the model's thinking).

Serves an HTML chat page on :8080 and proxies /v1/* to the model on :8000 so the
browser talks same-origin (no CORS).

Run:  python3 chat_ui.py            (or ./start_ui.sh)
Then open the forwarded port 8080 in your browser.
"""
import json
import sys
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UI_PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
MODEL_BASE = "http://127.0.0.1:8000"

PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>GLM-5.2 Chat</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
         background:#0d1117; color:#e6edf3; display:flex; flex-direction:column; height:100vh; }
  header { padding:10px 16px; border-bottom:1px solid #21262d; display:flex; align-items:center; gap:12px; }
  header h1 { font-size:15px; margin:0; font-weight:600; }
  header .meta { font-size:12px; color:#7d8590; margin-left:auto; }
  #log { flex:1; overflow-y:auto; padding:16px; display:flex; flex-direction:column; gap:14px; }
  .msg { max-width:820px; width:100%; margin:0 auto; }
  .role { font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:#7d8590; margin-bottom:4px; }
  .bubble { white-space:pre-wrap; line-height:1.5; padding:10px 14px; border-radius:8px; }
  .user .bubble { background:#1f6feb22; border:1px solid #1f6feb55; }
  .assistant .bubble { background:#161b22; border:1px solid #21262d; }
  details.reasoning { margin:0 0 8px 0; border:1px dashed #30363d; border-radius:8px; background:#0b0f14; }
  details.reasoning > summary { cursor:pointer; padding:6px 12px; font-size:12px; color:#9aa4ad; }
  details.reasoning .rc { white-space:pre-wrap; padding:0 12px 10px; font-size:13px; color:#8b949e; line-height:1.45; }
  footer { border-top:1px solid #21262d; padding:10px 16px; }
  .row { max-width:820px; margin:0 auto; display:flex; gap:8px; align-items:flex-end; }
  textarea { flex:1; resize:none; background:#0d1117; color:#e6edf3; border:1px solid #30363d;
             border-radius:8px; padding:10px 12px; font:inherit; min-height:44px; max-height:200px; }
  button { background:#238636; color:#fff; border:0; border-radius:8px; padding:0 18px; height:44px;
           font:inherit; font-weight:600; cursor:pointer; }
  button:disabled { background:#30363d; color:#7d8590; cursor:not-allowed; }
  .ctrls { max-width:820px; margin:6px auto 0; display:flex; gap:14px; font-size:12px; color:#7d8590; align-items:center; }
  .ctrls input[type=number] { width:64px; background:#0d1117; color:#e6edf3; border:1px solid #30363d; border-radius:6px; padding:3px 6px; }
  .stat { color:#3fb950; }
</style></head>
<body>
  <header>
    <h1 id="title">GLM-5.2</h1>
    <span class="meta" id="meta">2×H100 + kt-kernel · OpenAI /v1/chat/completions</span>
  </header>
  <div id="log"></div>
  <footer>
    <div class="row">
      <textarea id="inp" placeholder="Ask something…  (Enter to send, Shift+Enter for newline)"></textarea>
      <button id="send">Send</button>
    </div>
    <div class="ctrls">
      <label>max_tokens <input type="number" id="maxtok" placeholder="adaptive" min="1" max="8000"></label>
      <label>temperature <input type="number" id="temp" value="0.6" min="0" max="2" step="0.1"></label>
      <label><input type="checkbox" id="reset"> new chat each send</label>
      <span id="stat" class="stat"></span>
    </div>
  </footer>
<script>
const log = document.getElementById('log');
const inp = document.getElementById('inp');
const sendBtn = document.getElementById('send');
let history = [];
let MODEL = 'GLM5.2';

// discover the served model name so the request matches the server
fetch('/v1/models').then(r=>r.json()).then(j=>{
  if(j && j.data && j.data[0] && j.data[0].id){
    MODEL = j.data[0].id;
    document.getElementById('title').textContent = MODEL;
  }
}).catch(()=>{});

function el(cls, html){ const d=document.createElement('div'); d.className=cls; if(html!==undefined) d.innerHTML=html; return d; }
function addMsg(role){
  const m = el('msg '+role);
  m.appendChild(el('role', role));
  if(role==='assistant'){
    const det = document.createElement('details'); det.className='reasoning'; det.open=true;
    const sum=document.createElement('summary'); sum.textContent='💭 reasoning'; det.appendChild(sum);
    const rc=el('rc'); det.appendChild(rc); det.style.display='none';
    m.appendChild(det); m._det=det; m._rc=rc;
  }
  const b = el('bubble'); m.appendChild(b); m._b=b;
  log.appendChild(m); log.scrollTop=log.scrollHeight; return m;
}

async function send(){
  const text = inp.value.trim();
  if(!text) return;

  if(document.getElementById('reset').checked) history=[];

  inp.value='';
  sendBtn.disabled=true;

  addMsg('user')._b.textContent=text;
  history.push({role:'user', content:text});

  const am = addMsg('assistant');
  const stat=document.getElementById('stat');
  stat.textContent='…';

  const t0=performance.now();
  let content='', reasoning='', usage=null;
  const reqBody = {
    model: MODEL,
    messages: history,
    stream: true,
    stream_options: {include_usage: true},
    temperature: +document.getElementById('temp').value
  };
  const maxTokRaw = document.getElementById('maxtok').value.trim();
  if(maxTokRaw){
    reqBody.max_tokens = Math.max(1, Math.min(8000, +maxTokRaw));
  }

  try{
    const resp = await fetch('/v1/chat/completions', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify(reqBody)
    });

    if(!resp.ok){
      am._b.textContent='HTTP '+resp.status+': '+await resp.text();
      throw 0;
    }

    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf='';
    let firstTokT=null;

    while(true){
      const {value, done} = await reader.read();
      if(done) break;
      buf += dec.decode(value, {stream:true});
      let nl;
      while((nl = buf.indexOf('\n')) >= 0){
        let line = buf.slice(0, nl).trim();
        buf = buf.slice(nl+1);
        if(!line.startsWith('data:')) continue;
        const data = line.slice(5).trim();
        if(data === '[DONE]') continue;
        let j; try{ j = JSON.parse(data); }catch(e){ continue; }
        if(j.usage) usage = j.usage;
        const ch = (j.choices && j.choices[0]) || {};
        const d = ch.delta || {};
        if(d.reasoning_content){
          if(firstTokT===null) firstTokT=performance.now();
          reasoning += d.reasoning_content;
          am._det.style.display='';
          am._rc.textContent = reasoning;
        }
        if(d.content){
          if(firstTokT===null) firstTokT=performance.now();
          content += d.content;
          am._b.textContent = content;
        }
        log.scrollTop=log.scrollHeight;
      }
    }

    // collapse the reasoning panel once the answer is in
    if(reasoning && content) am._det.open=false;
    if(!content && !reasoning) am._b.textContent='[empty response]';
    history.push({role:'assistant', content});

    const secs = (performance.now()-t0)/1000;
    if(usage && usage.completion_tokens){
      const genSecs = firstTokT ? (performance.now()-firstTokT)/1000 : secs;
      stat.textContent =
        usage.completion_tokens + ' tok · ' +
        (usage.completion_tokens / genSecs).toFixed(2) + ' tok/s · ' +
        secs.toFixed(1) + 's e2e';
    } else {
      stat.textContent = secs.toFixed(2) + 's';
    }
    log.scrollTop=log.scrollHeight;
  }catch(e){
    if(!am._b.textContent) am._b.textContent='[error] '+e;
  }finally{
    sendBtn.disabled=false;
    inp.focus();
  }
}
sendBtn.onclick=send;
inp.addEventListener('keydown', e=>{ if(e.key==='Enter' && !e.shiftKey){ e.preventDefault(); send(); }});
inp.focus();
</script>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # quiet
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/v1/"):
            self._proxy("GET")
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path.startswith("/v1/") or self.path == "/generate":
            self._proxy("POST")
        else:
            self.send_error(404)

    def _proxy(self, method):
        body = None
        if method == "POST":
            n = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(n)
        req = urllib.request.Request(
            MODEL_BASE + self.path, data=body,
            headers={"Content-Type": "application/json"}, method=method)
        try:
            up = urllib.request.urlopen(req, timeout=600)
        except urllib.error.HTTPError as e:
            msg = e.read()
            self.send_response(e.code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)
            return
        except Exception as e:
            self.send_error(502, str(e))
            return
        # stream upstream -> client
        self.send_response(200)
        self.send_header("Content-Type", up.headers.get("Content-Type", "text/event-stream"))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            while True:
                line = up.readline()
                if not line:
                    break
                self.wfile.write(line)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


if __name__ == "__main__":
    print(f"Chat UI on http://0.0.0.0:{UI_PORT}  (proxying -> {MODEL_BASE})")
    print(f"Open the forwarded port {UI_PORT} in your browser.")
    ThreadingHTTPServer(("0.0.0.0", UI_PORT), Handler).serve_forever()
