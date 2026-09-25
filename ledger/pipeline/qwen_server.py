"""Persistent Qwen3.5-9B server: load the 19.3 GB model ONCE, serve every pipeline stage over HTTP.

Used by tagging (vision), tag canonicalization, tracker-collapse name grouping, and retrieval, so the
model is loaded once per machine instead of once per clip per stage.

Backend note: vLLM cannot serve this model -- neither installed vllm (0.10.2 / 0.7.3) registers
Qwen3_5ForConditionalGeneration. transformers 5.12.1 in `wilddet3d` does, and is the loading path
already proven by run_tagger_hf_vlm.py, so we use it. Same HTTP contract either way, so a vLLM
backend can be swapped in later without touching callers.

  CUDA_VISIBLE_DEVICES=0 python qwen_server.py --port 8077
  POST /generate {"prompt": str, "images": [paths], "max_new_tokens": int, "no_think": bool}
       -> {"text": str, "usage": {input_tokens, output_tokens, reasoning_tokens, total_tokens}}
  GET  /health   -> {"ok": true, "model": ..., "device": ..., "served": N}
"""
import argparse, json, threading, time, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATE = {"model": None, "proc": None, "lock": threading.Lock(), "served": 0, "name": None}

def load(model_id):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    print(f"[server] loading {model_id} ...", flush=True)
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    # match run_tagger_hf_vlm.py exactly: no device_map (avoids accelerate), move to the visible GPU
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch.bfloat16, trust_remote_code=True).eval().to("cuda")
    STATE.update(model=model, proc=proc, name=model_id)
    print(f"[server] READY in {time.time()-t0:.0f}s on {model.device}", flush=True)

def generate(prompt, images=None, max_new_tokens=160, no_think=True):
    import torch
    from PIL import Image
    model, proc = STATE["model"], STATE["proc"]
    content = []
    for p in (images or []):
        content.append({"type": "image", "image": Image.open(p).convert("RGB")})
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    kw = dict(add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
    if no_think: kw["enable_thinking"] = False
    try:
        inputs = proc.apply_chat_template(messages, **kw).to(model.device)
    except TypeError:
        kw.pop("enable_thinking", None)
        inputs = proc.apply_chat_template(messages, **kw).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    n_in = int(inputs["input_ids"].shape[1]); gen = out[0][n_in:]
    txt = proc.decode(gen, skip_special_tokens=True)
    n_out = int(gen.shape[0])
    # thinking is disabled by default (no_think); if it is ever enabled, everything up to </think>
    # is reasoning and must be reported separately rather than silently folded into the answer.
    n_think = 0
    if "</think>" in txt:
        head = txt.split("</think>", 1)[0]
        try: n_think = len(proc.tokenizer(head, add_special_tokens=False)["input_ids"])
        except Exception: n_think = 0
    return txt, {"input_tokens": n_in, "output_tokens": n_out, "reasoning_tokens": n_think,
                 "total_tokens": n_in + n_out}

class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path.startswith("/health"):
            m = STATE["model"]
            self._send(200, {"ok": m is not None, "model": STATE["name"],
                             "device": str(getattr(m, "device", None)), "served": STATE["served"]})
        else: self._send(404, {"error": "not found"})
    def do_POST(self):
        if not self.path.startswith("/generate"):
            self._send(404, {"error": "not found"}); return
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            self._send(400, {"error": f"bad json: {e}"}); return
        try:
            # one GPU, one generate at a time; concurrent callers queue rather than OOM
            with STATE["lock"]:
                txt, usage = generate(req.get("prompt", ""), req.get("images"),
                                      int(req.get("max_new_tokens", 160)), bool(req.get("no_think", True)))
                STATE["served"] += 1
            self._send(200, {"text": txt, "usage": usage})
        except Exception as e:
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--port", type=int, default=8077)
    a = ap.parse_args()
    load(a.model)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    print(f"[server] listening on http://127.0.0.1:{a.port}", flush=True)
    srv.serve_forever()
