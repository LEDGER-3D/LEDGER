"""Qwen3.5-9B text generation, greedy, thinking disabled -- the answering model of every example.

Two ways to reach the model, same generation settings:
  * in-process (default): the weights are downloaded from Hugging Face on first use (~19 GB, bf16,
    one 24 GB GPU);
  * --server URL: a running ledger/pipeline/qwen_server.py (the memory-construction stages use the
    same server, so one loaded model can serve both).
"""
import json, os, time, urllib.request

MODEL_ID = os.environ.get("LEDGER_QWEN", "Qwen/Qwen3.5-9B")


class Qwen:
    def __init__(self, server=None, model_id=MODEL_ID):
        self.server = server.rstrip("/") if server else None
        self.model_id = model_id
        self.model = self.proc = None
        if self.server is None:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
            t0 = time.time()
            self.proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id, dtype=torch.bfloat16, trust_remote_code=True).eval().to("cuda")
            print(f"[qwen] {model_id} loaded in {time.time()-t0:.0f}s", flush=True)

    def ask(self, prompt, max_new_tokens=320):
        if self.server:
            body = json.dumps({"prompt": prompt, "max_new_tokens": max_new_tokens, "no_think": True}).encode()
            rq = urllib.request.Request(self.server + "/generate", data=body,
                                        headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(rq, timeout=600) as r:
                return json.loads(r.read())["text"].strip()
        import torch
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        kw = dict(add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
                  enable_thinking=False)
        inputs = self.proc.apply_chat_template(messages, **kw).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                      pad_token_id=self.proc.tokenizer.eos_token_id)
        gen = out[0][int(inputs["input_ids"].shape[1]):]
        return self.proc.decode(gen, skip_special_tokens=True).strip()
