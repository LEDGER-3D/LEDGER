"""Open-source VLM as an open-vocabulary TAGGER (Qwen*-VL / Gemma-3 / any image-text-to-text model).
Prompts the model to list physical objects per grid frame, unions per clip. Same output schema as the
other taggers ({clip:{video,n_frames,tags:[...]}}). Shardable via --clips-file (write to per-shard --out,
merge after). Run in an env with a recent transformers (>=4.49).

ADDED (HD-EPIC port, ego4d path unchanged):
  --server URL   query a persistent qwen_server.py instead of loading the model in-process
                 (loads 19.3 GB once per machine instead of once per invocation)
  --dataset      ego4d (default) | hdepic — hdepic skips the Ego4D vq_val.json clip->video map
"""
import argparse, json, glob, os, re
import numpy as np
from PIL import Image
import ledger_paths as LP
DATA=LP.EGO4D
PROMPT=("List every distinct physical object visible (tools, parts, containers, furniture, equipment, "
        "vehicles). Be specific (e.g. screwdriver, bicycle seat, pliers, tire). Reply with ONLY a "
        "comma-separated list of object names, nothing else — no sentences, no numbering, no counts.")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",required=True)
    ap.add_argument("--frames",required=True); ap.add_argument("--out",required=True)
    ap.add_argument("--per-clip",type=int,default=60,help="frames/clip to tag (linspace over the grid)")
    ap.add_argument("--clips-file",default=None); ap.add_argument("--only-clip",default=None)
    ap.add_argument("--max-new-tokens",type=int,default=160)
    ap.add_argument("--no-think",action="store_true",help="pass enable_thinking=False (Qwen3 reasoning models)")
    ap.add_argument("--server",default=None,help="persistent model server URL, e.g. http://127.0.0.1:8077")
    ap.add_argument("--dataset",default="ucs", choices=["ucs"])
    a=ap.parse_args()

    if a.server:
        import urllib.request
        def raw_tags(path):
            body=json.dumps({"prompt":PROMPT,"images":[path],
                             "max_new_tokens":a.max_new_tokens,"no_think":bool(a.no_think)}).encode()
            rq=urllib.request.Request(a.server.rstrip("/")+"/generate",data=body,
                                      headers={"Content-Type":"application/json"})
            with urllib.request.urlopen(rq,timeout=300) as r:
                return json.loads(r.read())["text"]
        print(f"tagger using server {a.server}",flush=True)
    else:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        proc=AutoProcessor.from_pretrained(a.model, trust_remote_code=True)
        # avoid device_map (needs accelerate); load then move to the single visible GPU
        model=AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16,
                                                          trust_remote_code=True).eval().to("cuda")
        print("VLM loaded:",a.model,flush=True)
        def raw_tags(path):
            img=Image.open(path).convert("RGB")
            messages=[{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":PROMPT}]}]
            kw=dict(add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors="pt")
            if a.no_think: kw["enable_thinking"]=False
            try: inputs=proc.apply_chat_template(messages,**kw).to(model.device)
            except TypeError:
                kw.pop("enable_thinking",None); inputs=proc.apply_chat_template(messages,**kw).to(model.device)
            with torch.no_grad():
                out=model.generate(**inputs,max_new_tokens=a.max_new_tokens,do_sample=False)
            return proc.decode(out[0][inputs["input_ids"].shape[1]:],skip_special_tokens=True)

    vidof={}
    if a.dataset=="ego4d":
        vqv=json.load(open(DATA+"/annotations/vq_val.json"))
        for v in vqv["videos"]:
            for c in v["clips"]: vidof[c["clip_uid"]]=v["video_uid"]
    _dbg=[os.getenv("DBG_RAW")=="1"]
    def clean(txt):
        if _dbg[0]: print("  RAW>>>",repr(txt[:300]),flush=True); _dbg[0]=False
        # drop any reasoning block, then keep comma-list lines only
        txt=re.sub(r"<think>.*?</think>","",txt,flags=re.S)
        if "</think>" in txt: txt=txt.split("</think>")[-1]
        toks=[re.sub(r"^[\-\*\d\.\)\s]+","",w).strip().lower() for w in re.split(r"[,\n;]+",txt)]
        bad=("here","the ","i can","this ","object","list","okay","so ","let","no ","got","first","initial",
             "looks","wearing","image","comma","sure","of course","below","following","some ")
        out_t=[]
        for w in toks:
            if not (1<len(w)<40): continue
            if w.startswith(bad) or w.endswith((".",":","*")) or "**" in w or "'" in w: continue
            if len(w.split())>4: continue          # object names are short; drop prose fragments
            out_t.append(w)
        return out_t
    if a.only_clip: clips=[a.only_clip]
    elif a.clips_file: clips=[l.strip() for l in open(a.clips_file) if l.strip()]
    else: clips=sorted(d for d in os.listdir(a.frames) if os.path.isdir(f"{a.frames}/{d}"))
    out=json.load(open(a.out)) if os.path.isfile(a.out) else {}
    for cu in clips:
        if cu in out: continue
        imgs=sorted(glob.glob(f"{a.frames}/{cu}/*.jpg"))
        sel=[imgs[i] for i in np.linspace(0,len(imgs)-1,min(a.per_clip,len(imgs))).astype(int)] if imgs else []
        tags=set()
        for p in sel:
            try: tags.update(clean(raw_tags(p)))
            except Exception as e: print("  err",str(e)[:90],flush=True)
        out[cu]={"video":vidof.get(cu,cu if a.dataset in ("hdepic","ucs") else None),"n_frames":len(sel),"tags":sorted(tags)}
        json.dump(out,open(a.out,"w"),indent=1)
        print(f"[{cu[:8]}] {len(sel)}fr -> {len(tags)} tags",flush=True)
    print(f"DONE {len(out)} clips mean {np.mean([len(v['tags']) for v in out.values()]):.1f} -> {a.out}",flush=True)
if __name__=="__main__": main()
