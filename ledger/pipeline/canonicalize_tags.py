"""Collapse a noisy per-clip tag vocabulary to canonical physical-object names.
LLM (Responses API, text-only) per clip: merge synonyms/duplicates -> one canonical singular name,
DROP non-objects (verbs/actions, body parts, people/roles, bare colors/materials) and obvious
scene-impossible hallucinations. Query objects are prominent real objects -> survive naturally
(never explicitly injected). Output same schema as the taggers: {clip:{video,tags:[...]}}.
"""
import argparse, os, sys, json, re, time
import httpx

SYS=("You clean a noisy list of object tags harvested from a first-person video of ONE physical scene. "
     "Return a canonical vocabulary of the DISTINCT PHYSICAL OBJECTS actually present.\n"
     "Rules:\n"
     "1. MERGE synonyms and over-specific variants into ONE canonical singular name "
     "(e.g. 'bicycle saddle','bicycle seat','saddle','seatpost'->'saddle'; "
     "'front wheel','rear wheel','bicycle tire','wheel rim','rim'->'wheel'; "
     "'yellow-handled screwdriver','precision screwdriver'->'screwdriver').\n"
     "2. DROP non-objects: verbs/actions (adjust, lift, wash), body parts (hand, foot, lap), "
     "people/roles (man, woman, technician, child), bare colors or materials with no object.\n"
     "3. DROP obvious hallucinations that cannot belong to THIS scene given the other tags "
     "(e.g. 'arcade machine','elevator','mall','toilet','drone','telescope' in a bike-repair scene).\n"
     "4. KEEP every genuinely distinct object, fine-grained where it is a real different thing "
     "(e.g. 'screwdriver' and 'wrench' stay separate). Prefer the common generic name for the merged group.\n"
     'Return STRICT JSON: {"canonical":["name",...],"dropped":["tag",...]} singular, lowercase.')

def call(cx,url,key,model,tags):
    body={"model":model,"input":[{"role":"user","content":[
        {"type":"input_text","text":SYS+"\n\nTAGS:\n"+", ".join(tags)}]}]}
    r=cx.post(url,headers={"api-key":key,"Content-Type":"application/json"},json=body,timeout=180)
    txt="".join(c.get("text","") for o in r.json().get("output",[])
                for c in (o.get("content") or []) if c.get("type")=="output_text")
    m=re.search(r"\{.*\}",txt,re.S); j=json.loads(m.group(0))
    can=[str(x).strip().lower() for x in j.get("canonical",[]) if str(x).strip()]
    dr=[str(x).strip().lower() for x in j.get("dropped",[])]
    # de-dup, preserve order
    seen=set(); out=[]
    for t in can:
        if t not in seen: seen.add(t); out.append(t)
    return out,dr

EXTRA=("\n5. ALWAYS merge singular and plural into the SINGULAR ('shoes'->'shoe','cabinets'->'cabinet').\n"
       "6. ALWAYS merge an object with its own parts into the base object "
       "('bread bin lid','bread bin handle','bread bin knob'->'bread bin'; "
       "'cabinet door','cabinet handle'->'cabinet').\n"
       "7. Be aggressive: the output should be roughly HALF the input size or smaller.\n"
       "Output ONLY the JSON object, no markdown fence, no commentary.")

def _server_once(server,tags,max_new_tokens):
    import urllib.request
    body=json.dumps({"prompt":SYS+EXTRA+"\n\nTAGS:\n"+", ".join(tags),
                     "max_new_tokens":max_new_tokens,"no_think":True}).encode()
    rq=urllib.request.Request(server.rstrip("/")+"/generate",data=body,
                              headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(rq,timeout=900) as r:
        txt=json.loads(r.read())["text"]
    txt=re.sub(r"<think>.*?</think>","",txt,flags=re.S)
    m=re.search(r"\{.*\}",txt,re.S)
    if not m: raise ValueError(f"no JSON in {len(txt)} chars (truncated?)")
    j=json.loads(m.group(0))
    return ([str(x).strip().lower() for x in j.get("canonical",[]) if str(x).strip()],
            [str(x).strip().lower() for x in j.get("dropped",[])])

def call_server(server,tags,max_new_tokens=1400,chunk=60):
    """Local persistent Qwen server. CHUNKED: a 9B model truncates on ~150-name JSON outputs, so we
    canonicalize alphabetical blocks (which keeps synonyms/plurals together) and union the results.
    A failed chunk falls back to its raw tags -- we never lose vocabulary."""
    chunks=[tags[i:i+chunk] for i in range(0,len(tags),chunk)] or [[]]
    can_all=[]; dr_all=[]
    for ci,ch in enumerate(chunks):
        if not ch: continue
        try:
            can,dr=_server_once(server,ch,max_new_tokens)
        except Exception as e:
            print(f"    chunk {ci+1}/{len(chunks)} failed ({str(e)[:60]}); keeping raw",flush=True)
            can,dr=ch,[]
        can_all+=can; dr_all+=dr
    seen=set(); out=[]
    for t in can_all:
        if t and t not in seen: seen.add(t); out.append(t)
    return out,dr_all

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="per-clip tag json {clip:{video,tags:[...]}}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap", type=int, default=0, help="if >0, keep at most CAP canonical tags/clip")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--server", default=None,
                    help="use a local qwen_server.py URL instead of the Azure Responses API")
    a=ap.parse_args()
    url=os.getenv("LLM_BASE_URL"); key=os.getenv("LLM_API_KEY"); model=os.getenv("LLM_MODEL","gpt-5.4")
    if not a.server:
        assert url and key, "set LLM_BASE_URL and LLM_API_KEY (or pass --server for the local model)"
    src=json.load(open(a.inp))
    out=json.load(open(a.out)) if (a.resume and os.path.isfile(a.out)) else {}
    cx=httpx.Client()
    raws=[]; cans=[]
    for cu in sorted(src):
        if cu in out: cans.append(len(out[cu]["tags"])); raws.append(len(src[cu]["tags"])); continue
        tags=src[cu].get("tags",[]) or []
        try:
            can,dr=call_server(a.server,tags) if a.server else call(cx,url,key,model,tags)
        except Exception as e:
            print(f"[{cu[:8]}] FAILED {type(e).__name__}: {str(e)[:120]}",flush=True)
            can,dr=tags,[]                       # fall back to raw on error (never lose the clip)
        if a.cap and len(can)>a.cap: can=can[:a.cap]
        out[cu]={"video":src[cu].get("video"),"tags":can,"n_raw":len(tags),"n_dropped":len(dr)}
        json.dump(out,open(a.out,"w"),indent=1)  # incremental
        raws.append(len(tags)); cans.append(len(can))
        print(f"[{cu[:8]}] {len(tags)} -> {len(can)} canonical (dropped {len(dr)})",flush=True)
    import numpy as np
    print(f"DONE {len(out)} clips  raw mean {np.mean(raws):.0f} -> canonical mean {np.mean(cans):.0f}  "
          f"(max {int(np.max(cans))})  -> {a.out}",flush=True)

if __name__=="__main__":
    main()
