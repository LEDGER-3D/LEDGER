"""EVENT LAYER of the memory (UCS-Bench): for every 4 s grid frame of a recording -- the frames the object memory is built
from -- one short line saying what the camera wearer is doing (action + the object or person involved) and where, from the
same VLM as the description stage (Qwen3.5-9B). Stored per recording as <out>/<video>.json = [{t, text}].
  usage: build_events.py --videos <list file> --server http://127.0.0.1:8077 --frames-root <grid frames> --out <dir>
"""
import argparse, glob, json, os, time, urllib.request
FPS = 10.0            # grid frames are named by their 10 fps frame index: 000040.jpg = 4.0 s
PROMPT = ("This is one frame from a head-mounted camera. In ONE short sentence (at most 25 words), say what the camera wearer "
          "is doing right now -- the action and the object or person involved -- and where they are. If they are just looking "
          "or walking, say so and name what is in front of them. Plain words, no speculation.")
MAX_NEW_TOKENS = 60
def ask(server, img):
    body = json.dumps({"prompt": PROMPT, "images": [img], "max_new_tokens": MAX_NEW_TOKENS, "no_think": True}).encode()
    for k in range(4):
        try:
            r = urllib.request.urlopen(urllib.request.Request(server + "/generate", body, {"Content-Type": "application/json"}), timeout=180)
            return " ".join(json.loads(r.read())["text"].strip().split())
        except Exception as ex:
            err = ex; time.sleep(5 * (k + 1))
    raise err
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--videos", required=True); ap.add_argument("--server", required=True)
    ap.add_argument("--frames-root", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    for v in [l.strip() for l in open(a.videos) if l.strip()]:
        files = sorted(glob.glob(os.path.join(a.frames_root, v, "*.jpg")))
        if os.path.isfile(f"{a.out}/{v}.json") or not files: continue
        t0 = time.time(); ev = []
        for f in files:
            ev.append({"t": round(int(os.path.basename(f).split(".")[0]) / FPS, 1), "text": ask(a.server, f)})
        json.dump(ev, open(f"{a.out}/{v}.json.tmp", "w")); os.replace(f"{a.out}/{v}.json.tmp", f"{a.out}/{v}.json")
        print(f"[events] {v}: {len(ev)} frames in {time.time()-t0:.0f}s ({(time.time()-t0)/max(1,len(ev)):.1f} s/frame)", flush=True)
if __name__ == "__main__": main()
