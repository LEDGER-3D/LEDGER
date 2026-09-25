"""Answer an example's questions from its shipped memory with Qwen3.5-9B, and compare with the expected answers.

  python ledger/answer/run_example.py examples/ucs/egolife_A1_D6_045      # loads the model in-process
  python ledger/answer/run_example.py examples/hdepic/<video> --server http://127.0.0.1:8077
  python ledger/answer/run_example.py examples/vq3d/<clip> --show-prompt

Each example directory holds example.json (dataset, video id, questions, expected answers) and memory/
(the memory files the pipeline wrote). The model never sees the video.
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reader as R   # noqa: E402
import vq3d as V     # noqa: E402
from qwen import Qwen  # noqa: E402


def _load(path):
    return json.load(open(path)) if os.path.isfile(path) else None


def run(ex_dir, llm, show_prompt=False):
    ex = json.load(open(os.path.join(ex_dir, "example.json")))
    md = os.path.join(ex_dir, "memory")
    ds = ex["dataset"]
    out = []
    if ds == "ucs":
        mem = _load(f"{md}/object_memory.json")
        poses, rel, pl = _load(f"{md}/poses.json"), _load(f"{md}/relations.json"), _load(f"{md}/places.json")
        ev = _load(f"{md}/events.json"); ev = sorted(ev, key=lambda x: x["t"]) if ev else None
    elif ds == "hdepic":
        mem = _load(f"{md}/object_memory.json")
    elif ds == "vq3d":
        mem = _load(f"{md}/objects_3d_trajectories_tracker.json")
    else:
        raise SystemExit(f"unknown dataset {ds}")
    for q in ex["questions"]:
        if ds == "vq3d":
            xyz, rec = V.answer(llm, q["object_title"], mem)
            sc = V.score(xyz, q["gt_3d_scan_coords"]) if xyz is not None else {"L2_m": None, "success": False}
            rec.update(sc, question=f"Where is the {q['object_title']}?", predicted_xyz=xyz)
            print(f"\nQ: Where is the {q['object_title']}?   picked '{rec['picked']}' -> {xyz and [round(v, 2) for v in xyz]}"
                  f"   L2 = {sc['L2_m']} m   success = {sc['success']}")
            print(f"   {rec['reply'].strip()}")
            out.append(rec); continue
        if ds == "ucs":
            prompt, trace = R.ucs_prompt(q, mem, poses, rel, pl, ev)
        else:
            prompt, trace = R.hdepic_prompt(q, mem)
        if show_prompt: print("\n" + "=" * 100 + "\n" + prompt + "\n" + "=" * 100)
        free = llm.ask(prompt, 820)
        p = R.parse_reply(free, len(q["choices"]))
        ok = p["chosen"] == q["correct_idx"]
        print(f"\nQ: {q['question']}")
        for i, c in enumerate(q["choices"]):
            print(f"   {R.LET[i]}. {c}" + ("   <- expected" if i == q["correct_idx"] else ""))
        print(f"   retrieved: {trace['retrieved']}")
        print(f"   IDENTIFICATION: {p['identification']}\n   REASONING: {p['reasoning']}")
        print(f"   ANSWER: {R.LET[p['chosen']] if p['chosen'] is not None else None}   ->  {'CORRECT' if ok else 'WRONG'}")
        out.append({"question": q["question"], "chosen": p["chosen"], "correct_idx": q["correct_idx"],
                    "correct": ok, "reply": free, **trace})
    return ex, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("examples", nargs="+", help="example directories (examples/<dataset>/<id>)")
    ap.add_argument("--server", default=None, help="URL of a running ledger/pipeline/qwen_server.py; default: load in-process")
    ap.add_argument("--show-prompt", action="store_true", help="print the full memory prompt of each question")
    ap.add_argument("--out", default=None, help="write all answers to this JSON file")
    a = ap.parse_args()
    llm = Qwen(a.server)
    allrec = {}
    for d in a.examples:
        ex, recs = run(d, llm, a.show_prompt)
        allrec[d] = recs
        if ex["dataset"] == "vq3d":
            print(f"\n[{d}] success {sum(r['success'] for r in recs)}/{len(recs)}")
        else:
            print(f"\n[{d}] correct {sum(r['correct'] for r in recs)}/{len(recs)}")
    if a.out: json.dump(allrec, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
