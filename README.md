# Never Look Back: Understanding Persistence in 3D Object Memory from Egocentric Videos

**[Project page](https://ledger-3d.github.io/)** · **[Paper (preview PDF)](https://ledger-3d.github.io/paper.pdf)** · arXiv: coming soon

Shravan S Chaudhari<sup>1</sup>, William Paul<sup>2</sup>, Suchi Saria<sup>1</sup>, Rama Chellappa<sup>1\*</sup>, Homanga Bharadhwaj<sup>1\*</sup>
<br><sup>1</sup>Johns Hopkins University · <sup>2</sup>Johns Hopkins University Applied Physics Laboratory · <sup>\*</sup>equal advising

The project page shows the memory being built and queried in 3D, for a kitchen (HD-EPIC), a mall
(UCS-Bench), a workshop (Ego4D VQ3D) and a stream that joins three different kitchens.

## LEDGER

LEDGER (Long-horizon Egocentric Descriptions, Geometry, and Event Records) is a persistent 3D object memory.
It turns an egocentric video into a **text memory**: the objects that were seen, where each
one was in 3D over time, what it looked like at each place, how objects related to each other,
which place the wearer was in, and one line per sampled moment of what the wearer was doing. A
language model then answers questions about the video **from that memory alone**, without
looking at the video.

This repository is a minimal working example with three parts:

* `ledger/pipeline/` and `scripts/build_memory_ucs.sh` build the memory from a video. It is
  pose-free: camera poses and intrinsics come from FastVGGT, and metric scale comes from the 3D lift.
* `ledger/answer/` is the answering side, with Qwen3.5-9B reading the memory. The reader
  reproduces the evaluated prompts byte for byte. This was checked on all 1,626 UCS-Bench prompts
  and all 1,900 HD-EPIC prompts of the evaluation runs.
* `examples/` holds finished memories for a few videos from **UCS-Bench**, **HD-EPIC** and
  **Ego4D VQ3D**. Each comes with its questions, the expected answers and Qwen3.5-9B's answers
  from the memory. No videos are included (see [Getting the videos](#getting-the-videos)).

```
ledger/
  answer/      qwen.py        Qwen3.5-9B, greedy, thinking off (in-process, or via a qwen_server)
               reader.py      retrieval + memory rendering + answer prompt (UCS-Bench, HD-EPIC)
               vq3d.py        "where is the X?" -> 3D position, and the VQ3D metrics
               run_example.py answer an example directory and compare with the expected answers
  pipeline/    the memory-construction stages (one module per stage; see below)
scripts/
  build_memory_ucs.sh         all stages, end to end, for one UCS-Bench video
examples/
  ucs/<video>/  hdepic/<video>/  vq3d/<clip>/     example.json + memory/
```

## 1. Answer the shipped examples

Requirements: Python 3.10+ and one GPU with 24 GB. The first run downloads the weights, about 19 GB
(`Qwen/Qwen3.5-9B`).

```bash
pip install -r requirements.txt          # the first block is enough for answering
python ledger/answer/run_example.py examples/ucs/* examples/hdepic/* examples/vq3d/*
python ledger/answer/run_example.py examples/ucs/<video> --show-prompt      # print what the model reads
```

For each question the script prints the retrieved memory objects, the model's IDENTIFICATION and
REASONING, its answer and the expected answer. For VQ3D it prints the chosen memory object, its 3D
position, the L2 error in metres and whether the query counts as a success.

To reuse one loaded model across runs, start a server and pass `--server`:

```bash
CUDA_VISIBLE_DEVICES=0 python ledger/pipeline/qwen_server.py --port 8077 &
python ledger/answer/run_example.py examples/ucs/* --server http://127.0.0.1:8077
```

## The examples

Each question below was answered correctly by Qwen3.5-9B from the memory, with the same answer in three
independent runs (the last with the model loaded in-process by `run_example.py`). For UCS-Bench and HD-EPIC, Qwen answering with no memory got these questions wrong. The
model's full reply is stored with each question (`qwen_reference`).

| dataset | video | length | question | expected | Qwen3.5-9B from memory |
|---|---|---:|---|---|---|
| UCS-Bench | `egolife_A1_D3_001` | 11 min 32 s | Right now, can I touch the cup? | At this moment, you can reach the cup; the cup is in front of you. | **D** ✓ |
| UCS-Bench | `egolife_A1_D3_001` | 11 min 32 s | What is the movement path of the pen near the whiteboard on the first floor that I see? | You picked up a pen from next to the whiteboard on the first floor, climbed two flights of stairs, and arrived on the second floor. | **C** ✓ |
| UCS-Bench | `egolife_A1_D6_045` | 5 min 56 s | Which direction is the guitar relative to me? | The guitar should be above you to your right. | **D** ✓ |
| HD-EPIC | `P03-20240216-223126` | 2 min 32 s | Where did I take the object identified by ⟨box⟩ at ⟨00:02:31.667⟩ from before putting it at ⟨00:02:31.667⟩? | table left from radiator | **A** ✓ |
| HD-EPIC | `P03-20240216-223126` | 2 min 32 s | Which of these objects did the person take from the item indicated by bounding box ⟨box⟩ in ⟨00:02:14.967⟩? | Nothing | **B** ✓ |
| HD-EPIC | `P03-20240217-210126` | 3 min 04 s | Where did I put the object identified by ⟨box⟩ at ⟨00:01:36.633⟩ after taking it at ⟨00:01:36.633⟩? | dishwasher | **C** ✓ |
| HD-EPIC | `P03-20240217-210126` | 3 min 04 s | Which of these objects did the person put in/on the item indicated by bounding box ⟨box⟩ in ⟨00:02:11.967⟩? | Nothing | **E** ✓ |
| Ego4D VQ3D | `150bc185…` | 5 min 00 s | Where is the puller slide hammer? | 3D position (scan frame) | picks *hammer*: L2 0.76 m |
| Ego4D VQ3D | `150bc185…` | 5 min 00 s | Where is the fire extinguisher? | 3D position (scan frame) | picks *fire extinguisher*: L2 0.22 m |
| Ego4D VQ3D | `150bc185…` | 5 min 00 s | Where is the wheel rim? | 3D position (scan frame) | picks *hubcap*: L2 0.40 m |
| Ego4D VQ3D | `150bc185…` | 5 min 00 s | Where is the plastic bag? | 3D position (scan frame) | picks *bag*: L2 0.19 m |
| Ego4D VQ3D | `a044eb14…` | 5 min 00 s | Where is the screw driver? | 3D position (scan frame) | picks *screwdriver*: L2 0.89 m |
| Ego4D VQ3D | `a044eb14…` | 5 min 00 s | Where is the pliers? | 3D position (scan frame) | picks *pliers*: L2 0.28 m |
| Ego4D VQ3D | `a044eb14…` | 5 min 00 s | Where is the chair? | 3D position (scan frame) | picks *stool*: L2 0.43 m |

On VQ3D the query names only the object. Qwen matches it to a memory object (for example "wheel rim"
to *hubcap*, or "chair" to *stool*), and the answer is that object's 3D position. All seven
queries meet the benchmark's success criterion.


## 2. Build a memory for a UCS-Bench video

**Third-party code and weights**
- clone into `third_party/`:
  - [FastVGGT](https://github.com/mystorm16/FastVGGT), for camera poses and intrinsics. Its weights
    are `facebook/VGGT-1B` on Hugging Face.
  - [WildDet3D](https://github.com/allenai/WildDet3D), for the metric 3D lift of each 2D detection.
    Follow its README for the checkpoint.
- `weights/yolov8x-worldv2.pt`: the YOLO-World detector, from Ultralytics.
- SAM3 video tracker: `facebook/sam3` on Hugging Face (gated; request access), loaded through
  transformers.
- Qwen3.5-9B serves the tagging, name canonicalisation, descriptions, places and events stages
  through `ledger/pipeline/qwen_server.py`.

Every path is set by an environment variable in `ledger/pipeline/ledger_paths.py`, and each has a
default relative to the repository. `LEDGER_UCS` points at the UCS-Bench folder (`<video>.mp4` and
`QAs/<video>-mcq.json`), and `LEDGER_WORK` is where outputs go (`./work` by default).

```bash
CUDA_VISIBLE_DEVICES=1 python ledger/pipeline/qwen_server.py --port 8077 &      # one GPU for Qwen
QWEN_SERVERS=http://127.0.0.1:8077 bash scripts/build_memory_ucs.sh <video_id> 0   # GPU 0 for the vision models
```

The stages, each one a module in `ledger/pipeline/`:

| # | stage | module |
|---|---|---|
| 1 | frames on a 4 s grid | `ucs_adapter.py prep` |
| 2 | open-vocabulary tags per frame (Qwen3.5-9B) | `run_tagger_hf_vlm.py` |
| 3 | canonicalise tag names | `canonicalize_tags.py` |
| 4 | 2D detection of the tags (YOLO-World) | `build_yolo_detect.py` |
| 5 | camera poses + intrinsics (FastVGGT, chunked and Sim(3)-chained) | `ucs_adapter.py poses`, `vggt_poses_egoloc.py` |
| 6 | 3D lift of every detection (WildDet3D) | `build_mem2_lift.py` |
| 7 | one metric scale per video, from the lift's own camera-frame depths | `ucs_adapter.py scale` |
| 8 | 3D trajectories, then SAM3 tracker collapse of duplicate tracks | `build_trajectory_memory.py`, `build_tracker_collapse.py` |
| 12 | object-object relations (same-frame geometry) | `build_object_relations.py` |
| 13 | places (per-frame scene label) | `build_places.py` |
| 9 | per-position object descriptions (Qwen3.5-9B) | `build_instance_descriptions.py` |
| 10 | self-consistency check of poses and lift | `ucs_adapter.py check` |
| 14 | event layer: one line per frame of what the wearer is doing | `build_events.py` |

Outputs go to `work/memories/<video>/`:
- `object_memory.json`, the answering memory
- `relations.json` and `places.json`
- `work/poses/poses_<video>.json` and `work/events/<video>.json`

These are exactly the files an example's `memory/` directory holds.

## Getting the videos

No video is redistributed here. Each example's `example.json` gives the dataset's own ID.

- **UCS-Bench.** Videos and question files are on Hugging Face
  ([`cocowy1/UCS-Bench`](https://huggingface.co/datasets/cocowy1/UCS-Bench)):
  ```bash
  huggingface-cli download cocowy1/UCS-Bench --repo-type dataset --local-dir data/UCS-Bench \
      --include "<video>.mp4" "QAs/<video>-mcq.json"
  ```
- **HD-EPIC.** Register at [hd-epic.github.io](https://hd-epic.github.io/), then use the official
  downloader. Videos are stored as `HD-EPIC/Videos/<participant>/<video>.mp4`. The VQA questions
  are in [hd-epic-annotations](https://github.com/hd-epic/hd-epic-annotations) under
  `vqa-benchmark/`.
  ```bash
  git clone https://github.com/hd-epic/hd-epic-downloader && cd hd-epic-downloader
  python hd-epic-downloader.py <out> --videos --participants <n>      # e.g. 5 for P05-...
  ```
- **Ego4D VQ3D.** Accept the Ego4D licence at [ego4d-data.org](https://ego4d-data.org/), then use
  the Ego4D CLI. The VQ3D annotations and the evaluation code are in
  [EGO4D/episodic-memory](https://github.com/EGO4D/episodic-memory) under `VQ3D/`.
  ```bash
  ego4d --output_directory data/ego4d --datasets full_scale annotations --video_uids <video_uid>
  ```

## Notes

- The **HD-EPIC** and **VQ3D** example memories were built with those datasets' own camera poses:
  Aria MPS for HD-EPIC and the Ego4D VQ3D camera poses for VQ3D. The detections, 3D lift,
  trajectories, SAM3 tracker collapse and descriptions are the same stages as above. For the
  VQ3D examples the first stage was SAM3 detection. Only the pose-free (FastVGGT) path is
  released as code.
- HD-EPIC questions point at an image box. Its lifted 3D centre is shipped with each question
  (`question_point_3d`), and the candidates are ranked by 3D distance to it.
- The VQ3D examples use the tracker-collapsed memory (`objects_3d_trajectories_tracker.json`).
- Questions and expected answers come from the respective benchmarks and remain under their
  licences.

## License

The code is released under the [MIT License](LICENSE). The example memories are derived from
UCS-Bench, HD-EPIC and Ego4D and remain subject to those datasets' terms.

## Citation

```bibtex
@article{chaudhari2026neverlookback,
  title   = {Never Look Back: Understanding Persistence in 3D Object Memory from Egocentric Videos},
  author  = {Chaudhari, Shravan S and Paul, William and Saria, Suchi and Chellappa, Rama and Bharadhwaj, Homanga},
  year    = {2026}
}
```
