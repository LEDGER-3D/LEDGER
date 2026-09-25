#!/usr/bin/env bash
# LEDGER memory construction for ONE UCS-Bench video, end to end.
#   bash scripts/build_memory_ucs.sh <video_id> [gpu]
# Needs: the video under $LEDGER_UCS (see README), a running Qwen3.5-9B server (ledger/pipeline/qwen_server.py),
# and the third-party checkouts / weights listed in the README (WildDet3D, FastVGGT, YOLO-World, SAM3 via transformers).
#
# Stages: 1 prep (4 s grid frames) | 2 tag (Qwen3.5-9B) | 3 canonicalize tags | 4 detect (YOLO-World)
#         5 poses + intrinsics (FastVGGT, chunked and Sim(3)-chained; no dataset camera parameters) | 6 3D lift (WildDet3D) | 7 metric scale
#         8 trajectories + SAM3 tracker collapse | 12 object-object relations | 13 places | 9 object descriptions
#         10 check | 14 event layer (one line per frame: what the wearer is doing)
set -uo pipefail
V="${1:?usage: build_memory_ucs.sh <video_id> [gpu]}"
GPU="${2:-0}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
P="$ROOT/ledger/pipeline"                                   # pipeline modules
WORK="${LEDGER_WORK:-$ROOT/work}"; export LEDGER_WORK="$WORK"
# Python interpreters (one environment can serve all of them if it has every dependency; see README)
PY_MAIN="${PY_MAIN:-python}"        # numpy, opencv, httpx, pycocotools            (prep, scale, trajectories, relations, check)
PY_VLM="${PY_VLM:-$PY_MAIN}"         # + transformers>=5, torch                      (tagging, places, descriptions, events)
PY_DET="${PY_DET:-$PY_MAIN}"         # + ultralytics (YOLO-World)                    (detection)
PY_LIFT="${PY_LIFT:-$PY_MAIN}"       # + WildDet3D environment                       (3D lift, FastVGGT poses)
PY_DA3="${PY_DA3:-$PY_LIFT}"         # + depth_anything_3                            (DA3 poses, optional)
YOLO_WEIGHTS="${LEDGER_YOLO_WEIGHTS:-$ROOT/weights/yolov8x-worldv2.pt}"
SERVERS="${QWEN_SERVERS:-http://127.0.0.1:8077}"            # comma-separated Qwen3.5-9B servers
SERVER="${SERVERS%%,*}"
STAGES="${STAGES:-1,2,3,4,5,6,7,8,12,13,9,10,14}"
IMGSZ="${IMGSZ:-1280}"; CONF="${CONF:-0.25}"; TOPK="${TOPK:-40}"; MIN_FREE_MB="${MIN_FREE_MB:-6000}"
POSE_BACKEND="${POSE_BACKEND:-fastvggt}"   # fastvggt | auto (DA3 behind a quality gate, else FastVGGT) | da3
SEG_MODE="${SEG_MODE:-persist}"
want () { case ",$STAGES," in *",$1,"*) return 0;; *) return 1;; esac; }
die () { echo "=== [$V] STAGE $1 FAILED (rc=$2) -- aborting this video ===" >&2; exit "$2"; }

OUT="$WORK/memories"; GRID="$WORK/grid"; FRAMES="$WORK/frames_grid"; LOG="$WORK/logs"; TAGD="$WORK/tags"
POSED="$WORK/poses"; SCENE_TAGS="$WORK/scene_tags"; EVENTS="$WORK/events"
mkdir -p "$OUT" "$GRID" "$FRAMES" "$LOG" "$TAGD" "$POSED" "$SCENE_TAGS" "$EVENTS"
TAGS=$TAGD/tags_$V.json; CANON=$TAGD/canon_$V.json; POSES=$POSED/poses_$V.json

if want 2 || want 3 || want 8 || want 9 || want 13 || want 14; then
  curl -sf "$SERVER/health" >/dev/null || { echo "ERROR: no Qwen server at $SERVER (start ledger/pipeline/qwen_server.py)" >&2; exit 1; }
fi
if want 4 || want 5 || want 6 || want 8; then
  FREE=$(nvidia-smi --id="$GPU" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null || echo 0)
  [ "${FREE:-0}" -lt "$MIN_FREE_MB" ] && { echo "ERROR: GPU $GPU has ${FREE}MB free, need >=${MIN_FREE_MB}MB" >&2; exit 2; }
fi
echo "=== [$V] gpu=$GPU stages=$STAGES ==="
cd "$P"

if want 1; then echo "[1] prep (4 s grid frames)"
  $PY_MAIN -u ucs_adapter.py prep --videos "$V" --grid-root "$GRID" --frames-root "$FRAMES" >>"$LOG/$V.prep.log" 2>&1 || die 1 $?; fi
if [ "$POSE_BACKEND" = "auto" ] && want 5; then
  POSE_BACKEND=$($PY_MAIN ucs_adapter.py posebackend --video "$V" --frames-root "$FRAMES"); echo "    pose backend: $POSE_BACKEND"; fi
if want 2; then echo "[2] tagging (Qwen3.5-9B)"
  $PY_VLM -u run_tagger_hf_vlm.py --model Qwen/Qwen3.5-9B --no-think --dataset ucs --server "$SERVER" --frames "$FRAMES" \
      --out "$TAGS" --per-clip 80 --max-new-tokens 130 --only-clip "$V" >>"$LOG/$V.tag.log" 2>&1 || die 2 $?; fi
if want 3; then echo "[3] canonicalize tags"
  $PY_VLM -u canonicalize_tags.py --in "$TAGS" --out "$CANON" --server "$SERVER" --resume >>"$LOG/$V.canon.log" 2>&1 || die 3 $?; fi
if want 4; then echo "[4] YOLO-World detection (imgsz=$IMGSZ conf=$CONF topk=$TOPK)"
  CUDA_VISIBLE_DEVICES=$GPU $PY_DET -u build_yolo_detect.py --model "$YOLO_WEIGHTS" --mode world --frames-root "$FRAMES" \
      --tags-file "$CANON" --sam3-root "$GRID" --out-root "$OUT" --conf $CONF --topk $TOPK --imgsz $IMGSZ --only-clip "$V" \
      >>"$LOG/$V.yolo.log" 2>&1 || die 4 $?; fi
if want 5; then echo "[5] camera poses + intrinsics ($POSE_BACKEND)"
  PY_POSE=$PY_LIFT; [ "$POSE_BACKEND" = "da3" ] && PY_POSE=$PY_DA3
  CUDA_VISIBLE_DEVICES=$GPU $PY_POSE -u ucs_adapter.py poses --videos "$V" --backend "$POSE_BACKEND" --grid-root "$GRID" \
      --frames-root "$FRAMES" --out "$POSES" >>"$LOG/$V.poses.log" 2>&1 || die 5 $?
  if [ "$POSE_BACKEND" = "da3" ]; then
    GATE=$($PY_MAIN ucs_adapter.py posegate --video "$V" --poses "$POSES"); echo "    DA3 gate: $GATE"
    FB=$($PY_MAIN -c "import ucs_adapter as U; print(U.CONFIG['da3_gate_fallback'])")
    if [[ "$GATE" == FAIL* ]] && [ "$FB" != "da3" ]; then mv "$POSES" "${POSES%.json}.da3_rejected.json"; echo "    DA3 rejected -> $FB"
      CUDA_VISIBLE_DEVICES=$GPU $PY_LIFT -u ucs_adapter.py poses --videos "$V" --backend "$FB" --grid-root "$GRID" \
          --frames-root "$FRAMES" --out "$POSES" >>"$LOG/$V.poses.log" 2>&1 || die 5 $?; fi
  fi; fi
if want 6; then echo "[6] WildDet3D 3D lift"
  rm -f "$OUT/$V/objects_3d.json"
  CUDA_VISIBLE_DEVICES=$GPU $PY_LIFT -u build_mem2_lift.py --dataset ucs --poses "$POSES" --out-root "$OUT" --frames-grid "$FRAMES" \
      --only-clip "$V" >>"$LOG/$V.lift.log" 2>&1 || die 6 $?; fi
if want 7; then echo "[7] metric scale"
  $PY_MAIN -u ucs_adapter.py scale --videos "$V" --mem-root "$OUT" --poses "$POSES" --report "$LOG/scale_$V.json" >>"$LOG/$V.scale.log" 2>&1; fi
if want 8; then echo "[8] trajectories + SAM3 tracker collapse"
  $PY_MAIN -u build_trajectory_memory.py --dataset ucs --clip "$V" --mem-root "$OUT" >>"$LOG/$V.traj.log" 2>&1 || die 8 $?
  CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY_VLM -u build_tracker_collapse.py --clip "$V" \
      --mem-root "$OUT" --tau 0.3 --merge-max-dist 1.0 --server "$SERVER" >>"$LOG/$V.trk.log" 2>&1 || die 8 $?; fi
if want 12; then echo "[12] object-object relations"
  $PY_MAIN -u build_object_relations.py --dataset ucs --clip "$V" --mem-root "$OUT" >>"$LOG/$V.rel.log" 2>&1 || die 12 $?; fi
if want 13; then echo "[13] places"
  $PY_VLM -u build_places.py --dataset ucs --clip "$V" --mem-root "$OUT" --frames-root "$FRAMES" --grid-root "$GRID" \
      --tags-root "$SCENE_TAGS" --server "$SERVERS" >>"$LOG/$V.places.log" 2>&1 || die 13 $?; fi
if want 9; then echo "[9] object descriptions"
  rm -f "$OUT/$V/object_memory.json"; SEG_ARGS="--seg-thresh 0.3"
  if [ "$SEG_MODE" = "persist" ]; then
    read THR K UNITS < <($PY_MAIN ucs_adapter.py segparams --video "$V" --poses "$POSES" --grid-root "$GRID")
    SEG_ARGS="--seg-mode persist --seg-thresh $THR --seg-persist $K"; fi
  $PY_VLM -u build_instance_descriptions.py --clip "$V" --mem-root "$OUT" --dataset ucs --frames-grid "$FRAMES" $SEG_ARGS \
      --server "$SERVERS" >>"$LOG/$V.desc.log" 2>&1 || die 9 $?; fi
if want 10; then echo "[10] check"
  $PY_MAIN -u ucs_adapter.py check --videos "$V" --frames-root "$FRAMES" --mem-root "$OUT" --poses "$POSES" --report "$LOG/check_$V.json" 2>&1 | tail -3; fi
if want 14; then echo "[14] event layer"
  echo "$V" > "$LOG/$V.list"
  $PY_VLM -u build_events.py --videos "$LOG/$V.list" --server "$SERVER" --frames-root "$FRAMES" --out "$EVENTS" >>"$LOG/$V.events.log" 2>&1 || die 14 $?; fi
echo "=== [$V] DONE -> $OUT/$V/object_memory.json (+ relations.json, places.json), poses $POSES, events $EVENTS/$V.json ==="
