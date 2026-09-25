"""Every location the pipeline reads or writes, in one place. Set the environment variables to point at your copies;
the defaults are relative to the repository (./data for datasets, ./work for outputs, ./third_party for external code)."""
import os
CODE = os.path.dirname(os.path.abspath(__file__))                      # this directory (pipeline modules)
ROOT = os.path.abspath(os.path.join(CODE, "..", ".."))                 # repository root
DATA = os.environ.get("LEDGER_DATA", os.path.join(ROOT, "data"))
WORK = os.environ.get("LEDGER_WORK", os.path.join(ROOT, "work"))
THIRD = os.environ.get("LEDGER_THIRD_PARTY", os.path.join(ROOT, "third_party"))
UCS = os.environ.get("LEDGER_UCS", os.path.join(DATA, "UCS-Bench"))                           # UCS-Bench videos + QA files
EGO4D = os.environ.get("LEDGER_EGO4D", os.path.join(DATA, "ego4d", "v2"))                      # Ego4D v2 (VQ3D clips)
VQ3D = os.environ.get("LEDGER_VQ3D", os.path.join(DATA, "ego4d", "episodic-memory", "VQ3D"))  # Ego4D episodic-memory VQ3D
EGOLOC_POSES = os.environ.get("LEDGER_EGOLOC_POSES", os.path.join(DATA, "ego4d", "egoloc_poses", "all_val_test_pose.json"))
HDEPIC = os.environ.get("LEDGER_HDEPIC", os.path.join(DATA, "hd-epic"))                        # HD-EPIC videos + annotations
FASTVGGT = os.environ.get("LEDGER_FASTVGGT", os.path.join(THIRD, "FastVGGT"))
WILDDET3D = os.environ.get("LEDGER_WILDDET3D", os.path.join(THIRD, "WildDet3D"))
YOLO_WEIGHTS = os.environ.get("LEDGER_YOLO_WEIGHTS", os.path.join(ROOT, "weights", "yolov8x-worldv2.pt"))
