"""Central path config for the int4 dev/test harnesses (the Python analogue of
../config.sh — one place to point them at your weights).

Defaults are repo-relative (./weights/...). Override any of them via env:
    WEIGHTS_DIR   parent dir that holds the checkpoints (default: <repo>/weights)
    W4AFP8_MODEL  the W4AFP8 int4 checkpoint     (default: $WEIGHTS_DIR/GLM-5.2-W4AFP8)
    FP8_MODEL     the FP8 baseline checkpoint    (default: $WEIGHTS_DIR/GLM-5.2-FP8)
    GPTQ_EXPERTS_DIR  GPTQ-repacked int4 experts (default: $WEIGHTS_DIR/GLM-5.2-W4-GPTQ-experts)
"""
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR", os.path.join(REPO, "weights"))

# W4AFP8_MODEL / MODEL are both honored so these line up with config.sh + the
# launchers, which export MODEL.
W4 = (os.environ.get("W4AFP8_MODEL")
      or os.environ.get("MODEL")
      or os.path.join(WEIGHTS_DIR, "GLM-5.2-W4AFP8"))
FP8 = os.environ.get("FP8_MODEL", os.path.join(WEIGHTS_DIR, "GLM-5.2-FP8"))
GPTQ_EXPERTS_DIR = os.environ.get(
    "GPTQ_EXPERTS_DIR", os.path.join(WEIGHTS_DIR, "GLM-5.2-W4-GPTQ-experts"))
