#!/usr/bin/env bash
# Fail loudly and early, on the two things that are silent otherwise: a torch
# without sm_110 kernels, and missing weights.
set -euo pipefail

echo "[entrypoint] $(uname -m) | python $(python -V 2>&1 | cut -d' ' -f2)"

python - <<'PY'
import sys
try:
    import torch
except Exception as exc:
    print("[entrypoint] torch not importable: {}".format(exc), file=sys.stderr)
    raise SystemExit(1)

print("[entrypoint] torch {} cuda={}".format(torch.__version__, torch.version.cuda))
if not torch.cuda.is_available():
    print("[entrypoint] WARNING: no CUDA device. Did you pass --runtime nvidia "
          "(or --gpus all)?", file=sys.stderr)
    raise SystemExit(0)

name = torch.cuda.get_device_name(0)
major, minor = torch.cuda.get_device_capability(0)
arches = torch.cuda.get_arch_list()
print("[entrypoint] GPU: {} (sm_{}{}) | built for {}".format(name, major, minor, arches))
if "sm_{}{}".format(major, minor) not in arches:
    # This is the failure the Thor install path exists to prevent. It does not
    # surface at import -- only at the first kernel launch, as "no kernel image
    # is available for execution on the device".
    print("[entrypoint] FATAL: this torch has no sm_{}{} kernels. The image was "
          "built with the dGPU wheels instead of the Thor ones."
          .format(major, minor), file=sys.stderr)
    raise SystemExit(1)
PY

# Every GR00T checkpoint loads the VLM backbone nvidia/Cosmos-Reason2-2B, and
# that repo is GATED on Hugging Face. It is not in our checkpoint and not in
# this image, so it has to come from somewhere: a pre-staged HF cache (no
# network needed at run time) or an HF_TOKEN for an account that has accepted
# the terms. Without either, the failure lands 30 seconds into the model load
# as a 401 that names a repo nobody asked for.
BACKBONE="nvidia/Cosmos-Reason2-2B"
CACHE_ROOT="${HF_HOME:-$HOME/.cache/huggingface}"
if [ -n "${PEVAL_CHECKPOINT:-}" ] \
   && [ ! -d "$CACHE_ROOT/hub/models--nvidia--Cosmos-Reason2-2B" ] \
   && [ -z "${HF_TOKEN:-}" ]; then
    echo "[entrypoint] FATAL: the VLM backbone $BACKBONE is neither cached nor" >&2
    echo "[entrypoint] fetchable. It is a gated repo that every GR00T checkpoint" >&2
    echo "[entrypoint] loads. Either:" >&2
    echo "[entrypoint]   * pre-stage it and point HF_HOME at the cache:" >&2
    echo "[entrypoint]       HF_HOME=/opt/weights/hf-cache hf download $BACKBONE" >&2
    echo "[entrypoint]       docker run ... -e HF_HOME=/weights/hf-cache" >&2
    echo "[entrypoint]   * or pass a token for an account that has accepted the" >&2
    echo "[entrypoint]     terms at https://huggingface.co/$BACKBONE :" >&2
    echo "[entrypoint]       docker run ... -e HF_TOKEN=..." >&2
    echo "[entrypoint] Looked in: $CACHE_ROOT/hub" >&2
    exit 1
fi

CHECKPOINT="${PEVAL_CHECKPOINT:-}"
if [ -n "$CHECKPOINT" ] && [ ! -f "$CHECKPOINT/config.json" ]; then
    echo "[entrypoint] no checkpoint at $CHECKPOINT (config.json missing)." >&2
    echo "[entrypoint] Mount the weights read-only at /weights, or unset" >&2
    echo "[entrypoint] PEVAL_CHECKPOINT to run the hold-still policy." >&2
    exit 1
fi

exec "$@"
