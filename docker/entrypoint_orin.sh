#!/usr/bin/env bash
# The client's failure modes are network-shaped, so say what we can see before
# the loop starts blocking on sockets.
set -euo pipefail

echo "[entrypoint] $(uname -m) | python $(python3 -V 2>&1 | cut -d' ' -f2)"

python3 - <<'PY'
import sys
import numpy, msgpack, zmq, cv2, websockets
print("[entrypoint] numpy {} | msgpack {} | pyzmq {} | cv2 {} | websockets {}".format(
    numpy.__version__, msgpack.version, zmq.__version__, cv2.__version__,
    websockets.__version__))
# The sync API only grew keepalive arguments in 14.0; transport.py passes them
# only where they exist. Confirm the pair actually agrees at runtime.
from websockets.sync.client import connect
import inspect
print("[entrypoint] sync keepalive supported: {}".format(
    "ping_interval" in inspect.signature(connect).parameters))
PY

# We BIND :5556. If it is already taken, a previous client is still alive and
# the controller will keep talking to it, not to us.
if command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | grep -q ':5556 '; then
    echo "[entrypoint] FATAL: :5556 is already bound inside this network " \
         "namespace. Another policy client is still running." >&2
    exit 1
fi

echo "[entrypoint] thor=${PEVAL_THOR_HOST:-?}:${PEVAL_THOR_PORT:-?} lane=${PEVAL_LANE:-?}"
exec "$@"
