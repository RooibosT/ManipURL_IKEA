#!/usr/bin/env python3
"""Policy client — runs on the Jetson Orin NX onboard the G1. Team-owned.

    boundary.CameraStream  :5555 ─┐
                                  ├─> observation ─> Thor ─> (T,25) chunk ─┐
    boundary.StateStream   :5557 ─┘                                        │
                                                boundary.ActionSink :5556 <┘

    python components/client.py --lane decoupled --thor 192.168.100.1

Adapted from the template's reference loop. Three changes, all because our
server's contract is not the template's default:

  1. ROW RATE COMES FROM THE SERVER. The template computes staleness at a fixed
     20 Hz. Our chunks are published at whatever `--row-hz` the server declares
     (50 Hz by default, resampled from the checkpoint's 30 Hz), so the client
     reads `action_row_hz` out of the metadata frame instead of assuming. Get
     this wrong and latency compensation silently skips the wrong number of
     rows.

  2. RE-QUERY WHEN THE CHUNK RUNS OUT, not on a fixed timer. The server hands
     back exactly the rows it wants executed before replanning (8 model rows
     ≈ 0.27 s), so the natural period is the duration of what we just
     published. A fixed 20 Hz re-query would ask for a new chunk five times
     per chunk and throw four of them away.

  3. DECOUPLED ONLY. Our checkpoint emits joint targets, not a SONIC latent.
     The sonic path is refused rather than left in as untested code.

SAFETY: killing or pausing this client does NOT stop the robot. The whole-body
controller keeps replaying the last command it got. Only the organizer's
independent e-stop brings the robot to a safe state, and it does not go through
our code.

NETWORK: the Thor<->Orin link is a direct ethernet cable with static IPs. If
the client cannot reach the Thor, check `ip link` for a down interface before
looking at code.

Observations carry three 480x640x3 images, 2.76 MB raw. On our own Thor<->Orin
link that measured about 330 ms of round trip on top of the policy's own
185 ms -- enough that the client discarded 25 of every 26 published rows as
stale. So the images go over the wire as JPEG (--jpeg-quality, default 85),
which takes the observation to roughly a twentieth of that. The frames were
JPEG from the organizer's camera server to begin with and `boundary` decoded
them for us, so this is a second generation of the same artefacts.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from boundary import ActionSink, CameraStream, StateStream  # noqa: E402
from boundary.actions import ActionError  # noqa: E402
from components.imagecodec import DEFAULT_QUALITY, encode_images  # noqa: E402
from components.transport import PolicyLink  # noqa: E402

LANE = "decoupled"
DEFAULT_ROW_HZ = 50.0       # fallback if the server declares nothing
# One of the five subtask strings the checkpoint was trained on. The policy
# is language-conditioned, so this is a real input, not a label -- change it
# per attempt with --prompt. components/policy/bct.py lists all five.
DEFAULT_PROMPT = "pick table leg"


class Inference:
    """Runs one inference at a time on a worker thread, so the control loop
    never blocks on the network.

    ``submit`` snapshots the observation on the calling thread — sampling it
    inside the worker would time-stamp the observation at whenever the thread
    happened to start, which is exactly the error latency compensation is
    trying to correct.
    """

    def __init__(self, link, cameras, states, prompt, camera_keys,
                 jpeg_quality=DEFAULT_QUALITY):
        self._link = link
        self._cameras = cameras
        self._states = states
        self._prompt = prompt
        self._camera_keys = camera_keys
        self._jpeg_quality = int(jpeg_quality)
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="infer")
        self._pending = None            # the in-flight Future, or None
        self._missing_warned = set()

    @property
    def busy(self) -> bool:
        return self._pending is not None

    def observe(self):
        """Sample both input endpoints. None until each has produced once."""
        frame = self._cameras.read(timeout_ms=0) or self._cameras.latest()
        state = self._states.read(timeout_ms=0) or self._states.latest()
        if frame is None or state is None:
            return None
        # Only the cameras the server declared: three 480x640x3 images is
        # ~2.7 MB per step raw, so shipping ones the model ignores is pure
        # latency. A key the organizer has dropped is simply absent here; the
        # server reuses that camera's last good frame and says so.
        images = {}
        for key in self._camera_keys:
            if key in frame.images:
                images[key] = frame.images[key]
            elif key not in self._missing_warned:
                self._missing_warned.add(key)
                print(
                    "[client] WARNING: camera {!r} is not being published. The "
                    "server will reuse its last good frame if it ever had one.".format(key),
                    file=sys.stderr,
                )
        obs = {
            "body_q": state.body_q,
            "base_quat": state.base_quat,
            "prompt": self._prompt,
            "t": frame.received_at,
        }
        if self._jpeg_quality > 0:
            obs["images_jpeg"] = encode_images(images, self._jpeg_quality)
        else:
            obs["images"] = images
        return obs

    def submit(self) -> bool:
        """Kick off the next inference. False if one is already in flight or
        the endpoints have not warmed up yet."""
        if self._pending is not None:
            return False
        obs = self.observe()
        if obs is None:
            return False
        issued = time.monotonic()
        self._pending = self._pool.submit(
            lambda: (self._link.act(obs), time.monotonic() - issued)
        )
        return True

    def collect(self, block: bool = True):
        """Take the finished chunk and its measured latency in seconds."""
        if self._pending is None:
            return None
        if not block and not self._pending.done():
            return None
        try:
            return self._pending.result()
        finally:
            self._pending = None

    def close(self):
        self._pool.shutdown(wait=False, cancel_futures=True)


_warned_short_chunk = False


def _skip_rows(latency_s: float, rate_hz: float, chunk_length: int) -> int:
    """How many leading rows of a fresh chunk are already stale.

    A chunk that took L seconds to compute describes the world as it was L
    seconds ago. Always leaves at least one row: a chunk that took longer to
    compute than it lasts means the model is too slow for the cadence, not that
    we should send nothing.
    """
    global _warned_short_chunk
    skip = min(int(round(latency_s * rate_hz)), max(chunk_length - 1, 0))
    if not _warned_short_chunk and skip * 2 > chunk_length:
        _warned_short_chunk = True
        print(
            "[client] WARNING: {} of {} rows are stale on arrival ({:.0f}ms at "
            "{:.0f}Hz). Most of every chunk is being discarded. Raise the server's "
            "--execute-rows (to at least {}) or make inference faster, or the robot "
            "spends most of its time replaying the tail of an old chunk.".format(
                skip, chunk_length, latency_s * 1000, rate_hz,
                int(latency_s * rate_hz * 3),
            ),
            file=sys.stderr,
        )
    return skip


def run_decoupled(inference: Inference, sink, row_hz: float, min_period_s: float):
    """Publish a task-space chunk, then re-query while it plays out."""
    chunks = 0

    while not inference.submit():
        print("[client] waiting for camera/state...")
        time.sleep(0.1)

    while True:
        tick = time.monotonic()
        action, latency = inference.collect(block=True)
        chunk = sink.validate_chunk(action["actions"])

        # The adapter owns interpolation, so hand it the trajectory minus the
        # rows that inference latency already consumed.
        skip = _skip_rows(latency, row_hz, len(chunk))
        live = chunk[skip:]
        sink.send_chunk(live, issued_at=time.time() - latency)

        # Submit the next one before sleeping, not after.
        inference.submit()

        chunks += 1
        if chunks % 20 == 1:
            print(
                "[client] chunk T={} latency={:.0f}ms skip={} publish={} rows "
                "({:.0f}ms of motion)".format(
                    len(chunk), latency * 1000, skip, len(live),
                    len(live) / row_hz * 1000,
                )
            )

        # Ask again as the chunk we just published runs out, not on a timer
        # unrelated to how much motion it contained.
        period = max(len(live) / row_hz, min_period_s)
        remaining = period - (time.monotonic() - tick)
        if remaining > 0:
            time.sleep(remaining)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lane", default=os.environ.get("PEVAL_LANE", LANE),
                        help="Must match the manifest and the server.")
    parser.add_argument("--thor", default=os.environ.get("PEVAL_THOR_HOST", "192.168.100.1"),
                        help="Policy server host.")
    parser.add_argument("--thor-port", type=int,
                        default=int(os.environ.get("PEVAL_THOR_PORT", "8765")))
    parser.add_argument("--orin", default="127.0.0.1",
                        help="Host of the organizer's camera/state endpoints.")
    parser.add_argument("--prompt", default=os.environ.get("PEVAL_PROMPT", DEFAULT_PROMPT),
                        help="Task instruction handed to the policy.")
    parser.add_argument("--jpeg-quality", type=int, default=DEFAULT_QUALITY,
                        help="JPEG quality for the observation images, 1-100. "
                             "0 sends them raw, which costs about 330ms of round "
                             "trip on our link. The server accepts either.")
    parser.add_argument("--min-period-s", type=float, default=0.02,
                        help="Floor on the re-query period, so a degenerate "
                             "one-row chunk cannot spin the loop.")
    args = parser.parse_args()

    if args.lane != LANE:
        raise SystemExit(
            "[client] this submission is lane {!r}: the checkpoint emits joint "
            "targets, not a SONIC latent. Refusing to start on lane {!r}.".format(
                LANE, args.lane
            )
        )

    cameras = CameraStream(host=args.orin)
    states = StateStream(host=args.orin)
    sink = ActionSink.for_lane(args.lane)   # we BIND :5556; the controller dials in

    print("[client] lane={} thor={}:{} orin={}".format(
        args.lane, args.thor, args.thor_port, args.orin))
    print("[client] waiting for the organizer's endpoints...")
    cameras.wait_until_live()
    state = states.wait_until_live()
    print("[client] endpoints live (hand state {})".format(
        "present" if state.hands_present else "absent — Dex1-1 rig, as expected"))

    link = PolicyLink("ws://{}:{}".format(args.thor, args.thor_port))
    declared = link.metadata.get("lane")
    if declared != args.lane:
        raise SystemExit(
            "[client] lane mismatch: server declared {!r}, client is {!r}. "
            "Start both on the same lane.".format(declared, args.lane)
        )

    row_hz = float(link.metadata.get("action_row_hz", DEFAULT_ROW_HZ))
    camera_keys = link.metadata.get("camera_keys", ["ego_view"])
    print("[client] server: {} rows @ {:g} Hz, cameras {}, images {}".format(
        link.metadata.get("action_chunk_size"), row_hz, camera_keys,
        "jpeg q{}".format(args.jpeg_quality) if args.jpeg_quality > 0 else "raw"))

    link.reset()
    inference = Inference(link, cameras, states, args.prompt, camera_keys,
                          jpeg_quality=args.jpeg_quality)
    try:
        run_decoupled(inference, sink, row_hz, args.min_period_s)
    except KeyboardInterrupt:
        print("\n[client] interrupted — THE ROBOT IS STILL HOLDING ITS LAST COMMAND. "
              "Use the e-stop to bring it to a safe state.")
    except ActionError as exc:
        print("\n[client] ACTION REJECTED: {}".format(exc), file=sys.stderr)
        raise SystemExit(1)
    finally:
        inference.close()
        sink.close()
        cameras.close()
        states.close()


if __name__ == "__main__":
    main()
