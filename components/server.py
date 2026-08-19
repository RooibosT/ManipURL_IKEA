#!/usr/bin/env python3
"""Policy server — runs on the Jetson AGX Thor. Team-owned.

    Thor  192.168.100.1   this file, the checkpoint, the GPU
    Orin  192.168.100.2   components/client.py and the organizer's endpoints

    python components/server.py --lane decoupled --port 8765 \
        --checkpoint /weights/gr00t-n1.7-g1-dex1-bct-relarm-aug-30hz-h40

What happens per call to :meth:`Policy.act`:

    boundary obs (body_q 29, base_quat 4, 3 images)
        -> 46-dim BCT state  (policy/bct.py)
        -> GR00T N1.7 inference, horizon 40 @ 30 Hz, arms restored to absolute
        -> take the first --execute-rows rows, drop the waist group
        -> joint-domain gates                     (policy/taskspace.py)
        -> FK -> end-effector poses, resampled to --row-hz
        -> (T, 25) task-space chunk for the decoupled lane

The lane is not a preference: this checkpoint is GR00T N1.7 under the
``new_embodiment`` tag and emits joint targets, not a 64-dim SONIC latent, so
it is ``decoupled``. ``--lane sonic`` is refused rather than silently wrong.

NO CHECKPOINT? The server starts anyway on a hold-still policy that repeats
the measured arm pose through the same FK and encoding path. That is what
makes `conformance.py` meaningful without weights, and it is the behaviour the
bench sees if the weights fail to mount -- a robot that holds still, not one
that publishes zeros.

THOR SETUP, THE PART THAT BITES: a plain `uv sync` installs the dGPU torch
build (sm_80/90/100/120) and every kernel launch dies with "no kernel image
available" on Thor's sm_110. docker/Dockerfile.thor uses the Isaac-GR00T Thor
install path, which is the only one that produces sm_110 kernels.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from components.policy.bct import (  # noqa: E402
    TRAINED_PROMPTS,
    Gr00tBctPolicy,
    HoldStillPolicy,
)
from components.policy.kinematics import G1WristKinematics  # noqa: E402
from components.policy.taskspace import (  # noqa: E402
    GRIPPER_OPEN_RAD,
    JointChunkError,
    TaskSpaceEncoder,
    validate_joint_chunk,
)
from components.transport import serve_policy  # noqa: E402

LANE = "decoupled"

# Boundary camera key for each view the checkpoint consumes. cam_head was
# trained on cam_0 of the source recording, which is the LEFT eye of the head
# stereo pair, so the stereo boundary key is the faithful one; --head-camera
# ego_view falls back to the mono frame, whose eye the organizer has not
# documented (see INSTRUCTIONS.md, "Open questions").
HEAD_CAMERA_CHOICES = ("ego_view_left", "ego_view")

# How often to repeat the 'holding still, camera never arrived' line. Once is
# not enough -- it scrolls away and the robot looks merely idle.
BLOCKED_LOG_PERIOD_S = 5.0


class Policy:
    """BCT checkpoint -> decoupled-lane task-space chunks."""

    OBS_CHUNK = 1

    def __init__(self, args: argparse.Namespace):
        if args.lane != LANE:
            raise SystemExit(
                "[server] this submission is lane {!r}: the checkpoint emits joint "
                "targets, not a SONIC latent. Refusing to start on lane {!r}.".format(
                    LANE, args.lane
                )
            )
        self.args = args
        self.kinematics = G1WristKinematics(
            urdf_path=args.urdf, ee_offset_m=args.ee_offset_m
        )
        self.encoder = TaskSpaceEncoder(
            kinematics=self.kinematics,
            model_row_hz=args.model_row_hz,
            output_row_hz=args.row_hz,
            use_measured_waist=(args.ee_frame == "pelvis"),
        )
        self.view_to_boundary_key = {
            "head": args.head_camera,
            "left_wrist": "left_wrist",
            "right_wrist": "right_wrist",
        }
        # The Dex1-1 rig publishes no hand state, so the gripper dims of the
        # 46-dim state are fed from our own last command.
        self._gripper_q = np.full(2, float(args.initial_gripper_rad))
        # Newest good frame per camera key: the organizer's server drops a
        # wrist key when that camera fails, and the checkpoint needs all three.
        self._last_images: Dict[str, np.ndarray] = {}
        self._degraded_since: Optional[float] = None
        self._blocked_logged_at = 0.0
        self._gate_failures = 0
        self._prompts_warned = set()

        self.policy = self._load_policy()

    # -- setup --------------------------------------------------------------

    def _load_policy(self):
        if not self.args.checkpoint:
            print(
                "[server] no --checkpoint: running the HOLD-STILL policy. The robot "
                "will not move. This is the correct mode for conformance and a "
                "deliberate fallback on the bench, not a working submission.",
                file=sys.stderr,
            )
            return HoldStillPolicy(delay_s=self.args.delay_ms / 1000.0)

        # Check the directory ourselves. Handed a path that does not exist,
        # `AutoModel.from_pretrained` decides it must be a Hugging Face repo id
        # and fails with "Repo id must be in the form 'repo_name'..." -- which
        # says nothing about the actual problem, that the weights are not
        # mounted. The entrypoint checks this too, but `--entrypoint bash`
        # skips it, so the server does not rely on that.
        checkpoint = Path(self.args.checkpoint)
        if not (checkpoint / "config.json").is_file():
            raise SystemExit(
                "[server] {} is not a checkpoint directory (no config.json).\n"
                "[server]   * on the bench: mount the weights read-only, e.g.\n"
                "[server]     -v /opt/weights:/weights:ro\n"
                "[server]   * to run without weights (conformance, wiring checks):\n"
                "[server]     pass an empty checkpoint, e.g. -e PEVAL_CHECKPOINT=\n"
                "[server]     which starts the hold-still policy instead.".format(
                    checkpoint
                )
            )
        try:
            policy = Gr00tBctPolicy(
                checkpoint_path=self.args.checkpoint,
                kinematics=self.kinematics,
                embodiment_tag=self.args.embodiment_tag,
                device=self.args.device,
                denoising_steps=self.args.denoising_steps,
                horizon=self.args.horizon,
                row_hz=self.args.model_row_hz,
            )
        except Exception:
            print(traceback.format_exc(), file=sys.stderr)
            raise SystemExit(
                "[server] failed to load {}. Refusing to fall back to hold-still "
                "silently: a checkpoint was asked for, so a load failure is a "
                "configuration error, not a runtime condition.".format(
                    self.args.checkpoint
                )
            )
        print(
            "[server] loaded {} (tag={}, horizon={}, {} denoising steps)".format(
                self.args.checkpoint,
                self.args.embodiment_tag,
                policy.horizon,
                self.args.denoising_steps,
            )
        )
        return policy

    # -- the boundary-facing contract ---------------------------------------

    @property
    def metadata(self) -> dict:
        """Announced to the client on connect, before any observation."""
        execute_rows = min(self.args.execute_rows, self.policy.horizon)
        return {
            "lane": LANE,
            # Rows in the chunk we return, at action_row_hz -- not the model's
            # own horizon, which is longer than we ever execute.
            "action_chunk_size": self.encoder.rows_for(execute_rows),
            "obs_chunk_size": self.OBS_CHUNK,
            "camera_keys": [
                self.view_to_boundary_key["head"],
                "left_wrist",
                "right_wrist",
            ],
            "wants_state": True,
            "wants_prompt": True,
            # --- team extensions; the organizer's boundary ignores these -----
            # Row spacing of the chunk we publish. The client needs it to know
            # how many leading rows inference latency has already eaten.
            "action_row_hz": self.args.row_hz,
            "model_row_hz": self.args.model_row_hz,
            "model_horizon": self.policy.horizon,
            "execute_rows": execute_rows,
            "policy": type(self.policy).__name__,
            "checkpoint": self.args.checkpoint or "<hold-still>",
            "ee_frame": self.args.ee_frame,
            "ee_offset_m": self.args.ee_offset_m,
        }

    def act(self, obs: dict) -> dict:
        """One inference step. Returns {"actions": (T, 25) float32}."""
        body_q = np.asarray(obs["body_q"], dtype=np.float64)
        base_quat = np.asarray(obs["base_quat"], dtype=np.float64)
        prompt = obs.get("prompt", "")
        self._check_prompt(prompt)
        images, blocked = self._collect_images(obs.get("images", {}))

        if blocked and self.policy.needs_images:
            # A camera the checkpoint needs has never arrived. Hold the measured
            # pose and keep saying why: a server that dies here takes the
            # control loop with it, and a policy fed a black frame is worse than
            # one that does nothing.
            self._report_blocked(blocked)
            arm_targets, gripper_targets = self._hold_still_targets(body_q)
        else:
            arm_targets, gripper_targets = self._infer(
                images, body_q, base_quat, prompt
            )

        execute_rows = min(self.args.execute_rows, len(arm_targets))
        arm_targets = arm_targets[:execute_rows]
        gripper_targets = gripper_targets[:execute_rows]

        measured_arm = np.concatenate((body_q[15:22], body_q[22:29]))
        try:
            validate_joint_chunk(arm_targets, gripper_targets, measured_arm)
        except JointChunkError as exc:
            # The joint gates see failures FK would hide -- an undecoded
            # relative chunk still produces a perfectly valid-looking pose.
            # Hold the measured pose for this tick rather than publish it.
            self._gate_failures += 1
            print(
                "[server] JOINT GATE REJECTED chunk #{}: {} -- holding the "
                "measured pose for this tick".format(self._gate_failures, exc),
                file=sys.stderr,
            )
            arm_targets = np.tile(measured_arm, (execute_rows, 1))
            gripper_targets = np.tile(self._gripper_q, (execute_rows, 1))

        actions = self.encoder.encode(
            arm_targets, gripper_targets, waist_q=body_q[12:15]
        )
        # By the next observation the gripper should be tracking the last row
        # we published, and that is the best estimate of its state we have.
        self._gripper_q = np.asarray(gripper_targets[-1], dtype=np.float64).reshape(2)
        return {"actions": actions}

    def reset(self) -> dict:
        """Called once at the start of every attempt. Drop episode state."""
        self.policy.reset()
        self._gripper_q = np.full(2, float(self.args.initial_gripper_rad))
        self._last_images.clear()
        self._degraded_since = None
        self._blocked_logged_at = 0.0
        self._gate_failures = 0
        self._prompts_warned.clear()
        print("[server] reset")
        return {"ok": True}

    # -- helpers ------------------------------------------------------------

    def _infer(self, images, body_q, base_quat, prompt):
        if isinstance(self.policy, HoldStillPolicy):
            return self.policy.infer(
                images, body_q, base_quat, self._gripper_q, prompt
            )
        return self.policy.infer(
            images,
            body_q,
            base_quat,
            self._gripper_q,
            prompt,
            view_to_boundary_key=self.view_to_boundary_key,
        )

    def _check_prompt(self, prompt: str) -> None:
        """An unseen instruction is out of distribution, and it never errors."""
        if prompt in TRAINED_PROMPTS or prompt in self._prompts_warned:
            return
        self._prompts_warned.add(prompt)
        print(
            "[server] WARNING: prompt {!r} is not one of the five the checkpoint "
            "was trained on {}. The policy will still return a confident-looking "
            "chunk -- it just was not asked anything it knows.".format(
                prompt, list(TRAINED_PROMPTS)
            ),
            file=sys.stderr,
        )

    def _hold_still_targets(self, body_q: np.ndarray):
        arm = np.concatenate((body_q[15:22], body_q[22:29]))
        rows = min(self.args.execute_rows, self.policy.horizon)
        return np.tile(arm, (rows, 1)), np.tile(self._gripper_q, (rows, 1))

    def _collect_images(self, images: dict):
        """Fill each declared camera, reusing its newest frame when one drops.

        Returns ``(images, blocked)`` where ``blocked`` names the cameras that
        have never produced a frame at all. The organizer's camera server drops
        individual wrist keys when those cameras fail, and the checkpoint has no
        missing-view mode, so a stale frame beats a black one -- for as long as
        it takes someone to read the log, hence the warning.
        """
        filled = {}
        stale = []
        blocked = []
        for key in sorted(set(self.view_to_boundary_key.values())):
            image = images.get(key)
            if image is not None:
                self._last_images[key] = image
                filled[key] = image
            elif key in self._last_images:
                filled[key] = self._last_images[key]
                stale.append(key)
            else:
                blocked.append(key)

        now = time.time()
        if stale:
            if self._degraded_since is None:
                self._degraded_since = now
                print(
                    "[server] WARNING: camera(s) {} stopped publishing; reusing "
                    "their last good frame. The policy is acting on a stale "
                    "view.".format(stale),
                    file=sys.stderr,
                )
        elif self._degraded_since is not None:
            print("[server] all cameras back after {:.1f}s".format(
                now - self._degraded_since))
            self._degraded_since = None
        return filled, blocked

    def _report_blocked(self, blocked):
        """Say why we are holding still, on first occurrence and then rarely."""
        now = time.time()
        if now - self._blocked_logged_at < BLOCKED_LOG_PERIOD_S:
            return
        self._blocked_logged_at = now
        hint = ""
        if self.view_to_boundary_key["head"] in blocked:
            hint = (
                " The head key {!r} is the stereo one; if the organizer has not "
                "enabled stereo for us, restart with --head-camera ego_view."
            ).format(self.view_to_boundary_key["head"])
        print(
            "[server] HOLDING STILL: camera(s) {} have never published, and this "
            "checkpoint needs all three views.{}".format(sorted(blocked), hint),
            file=sys.stderr,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lane", default=os.environ.get("PEVAL_LANE", LANE),
                        help="Must match the manifest. This submission is 'decoupled'.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("PEVAL_THOR_PORT", "8765")))

    model = parser.add_argument_group("model")
    model.add_argument("--checkpoint", default=os.environ.get("PEVAL_CHECKPOINT", ""),
                       help="Local path to the checkpoint directory. Empty runs the "
                            "hold-still policy.")
    model.add_argument("--embodiment-tag", default="new_embodiment")
    model.add_argument("--device", default="cuda")
    model.add_argument("--denoising-steps", type=int, default=4,
                       help="4 is this checkpoint's calibrated value; every "
                            "open-loop number was measured there.")
    model.add_argument("--horizon", type=int, default=40,
                       help="Rows the checkpoint predicts. 40 @ 30 Hz = 1.33 s.")
    model.add_argument("--model-row-hz", type=float, default=30.0,
                       help="Row spacing the checkpoint was trained at.")

    control = parser.add_argument_group("control")
    control.add_argument("--execute-rows", type=int, default=16,
                         help="Model rows executed per inference before replanning. "
                              "Must cover inference latency twice over: the client "
                              "discards the leading rows that latency already ate, "
                              "and what is left has to last until the next chunk "
                              "arrives. At the 185 ms measured on the Thor, 8 rows "
                              "leaves 3 of 12 alive -- 60 ms of motion per 185 ms "
                              "cycle, so the controller holds its last command two "
                              "thirds of the time. 16 rows leaves 17 of 26, i.e. "
                              "340 ms of motion per cycle. The cost is accuracy "
                              "deeper into the chunk: arm MAE is 1.20 deg at 5 rows, "
                              "1.50 at 8, 2.25 at 16, 3.98 at 40.")
    control.add_argument("--row-hz", type=float, default=50.0,
                         help="Row spacing of the published (T, 25) chunk. 50 Hz "
                              "matches the controller cadence; the model's 30 Hz "
                              "rows are resampled in joint space.")
    control.add_argument("--head-camera", choices=HEAD_CAMERA_CHOICES,
                         default=os.environ.get("PEVAL_HEAD_CAMERA", "ego_view_left"),
                         help="Boundary key for the head view. The checkpoint was "
                              "trained on the LEFT eye.")
    control.add_argument("--initial-gripper-rad", type=float, default=GRIPPER_OPEN_RAD,
                         help="Assumed gripper position at reset, in Dex1-1 motor "
                              "radians (0 closed, 5.40 open). No hand state is "
                              "published for a Dex1-1 rig.")

    kin = parser.add_argument_group("kinematics")
    kin.add_argument("--ee-frame", choices=("pelvis", "torso"), default="pelvis",
                     help="Frame of the published end-effector poses. 'pelvis' runs "
                          "FK with the measured waist; 'torso' zeroes it, matching "
                          "the checkpoint's own state block. See INSTRUCTIONS.md.")
    kin.add_argument("--ee-offset-m", type=float, default=0.05,
                     help="Distance from the wrist_yaw link origin to the commanded "
                          "point, along the link's local +x.")
    kin.add_argument("--urdf", type=Path, default=None,
                     help="Override the bundled assets/g1/g1_body29_hand14.urdf.")

    parser.add_argument("--delay-ms", type=float, default=0.0,
                        help="Hold-still policy only: fake inference time, to see "
                             "how the client behaves at realistic latency.")
    return parser


def main():
    args = build_parser().parse_args()
    policy = Policy(args)
    meta = policy.metadata
    print(
        "[server] lane={} policy={} chunk={} rows @ {:g} Hz "
        "(execute {} model rows of {}) ee_frame={}".format(
            meta["lane"], meta["policy"], meta["action_chunk_size"], meta["action_row_hz"],
            meta["execute_rows"], meta["model_horizon"], meta["ee_frame"],
        )
    )
    serve_policy(policy, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
