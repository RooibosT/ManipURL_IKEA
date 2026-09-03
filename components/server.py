#!/usr/bin/env python3
"""Policy server — runs on the Jetson AGX Thor. Team-owned.

    Thor  192.168.100.1   this file, the checkpoint, the GPU
    Orin  192.168.100.2   components/client.py and the organizer's endpoints

    python components/server.py --lane decoupled --port 8765 \
        --checkpoint /weights/gr00t-n1.7-g1-dex1-ikea-3task-46d-30hz-h40

What happens per call to :meth:`Policy.act`:

    boundary obs (body_q 29, base_quat 4, 3 images)
        -> 46-dim BCT state  (policy/bct.py)
        -> GR00T N1.7 inference, horizon 40 @ 30 Hz, arms restored to absolute
        -> take the first --execute-rows rows
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
    READY_ARM_Q_BY_TASK,
    TRAINED_PROMPTS,
    Gr00tBctPolicy,
    HoldStillPolicy,
)
from components.imagecodec import decode_images  # noqa: E402
from components.policy.kinematics import (  # noqa: E402
    ACTION_EE_OFFSET_M,
    TRAINING_EE_OFFSET_M,
    G1WristKinematics,
)
from components.policy.taskspace import (  # noqa: E402
    MAX_ARM_STEP_RAD,
    JointChunkError,
    TaskSpaceEncoder,
    GRIPPER_RESET_RAD_BY_TASK,
    command_to_measured_rad,
    measured_to_command_rad,
    reset_gripper_rad,
    validate_joint_chunk,
)
from components.transport import serve_policy  # noqa: E402
try:  # noqa: E402
    from boundary.actions import MAX_CHUNK_LENGTH
except ImportError:                     # boundary/__init__ pulls cv2 and zmq,
    # and docker/requirements-thor.txt says the server never needs them. Stay
    # in sync with the contract when they are installed, and keep starting when
    # they are not -- this is a bound we read, not a wire we speak.
    MAX_CHUNK_LENGTH = 64

LANE = "decoupled"

# Boundary camera key for each view the checkpoint consumes. cam_left_high was
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
        # TWO kinematics, one per convention, and they must not be merged.
        # `kinematics` feeds the model's 46-dim state and is pinned to the
        # offset the checkpoint was trained with. `action_kinematics` feeds the
        # wire and carries whatever the organizer's IK adapter wants (zero).
        # Sharing one instance is what made --ee-offset-m a trap: it moved 6 of
        # the 46 state dims off the training distribution as a side effect of
        # fixing the wire. The URDF parse is cached, so the pair is free.
        self.kinematics = G1WristKinematics(
            urdf_path=args.urdf, ee_offset_m=TRAINING_EE_OFFSET_M
        )
        self.action_kinematics = G1WristKinematics(
            urdf_path=args.urdf, ee_offset_m=args.ee_offset_m
        )
        self.encoder = TaskSpaceEncoder(
            kinematics=self.action_kinematics,
            model_row_hz=args.model_row_hz,
            output_row_hz=args.row_hz,
            use_measured_waist=(args.ee_frame == "pelvis"),
        )
        self.view_to_boundary_key = {
            "head": args.head_camera,
            "left_wrist": "left_wrist",
            "right_wrist": "right_wrist",
        }
        # The Dex1-1 rig publishes no hand state, so we carry two jaw values
        # and they are NOT interchangeable:
        #   _gripper_q   the MEASURED jaw the model reads, in its state dims
        #   _gripper_cmd the last COMMAND we published, for holding it
        # They differ by the training rig's calibration. Keeping one field and
        # converting at each use is what let the calibration be applied to its
        # own output every tick -- a geometric drift toward 4.45/4.61 rad that
        # opens a closed hand in about three ticks and drops whatever it holds.
        # Both are None until the first observation: the seed is per-task and
        # the prompt arrives with the observation, not at reset.
        self._gripper_q: Optional[np.ndarray] = None
        self._gripper_cmd: Optional[np.ndarray] = None
        self._seeded_prompt: Optional[str] = None
        # Newest good frame per camera key: the organizer's server drops a
        # wrist key when that camera fails, and the checkpoint needs all three.
        self._last_images: Dict[str, np.ndarray] = {}
        self._degraded_since: Optional[float] = None
        self._blocked_logged_at = 0.0
        self._gate_failures = 0
        self._prompts_warned = set()
        # Ready move: hands, then arms, then the policy. None until the first
        # observation names the task.
        self._ready_phase = "hands" if args.ready_move else "done"
        self._ready_since: Optional[float] = None
        self._ready_started_at: Optional[float] = None

        self.policy = self._load_policy()
        self.execute_rows = self._clamp_execute_rows()
        self._check_ready_velocity()

    def _check_ready_velocity(self) -> None:
        """A ready ramp that trips our own joint gate would reject every chunk.

        `_ramp`'s per-row step is velocity / model_row_hz, and the gate rejects
        anything over MAX_ARM_STEP_RAD. Over that, the ready move never
        converges, the timeout fires 15 s later and the arm never gets there --
        so fail at startup, the way the chunk length does, rather than on the
        bench mid-move. Both of these are advertised as no-rebuild knobs.
        """
        if not self.args.ready_move:
            return
        for name, velocity in (("--ready-velocity-rad-s",
                                self.args.ready_velocity_rad_s),):
            step = velocity / self.args.model_row_hz
            if step > MAX_ARM_STEP_RAD:
                raise SystemExit(
                    "[server] {} {:g} is {:.3f} rad per model row at {:g} Hz, over "
                    "the {:.2f} rad joint gate -- every ready chunk would reject "
                    "itself. Use at most {:g}.".format(
                        name, velocity, step, self.args.model_row_hz,
                        MAX_ARM_STEP_RAD, MAX_ARM_STEP_RAD * self.args.model_row_hz)
                )

    def _clamp_execute_rows(self) -> int:
        """Model rows we execute, capped so the encoded chunk fits the contract.

        Resampling 30 Hz model rows up to 50 Hz multiplies them by 5/3, and
        `DecoupledSink` rejects anything longer than MAX_CHUNK_LENGTH rows. At
        those rates the ceiling is 39 model rows (which resample to exactly 64)
        -- so `--execute-rows 40`, the
        obvious "execute the whole horizon" value and equal to `--horizon`'s
        own default, would have the client die on its FIRST publish with an
        ActionError, on the bench, with the robot holding its last command.
        Clamp here rather than discover it there.
        """
        # max(1, ...): --execute-rows 0 declared a 0-row chunk and then raised
        # IndexError on gripper_targets[-1] at the first observation, killing
        # the control loop with the robot live; negatives sliced from the end
        # and published a chunk the metadata had not declared.
        rows = max(1, min(self.args.execute_rows, self.policy.horizon))
        capped = rows
        while capped > 1 and self.encoder.rows_for(capped) > MAX_CHUNK_LENGTH:
            capped -= 1
        if capped != self.args.execute_rows:
            if capped < rows:
                why = ("{} model rows at {:g} Hz resample to {} rows at {:g} Hz, "
                       "over the contract's {}-row limit".format(
                           rows, self.args.model_row_hz,
                           self.encoder.rows_for(rows), self.args.row_hz,
                           MAX_CHUNK_LENGTH))
            else:
                why = "the checkpoint only predicts {} rows".format(
                    self.policy.horizon)
            print("[server] --execute-rows {} -> {}: {}.".format(
                self.args.execute_rows, capped, why), file=sys.stderr)
        return capped

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
        execute_rows = self.execute_rows
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
            "accepts_jpeg": True,
            "policy": type(self.policy).__name__,
            "checkpoint": self.args.checkpoint or "<hold-still>",
            "ready_move": self.args.ready_move,
            "ee_frame": self.args.ee_frame,
            "ee_offset_m": self.args.ee_offset_m,          # on the wire
            "state_ee_offset_m": TRAINING_EE_OFFSET_M,     # into the model
        }

    def act(self, obs: dict) -> dict:
        """One inference step. Returns {"actions": (T, 25) float32}."""
        body_q = np.asarray(obs["body_q"], dtype=np.float64)
        base_quat = np.asarray(obs["base_quat"], dtype=np.float64)
        prompt = obs.get("prompt", "")
        self._check_prompt(prompt)
        self._seed_gripper(prompt)
        images, blocked = self._collect_images(self._images_from(obs))

        if blocked and self.policy.needs_images:
            # A camera the checkpoint needs has never arrived. Hold the measured
            # pose and keep saying why: a server that dies here takes the
            # control loop with it, and a policy fed a black frame is worse than
            # one that does nothing.
            #
            # This is checked BEFORE the ready move, not after. Ramping the arms
            # 13-26 cm while blind is worse than not ramping, and burying the
            # one message that names the fix (--head-camera ego_view) under 15 s
            # of ready move defeats the reason it repeats at all.
            self._report_blocked(blocked)
            arm_targets, gripper_targets = self._hold_still_targets(body_q)
        elif self._ready_phase != "done":
            # The policy does not run until the upper body is where the
            # demonstrations start. Inference is skipped entirely, so the first
            # real observation is taken from the right arm AND the right jaws.
            arm_targets, gripper_targets = self._ready_targets(body_q, prompt)
        else:
            arm_targets, gripper_targets = self._infer(
                images, body_q, base_quat, prompt
            )

        execute_rows = min(self.execute_rows, len(arm_targets))
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
            gripper_targets = np.tile(self._gripper_cmd, (execute_rows, 1))

        actions = self.encoder.encode(
            arm_targets, gripper_targets, waist_q=body_q[12:15]
        )
        # By the next observation the jaw should be tracking the last row we
        # published. The model reads the MEASURED position, which on the
        # training rig is an affine function of the command rather than a copy
        # of it, so go through that calibration rather than feeding the command
        # back raw -- it cuts the left hand's error from 0.170 to 0.008 rad.
        self._gripper_cmd = np.asarray(gripper_targets[-1], dtype=np.float64).reshape(2)
        self._gripper_q = command_to_measured_rad(self._gripper_cmd)
        return {"actions": actions}

    def reset(self) -> dict:
        """Called once at the start of every attempt. Drop episode state."""
        self.policy.reset()
        self._gripper_q = None          # re-seeded from the next prompt
        self._gripper_cmd = None
        self._seeded_prompt = None
        self._last_images.clear()
        self._degraded_since = None
        self._blocked_logged_at = 0.0
        self._gate_failures = 0
        self._prompts_warned.clear()
        self._ready_phase = "hands" if self.args.ready_move else "done"
        self._ready_since = None
        self._ready_started_at = None
        print("[server] reset ({})".format(
            "ready move armed" if self.args.ready_move else "no ready move"))
        return {"ok": True}

    # -- helpers ------------------------------------------------------------

    def _infer(self, images, body_q, base_quat, prompt):
        if isinstance(self.policy, HoldStillPolicy):
            # HoldStillPolicy echoes this straight back as the action, so it
            # wants the COMMAND. The real policy wants the measurement, because
            # its argument goes into the model's state block.
            return self.policy.infer(
                images, body_q, base_quat, self._gripper_cmd, prompt
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
            "[server] WARNING: prompt {!r} is not one of the three the checkpoint "
            "was trained on {}. The policy will still return a confident-looking "
            "chunk -- it just was not asked anything it knows.".format(
                prompt, list(TRAINED_PROMPTS)
            ),
            file=sys.stderr,
        )

    def _seed_gripper(self, prompt: str) -> None:
        """Assume a jaw position for the first inference of an attempt.

        Each of the three subtasks starts from a different grasp state -- two of
        them begin with the leg already held -- so this is per-task and per-hand
        and cannot be one number. `--initial-gripper-rad` overrides both hands
        when you want to pin it by hand.
        """
        if self._gripper_q is not None:
            # Re-arm in exactly one case: we seeded off a prompt with no
            # training-set start and a real one has now arrived. One empty or
            # mistyped prompt on the first observation would otherwise latch
            # the both-open fallback and the skipped ready move for the whole
            # attempt -- and on `rotate leg to tighten` that tells the model a
            # closed hand is open, wrong by the entire 5.2 rad stroke, on the
            # inference that decides the opening move. Narrow on purpose: an
            # unknown prompt means nothing meaningful was under way, so there
            # is no attempt in progress to disturb.
            if (self._seeded_prompt in GRIPPER_RESET_RAD_BY_TASK
                    or prompt not in GRIPPER_RESET_RAD_BY_TASK):
                return
            print("[server] prompt {!r} -> {!r}: re-seeding off a real "
                  "training-set start".format(self._seeded_prompt, prompt))
            self._ready_phase = "hands" if self.args.ready_move else "done"
            self._ready_since = None
            self._ready_started_at = None
        self._gripper_q = self._target_gripper_q(prompt)
        self._gripper_cmd = measured_to_command_rad(self._gripper_q)
        if self.args.initial_gripper_rad is not None:
            source = "--initial-gripper-rad"
        elif prompt in GRIPPER_RESET_RAD_BY_TASK:
            source = "training-set start for {!r}".format(prompt)
        else:
            # reset_gripper_rad falls through to both-open for an unknown key,
            # and saying "training-set start" there would be a lie about a
            # value that can be wrong by the whole stroke on a subtask that
            # begins holding the leg.
            source = ("the both-open FALLBACK -- {!r} has no training-set start"
                      .format(prompt))
        print("[server] gripper state seeded to (left {:.2f}, right {:.2f}) rad "
              "from {}".format(self._gripper_q[0], self._gripper_q[1], source))
        self._seeded_prompt = prompt

    def _target_gripper_q(self, prompt: str) -> np.ndarray:
        """The MEASURED jaw position this task starts from, both hands.

        One source for the state seed and the ready move's target, so
        `--initial-gripper-rad` cannot pin one and leave the other on the
        table's value.
        """
        if self.args.initial_gripper_rad is not None:
            return np.full(2, float(self.args.initial_gripper_rad))
        return reset_gripper_rad(prompt)

    def _ready_targets(self, body_q: np.ndarray, prompt: str):
        """Ramp the upper body to this task's recorded start pose.

        Hands first, then arms -- borrowed from the team's own Thor deployment
        for a reason worth keeping: a jaw that is holding something should drop
        it from where the arms are now, not from wherever the ready pose puts
        them.

        The ramp is slow on purpose. We publish end-effector poses and the
        ORGANIZER'S adapter decides the joint velocity that realises them, and
        their interpolator is what produced the -7.15 and -8.99 rad/s
        shutdowns on 2026-09-03. At the default 0.35 rad/s the chunk asks for
        0.012 rad per model row, seventeen times inside our own
        MAX_ARM_STEP_RAD gate, so these chunks take exactly the same
        validation, FK and encoding path a real inference does.
        """
        rows = self.execute_rows
        measured = np.concatenate((body_q[15:22], body_q[22:29]))
        target = self._ready_arm_q(prompt)
        if target is None:
            print(
                "[server] no recorded start pose for prompt {!r} (known: {}); "
                "SKIPPING the ready move and handing the policy the arms where "
                "they are.".format(prompt, sorted(READY_ARM_Q_BY_TASK)),
                file=sys.stderr,
            )
            self._ready_phase = "done"
            return self._hold_still_targets(body_q)
        # The table stores the MEASURED jaw, so aim at the command that
        # produces it and ramp in command space -- what we publish is a command.
        grip_target = measured_to_command_rad(self._target_gripper_q(prompt))
        # monotonic, not time.time(): a Jetson with no RTC battery sets its
        # clock from NTP right about when the first attempt starts, and a step
        # either way decides between an instant bogus timeout and a move that
        # never ends. components/client.py uses monotonic for the same reason.
        now = time.monotonic()
        if self._ready_started_at is None:
            self._ready_started_at = now

        if self._ready_phase == "hands":
            # Hold the arms still and command the jaws. We cannot see the jaw,
            # so there is nothing to converge on -- we wait a fixed dwell for it
            # to physically travel instead. Without that this phase would be a
            # single chunk and the arms would start moving while the jaw was
            # still opening, which is exactly the ordering it exists to prevent.
            # A STEP, not a ramp. Ramping needs a start, and no jaw position
            # reaches us -- our own estimate is by construction already at the
            # target, so any ramp from it is the identity. The jaw is commanded
            # outright and `--ready-hand-dwell-s` is the travel allowance.
            arm = np.tile(measured, (rows, 1))
            grip = np.tile(grip_target, (rows, 1))
            if now - self._ready_started_at >= self.args.ready_hand_dwell_s:
                # No assignment here: act() already ends by running the last
                # published command back through the calibration, which is the
                # same estimate and keeps one path for it.
                print("[server] ready: jaws commanded to (left {:.2f}, right "
                      "{:.2f}) rad and given {:.1f}s to travel; moving the arms"
                      .format(grip_target[0], grip_target[1],
                              self.args.ready_hand_dwell_s))
                self._ready_phase = "arms"
                self._ready_started_at = now      # the arm timeout starts here
            return arm, grip

        arm = self._ramp(measured, target, rows, self.args.ready_velocity_rad_s,
                         self.args.model_row_hz)
        grip = np.tile(grip_target, (rows, 1))
        error = float(np.max(np.abs(measured - target)))

        if error <= self.args.ready_tolerance_rad:
            if self._ready_since is None:
                self._ready_since = now
            elif now - self._ready_since >= self.args.ready_stable_s:
                print("[server] READY: arms within {:.3f} rad of the {!r} start "
                      "pose, held {:.1f}s. Policy starts now.".format(
                          error, prompt, self.args.ready_stable_s))
                self._ready_phase = "done"
        else:
            self._ready_since = None
            if now - self._ready_started_at > self.args.ready_timeout_s:
                # Proceeding beats refusing to start: an attempt that never
                # begins scores zero for certain. Say so loudly -- the arm is
                # out of distribution and every number from this run inherits
                # that. The likeliest cause is the organizer's IK choosing a
                # different elbow, which a 6-DoF pose cannot pin on a 7-DoF arm.
                print(
                    "[server] READY MOVE TIMED OUT after {:.0f}s at {:.3f} rad "
                    "(tolerance {:.3f}). Starting the policy anyway on an arm "
                    "that is NOT at the demonstration start pose.".format(
                        self.args.ready_timeout_s, error,
                        self.args.ready_tolerance_rad),
                    file=sys.stderr,
                )
                self._ready_phase = "done"
        return arm, grip

    def _ready_arm_q(self, prompt: str) -> Optional[np.ndarray]:
        """The recorded start pose for ``prompt``, or None if we have none.

        Another subtask's pose does not transfer -- borrowing one is worse than
        not moving, because a wrong recorded pose is indistinguishable from the
        right one once the arm is already there. So an unknown prompt skips the
        ready move rather than guessing.

        It does NOT kill the server. This runs on the first observation, with
        the robot live and the client's control loop depending on the reply --
        the same reason a missing camera holds the pose instead of raising. The
        prompt is out of distribution either way and `_check_prompt` has
        already said so.
        """
        pose = READY_ARM_Q_BY_TASK.get(prompt)
        if pose is None:
            return None
        return np.asarray(pose, dtype=np.float64)

    @staticmethod
    def _ramp(start, target, rows: int, velocity_rad_s: float,
              row_hz: float = 30.0) -> np.ndarray:
        """``rows`` model rows marching from ``start`` toward ``target``."""
        start = np.asarray(start, dtype=np.float64).reshape(-1)
        target = np.asarray(target, dtype=np.float64).reshape(-1)
        delta = target - start
        distance = float(np.max(np.abs(delta)))
        if distance < 1e-9:
            return np.tile(target, (rows, 1))
        step = velocity_rad_s / float(row_hz)   # per model row, at the model rate
        fractions = np.minimum(step * np.arange(1, rows + 1) / distance, 1.0)
        return start + fractions[:, None] * delta

    def _hold_still_targets(self, body_q: np.ndarray):
        arm = np.concatenate((body_q[15:22], body_q[22:29]))
        rows = self.execute_rows
        return np.tile(arm, (rows, 1)), np.tile(self._gripper_cmd, (rows, 1))

    @staticmethod
    def _images_from(obs: dict) -> dict:
        """Accept either raw arrays or JPEG blobs; the client decides which.

        Raw is simpler and lossless; JPEG is what makes the link fast enough to
        matter. Supporting both means the choice stays a client flag rather than
        a rebuild of two images.
        """
        blobs = obs.get("images_jpeg")
        if blobs:
            return decode_images(blobs)
        return obs.get("images", {})

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
                              "deeper into the chunk; for this checkpoint only the "
                              "1-8 row window has been scored (arm MAE 1.45 deg, "
                              "wrist 11.8 mm), so the 16-row figure is not measured.")
    control.add_argument("--row-hz", type=float, default=50.0,
                         help="Row spacing of the published (T, 25) chunk. 50 Hz "
                              "matches the controller cadence; the model's 30 Hz "
                              "rows are resampled in joint space.")
    control.add_argument("--head-camera", choices=HEAD_CAMERA_CHOICES,
                         default=os.environ.get("PEVAL_HEAD_CAMERA", "ego_view_left"),
                         help="Boundary key for the head view. The checkpoint was "
                              "trained on the LEFT eye.")
    control.add_argument("--initial-gripper-rad", type=float, default=None,
                         help="Override the assumed MEASURED jaw at reset, both hands, "
                              "in Dex1-1 motor radians. THIS MOVES HARDWARE: it is "
                              "also the ready move's jaw setpoint, so on a subtask "
                              "that starts holding the leg (`rotate leg to tighten`, "
                              "left 0.17) passing an open value commands the hand open "
                              "and drops it. Stay inside the measured range the "
                              "training set actually contains, [0.177, 5.362] left and "
                              "[0.312, 5.346] right -- 5.40 is the mechanical end stop "
                              "and above anything the checkpoint ever saw. Left unset, "
                              "the seed is the training set's per-task, per-hand "
                              "episode start, which one scalar cannot express since "
                              "two of the three subtasks begin with the leg held.")

    ready = parser.add_argument_group("ready move")
    ready.add_argument("--ready-move", dest="ready_move", action="store_true",
                       default=True,
                       help="Before the policy runs, ramp the upper body to the "
                            "recorded start pose for the active prompt. The 2026-09-03 "
                            "dry run sat ~1.0 rad away from it for 11.4 minutes.")
    ready.add_argument("--no-ready-move", dest="ready_move", action="store_false",
                       help="Hand the arms straight to the policy, wherever they are.")
    ready.add_argument("--ready-velocity-rad-s", type=float, default=0.35,
                       help="Arm ramp speed. WE DO NOT SET THE JOINT VELOCITY -- we "
                            "publish poses and the organizer's adapter realises them, "
                            "and their interpolator is what produced the rad/s "
                            "shutdowns on 2026-09-03. Slow is the whole point.")
    ready.add_argument("--ready-hand-dwell-s", type=float, default=1.0,
                       help="How long to command the ready jaw position before the "
                            "arms move. No jaw position reaches us, so there is "
                            "nothing to converge on -- this is the travel time we "
                            "allow it. Jaws move first so a hand that is holding "
                            "something drops it from where the arms are now.")
    ready.add_argument("--ready-tolerance-rad", type=float, default=0.10,
                       help="Joint error that counts as arrived. Looser than a "
                            "direct-drive move would need, because the organizer's "
                            "IK picks the elbow swivel a 6-DoF pose cannot pin.")
    ready.add_argument("--ready-stable-s", type=float, default=0.30,
                       help="How long the error must hold before the policy starts.")
    ready.add_argument("--ready-timeout-s", type=float, default=15.0,
                       help="Give up and start the policy anyway, loudly. An attempt "
                            "that never begins scores zero for certain.")

    kin = parser.add_argument_group("kinematics")
    kin.add_argument("--ee-frame", choices=("pelvis", "torso"), default="pelvis",
                     help="Waist handling for the published poses, NOT two different "
                          "frames -- both are pelvis-origin. 'pelvis' runs FK with the "
                          "measured waist, i.e. where the wrist actually is. 'torso' "
                          "locks the waist at zero, reproducing the checkpoint's own "
                          "state convention; the name is historical and misleading, "
                          "since a true torso_link-origin pose is a further 4.42 cm "
                          "away. See INSTRUCTIONS.md.")
    kin.add_argument("--ee-offset-m", type=float, default=ACTION_EE_OFFSET_M,
                     help="Tool offset of the PUBLISHED pose only, in metres along "
                          "the wrist_yaw link's local +x. 0 targets the bare link "
                          "origin, which is what the organizer's IK adapter wants "
                          "(confirmed 2026-09-03). The model's own state block is "
                          "NOT affected by this flag -- it stays pinned at the "
                          "{:g} m the checkpoint was trained with.".format(
                              TRAINING_EE_OFFSET_M))
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
        "(execute {} model rows of {}) ee_frame={} ready_move={}".format(
            meta["lane"], meta["policy"], meta["action_chunk_size"], meta["action_row_hz"],
            meta["execute_rows"], meta["model_horizon"], meta["ee_frame"],
            "on" if meta["ready_move"] else "OFF",
        )
    )
    serve_policy(policy, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
