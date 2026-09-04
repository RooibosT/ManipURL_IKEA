#!/usr/bin/env python3
"""The two invariants nothing else can catch. No weights, no GPU, ~1 second.

    python scripts/check_conventions.py

1. THE STATE OFFSET IS INDEPENDENT OF THE WIRE OFFSET. Both used to come from
   one `--ee-offset-m`, so setting the wire offset the organizer's IK wants
   also moved 6 of the 46 state dims 5 cm off the training distribution. That
   failure is silent: every chunk still passes `DecoupledSink.validate_chunk`,
   and the policy just gets worse.

2. THE PUBLISHED CHUNK FITS THE CONTRACT AT EVERY --execute-rows. Resampling
   30 Hz rows to 50 Hz multiplies them by 5/3, so 40 model rows overflow the
   64-row limit (39 fits at exactly 64) and the client dies on its first
   publish. Zero and negatives are checked too: both used to survive.

3. THE GRIPPER STATE WE SYNTHESIZE STAYS INSIDE THE TRAINING DISTRIBUTION. No
   jaw position is on the wire, so we manufacture that model input; the numbers
   below are pinned to what URL-RFM/IKEA_pickuptheleg actually contains.

4. THE READY MOVE STAYS INSIDE OUR OWN JOINT GATE. We publish poses and the
   organizer's adapter picks the joint velocity that realises them, so the only
   speed control we have is how small a step we ask for. A ramp that trips
   MAX_ARM_STEP_RAD would be rejected by our own gate at run time -- on the
   bench, mid-move.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from boundary.actions import MAX_CHUNK_LENGTH, DecoupledSink  # noqa: E402
from components.policy.bct import READY_ARM_Q_BY_TASK, TRAINED_PROMPTS, build_state  # noqa: E402
from components.server import Policy, build_parser  # noqa: E402
from components.policy.kinematics import (  # noqa: E402
    ACTION_EE_OFFSET_M,
    TRAINING_EE_OFFSET_M,
    G1WristKinematics,
)
from components.policy.taskspace import (  # noqa: E402
    GRIPPER_CLOSED_RAD,
    GRIPPER_OPEN_RAD,
    GRIPPER_RESET_RAD_BY_TASK,
    MAX_ARM_STEP_RAD,
    MAX_FIRST_ARM_ERROR_RAD,
    TaskSpaceEncoder,
    command_to_measured_rad,
    gripper_rad_to_command,
    measured_to_command_rad,
    reset_gripper_rad,
    validate_joint_chunk,
)

# Measured off URL-RFM/IKEA_pickuptheleg, 322 episodes / 149,437 frames.
TRAIN_STATE_MIN, TRAIN_STATE_MAX = -0.0255, 5.3921
TRAIN_ACTION_MIN, TRAIN_ACTION_MAX = 0.0, 5.4000

BODY_Q = 0.15 * np.sin(np.arange(29) * 0.2)
BASE_QUAT = np.array([1.0, 0.0, 0.0, 0.0])
GRIP = np.array([4.46, 4.46])
ARM = np.concatenate((BODY_Q[15:22], BODY_Q[22:29]))


def build_server(*argv) -> Policy:
    """A real Policy, so these checks see the server's wiring and not a copy.

    Rebuilding the wiring here is what made the first version of this file
    useless: it asserted things about objects it had constructed itself, so
    re-merging the two offsets in server.py left every check green.
    """
    with open(os.devnull, "w") as quiet:
        stderr, sys.stderr = sys.stderr, quiet
        try:
            return Policy(build_parser().parse_args(["--lane", "decoupled", *argv]))
        finally:
            sys.stderr = stderr


def eef_state(policy: Policy) -> np.ndarray:
    """The eef state blocks THIS server would hand the model."""
    s = build_state(BODY_Q, BASE_QUAT, GRIP, policy.kinematics)
    return np.concatenate([s["left_eef"].reshape(-1), s["right_eef"].reshape(-1)])


def main() -> int:
    # 1. the SERVER's state path must not move when its wire offset does
    baseline = None
    poses = {}
    for wire in (0.0, 0.05, 0.12, -0.03):
        srv = build_server("--ee-offset-m", str(wire))
        assert srv.kinematics.ee_offset_m == TRAINING_EE_OFFSET_M, (
            "server state kinematics carries {} m, not the training offset -- "
            "the two offsets have been merged again".format(srv.kinematics.ee_offset_m)
        )
        assert srv.encoder.kinematics.ee_offset_m == wire, (
            "server wire kinematics carries {} m, not --ee-offset-m {}".format(
                srv.encoder.kinematics.ee_offset_m, wire)
        )
        got = eef_state(srv)
        if baseline is None:
            baseline = got
        drift = float(np.abs(got - baseline).max())
        assert drift == 0.0, (
            "state eef moved {:.4f} m at --ee-offset-m {}".format(drift, wire)
        )
        poses[wire] = srv.encoder.encode(
            np.tile(ARM, (2, 1)), np.tile(GRIP, (2, 1)), waist_q=BODY_Q[12:15]
        )[0, 4:7]
    print("  server state eef is independent of --ee-offset-m       OK")

    # ...and the wire must actually honour it, or the fix does nothing
    moved = float(np.linalg.norm(poses[0.05] - poses[0.0]))
    # the chunk is float32, so compare at float32 precision, not float64
    assert abs(moved - 0.05) < 1e-6, "wire offset ignored: pose moved {:.4f} m".format(moved)
    print("  server wire pose tracks --ee-offset-m ({:.2f} cm)        OK".format(moved * 100))

    # 2. the SERVER's own clamp must keep every --execute-rows publishable
    for rows in (0, -5, 1, 8, 16, 26, 38, 39, 40, 100):
        srv = build_server("--execute-rows", str(rows))
        assert srv.execute_rows >= 1, "--execute-rows {} left {}".format(
            rows, srv.execute_rows)
        declared = srv.metadata["action_chunk_size"]
        chunk = srv.encoder.encode(
            np.tile(ARM, (srv.execute_rows, 1)),
            np.tile(GRIP, (srv.execute_rows, 1)), waist_q=BODY_Q[12:15])
        assert len(chunk) == declared, (
            "--execute-rows {}: declared {} rows, encoded {}".format(
                rows, declared, len(chunk))
        )
        DecoupledSink.validate_chunk(chunk)          # raises if over the limit
    print("  server clamps --execute-rows into the {}-row contract   OK".format(
        MAX_CHUNK_LENGTH))

    # 3. holding a pose must not move the jaw. _gripper_q is the MEASURED jaw
    #    and _gripper_cmd the last command; publishing the former re-applies
    #    the calibration to its own output every tick, which opened a closed
    #    hand in three ticks and dropped the leg.
    srv = build_server()
    obs = {"body_q": BODY_Q, "base_quat": BASE_QUAT, "images": {},
           "prompt": "insert table leg to table base"}
    srv.act(obs)
    srv._ready_phase = "done"
    srv.policy.needs_images = True
    srv._last_images.clear()                          # force the hold-still path
    held = [srv.act(obs)["actions"][0, 2] for _ in range(40)]
    spread = float(np.max(held) - np.min(held))
    assert spread < 1e-6, (
        "the held jaw command drifted {:.4f} over 40 ticks (from {:.4f} to "
        "{:.4f}) -- a measured value is being republished as a command".format(
            spread, held[0], held[-1])
    )
    print("  a held jaw command does not drift over 40 ticks        OK")

    # 3. the gripper state we manufacture must land inside the training range
    assert (GRIPPER_CLOSED_RAD, GRIPPER_OPEN_RAD) == (TRAIN_ACTION_MIN, TRAIN_ACTION_MAX), (
        "command scale endpoints {} do not match the training action range {}".format(
            (GRIPPER_CLOSED_RAD, GRIPPER_OPEN_RAD), (TRAIN_ACTION_MIN, TRAIN_ACTION_MAX))
    )
    for cmd in (GRIPPER_CLOSED_RAD, 2.35, GRIPPER_OPEN_RAD):
        m = command_to_measured_rad(np.full(2, cmd))
        assert TRAIN_STATE_MIN <= m.min() and m.max() <= TRAIN_STATE_MAX, (
            "command {} maps to measured {}, outside the training state range "
            "[{}, {}]".format(cmd, m, TRAIN_STATE_MIN, TRAIN_STATE_MAX)
        )
    print("  synthesized gripper state stays in the training range  OK")

    for prompt in GRIPPER_RESET_RAD_BY_TASK:
        seed = reset_gripper_rad(prompt)
        assert TRAIN_STATE_MIN <= seed.min() and seed.max() <= TRAIN_STATE_MAX, \
            "reset seed for {!r} is {}, outside the training range".format(prompt, seed)
    # the three tasks do NOT share a seed -- if they ever do, the table is broken
    seeds = {tuple(np.round(reset_gripper_rad(p), 3)) for p in GRIPPER_RESET_RAD_BY_TASK}
    assert len(seeds) == len(GRIPPER_RESET_RAD_BY_TASK), (
        "the per-task reset seeds collapsed to {} distinct values -- each subtask "
        "starts from a different grasp state".format(len(seeds))
    )
    print("  per-task reset seeds are distinct and in range      OK")

    # the boundary mapping must still put the endpoints at exactly -1 / +1
    assert abs(float(gripper_rad_to_command(np.array([GRIPPER_OPEN_RAD]))[0]) + 1.0) < 1e-9
    assert abs(float(gripper_rad_to_command(np.array([GRIPPER_CLOSED_RAD]))[0]) - 1.0) < 1e-9
    print("  open/closed map to boundary -1 / +1 exactly         OK")

    # the calibration must invert: the reset table stores MEASURED jaw, and the
    # ready move has to aim at the command that produces it, not at it directly.
    for prompt in GRIPPER_RESET_RAD_BY_TASK:
        want = reset_gripper_rad(prompt)
        got = command_to_measured_rad(measured_to_command_rad(want))
        reach_lo = command_to_measured_rad(np.full(2, GRIPPER_CLOSED_RAD))
        reach_hi = command_to_measured_rad(np.full(2, GRIPPER_OPEN_RAD))
        clipped = np.clip(want, reach_lo, reach_hi)
        assert np.abs(got - clipped).max() < 1e-9, (
            "{!r}: aiming at measured {} lands at {}, not the nearest reachable "
            "{}".format(prompt, want, got, clipped)
        )
    print("  gripper calibration inverts (ready aims at a command)  OK")

    # 4. every ready ramp must clear our own gate from the worst start we know
    #    of -- the pose the 2026-09-03 dry run actually began from.
    DRY_RUN_START = np.array([
        -0.164, 0.068, 0.093, 0.928, -0.061, -0.715, -0.023,
        0.071, 0.007, -0.053, 0.468, -0.016, -0.430, 0.066])
    assert set(READY_ARM_Q_BY_TASK) == set(TRAINED_PROMPTS), (
        "every trained prompt needs a start pose; missing {}".format(
            set(TRAINED_PROMPTS) - set(READY_ARM_Q_BY_TASK))
    )
    for prompt, target in READY_ARM_Q_BY_TASK.items():
        target = np.asarray(target)
        for start in (DRY_RUN_START, np.zeros(14), target):
            rows = Policy._ramp(start, target, 16, 0.35)
            grip = np.tile(reset_gripper_rad(prompt), (16, 1))
            validate_joint_chunk(rows, grip, start)     # raises if the gate trips
            step = float(np.abs(np.diff(rows, axis=0)).max()) if len(rows) > 1 else 0.0
            assert step <= MAX_ARM_STEP_RAD, "ramp step {:.3f} trips the gate".format(step)
    print("  ready ramps clear the joint gate for all 3 tasks    OK")

    print("\nALL CONVENTION CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
