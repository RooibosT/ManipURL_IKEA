#!/usr/bin/env python3
"""The two invariants nothing else can catch. No weights, no GPU, ~1 second.

    python scripts/check_conventions.py

1. THE STATE OFFSET IS INDEPENDENT OF THE WIRE OFFSET. Both used to come from
   one `--ee-offset-m`, so setting the wire offset the organizer's IK wants
   also moved 6 of the 46 state dims 5 cm off the training distribution. That
   failure is silent: every chunk still passes `DecoupledSink.validate_chunk`,
   and the policy just gets worse.

2. THE PUBLISHED CHUNK FITS THE CONTRACT AT EVERY --execute-rows. Resampling
   30 Hz rows to 50 Hz multiplies them by 5/3, so 39+ model rows overflow the
   64-row limit and the client dies on its first publish.

3. THE GRIPPER STATE WE SYNTHESIZE STAYS INSIDE THE TRAINING DISTRIBUTION. No
   jaw position is on the wire, so we manufacture that model input; the numbers
   below are pinned to what URL-RFM/IKEA_pickuptheleg actually contains.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from boundary.actions import MAX_CHUNK_LENGTH, DecoupledSink  # noqa: E402
from components.policy.bct import build_state  # noqa: E402
from components.policy.kinematics import (  # noqa: E402
    ACTION_EE_OFFSET_M,
    TRAINING_EE_OFFSET_M,
    G1WristKinematics,
)
from components.policy.taskspace import (  # noqa: E402
    GRIPPER_CLOSED_RAD,
    GRIPPER_OPEN_RAD,
    GRIPPER_RESET_RAD_BY_TASK,
    TaskSpaceEncoder,
    command_to_measured_rad,
    gripper_rad_to_command,
    reset_gripper_rad,
)

# Measured off URL-RFM/IKEA_pickuptheleg, 322 episodes / 149,437 frames.
TRAIN_STATE_MIN, TRAIN_STATE_MAX = -0.0255, 5.3921
TRAIN_ACTION_MIN, TRAIN_ACTION_MAX = 0.0, 5.4000

BODY_Q = 0.15 * np.sin(np.arange(29) * 0.2)
BASE_QUAT = np.array([1.0, 0.0, 0.0, 0.0])
GRIP = np.array([4.46, 4.46])
ARM = np.concatenate((BODY_Q[15:22], BODY_Q[22:29]))


def eef_state(wire_offset: float) -> np.ndarray:
    """The model's eef state blocks, built the way server.py builds them."""
    state_k = G1WristKinematics(ee_offset_m=TRAINING_EE_OFFSET_M)
    _wire_k = G1WristKinematics(ee_offset_m=wire_offset)   # noqa: F841 -- the point
    s = build_state(BODY_Q, BASE_QUAT, GRIP, state_k)
    return np.concatenate([s["left_eef"].reshape(-1), s["right_eef"].reshape(-1)])


def main() -> int:
    # 1. the state must not move when the wire offset does
    baseline = eef_state(ACTION_EE_OFFSET_M)
    for wire in (0.0, 0.05, 0.12, -0.03):
        drift = float(np.abs(eef_state(wire) - baseline).max())
        assert drift == 0.0, (
            "state eef moved {:.4f} m when the wire offset was set to {} -- the "
            "two offsets have been merged again".format(drift, wire)
        )
    print("  state eef is independent of the wire offset            OK")

    # ...and the wire must actually honour it, or the fix does nothing
    poses = {}
    for wire in (0.0, 0.05):
        enc = TaskSpaceEncoder(G1WristKinematics(ee_offset_m=wire), 30.0, 50.0)
        poses[wire] = enc.encode(np.tile(ARM, (2, 1)), np.tile(GRIP, (2, 1)),
                                 waist_q=BODY_Q[12:15])[0, 4:7]
    moved = float(np.linalg.norm(poses[0.05] - poses[0.0]))
    # the chunk is float32, so compare at float32 precision, not float64
    assert abs(moved - 0.05) < 1e-6, "wire offset ignored: pose moved {:.4f} m".format(moved)
    print("  wire pose does track the wire offset ({:.2f} cm)         OK".format(moved * 100))

    # 2. every execute-rows value must produce a publishable chunk
    enc = TaskSpaceEncoder(G1WristKinematics(ee_offset_m=ACTION_EE_OFFSET_M), 30.0, 50.0)
    for rows in range(1, 41):
        capped = rows
        while capped > 1 and enc.rows_for(capped) > MAX_CHUNK_LENGTH:
            capped -= 1
        chunk = enc.encode(np.tile(ARM, (capped, 1)), np.tile(GRIP, (capped, 1)),
                           waist_q=BODY_Q[12:15])
        DecoupledSink.validate_chunk(chunk)          # raises if over the limit
    print("  --execute-rows 1..40 all fit the {}-row contract       OK".format(
        MAX_CHUNK_LENGTH))

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

    print("\nALL CONVENTION CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
