#!/usr/bin/env python3
"""Run the real checkpoint through the server's own code path and check it.

`conformance.py` proves we speak the boundary contract, but it runs the
hold-still policy -- it never loads the checkpoint. This closes that gap: it
loads the weights, checks that what the checkpoint declares matches what
components/policy/bct.py assumes, runs a real inference, and pushes the result
all the way through the joint gates, FK and `boundary`'s own validator.

    python scripts/contract_check.py --checkpoint /weights/<name>

Run it on the Thor after building the image. The peak-memory line it prints is
the number `manifest.yaml` wants: measure it on the machine that will run it,
because allocator behaviour and the CUDA version both move it.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from boundary.actions import DecoupledSink  # noqa: E402
from components.policy.bct import (  # noqa: E402
    ACTION_KEYS,
    IGNORED_ACTION_KEYS,
    STATE_KEYS,
    TRAINED_PROMPTS,
    Gr00tBctPolicy,
    build_state,
)
from components.policy.kinematics import G1WristKinematics  # noqa: E402
from components.policy.taskspace import (  # noqa: E402
    JointChunkError,
    TaskSpaceEncoder,
    validate_joint_chunk,
)

VIEW_MAP = {
    "head": "ego_view_left",
    "left_wrist": "left_wrist",
    "right_wrist": "right_wrist",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--denoising-steps", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=40)
    parser.add_argument("--execute-rows", type=int, default=8)
    parser.add_argument("--row-hz", type=float, default=50.0)
    parser.add_argument("--repeats", type=int, default=4,
                        help="Inferences to time. The first includes warmup.")
    args = parser.parse_args()

    import torch

    failures = []

    def check(label, ok, detail=""):
        print("  {:<52s} {}{}".format(label, "OK" if ok else "FAIL",
                                      " -- " + detail if detail else ""))
        if not ok:
            failures.append(label)

    kinematics = G1WristKinematics()
    body_q = 0.15 * np.sin(np.arange(29) * 0.2)
    base_quat = np.array([0.9995, 0.01, 0.02, 0.0])
    base_quat /= np.linalg.norm(base_quat)
    gripper_q = np.array([5.4, 5.4])

    print("\n1. observation state")
    state = build_state(body_q, base_quat, gripper_q, kinematics)
    dims = sum(v.shape[-1] for v in state.values())
    check("46 dims across 9 groups", dims == 46, "got {}".format(dims))

    print("\n2. loading the checkpoint")
    started = time.time()
    policy = Gr00tBctPolicy(
        checkpoint_path=args.checkpoint,
        kinematics=kinematics,
        device=args.device,
        denoising_steps=args.denoising_steps,
        horizon=args.horizon,
    )
    print("  loaded in {:.1f}s".format(time.time() - started))

    declared = policy.policy.get_modality_config()
    # The checkpoint knows its own contract; ours has to be the same one.
    check("state keys match ours",
          tuple(declared["state"].modality_keys) == STATE_KEYS,
          str(list(declared["state"].modality_keys)))
    check("video keys match ours",
          tuple(declared["video"].modality_keys) == tuple(policy.video_keys.values()),
          str(list(declared["video"].modality_keys)))
    check("action keys are ours plus the dropped waist",
          tuple(declared["action"].modality_keys) == IGNORED_ACTION_KEYS + ACTION_KEYS,
          str(list(declared["action"].modality_keys)))
    horizon = len(declared["action"].delta_indices)
    check("action horizon is {}".format(args.horizon), horizon == args.horizon,
          "got {}".format(horizon))

    print("\n3. inference")
    rng = np.random.default_rng(0)
    images = {key: rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
              for key in VIEW_MAP.values()}
    torch.cuda.reset_peak_memory_stats()
    latencies = []
    for _ in range(args.repeats):
        started = time.time()
        arm, grip = policy.infer(images, body_q, base_quat, gripper_q,
                                 TRAINED_PROMPTS[0], view_to_boundary_key=VIEW_MAP)
        latencies.append(time.time() - started)
    check("arm targets are ({}, 14)".format(args.horizon), arm.shape == (args.horizon, 14),
          str(arm.shape))
    check("gripper targets are ({}, 2)".format(args.horizon), grip.shape == (args.horizon, 2),
          str(grip.shape))
    print("  latency: warmup {:.0f}ms, then {}".format(
        latencies[0] * 1000, ["{:.0f}ms".format(x * 1000) for x in latencies[1:]]))
    print("  PEAK GPU: allocated {:.2f} GiB | reserved {:.2f} GiB   <- manifest number"
          .format(torch.cuda.max_memory_allocated() / 2 ** 30,
                  torch.cuda.max_memory_reserved() / 2 ** 30))

    # If the processor had not restored the relative arm prediction to absolute
    # targets, row 0 would sit near the origin instead of near the arm.
    measured_arm = np.concatenate((body_q[15:22], body_q[22:29]))
    drift = float(np.abs(arm[0] - measured_arm).max())
    check("row 0 is an absolute target, not a delta", drift < 0.45,
          "max |row0 - observed| = {:.4f} rad".format(drift))
    check("gripper targets inside the Dex1-1 range",
          bool(grip.min() >= -1e-6 and grip.max() <= 5.4 + 1e-6),
          "[{:.3f}, {:.3f}]".format(grip.min(), grip.max()))

    print("\n4. joint gates, FK, and the boundary's own validator")
    arm_x, grip_x = arm[:args.execute_rows], grip[:args.execute_rows]
    try:
        validate_joint_chunk(arm_x, grip_x, measured_arm)
        check("joint gates", True)
    except JointChunkError as exc:
        check("joint gates", False, str(exc))
    encoder = TaskSpaceEncoder(kinematics, output_row_hz=args.row_hz)
    chunk = encoder.encode(arm_x, grip_x, waist_q=body_q[12:15])
    try:
        DecoupledSink.validate_chunk(chunk)
        check("boundary DecoupledSink.validate_chunk on {}".format(chunk.shape), True)
    except Exception as exc:                        # noqa: BLE001 -- report it
        check("boundary DecoupledSink.validate_chunk", False, str(exc))
    print("  left  ee pos {} quat {}".format(np.round(chunk[0, 4:7], 4),
                                             np.round(chunk[0, 7:11], 4)))
    print("  right ee pos {} quat {}".format(np.round(chunk[0, 11:14], 4),
                                             np.round(chunk[0, 14:18], 4)))
    print("  hands {} | nav/height/torso {}".format(np.round(chunk[0, 0:4], 3),
                                                    chunk[0, 18:25]))

    print()
    if failures:
        print("FAILED: {}".format(", ".join(failures)))
        return 1
    print("ALL CONTRACT CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
