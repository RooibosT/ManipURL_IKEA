"""The BCT checkpoint: observation assembly, inference, and a hold-still stand-in.

The checkpoint is GR00T N1.7 finetuned on Dex1 whole-body teleop, registered
under the ``new_embodiment`` tag. Its contract, which deployment has to
reproduce exactly:

    video   cam_left_high, cam_left_wrist, cam_right_wrist  (480, 640, 3) uint8 RGB
    state   46 dims, in this key order:
              legs 12 | waist 3 | left_arm 7 | right_arm 7
              left_gripper 1 | right_gripper 1
              base_gravity 3 | left_eef 6 | right_eef 6
    action  left_arm 7 | right_arm 7 | left_gripper 1 | right_gripper 1
            horizon 40 at 30 Hz, arms RELATIVE (the processor restores absolute
            targets before returning), grippers ABSOLUTE
    4 denoising steps -- every open-loop number for this checkpoint was
    measured at 4, and lowering it changes the actions that get executed.

Everything the boundary hands us is in ``body_q`` (29,) and ``base_quat`` (4,).
The three derived blocks are computed here: ``base_gravity`` from the base
quaternion, the two ``*_eef`` blocks by FK on the arm joints in the same row
(waist zeroed -- see kinematics).

    HAND STATE IS NOT ON THE WIRE. The competition G1 carries Dex1-1 grippers,
    so ``:5557`` publishes no hand vector, and when it does publish one it is a
    7-DoF Dex3 vector in different units. The two gripper state dims are
    therefore fed from our own last command, which is what the gripper is
    tracking to anyway. The caller owns that value; see server.py.

This checkpoint does not predict the waist at all: it is in the state but not
in the action space. That is the same conclusion the earlier checkpoint's
discarded waist group reached, now made upstream -- in the source recordings
the waist is an output of the locomotion controller rather than the
teleoperator (|d waist| correlates +0.85 with |d legs| and +0.12 with
|d arms|), so there was only a balance controller to imitate, and on this
bench the organizer's whole-body controller owns those joints anyway.
``IGNORED_ACTION_KEYS`` is therefore empty; the published chunk still carries
the waist we *measured*, taken from the observation.
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

import numpy as np

from .kinematics import G1WristKinematics, projected_gravity

BODY_DOF = 29
LEGS = slice(0, 12)
WAIST = slice(12, 15)
LEFT_ARM = slice(15, 22)
RIGHT_ARM = slice(22, 29)

STATE_KEYS = (
    "legs",
    "waist",
    "left_arm",
    "right_arm",
    "left_gripper",
    "right_gripper",
    "base_gravity",
    "left_eef",
    "right_eef",
)
ACTION_KEYS = ("left_arm", "right_arm", "left_gripper", "right_gripper")
IGNORED_ACTION_KEYS = ()          # this checkpoint predicts no waist group

LANGUAGE_KEY = "annotation.human.task_description"

# The checkpoint is language-conditioned, and these THREE strings are the whole
# vocabulary it was trained on. Anything else is out of distribution, and the
# failure is quiet: the policy still produces a confident-looking chunk.
#
# This is a three-task model on purpose. The five-task variants of the same
# recipe also carry "rotate table base" and "flip table", and paying for those
# two costs the original three about 12% on `insert`; dropping them is what
# makes this the most accurate 46-dim checkpoint on `pick table leg`. Sending
# either of those two strings here is undefined -- they were never trained.
#
# Locomotion segments ("move to table", "move table base") were deliberately
# excluded from the finetune, so there is no prompt that makes it walk.
TRAINED_PROMPTS = (
    "pick table leg",
    "insert table leg to table base",
    "rotate leg to tighten",
)

# --- Recorded start pose, per subtask -------------------------------------
# The arm joints (14,) an episode of each task begins from, taken from the
# training set (URL-RFM/IKEA_pickuptheleg): the real recorded start closest in
# L2 to the per-joint median, over the episodes whose jaws start in that task's
# typical grip state. Filtering by grip is not cosmetic -- an arm holding a leg
# sits nowhere near an arm about to reach for one.
#
#   task                              eps  kept  worst joint p5-p95
#   pick table leg                    110   100        0.769 rad
#   insert table leg to table base    109    99        0.827 rad
#   rotate leg to tighten             103   100        1.125 rad
#
# THIS TABLE BELONGS TO THIS CHECKPOINT AND DOES NOT TRANSFER. The same poses
# derived from the five-task recording differ from these by up to 0.48 rad, so
# a pose borrowed from another dataset puts the policy out of distribution at
# the exact moment it is handed the arms -- and nothing downstream can tell.
#
# The spread is wide because the arms are NOT a fixed start: only the stance is
# (legs std 0.048 rad, waist 0.064; arms 0.33-0.44). Moving here does not put
# the arm on a precise target, it moves it from outside the distribution to the
# middle of it -- the 2026-09-03 dry run sat about 1.0 rad away for its whole
# 11.4 minutes.
READY_ARM_Q_BY_TASK = {
    "pick table leg": (
        -0.158371, +0.008859, +0.023980, +0.200029, -0.041933, +0.031051, -0.061335,
        +0.027624, -0.000865, -0.071210, +0.018935, +0.124504, +0.122575, +0.067399,
    ),
    "insert table leg to table base": (
        -0.409393, +0.370231, +0.078856, -0.389128, -0.660606, +0.443296, -0.693899,
        -0.250830, -0.256261, -0.077826, -0.342162, -0.012512, +0.466450, +0.252424,
    ),
    "rotate leg to tighten": (
        -0.957588, +0.360344, -0.444423, +0.654974, +0.127201, +0.114917, -1.245904,
        +0.053462, -0.418288, -0.269489, -0.276848, +0.379996, +0.262119, +0.508287,
    ),
}


# Deployment view -> the checkpoint's own video key. cam_left_high is cam_0 of
# the source recording, which is the LEFT eye of the head stereo pair.
DEFAULT_VIDEO_KEYS = {
    "head": "cam_left_high",
    "left_wrist": "cam_left_wrist",
    "right_wrist": "cam_right_wrist",
}


def _batched(values: np.ndarray) -> np.ndarray:
    """GR00T wants every state group as (batch, time, dim)."""
    return np.asarray(values, dtype=np.float32).reshape(1, 1, -1)


def split_body_q(body_q: np.ndarray) -> Dict[str, np.ndarray]:
    """Canonical G1 29-DoF vector -> the checkpoint's joint groups."""
    q = np.asarray(body_q, dtype=np.float64).reshape(-1)
    if q.size != BODY_DOF:
        raise ValueError("body_q must be ({},), got {}".format(BODY_DOF, q.size))
    return {
        "legs": q[LEGS],
        "waist": q[WAIST],
        "left_arm": q[LEFT_ARM],
        "right_arm": q[RIGHT_ARM],
        "arm": np.concatenate((q[LEFT_ARM], q[RIGHT_ARM])),
    }


def build_state(
    body_q: np.ndarray,
    base_quat: np.ndarray,
    gripper_q: np.ndarray,
    kinematics: G1WristKinematics,
) -> Dict[str, np.ndarray]:
    """The 46-dim BCT state, in its declared key order."""
    groups = split_body_q(body_q)
    grip = np.asarray(gripper_q, dtype=np.float64).reshape(-1)
    if grip.size != 2:
        raise ValueError("gripper_q must be (2,), got {}".format(grip.size))
    # Waist zeroed: the training pipeline expressed these in the torso frame.
    left_eef, right_eef = kinematics.both_wrist_poses(groups["arm"], None)
    return {
        "legs": _batched(groups["legs"]),
        "waist": _batched(groups["waist"]),
        "left_arm": _batched(groups["left_arm"]),
        "right_arm": _batched(groups["right_arm"]),
        "left_gripper": _batched(grip[:1]),
        "right_gripper": _batched(grip[1:]),
        "base_gravity": _batched(projected_gravity(base_quat)),
        "left_eef": _batched(left_eef),
        "right_eef": _batched(right_eef),
    }


def build_video(
    images: Dict[str, np.ndarray],
    view_to_boundary_key: Dict[str, str],
    video_keys: Optional[Dict[str, str]] = None,
) -> Dict[str, np.ndarray]:
    """Boundary camera keys -> the checkpoint's video keys, as (1, 1, H, W, 3)."""
    keys = video_keys or DEFAULT_VIDEO_KEYS
    out = {}
    for view, boundary_key in view_to_boundary_key.items():
        if boundary_key not in images:
            raise KeyError(
                "camera {!r} (view {!r}) is not in this observation; the client "
                "sends exactly the keys the server declared".format(boundary_key, view)
            )
        image = np.asarray(images[boundary_key])
        if image.shape != (480, 640, 3) or image.dtype != np.uint8:
            raise ValueError(
                "{} must be (480, 640, 3) uint8, got {} {}".format(
                    boundary_key, image.shape, image.dtype
                )
            )
        out[keys[view]] = image[None, None]
    return out


class HoldStillPolicy:
    """Repeats the measured arm pose. No checkpoint, no GPU, no intelligence.

    This is not a stub in the usual sense: its output takes the same joint
    gates, the same FK and the same task-space encoding a real inference does,
    so `conformance.py` exercises the real publish path. It is also what the
    server falls back to when no checkpoint is mounted, which is the state the
    bench will see if the weights fail to download.
    """

    horizon = 40
    row_hz = 30.0
    # The hold-still policy ignores the cameras, so a camera that never arrives
    # is a warning here rather than a reason to stop acting. Conformance runs
    # `mock_orin --no-wrists`, which publishes none of our declared keys.
    needs_images = False

    def __init__(self, delay_s: float = 0.0):
        self._delay_s = float(delay_s)
        self.steps = 0

    def infer(
        self,
        images: Dict[str, np.ndarray],
        body_q: np.ndarray,
        base_quat: np.ndarray,
        gripper_q: np.ndarray,
        prompt: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self._delay_s:
            time.sleep(self._delay_s)   # stand-in for real inference time
        self.steps += 1
        arm = split_body_q(body_q)["arm"]
        grip = np.asarray(gripper_q, dtype=np.float64).reshape(2)
        return (
            np.tile(arm, (self.horizon, 1)),
            np.tile(grip, (self.horizon, 1)),
        )

    def reset(self) -> None:
        self.steps = 0


class Gr00tBctPolicy:
    """The real checkpoint. Import and load are lazy so the module stays cheap."""

    # Three views, no missing-view mode. A camera that has never published is a
    # reason to hold still, not to feed the model a black frame.
    needs_images = True

    def __init__(
        self,
        checkpoint_path: str,
        kinematics: G1WristKinematics,
        embodiment_tag: str = "new_embodiment",
        device: str = "cuda",
        denoising_steps: int = 4,
        horizon: int = 40,
        row_hz: float = 30.0,
        video_keys: Optional[Dict[str, str]] = None,
    ):
        from gr00t.policy.gr00t_policy import Gr00tPolicy   # heavy; Thor only

        self.kinematics = kinematics
        self.horizon = int(horizon)
        self.row_hz = float(row_hz)
        self.video_keys = video_keys or DEFAULT_VIDEO_KEYS
        self.checkpoint_path = checkpoint_path
        self.steps = 0

        self.policy = Gr00tPolicy(
            embodiment_tag=embodiment_tag,
            model_path=checkpoint_path,
            device=device,
        )
        # The action head reads this at inference time. 4 is this checkpoint's
        # calibrated value -- see the module docstring.
        self.policy.model.action_head.num_inference_timesteps = int(denoising_steps)

    def infer(
        self,
        images: Dict[str, np.ndarray],
        body_q: np.ndarray,
        base_quat: np.ndarray,
        gripper_q: np.ndarray,
        prompt: str,
        view_to_boundary_key: Optional[Dict[str, str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if view_to_boundary_key is None:
            raise ValueError("view_to_boundary_key is required")
        observation = {
            "video": build_video(images, view_to_boundary_key, self.video_keys),
            "state": build_state(body_q, base_quat, gripper_q, self.kinematics),
            "language": {LANGUAGE_KEY: [[prompt]]},
        }
        action, _info = self.policy.get_action(observation)
        self.steps += 1
        return self._flatten(action)

    def _flatten(self, action: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Named action groups -> (T, 14) arm targets and (T, 2) gripper targets.

        The arms come back ABSOLUTE: the processor restores them from the
        relative prediction against the observation state before returning, so
        the caller must not add the arm state again.
        """
        parts = {}
        for key in ACTION_KEYS:
            if key not in action:
                raise KeyError("checkpoint action is missing {!r}".format(key))
            value = np.asarray(action[key], dtype=np.float64)
            if value.ndim == 3 and value.shape[0] == 1:
                value = value[0]
            if value.ndim != 2:
                raise ValueError(
                    "action {!r} has shape {}, expected (T, D)".format(key, value.shape)
                )
            parts[key] = value
        horizons = {key: len(value) for key, value in parts.items()}
        if len(set(horizons.values())) != 1:
            raise ValueError("action horizons disagree: {}".format(horizons))
        arm = np.concatenate((parts["left_arm"], parts["right_arm"]), axis=1)
        grip = np.concatenate((parts["left_gripper"], parts["right_gripper"]), axis=1)
        if arm.shape[1] != 14 or grip.shape[1] != 2:
            raise ValueError(
                "unexpected action widths: arm {}, gripper {}".format(
                    arm.shape, grip.shape
                )
            )
        return arm, grip

    def reset(self) -> None:
        self.steps = 0
        reset = getattr(self.policy, "reset", None)
        if callable(reset):
            reset()
