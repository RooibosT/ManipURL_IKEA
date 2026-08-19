"""G1 forward kinematics and projected gravity, in the BCT training convention.

Two consumers with two different conventions live here, and mixing them up is
silent:

  * The 46-dim OBSERVATION state feeds the checkpoint. Its ``base_gravity`` and
    ``left_eef``/``right_eef`` blocks were *computed* by the training pipeline,
    not recorded, so deployment has to recompute them identically or those 15
    inputs are noise -- ``state_dropout`` makes the policy robust to a missing
    state, not to a wrong one. Convention, which must not drift:
    ``g1_body29_hand14.urdf``, **waist held at zero** so the pose is in the
    torso frame, the ``wrist_yaw`` link origin translated 0.05 m along its
    local +x, orientation as extrinsic-xyz Euler (URDF/ROS RPY).

  * The ACTION we publish on the decoupled lane is an end-effector pose for the
    organizer's IK adapter. That one wants the robot's actual pose, so it uses
    the **measured waist** and returns a rotation matrix we turn into a w-first
    quaternion without a detour through Euler angles.

``wrist_pose`` (state) and ``wrist_pose_matrix`` (action) therefore share a
chain walk but not a waist policy. Both are exercised by conformance.

Adapted from url_groot_deploy/g1_kinematics.py, which is the file the training
pipeline and the team's own Thor deployment already agree on.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

# Distance from the wrist_yaw link origin to the point the dataset calls the
# end effector, along the link's local +x.
DEFAULT_EE_OFFSET_M = 0.05

DEFAULT_URDF = (
    Path(__file__).resolve().parents[2] / "assets" / "g1" / "g1_body29_hand14.urdf"
)

# Contiguous parent->child chain from the pelvis out to each wrist. Verified
# against the URDF: waist_yaw's parent is `pelvis`, waist_pitch's child is
# `torso_link`, and each shoulder hangs off `torso_link`.
_CHAIN = {
    "left": (
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
    ),
    "right": (
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ),
}


def projected_gravity(quaternion_wxyz: np.ndarray) -> np.ndarray:
    """Gravity direction in the pelvis frame; yaw-invariant, unit length.

    Chosen over the raw quaternion during training because the recorded
    attitude is dominated by session heading (yaw std 109 deg vs roll 0.6),
    which is arbitrary. Upright returns about (0, 0, -1).
    """
    q = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm < 1e-6:
        raise ValueError("degenerate base quaternion: {}".format(q))
    w, x, y, z = q / norm
    # World -z rotated into the body frame: the third column of R transposed.
    return -np.array(
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
        dtype=np.float64,
    )


def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    a = axis / np.linalg.norm(axis)
    k = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


def matrix_to_rpy(rotation: np.ndarray) -> np.ndarray:
    """Extrinsic xyz Euler, the URDF/ROS RPY convention."""
    sy = float(np.hypot(rotation[0, 0], rotation[1, 0]))
    if sy > 1e-9:
        return np.array(
            [
                np.arctan2(rotation[2, 1], rotation[2, 2]),
                np.arctan2(-rotation[2, 0], sy),
                np.arctan2(rotation[1, 0], rotation[0, 0]),
            ]
        )
    return np.array(
        [
            np.arctan2(-rotation[1, 2], rotation[1, 1]),
            np.arctan2(-rotation[2, 0], sy),
            0.0,
        ]
    )


def matrix_to_quat_wxyz(rotation: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion, **w first**.

    The boundary's (T, 25) layout is (w, x, y, z) and a wrongly-ordered unit
    quaternion still passes its norm check, so this is one of the places the
    contract cannot catch a mistake for us. Shepperd's method: pick the branch
    with the largest denominator so the square root never loses precision.
    """
    r = np.asarray(rotation, dtype=np.float64)
    trace = r[0, 0] + r[1, 1] + r[2, 2]
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        q = np.array(
            [
                0.25 * s,
                (r[2, 1] - r[1, 2]) / s,
                (r[0, 2] - r[2, 0]) / s,
                (r[1, 0] - r[0, 1]) / s,
            ]
        )
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        q = np.array(
            [
                (r[2, 1] - r[1, 2]) / s,
                0.25 * s,
                (r[0, 1] + r[1, 0]) / s,
                (r[0, 2] + r[2, 0]) / s,
            ]
        )
    elif r[1, 1] > r[2, 2]:
        s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        q = np.array(
            [
                (r[0, 2] - r[2, 0]) / s,
                (r[0, 1] + r[1, 0]) / s,
                0.25 * s,
                (r[1, 2] + r[2, 1]) / s,
            ]
        )
    else:
        s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        q = np.array(
            [
                (r[1, 0] - r[0, 1]) / s,
                (r[0, 2] + r[2, 0]) / s,
                (r[1, 2] + r[2, 1]) / s,
                0.25 * s,
            ]
        )
    q = q / np.linalg.norm(q)
    # q and -q are the same rotation; fixing the sign keeps a chunk's rows from
    # flipping hemisphere mid-trajectory, which would make interpolation take
    # the long way round.
    return q if q[0] >= 0.0 else -q


@lru_cache(maxsize=4)
def _load_joints(urdf_path: str) -> Dict[str, tuple]:
    root = ET.parse(urdf_path).getroot()
    joints = {}
    for joint in root.findall("joint"):
        origin = joint.find("origin")
        xyz = np.zeros(3)
        rpy = np.zeros(3)
        if origin is not None:
            xyz = np.array([float(v) for v in origin.get("xyz", "0 0 0").split()])
            rpy = np.array([float(v) for v in origin.get("rpy", "0 0 0").split()])
        axis_element = joint.find("axis")
        axis = (
            np.array([float(v) for v in axis_element.get("xyz").split()])
            if axis_element is not None
            else np.array([0.0, 0.0, 1.0])
        )
        joints[str(joint.get("name"))] = (xyz, rpy, axis, joint.get("type"))
    return joints


class G1WristKinematics:
    """FK for both wrists off the pelvis, in the BCT dataset's convention."""

    def __init__(
        self,
        urdf_path: Optional[Path] = None,
        ee_offset_m: float = DEFAULT_EE_OFFSET_M,
    ):
        urdf = Path(urdf_path) if urdf_path is not None else DEFAULT_URDF
        if not urdf.is_file():
            raise FileNotFoundError("G1 URDF not found: {}".format(urdf))
        self.urdf_path = urdf
        self.ee_offset_m = float(ee_offset_m)
        self._joints = _load_joints(str(urdf))
        missing = [
            name
            for chain in _CHAIN.values()
            for name in chain
            if name not in self._joints
        ]
        if missing:
            raise RuntimeError("{} is missing joints: {}".format(urdf, missing))

    def wrist_pose_matrix(
        self,
        side: str,
        arm_q: np.ndarray,
        waist_q: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (position (3,), rotation (3, 3)) of the end-effector point.

        ``waist_q`` None means the waist is held at zero, i.e. the torso frame
        the training pipeline used. Pass the measured waist to get the true
        pelvis-frame pose, which is what an IK target wants.
        """
        try:
            chain = _CHAIN[side]
        except KeyError:
            raise ValueError("side must be 'left' or 'right', got {!r}".format(side))
        arm = np.asarray(arm_q, dtype=np.float64).reshape(-1)
        if arm.size != 7:
            raise ValueError("arm_q must have 7 entries, got {}".format(arm.size))
        if waist_q is None:
            waist = np.zeros(3)
        else:
            waist = np.asarray(waist_q, dtype=np.float64).reshape(-1)
            if waist.size != 3:
                raise ValueError("waist_q must have 3 entries, got {}".format(waist.size))

        rotation = np.eye(3)
        position = np.zeros(3)
        for name, angle in zip(chain, np.concatenate((waist, arm))):
            xyz, rpy, axis, joint_type = self._joints[name]
            position = position + rotation @ xyz
            rotation = rotation @ _rpy_to_matrix(*rpy)
            if joint_type in ("revolute", "continuous"):
                rotation = rotation @ _axis_angle(axis, float(angle))
        tip = position + rotation @ np.array([self.ee_offset_m, 0.0, 0.0])
        return tip, rotation

    def wrist_pose(
        self,
        side: str,
        arm_q: np.ndarray,
        waist_q: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """(6,) = xyz metres then extrinsic-xyz rpy radians -- the state block."""
        tip, rotation = self.wrist_pose_matrix(side, arm_q, waist_q)
        return np.concatenate((tip, matrix_to_rpy(rotation)))

    def both_wrist_poses(
        self,
        arm_q: np.ndarray,
        waist_q: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Left then right, from the 14-wide arm vector used everywhere else."""
        arm = np.asarray(arm_q, dtype=np.float64).reshape(-1)
        if arm.size != 14:
            raise ValueError("arm_q must have 14 entries, got {}".format(arm.size))
        return (
            self.wrist_pose("left", arm[:7], waist_q),
            self.wrist_pose("right", arm[7:], waist_q),
        )
