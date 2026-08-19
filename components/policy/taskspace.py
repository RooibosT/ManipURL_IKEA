"""Joint-space policy output -> the decoupled lane's (T, 25) task-space chunk.

Our checkpoint predicts arm joint targets and gripper positions. The decoupled
lane has no joint channel: the 25 columns are hands, two end-effector poses,
and a locomotion command. So the server runs forward kinematics on the joint
targets and publishes the resulting wrist poses; the organizer's adapter runs
inverse kinematics and drives the whole-body controller.

That round trip costs something and it is worth being explicit about what: the
G1 arm is 7-DoF and a pose is 6-DoF, so the elbow swivel is not determined by
what we send. The organizer's IK picks it. Nothing in the contract can flag a
disagreement -- it shows up as an oddly-posed elbow, not an error.

Three things this module does that the boundary cannot check for us:

  * ORDERING. Quaternions go out (w, x, y, z). A (x, y, z, w) quaternion is
    still unit length and still passes ``DecoupledSink.validate_chunk``.
  * RESAMPLING. The checkpoint's rows are 1/30 s apart. The controller runs at
    50 Hz. We interpolate in joint space -- before FK -- because that is how
    the arm actually moves between two joint targets; interpolating poses and
    re-solving would invent a different path.
  * JOINT-DOMAIN GATES. Ported from the team's own Thor deployment: an
    undecoded relative chunk (row 0 near the origin instead of near the arm)
    and per-tick jumps larger than anything in the training data are caught
    here, in the space the model actually predicts in, before FK hides them.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .kinematics import G1WristKinematics, matrix_to_quat_wxyz

# --- Dex1-1 parallel gripper ----------------------------------------------
# Motor radians: 0.0 fully closed, 5.40 fully open (0.6 rad/cm over a 9 cm jaw
# stroke). The boundary's convention is the opposite sign and normalized:
# -1 open, +1 closed.
GRIPPER_CLOSED_RAD = 0.0
GRIPPER_OPEN_RAD = 5.40

# --- Joint-domain gates, measured on the training set ----------------------
# Absolute targets predict the teleop command, which sits off the measured
# state even when perfect (|action - state| p99 0.153 rad, max 0.245).
MAX_FIRST_ARM_ERROR_RAD = 0.45
# Training per-tick |delta action| tops out at 0.116 rad.
MAX_ARM_STEP_RAD = 0.20
# |d gripper| per 30 Hz step is p99.9 0.558, max 0.700 on the training actions.
MAX_GRIPPER_STEP_RAD = 0.80

TASKSPACE_DIM = 25
ARM_DOF = 7
DUAL_ARM_DOF = 14


class JointChunkError(ValueError):
    """A joint chunk that fails a training-range gate. Never reaches FK."""


def gripper_rad_to_command(q_rad: np.ndarray) -> np.ndarray:
    """Dex1-1 motor radians -> the boundary's -1 open / +1 closed scale."""
    q = np.clip(np.asarray(q_rad, dtype=np.float64), GRIPPER_CLOSED_RAD, GRIPPER_OPEN_RAD)
    return 1.0 - 2.0 * (q - GRIPPER_CLOSED_RAD) / (GRIPPER_OPEN_RAD - GRIPPER_CLOSED_RAD)


def validate_joint_chunk(
    arm_targets: np.ndarray,
    gripper_targets: np.ndarray,
    reference_arm_q: np.ndarray,
    max_first_arm_error_rad: float = MAX_FIRST_ARM_ERROR_RAD,
    max_arm_step_rad: float = MAX_ARM_STEP_RAD,
    max_gripper_step_rad: float = MAX_GRIPPER_STEP_RAD,
) -> None:
    """Check a chunk in the space the model predicts in. Raises on violation."""
    arm = np.asarray(arm_targets, dtype=np.float64)
    grip = np.asarray(gripper_targets, dtype=np.float64)
    reference = np.asarray(reference_arm_q, dtype=np.float64).reshape(-1)

    if arm.ndim != 2 or arm.shape[1] != DUAL_ARM_DOF:
        raise JointChunkError(
            "arm targets must be (T, {}), got {}".format(DUAL_ARM_DOF, arm.shape)
        )
    if grip.ndim != 2 or grip.shape[1] != 2:
        raise JointChunkError("gripper targets must be (T, 2), got {}".format(grip.shape))
    if len(arm) != len(grip):
        raise JointChunkError(
            "arm and gripper horizons disagree: {} vs {}".format(len(arm), len(grip))
        )
    if len(arm) == 0:
        raise JointChunkError("joint chunk is empty")
    if not np.isfinite(arm).all() or not np.isfinite(grip).all():
        raise JointChunkError("joint chunk contains non-finite values")
    if reference.size != DUAL_ARM_DOF:
        raise JointChunkError(
            "reference_arm_q must be ({},), got {}".format(DUAL_ARM_DOF, reference.size)
        )

    first_row = arm[0]
    joint = int(np.argmax(np.abs(first_row - reference)))
    first_error = float(abs(first_row[joint] - reference[joint]))
    if first_error > max_first_arm_error_rad:
        detail = ""
        if first_error > float(np.max(np.abs(first_row))):
            # Row 0 sits nearer the origin than the arm does: undecoded
            # relative deltas. Commanding them collapses the arm toward zero.
            detail = " -- chunk looks like relative deltas, not absolute targets"
        raise JointChunkError(
            "first arm target is {:.3f} rad from the observed arm (limit {:.3f}) "
            "at joint {}{}".format(first_error, max_first_arm_error_rad, joint, detail)
        )

    if len(arm) > 1:
        steps = np.abs(np.diff(arm, axis=0))
        row, joint = np.unravel_index(int(np.argmax(steps)), steps.shape)
        worst = float(steps[row, joint])
        if worst > max_arm_step_rad:
            raise JointChunkError(
                "arm step {:.3f} rad at row {} joint {} exceeds {:.3f}".format(
                    worst, int(row) + 1, int(joint), max_arm_step_rad
                )
            )
        gsteps = np.abs(np.diff(grip, axis=0))
        worst_g = float(np.max(gsteps))
        if worst_g > max_gripper_step_rad:
            raise JointChunkError(
                "gripper step {:.3f} rad exceeds {:.3f}".format(
                    worst_g, max_gripper_step_rad
                )
            )


def resample_rows(values: np.ndarray, source_hz: float, target_hz: float) -> np.ndarray:
    """Linearly resample (T, D) rows from one row rate to another.

    Row i of the input is at t = i / source_hz. The output covers the same
    span, starting at t = 0, at 1 / target_hz spacing. A single input row is
    returned unchanged -- there is nothing to interpolate between.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("expected (T, D) rows, got {}".format(arr.shape))
    if len(arr) < 2 or abs(source_hz - target_hz) < 1e-9:
        return arr.copy()
    duration = (len(arr) - 1) / float(source_hz)
    n_out = int(np.floor(duration * float(target_hz))) + 1
    src_t = np.arange(len(arr), dtype=np.float64) / float(source_hz)
    dst_t = np.arange(n_out, dtype=np.float64) / float(target_hz)
    return np.stack(
        [np.interp(dst_t, src_t, arr[:, d]) for d in range(arr.shape[1])], axis=1
    )


class TaskSpaceEncoder:
    """Turns joint-space chunks into the decoupled lane's (T, 25) rows."""

    def __init__(
        self,
        kinematics: G1WristKinematics,
        model_row_hz: float = 30.0,
        output_row_hz: float = 50.0,
        use_measured_waist: bool = True,
    ):
        self.kinematics = kinematics
        self.model_row_hz = float(model_row_hz)
        self.output_row_hz = float(output_row_hz)
        # True  -> FK with the measured waist: the wrist's true pelvis-frame
        #          pose, which is what an IK target should be.
        # False -> waist held at zero, i.e. the torso frame the checkpoint's
        #          own state block uses.
        # The organizer has not documented which frame the decoupled adapter
        # expects; see INSTRUCTIONS.md, "Open questions".
        self.use_measured_waist = bool(use_measured_waist)

    def encode(
        self,
        arm_targets: np.ndarray,
        gripper_targets: np.ndarray,
        waist_q: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """(T_model, 14) + (T_model, 2) -> (T_out, 25) float32.

        ``waist_q`` is the measured waist at observation time, held constant
        across the chunk: the checkpoint does not predict the waist and the
        organizer's controller owns it, so we have no better estimate of where
        it will be.
        """
        arm = np.asarray(arm_targets, dtype=np.float64)
        grip = np.asarray(gripper_targets, dtype=np.float64)
        joint_rows = np.concatenate((arm, grip), axis=1)
        rows = resample_rows(joint_rows, self.model_row_hz, self.output_row_hz)

        waist = None
        if self.use_measured_waist and waist_q is not None:
            waist = np.asarray(waist_q, dtype=np.float64).reshape(-1)

        out = np.zeros((len(rows), TASKSPACE_DIM), dtype=np.float32)
        for i, row in enumerate(rows):
            left_pos, left_rot = self.kinematics.wrist_pose_matrix(
                "left", row[:ARM_DOF], waist
            )
            right_pos, right_rot = self.kinematics.wrist_pose_matrix(
                "right", row[ARM_DOF:DUAL_ARM_DOF], waist
            )
            hands = gripper_rad_to_command(row[DUAL_ARM_DOF:DUAL_ARM_DOF + 2])
            out[i, 0:2] = hands[0]      # left gripper, both finger joints
            out[i, 2:4] = hands[1]      # right gripper
            out[i, 4:7] = left_pos
            out[i, 7:11] = matrix_to_quat_wxyz(left_rot)
            out[i, 11:14] = right_pos
            out[i, 14:18] = matrix_to_quat_wxyz(right_rot)
            # [18:21] navigate_cmd, [21] base_height_cmd, [22:25] torso rpy all
            # stay zero. The checkpoint predicts no locomotion command, and its
            # `waist` output only imitates the balance controller (|d waist|
            # correlates +0.85 with |d legs|, +0.12 with |d arms|), so feeding
            # it to torso_rpy would fight the controller that owns balance.
        return out

    def rows_for(self, model_rows: int) -> int:
        """How many output rows ``model_rows`` model rows become."""
        if model_rows < 2:
            return max(model_rows, 0)
        duration = (model_rows - 1) / self.model_row_hz
        return int(np.floor(duration * self.output_row_hz)) + 1


def hold_still_chunk(
    encoder: TaskSpaceEncoder,
    arm_q: np.ndarray,
    gripper_q: np.ndarray,
    waist_q: Optional[np.ndarray],
    model_rows: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Repeat the measured pose: a valid chunk that commands no motion.

    Returned as joint targets so it takes the same validation and FK path a
    real inference does -- that is the point of it, and what makes conformance
    exercise the encoder rather than a hard-coded array of zeros.
    """
    arm = np.asarray(arm_q, dtype=np.float64).reshape(-1)
    grip = np.asarray(gripper_q, dtype=np.float64).reshape(-1)
    return (
        np.tile(arm, (model_rows, 1)),
        np.tile(grip, (model_rows, 1)),
    )
