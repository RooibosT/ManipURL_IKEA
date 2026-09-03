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

from typing import Optional

import numpy as np

from .kinematics import G1WristKinematics, matrix_to_quat_wxyz

# --- Dex1-1 parallel gripper ----------------------------------------------
# Motor radians: 0.0 fully closed, 5.40 fully open. CONFIRMED against the
# training set (URL-RFM/IKEA_pickuptheleg, 322 episodes / 149,437 frames): the
# action channels span exactly [0.0000, 5.4000], so these are the right
# endpoints for the command scale. The boundary's convention is the opposite
# sign and normalized: -1 open, +1 closed.
GRIPPER_CLOSED_RAD = 0.0
GRIPPER_OPEN_RAD = 5.40

# --- Synthesizing the gripper STATE the model expects to read --------------
# No jaw position is on the wire (see INSTRUCTIONS.md §6.5), so the two gripper
# state dims are fed from our own last command. The model was trained on the
# MEASURED jaw, and on the training rig the measurement is a systematic affine
# function of the command, not a copy of it: the jaw cannot quite reach either
# end stop.
#
# Fitted on 148,793 same-episode frame pairs at the best lag (1 frame,
# corr 0.9998), per hand because the two jaws are calibrated differently:
#
#            measured ~= SCALE * command + OFFSET      |err| median  -> after
#   left     0.9602 * cmd + 0.1770                          0.1700     0.0083
#   right    0.9322 * cmd + 0.3122                          0.0603     0.0118
#
# A 20x error reduction on the left hand for two constants. Worth stating why
# the TRAINING rig's calibration is the right one to use even though we deploy
# on a different robot: we are not trying to reproduce the competition jaw's
# true position, we are trying to reproduce the number the checkpoint was
# trained to read. That is the training rig's measurement, by definition.
#
# ponytail: fitted on the training rig; if the competition Dex1-1 is
# calibrated differently these drift, but only by the difference between two
# jaw calibrations (~0.2 rad at worst). Re-fit if jaw position ever reaches us.
GRIPPER_CMD_TO_MEASURED = {
    "left": (0.9602, 0.1770),
    "right": (0.9322, 0.3122),
}

# --- Assumed jaw at reset, per task ---------------------------------------
# Median measured jaw at frame 0 of every episode, by task, from the training
# set. It is per-task AND per-hand, so one scalar cannot express it: each
# subtask starts from a different grasp state, because the episodes are
# segmented per subtask and two of the three begin with the leg already held.
#
#   task                              n    left   right
#   pick table leg                  110    5.35    5.34    both open
#   insert table leg to table base  109    5.36    2.35    right holds the leg
#   rotate leg to tighten           103    0.17    5.34    left holds the leg
#
# Seeding "both hands open" for `rotate leg to tighten` would tell the policy
# the left hand is open when the training data says it is closed on the leg --
# wrong by 5.2 rad, the entire stroke, on the inference that decides the
# attempt's opening move.
GRIPPER_RESET_RAD_BY_TASK = {
    "pick table leg": (5.35, 5.34),
    "insert table leg to table base": (5.36, 2.35),
    "rotate leg to tighten": (0.17, 5.34),
}
# Fallback for a prompt we do not recognise: both hands open, the `pick` start.
GRIPPER_RESET_RAD = 5.35

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


def command_to_measured_rad(command_rad: np.ndarray) -> np.ndarray:
    """Our (left, right) gripper command -> the measured jaw the model expects.

    See GRIPPER_CMD_TO_MEASURED. Input and output are both Dex1-1 motor
    radians; this is a calibration, not a unit change.
    """
    cmd = np.asarray(command_rad, dtype=np.float64).reshape(-1)
    if cmd.size != 2:
        raise ValueError("command_rad must be (2,), got {}".format(cmd.size))
    out = np.empty(2, dtype=np.float64)
    for i, side in enumerate(("left", "right")):
        scale, offset = GRIPPER_CMD_TO_MEASURED[side]
        out[i] = scale * cmd[i] + offset
    return np.clip(out, GRIPPER_CLOSED_RAD, GRIPPER_OPEN_RAD)


def measured_to_command_rad(measured_rad: np.ndarray) -> np.ndarray:
    """Inverse of :func:`command_to_measured_rad`.

    The dataset records the MEASURED jaw, so a value taken from it -- the
    per-task start in GRIPPER_RESET_RAD_BY_TASK, say -- is where we want the
    jaw to end up, not what to ask for. Commanding it raw lands 0.15 rad off on
    the `holding a leg` value, which is the same size of error the calibration
    exists to remove.
    """
    meas = np.asarray(measured_rad, dtype=np.float64).reshape(-1)
    if meas.size != 2:
        raise ValueError("measured_rad must be (2,), got {}".format(meas.size))
    out = np.empty(2, dtype=np.float64)
    for i, side in enumerate(("left", "right")):
        scale, offset = GRIPPER_CMD_TO_MEASURED[side]
        out[i] = (meas[i] - offset) / scale
    return np.clip(out, GRIPPER_CLOSED_RAD, GRIPPER_OPEN_RAD)


def reset_gripper_rad(prompt: str) -> np.ndarray:
    """Assumed (left, right) measured jaw at the start of ``prompt``'s task."""
    pair = GRIPPER_RESET_RAD_BY_TASK.get(
        prompt, (GRIPPER_RESET_RAD, GRIPPER_RESET_RAD)
    )
    return np.asarray(pair, dtype=np.float64)


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
        # False -> waist locked at zero, the convention the checkpoint's own
        #          state block uses. Still pelvis-origin: torso_link is a
        #          further 4.42 cm out, so this is not "the torso frame".
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
