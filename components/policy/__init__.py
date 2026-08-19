"""Our policy: the BCT checkpoint and the joint -> task-space adapter.

    kinematics.py   G1 FK and projected gravity, in the training convention
    bct.py          observation assembly, GR00T inference, hold-still fallback
    taskspace.py    joint targets -> the decoupled lane's (T, 25) chunk

Nothing here is imported by the Orin client -- the client moves images and
actions, it does not infer. Only the Thor image needs these dependencies.
"""

from __future__ import annotations

from .bct import Gr00tBctPolicy, HoldStillPolicy, build_state, split_body_q
from .kinematics import G1WristKinematics, projected_gravity
from .taskspace import JointChunkError, TaskSpaceEncoder, validate_joint_chunk

__all__ = [
    "Gr00tBctPolicy",
    "HoldStillPolicy",
    "G1WristKinematics",
    "TaskSpaceEncoder",
    "JointChunkError",
    "validate_joint_chunk",
    "build_state",
    "split_body_q",
    "projected_gravity",
]
