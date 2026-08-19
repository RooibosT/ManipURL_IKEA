# ManipURL — IKEA IROS Assembly Challenge submission

Two containers for the Unitree G1 EDU: a policy server on the Jetson AGX Thor
and a policy client on the Orin NX. Lane **`decoupled`**.

**Run commands, mounts and env: [`INSTRUCTIONS.md`](INSTRUCTIONS.md).
Declarations: [`manifest.yaml`](manifest.yaml).**

---

## What the policy is

GR00T N1.7 finetuned on Dex1 whole-body teleop of an IKEA children's table
build, registered under the `new_embodiment` tag:

| | |
|---|---|
| Checkpoint | `RooibosT/gr00t-n1.7-g1-dex1-bct-relarm-aug-30hz-h40` (private HF) |
| Views | head (left eye) + both wrists, `(480,640,3)` RGB |
| State | 46 dims — legs 12, waist 3, arms 14, grippers 2, projected gravity 3, FK wrist poses 12 |
| Action | horizon 40 at 30 Hz; arms RELATIVE (restored to absolute by the processor), waist and grippers ABSOLUTE |
| Denoising | 4 steps — every open-loop number for this checkpoint was measured there |
| Executed | first 8 rows (~0.27 s), then replan |
| Prompts | five subtask strings; see INSTRUCTIONS.md |

Arm MAE against held-out teleop, by how deep into the chunk you execute:
5 rows 1.20°, **8 rows 1.50°**, 16 rows 2.25°, 40 rows 3.98°.

## Why `decoupled`

The lane follows from the model, and this one emits joint targets rather than a
64-dim SONIC latent, so `decoupled` is the only lane it can speak. Both halves
refuse to start on `sonic` rather than be quietly wrong.

That has a consequence worth stating plainly: the decoupled lane has no joint
channel, so the server runs **forward kinematics** on the predicted arm joints
and publishes wrist poses, and the organizer's adapter runs inverse kinematics
to get back to joints. The G1 arm is 7-DoF and a pose is 6-DoF, so the elbow
swivel is not determined by what we send — their IK picks it. Nothing in the
contract can flag a disagreement; it shows up as an oddly-posed elbow. We are
measuring that round-trip offline against a stand-in IK, and an EE-space variant
of the checkpoint (whose action space *is* the lane's) is the fallback if the
loss turns out to matter.

## What runs where

```
JETSON AGX THOR  192.168.100.1              JETSON ORIN NX  192.168.100.2
┌──────────────────────────────┐            ┌────────────────────────────────┐
│ components/server.py         │            │ components/client.py           │
│   policy/bct.py    46-dim    │◄─ ws:8765 ─►│                                │
│   policy/kinematics.py  FK   │  msgpack   │   boundary.CameraStream  :5555 │◄─ cameras
│   policy/taskspace.py (T,25) │            │   boundary.StateStream   :5557 │◄─ state
│   GR00T N1.7, sm_110         │            │   boundary.ActionSink    :5556 │──► WBC
└──────────────────────────────┘            └────────────────────────────────┘
```

| Path | Owner | Notes |
|---|---|---|
| `boundary/` | organizer | Verbatim from the template, never edited. `scripts/check_boundary.sh` proves it. |
| `mocks/`, `conformance.py`, `requirements.txt` | organizer | Verbatim. |
| `components/server.py` | us | Observation → inference → joint gates → FK → `(T,25)`. |
| `components/client.py` | us | The template's loop, adapted: row rate from the server's metadata, re-query when the chunk runs out, decoupled only. |
| `components/transport.py` | us | The template's WebSocket link, plus a version fix (below). |
| `components/policy/` | us | The checkpoint wrapper, G1 FK, and the task-space encoder. |
| `assets/g1/` | Unitree (Apache-2.0) | `g1_body29_hand14.urdf`, for FK. Joint origins only — no meshes needed. |
| `docker/` | us | One Dockerfile per machine. |

## Three things worth knowing about the code

**The Orin caps us at websockets 13.** Keepalive reached the *sync* websockets
API in 14.0, and websockets 14 requires Python ≥ 3.9 — but JetPack 5.1.1 ships
Python 3.8. The template's `transport.py` passes `ping_interval=None`
unconditionally, which is a `TypeError` on 13.x. Ours passes the keepalive
arguments only where they exist; the intent is identical on both versions
(nothing ever pings). Verified by running the full conformance loop under
Python 3.8 / websockets 13.1 *and* under 3.10 / websockets 15.0.

**No gripper state is published.** The Dex1-1 rig means `:5557` carries no hand
vector, but our 46-dim state has two gripper dims. They are fed from our own
last command, which is what the gripper is tracking to anyway.

**A missing camera means hold still, not crash.** The checkpoint has no
missing-view mode. A camera that drops after working reuses its last good frame
with a warning; one that has *never* published makes the server hold the
measured pose and say why every 5 seconds. Feeding a black frame to a policy
that has never seen one is worse than doing nothing.

## Checks

```bash
pip install -r requirements.txt
python conformance.py --lane decoupled     # log: docs/conformance_decoupled.log
scripts/check_boundary.sh                  # boundary/ unmodified
scripts/dev_stack.sh                       # full loop, all three declared cameras
```

```bash
python scripts/contract_check.py --checkpoint /weights/<name>   # needs the weights
```

`conformance.py` runs the organizer's single-camera mock, so it validates the
action contract on the hold-still policy; `scripts/dev_stack.sh` covers the
camera path; `scripts/contract_check.py` is the one that loads the real
checkpoint, and it prints the peak GPU figure `manifest.yaml` wants.

## Status

- [x] `boundary/` verbatim, checksummed
- [x] Both Dockerfiles, entrypoints with architecture and weight preflight
- [x] `conformance.py --lane decoupled` passing (Python 3.8 and 3.10)
- [x] Full task-space path — FK, quaternion ordering, gripper mapping — checked
      against `boundary`'s own validator
- [x] Real checkpoint loaded and inferred through the server's own code path;
      the checkpoint's declared state, video and action keys match what
      `policy/bct.py` assumes (`scripts/contract_check.py`,
      `docs/contract_check.log`)
- [ ] Images built and pushed on Thor/Orin silicon; digests into `manifest.yaml`
      (`scripts/build_and_push.sh` does both)
- [ ] Peak GPU memory re-measured on the Thor (6.10 GiB on an A100; declared
      12 GB with headroom)
- [ ] Closed-loop run against the real checkpoint
