# ManipURL — IKEA IROS Assembly Challenge submission

Two containers for the Unitree G1 EDU: a policy server on the Jetson AGX Thor
and a policy client on the Orin NX. Lane **`decoupled`**.

**Run commands, mounts and env: [`INSTRUCTIONS.md`](INSTRUCTIONS.md).
Declarations: [`manifest.yaml`](manifest.yaml).**

---

## What the policy is

GR00T N1.7 finetuned on Dex1 teleop of an IKEA children's table build, from a
fixed standing pose, registered under the `new_embodiment` tag:

| | |
|---|---|
| Checkpoint | `URL-RFM/gr00t-n1.7-g1-dex1-ikea-3task-46d-30hz-h40` (public HF, apache-2.0, 12.6 GB fp32 on disk, bf16 at load) |
| Views | head (left eye) + both wrists, `(480,640,3)` RGB |
| State | 46 dims — legs 12, waist 3, arms 14, grippers 2, projected gravity 3, FK wrist poses 12. Convention confirmed against the dataset's own `meta/modality.json`: pelvis reference, waist excluded from FK, 0.05 m offset from `wrist_yaw_joint`, extrinsic-xyz Euler, and a URDF whose sha256 matches ours byte-for-byte |
| Action | horizon 40 at 30 Hz; arms RELATIVE (restored to absolute by the processor), grippers ABSOLUTE; no waist group |
| Denoising | 4 steps — every open-loop number for this checkpoint was measured there |
| Executed | first 16 rows (0.5 s), then replan — sized against the 185 ms measured on our Thor |
| Prompts | three subtask strings; see INSTRUCTIONS.md |

Open-loop accuracy on the held-out split, over rows 1-8 — the window async
deployment actually executes once latency compensation has eaten the lead:

| task | n | arm MAE | wrist error |
|---|---:|---:|---:|
| pick table leg | 180 | 1.23° | 8.99 mm |
| insert table leg to table base | 182 | 1.37° | 13.43 mm |
| rotate leg to tighten | 335 | 1.60° | 12.40 mm |
| **all** | 697 | **1.45°** | **11.78 mm** |

This is a *three*-task model on purpose: of every 46-dim checkpoint trained
here, it has the lowest wrist error on `pick table leg`, because it is not
paying for the two extra tasks the five-task variants carry. Two consequences
worth stating — `rotate table base` and `flip table` are **not** in its
vocabulary, and a 46-dim model is 15-20% behind its 60-dim counterpart, which
we cannot use because arm joint *velocity* is not on the wire.

## Why `decoupled`

The lane follows from the model, and this one emits joint targets rather than a
64-dim SONIC latent, so `decoupled` is the only lane it can speak. Both halves
refuse to start on `sonic` rather than be quietly wrong.

That has a consequence worth stating plainly: the decoupled lane has no joint
channel, so the server runs **forward kinematics** on the predicted arm joints
and publishes wrist poses, and the organizer's adapter runs inverse kinematics
to get back to joints. The G1 arm is 7-DoF and a pose is 6-DoF, so the elbow
swivel is not determined by what we send — their IK picks it. Nothing in the
contract can flag a disagreement; it shows up as an oddly-posed elbow. An
EE-space variant of the checkpoint (whose action space *is* the lane's) is the
fallback if the loss turns out to matter.

The 2026-09-03 dry run found the first real cost of that round trip, and it was
a frame convention rather than the DoF gap: their IK targets the bare
`wrist_yaw_link` origin, while we published it translated 0.05 m along local +x
— the offset our checkpoint was trained with. 5.00 cm on every pose, and IK
accept held to 15-20% for the whole run. **The published pose now carries a zero
offset and the model's state block keeps the 0.05 m**; those were one shared
value, which made the obvious one-flag fix a trap. See INSTRUCTIONS.md §6.2.

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

**The link carries JPEG, not raw frames.** Three 480x640x3 images are 2.76 MB
per observation, and on our own Thor↔Orin link that measured ~330 ms of round
trip — more than the 185 ms the policy itself takes, and enough that the client
discarded 25 of every 26 published rows as stale. The cause was our Thor's NIC
negotiating 100 Mb/s: 22.1 Mbit at 100 Mbit/s is 221 ms before framing. A
gigabit link would have hidden it, which is the point — at ~150 KB per
observation the encoding costs ~12 ms even at 100 Mb/s, so the chunk sizing
holds whatever the link negotiates. The frames arrived as JPEG from the
organizer's camera server in the first place, so this is a second generation of
the same artefacts. `--jpeg-quality 0` sends raw; the server accepts either.

Measured on the Orin rather than estimated: 7-10 ms to encode all three frames,
and 56 KB per observation — but that 56 KB is from `mock_orin`'s synthetic
gradient frames, which compress far better than a real scene. Re-measured at
q85: **54 KB for the mock pattern, 227 KB for textured content, 681 KB for
noise**, against 2.7 MB raw. Real camera frames of a table build belong in the
150-250 KB band, which is what `components/imagecodec.py` has always said.

The conclusion is unchanged and does not depend on which number you take — at
227 KB the link costs ~30 ms at 100 Mb/s against ~221 ms raw, so JPEG wins at
any speed. Only the headline figure was optimistic. (An earlier version of this
file said raw would be faster on gigabit; that was arithmetic on an encode cost
three times the real one.)

**No gripper state is published, and none can be.** The Dex1-1 rig means
`:5557` carries no hand vector — and `boundary/states.py` types the optional
hand slots as `(7,)`, the Dex3 shape, rejecting anything else, so a 1-DoF jaw
position has no schema-valid way through. Our 46-dim state has two gripper dims,
fed from our own last command.

We checked what that substitution costs against the training set
(`URL-RFM/IKEA_pickuptheleg`, 322 episodes / 149,437 frames), and it is far
less than an earlier version of this file claimed. We said the loop breaks the
moment the jaw closes on something — the jaw stalls at the object while our
command keeps going. **The data contains no such regime.** The teleoperator
commanded the grip *width* instead of slamming to zero: on `insert table leg to
table base` the right-hand command sits at ~2.17 rad and the jaw at ~2.35, the
leg's width, and never goes below 1.0 rad on that task at all.

What it actually costs, over 148,793 same-episode frame pairs at the best lag
(1 frame, correlation 0.9998): `|measured − command|` is 0.170 rad median on the
left hand and 0.060 on the right, worst case 0.30 of a 5.40 rad stroke. And it
is systematic, not random — the jaw cannot quite reach either end stop, so
measured is an affine function of command. Fitting that per hand takes the left
hand's median error to **0.008 rad**. That fit is now applied every step.

The training rig's calibration is the right one to use even though we deploy on
a different robot: we are not reproducing the competition jaw's true position,
we are reproducing the number the checkpoint was trained to read.

Two things the same analysis settled. The command scale endpoints are exactly
right — the training action channels span `[0.0000, 5.4000]`, so 0 closed and
5.40 open are the correct denominators. And the reset seed cannot be one
number: each subtask starts from a different grasp state, `pick table leg`
with both hands open (5.35 / 5.34), `insert table leg to table base` with the
right already holding the leg (5.36 / 2.35), `rotate leg to tighten` with the
left (0.17 / 5.34). It is now seeded per task from the prompt.

**The policy does not start until the arms are where the demonstrations
start.** The checkpoint's episodes begin from a recorded arm pose, per subtask,
and the dry run sat ~1.0 rad away from it for its whole 11.4 minutes — 45% of
the left arm's samples outside the training envelope. So the server ramps the
jaws, dwells a second, ramps the arms at 0.35 rad/s, waits for 0.10 rad of joint
error to hold for 0.3 s, and only then runs inference. About 4 s.

The ramp is slow because *we do not set the joint velocity*: we publish poses,
the organizer's adapter realises them, and their interpolator is what produced
the rad/s shutdowns on 2026-09-03. At 0.35 rad/s a chunk asks for 0.012 rad per
model row — seventeen times inside our own `MAX_ARM_STEP_RAD` gate — and ready
chunks take the same gate, FK and encoding path a real inference does.
`--no-ready-move` turns it off. On timeout it starts anyway and says so loudly:
an attempt that never begins scores zero for certain.

Only the upper body. Legs and waist have no channel in this lane, and the dry
run was outside the training envelope on both for 100% of the run — that one is
an ask, not a fix (INSTRUCTIONS.md §5).

**A missing camera means hold still, not crash.** The checkpoint has no
missing-view mode. A camera that drops after working reuses its last good frame
with a warning; one that has *never* published makes the server hold the
measured pose and say why every 5 seconds. Feeding a black frame to a policy
that has never seen one is worse than doing nothing.

## Checks

```bash
pip install -r requirements.txt
python conformance.py --lane decoupled       # log: docs/conformance_decoupled.log
scripts/check_boundary.sh                    # boundary/ unmodified
python scripts/check_conventions.py          # the two silent invariants (below)
scripts/dev_stack.sh                         # full loop, all three declared cameras
```

```bash
python scripts/contract_check.py --checkpoint /weights/<name>   # needs the weights
```

`conformance.py` runs the organizer's single-camera mock, so it validates the
action contract on the hold-still policy; `scripts/dev_stack.sh` covers the
camera path; `scripts/contract_check.py` is the one that loads the real
checkpoint, and it prints the peak GPU figure `manifest.yaml` wants.

`scripts/check_conventions.py` needs no weights and covers the two mistakes
nothing else can see, because both produce chunks that pass every contract
check and merely make the robot worse: the model's state offset drifting with
the wire offset, and an `--execute-rows` value whose resampled chunk overflows
the contract's 64-row limit (39+ model rows at 30→50 Hz, which the server now
clamps rather than discovering on the bench).

## Rehearsing on our own Thor and Orin

`192.168.100.1/.2` is the bench's addressing, not ours. Our pair sits on
`192.168.123.x` (Thor `.2`, Orin `.164`), so point the client at it:

```bash
# Orin container
-e PEVAL_THOR_HOST=192.168.123.2
```

Do not read a successful `ping 192.168.100.1` from the Orin as the link being
up — something else on the office network answers that address. Ping the Thor's
real address instead.

Two differences from the bench worth tracking:

| | our hardware | the bench |
|---|---|---|
| Orin | JetPack 5.1.1, L4T **R35.3.1** | R35.3.1 — exact match |
| Thor | JetPack **7.1-b112**, L4T R38.4, CUDA 13.0 | JetPack 7.2, L4T R39.2, CUDA 13.0 |

The Orin image pins an `l4t-*` tag and must match its host exactly, and ours
does. The Thor image is a plain NGC CUDA image rather than an `l4t-*` tag, so
it is not pinned to an L4T revision — and building against CUDA 13.0 on 7.1 to
run on 7.2 is the forward-compatible direction. Still worth confirming on the
bench rather than assuming.

## Status

- [x] `boundary/` verbatim, checksummed
- [x] Both Dockerfiles, entrypoints with architecture and weight preflight
- [x] `conformance.py --lane decoupled` passing **inside both built images on
      their own silicon** — `docs/conformance_decoupled_thor.log` (Thor, sm_110
      confirmed in the arch list) and `docs/conformance_decoupled_orin.log`
      (Orin NX, L4T R35.3.1, Python 3.8); also on x86 Python 3.8 and 3.10
- [x] Full task-space path — FK, quaternion ordering, gripper mapping — checked
      against `boundary`'s own validator
- [ ] **Re-run pending for the new checkpoint.** Real checkpoint loaded and
      inferred **on the Thor, in the built image**, through the server's own
      code path: contract matches, 185 ms per inference, 6.10 GiB peak
      (`docs/contract_check_thor.log`) — recorded against the previous
      checkpoint. The architecture is identical, so the latency and memory
      figures should carry; the contract keys changed (`cam_left_high`, no
      waist action) and have only been checked against the repo's declared
      config, not on hardware.
- [x] Both images build and run on their own silicon; the Thor image's torch
      carries sm_110 kernels
- [x] Both pushed to `ghcr.io/rooibost/` and their digests written into
      `manifest.yaml`
- [x] Peak GPU memory measured on the Thor: 6.10 GiB reserved, declared 8 GB
- [ ] **Re-run pending for the new checkpoint.** Closed loop: the real
      checkpoint on the Thor driving the Orin client, both from the pushed
      digest images — 185 ms round trip, 320-340 ms of motion published per
      cycle (`docs/integration_thor_orin.log`), again on the previous
      checkpoint
- [x] **Live dry run on the real G1, 2026-09-03** (organizer-run, four
      shakedown attempts, none scored). Zero contract violations, zero rejected,
      zero stale across all four; 2,222 chunks over 11.4 min of continuous
      publishing at 304 ms median with no gap over 500 ms. Every crash was
      organizer-side and is fixed on their end. **Ran the previous checkpoint** —
      see INSTRUCTIONS.md §9 for what carries over.
- [x] Tool-frame fix from that run: published pose at zero offset, model state
      pinned at 0.05 m, the two no longer shareable. Validated by replaying
      their `log.jsonl` — wire moves exactly 5.000 cm, state moves 0.000000 cm.
- [ ] **Gripper channel to re-measure on the current checkpoint.** The dry run
      found it never commands a close (INSTRUCTIONS.md §6.5); the cause is not
      separable from the tool-frame bug on that data.
