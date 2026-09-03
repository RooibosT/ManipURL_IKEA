# How to run our two containers

Team **ManipURL** · lane **`decoupled`** · Unitree G1 EDU with Dex1-1 grippers.

Everything below is the command as we expect you to type it. `manifest.yaml`
carries the same values in machine-readable form.

## Where each onboarding item lives

| # | Item | Where |
|---|---|---|
| 1 | Repo with an unmodified `boundary/` and a Dockerfile per container | this repo; `docker/Dockerfile.thor`, `docker/Dockerfile.orin`; verify with `scripts/check_boundary.sh` |
| 2 | Thor image, by digest | `manifest.yaml` → `images.thor.digest` |
| 3 | Orin image, by digest | `manifest.yaml` → `images.orin.digest` |
| 4 | Model weights, not baked in | `manifest.yaml` → `weights`; download in §0 below — the repo is public, nothing to request |
| 5 | Manifest | `manifest.yaml` |
| 6 | Conformance log | `docs/conformance_decoupled_orin.log` (on our Orin, inside the built image); `docs/conformance_decoupled.log` and `..._py38.log` are the same check off-hardware |
| 7 | Run command per container | §1 and §2 below |

Neither container needs internet once the images and weights are staged; see §3.

---

## 0 · Before the first run: the weights

The checkpoint is **not** baked into the image — mount it.

```bash
# on the Thor, once
sudo mkdir -p /opt/weights && sudo chown "$USER" /opt/weights
pip install -U "huggingface_hub[cli]"
hf download URL-RFM/gr00t-n1.7-g1-dex1-ikea-3task-46d-30hz-h40 \
    --local-dir /opt/weights/gr00t-n1.7-g1-dex1-ikea-3task-46d-30hz-h40
```

<https://huggingface.co/URL-RFM/gr00t-n1.7-g1-dex1-ikea-3task-46d-30hz-h40>

About **12.6 GB** to download (the checkpoint is stored fp32 and cast to
bfloat16 at load, so it occupies roughly 6 GB on the GPU).

> **Nothing is needed from you for this repo.** It is public, ungated and
> apache-2.0 — no account, no token, no access request, no license
> click-through. An earlier version of this file asked you for Hugging Face
> usernames so we could grant access to a private repo; that ask is withdrawn.
> The backbone below is a different story.

### And one more download: the VLM backbone

Every GR00T checkpoint — ours and NVIDIA's base model alike — loads its vision-
language backbone `nvidia/Cosmos-Reason2-2B` from Hugging Face when the model is
constructed. It is not inside our checkpoint and not baked into the image, and
**it is gated**. Miss it and the server dies 30 seconds into loading with a 401
about a repo you never asked for.

The gate is automatic approval, not a request queue: open
<https://huggingface.co/nvidia/Cosmos-Reason2-2B>, accept the terms, done.

Then pre-stage it next to the weights, so the container needs no network at run
time:

```bash
hf auth whoami                       # confirm this is the account that accepted
export HF_TOKEN="$(cat ~/.cache/huggingface/token)"
HF_HUB_CACHE=/opt/weights/hf-cache/hub hf download nvidia/Cosmos-Reason2-2B  # ~4.9 GB
```

`HF_HUB_CACHE`, not `HF_HOME`: `HF_HOME` moves the token as well as the cache,
so setting it here makes the download unauthenticated and the gate rejects it
with "Access denied. This repository requires approval" even when your account
has access. Inside the container `HF_HOME` is the right variable, because by
then there is no token to find and nothing to fetch.

Run with `-e HF_HOME=/weights/hf-cache` (already in the command below). If
you would rather let the container fetch it, pass `-e HF_TOKEN=<your token>`
instead and give it network access. The entrypoint checks for one or the other
before loading anything and says which is missing.

Both images also come from `nvcr.io`, which needs an NGC login even though the
base images are public: `docker login nvcr.io` with username `$oauthtoken` and
an NGC API key.

Total to stage on the Thor: **12.6 GB** checkpoint + **4.9 GB** backbone.

---

## 1 · Thor — policy server

```bash
docker run --rm -it \
    --runtime nvidia \
    --network host \
    --ipc host \
    -v /opt/weights:/weights:ro \
    -e PEVAL_CHECKPOINT=/weights/gr00t-n1.7-g1-dex1-ikea-3task-46d-30hz-h40 \
    -e HF_HOME=/weights/hf-cache \
    -e HF_HUB_OFFLINE=1 \
    <thor-image>@<thor-digest> \
    python components/server.py --lane decoupled --port 8765
```

| Flag | Why it is there |
|---|---|
| `--runtime nvidia` | Required. The policy runs on the GPU. |
| `--network host` | So the server is reachable at `192.168.100.1:8765` over the direct link without port mapping. `-p 8765:8765` works too if you prefer. |
| `--ipc host` | PyTorch dataloader/shared-memory headroom. Not strictly required at batch 1; drop it if it conflicts with your setup. |
| `-v /opt/weights:/weights:ro` | The checkpoint. Read-only is enough. |
| `-e PEVAL_CHECKPOINT=` | Which directory under the mount to load. Already the image default; override only if you put the weights elsewhere. |
| `-e HF_HOME=/weights/hf-cache` | Where the pre-staged `nvidia/Cosmos-Reason2-2B` backbone lives. Drop it and pass `-e HF_TOKEN` instead if you would rather fetch it at run time. |
| `-e HF_HUB_OFFLINE=1` | With the backbone pre-staged there is nothing left to fetch, so this keeps model loading off the network entirely. Drop it if you are using `HF_TOKEN`. |

The entrypoint refuses to start if `torch` has no `sm_110` kernels or if the
checkpoint directory is missing, because both of those otherwise fail later and
less clearly — a wrong-architecture torch imports fine and dies at the first
kernel launch.

**Smoke test without weights.** `-e PEVAL_CHECKPOINT=` (empty) starts a
hold-still policy that repeats the measured arm pose through the same FK and
encoding path. The robot will not move, and the whole publish chain is
exercised. Good for a first bring-up before the download finishes.

Expect roughly **24 GB** of the Thor's 128 GB unified pool; see `manifest.yaml`
for the breakdown.

---

## 2 · Orin — policy client

```bash
docker run --rm -it \
    --runtime nvidia \
    --network host \
    -e PEVAL_THOR_HOST=192.168.100.1 \
    <orin-image>@<orin-digest> \
    python3 components/client.py --lane decoupled --thor 192.168.100.1 \
        --prompt "pick table leg"
```

| Flag | Why it is there |
|---|---|
| `--network host` | **Required.** We *bind* `:5556` for your controller to dial into, and we subscribe to `:5555` / `:5557` at `127.0.0.1`. Bridge networking breaks both halves. |
| `--runtime nvidia` | Not actually needed — the client does no inference and nothing in the image is compiled for the GPU. Included only because the base image is an `l4t-*` tag and you may want the mounts consistent. Drop it freely. |
| `--prompt` | See below. This is a live model input, not a label. |

The client does not need a GPU, a checkpoint, or `pinocchio`. It reads the two
input endpoints, ships the observation to the Thor, and publishes what comes
back.

### The prompt matters

The checkpoint is language-conditioned and was trained on exactly three subtask
strings:

```
pick table leg
insert table leg to table base
rotate leg to tighten
```

Anything else is out of distribution, and the failure is quiet — the policy
still returns a confident-looking chunk. The server logs a warning once per
unseen prompt. Restart the client with a different `--prompt` to switch
subtask; nothing else needs restarting.

**`rotate table base` and `flip table` are not in this model's vocabulary.**
We train five-task variants too, but carrying those two costs the other three
about 12% on `insert`, so the checkpoint we are submitting drops them. If the
run order needs either subtask, tell us and we will swap in the five-task
checkpoint — same contract, same image, only the mounted directory changes.

There is **no prompt that makes the robot walk.** Locomotion segments were
excluded from the finetune on purpose, so `navigate_cmd`, `base_height_cmd` and
`torso_rpy` all go out as zeros and your controller keeps the lower body.

---

## 3 · Neither container needs the network at run time

Everything is staged beforehand, so both containers run on an air-gapped bench:

| Needs network | Once, beforehand |
|---|---|
| `docker pull` both images by digest | yes |
| `hf download` the checkpoint (12.6 GB) | yes |
| `hf download` the backbone (4.9 GB) | yes |

| Needs no network | At run time |
|---|---|
| Thor server | `HF_HUB_OFFLINE=1` with the backbone pre-staged; loads entirely from the mount |
| Orin client | only local sockets: `:5555`/`:5557` on the Orin itself, `:5556` bound for your controller, and the Thor over the direct link |

We verified the Thor side of this: the run in `docs/contract_check_thor.log`
loaded the model from a **read-only** mount with `HF_HUB_OFFLINE=1`.

## 4 · Order of operations

1. Your camera and state servers up on the Orin.
2. Thor container. Wait for `policy server listening on ws://0.0.0.0:8765`.
3. Orin container. It waits for both endpoints, then dials the Thor and prints
   the metadata it got back.
4. Your controller dials into `:5556`.

Starting the client before the server is fine — `PolicyLink` retries for 60 s.

**Stopping the client does not stop the robot.** Your controller keeps replaying
its last command. Only your e-stop brings it to a safe state.

---

## 5 · What we need from you

**Read §9 first if you are scheduling the re-run.** The four dry-run attempts
ran the checkpoint we certified at onboarding, which is no longer the one we
ship. Which of your findings carry over depends on that, and one of them
does not.

**The upper body moves for ~4 seconds before the policy starts.** New since the
dry run, and you should know it is coming. Our checkpoint's demonstrations begin
from a recorded arm pose, and the 2026-09-03 run sat about **1.0 rad away from
it for the whole 11.4 minutes** — 45% of the left arm's samples were outside the
training envelope entirely. So the server now ramps the arms and jaws there
first, and only then hands over to GR00T.

What you will see at the start of every attempt: jaws commanded to the task's
start position, one second of dwell, then a **0.35 rad/s** arm ramp of 0.75-1.32
rad (13-26 cm of end-effector travel, depending on the subtask), then the policy.
It converges in about 4 s and gives up at 15 s.

We are flagging it because *we do not set the joint velocity* — we publish poses
and your adapter realises them, and your interpolator is what produced the
-7.15 and -8.99 rad/s shutdowns. Our ramp asks for 0.012 rad per model row,
seventeen times inside our own step gate, and the chunks take the same
validation path as policy output. But it is a larger displacement than anything
the dry run published, and your velocity-limit fix has not been exercised on
one. **If you would rather we not do this on the first re-run, say so and we
start with `--no-ready-move`** — it is a flag, no rebuild.

**Stereo `ego_view_left`, please.** Our server declares
`["ego_view_left", "left_wrist", "right_wrist"]`. The checkpoint's head view was
trained on `cam_0` of the source recording, which is the **left eye** of the
head stereo pair — we confirmed this by matching `cam_0` against `cam_1` (its
features sit 11 px further right, and the scene sits further right in the image
of the camera that is further left). The mono `ego_view` may well be the same
eye, but which one is not documented anywhere we can find, and using the wrong
eye degrades accuracy without producing a single error.

If stereo is not available for our slot, start the server with
`--head-camera ego_view` and everything runs — just tell us, so we know the
result came from a fallback.

If a declared camera never publishes, the server holds the measured pose and
says why every 5 seconds rather than feeding the policy a black frame.

---

## 6 · Open questions on the decoupled contract

Two of these are now **closed** by the 2026-09-03 dry run. The rest still stand,
and all of them are flags, so an answer costs no rebuild.

1. ~~**What frame are `left_ee_pos` / `right_ee_pos` in?**~~ **ANSWERED — pelvis,
   by your own data.** We publish the wrist pose in the **pelvis** frame, from FK
   with the measured waist, and your offline IK test confirms that is what your
   solver expects. The reasoning, because it is not obvious: your `(target, seed)`
   replay reached **100.0% accept at ~0.0000 median residual** once the tool
   offset was removed. A residual that small is only possible if your FK and our
   target agree on the waist, and the waist was **not** near zero during the run
   — `log.jsonl` puts `waist_yaw` at a steady **-0.234 rad** (|waist| median
   0.228, max 0.307). At that waist the pelvis- and torso-frame poses are
   **7.4 cm apart** (median over the run; 6.0-11.3 cm range). A torso-frame
   mismatch would have shown up as centimetres of residual, not 0.0000. So
   `--ee-frame pelvis` stays, and we are no longer guessing.

   Say so if you read that differently — it is still one flag either way.

2. ~~**Where is the commanded point on the end effector?**~~ **ANSWERED — zero
   offset, and fixed on our side.** Your IK targets the bare `wrist_yaw_link`
   origin; we were publishing it translated 0.05 m along local +x, which is
   exactly the 5.00 cm error you measured. Fixed.

   The fix is the one you specified, and only that: the **published pose** now
   uses a zero offset while the **46-dim state block keeps the 0.05 m** the
   checkpoint was trained with. Those were one shared value in our code, which
   is precisely the trap you flagged — a blanket `--ee-offset-m 0` would have
   fixed the wire and silently moved 6 of the 46 state dims 5 cm off the
   training distribution. They are now two independent values
   (`TRAINING_EE_OFFSET_M` and `ACTION_EE_OFFSET_M` in
   `components/policy/kinematics.py`), the server declares both in its metadata
   frame (`ee_offset_m` for the wire, `state_ee_offset_m` for the model), and
   `scripts/check_conventions.py` fails if they are ever merged again.

   Verified by replaying your `log.jsonl` states through the new code: every
   published pose moves **exactly 5.000 cm**, the state eef blocks move
   **0.000000 cm**, and the quaternion columns are bit-identical.

   `--ee-offset-m` still exists and is now the wire offset alone, so if your
   solver's convention moves again it is one flag and no rebuild.

3. **What does the Thor<->Orin link negotiate?** One command on your side
   (`ethtool <iface> | grep Speed`) and we stop guessing. We ship the
   observation images as JPEG because we cannot see your link: three 480x640x3
   frames are 2.76 MB raw, which is ~22 ms at gigabit but ~221 ms at 100 Mb/s,
   and at 100 Mb/s that alone would leave 1 of our 26 published rows alive.
   JPEG costs 7-10 ms to encode on our Orin. (An earlier draft quoted 56 KB per
   observation here; that figure came from the mock's synthetic frames and a
   real scene is 150-250 KB. The conclusion is unchanged -- see README.) So it
   is the faster option at any link speed we can imagine and we are not asking
   you to change anything. `--jpeg-quality 0` sends raw if you ever want to
   compare; it is a client flag, so no rebuild.

   (We found this on our own bench, where the Thor's NIC had negotiated
   100 Mb/s on a two-pair cable. Ours, not yours — but it is why we would
   rather know than assume.)

4. **What row spacing does the adapter assume?** We publish at **50 Hz**,
   resampled in joint space from the checkpoint's 30 Hz rows, because 50 Hz is
   the controller cadence named in the README. If your adapter reads chunks at
   a different row rate, `--row-hz` sets ours and the server re-declares it in
   its metadata frame, which the client reads for latency compensation. (The
   template client uses `DECOUPLED_CHUNK_HZ = 20` as the row rate for staleness,
   which is where the ambiguity comes from.)

5. **Can the Dex1-1 gripper position reach us at all?** Still no, and we still
   think that is a gap rather than a decision — but **we have measured what it
   costs and it is much smaller than we told you.** Correcting our own claim
   first, then the remaining ask.

   As shipped it cannot reach us. `boundary/states.py` declares the optional
   hand slots as `(7,)` — the Dex3 shape — and `_as_vector` rejects anything
   else outright, so a 1-DoF Dex1-1 jaw position has no schema-valid way onto
   `:5557`. The README's advice ("synthesize whatever your model expects") is
   what we do.

   **What we previously claimed, and withdraw.** We told you the loop breaks
   the moment the jaw closes on something: the jaw stalls at the object, our
   command keeps going, and the policy reads "closed" while holding a leg. We
   have now checked that against our own training set
   (`URL-RFM/IKEA_pickuptheleg`, 322 episodes / 149,437 frames) and **there is
   no such regime in the data.** The teleoperator commanded the grip *width*
   rather than slamming to zero against the object — on `insert table leg to
   table base` the right-hand command sits at ~2.17 rad and the jaw at ~2.35,
   the leg's width. The command never goes below 1.0 rad on that task at all.
   So the stall we warned you about is not something the demonstrations
   contain, and a policy trained on them commands widths too.

   **What it actually costs.** Over 148,793 same-episode frame pairs, at the
   best lag (1 frame, correlation 0.9998):

   | | median | q99 | max |
   |---|---:|---:|---:|
   | `\|measured - command\|`, left | 0.170 | 0.180 | 0.242 |
   | `\|measured - command\|`, right | 0.060 | 0.181 | 0.302 |

   Worst case 0.30 rad of a 5.40 rad stroke — about 6%. And the residual is
   systematic rather than random: the jaw cannot quite reach either end stop,
   so measured is an affine function of command. Fitting that per hand
   (`measured ≈ 0.9602·cmd + 0.1770` left, `0.9322·cmd + 0.3122` right) takes
   the left hand's median error from **0.170 rad to 0.008**. That is now in
   `components/policy/taskspace.py` and applied every step.

   So: **this is a ~0.01 rad problem, not the loop-breaker we described.** We
   are sorry for the overstatement — it was reasoning from the mechanism
   without checking the data. If you were considering a boundary change on our
   account, this is no longer worth one.

   The ask that remains is smaller and you may reasonably decline it: the
   affine fit above is the *training* rig's jaw calibration, and we deploy on
   yours. If the two jaws are calibrated differently our synthesized state
   drifts by the difference — bounded by roughly 0.2 rad, but unmeasurable from
   where we sit. A single number would settle it: **the measured jaw position at
   any one known command**, sent however you like, even in an email. No
   boundary change, no new key, no per-step stream.

   One thing the same analysis fixed on our side, worth flagging because it
   was ours: we were seeding the first inference of every attempt with one
   scalar for both hands. The training set says each subtask starts from a
   different grasp state — `pick table leg` opens both (5.35 / 5.34),
   `insert table leg to table base` starts with the right hand already holding
   the leg (5.36 / 2.35), and `rotate leg to tighten` with the left
   (0.17 / 5.34). Seeding "both open" for the third told the policy a closed
   hand was open, wrong by the entire stroke, on the inference that decides an
   attempt's opening move. Now seeded per task from the prompt.

Two smaller ones we resolved by following the template's own reference: that
`base_height_cmd = 0` and `torso_rpy = 0` mean *neutral / hold* rather than an
absolute target of zero height. If that is wrong, it is the one place our chunks
could surprise you, so please say so.

---

## 7 · Verifying what we sent you

From a checkout:

```bash
scripts/check_boundary.sh                 # boundary/ is byte-identical to the template
python conformance.py --lane decoupled    # our log: docs/conformance_decoupled.log
scripts/dev_stack.sh                      # full loop against the mocks, with our
                                          # three declared cameras
```

Or inside either image, without weights. `PEVAL_CHECKPOINT=` (empty) is what
selects the hold-still policy; leave it set and the server refuses to start
without the weights mounted, which is what you want on the bench and not what
you want here:

```bash
docker run --rm --network host -e PEVAL_CHECKPOINT= --entrypoint bash \
    <image>@<digest> -c \
    "cd /submission && scripts/check_boundary.sh && python conformance.py --lane decoupled"
```

`conformance.py` runs `mock_orin --no-wrists`, which publishes only the mono
`ego_view` — so it validates the action contract with the hold-still policy, not
the camera path. `scripts/dev_stack.sh` uses `--stereo-ego` and all three
cameras, which is the closer rehearsal.

---

## 8 · Building the images (for reference)

Both are aarch64 and neither cross-builds usefully — build each on its own
machine:

```bash
scripts/build_and_push.sh thor <registry>/manipurl-thor v1   # on the Thor
scripts/build_and_push.sh orin <registry>/manipurl-orin v1   # on the Orin
```

The script pushes, reads the digest back from the registry, and writes it into
`manifest.yaml`.

One note on base images: the onboarding brief asks for `nvcr.io/nvidia/l4t-*`
for both machines, but the README's hardware section names
`nvcr.io/nvidia/cuda:13.0.0-devel-ubuntu24.04` for the Thor, since JetPack 7
uses unified Arm CUDA and there is no matching `l4t-*` tag. We followed the
README as the newer of the two. Say the word if you want the `l4t-*` form.

---

## 9 · The dry run tested a checkpoint we no longer ship

The 2026-09-03 report names `gr00t-n1.7-g1-dex1-bct-relarm-aug-30hz-h40`, and
notes the build was unchanged from onboarding certification — correct. But we
swapped the checkpoint after that certification and before the dry run was
scheduled, so what ran on the robot is one generation behind
`manifest.yaml`:

| | ran on the robot 09-03 | what we ship now |
|---|---|---|
| Checkpoint | `bct-relarm-aug-30hz-h40` | `URL-RFM/…-ikea-3task-46d-30hz-h40` |
| Trained prompts | five | **three** (`rotate table base`, `flip table` dropped) |
| Head video key | `cam_head` | **`cam_left_high`** |
| Waist action group | present, discarded by us | **absent from the action space** |
| State / horizon / denoising | 46-dim, 40 @ 30 Hz, 4 steps | unchanged |
| Gripper units and action space | Dex1-1 rad, arms+grippers | unchanged |

Nothing in the boundary-facing contract moved, which is why the plumbing
result stands unchanged. What it means per finding:

* **Finding 1 (tool frame) transfers completely.** It is our FK convention, not
  the checkpoint's. Fixed, and validated by replaying your `log.jsonl`.
* **Finding 3 (safety) is unaffected.** Our action space never contained joint
  velocity in either build.
* **Finding 4 (gripper) — the units transfer, the feedback path does NOT, and
  §6.5 has been rewritten.** The gripper units and the command mapping are
  identical between the two builds. The open-loop state feedback is not: it was
  a raw echo of our last command on 2026-09-03 and now runs through a measured
  calibration, so what you observed is not what will run. And the jaw-stall
  mechanism we originally offered you in §6.5 has been **withdrawn** — the
  training set contains no such regime; see that section for what replaced it.
  4.271-4.500 rad is the *old* checkpoint's output under the *old* feedback
  path. We are re-measuring on the current one and will send the figures before
  the next session.
* **Finding 5 (repetitive motion) is not testable on this data** — and now for
  two reasons rather than one: the IK confound you already identified, plus a
  different policy.

We are not asking you to re-run anything you had not already planned. This is
so the next report is not comparing across a checkpoint boundary without
knowing it.
