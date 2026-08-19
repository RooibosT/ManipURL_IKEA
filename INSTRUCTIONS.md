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
| 4 | Model weights, not baked in | `manifest.yaml` → `weights`; download and access in §0 below |
| 5 | Manifest | `manifest.yaml` |
| 6 | Conformance log | `docs/conformance_decoupled_orin.log` (on our Orin, inside the built image); `docs/conformance_decoupled.log` and `..._py38.log` are the same check off-hardware |
| 7 | Run command per container | §1 and §2 below |

---

## 0 · Before the first run: the weights

The checkpoint is **not** baked into the image — mount it.

```bash
# on the Thor, once
sudo mkdir -p /opt/weights && sudo chown "$USER" /opt/weights
pip install -U "huggingface_hub[cli]"
hf auth login                       # your own HF account, once we have added it
hf download RooibosT/gr00t-n1.7-g1-dex1-bct-relarm-aug-30hz-h40 \
    --local-dir /opt/weights/gr00t-n1.7-g1-dex1-bct-relarm-aug-30hz-h40
```

<https://huggingface.co/RooibosT/gr00t-n1.7-g1-dex1-bct-relarm-aug-30hz-h40>

About **12.6 GB** to download (the checkpoint is stored fp32 and cast to
bfloat16 at load, so it occupies roughly 6 GB on the GPU).

> **The one thing we need from you to make this work:** the repo is private, so
> send us the Hugging Face username(s) of whoever will pull it and we will add
> them as readers. Then `hf auth login` with your own token and the command
> above just works. There is no gate, no license click-through and no request
> form. We would rather grant per-account access than mail you a shared token,
> but say the word and we will issue a fine-grained read-only token instead.

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
    -e PEVAL_CHECKPOINT=/weights/gr00t-n1.7-g1-dex1-bct-relarm-aug-30hz-h40 \
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

The checkpoint is language-conditioned and was trained on exactly five subtask
strings:

```
pick table leg
rotate leg to tighten
insert table leg to table base
rotate table base
flip table
```

Anything else is out of distribution, and the failure is quiet — the policy
still returns a confident-looking chunk. The server logs a warning once per
unseen prompt. Restart the client with a different `--prompt` to switch
subtask; nothing else needs restarting.

There is **no prompt that makes the robot walk.** Locomotion segments were
excluded from the finetune on purpose, so `navigate_cmd`, `base_height_cmd` and
`torso_rpy` all go out as zeros and your controller keeps the lower body.

---

## 3 · Order of operations

1. Your camera and state servers up on the Orin.
2. Thor container. Wait for `policy server listening on ws://0.0.0.0:8765`.
3. Orin container. It waits for both endpoints, then dials the Thor and prints
   the metadata it got back.
4. Your controller dials into `:5556`.

Starting the client before the server is fine — `PolicyLink` retries for 60 s.

**Stopping the client does not stop the robot.** Your controller keeps replaying
its last command. Only your e-stop brings it to a safe state.

---

## 4 · What we need from you

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

## 5 · Open questions on the decoupled contract

We had to guess three things. All three are configurable, so a one-line answer
is enough to correct any of them, and none needs a rebuild.

1. **What frame are `left_ee_pos` / `right_ee_pos` in?** We publish the wrist
   pose in the **pelvis** frame, from FK with the measured waist. Switch with
   `--ee-frame torso` (waist held at zero, which is the frame our checkpoint's
   own state block uses).

2. **Where is the commanded point on the end effector?** We use the
   `wrist_yaw` link origin translated **0.05 m** along its local +x, which is
   the convention the checkpoint was trained against. Change with
   `--ee-offset-m`.

3. **What does the Thor<->Orin link negotiate?** One command on your side
   (`ethtool <iface> | grep Speed`) and we stop guessing. We ship the
   observation images as JPEG because we cannot see your link: three 480x640x3
   frames are 2.76 MB raw, which is ~22 ms at gigabit but ~221 ms at 100 Mb/s,
   and at 100 Mb/s that alone would leave 1 of our 26 published rows alive.
   JPEG costs about 55 ms end to end regardless of link speed, so it is the
   safe default and mildly wasteful on gigabit. If your link is gigabit, add
   `--jpeg-quality 0` to the client command and we send raw — it is a client
   flag, so no rebuild.

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

5. **Can the Dex1-1 gripper position reach us at all?** As shipped it cannot,
   and we think that is a gap rather than a decision. `boundary/states.py`
   declares the optional hand slots as `(7,)` — the Dex3 shape — and
   `_as_vector` rejects anything else outright, so a 1-DoF Dex1-1 jaw position
   has no schema-valid way onto `:5557`. The README's advice ("synthesize
   whatever your model expects") is what we do: our two gripper state dims are
   fed from our own last command.

   That is fine right up to the moment it matters. Our checkpoint was trained
   on the *measured* jaw position, and command and measurement agree only while
   the gripper is moving freely. The instant it closes on a table leg the jaw
   stops at the object and our command keeps going, so the policy sees "closed"
   while the hardware is holding something — exactly the state it needs to read
   correctly to decide whether a grasp succeeded.

   If you can publish it we will use it, in whatever form is least disruptive:
   the real value repeated across the 7-wide vector, a new key, anything. If
   you would rather not touch the boundary, tell us and we will stop asking; we
   just would rather you knew that no team on a Dex1-1 rig can close that loop
   today.

Two smaller ones we resolved by following the template's own reference: that
`base_height_cmd = 0` and `torso_rpy = 0` mean *neutral / hold* rather than an
absolute target of zero height. If that is wrong, it is the one place our chunks
could surprise you, so please say so.

---

## 6 · Verifying what we sent you

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

## 7 · Building the images (for reference)

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
