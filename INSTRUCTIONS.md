# How to run our two containers

Team **ManipURL** · lane **`decoupled`** · Unitree G1 EDU with Dex1-1 grippers.

Everything below is the command as we expect you to type it. `manifest.yaml`
carries the same values in machine-readable form.

---

## 0 · Before the first run: the weights

The checkpoint is **not** baked into the image — mount it.

```bash
# on the Thor, once
pip install -U "huggingface_hub[cli]"
hf auth login                       # or: export HF_TOKEN=<token we give you>
hf download RooibosT/gr00t-n1.7-g1-dex1-bct-relarm-aug-30hz-h40 \
    --local-dir /opt/weights/gr00t-n1.7-g1-dex1-bct-relarm-aug-30hz-h40
```

About 6.9 GB. The repo is **private**: tell us whether you would rather we add
your Hugging Face account as a reader or hand you a scoped read token, and we
will do that before the slot. There is no other gating — no license click-through,
no request form.

Both images also come from `nvcr.io`, which needs an NGC login even though the
base images are public: `docker login nvcr.io` with username `$oauthtoken` and
an NGC API key.

---

## 1 · Thor — policy server

```bash
docker run --rm -it \
    --runtime nvidia \
    --network host \
    --ipc host \
    -v /opt/weights:/weights:ro \
    -e PEVAL_CHECKPOINT=/weights/gr00t-n1.7-g1-dex1-bct-relarm-aug-30hz-h40 \
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

3. **What row spacing does the adapter assume?** We publish at **50 Hz**,
   resampled in joint space from the checkpoint's 30 Hz rows, because 50 Hz is
   the controller cadence named in the README. If your adapter reads chunks at
   a different row rate, `--row-hz` sets ours and the server re-declares it in
   its metadata frame, which the client reads for latency compensation. (The
   template client uses `DECOUPLED_CHUNK_HZ = 20` as the row rate for staleness,
   which is where the ambiguity comes from.)

Two smaller ones we resolved by following the template's own reference: that
`base_height_cmd = 0` and `torso_rpy = 0` mean *neutral / hold* rather than an
absolute target of zero height. If that is wrong, it is the one place our chunks
could surprise you, so please say so.

---

## 6 · Verifying what we sent you

```bash
scripts/check_boundary.sh                 # boundary/ is byte-identical to the template
python conformance.py --lane decoupled    # our log: docs/conformance_decoupled.log
scripts/dev_stack.sh                      # full loop against the mocks, with our
                                          # three declared cameras
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
