#!/usr/bin/env bash
# Build one container, push it, and write the digest the registry returns back
# into manifest.yaml. The digest is the thing the organizer runs, so it has to
# come from the registry rather than from a local build ID.
#
#   scripts/build_and_push.sh thor ghcr.io/rooibost/manipurl-thor v1
#   scripts/build_and_push.sh orin ghcr.io/rooibost/manipurl-orin v1
#
# RUN EACH ON ITS OWN MACHINE. Both images are aarch64 and neither cross-builds
# usefully: the Thor installer refuses any architecture but aarch64, and the
# Orin's l4t tag has to match the device's L4T. Build the Thor image on the
# Thor and the Orin image on the Orin (or on a matching JetPack host).
set -euo pipefail
cd "$(dirname "$0")/.."

TARGET="${1:?usage: build_and_push.sh <thor|orin> <image> [tag]}"
IMAGE="${2:?usage: build_and_push.sh <thor|orin> <image> [tag]}"
TAG="${3:-$(git rev-parse --short HEAD 2>/dev/null || echo latest)}"

case "$TARGET" in
    thor) DOCKERFILE=docker/Dockerfile.thor ;;
    orin) DOCKERFILE=docker/Dockerfile.orin ;;
    *) echo "target must be 'thor' or 'orin', got '$TARGET'" >&2; exit 2 ;;
esac

if [ "$(uname -m)" != "aarch64" ]; then
    echo "WARNING: building on $(uname -m). Both images target aarch64 Jetsons;" >&2
    echo "         an x86 build will not run on the bench." >&2
fi

# A modified boundary/ is a submission defect, so catch it before the push.
scripts/check_boundary.sh

echo "--- building $IMAGE:$TAG from $DOCKERFILE ---"
docker build -f "$DOCKERFILE" -t "$IMAGE:$TAG" .

echo "--- pushing ---"
docker push "$IMAGE:$TAG"

DIGEST="$(docker inspect --format '{{index .RepoDigests 0}}' "$IMAGE:$TAG" | cut -d@ -f2)"
if [ -z "$DIGEST" ]; then
    echo "could not read a digest back for $IMAGE:$TAG" >&2
    exit 1
fi
echo "--- $TARGET digest: $DIGEST ---"

python3 - "$TARGET" "$IMAGE:$TAG" "$DIGEST" <<'PY'
import re
import sys

target, image, digest = sys.argv[1:4]
path = "manifest.yaml"
lines = open(path).read().splitlines(keepends=True)

# Walk to the target's block under images:, then rewrite only its image/digest
# lines. Editing in place keeps the comments, which carry most of the file.
in_images = False
in_target = False
patched = {"image": False, "digest": False}
for i, line in enumerate(lines):
    if re.match(r"^images:\s*$", line):
        in_images = True
        continue
    if in_images and re.match(r"^\S", line):
        break                              # left the images: block
    if in_images and re.match(r"^  \w+:\s*$", line):
        in_target = line.strip().rstrip(":") == target
        continue
    if not in_target:
        continue
    for key, value in (("image", image), ("digest", digest)):
        m = re.match(r"^(    {}:\s+)(\S+)(.*)$".format(key), line)
        if m:
            lines[i] = "{}{}\n".format(m.group(1), value)
            patched[key] = True

missing = [k for k, ok in patched.items() if not ok]
if missing:
    raise SystemExit(
        "manifest.yaml: could not find {} under images.{} -- edit it by hand".format(
            ", ".join(missing), target
        )
    )
open(path, "w").write("".join(lines))
print("manifest.yaml updated: images.{}.image / .digest".format(target))
PY

echo
echo "Done. Commit manifest.yaml so the digest ships with the repo."
