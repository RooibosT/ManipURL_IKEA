"""JPEG on the Thor<->Orin link. Team-owned, invisible to the organizer.

One observation is three 480x640x3 frames, 2.76 MB raw. Measured on our own
Thor<->Orin link that costs about 330 ms of the round trip -- far more than the
185 ms the policy itself takes, and enough that the client discards 25 of every
26 published rows as stale. JPEG at quality 85 takes the same observation to
roughly 150 KB, which is the difference between a robot that moves and one that
holds its last command.

The frames arrived as JPEG from the organizer's camera server in the first
place; `boundary/cameras.py` decodes them for us, so this re-encodes what was
already lossy. That is a second generation of JPEG artefacts on top of the
first, at a quality where the second pass is close to a no-op.

Channel order is preserved end to end: RGB in, RGB out. cv2 wants BGR, so both
ends convert, rather than relying on the asymmetry cancelling -- it nearly does,
because chroma subsampling weights the channels differently, but "nearly" is not
something to leave in a control path.
"""

from __future__ import annotations

from typing import Dict

import cv2
import numpy as np

DEFAULT_QUALITY = 85


def encode_jpeg(image: np.ndarray, quality: int = DEFAULT_QUALITY) -> bytes:
    """(H, W, 3) uint8 RGB -> JPEG bytes."""
    bgr = cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise ValueError("cv2 refused to encode a {} {} image".format(
            image.shape, image.dtype))
    return buffer.tobytes()


def decode_jpeg(blob: bytes) -> np.ndarray:
    """JPEG bytes -> (H, W, 3) uint8 RGB."""
    bgr = cv2.imdecode(np.frombuffer(blob, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("undecodable JPEG frame ({} bytes)".format(len(blob)))
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def encode_images(images: Dict[str, np.ndarray], quality: int = DEFAULT_QUALITY):
    return {key: encode_jpeg(value, quality) for key, value in images.items()}


def decode_images(blobs: Dict[str, bytes]):
    return {key: decode_jpeg(value) for key, value in blobs.items()}
