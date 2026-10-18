"""Fixed pixel patch encoder for the finite wafer-image setting; no CLIP weights."""

import numpy as np
from PIL import Image


def texture_tokens(path):
    with Image.open(path) as opened:
        image = opened.convert("RGB")
        width, height = image.size
        tokens = []
        for top, bottom in [(0, height // 2), (height // 2, height)]:
            for left, right in [(0, width // 2), (width // 2, width)]:
                patch = image.crop((left, top, right, bottom)).resize((16, 16))
                tokens.append(np.asarray(patch, dtype=np.float32).ravel() / 255)
    return np.stack(tokens)
