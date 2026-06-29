#!/usr/bin/env python3
"""Generate the Orienta app icon: an IPF orientation triangle (barycentric RGB
blend: red top, green bottom-right, blue bottom-left) in a dark rounded square.

Outputs:
  resources/icon.png        (256x256, app/window icon)
  resources/icon.ico        (multi-size Windows installer icon)
  branding/orienta-icon-512.png  (512, for README / Zenodo / web)
"""
import os
import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S = 1024  # supersample, then downscale for anti-aliasing

# Triangle vertices in the 120-unit design space (match the SVG):
def vt(x, y):
    return np.array([x / 120 * S, y / 120 * S])

P0 = vt(60, 25)   # top    -> RED
P1 = vt(98, 91)   # b-right -> GREEN
P2 = vt(22, 91)   # b-left  -> BLUE
RED = np.array([255, 70, 70], float)
GREEN = np.array([40, 220, 110], float)
BLUE = np.array([60, 130, 255], float)
BG = np.array([15, 24, 48], float)        # triangle base navy (#0f1830)
CARD = np.array([15, 20, 27], float)      # rounded-square bg (#0f141b)

ys, xs = np.mgrid[0:S, 0:S]
px = np.stack([xs, ys], axis=-1).astype(float)

# Barycentric coordinates relative to (P0,P1,P2)
v0 = P1 - P0
v1 = P2 - P0
v2 = px - P0
d00 = v0 @ v0
d01 = v0 @ v1
d11 = v1 @ v1
d20 = (v2 * v0).sum(-1)
d21 = (v2 * v1).sum(-1)
den = d00 * d11 - d01 * d01
v = (d11 * d20 - d01 * d21) / den
w = (d00 * d21 - d01 * d20) / den
u = 1 - v - w
inside = (u >= 0) & (v >= 0) & (w >= 0)

# Color = u*RED + v*GREEN + w*BLUE, then brighten balanced (center) regions
col = u[..., None] * RED + v[..., None] * GREEN + w[..., None] * BLUE
balance = np.clip(np.minimum(np.minimum(u, v), w), 0, None)
col = np.clip(col + balance[..., None] * 140, 0, 255)

img = np.empty((S, S, 3), float)
img[:] = CARD
img[inside] = BG  # behind blend, only matters at AA edges
img[inside] = col[inside]
arr = img.astype(np.uint8)
pim = Image.fromarray(arr, "RGB").convert("RGBA")

# faint white triangle outline for definition
d = ImageDraw.Draw(pim)
d.line([tuple(P0), tuple(P1), tuple(P2), tuple(P0)], fill=(255, 255, 255, 90),
       width=max(2, S // 240), joint="curve")

# rounded-square alpha mask
mask = Image.new("L", (S, S), 0)
dm = ImageDraw.Draw(mask)
dm.rounded_rectangle(
    [int(5 / 120 * S), int(5 / 120 * S), int(115 / 120 * S), int(115 / 120 * S)],
    radius=int(27 / 120 * S), fill=255,
)
out = Image.new("RGBA", (S, S), (0, 0, 0, 0))
out.paste(pim, (0, 0), mask)

os.makedirs(os.path.join(ROOT, "resources"), exist_ok=True)
icon256 = out.resize((256, 256), Image.LANCZOS)
icon256.save(os.path.join(ROOT, "resources", "icon.png"))
icon256.save(os.path.join(ROOT, "resources", "icon.ico"),
             sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
out.resize((512, 512), Image.LANCZOS).save(os.path.join(ROOT, "branding", "orienta-icon-512.png"))
print("wrote resources/icon.png, resources/icon.ico, branding/orienta-icon-512.png")
