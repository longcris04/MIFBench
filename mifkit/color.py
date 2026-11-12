"""Colour handling shared by every model.

PET and SPECT are colour images.  Every model fuses the luminance Y of the
functional image with the (grey) MRI; the colour result is that fused Y put
back together with the functional image's own Cb/Cr.  The conversion is
OpenCV's full-range BT.601 YCrCb on 8-bit images -- the one the released
code of LRFNet, ALMFnet, VDMUFusion and MixFuse's data loader use:

    Y  = 0.299 R + 0.587 G + 0.114 B
    Cr = (R - Y) * 0.713 + 128
    Cb = (B - Y) * 0.564 + 128

CT and MRI are stored as RGB with R = G = B; their Y is that grey level.
"""

import cv2
import numpy as np


def rgb_to_ycbcr(rgb):
    """uint8 (H,W,3) RGB -> uint8 Y, Cb, Cr, each (H,W)."""
    ycrcb = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2YCrCb)
    return ycrcb[..., 0], ycrcb[..., 2], ycrcb[..., 1]


def ycbcr_to_rgb(y, cb, cr):
    """uint8 Y, Cb, Cr (H,W) -> uint8 (H,W,3) RGB."""
    ycrcb = np.stack([y, cr, cb], axis=-1).astype(np.uint8)
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def to_uint8(x):
    """float image in [0,1] -> uint8, rounded and clipped."""
    return np.clip(np.rint(np.asarray(x, dtype=np.float64) * 255.0), 0, 255).astype(np.uint8)
