"""Images of a pair, held in memory, and the datasets the training loop reads.

A sample is a dict:

    func      (1,H,W) float in [0,1]  luminance Y of the functional image (CT / PET / SPECT)
    mri       (1,H,W) float in [0,1]  MRI (grey)
    func_rgb  (3,H,W) float in [0,1]  the functional image as stored (RGB; grey CT replicated)
    index     int                     position of the image in PairData (all images of the pair)
    flip      int                     augmentation applied: bit 0 horizontal, bit 1 vertical
    name      str                     file stem

Images are used at their native 256x256 -- no resize.  The augmentation is a
random horizontal and a random vertical flip (each p=0.5), applied identically
to both modalities (an adapter can switch it off).  A model may also train on
random ``crop`` x ``crop`` patches (same position in both modalities), drawn
anew at every pass; with ``repeat`` passes per epoch that is ``repeat`` random
crops of every training image per epoch (MMIF-INet).  Test images are never
flipped or cropped.

An adapter may add per-sample tensors computed in the loader workers (e.g.
LRFNet's detail weights) through ``sample_extras(func_u8, mri_u8, rgb_u8)``.
"""

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .color import rgb_to_ycbcr
from .folds import load_folds
from .paths import PAIRS

FLIP_NAMES = {0: "none", 1: "h", 2: "v", 3: "hv"}


def apply_flip(array, code):
    """Flip an (H,W) or (H,W,C) array: bit 0 of ``code`` horizontally, bit 1 vertically."""
    if code & 1:
        array = array[:, ::-1]
    if code & 2:
        array = array[::-1]
    return np.ascontiguousarray(array)


def _read_rgb(path):
    with Image.open(path) as image:
        return np.array(image.convert("RGB"))


class PairData:
    """Every image of a pair, decoded once: functional RGB/Y/Cb/Cr and MRI grey, all uint8."""

    def __init__(self, pair):
        import os

        spec = PAIRS[pair]
        folds = load_folds(pair)
        files = folds["files"]
        self.pair = pair
        self.spec = spec
        self.names = sorted(files[spec.func], key=lambda s: (len(s), s))
        self.index = {stem: i for i, stem in enumerate(self.names)}
        rgb, y, cb, cr, mri = [], [], [], [], []
        for stem in self.names:
            f_rgb = _read_rgb(os.path.join(spec.image_dir(spec.func), files[spec.func][stem]))
            f_y, f_cb, f_cr = rgb_to_ycbcr(f_rgb)
            m_rgb = _read_rgb(os.path.join(spec.image_dir(spec.mri), files[spec.mri][stem]))
            m_y, _, _ = rgb_to_ycbcr(m_rgb)
            rgb.append(f_rgb), y.append(f_y), cb.append(f_cb), cr.append(f_cr), mri.append(m_y)
        self.func_rgb = np.stack(rgb)
        self.func_y = np.stack(y)
        self.func_cb = np.stack(cb)
        self.func_cr = np.stack(cr)
        self.mri = np.stack(mri)
        self.shape = self.mri.shape[1:]

    def indices(self, stems):
        return [self.index[s] for s in stems]


_PAIR_DATA = {}


def pair_data(pair):
    if pair not in _PAIR_DATA:
        _PAIR_DATA[pair] = PairData(pair)
    return _PAIR_DATA[pair]


class FusionDataset(Dataset):
    def __init__(self, data, stems, train, flip=True, extras=None, repeat=1, crop=None):
        self.data = data
        self.idx = data.indices(stems)
        self.train = train
        self.flip = flip and train
        self.extras = extras
        self.repeat = max(1, int(repeat)) if train else 1
        self.crop = int(crop) if (crop and train) else None

    def __len__(self):
        return len(self.idx) * self.repeat

    def __getitem__(self, i):
        d = self.data
        idx = self.idx[i % len(self.idx)]
        code = 0
        if self.flip:
            r = torch.rand(2)
            code = int(r[0] < 0.5) | (int(r[1] < 0.5) << 1)
        func = apply_flip(d.func_y[idx], code)
        mri = apply_flip(d.mri[idx], code)
        rgb = apply_flip(d.func_rgb[idx], code)
        if self.crop:
            c = self.crop
            top = int(torch.randint(0, func.shape[0] - c + 1, (1,)))
            left = int(torch.randint(0, func.shape[1] - c + 1, (1,)))
            func = np.ascontiguousarray(func[top:top + c, left:left + c])
            mri = np.ascontiguousarray(mri[top:top + c, left:left + c])
            rgb = np.ascontiguousarray(rgb[top:top + c, left:left + c])
        sample = {
            "func": torch.from_numpy(func).float().div_(255.0).unsqueeze(0),
            "mri": torch.from_numpy(mri).float().div_(255.0).unsqueeze(0),
            "func_rgb": torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float().div_(255.0),
            "index": idx,
            "flip": code,
            "name": d.names[idx],
        }
        if self.extras is not None:
            sample.update(self.extras(func, mri, rgb))
        return sample
