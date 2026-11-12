"""The 4-fold split of each pair: 25 % test / 75 % train per fold.

Every image is the test image of exactly one fold, so the four test splits
together cover the whole dataset once.

The file names of the dataset encode the scan they come from: ``16003`` is
slice 3 of case 16, ``2004`` slice 4 of case 2.  The 3-digit names (``032``)
come from a separate collection whose case ids are not encoded; they form a
group of their own.  The split is stratified by that case id: within each case
the slices are shuffled and dealt to the folds in turn, continuing the deal
from one case to the next, so

* the four test sets differ in size by at most one image, and
* every case contributes the same number of slices (+-1) to every test set,
  i.e. each fold sees every case, in the same proportions, in train and test.

Neighbouring slices of a case look alike, so a case-level split would be the
stricter protocol; it cannot give 25 % test sets here (PET-MRI has only four
cases, of 37 to 93 slices), which is why the split is per slice.
"""

import json
import os
import random
from collections import OrderedDict

from .paths import N_FOLDS, PAIRS

IMG_EXT = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")
DEFAULT_SEED = 42


def case_of(stem):
    """Case id encoded in a file stem: all but the last three digits ('' for 3-digit names)."""
    return stem[:-3] if len(stem) > 3 and stem.isdigit() else ""


def list_images(folder):
    """{stem: file name} of the images in ``folder``."""
    return OrderedDict((os.path.splitext(f)[0], f) for f in sorted(os.listdir(folder))
                       if f.lower().endswith(IMG_EXT))


def paired_stems(spec):
    func = list_images(spec.image_dir(spec.func))
    mri = list_images(spec.image_dir(spec.mri))
    if set(func) != set(mri):
        raise ValueError(f"{spec.name}: {spec.func} and {spec.mri} hold different images: "
                         f"{sorted(set(func) ^ set(mri))[:10]}")
    return sorted(func, key=_natural), func, mri


def _natural(stem):
    return (len(stem), stem)


def make_split(stems, n_folds=N_FOLDS, seed=DEFAULT_SEED):
    """{stem: fold (1-based)}, stratified by case, balanced to +-1 image per fold."""
    rng = random.Random(seed)
    cases = OrderedDict()
    for stem in sorted(stems, key=_natural):
        cases.setdefault(case_of(stem), []).append(stem)
    assignment = {}
    turn = 0
    for case in sorted(cases, key=lambda c: (c == "", len(c), c)):
        members = list(cases[case])
        rng.shuffle(members)
        for stem in members:
            assignment[stem] = turn % n_folds + 1
            turn += 1
    return assignment, cases


def make_folds(pair, n_folds=N_FOLDS, seed=DEFAULT_SEED):
    """Write ``splits/<PAIR>.json``, the split of one pair (the shipped files were made with seed 42)."""
    spec = PAIRS[pair]
    stems, func_files, mri_files = paired_stems(spec)
    assignment, cases = make_split(stems, n_folds, seed)
    folds = OrderedDict()
    for fold in range(1, n_folds + 1):
        test = [s for s in stems if assignment[s] == fold]
        train = [s for s in stems if assignment[s] != fold]
        folds[str(fold)] = {"train": train, "test": test}
    payload = OrderedDict([
        ("pair", pair),
        ("n_folds", n_folds),
        ("seed", seed),
        ("n_images", len(stems)),
        ("method", "per-slice split stratified by case id (file stem minus its last 3 digits); "
                   "slices of each case shuffled and dealt to the folds in turn"),
        ("cases", OrderedDict((c or "3-digit names", len(m)) for c, m in cases.items())),
        ("files", {spec.func: func_files, spec.mri: mri_files}),
        ("folds", folds),
    ])
    os.makedirs(os.path.dirname(spec.folds_file), exist_ok=True)
    with open(spec.folds_file, "w") as handle:
        json.dump(payload, handle, indent=1)
    return payload


_CACHE = {}


def load_folds(pair):
    spec = PAIRS[pair]
    if pair not in _CACHE:
        if not os.path.exists(spec.folds_file):
            raise FileNotFoundError(f"{spec.folds_file} missing (shipped in splits/; "
                                    f"`python -m mifkit.cli.make_folds` rebuilds it)")
        with open(spec.folds_file) as handle:
            _CACHE[pair] = json.load(handle)
    return _CACHE[pair]


def split_stems(pair, fold, split):
    """Stems of the ``train`` or ``test`` split of ``fold`` (1-based)."""
    return list(load_folds(pair)["folds"][str(int(fold))][split])


def fold_of(pair):
    """{stem: fold} -- the fold in which each image is a test image."""
    out = {}
    for fold, splits in load_folds(pair)["folds"].items():
        for stem in splits["test"]:
            out[stem] = int(fold)
    return out


def file_of(pair, modality, stem):
    return load_folds(pair)["files"][modality][stem]
