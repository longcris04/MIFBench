"""Where everything lives: dataset, folds, model repositories and run folders.

Layout (relative to the repository root ``ROOT``)::

    data/<PAIR>/<MODALITY>/<stem>.bmp        every image of the pair (download, see README)
    splits/<PAIR>.json                       the 4-fold split
    models/<MODEL>/                          the model's released code + mif_adapter.py
    models/<MODEL>/mif_runs/<PAIR>/fold<k>/  one training run
    evaluation/                              the MATLAB metrics (run under GNU Octave)

``MIF_DATA_ROOT`` points the pipeline at another copy of ``data/``.
"""

import os
from collections import OrderedDict
from dataclasses import dataclass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.environ.get("MIF_DATA_ROOT", os.path.join(ROOT, "data"))
SPLITS_DIR = os.path.join(ROOT, "splits")
MODELS_DIR = os.path.join(ROOT, "models")
EVAL_CODE = os.path.join(ROOT, "evaluation")
CACHE_DIR = os.environ.get("MIF_CACHE", os.path.join(ROOT, ".mif_cache"))

N_FOLDS = 4
FOLDS = tuple(range(1, N_FOLDS + 1))
# MIF_RUNS_FOLDER redirects every run (e.g. "mif_runs_test" for a dry run of the fold runner)
RUNS_FOLDER = os.environ.get("MIF_RUNS_FOLDER", "mif_runs")


@dataclass(frozen=True)
class PairSpec:
    name: str        # "PET-MRI"
    func: str        # functional / other modality folder: "CT", "PET", "SPECT"
    color: bool      # True when the functional modality is a colour image
    mri: str = "MRI"

    @property
    def root(self):
        return os.path.join(DATA_ROOT, self.name)

    def image_dir(self, modality):
        """Folder holding every image of ``modality``."""
        return os.path.join(self.root, modality)

    @property
    def folds_file(self):
        return os.path.join(SPLITS_DIR, f"{self.name}.json")


PAIRS = OrderedDict((spec.name, spec) for spec in (
    PairSpec("CT-MRI", "CT", color=False),
    PairSpec("PET-MRI", "PET", color=True),
    PairSpec("SPECT-MRI", "SPECT", color=True),
))


def model_dir(model):
    return os.path.join(MODELS_DIR, model)


def run_dir(model, pair, fold, folder=RUNS_FOLDER):
    return os.path.join(model_dir(model), folder, pair, f"fold{int(fold)}")


def epoch_dir(run, epoch):
    return os.path.join(run, "results", f"epoch_{int(epoch):03d}")
