"""The benchmark's models and how to import their adapters."""

import importlib.util
import os
import sys

from .paths import model_dir

#: the benchmarked models (folders under models/), in table order
MODELS = ["ALMFnet", "DATFuse", "ECINFusion", "FATFusion", "LRFNet", "MMIF-CDDFuse",
          "MMIF-INet", "MixFuse", "TUFusion", "VDMUFusion",
          "DSPFusion", "FusionGAN", "GeSeNet", "MM-Net", "MMA-UNet", "SigFusion", "SwinFusion",
          "TC-MoA", "U2Fusion", "YDTR"]
#: order in which a fold job trains them, by cost
#: (VDMUFusion's 2000-step sampler at every epoch makes it by far the longest run)
RUN_ORDER = ["DATFuse", "ALMFnet", "LRFNet", "MMIF-INet", "TUFusion", "ECINFusion",
             "MMIF-CDDFuse", "MixFuse", "FATFusion",
             "YDTR", "GeSeNet", "FusionGAN", "MM-Net", "MMA-UNet", "U2Fusion", "SigFusion",
             "SwinFusion", "TC-MoA", "DSPFusion",
             "VDMUFusion"]
#: extra models trained with the same protocol (a folder under models/ with a mif_adapter.py):
#: a fold job trains them only when named (``run_fold --models``), the summary tables leave
#: them out unless named, the review server shows them after the benchmark's models
EXPERIMENTS = []
ALL_MODELS = MODELS + EXPERIMENTS


def load_adapter(model):
    """Import ``models/<model>/mif_adapter.py`` with the model's repository first on sys.path.

    Every repository uses top-level module names (``net``, ``losses``, ``config``,
    ``utils`` ...) that collide between models, so one process trains one model.
    """
    folder = model_dir(model)
    path = os.path.join(folder, "mif_adapter.py")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no adapter for {model}: {path}")
    if folder not in sys.path:
        sys.path.insert(0, folder)
    spec = importlib.util.spec_from_file_location(f"mif_adapter_{model.replace('-', '_')}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
