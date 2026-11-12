"""TC-MoA adapter for the shared MIF benchmark (mifkit).

Model and losses are the release's (``model/ViT_MAE.py`` ``mae_vit_large_patch16`` with
the TC-MoA adapters, ``config/base.yaml``), trained as ``main_train.py`` /
``engine_train.py`` do, on one task:

* backbone: MAE ViT-L/16 with GAN loss (``mae_visualize_vit_large_ganloss.pth`` from the
  MAE release, as the README says), loaded by ``load_pretrained_weights``: every weight in
  that checkpoint is frozen, the rest (adapters, routers, fusion layers, window position
  tables, alphas) trains; interval_tau 4, tau_shift 2, window 14 patches, 3 task slots,
  ``upsample: true`` (the network upsamples its input 2x);
* the VIF task (``task_index`` 0) and its loss ``forward_loss_taskRGBT``: epochs < 2
  (``warmup_epochs``) pixel + prompt + SSIM + aux/(epoch+1), then
  1.5 max-gradient + 1.5 max-pixel + prompt + 2 SSIM + aux/(epoch+1) -- medical images are
  fused as the VIF task in the release too (``dataloader_evaluate.py``: "MandP" -> 0);
* AdamW(lr 3e-4, betas (0.9, 0.95), weight decay 0.05, none on biases and 1-D weights --
  timm ``add_weight_decay``), the learning rate set every iteration by ``lr_sched``:
  linear warm-up over 2 epochs, then half-cosine to 1e-5 at epoch 20;
* mixed precision (``torch.cuda.amp.autocast`` + ``NativeScalerWithGradNormCount``,
  no clipping).  The release's EMA is not reproduced: it writes the averages into a
  ``state_dict()`` copy that is never loaded back, so it never changes the model.

Inputs as the release's VIF loaders: both images as 3-channel (grey replicated), resized
to the multiple of 112 that ``win_HW`` needs (256 -> 336; training: bicubic, as the
loader's resized crop; fusion: bilinear, as ``dataloader_evaluate.py``), ImageNet
normalisation.  "rgb" (visible) = MRI, "t" (infrared) = the functional image (Y).
Fusion as ``main_predict.py``: ``model(A, B, 0)``, prediction resized back to 256, saved
with ``scipy.misc.imsave`` -- which min-max scales a float image -- then its BT.601 luma
is the fused Y.

Benchmark deviations: no random resized crop (whole images, scale 1), batch 16 (release:
3 per task per GPU on 3 GPUs), one task instead of VIF + MEF + MFF jointly.

Compatibility: ``util/pos_embed.py`` uses ``np.float``, removed in NumPy 1.24; the alias is
restored as the built-in ``float`` it always was (files untouched).

The source embeddings S of the paper's Eq. (5) (``BiMixtureOfAdapters.modal_shifts``) are a
plain Python list in the release, so they are neither trained nor in ``state_dict()``:
they keep their random initialisation (normal, std 0.02).  That is reproduced; the
checkpoint stores them so that a model loaded back fuses identically.
"""

import math
import os

import numpy as np
import torch
import torch.nn.functional as F

if not hasattr(np, "float"):
    np.float = float          # util/pos_embed.py: np.arange(..., dtype=np.float)

import model.ViT_MAE as VIT_MAE_Origin  # noqa: E402
from mifkit.folds import split_stems

NAME = "TC-MoA"
# two ViT-L streams on 42x42 tokens (336 px, upsampled 2x): batch 24 and above run out of memory on an
# 80 GB A100; 16 peaks at 75.7 GB (release: 3 per task and GPU)
DEFAULTS = {"batch_size": 16, "lr": 3e-4}
HERE = os.path.dirname(os.path.abspath(__file__))
MAE_WEIGHTS = os.path.join(HERE, "checkpoint", "mae_visualize_vit_large_ganloss.pth")
CONFIG = {   # config/base.yaml (model and schedule keys)
    "upsample": True, "interval_tau": 4, "task_num": 3, "tau_shift_value": 2, "shift_window_size": 14,
    "model_type": "mae_vit_large_patch16", "warmup_epochs": 2, "min_lr": 1e-5, "weight_decay": 0.05,
}
TASK_VIF = 0
WIN_HW = 112                       # dataloader_VIF: 112 with upsample
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
HPARAMS = {
    "backbone": "MAE ViT-L/16 GAN-loss (mae_visualize_vit_large_ganloss.pth), frozen; adapters trained",
    "optimizer": "AdamW(lr=3e-4, betas=(0.9, 0.95), weight_decay=0.05, none on bias/1-D), per-iteration "
                 "schedule: 2-epoch linear warm-up, half-cosine to 1e-5 -- base.yaml, lr_sched.py",
    "loss": "VIF task (forward_loss_taskRGBT): warm-up pixel+prompt+ssim+aux/(e+1); then "
            "1.5 maxgrad + 1.5 maxpixel + prompt + 2 ssim + aux/(e+1)",
    "precision": "fp16 autocast + GradScaler (NativeScalerWithGradNormCount), no clipping",
    "ema": "not reproduced (no effect in the release: averages written to a discarded state_dict copy)",
    "inputs": "rgb = MRI, t = func Y, grey replicated to 3 channels, resized 256->336 (train bicubic, fuse "
              "bilinear), ImageNet normalisation",
    "fusion": "prediction resized to 256, min-max scaled (scipy.misc.imsave), BT.601 luma",
    "code_defaults_not_used": "random resized crop 0.81-1 + h-flip, batch 3/task/GPU x 3 GPUs, VIF+MEF+MFF "
                              "jointly (benchmark protocol instead)",
}


class State:
    def __init__(self, model, optimizer, scaler, device):
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.device = device
        self.steps_per_epoch = 1
        self.iteration = 0          # within the current epoch
        self.lr_max = None


def _param_groups(model, weight_decay):
    """timm 0.3.2 optim_factory.add_weight_decay (main_train.py, no skip list)."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [{"params": no_decay, "weight_decay": 0.0}, {"params": decay, "weight_decay": weight_decay}]


def build(args, device, pair):
    if not os.path.exists(MAE_WEIGHTS):
        raise SystemExit(f"TC-MoA: MAE weights missing: {MAE_WEIGHTS}")
    model = getattr(VIT_MAE_Origin, CONFIG["model_type"])(dict(CONFIG)).to(device)
    *_, model = VIT_MAE_Origin.load_pretrained_weights(model, model_name=None, weights_path=MAE_WEIGHTS,
                                                       verbose=False)
    optimizer = torch.optim.AdamW(_param_groups(model, CONFIG["weight_decay"]), lr=args.lr, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    state = State(model, optimizer, scaler, device)
    state.lr_max = args.lr
    return state


def prepare(state, args, data, device):
    n = len(split_stems(args.pair, args.fold, "train")) * max(1, args.repeat or 1)
    state.steps_per_epoch = n // args.batch_size if n % args.batch_size == 1 else math.ceil(n / args.batch_size)


def describe(state):
    total = sum(p.numel() for p in state.model.parameters())
    trained = sum(p.numel() for p in state.model.parameters() if p.requires_grad)
    return f"TC-MoA: {total / 1e6:.1f}M parameters, {trained / 1e6:.2f}M trained (MAE ViT-L frozen)"


def current_lr(state):
    return state.optimizer.param_groups[0]["lr"]


def _adjust_lr(state, epoch_float, args):
    """util/lr_sched.adjust_learning_rate"""
    warm = CONFIG["warmup_epochs"]
    if epoch_float < warm:
        lr = state.lr_max * epoch_float / warm
    else:
        lr = CONFIG["min_lr"] + (state.lr_max - CONFIG["min_lr"]) * 0.5 * \
            (1. + math.cos(math.pi * (epoch_float - warm) / (args.epochs - warm)))
    for group in state.optimizer.param_groups:
        group["lr"] = lr


def on_epoch_start(state, epoch, args):
    state.iteration = 0
    state.optimizer.zero_grad()


def set_mode(state, train):
    state.model.train(train)


def _prep(x, mode):
    h, w = x.shape[-2:]
    size = (math.ceil(h / WIN_HW) * WIN_HW, math.ceil(w / WIN_HW) * WIN_HW)
    x = F.interpolate(x.expand(-1, 3, -1, -1), size=size, mode=mode, align_corners=False)
    if mode == "bicubic":
        x = x.clamp(0, 1)        # PIL resizes 8-bit images
    mean = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def _autocast(state):
    return torch.autocast(device_type=state.device.type, dtype=torch.float16, enabled=state.device.type == "cuda")


def train_step(state, batch, args, epoch):
    _adjust_lr(state, state.iteration / state.steps_per_epoch + (epoch - 1), args)
    a = _prep(batch["mri"], "bicubic")
    b = _prep(batch["func"], "bicubic")
    with _autocast(state):
        loss_dict, _, _ = state.model(a, b, TASK_VIF, epoch - 1)
        loss = loss_dict["loss"]
    state.scaler.scale(loss).backward()
    state.scaler.unscale_(state.optimizer)
    state.scaler.step(state.optimizer)
    state.scaler.update()
    state.optimizer.zero_grad()
    state.iteration += 1
    return {k: float(v) for k, v in loss_dict.items() if torch.is_tensor(v) or isinstance(v, (int, float))}


@torch.no_grad()
def fuse(state, batch, args):
    h, w = batch["mri"].shape[-2:]
    a = _prep(batch["mri"], "bilinear")
    b = _prep(batch["func"], "bilinear")
    with _autocast(state):
        _, pred, _ = state.model(a, b, TASK_VIF)
    pred = F.interpolate(pred.float(), size=(h, w), mode="bilinear", align_corners=False)
    lo = pred.amin(dim=(1, 2, 3), keepdim=True)
    hi = pred.amax(dim=(1, 2, 3), keepdim=True)
    pred = (pred - lo) / (hi - lo).clamp_min(1e-8)          # scipy.misc.imsave bytescale
    return 0.299 * pred[:, 0:1] + 0.587 * pred[:, 1:2] + 0.114 * pred[:, 2:3]


def _modal_shifts(model):
    return {name: m.modal_shifts for name, m in model.named_modules() if hasattr(m, "modal_shifts")}


def state_dict(state):
    trained = {k: v for k, v in state.model.named_parameters() if v.requires_grad}
    return {"trained": {k: v.detach().clone() for k, v in trained.items()},
            "modal_shifts": {k: [t.detach().cpu().clone() for t in v] for k, v in _modal_shifts(state.model).items()},
            "optimizer": state.optimizer.state_dict(), "scaler": state.scaler.state_dict()}


def load_state_dict(state, blob):
    missing = state.model.load_state_dict(blob["trained"], strict=False)
    if missing.unexpected_keys:
        raise RuntimeError(f"TC-MoA checkpoint: unexpected keys {missing.unexpected_keys[:5]}")
    with torch.no_grad():
        for name, tensors in _modal_shifts(state.model).items():
            for t, saved in zip(tensors, blob["modal_shifts"][name]):
                t.copy_(saved)
    if "optimizer" in blob:
        state.optimizer.load_state_dict(blob["optimizer"])
    if "scaler" in blob:
        state.scaler.load_state_dict(blob["scaler"])
