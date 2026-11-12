"""CDDFuse adapter for the shared MIF benchmark (mifkit).

``net.py`` and ``utils/loss.py`` are the published ones, and the two-phase
loop of ``train.py`` is kept: Adam per module (lr 1e-4, weight decay 0),
coefficients 1 / 1 / 2 / 5, gradient clipping at 0.01, StepLR(gamma 0.5) with
a 1e-6 floor, the SSIM term from kornia, and

  Phase I  -- encoder + decoder reconstruct each modality
              (5*SSIM + MSE per modality, decomposition loss, gradient loss);
  Phase II -- all four modules, ``Fusionloss`` (max-intensity L1 + 10 * max-gradient L1)
              + 2 * decomposition loss.

``train.py`` runs 120 epochs with the phase switch at 40 and the LR step every
20; both are kept as fractions of the run (1/3 and 1/6), so with 20 epochs
phase II starts at epoch 8 and the LR halves every 3 epochs.

Medical-fusion specifics, from ``test_MIF.py``: VIS = MRI, IR = the functional
modality; the MIF model decodes without the residual input image
(``Decoder(None, ...)``) and the fused image is min-max normalised.  The
decoder is trained the same way (``inp_img=None``) so training and inference
use one decoder; ``train.py`` -- the infrared-visible recipe -- passes VIS there.

Not kept: ``nn.DataParallel`` (one GPU per job); ``kornia.losses.SSIM`` is
called by its current name ``SSIMLoss`` (same function).
"""

import kornia
import torch
import torch.nn as nn
from kornia.losses import SSIMLoss

from net import BaseFeatureExtraction, DetailFeatureExtraction, Restormer_Decoder, Restormer_Encoder
from utils.loss import Fusionloss, cc

NAME = "MMIF-CDDFuse"
# Restormer blocks at full resolution: at 256x256 batch 8 and above run out of memory on an
# 80 GB A100; 6 peaks at 61.6 GB.
DEFAULTS = {"batch_size": 6, "lr": 1e-4}

COEFF_MSE_VF = 1.0
COEFF_MSE_IF = 1.0
COEFF_DECOMP = 2.0
COEFF_TV = 5.0
CLIP_GRAD_NORM = 0.01
OPTIM_GAMMA = 0.5
LR_FLOOR = 1e-6
NUM_EPOCHS, EPOCH_GAP, OPTIM_STEP = 120, 40, 20   # train.py
WEIGHT_DECAY = 0.0

HPARAMS = {
    "optimizer": "Adam(lr=1e-4, weight_decay=0) for each of encoder/decoder/base/detail -- train.py",
    "scheduler": "StepLR(step=20 of 120 epochs, gamma=0.5), lr floor 1e-6; phase switch at epoch 40 of 120 "
                 "-> scaled to the run length (20 epochs: phase II from epoch 8, step every 3 epochs)",
    "loss": "phase I: 1*(5*SSIM+MSE)_V + 1*(5*SSIM+MSE)_I + 2*decomp + 5*grad; "
            "phase II: Fusionloss (L1 max-intensity + 10*L1 max-gradient) + 2*decomp",
    "clip_grad_norm": 0.01,
    "inputs": "VIS = MRI, IR = func Y; Decoder(None, ...) and min-max output as test_MIF.py",
    "code_defaults_not_used": "batch 8, 120 epochs, 128x128 MSRS patches (benchmark protocol instead)",
}


class State:
    def __init__(self, args, device):
        self.device = device
        self.encoder = Restormer_Encoder().to(device)
        self.decoder = Restormer_Decoder().to(device)
        self.base_fuse = BaseFeatureExtraction(dim=64, num_heads=8).to(device)
        self.detail_fuse = DetailFeatureExtraction(num_layers=1).to(device)
        self.modules_ = [self.encoder, self.decoder, self.base_fuse, self.detail_fuse]
        self.optimizers = [torch.optim.Adam(m.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
                           for m in self.modules_]
        self.step_size = max(1, round(args.epochs * OPTIM_STEP / NUM_EPOCHS))
        self.epoch_gap = max(1, round(args.epochs * EPOCH_GAP / NUM_EPOCHS))
        self.schedulers = [torch.optim.lr_scheduler.StepLR(o, step_size=self.step_size, gamma=OPTIM_GAMMA)
                           for o in self.optimizers]
        self.criteria_fusion = Fusionloss().to(device)
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.ssim = SSIMLoss(11, reduction="mean").to(device)


def build(args, device, pair):
    return State(args, device)


def describe(state):
    total = sum(p.numel() for m in state.modules_ for p in m.parameters())
    return (f"CDDFuse: {total / 1e6:.3f}M parameters; phase I epochs 1-{state.epoch_gap}, "
            f"LR x{OPTIM_GAMMA} every {state.step_size} epochs")


def current_lr(state):
    return state.optimizers[0].param_groups[0]["lr"]


def set_mode(state, train):
    for module in state.modules_:
        module.train(train)


def _clip(*modules):
    for module in modules:
        nn.utils.clip_grad_norm_(module.parameters(), max_norm=CLIP_GRAD_NORM, norm_type=2)


def train_step(state, batch, args, epoch):
    data_VIS, data_IR = batch["mri"], batch["func"]
    for module in state.modules_:
        module.zero_grad()
    for optimizer in state.optimizers:
        optimizer.zero_grad()

    if epoch <= state.epoch_gap:  # Phase I (train.py: epoch < epoch_gap, 0-based)
        feature_V_B, feature_V_D, _ = state.encoder(data_VIS)
        feature_I_B, feature_I_D, _ = state.encoder(data_IR)
        data_VIS_hat, _ = state.decoder(None, feature_V_B, feature_V_D)
        data_IR_hat, _ = state.decoder(None, feature_I_B, feature_I_D)

        cc_loss_B = cc(feature_V_B, feature_I_B)
        cc_loss_D = cc(feature_V_D, feature_I_D)
        mse_loss_V = 5 * state.ssim(data_VIS, data_VIS_hat) + state.mse(data_VIS, data_VIS_hat)
        mse_loss_I = 5 * state.ssim(data_IR, data_IR_hat) + state.mse(data_IR, data_IR_hat)
        Gradient_loss = state.l1(kornia.filters.SpatialGradient()(data_VIS),
                                 kornia.filters.SpatialGradient()(data_VIS_hat))
        loss_decomp = (cc_loss_D) ** 2 / (1.01 + cc_loss_B)
        loss = (COEFF_MSE_VF * mse_loss_V + COEFF_MSE_IF * mse_loss_I
                + COEFF_DECOMP * loss_decomp + COEFF_TV * Gradient_loss)
        loss.backward()
        _clip(state.encoder, state.decoder)
        state.optimizers[0].step()
        state.optimizers[1].step()
        return {"loss": loss.item(), "phase": 1.0, "mse_v": mse_loss_V.item(), "mse_i": mse_loss_I.item(),
                "decomp": loss_decomp.item(), "grad": Gradient_loss.item()}

    # Phase II
    feature_V_B, feature_V_D, _ = state.encoder(data_VIS)
    feature_I_B, feature_I_D, _ = state.encoder(data_IR)
    feature_F_B = state.base_fuse(feature_I_B + feature_V_B)
    feature_F_D = state.detail_fuse(feature_I_D + feature_V_D)
    data_Fuse, _ = state.decoder(None, feature_F_B, feature_F_D)

    cc_loss_B = cc(feature_V_B, feature_I_B)
    cc_loss_D = cc(feature_V_D, feature_I_D)
    loss_decomp = (cc_loss_D) ** 2 / (1.01 + cc_loss_B)
    fusionloss, loss_in, loss_grad = state.criteria_fusion(data_VIS, data_IR, data_Fuse)
    loss = fusionloss + COEFF_DECOMP * loss_decomp
    loss.backward()
    _clip(*state.modules_)
    for optimizer in state.optimizers:
        optimizer.step()
    return {"loss": loss.item(), "phase": 2.0, "fusion": fusionloss.item(), "in": loss_in.item(),
            "grad_fuse": loss_grad.item(), "decomp": loss_decomp.item()}


def on_epoch_end(state, epoch, args):
    state.schedulers[0].step()
    state.schedulers[1].step()
    if epoch > state.epoch_gap:
        state.schedulers[2].step()
        state.schedulers[3].step()
    for optimizer in state.optimizers:
        if optimizer.param_groups[0]["lr"] <= LR_FLOOR:
            optimizer.param_groups[0]["lr"] = LR_FLOOR


@torch.no_grad()
def fuse(state, batch, args):
    data_VIS, data_IR = batch["mri"], batch["func"]
    feature_V_B, feature_V_D, _ = state.encoder(data_VIS)
    feature_I_B, feature_I_D, _ = state.encoder(data_IR)
    feature_F_B = state.base_fuse(feature_V_B + feature_I_B)
    feature_F_D = state.detail_fuse(feature_V_D + feature_I_D)
    data_Fuse, _ = state.decoder(None, feature_F_B, feature_F_D)
    # test_MIF.py: per-image min-max (it fuses one image at a time)
    lo = data_Fuse.amin(dim=(1, 2, 3), keepdim=True)
    hi = data_Fuse.amax(dim=(1, 2, 3), keepdim=True)
    return (data_Fuse - lo) / (hi - lo).clamp_min(1e-8)


def state_dict(state):
    return {"encoder": state.encoder.state_dict(), "decoder": state.decoder.state_dict(),
            "base_fuse": state.base_fuse.state_dict(), "detail_fuse": state.detail_fuse.state_dict(),
            "optimizers": [o.state_dict() for o in state.optimizers],
            "schedulers": [s.state_dict() for s in state.schedulers]}


def load_state_dict(state, blob):
    state.encoder.load_state_dict(blob["encoder"])
    state.decoder.load_state_dict(blob["decoder"])
    state.base_fuse.load_state_dict(blob["base_fuse"])
    state.detail_fuse.load_state_dict(blob["detail_fuse"])
    for optimizer, sd in zip(state.optimizers, blob.get("optimizers", [])):
        optimizer.load_state_dict(sd)
    for scheduler, sd in zip(state.schedulers, blob.get("schedulers", [])):
        scheduler.load_state_dict(sd)
