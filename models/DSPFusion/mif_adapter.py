"""DSPFusion adapter for the shared MIF benchmark (mifkit).

Networks, losses and optimiser are the release's (``options/train/DSPF_S1.yml``,
``DSPF_S2.yml``, ``DSPF/models/DSPF_S1_model.py``, ``DSPF_S2_model.py``):

* Stage I -- the fusion network ``Transformer_DSPF`` (dim 32, blocks [2,2,4,4], heads
  [1,1,2,2]), the degradation-prior encoder ``degradation_encoder_gelu`` and the
  semantic-prior encoder ``semantic_encoder_gelu`` (on the high-quality pair) are trained
  together with ``fusion_loss`` (basicsr ``basic_loss.py``: 2 SSIM + 15 max-intensity +
  consistency + 20 colour + 10 max-gradient), gradients clipped to norm 0.01;
* Stage II -- those three are frozen and the semantic-prior diffusion model (``denoising``
  in ``ldm/cddpm.DDPM``, T = 10, linear betas 0.1-0.99, x0-parameterised) learns to
  restore the high-quality semantic prior from the low-quality inputs:
  ``fusion_loss + 10 * L1(prior_gt, prior)``, gradients clipped to 0.01;
* AdamW(lr 2e-4, weight decay 1e-4, betas (0.9, 0.999)) per stage, and the
  ``CosineAnnealingRestartCyclicLR`` shape of the configs: lr constant for the first 60 %
  of a stage (periods 300k + 200k with eta_min 2e-4, then 1e-6), half-cosine to 1e-6 over
  the last 40 %, per iteration;
* the two stages have the same length (paper: 50k iterations each; configs: 500k each):
  epochs 1-10 and 11-20.  Fusion uses the stage's own test path: Stage I -- the semantic
  prior encoded from the pair; Stage II -- the prior sampled by the diffusion model.

What medical images cannot provide: DSPFusion is trained on degraded / restored pairs
(``lq`` / ``gt``) with a contrastive loss over degradation types (positives of the same
degradation from other scenes, negatives of other degradations).  The benchmark's images
are clean and have no degradation labels: ``lq = gt`` = the source images, and the
contrastive loss is left out (its configs list ``contrastive_opt``, weight 1).

Inputs: 3-channel images in [0,1] (grey replicated); "ir" = the functional image (Y),
"vi" = MRI.  Output: the network's RGB fusion clipped to [0,1] (``tensor2img``), its
BT.601 luma is the fused Y.
"""

import math

import torch
import torch.nn.functional as F

from basicsr.losses.basic_loss import fusion_loss
from DSPF.archs.degradation_encoder_arch import degradation_encoder_gelu
from DSPF.archs.denoising_arch import denoising
from DSPF.archs.semantic_encoder_arch import semantic_encoder_gelu
from DSPF.archs.Transformer_DSPF_arch import Transformer_DSPF
from ldm.cddpm import DDPM
from mifkit.folds import split_stems

NAME = "DSPFusion"
# at 256x256 batch 16 and above run out of memory on an
# 80 GB A100; 12 peaks at 67.7 GB over both stages (configs: 4 / 3 per GPU)
DEFAULTS = {"batch_size": 12, "lr": 2e-4}
NETWORK_G = dict(inp_channels=3, out_channels=3, dim=32, num_blocks=[2, 2, 4, 4], num_refinement_blocks=1,
                 heads=[1, 1, 2, 2], ffn_expansion_factor=2.66, bias=False, LayerNorm_type="WithBias",
                 dual_pixel_task=False, embed_dim=64, type_embed_dim=32, group=4, with_contra=False,
                 with_SFP=False)
NETWORK_DP = dict(in_chans=3, embed_dim=32, block_num=4, group=4, stage=1, patch_expansion=0.5, channel_expansion=4)
NETWORK_SP = dict(in_chans=6, embed_dim=64, block_num=6, group=4, stage=1, patch_expansion=0.5, channel_expansion=4)
NETWORK_DM = dict(in_channel=256, out_channel=256, inner_channel=512, block_num=6, group=4, patch_expansion=0.5,
                  channel_expansion=2, dp_channel=128)
DIFFUSION = dict(timesteps=10, linear_start=0.1, linear_end=0.99)
WEIGHT_DECAY, BETAS = 1e-4, (0.9, 0.999)
ETA_MIN = 1e-6
CONSTANT_FRACTION = 0.6          # periods [300000, 200000]: constant (eta_min = lr), then cosine
CLIP = 0.01
DIFF_WEIGHT = 10.0
HPARAMS = {
    "stages": "epochs 1-10 Stage I (fusion net + degradation/semantic encoders), 11-20 Stage II (prior "
              "diffusion, the rest frozen) -- equal stages as paper / configs",
    "optimizer": "AdamW(lr=2e-4, weight_decay=1e-4, betas=(0.9,0.999)) per stage; lr constant for 60 % of a "
                 "stage, cosine to 1e-6 over the last 40 % (CosineAnnealingRestartCyclicLR periods 300k+200k)",
    "loss_stage1": "fusion_loss (2 ssim + 15 max + 1 consist + 20 color + 10 grad), clip 0.01",
    "loss_stage2": "fusion_loss + 10*L1(prior_gt, prior), clip 0.01",
    "not_reproduced": "contrastive degradation loss (no degradations / labels in clean medical images); "
                      "lq = gt = source images",
    "networks": {"g": NETWORK_G, "dp": NETWORK_DP, "sp": NETWORK_SP, "dm": NETWORK_DM, "diffusion": DIFFUSION},
    "inputs": "ir = func Y, vi = MRI, grey replicated to 3 channels",
    "fusion": "stage's test path; RGB output clipped to [0,1], BT.601 luma",
    "code_defaults_not_used": "500k iterations per stage, batch 4/3 on 224/256 crops (benchmark protocol instead)",
}


class State:
    def __init__(self, args, device):
        self.net_g = Transformer_DSPF(**NETWORK_G).to(device)
        self.net_dp = degradation_encoder_gelu(**NETWORK_DP).to(device)
        self.net_sp = semantic_encoder_gelu(**NETWORK_SP).to(device)
        self.net_dm = denoising(**NETWORK_DM).to(device)
        self.diffusion = DDPM(denoise=self.net_dm, n_feats=NETWORK_G["embed_dim"], group=NETWORK_G["group"],
                              **DIFFUSION).to(device)
        self.stage1_params = (list(self.net_g.parameters()) + list(self.net_dp.parameters())
                              + list(self.net_sp.parameters()))
        self.opt1 = torch.optim.AdamW(self.stage1_params, lr=args.lr, weight_decay=WEIGHT_DECAY, betas=BETAS)
        self.opt2 = torch.optim.AdamW(self.net_dm.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY, betas=BETAS)
        self.criterion = fusion_loss().to(device)
        self.stage1_epochs = max(1, args.epochs // 2)
        self.lr_max = args.lr
        self.steps_per_epoch = 1
        self.stage = 1
        self.iteration = 0          # within the current epoch

    def stage_of(self, epoch):
        return 1 if epoch <= self.stage1_epochs else 2

    @property
    def optimizer(self):
        return self.opt1 if self.stage == 1 else self.opt2


def build(args, device, pair):
    return State(args, device)


def prepare(state, args, data, device):
    n = len(split_stems(args.pair, args.fold, "train")) * max(1, args.repeat or 1)
    state.steps_per_epoch = n // args.batch_size if n % args.batch_size == 1 else math.ceil(n / args.batch_size)


def describe(state):
    def count(*ms):
        return sum(p.numel() for m in ms for p in m.parameters()) / 1e6
    return (f"DSPFusion: fusion net {count(state.net_g):.2f}M, priors {count(state.net_dp, state.net_sp):.2f}M, "
            f"diffusion {count(state.net_dm):.2f}M parameters; Stage II from epoch {state.stage1_epochs + 1}")


def current_lr(state):
    return state.optimizer.param_groups[0]["lr"]


def _set_lr(state, epoch, args):
    first = 1 if state.stage == 1 else state.stage1_epochs + 1
    length = state.stage1_epochs if state.stage == 1 else args.epochs - state.stage1_epochs
    progress = ((epoch - first) * state.steps_per_epoch + state.iteration) / max(1, length * state.steps_per_epoch)
    if progress < CONSTANT_FRACTION:
        lr = state.lr_max
    else:
        p = (progress - CONSTANT_FRACTION) / (1 - CONSTANT_FRACTION)
        lr = ETA_MIN + 0.5 * (state.lr_max - ETA_MIN) * (1 + math.cos(math.pi * p))
    for group in state.optimizer.param_groups:
        group["lr"] = lr


def on_epoch_start(state, epoch, args):
    state.stage = state.stage_of(epoch)
    state.iteration = 0
    frozen = state.stage == 2
    for p in state.stage1_params:
        p.requires_grad_(not frozen)


def set_mode(state, train):
    if state.stage == 1:
        for net in (state.net_g, state.net_dp, state.net_sp):
            net.train(train)
        state.net_dm.eval()
        state.diffusion.eval()
    else:      # DSPF_S2.init_training_settings: net_g / net_dp / net_sp eval, net_dm trains
        for net in (state.net_g, state.net_dp, state.net_sp):
            net.eval()
        state.net_dm.train(train)
        state.diffusion.train(train)


def _rgb(x):
    return x.expand(-1, 3, -1, -1)


def train_step(state, batch, args, epoch):
    _set_lr(state, epoch, args)
    ir, vi = _rgb(batch["func"]), _rgb(batch["mri"])
    opt = state.optimizer
    opt.zero_grad()
    if state.stage == 1:
        semantic_prior = state.net_sp(torch.cat([ir, vi], 1))
        deg_ir, _ = state.net_dp(ir)
        deg_vi, _ = state.net_dp(vi)
        out = state.net_g(ir, vi, deg_ir, deg_vi, semantic_prior)["fusion"]
        losses, _ = state.criterion(out, ir, vi)
        loss = losses["loss_fusion"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(state.stage1_params, CLIP)
        stats = {"loss": loss.item(), "stage": 1.0}
    else:
        with torch.no_grad():
            prior_lq = state.net_sp(torch.cat([ir, vi], 1))
            prior_gt = prior_lq                                   # lq = gt for clean images
            deg_ir, _ = state.net_dp(ir)
            deg_vi, _ = state.net_dp(vi)
        condition = torch.cat([prior_lq, deg_ir, deg_vi], dim=-1)
        prior, _ = state.diffusion(ir, vi, condition, prior_gt)
        out = state.net_g(ir, vi, deg_ir, deg_vi, prior)["fusion"]
        losses, _ = state.criterion(out, ir, vi)
        l_diff = DIFF_WEIGHT * F.l1_loss(prior, prior_gt)
        loss = losses["loss_fusion"] + l_diff
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(state.diffusion.parameters()), CLIP)
        stats = {"loss": loss.item(), "stage": 2.0, "diff": l_diff.item()}
    opt.step()
    state.iteration += 1
    for k in ("loss_ssim", "loss_intensity_max", "loss_intensity_consist", "loss_color", "loss_grad"):
        stats[k] = float(losses[k])
    return stats


@torch.no_grad()
def fuse(state, batch, args):
    ir, vi = _rgb(batch["func"]), _rgb(batch["mri"])
    deg_ir, _ = state.net_dp(ir)
    deg_vi, _ = state.net_dp(vi)
    prior = state.net_sp(torch.cat([ir, vi], 1))
    if state.stage == 2:     # DSPF_S2.test: the prior sampled by the diffusion model
        prior = state.diffusion(ir, vi, torch.cat([prior, deg_ir, deg_vi], dim=-1))
    out = state.net_g(ir, vi, deg_ir, deg_vi, prior)["fusion"].clamp(0, 1)
    return 0.299 * out[:, 0:1] + 0.587 * out[:, 1:2] + 0.114 * out[:, 2:3]


def state_dict(state):
    return {"net_g": state.net_g.state_dict(), "net_dp": state.net_dp.state_dict(),
            "net_sp": state.net_sp.state_dict(), "diffusion": state.diffusion.state_dict(),
            "opt1": state.opt1.state_dict(), "opt2": state.opt2.state_dict(), "stage": state.stage}


def load_state_dict(state, blob):
    state.net_g.load_state_dict(blob["net_g"])
    state.net_dp.load_state_dict(blob["net_dp"])
    state.net_sp.load_state_dict(blob["net_sp"])
    state.diffusion.load_state_dict(blob["diffusion"])
    if "opt1" in blob:
        state.opt1.load_state_dict(blob["opt1"])
        state.opt2.load_state_dict(blob["opt2"])
    # the stage of the saved epoch: fusing from a checkpoint (time_inference, ...) takes its path
    state.stage = int(blob.get("stage", 1))
    for p in state.stage1_params:
        p.requires_grad_(state.stage == 1)
