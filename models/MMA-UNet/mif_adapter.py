"""MMA-UNet adapter for the shared MIF benchmark (mifkit).

The release has the networks (``unet/unet_model.py`` VI-UNet, ``unet/unet_ir.py`` IR-UNet,
``unet/unetv2.py`` fusion model with its loss, ``unet/fs_loss.py``) and the inference
scripts (``predict.py``, ``Test2.py``) -- no training script.  The three training stages
are written from the paper (``papers_MIF/MMA-UNet.pdf``, Sec. III-D and IV-A):

  stage I-a  VI-UNet ``UNet(3, 3)``: reconstruct the VI image, MSE (Eq. 3);
  stage I-b  IR-UNet ``UNet(3, 3)``: reconstruct the IR image, MSE, its encoder guided by
             the frozen VI encoder's features through the SE blocks (Eq. 1);
  stage II   fusion model ``Unet`` on the frozen encoders' features, the release's
             ``Unet.forward_loss``: ``10 * L_det + 0.5 * L_ssim + L_mse`` (``fs_loss.Fusionloss``:
             kornia SSIM loss, MSE to (IR + VI) / 2, squared L1 of Sobel gradient to the max)
             -- alpha 0.5, beta 10 as the paper.

Optimiser (paper): AdamW, weight decay 5e-3, cosine annealing; learning rate 1e-3 for the
two auto-encoders, 1e-4 for the fusion model.  Not in the paper (epochs, batch, stage
lengths) -> the benchmark's rule, as APLFusion: 20 epochs in all (batch: the protocol's); the stages split
them 5 / 5 / 10 (the two auto-encoders share the first half, "the experimental settings
of VI-UNet and IR-UNet are same", the fusion model has the second half, as APLFusion's
two stages switch at epoch 10); the cosine schedule runs over each stage's epochs.
Epochs 1-10 fuse with the untrained fusion model (as CDDFuse's phase I epochs).

Inputs as ``util/loader.py``: grey images replicated to 3 channels in [0,1];
VI = MRI, IR = the functional image (Y).  Output as ``predict.py``: the fusion model's
3-channel logits, min-max normalised per image (``tensor2img(is_norm=True)``), channel 0.
The frozen networks run in eval mode (batch-norm running statistics) in later stages.

``fs_loss.py`` calls ``kornia.losses.ssim(img1, img2, 11)``, the SSIM loss function of
the kornia it was written for; in current kornia that name is a module and the function
is ``kornia.losses.ssim_loss`` (same loss, mean of (1 - SSIM) / 2): the module attribute
is pointed at it (file untouched).
"""

import kornia
import torch

import unet.fs_loss as fs_loss
from unet.unet_ir import UNet as UNet_ir
from unet.unet_model import UNet as UNet_vi
from unet.unetv2 import Unet as FusionNet

fs_loss.ssim = kornia.losses.ssim_loss

NAME = "MMA-UNet"
# batch: the protocol's 64, peak 27.4 GB over the three stages (not stated in the paper)
DEFAULTS = {"batch_size": 64, "lr": 1e-3}
WEIGHT_DECAY = 5e-3
FUSION_LR_FACTOR = 0.1          # 1e-3 -> 1e-4 for the fusion model
STAGE_FRACTIONS = (0.25, 0.25, 0.5)
HPARAMS = {
    "optimizer": "AdamW(weight_decay=5e-3), cosine annealing; lr 1e-3 (VI-UNet, IR-UNet), 1e-4 (fusion) -- "
                 "paper Sec. IV-A; no training code released",
    "stages": "epochs 1-5 VI-UNet (MSE), 6-10 IR-UNet (MSE, SE-guided by frozen VI encoder), 11-20 fusion "
              "(frozen encoders) -- split not in the paper: benchmark rule (as APLFusion, switch at epoch 10)",
    "loss_fusion": "Unet.forward_loss: 10*L1(sobel F, max(sobel ir, sobel vi))^2 + 0.5*(kornia ssim loss vs vi + vs ir)"
                   " + MSE(F, (ir+vi)/2) -- unetv2.py / fs_loss.py, alpha 0.5 beta 10 (paper)",
    "inputs": "VI = MRI, IR = func Y, grey replicated to 3 channels -- util/loader.py",
    "fusion": "logits min-max normalised per image (3 channels), channel 0 -- predict.py",
}


class State:
    def __init__(self, args, device):
        self.vi = UNet_vi(3, 3).to(device)
        self.ir = UNet_ir(3, 3).to(device)
        self.fusion = FusionNet(True).to(device)
        e = args.epochs
        n_vi = max(1, round(e * STAGE_FRACTIONS[0]))
        n_ir = max(1, round(e * STAGE_FRACTIONS[1]))
        self.bounds = (n_vi, n_vi + n_ir)          # last epoch of stage I-a and of stage I-b
        n_fu = max(1, e - n_vi - n_ir)
        self.optimizers, self.schedulers = [], []
        for net, lr, n in ((self.vi, args.lr, n_vi), (self.ir, args.lr, n_ir),
                           (self.fusion, args.lr * FUSION_LR_FACTOR, n_fu)):
            opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
            self.optimizers.append(opt)
            self.schedulers.append(torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n))
        self.stage = 0

    def stage_of(self, epoch):
        return 0 if epoch <= self.bounds[0] else (1 if epoch <= self.bounds[1] else 2)


def build(args, device, pair):
    return State(args, device)


def describe(state):
    n = [sum(p.numel() for p in m.parameters()) for m in (state.vi, state.ir, state.fusion)]
    return (f"MMA-UNet: VI-UNet {n[0] / 1e6:.2f}M, IR-UNet {n[1] / 1e6:.2f}M, fusion {n[2] / 1e6:.2f}M parameters; "
            f"stages end at epochs {state.bounds[0]}, {state.bounds[1]}, then fusion")


def current_lr(state):
    return state.optimizers[state.stage].param_groups[0]["lr"]


def on_epoch_start(state, epoch, args):
    state.stage = state.stage_of(epoch)


def set_mode(state, train):
    # only the network of the current stage trains; the frozen ones stay in eval mode
    for k, net in enumerate((state.vi, state.ir, state.fusion)):
        net.train(train and k == state.stage)


def _rgb(x):
    return x.expand(-1, 3, -1, -1)


def train_step(state, batch, args, epoch):
    vis, ir = _rgb(batch["mri"]), _rgb(batch["func"])
    stage = state.stage
    opt = state.optimizers[stage]
    opt.zero_grad()
    if stage == 0:
        rec, _ = state.vi(vis)
        loss = torch.mean((rec - vis) ** 2)
        out = {"loss": loss.item(), "stage": 1.0, "mse_vi": loss.item()}
    elif stage == 1:
        with torch.no_grad():
            _, fea_vis = state.vi(vis)
        rec, _ = state.ir(ir, fea_vis)
        loss = torch.mean((rec - ir) ** 2)
        out = {"loss": loss.item(), "stage": 2.0, "mse_ir": loss.item()}
    else:
        with torch.no_grad():
            _, fea_vis = state.vi(vis)
            _, fea_ir = state.ir(ir, fea_vis)
        _, loss = state.fusion(fea_ir, fea_vis, ir, vis)
        out = {"loss": loss.item(), "stage": 3.0}
    loss.backward()
    opt.step()
    return out


def on_epoch_end(state, epoch, args):
    state.schedulers[state.stage].step()


@torch.no_grad()
def fuse(state, batch, args):
    vis, ir = _rgb(batch["mri"]), _rgb(batch["func"])
    _, fea_vis = state.vi(vis)
    _, fea_ir = state.ir(ir, fea_vis)
    logits, _ = state.fusion(fea_ir, fea_vis, ir, vis)
    lo = logits.amin(dim=(1, 2, 3), keepdim=True)
    hi = logits.amax(dim=(1, 2, 3), keepdim=True)
    return ((logits - lo) / (hi - lo).clamp_min(1e-8))[:, :1]


def state_dict(state):
    return {"vi": state.vi.state_dict(), "ir": state.ir.state_dict(), "fusion": state.fusion.state_dict(),
            "optimizers": [o.state_dict() for o in state.optimizers],
            "schedulers": [s.state_dict() for s in state.schedulers]}


def load_state_dict(state, blob):
    state.vi.load_state_dict(blob["vi"])
    state.ir.load_state_dict(blob["ir"])
    state.fusion.load_state_dict(blob["fusion"])
    for opt, sd in zip(state.optimizers, blob.get("optimizers", [])):
        opt.load_state_dict(sd)
    for sch, sd in zip(state.schedulers, blob.get("schedulers", [])):
        sch.load_state_dict(sd)
