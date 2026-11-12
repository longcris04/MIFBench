"""VDMUFusion adapter for the shared MIF benchmark (mifkit).

``Diffusion.py`` (DDPM, training losses, reverse process) and
``Condition_Noise_Predictor/UNet.py`` are the published ones, called directly
with the settings of ``config.json``: T = 2000, cosine betas, loss_scale 128,
``add_noise`` false, UNet(4 -> 1, 64 channels, 2 res-blocks, dropout 0.1,
mult [1, 2, 4, 8]).

``train.py`` trains one model on four fusion tasks, drawing one task per
epoch; only its medical branch belongs to this benchmark, and that is what is
reproduced:

    noise_loss, fusion_loss = diffusion.train_losses(model, MRI, OTH_Y, t, 128)
    loss = noise_loss + 10 * fusion_loss            t ~ U{0, ..., T-1} per sample

with AdamW(lr 2e-4) -- PyTorch's default weight decay 0.01, since ``train.py``
passes none -- and StepLR(1000 epochs, 0.99), which does not fire within 20
epochs.  Inputs as ``dataset.py``'s medical branch: sourceImg1 = MRI,
sourceImg2 = Y of the colour image, both mapped to [-1, 1].

Fusion: ``p_sample_loop``, the full T = 2000-step reverse process (no
strided sampler exists in the release), output mapped back to [0, 1].  It
starts from Gaussian noise; the pipeline fuses under a fixed seed, so every
epoch is sampled from the same noise.

``Diffusion.py`` hard-codes ``device = "cuda:1"`` at module level; it is
pointed at this job's GPU before use (module attribute, file untouched).
"""

import json
import os

import torch

import Diffusion as diffusion_module
from Condition_Noise_Predictor.UNet import NoisePred
from Diffusion import GaussianDiffusion

NAME = "VDMUFusion"
DEFAULTS = {"batch_size": 64, "lr": 2e-4, "fuse_batch_size": 96}
FUSION_LOSS_WEIGHT = 10   # train.py: loss_total = noise_loss + 10 * fusion_loss


def _config():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")) as handle:
        return json.load(handle)


CONFIG = _config()
HPARAMS = {
    "optimizer": f"AdamW(lr={CONFIG['optimizer']['init_lr']}, weight_decay=0.01 [torch default]) -- train.py",
    "scheduler": f"StepLR(step={CONFIG['optimizer']['StepLR_size']} epochs, gamma="
                 f"{CONFIG['optimizer']['StepLR_gamma']}) -- inactive within 20 epochs",
    "diffusion": {k: CONFIG["diffusion_model"][k] for k in ("T", "beta_schedule_type", "add_noise", "loss_scale")},
    "unet": CONFIG["Condition_Noise_Predictor"]["UNet"],
    "loss": "noise_loss + 10*fusion_loss (Diffusion.train_losses, train.py)",
    "inputs": "sourceImg1 = MRI, sourceImg2 = func Y, scaled to [-1,1] (dataset.py medical branch)",
    "fusion": "p_sample_loop, 2000 reverse steps, fixed noise seed",
    "code_defaults_not_used": f"batch {CONFIG['dataset']['train']['batch_size']}, "
                              f"{CONFIG['hyperParameter']['epochs']} epochs over four tasks (benchmark protocol)",
}


class State:
    def __init__(self, model, optimizer, scheduler, diffusion):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.diffusion = diffusion
        self.concat_type = CONFIG["Condition_Noise_Predictor"]["concat_type"]
        self.add_noise = CONFIG["diffusion_model"]["add_noise"]
        self.loss_scale = CONFIG["diffusion_model"]["loss_scale"]


def build(args, device, pair):
    diffusion_module.device = device          # Diffusion.py: device = "cuda:1"
    import Condition_Noise_Predictor.UNet as unet_module
    if hasattr(unet_module, "device"):
        unet_module.device = device
    u = CONFIG["Condition_Noise_Predictor"]["UNet"]
    model = NoisePred(u["in_channels"], u["out_channels"], u["model_channels"], u["num_res_blocks"],
                      u["dropout"], u["time_embed_dim_mult"], u["down_sample_mult"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=CONFIG["optimizer"]["StepLR_size"],
                                                gamma=CONFIG["optimizer"]["StepLR_gamma"])
    diffusion = GaussianDiffusion(CONFIG["diffusion_model"]["T"], CONFIG["diffusion_model"]["beta_schedule_type"])
    return State(model, optimizer, scheduler, diffusion)


def describe(state):
    n = sum(p.numel() for p in state.model.parameters())
    return f"VDMUFusion noise predictor: {n / 1e6:.3f}M parameters, T={state.diffusion.timesteps}"


def set_mode(state, train):
    state.model.train(train)


def _to_pm1(x):
    """dataset.py: transforms.Lambda(lambda t: (t * 2) - 1)"""
    return x * 2 - 1


def train_step(state, batch, args, epoch):
    state.optimizer.zero_grad()
    src1 = _to_pm1(batch["mri"])
    src2 = _to_pm1(batch["func"])
    t = torch.randint(0, state.diffusion.timesteps, (src1.shape[0],), device=src1.device).long()
    noise_loss, fusion_loss = state.diffusion.train_losses(state.model, src1, src2, t, state.loss_scale)
    loss_total = noise_loss + FUSION_LOSS_WEIGHT * fusion_loss
    loss_total.backward()
    state.optimizer.step()
    return {"loss": loss_total.item(), "noise": noise_loss.item(),
            "fusion": FUSION_LOSS_WEIGHT * fusion_loss.item()}


def on_epoch_end(state, epoch, args):
    state.scheduler.step()


@torch.no_grad()
def fuse(state, batch, args):
    src1 = _to_pm1(batch["mri"])
    src2 = _to_pm1(batch["func"])
    log_info = [1, 1, 0, 1]   # [step, valid_step_sum, num, generat_imgs_num]: progress printing only
    imgs = state.diffusion.p_sample_loop(state.model, src1, src2, state.concat_type, state.add_noise, log_info)
    return (imgs + 1) / 2     # utils.tensor2img: (t + 1) / 2


def state_dict(state):
    return {"model": state.model.state_dict(), "optimizer": state.optimizer.state_dict(),
            "scheduler": state.scheduler.state_dict()}


def load_state_dict(state, blob):
    state.model.load_state_dict(blob["model"])
    if "optimizer" in blob:
        state.optimizer.load_state_dict(blob["optimizer"])
    if "scheduler" in blob:
        state.scheduler.load_state_dict(blob["scheduler"])
