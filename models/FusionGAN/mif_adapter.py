"""FusionGAN adapter for the shared MIF benchmark (mifkit).

The release is TensorFlow 1 (``model.py``, ``main.py``, ``utils.py``); the benchmark runs
the PyTorch port in ``fusiongan_torch.py`` (same layers, spectral norm, initialisation
and losses -- see its docstring).  Training as ``main.py`` / ``CGAN.train``:

* Adam(lr 1e-4) for the generator and for the discriminator (TF defaults: betas
  (0.9, 0.999), eps 1e-8);
* every step: two discriminator updates, then one generator update;
* D loss: ``mean((D(vi) - U(0.7, 1.2))^2) + mean((D(F) - U(0, 0.3))^2)`` (label noise
  drawn anew at every evaluation);
* G loss: ``mean((D(F) - U(0.7, 1.2))^2) + 100 * (mean((F - ir)^2)
  + 5 * mean((lap(F) - lap(vi))^2))``, ``lap`` the 3x3 Laplacian of ``utils.gradient``;
* images scaled to [-1, 1] (``(img - 127.5) / 127.5``); the generator sees the pair
  edge-padded by 6 pixels and outputs the unpadded size; output ``F * 127.5 + 127.5``.

Role mapping: FusionGAN's "ir" (intensity target) is the functional modality
(CT / PET-Y / SPECT-Y), its "vi" (gradient target, the discriminator's real images) is
MRI -- as for DATFuse / YDTR.

Benchmark deviations: whole 256x256 images instead of 132/120 patches (the
discriminator's last layer has 15*15*256 inputs instead of 6*6*256); batch 64 and
20 epochs instead of 32 and 10.  ``test_one_image.py`` rebuilds the generator from the
raw weights without spectral normalisation (an inconsistency with training); the
benchmark fuses with the generator as trained (spectral-normalised weights), one image
at a time as ``test_one_image.py`` does, so that batch norm sees one image's statistics.
"""

import torch

from fusiongan_torch import Discriminator, Generator, d_loss, g_loss

NAME = "FusionGAN"
# batch: the protocol's 64, peak 33.3 GB (main.py: 32)
DEFAULTS = {"batch_size": 64, "lr": 1e-4}
D_STEPS = 2   # CGAN.train: `for i in range(2)` discriminator updates per generator update
IMAGE_SIZE = 256
HPARAMS = {
    "framework": "PyTorch port of the TF1 release (fusiongan_torch.py)",
    "optimizer": "Adam(lr=1e-4) for G and D, TF defaults betas (0.9, 0.999) eps 1e-8 -- main.py",
    "schedule": "2 D steps + 1 G step per batch -- CGAN.train",
    "loss_d": "mean((D(vi)-U(0.7,1.2))^2) + mean((D(F)-U(0,0.3))^2)",
    "loss_g": "mean((D(F)-U(0.7,1.2))^2) + 100*(mean((F-ir)^2) + 5*mean((lap F - lap vi)^2))",
    "inputs": "ir = func Y, vi = MRI, scaled to [-1,1]; generator input edge-padded by 6",
    "fusion": "generator with spectral-normalised weights, one image at a time (batch-norm statistics "
              "of that image, as test_one_image.py)",
    "code_defaults_not_used": "batch 32, 10 epochs, 132/120 patches with stride 14 (benchmark protocol instead)",
}


class State:
    def __init__(self, gen, disc, opt_g, opt_d):
        self.gen = gen
        self.disc = disc
        self.opt_g = opt_g
        self.opt_d = opt_d
        self.optimizer = opt_g


def build(args, device, pair):
    gen = Generator().to(device)
    disc = Discriminator(IMAGE_SIZE).to(device)
    opt_g = torch.optim.Adam(gen.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    opt_d = torch.optim.Adam(disc.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    return State(gen, disc, opt_g, opt_d)


def describe(state):
    ng = sum(p.numel() for p in state.gen.parameters())
    nd = sum(p.numel() for p in state.disc.parameters())
    return f"FusionGAN (PyTorch port): generator {ng / 1e6:.3f}M, discriminator {nd / 1e6:.3f}M parameters"


def set_mode(state, train):
    state.gen.train(train)
    state.disc.train(train)


def _pm1(x):
    return x * 2 - 1


def train_step(state, batch, args, epoch):
    ir, vi = _pm1(batch["func"]), _pm1(batch["mri"])
    loss_d = None
    for _ in range(D_STEPS):
        with torch.no_grad():
            fused = state.gen(ir, vi, update=True)
        # D(F) with update_collection='NO_OPS' first, then D(vi), which updates u
        d_fake = state.disc(fused, update=False)
        d_real = state.disc(vi, update=True)
        loss_d = d_loss(d_real, d_fake)
        state.opt_d.zero_grad()
        loss_d.backward()
        state.opt_d.step()

    fused = state.gen(ir, vi, update=True)
    d_fake = state.disc(fused, update=False)
    loss_g, adv, content = g_loss(d_fake, fused, ir, vi)
    state.opt_g.zero_grad()
    loss_g.backward(inputs=list(state.gen.parameters()))
    state.opt_g.step()
    return {"loss": loss_g.item(), "g_adv": adv.item(), "g_content": content.item(), "d": loss_d.item()}


@torch.no_grad()
def fuse(state, batch, args):
    ir, vi = _pm1(batch["func"]), _pm1(batch["mri"])
    out = torch.cat([state.gen(ir[k:k + 1], vi[k:k + 1], update=False) for k in range(ir.shape[0])])
    return out * 0.5 + 0.5      # result * 127.5 + 127.5, on the [0, 1] scale


def state_dict(state):
    return {"gen": state.gen.state_dict(), "disc": state.disc.state_dict(),
            "opt_g": state.opt_g.state_dict(), "opt_d": state.opt_d.state_dict()}


def load_state_dict(state, blob):
    state.gen.load_state_dict(blob["gen"])
    state.disc.load_state_dict(blob["disc"])
    if "opt_g" in blob:
        state.opt_g.load_state_dict(blob["opt_g"])
        state.opt_d.load_state_dict(blob["opt_d"])
