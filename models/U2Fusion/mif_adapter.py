"""U2Fusion adapter for the shared MIF benchmark (mifkit), on the authors' PyTorch port
(``hanna-xu/U2Fusion-pytorch``; the original release is TensorFlow 1).

``train.py`` trains three tasks in sequence (multi-modal, multi-exposure, multi-focus)
with EWC between them; only the multi-modal task belongs to this benchmark, and its
setting is what is reproduced:

* ``FusionModel`` (``model.py``): the dense-block generator of ``generator.py`` and its
  ``content_loss(S1, S2, fused, c)`` -- SSIM and MSE to each source, weighted per pair by
  the softmax of the VGG16 feature-gradient information of each source divided by ``c``;
  ``c = 3200`` (``C_VALUES[0]``, task 1); ``loss = ssim + 20 * mse``;
* RMSprop(lr 1e-4, momentum 0.15, alpha 0.9), gradients clipped to +-50 by value;
* the order of the two sources is drawn at random for every batch
  (``collate_random_channels``); EWC is inactive (``LAM = 0``) and irrelevant to one task.

VGG16: ``model.py`` loads ``vgg16.pth``, converted (``convert_vgg.py``) from the
TF ``vgg16.npy`` -- the original Caffe VGG16 (BGR, mean-subtracted 0-255 input, which is
what ``vgg16.py`` feeds) -- that the README links on Baidu only.  The same Caffe weights
are taken from OpenMMLab's ``vgg16_caffe-292e1171.pth`` (``pretrained/``; conv1_1 ...
conv5_3 = its ``features.0`` ... ``features.28``), written to ``vgg16.pth`` under
``model.py``'s names.

Inputs as ``test.py``'s medical setting: S1 = the colour (functional) image's Y,
S2 = MRI, in [0,1]; output ``tanh / 2 + 0.5``.
Benchmark deviations: whole 256x256 images (release: 64x64 patches), batch 64 (18),
20 epochs (3 for task 1).
"""

import os

import torch

from model import FusionModel

NAME = "U2Fusion"
# batch: the protocol's 64, peak 45.4 GB (train.py: 18)
DEFAULTS = {"batch_size": 64, "lr": 1e-4}
C_MULTIMODAL = 3200          # train.py C_VALUES[0], task 1 (multi-modal)
CLIP_VALUE = 50
VGG_WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vgg16.pth")
HPARAMS = {
    "optimizer": "RMSprop(lr=1e-4, momentum=0.15, alpha=0.9), clip_grad_value 50 -- train.py",
    "loss": "content_loss with c=3200 (task 1): w1*(1-ssim1)+w2*(1-ssim2) + 20*(w1*mse1+w2*mse2), "
            "w = softmax(VGG16 feature-gradient information / c) -- model.py",
    "source_order": "S1/S2 order drawn at random per batch (collate_random_channels)",
    "vgg16": "Caffe VGG16 conv weights from OpenMMLab vgg16_caffe-292e1171.pth (the release's vgg16.npy is "
             "on Baidu only), converted to vgg16.pth",
    "inputs": "fusion: S1 = func Y, S2 = MRI -- test.py medical",
    "code_defaults_not_used": "batch 18, 3 epochs of task 1, 64x64 patches; tasks 2-3 and EWC "
                              "(benchmark protocol instead)",
}


class State:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer


def build(args, device, pair):
    model = FusionModel(vgg_weights_path=VGG_WEIGHTS).to(device)
    expected = torch.load(VGG_WEIGHTS, map_location=device, weights_only=True)
    if not torch.equal(model.vgg.conv5_3.weight, expected["conv5_3.weight"]):
        raise RuntimeError("VGG16 weights were not loaded")
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, momentum=0.15, alpha=0.9)
    return State(model, optimizer)


def describe(state):
    n = sum(p.numel() for p in state.model.generator.parameters())
    return f"U2Fusion generator: {n / 1e6:.3f}M parameters (+ frozen VGG16)"


def set_mode(state, train):
    state.model.generator.train(train)
    state.model.vgg.eval()


def train_step(state, batch, args, epoch):
    a, b = batch["func"], batch["mri"]
    if int(torch.randint(0, 2, (1,))) == 1:     # collate_random_channels
        a, b = b, a
    state.optimizer.zero_grad()
    fused = state.model(a, b)
    loss = state.model.content_loss(a, b, fused, C_MULTIMODAL)
    loss.backward()
    torch.nn.utils.clip_grad_value_(state.model.parameters(), CLIP_VALUE)
    state.optimizer.step()
    return {"loss": loss.item()}


@torch.no_grad()
def fuse(state, batch, args):
    return state.model(batch["func"], batch["mri"])


def state_dict(state):
    return {"generator": state.model.generator.state_dict(), "optimizer": state.optimizer.state_dict()}


def load_state_dict(state, blob):
    state.model.generator.load_state_dict(blob["generator"])
    if "optimizer" in blob:
        state.optimizer.load_state_dict(blob["optimizer"])
