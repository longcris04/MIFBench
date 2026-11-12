"""TUFusion adapter for the shared MIF benchmark (mifkit).

TUFusion is an auto-encoder: ``train_tufusion.py`` trains it to reconstruct
single images, and ``test_image.py`` fuses by combining the two encodings with
a training-free strategy.  Both halves are kept:

* training -- every image of a pair (functional Y and MRI) is a reconstruction
  target, with the published loss ``pixel + ssim_weight*(1 - MS-SSIM) +
  grd_weight*grad`` at the weights ``main()`` selects (i = 0, j = 2 -> 1 and 1),
  Adam(lr 1e-4, ``args_fusion.lr``), no weight decay;
* fusion -- ``test_image.py``'s ``_generate_fusion_image``: encoder on both,
  ``model.fusion2(en_r, en_v, 'attention_max')``, decoder.

``fusion2`` builds a new ``AxialAttentionTransformer`` with random weights at
every call (that is the published code); it is moved to the GPU and seeded
the same way at every epoch (the pipeline fuses under a fixed RNG), so the
strategy is identical from one epoch to the next.

Two device fixes, values unchanged: ``utils.gradient`` creates its Laplacian
conv on the CPU at every call (the published run was CPU-only,
``args_fusion.cuda = 0``), so the same kernel is applied here with
``F.conv2d`` on the batch's device; and the axial transformer of ``fusion2``
is created on the CPU, so it is constructed here and moved to the GPU.

Scale: ``train_tufusion.py`` reads training images through
``skimage.transform.resize`` (floats in [0,1]), and that is what the network
is trained and run on here.  (``test_image.py`` feeds unresized uint8 images,
i.e. [0,255], to the network trained on [0,1]; that mismatch is not reproduced.)

The ViT branch is fixed to 256x256 inputs (``image_size=256``, ``h=16``).
"""

import torch
import torch.nn.functional as F

import fusion_strategy
import pytorch_msssim
from net import TUFusion_net

NAME = "TUFusion"
# Batch 8, not the protocol's 64: at 64 (3-5 steps per epoch) the
# 20-epoch run ends with blocky 16x16 ViT-patch images (fold 1 Qabf 0.03-0.09). Study on fold 1,
# epoch 20, Qabf CT-MRI / PET-MRI: batch 64 0.03/0.07, 32 0.08/0.15, 16 0.28/0.40, 8 0.35/0.42,
# 4 (args_fusion.py) 0.36/0.42 -- 8 is the largest batch that gives clean images.
DEFAULTS = {"batch_size": 8, "lr": 1e-4}
SSIM_WEIGHT = 1.0   # args_fusion.args.ssim_weight[0]  (main(): i = 0)
GRD_WEIGHT = 1.0    # args_fusion.args.grd_weight[2]   (main(): j = 2)
P_TYPE = "attention_max"   # test_image.py fusion_type[0]
HPARAMS = {
    "optimizer": "Adam(lr=1e-4) -- args_fusion.lr; no weight decay",
    "loss": "mse + 1*(1 - msssim(normalize=True)) + 1*mse(laplacian) (train_tufusion.py, i=0, j=2), "
            "averaged over the decoder outputs; both images of a pair are reconstruction targets",
    "fusion": "encoder -> fusion2(en_func, en_mri, 'attention_max') -> decoder (test_image.py)",
    "batch_size": "8 instead of the protocol's 64: at 64 the outputs stay blocky after 20 epochs "
                  "(batch-size study on fold 1); args_fusion.py uses 4",
    "code_defaults_not_used": "batch 4, 4 epochs, MS-COCO (benchmark protocol and data instead)",
}

_LAPLACIAN = torch.tensor([[-1., -1., -1.], [-1., 8., -1.], [-1., -1., -1.]]).view(1, 1, 3, 3)


def _gradient(x):
    """utils.gradient: 3x3 Laplacian (8-neighbour), no padding."""
    return F.conv2d(x, _LAPLACIAN.to(x.device, x.dtype), padding=0)


def _attention_mechanism_fusion(tensor1, tensor2, p_type):
    """fusion_strategy.attention_mechanism_fusion with the axial transformer on the tensors' device."""
    se = fusion_strategy.AxialAttentionTransformer(dim=65, depth=2, reversible=True).to(tensor1.device)
    tensor3 = se(tensor1)
    tensor4 = se(tensor2)
    tensor5 = (tensor3 + 1.0 * tensor4) * 0.6
    tensor6 = fusion_strategy.attention_fusion_channel(tensor3, tensor4, p_type)
    tensor7 = tensor6[1]
    tensor8 = tensor6[2]
    tensor9 = (tensor7 + tensor8) * 1.25
    return (tensor5 + tensor9) / 2


class State:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.mse = torch.nn.MSELoss()


def build(args, device, pair):
    model = TUFusion_net(1, 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    return State(model, optimizer)


def describe(state):
    n = sum(p.numel() for p in state.model.parameters())
    return f"TUFusion: {n / 1e6:.3f}M parameters"


def set_mode(state, train):
    state.model.train(train)


def _reconstruction_terms(state, img):
    en = state.model.encoder(img)
    outputs = state.model.decoder(en)
    x = img.detach()
    ssim_loss_value = 0.
    pixel_loss_value = 0.
    gra_loss_value = 0.
    for output in outputs:
        pixel_loss_value += state.mse(output, x)
        ssim_loss_value += (1 - pytorch_msssim.msssim(output, x, normalize=True))
        gra_loss_value += state.mse(_gradient(output), _gradient(x))
    n = len(outputs)
    return pixel_loss_value / n, ssim_loss_value / n, gra_loss_value / n


def train_step(state, batch, args, epoch):
    state.optimizer.zero_grad()
    # one batch of training images = the functional and the MRI images of the batch's pairs
    img = torch.cat([batch["func"], batch["mri"]], dim=0)
    pixel, ssim, grad = _reconstruction_terms(state, img)
    total_loss = pixel + SSIM_WEIGHT * ssim + GRD_WEIGHT * grad
    total_loss.backward()
    state.optimizer.step()
    return {"loss": total_loss.item(), "pixel": pixel.item(), "ssim": ssim.item(), "grad": grad.item()}


@torch.no_grad()
def fuse(state, batch, args):
    en_r = state.model.encoder(batch["func"])
    en_v = state.model.encoder(batch["mri"])
    f = [_attention_mechanism_fusion(en_r[0], en_v[0], P_TYPE)]   # model.fusion2(en_r, en_v, p_type)
    return state.model.decoder(f)[0]


def state_dict(state):
    return {"model": state.model.state_dict(), "optimizer": state.optimizer.state_dict()}


def load_state_dict(state, blob):
    state.model.load_state_dict(blob["model"])
    if "optimizer" in blob:
        state.optimizer.load_state_dict(blob["optimizer"])
