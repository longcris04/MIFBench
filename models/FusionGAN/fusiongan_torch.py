"""PyTorch port of FusionGAN's TensorFlow 1 graph (``model.py`` / ``utils.py``), for the
MIF benchmark (the release is TF1-only).  Written to match the TF code layer by layer:

* generator ``fusion_model``: five VALID convolutions 5x5 (2->256), 5x5 (256->128),
  3x3 (128->64), 3x3 (64->32), 1x1 (32->1); the first four followed by
  ``tf.contrib.layers.batch_norm(decay=0.9, epsilon=1e-5, scale=True)`` and ``lrelu``
  (leak 0.2), the last by tanh.  The input is the IR/VI pair padded by 6 pixels, so the
  output has the size of the unpadded images (``image_size`` 132 -> ``label_size`` 120);
* discriminator: four 3x3 stride-2 VALID convolutions (1->32->64->128->256), batch norm on
  the last three, ``lrelu``, then a linear layer to one logit (``6*6*256`` inputs for
  120x120 patches; ``15*15*256`` for the benchmark's 256x256 images);
* every weight goes through ``weights_spectral_norm``: one power iteration per call from
  the stored vector ``u`` (shape ``[1, out]``), ``W / sigma`` with
  ``sigma = v W u^T``; ``u`` is updated by the generator's calls and the discriminator's
  call on real images, not by its call on fused images (``update_collection='NO_OPS'``);
  as in the TF graph no gradient is stopped through ``u``/``v``;
* initialisation: weights ``truncated_normal(stddev=1e-3)``, biases 0, ``u``
  ``truncated_normal(stddev=1)``;
* batch norm always normalises with the statistics of the batch it sees: the TF code
  never passes ``is_training=False`` (training or ``test_one_image.py``), so its moving
  averages are never used.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def truncated_normal_(tensor, std):
    """tf.truncated_normal_initializer: normal(0, std) redrawn outside +-2 std."""
    with torch.no_grad():
        return nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-2 * std, b=2 * std)


def _l2_norm(x, eps=1e-12):
    """utils.l2_norm: x / (||x||_2 + eps), over the whole tensor."""
    return x / (x.pow(2).sum().sqrt() + eps)


class SpectralNormWeight(nn.Module):
    """``weights_spectral_norm`` around one weight; ``out_dim`` is the TF last axis."""

    def __init__(self, weight_shape, out_dim, std=1e-3):
        super().__init__()
        self.weight = nn.Parameter(truncated_normal_(torch.empty(weight_shape), std))
        self.register_buffer("u", truncated_normal_(torch.empty(1, out_dim), 1.0))

    def forward(self, update):
        w = self.weight
        out = w.shape[0]
        w_mat = w.reshape(out, -1)                    # = TF's reshape(w, [-1, out]) transposed
        u = self.u.clone()                            # the buffer may be updated before backward
        v_hat = _l2_norm(u @ w_mat)                   # [1, K]
        u_hat = _l2_norm(v_hat @ w_mat.t())           # [1, out]
        sigma = (u_hat @ w_mat @ v_hat.t()).squeeze()
        if update:
            self.u.copy_(u_hat.detach())
        return w / sigma


class SNConv2d(nn.Module):
    def __init__(self, cin, cout, k, stride=1):
        super().__init__()
        self.stride = stride
        self.w = SpectralNormWeight((cout, cin, k, k), cout)
        self.bias = nn.Parameter(torch.zeros(cout))

    def forward(self, x, update):
        return F.conv2d(x, self.w(update), self.bias, stride=self.stride)   # VALID


class SNLinear(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.w = SpectralNormWeight((cout, cin), cout)
        self.bias = nn.Parameter(torch.zeros(cout))

    def forward(self, x, update):
        return F.linear(x, self.w(update), self.bias)


def batch_norm(channels):
    # tf.contrib.layers.batch_norm(..., epsilon=1e-5, scale=True), always on batch statistics
    return nn.BatchNorm2d(channels, eps=1e-5, affine=True, track_running_stats=False)


def lrelu(x):
    return torch.maximum(x, 0.2 * x)


class Generator(nn.Module):
    """``CGAN.fusion_model``"""

    PAD = 6   # (image_size - label_size) / 2

    def __init__(self):
        super().__init__()
        spec = [(2, 256, 5), (256, 128, 5), (128, 64, 3), (64, 32, 3), (32, 1, 1)]
        self.convs = nn.ModuleList([SNConv2d(cin, cout, k) for cin, cout, k in spec])
        self.bns = nn.ModuleList([batch_norm(cout) for _, cout, _ in spec[:4]])

    def forward(self, ir, vi, update=True):
        """ir, vi in [-1, 1], (B,1,H,W) -> (B,1,H,W) in [-1, 1] (edge-padded by 6 first)."""
        x = torch.cat([ir, vi], dim=1)
        x = F.pad(x, (self.PAD,) * 4, mode="replicate")          # np.lib.pad(..., 'edge')
        for conv, bn in zip(self.convs[:4], self.bns):
            x = lrelu(bn(conv(x, update)))
        return torch.tanh(self.convs[4](x, update))


class Discriminator(nn.Module):
    """``CGAN.discriminator``; ``size`` is the side of the images it judges."""

    def __init__(self, size):
        super().__init__()
        spec = [(1, 32), (32, 64), (64, 128), (128, 256)]
        self.convs = nn.ModuleList([SNConv2d(cin, cout, 3, stride=2) for cin, cout in spec])
        self.bns = nn.ModuleList([batch_norm(cout) for _, cout in spec[1:]])
        side = size
        for _ in spec:
            side = (side - 3) // 2 + 1
        self.line = SNLinear(side * side * 256, 1)

    def forward(self, img, update):
        x = lrelu(self.convs[0](img, update))
        for conv, bn in zip(self.convs[1:], self.bns):
            x = lrelu(bn(conv(x, update)))
        # TF flattens NHWC; the order of the features does not matter for a freshly trained layer
        return self.line(x.flatten(1), update)


def gradient(x):
    """utils.gradient: Laplacian [[0,1,0],[1,-4,1],[0,1,0]], SAME (zero) padding."""
    k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], device=x.device, dtype=x.dtype)
    return F.conv2d(x, k.view(1, 1, 3, 3), padding=1)


def uniform_like(logits, low, high):
    """tf.random_uniform(shape=[batch_size, 1], minval=low, maxval=high)"""
    return torch.empty_like(logits).uniform_(low, high)


def d_loss(d_real, d_fake):
    pos = torch.mean(torch.square(d_real - uniform_like(d_real, 0.7, 1.2)))
    neg = torch.mean(torch.square(d_fake - uniform_like(d_fake, 0.0, 0.3)))
    return neg + pos


def g_loss(d_fake, fused, ir, vi):
    adv = torch.mean(torch.square(d_fake - uniform_like(d_fake, 0.7, 1.2)))
    content = torch.mean(torch.square(fused - ir)) + 5 * torch.mean(torch.square(gradient(fused) - gradient(vi)))
    return adv + 100 * content, adv, content


def check_spectral_norm(seed=0):
    """After enough power iterations the normalised weight has largest singular value 1."""
    torch.manual_seed(seed)
    w = SpectralNormWeight((8, 4, 3, 3), 8)
    with torch.no_grad():
        for _ in range(200):
            w(True)
        s_max = float(torch.linalg.svdvals(w(False).reshape(8, -1))[0])
    return s_max, math.isclose(s_max, 1.0, rel_tol=1e-3)
