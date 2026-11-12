"""Download the two pretrained networks that benchmark models use as frozen parts.

    $MIF_PYTHON scripts/download_weights.py                 # both
    $MIF_PYTHON scripts/download_weights.py --only u2fusion
    $MIF_PYTHON scripts/download_weights.py --vgg-caffe /path/to/vgg16_caffe-292e1171.pth   # local copy

* U2Fusion's loss weights the sources by VGG16 features.  The release loads ``vgg16.pth``,
  converted from a TF ``vgg16.npy`` that is linked on Baidu only; it is the original Caffe
  VGG16, which OpenMMLab publishes as ``vgg16_caffe-292e1171.pth``.  Its 13 conv layers
  (``features.0`` ... ``features.28``) are written to ``models/U2Fusion/vgg16.pth`` under
  ``model.py``'s names (``conv1_1`` ... ``conv5_3``) -- the file the benchmark runs used.
* TC-MoA's frozen backbone is MAE ViT-L/16 trained with the GAN loss
  (``mae_visualize_vit_large_ganloss.pth`` from the MAE release, as TC-MoA's README says),
  saved to ``models/TC-MoA/checkpoint/``.

The other models train from scratch and need no download.
"""

import argparse
import hashlib
import os
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VGG_URL = "https://download.openmmlab.com/pretrain/third_party/vgg16_caffe-292e1171.pth"
VGG_MD5 = "b191dbef885dca139aca6e76a23076f7"
VGG_OUT = os.path.join(ROOT, "models", "U2Fusion", "vgg16.pth")
VGG_LAYERS = ["conv1_1", "conv1_2", "conv2_1", "conv2_2", "conv3_1", "conv3_2", "conv3_3",
              "conv4_1", "conv4_2", "conv4_3", "conv5_1", "conv5_2", "conv5_3"]
VGG_FEATURES = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]   # torchvision VGG16 conv indices

MAE_URL = "https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_large_ganloss.pth"
MAE_MD5 = "aa2eaf8b3799d0ac9af2f1df69fa92eb"
MAE_OUT = os.path.join(ROOT, "models", "TC-MoA", "checkpoint", "mae_visualize_vit_large_ganloss.pth")


def md5(path):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url, path, expected_md5):
    if os.path.exists(path) and md5(path) == expected_md5:
        print(f"ok (already there): {path}")
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    print(f"downloading {url}\n         -> {path}")
    with urllib.request.urlopen(url) as response, open(tmp, "wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        while True:
            block = response.read(1 << 20)
            if not block:
                break
            out.write(block)
            done += len(block)
            if total:
                print(f"\r  {done / 2 ** 20:7.1f} / {total / 2 ** 20:.1f} MB", end="", flush=True)
    print()
    got = md5(tmp)
    if got != expected_md5:
        os.remove(tmp)
        raise SystemExit(f"md5 mismatch for {url}: {got} (expected {expected_md5})")
    os.replace(tmp, path)
    return path


def convert_vgg(caffe_path, out_path=VGG_OUT):
    import torch

    blob = torch.load(caffe_path, map_location="cpu", weights_only=False)
    source = blob.get("state_dict", blob)
    state = {}
    for name, index in zip(VGG_LAYERS, VGG_FEATURES):
        state[f"{name}.weight"] = source[f"features.{index}.weight"].float().clone()
        state[f"{name}.bias"] = source[f"features.{index}.bias"].float().clone()
    torch.save(state, out_path)
    print(f"ok: {out_path} ({len(VGG_LAYERS)} conv layers from {os.path.basename(caffe_path)})")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=["u2fusion", "tc-moa"], default=None)
    parser.add_argument("--vgg-caffe", default=None,
                        help="use this local vgg16_caffe-292e1171.pth instead of downloading it")
    args = parser.parse_args(argv)

    if args.only in (None, "u2fusion"):
        caffe = args.vgg_caffe
        if caffe is None:
            caffe = download(VGG_URL, os.path.join(ROOT, "models", "U2Fusion", "pretrained",
                                                   os.path.basename(VGG_URL)), VGG_MD5)
        elif md5(caffe) != VGG_MD5:
            raise SystemExit(f"{caffe}: md5 is not that of vgg16_caffe-292e1171.pth")
        convert_vgg(caffe)
    if args.only in (None, "tc-moa"):
        download(MAE_URL, MAE_OUT, MAE_MD5)   # 1.3 GB


if __name__ == "__main__":
    main()
