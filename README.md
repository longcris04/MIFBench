# MIFBench

A benchmark of 20 published image fusion methods on multimodal medical image fusion
(CT-MRI, PET-MRI and SPECT-MRI), all trained, fused and scored under one protocol. It is the
comparison of the [APLFusion](https://github.com/longcris04/APLFusion) paper: every method is
trained on the same 4-fold splits, for the same 20 epochs on the same 256×256 images, and every
fused image is scored with the same MATLAB metric code.

Each method's released code is kept as its authors published it, in `models/<METHOD>/`. One file
is added next to it, `mif_adapter.py`, which tells the shared pipeline (`mifkit/`) how to build,
train and run the network with the method's own loss, optimiser and learning rate. Where the
authors released no training code, the adapter follows the paper; every departure from the
released code is written in the adapter's docstring and in its `HPARAMS`, which every run
copies into its `config.json`.

## Protocol

| | |
|---|---|
| data | the Harvard Whole Brain Atlas pairs of the APLFusion paper: CT-MRI 187, PET-MRI 288, SPECT-MRI 370 registered 256×256 pairs ([Data](#data)) |
| folds | 4 folds; each tests on 25 % of the pairs (CT-MRI 47 / 47 / 47 / 46, PET-MRI 72, SPECT-MRI 93 / 93 / 92 / 92), and every pair is a test pair in exactly one fold (`splits/<PAIR>.json`) |
| input | whole 256×256 images, no crop, no resize |
| augmentation | random horizontal and vertical flips (p = 0.5 each), the same on both sources |
| colour | PET / SPECT are fused on their Y channel (OpenCV BT.601 YCbCr); the colour result is the fused Y with the source's Cb / Cr |
| epochs | 20 |
| batch | 64; smaller where 64 does not fit an 80 GB A100 (the largest of 56 / 48 / ... / 6 that fits) or where the method does not train at 64 (DATFuse, ECINFusion, MMIF-INet, TUFusion), see the table below |
| optimiser, learning rate, loss | each method's own (table below) |
| every epoch | the test split is fused and saved (grey and colour), the checkpoint is written, and the fused images are scored with the MATLAB metrics under GNU Octave |
| seed | 0 |

The split is per slice and stratified by case: a file name is a case id followed by a 3-digit
slice number (`16003` is slice 3 of case 16; the 3-digit names form one more group), and the
slices of each case are dealt to the four folds in turn (seed 42, `mifkit/folds.py`). A
case-level split cannot give 25 % test sets here (PET-MRI has four cases), so test scores are on
unseen slices of seen patients.

## Methods

| method | code (commit) | training from | batch | optimiser / lr | notes |
|---|---|---|---|---|---|
| ALMFnet | [RollingPlain/ALMFnet](https://github.com/RollingPlain/ALMFnet) (`7801a02`) | `train.py` | 64 | Adam 0.025 (`train.py`; paper 1e-4) | saliency / segmentation masks rebuilt from the paper (the release ships them for 8 samples) |
| DATFuse | [tthinking/DATFuse](https://github.com/tthinking/DATFuse) (`0cd93f1`) | `Train.py` | 32 | Adam 1e-3 | batch 32 trains better than 64 (`Train.py`: 128) |
| ECINFusion | [SadInSummer/ECINFusion](https://github.com/SadInSummer/ECINFusion) (`cdcb3ed`) | `Train.py` | 8 | Adam 1e-3 (`Train.py`; paper 1e-4) | 64 does not fit (double-precision RMI loss) |
| FATFusion | [tthinking/FATFusion](https://github.com/tthinking/FATFusion) (`9873c34`) | `Train.py` | 8 | Adam 1e-3 (`Train.py`; paper 0.03) | 12 does not fit |
| LRFNet | [HeDan-11/LRFNet](https://github.com/HeDan-11/LRFNet) (`f5aae42`) | paper + `test.py` | 64 | Adam 1e-3 | no training code: DGSSIM (paper Eq. 11) + MSE on 0-255 with `test.py`'s weights; SSIM alone collapses to black |
| MMIF-CDDFuse | [Zhaozixiang1228/MMIF-CDDFuse](https://github.com/Zhaozixiang1228/MMIF-CDDFuse) (`73681d2`) | `train.py` | 6 | Adam 1e-4 ×4, StepLR 0.5, clip 0.01 | two phases, switch at 1/3 of the run (as 40 / 120); MIF decoder as `test_MIF.py` |
| MMIF-INet | [HeDan-11/MMIF-INet](https://github.com/HeDan-11/MMIF-INet) (`5a3c882`) | `train_logging.py`, `config.py` | 16 | Adam **1e-3** (`config.py`: 10^-4.5), betas (0.5, 0.999), wd 1e-5 | at the released lr the output stays black in 20 epochs; whole images instead of 128×128 crops |
| MixFuse | [Bitlijinfu/MixFuse](https://github.com/Bitlijinfu/MixFuse) (`7839718`) | `loss/` (no training script) | 12 | Adam 2e-4 (not released; SwinFusion's) | `fusion_loss_vif` 10 / 5 / 4 |
| TUFusion | [windrunners/TUFusion](https://github.com/windrunners/TUFusion) (`985b43d`) | `train_tufusion.py` | 8 | Adam 1e-4 | at batch 64 the images stay blocky; fusion with `fusion2` (`test_image.py`) |
| VDMUFusion | [yuliu316316/VDMUFusion](https://github.com/yuliu316316/VDMUFusion) (`6e2d910`) | `train.py`, `config.json` | 64 | AdamW 2e-4 | 2000-step sampler at every epoch (~8 h of a fold) |
| DSPFusion | [Linfeng-Tang/DSPFusion](https://github.com/Linfeng-Tang/DSPFusion) (`032c061`) | `DSPF_S1.yml`, `DSPF_S2.yml` | 12 | AdamW 2e-4, wd 1e-4 | stage I epochs 1-10, stage II 11-20; clean images: lq = gt, no degradation loss |
| FusionGAN | [jiayi-ma/FusionGAN](https://github.com/jiayi-ma/FusionGAN) (`15e7ea1`) | TF1 release, ported to PyTorch (`fusiongan_torch.py`) | 64 | Adam 1e-4 (G and D) | 2 D steps per G step, as the TF code |
| GeSeNet | [lok-18/GeSeNet](https://github.com/lok-18/GeSeNet) (`89f7fe9`) | paper (no training code) | 64 | Adam 1e-3, wd 2e-4 | semantic loss left out (its pre-trained module is not released) |
| MM-Net | [yuliu316316/MM-Net-Fusion](https://github.com/yuliu316316/MM-Net-Fusion) (`6929db4`) | paper + `loss.py` | 56 | Adam 1e-4, wd 1e-4 (`config.py`) | the network predicts a weight map (`test.py`) |
| MMA-UNet | [JasonWong30/MMA-UNet](https://github.com/JasonWong30/MMA-UNet) (`a63547b`) | paper + `Unet.forward_loss` | 64 | AdamW wd 5e-3, cosine; 1e-3 (auto-encoders), 1e-4 (fusion) | stages: VI-UNet 1-5, IR-UNet 6-10, fusion 11-20 (lengths not in the paper) |
| SigFusion | [fengjiawei123/SigFusion](https://github.com/fengjiawei123/SigFusion) (`b871418`) | paper (no training code) | 24 | Adam 1e-4 | stage I (PLGN pre-training) not released: trained from scratch |
| SwinFusion | [Linfeng-Tang/SwinFusion](https://github.com/Linfeng-Tang/SwinFusion) (`ab8c9ee`) | `train_swinfusion_med.json` | 8 | Adam 2e-4 | fused with the EMA model, decay scaled to the run's length |
| TC-MoA | [YangSun22/TC-MoA](https://github.com/YangSun22/TC-MoA) (`48e4a54`) | `main_train.py`, `base.yaml` | 16 | AdamW 3e-4, 2-epoch warm-up + cosine | frozen MAE ViT-L backbone ([weights](#pretrained-weights)); images resized 256 → 336 and back |
| U2Fusion | [hanna-xu/U2Fusion-pytorch](https://github.com/hanna-xu/U2Fusion-pytorch) (`6b2acd5`) | `train.py`, task 1 | 64 | RMSprop 1e-4, momentum 0.15, clip 50 | VGG16 = the original Caffe weights ([weights](#pretrained-weights)) |
| YDTR | [tthinking/YDTR](https://github.com/tthinking/YDTR) (`b1f01b4`) | `Train.py` | 40 | Adam 1e-3 | same training template as DATFuse |

The adapters (`models/<METHOD>/mif_adapter.py`) give the full setting of each method -- loss
weights, which source plays which role, output range -- and where each value comes from. When
neither the code nor the paper states a value (e.g. the stage lengths of a staged method), it
follows APLFusion: 20 epochs, two stages switching at epoch 10.

## Repository

```
mifkit/                    the pipeline: data, folds, training loop, Octave scoring, command-line tools
models/<METHOD>/           the method's released code + mif_adapter.py
models/<METHOD>/mif_runs/  the runs (written by the pipeline)
evaluation/                MATLAB metric code (Run_Foder_Updated.m, indexes/) + octave_eval/ (Octave entry point)
splits/<PAIR>.json         the 4-fold splits (file names of the training and test pairs)
data/                      (empty) the image pairs go here, see Data
results/                   reference scores of the benchmark (epoch 20, every metric), see Results
scripts/                   environment setup, weight download, Slurm job scripts
review_server/             a local web page to browse the fused images and scores of the runs
```

## Installation

```bash
scripts/setup_env.sh        # conda envs `mifbench` (PyTorch) and `mifoctave` (GNU Octave + image package)
source scripts/env.sh       # exports MIF_PYTHON, MIF_OCTAVE, OCTAVE_HOME, PYTHONPATH
```

`mifbench` is Python 3.11 with PyTorch 2.6.0 (CUDA 12.4) and `requirements.txt`; `mifoctave` is
GNU Octave 10.3 from conda-forge with the Octave Forge `image` package. The benchmark was run on
NVIDIA A100 80 GB GPUs; training needs CUDA (some released code calls `.cuda()` directly). The
metrics also run under MATLAB (`evaluation/Run_Foder_Updated.m`), but the pipeline drives Octave.

### Pretrained weights

Two methods use a frozen pretrained network; the others train from scratch.

```bash
$MIF_PYTHON scripts/download_weights.py
```

* U2Fusion: the Caffe VGG16 of its loss. The release's `vgg16.npy` is on Baidu only; the same
  weights are OpenMMLab's `vgg16_caffe-292e1171.pth`, converted to `models/U2Fusion/vgg16.pth`.
* TC-MoA: the MAE ViT-L/16 (GAN loss) backbone, `mae_visualize_vit_large_ganloss.pth` (1.3 GB),
  saved to `models/TC-MoA/checkpoint/`.

The script checks the MD5 of both downloads.

## Data

The 845 image pairs, already in the layout below (the same file as the APLFusion repository):
[`APLFusion_data.zip`](https://drive.google.com/file/d/1rgL2uj34P1Kt51joLEjkotuyWQEzvDOp/view)
(Google Drive, 34 MB). Unzip it in the root of this repository:

```bash
pip install gdown
gdown 1rgL2uj34P1Kt51joLEjkotuyWQEzvDOp   # downloads APLFusion_data.zip
unzip APLFusion_data.zip                  # creates data/CT-MRI, data/PET-MRI and data/SPECT-MRI
```

```
data/
├── CT-MRI/
│   ├── CT/      000.bmp  001.bmp  ...  2004.bmp  ...
│   └── MRI/     (the same file names)
├── PET-MRI/
│   ├── PET/
│   └── MRI/
└── SPECT-MRI/
    ├── SPECT/
    └── MRI/
```

The images come from the [Harvard Whole Brain Atlas](https://www.med.harvard.edu/AANLIB/). PET
and SPECT are colour images, CT and MRI grey; the two images of a pair have the same file name.
`MIF_DATA_ROOT=/path/to/data` points the pipeline at a copy elsewhere.

## Running the benchmark

**One run** (one method, one task, one fold):

```bash
source scripts/env.sh
$MIF_PYTHON -m mifkit.cli.train --model SwinFusion --pair PET-MRI --fold 1
```

It trains for 20 epochs; after each epoch it fuses the fold's test pairs, saves a checkpoint and
scores the fused images in background Octave processes (`--eval-workers`, default 16), then
waits for the scores at the end. `--help --model <METHOD>` lists the method's own options.

**One fold** (every method × task, one after another on one GPU):

```bash
$MIF_PYTHON -m mifkit.cli.run_fold --fold 1                     # 20 methods x 3 tasks
$MIF_PYTHON -m mifkit.cli.run_fold --fold 1 --models DATFuse GeSeNet --pairs CT-MRI
```

Each (method, task) run is its own process, so one method failing does not stop the others; the
Octave scoring runs on the CPUs while the GPU trains the next epoch or method. A fold takes about
15 GPU-hours on an A100 (VDMUFusion ~8 h of it, SwinFusion ~1 h, FATFusion and MixFuse ~1 h
each, most others minutes); the scoring needs CPU cores alongside (the Slurm script asks for 24).

**All four folds on Slurm** (one job per fold: 1 GPU, 24 CPUs):

```bash
scripts/submit_folds.sh                                   # folds 1-4
MIF_SBATCH_ARGS="-p <partition> --qos <qos>" scripts/submit_folds.sh
scripts/submit_folds.sh 2 -- --models SwinFusion          # run_fold arguments after --
tail -f logs/fold1/run_fold.log
```

Running the same command again resumes: finished runs return at once, interrupted runs continue
from their last checkpoint, and fused but unscored epochs are scored. A run trained with other
settings (e.g. a changed batch size in an adapter) is deleted and trained again.

**Progress and tables**:

```bash
$MIF_PYTHON -m mifkit.cli.status                  # epochs trained / scored per run, last Qabf
$MIF_PYTHON -m mifkit.cli.summary                 # cross-fold tables in summary/
review_server/run.sh                              # http://127.0.0.1:8777: images and scores of every run
```

`mifkit.cli.summary` writes, per task, `summary/<PAIR>_final.csv` (every metric at the last
epoch that every fold has scored, or at `--epoch`: the mean over all pairs of the task, each
scored by the model of the fold it is a test pair of, and the mean ± std of the four fold
means), `<PAIR>_by_epoch.csv` (every epoch) and a Markdown table.

**Other tools**: `scripts/smoke_test.sh [methods]` (2 epochs × 3 steps per method, a quick check
of an installation), `scripts/probe_batch.py <METHOD>` (largest batch that fits the GPU),
`mifkit.cli.score` (score fused images again), `mifkit.cli.time_inference` and
`mifkit.cli.count_flops` (inference time, parameters and FLOPs of the trained fold-1 models).

### What a run writes

```
models/<METHOD>/mif_runs/<PAIR>/fold<k>/
    config.json             protocol, the method's hyper-parameters, the split, the environment
    metrics_by_epoch.csv    per epoch: the mean of every metric over the test pairs, loss terms, timings
    scores/epoch_NNN.csv    per epoch: every test pair's metrics
    results/epoch_NNN/{gray,color}/<name>.png    the fused test images of every epoch
    checkpoints/last.pt     the last epoch (model, optimiser, RNG state), for resuming
    logs/                   train.log, train_steps.jsonl (every step's loss terms), epochs.json
```

## Evaluation

`evaluation/octave_eval/mif_eval_list.m` runs the per-image loop of
`evaluation/Run_Foder_Updated.m` under GNU Octave, with the same 29 columns: for each pair,
`path_1` is the functional source as stored (CT, or colour PET / SPECT through `rgb2gray`),
`path_2` the MRI and `path_3` the fused grey image. The metric code in `evaluation/indexes/` is
unchanged; `octave_eval/` adds Octave replacements (`dct2`, NIQE helpers) and vectorised versions
of `fmi`, `Qabf`, `mutual_information` and `NCC` that give identical numbers about 1000× faster
(see `evaluation/octave_eval/README.md`). Lower is better for NIQE, SSEQ and Chen; `metricZhao`,
`x1`, `x2` and `Time` carry no information.

The APLFusion paper reports eight of the columns:

| paper | column | metric |
|---|---|---|
| SD | `SD` | standard deviation of the fused image, `DanhGiaAnh.m` |
| Q<sup>AB/F</sup> | `Qabf` | edge-information preservation, `Qabf.m` |
| FMI | `FMI` | feature mutual information on edge features, `fmi(…, 'edge', 3)` |
| Q<sub>S</sub> | `Q` | Piella's metric (Liu et al. 2012 naming), `metricPeilla(…, 1)` |
| Q<sub>W</sub> | `Qw` | weighted Piella metric, `metricPeilla(…, 2)` |
| Q<sub>E</sub> | `Qe` | edge-dependent Piella metric, `metricPeilla(…, 3)` |
| SSIM | `SSIM` | mean of SSIM(A, F) and SSIM(B, F), `MS_SSIM.m` |
| Q<sub>CB</sub> | `CB` | Chen-Blum metric, `metricChenBlum.m` |

## Results

Means over all pairs of a task at epoch 20, each pair scored once, as a test pair, by the model
of its fold (`results/<PAIR>_epoch20.csv` has every metric). These are the runs of the
benchmark, trained with the settings in this repository.

**CT-MRI** (187 pairs)

| method | SD | Q<sup>AB/F</sup> | FMI | Q<sub>S</sub> | Q<sub>W</sub> | Q<sub>E</sub> | SSIM | Q<sub>CB</sub> |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ALMFnet | 0.3409 | 0.4455 | 0.8593 | 0.7754 | 0.6479 | 0.4068 | 0.7252 | 0.6455 |
| DATFuse | 0.3094 | 0.3106 | 0.8477 | 0.7133 | 0.5415 | 0.2217 | 0.6822 | 0.6255 |
| ECINFusion | 0.1067 | 0.2664 | 0.8689 | 0.4452 | 0.1784 | 0.0188 | 0.4325 | 0.3723 |
| FATFusion | 0.1717 | 0.3207 | 0.8487 | 0.7133 | 0.2154 | 0.0302 | 0.6969 | 0.5504 |
| LRFNet | 0.2374 | 0.2708 | 0.8670 | 0.6849 | 0.5610 | 0.2269 | 0.6395 | 0.5311 |
| MMIF-CDDFuse | 0.2899 | 0.5389 | 0.8578 | 0.8310 | 0.7555 | 0.5409 | 0.7423 | 0.6159 |
| MMIF-INet | 0.1142 | 0.4217 | 0.8501 | 0.5404 | 0.4682 | 0.2153 | 0.4776 | 0.3263 |
| MixFuse | 0.2888 | 0.4659 | 0.8639 | 0.2995 | 0.7021 | 0.5758 | 0.2429 | 0.2143 |
| TUFusion | 0.2287 | 0.3433 | 0.8662 | 0.5596 | 0.5755 | 0.3341 | 0.5238 | 0.4543 |
| VDMUFusion | 0.2172 | 0.2533 | 0.8591 | 0.2706 | 0.5323 | 0.1715 | 0.2238 | 0.1712 |
| DSPFusion | 0.3475 | 0.4918 | 0.8667 | 0.8249 | 0.7463 | 0.5714 | 0.7626 | 0.6124 |
| FusionGAN | 0.3012 | 0.2422 | 0.8408 | 0.4389 | 0.5123 | 0.1020 | 0.4037 | 0.5153 |
| GeSeNet | 0.2975 | 0.4714 | 0.8599 | 0.8005 | 0.7028 | 0.4619 | 0.7277 | 0.6518 |
| MM-Net | 0.2161 | 0.4363 | 0.8679 | 0.8025 | 0.4898 | 0.1404 | 0.7486 | 0.6673 |
| MMA-UNet | 0.0669 | 0.0210 | 0.7310 | 0.0229 | 0.0140 | 0.0000 | 0.0286 | 0.0566 |
| SigFusion | 0.3201 | 0.4760 | 0.8654 | 0.8177 | 0.7506 | 0.5634 | 0.7485 | 0.6403 |
| SwinFusion | 0.2996 | 0.4990 | 0.8623 | 0.5502 | 0.7135 | 0.5484 | 0.4888 | 0.4284 |
| TC-MoA | 0.0000 | 0.0000 | 0.8815 | 0.4779 | 0.0017 | 0.0000 | 0.5112 | – |
| U2Fusion | 0.3068 | 0.2074 | 0.8644 | 0.6960 | 0.4725 | 0.1580 | 0.7004 | 0.6353 |
| YDTR | 0.0471 | 0.0227 | 0.8568 | 0.4636 | -0.0074 | 0.0072 | 0.4854 | 0.4537 |
| *APLFusion* ² | 0.3242 | 0.4833 | 0.8739 | 0.7929 | 0.7641 | 0.6477 | 0.7520 | 0.6396 |

**PET-MRI** (288 pairs)

| method | SD | Q<sup>AB/F</sup> | FMI | Q<sub>S</sub> | Q<sub>W</sub> | Q<sub>E</sub> | SSIM | Q<sub>CB</sub> |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ALMFnet | 0.3594 | 0.6579 | 0.8337 | 0.7936 | 0.8905 | 0.8094 | 0.6257 | 0.5425 |
| DATFuse | 0.2955 | 0.5001 | 0.8132 | 0.7805 | 0.8115 | 0.5975 | 0.6463 | 0.5669 |
| ECINFusion | 0.1919 | 0.5981 | 0.8442 | 0.6488 | 0.7143 | 0.5561 | 0.5208 | 0.5065 |
| FATFusion | 0.3167 | 0.6762 | 0.8305 | 0.7151 | 0.8890 | 0.8352 | 0.5379 | 0.4517 |
| LRFNet | 0.2742 | 0.4856 | 0.8404 | 0.7267 | 0.6882 | 0.4502 | 0.6030 | 0.4853 |
| MMIF-CDDFuse | 0.3145 | 0.7003 | 0.8325 | 0.8980 | 0.9224 | 0.8583 | 0.7063 | 0.5510 |
| MMIF-INet | 0.1542 | 0.5654 | 0.8141 | 0.4360 | 0.6387 | 0.5980 | 0.3064 | 0.3367 |
| MixFuse | 0.2881 | 0.7189 | 0.8371 | 0.4432 | 0.8576 | 0.8626 | 0.2851 | 0.2766 |
| TUFusion | 0.2269 | 0.4017 | 0.8236 | 0.5861 | 0.5466 | 0.3460 | 0.5215 | 0.4772 |
| VDMUFusion | 0.2122 | 0.1675 | 0.8114 | 0.3211 | 0.5180 | 0.1029 | 0.2312 | 0.2198 |
| DSPFusion | 0.3162 | 0.7491 | 0.8520 | 0.9115 | 0.9319 | 0.8959 | 0.7243 | 0.5914 |
| FusionGAN | 0.2358 | 0.4105 | 0.8030 | 0.4871 | 0.6156 | 0.3411 | 0.3695 | 0.3113 |
| GeSeNet | 0.3165 | 0.7326 | 0.8459 | 0.8856 | 0.9116 | 0.8828 | 0.7085 | 0.6209 |
| MM-Net | 0.2616 | 0.6456 | 0.8518 | 0.8765 | 0.8401 | 0.6694 | 0.7138 | 0.6395 |
| MMA-UNet | 0.0710 | 0.0224 | 0.7222 | 0.0202 | 0.0144 | 0.0000 | 0.0191 | 0.0479 |
| SigFusion | 0.3232 | 0.7561 | 0.8522 | 0.9029 | 0.9238 | 0.8952 | 0.7179 | 0.6201 |
| SwinFusion | 0.3252 | 0.7723 | 0.8635 | 0.9057 | 0.9380 | 0.9113 | 0.7141 | 0.5992 |
| TC-MoA ¹ | 0.0000 | 0.0001 | 0.8696 | 0.3873 | 0.0011 | 0.0000 | 0.5397 | – |
| U2Fusion | 0.3026 | 0.4024 | 0.8086 | 0.7899 | 0.7710 | 0.4341 | 0.6702 | 0.5801 |
| YDTR | 0.0040 | 0.0004 | 0.8414 | 0.3920 | 0.0012 | 0.0000 | 0.5430 | 0.5160 |
| *APLFusion* ² | 0.3079 | 0.7603 | 0.8512 | 0.8953 | 0.9225 | 0.8957 | 0.7172 | 0.6091 |

**SPECT-MRI** (370 pairs)

| method | SD | Q<sup>AB/F</sup> | FMI | Q<sub>S</sub> | Q<sub>W</sub> | Q<sub>E</sub> | SSIM | Q<sub>CB</sub> |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ALMFnet | 0.2507 | 0.6752 | 0.8571 | 0.8306 | 0.8644 | 0.7766 | 0.6855 | 0.5750 |
| DATFuse | 0.2153 | 0.4506 | 0.8285 | 0.5905 | 0.7444 | 0.4797 | 0.4840 | 0.4195 |
| ECINFusion | 0.1197 | 0.4830 | 0.8573 | 0.6709 | 0.6531 | 0.4396 | 0.5844 | 0.4996 |
| FATFusion | 0.2589 | 0.6024 | 0.8421 | 0.8245 | 0.8412 | 0.7385 | 0.6800 | 0.5088 |
| LRFNet | 0.1706 | 0.4955 | 0.8604 | 0.6777 | 0.7502 | 0.4986 | 0.5769 | 0.4397 |
| MMIF-CDDFuse | 0.2399 | 0.7103 | 0.8616 | 0.8735 | 0.8724 | 0.8273 | 0.7191 | 0.6096 |
| MMIF-INet | 0.1344 | 0.6458 | 0.8484 | 0.6599 | 0.7604 | 0.7483 | 0.5166 | 0.3618 |
| MixFuse | 0.2259 | 0.6365 | 0.8492 | 0.3068 | 0.6642 | 0.7414 | 0.2016 | 0.1850 |
| TUFusion | 0.1582 | 0.3630 | 0.8390 | 0.5396 | 0.5672 | 0.3382 | 0.4980 | 0.4351 |
| VDMUFusion | 0.1530 | 0.1662 | 0.8134 | 0.2656 | 0.5234 | 0.1269 | 0.2144 | 0.1569 |
| DSPFusion | 0.2217 | 0.7285 | 0.8660 | 0.9142 | 0.9015 | 0.8522 | 0.7523 | 0.6097 |
| FusionGAN | 0.1589 | 0.4047 | 0.8329 | 0.5779 | 0.6560 | 0.3989 | 0.4880 | 0.3360 |
| GeSeNet | 0.2380 | 0.7007 | 0.8654 | 0.8773 | 0.8797 | 0.8368 | 0.7339 | 0.6538 |
| MM-Net | 0.1915 | 0.6735 | 0.8707 | 0.8896 | 0.8438 | 0.6814 | 0.7541 | 0.6647 |
| MMA-UNet | 0.0560 | 0.0190 | 0.6971 | 0.0276 | 0.0322 | 0.0001 | 0.0296 | 0.0388 |
| SigFusion | 0.2253 | 0.7322 | 0.8686 | 0.8979 | 0.8998 | 0.8592 | 0.7480 | 0.6506 |
| SwinFusion | 0.2227 | 0.7529 | 0.8795 | 0.9004 | 0.9058 | 0.8733 | 0.7537 | 0.6984 |
| TC-MoA | 0.0000 | 0.0001 | 0.8766 | 0.4473 | 0.0037 | 0.0000 | 0.5264 | – |
| U2Fusion | 0.2027 | 0.3494 | 0.8342 | 0.7827 | 0.7237 | 0.3243 | 0.7140 | 0.5828 |
| YDTR | 0.0050 | 0.0027 | 0.8063 | 0.4429 | 0.0050 | 0.0000 | 0.5193 | 0.4478 |
| *APLFusion* ² | 0.2215 | 0.7292 | 0.8713 | 0.8934 | 0.8988 | 0.8603 | 0.7471 | 0.6525 |

¹ TC-MoA crashed in epoch 4 of PET-MRI fold 2 (NaN router gates); its PET-MRI row covers folds 1, 3
and 4 (216 pairs). Q<sub>CB</sub> is undefined for its all-black output.
² APLFusion is trained and released in its own repository,
[longcris04/APLFusion](https://github.com/longcris04/APLFusion), with the same data, splits and
metric code; its row is the paper's models (the pretrained models of that repository).

Three methods do not train under this protocol: their output is (nearly) black at epoch 20, and
the rows above are what such images score. **TC-MoA**: the warm-up loss on mostly black medical
images drives every pixel to 0 within the first epochs, where the output clipping has no gradient.
**YDTR**: the tanh output drifts negative and is clipped to black. **MMA-UNet**: the fusion stage
gets only 30-50 optimiser steps at lr 1e-4 (3-5 steps per epoch for 10 epochs) and does not learn
a fusion in that time.

### Rerunning the benchmark

Retrained runs come close to these numbers but do not repeat them exactly: the pipeline uses
cuDNN's benchmark mode (`torch.backends.cudnn.benchmark = True`), which picks convolution
algorithms by timing, and some GPU operations (e.g. the backward pass of reflection padding and
interpolation) are not deterministic, so two runs of the same setting can differ slightly. The
seed fixes the initial weights, the order of the batches and the flips. To compare a rerun with
the table, run `mifkit.cli.summary --epoch 20` and read `summary/<PAIR>_final.csv` (the same
pooled means).

## Things to keep in mind

* 20 epochs of 3-5 optimiser steps each (batch 64, 140-278 training pairs per fold) is far shorter
  than any method's own schedule (10-1000 epochs). Some methods are still improving at epoch 20
  (e.g. MMIF-INet); the runs' `metrics_by_epoch.csv` and `summary/<PAIR>_by_epoch.csv` show
  every epoch.
* Methods without released training code (LRFNet, MixFuse's optimiser, GeSeNet, MM-Net, MMA-UNet,
  SigFusion) follow their papers; parts the papers leave out are listed in the adapters.
* Fused outputs are clipped to [0, 1] before saving (several networks end in tanh or have no
  output activation; the released test scripts wrap such values with `astype(uint8)`).
* VDMUFusion's sampler draws noise and TUFusion re-initialises a random axial transformer at every
  fusion; the pipeline fixes the RNG while fusing, so each epoch is fused with the same noise.

## Adding a method

Put the method's code in `models/<NAME>/`, add a `mif_adapter.py` (the interface is at the top of
`mifkit/pipeline.py`; the adapters of the 20 methods are examples) and add `<NAME>` to `MODELS`
and `RUN_ORDER` in `mifkit/models.py` (or to `EXPERIMENTS`, to train it only when named). The
adapter receives batches of `func` (CT, or the Y of PET / SPECT) and `mri`, both (B, 1, 256, 256)
in [0, 1], and returns the fused luminance in the same format.

## Third-party code

`models/<METHOD>/` contains code from the repositories in the [Methods](#methods) table, at the
listed commits, without the authors' sample images and pretrained weights (which the benchmark
does not use); only `mif_adapter.py` (and, for FusionGAN, the PyTorch port `fusiongan_torch.py`)
is added. ALMFnet, FusionGAN and GeSeNet are under the MIT license and MMA-UNet under the GPL
(their `LICENSE` files are kept); the other repositories do not state a license, and their code
remains their authors'. `evaluation/` contains the MATLAB metric code with its third-party parts
(`indexes/imageFusionMetrics-master`, `indexes/SSEQ-main` with libsvm). Please cite the original
papers of the methods you compare with.
