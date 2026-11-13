# Prithvi Landslide Detection Workshop

Hands-on workshop: fine-tuning the **Prithvi-EO-2.0** geospatial foundation
model for **landslide semantic segmentation** on the **Landslide4Sense**
dataset, using **TerraTorch**.

| Session | Notebook | Duration |
|---------|----------|----------|
| Part 1 — Fine-tuning | `Part1_Finetuning_Prithvi_Landslide4Sense.ipynb` | ~1.5 h |
| Part 2 — Inference & evaluation | `Part2_Inference_Prithvi_Landslide4Sense.ipynb` | ~1 h |

## Folder contents

```
prithvi-landslide-workshop/
├── Part1_Finetuning_Prithvi_Landslide4Sense.ipynb   # fine-tuning session
├── Part2_Inference_Prithvi_Landslide4Sense.ipynb    # inference session
├── configs/
│   └── prithvi_eo_v2_300_landslide4sense.yaml       # TerraTorch experiment config
├── outputs/                                         # figures, logs, metrics (created by the notebooks)
├── checkpoints/                                     # fine-tuned weights (created by Part 1)
├── data/                                            # Landslide4Sense download (created by the notebooks)
├── requirements.txt                                 # pinned, verified package versions
└── README.md
```

The notebooks are fully self-contained: they install packages, download the
dataset (~2 GB from Hugging Face), train, and evaluate. `data/` and
`checkpoints/` should **not** be committed to git (checkpoints are ~3.7 GB);
`outputs/` holds the figures and metrics from a verified full run and is worth
keeping in the repo as a reference.

## Environment setup

Any fresh single-GPU environment works (Colab, JupyterHub, a cloud VM).
Both notebooks begin with a `%pip install` cell, so on Colab you can simply
upload the notebook plus `configs/` and run. For a local/VM setup:

```bash
conda create -n landslide python=3.11 -y
conda activate landslide
pip install -r requirements.txt
```

**Hardware:** Part 1's full 30-epoch fine-tune takes ~20–30 min on an A100
and a few hours on a T4 (reduce `NUM_EPOCHS` accordingly). The code is
device-agnostic (CUDA / Apple MPS / CPU), but CPU is only realistic for the
2-epoch smoke test. Peak GPU memory at batch 16 with mixed precision is
roughly 20 GB; halve the batch size on smaller cards.

**Hugging Face access:** no account is required, but anonymous downloads of
the ~9,700-file dataset can hit rate limits. If that happens, set a (free)
token first: `export HF_TOKEN=hf_...`.

## How to run

1. Start Jupyter **from this folder** (both notebooks resolve paths relative
   to the working directory):

   ```bash
   jupyter lab
   ```

2. Run `Part1_Finetuning_Prithvi_Landslide4Sense.ipynb` top to bottom.
   The training cell has one knob:

   ```python
   NUM_EPOCHS = 30   # reduce to 2 for a quick smoke test
   ```

   Part 1 writes the best checkpoint to `checkpoints/` and figures/logs to
   `outputs/`.

3. Run `Part2_Inference_Prithvi_Landslide4Sense.ipynb`. It picks up the
   Part 1 checkpoint from `checkpoints/` automatically (best epoch first,
   then `last.ckpt`). Part 1 must have been run at least once — its 2-epoch
   smoke-test setting is enough to produce a usable checkpoint.

### Running from the config instead of the notebook

The whole Part 1 experiment also runs headless from the YAML config:

```bash
terratorch fit --config configs/prithvi_eo_v2_300_landslide4sense.yaml
```

## Task summary

- **Data:** [Landslide4Sense](https://huggingface.co/datasets/ibm-nasa-geospatial/Landslide4sense)
  — 128×128 patches, 14 bands (Sentinel-2 B1–B12 + slope + DEM), binary
  landslide masks; splits 3799 / 245 / 800 (train / val / test).
- **Model:** [Prithvi-EO-2.0-300M](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M)
  ViT encoder (6 HLS bands: Blue, Green, Red, NIR, SWIR-1, SWIR-2) + UNet
  decoder, assembled by TerraTorch's `EncoderDecoderFactory`.
- **Band mapping:** the 6 Prithvi bands are selected from the 14 by name;
  Sentinel-2's broad NIR (B8) stands in for HLS's narrow NIR (B8A) — Part 1,
  Section 4 discusses this choice.

## Acknowledgements

The IBM–NASA Prithvi team (foundation models), the TerraTorch developers
(fine-tuning toolkit), and the Landslide4Sense creators (Ghorbanzadeh et al.)
— see the Acknowledgements section at the end of each notebook.
