# Phyto-VLMChatbot

Phyto-VLMChatbot is the source repository for **PhytoSpectra-VLM**, a research
pipeline and local chatbot for hyperspectral wheat-leaf analysis. Its
spectral-spatial PhytoCore predicts GrainWeight, Gsw, PhiPS2, Fertilizer, and
Cultivar. A question-conditioned visual bridge connects PhytoCore features to
Qwen2.5-1.5B-Instruct; a symbolic verifier resolves requested values from
PhytoCore predictions rather than treating generated text as a measurement.

This repository contains the model and training code, API backend, React
frontend, configuration, launch scripts, notebooks, and tests. **It does not
contain HyperLeaf2024 images or labels, trained checkpoints, model weights,
experiment results, logs, generated figures, or uploaded files.** A clone is
therefore source-complete but not immediately inference-ready: you must obtain
the dataset and train the model, or supply your own compatible locally trained
artifacts.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/phytospectra/` | Preprocessing, splits, PhytoCore, evidence, VLM, verification, training, and evaluation code |
| `configs/experiment.yaml` | Research protocol and model settings |
| `scripts/` | Single-run and 3-seed × 5-fold PowerShell launchers |
| `wheatspectra_app/backend/` | FastAPI upload, inference, knowledge, evidence, and chat service |
| `wheatspectra_app/frontend/` | React/Vite WheatSpectraChat UI; `package-lock.json` supports reproducible installs |
| `tests/` | Python tests; frontend tests live alongside the UI source |
| `notebooks/` | Optional execution and inspection notebooks |

Generated content belongs in `data/`, `artifacts/`, `results/`, `reports/`,
`logs/`, and the chatbot runtime directories. These paths are ignored by Git.

## Requirements

- Windows PowerShell for the research `.ps1` scripts; the app uses `.bat` launchers.
- Python **3.11** (`pyproject.toml` requires `>=3.11,<3.12`).
- Node.js **20 or newer** and npm for the frontend.
- Sufficient local storage and, for practical training and Qwen inference, a
  CUDA-capable GPU. The complete 3 × 5 campaign is substantially more
  demanding than starting the web UI.
- A locally cached copy of [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
  for VLM training and inference. The frozen experiment configuration sets
  `local_files_only: true`; the base language-model weights are not
  distributed here.

## 1. Install the code

In PowerShell, from the cloned `Phyto-VLMChatbot` directory:

```powershell
py -3.11 -m venv .venv
$Python = (Resolve-Path '.\.venv\Scripts\python.exe').Path
& $Python -m pip install --upgrade pip
& $Python -m pip install -e '.[app,dev]'
```

The project dependencies are declared in `pyproject.toml`. Install a PyTorch
build appropriate for your machine if the default package does not support
your CUDA setup. Before a VLM run, download the base Qwen model into the local
Hugging Face cache while internet access is available:

```powershell
& $Python -c 'from huggingface_hub import snapshot_download; snapshot_download("Qwen/Qwen2.5-1.5B-Instruct")'
```

This download is stored outside the Git repository and is not redistributed.

## 2. Obtain HyperLeaf2024 locally

Download the original dataset from the authors' [HyperLeaf2024 Kaggle data
page](https://www.kaggle.com/competitions/HyperLeaf2024/data), subject to its
access requirements and terms. The accompanying [dataset
paper](https://openaccess.thecvf.com/content/CVPR2024W/FGVC11/html/Laprade_HyperLeaf2024_-_A_Hyperspectral_Imaging_Dataset_for_Classification_and_Regression_CVPRW_2024_paper.html)
describes its acquisition and labels. Do not commit or redistribute the data
through this repository.

Arrange the downloaded training files as follows (or use any other local
directory with the same layout):

```text
data/raw/
  train.csv
  images/
    00000.tiff
    ...
```

The pipeline uses the **1,590 labelled training images** in `train.csv`; it
does not use the competition's hidden-label test partition. See `DATASET.md`
for the exact columns required by the loader. Set the data path for the
commands below:

```powershell
$RawData = (Resolve-Path '.\data\raw').Path
```

The optional `notebooks/00_Run_From_Scratch.ipynb` contains example local
paths; adjust its interpreter and dataset variables before running it on a
different computer. The PowerShell commands in this README are portable and
pass both paths explicitly.

## 3. Check installation and run a single experiment

```powershell
& '.\scripts\run_pipeline.ps1' -Stage all -Smoke -RawData $RawData -RunId 'smoke_v1' -Seed 2026 -Fold 0 -Python $Python
```

`-Smoke` is a bounded integration run, **not** a publishable result. It still
requires the dataset and local base Qwen model. To run one complete fold,
omit `-Smoke` and use a new run ID:

```powershell
& '.\scripts\run_pipeline.ps1' -Stage all -RawData $RawData -RunId 'my_full_run' -Seed 2026 -Fold 0 -Python $Python
```

The research protocol uses seeds 2026, 2027, and 2028 across five image-wise
folds. Only after a smoke run and a full fold have been checked, the complete
campaign can be launched explicitly:

```powershell
& '.\scripts\run_3x5.ps1' -RunId 'my_campaign' -RawData $RawData -Python $Python -Resume -ConfirmFullCampaign
```

The campaign writes run-specific outputs under `artifacts/`, `results/`, and
`reports/`, then creates an aggregate report and figures after the runs pass.
All of these outputs remain local. This is **image-wise internal
cross-validation**, not validation on unseen plots, sites, seasons, or devices.

## 4. Run the local chatbot

The only web app is [`wheatspectra_app/`](wheatspectra_app/README.md). It uses
the completed `stage42d_exact_v1` seed 2026/fold 0 checkpoint for live
inference; the 15-run aggregate is an evaluation result, not a deployable
checkpoint. The local PhytoCore checkpoint, fold normalisation files, VLM
bundle, selected run report, and PASS aggregate report must be available under
`artifacts/stage42d_exact_v1/` and `reports/stage42d_exact_v1/`. These files
are not included in Git. Without them, image inference fails closed rather
than substituting another model.

From the repository directory, install the additional backend packages and
frontend dependencies:

```powershell
& $Python -m pip install -r '.\wheatspectra_app\backend\requirements.txt'
Push-Location '.\wheatspectra_app\frontend'
npm ci
Pop-Location
```

Open two PowerShell windows in `wheatspectra_app/`. Set `PHYTO_PYTHON` to the
Python 3.11 environment installed above, then start the backend in one window
and the frontend in the other:

```powershell
$env:PHYTO_PYTHON = (Resolve-Path '..\.venv\Scripts\python.exe').Path
& '.\start_backend.bat'
```

```powershell
& '.\start_frontend.bat'
```

Open `http://127.0.0.1:5173`; the API listens on `http://127.0.0.1:8000`.
Upload a 48 × 352 × 204 (or band-first 204 × 48 × 352) unsigned 16-bit TIFF
image. For the full upload, evidence, knowledge, and chat behaviour, see the
[app instructions](wheatspectra_app/README.md). To use a nondefault API
address, copy `wheatspectra_app/frontend/.env.example` to a local `.env` and
set `VITE_API_URL`. Local `.env` files are ignored by Git.

## Tests and publication notes

The two legacy tests `tests/test_chatbot_upload.py` and `tests/test_query.py`
still import the removed app and need migration before the entire root test
suite can run. To run the remaining research tests and the current frontend
tests:

```powershell
& $Python -m pytest -q --ignore=tests/test_chatbot_upload.py --ignore=tests/test_query.py
Push-Location '.\wheatspectra_app\frontend'
npm ci
npm test
npm run build
Pop-Location
```

The optional chatbot-routing notebook and the source-freeze file list in
`src/phytospectra/provenance.py` also still refer to the removed app. Update
those references before using the notebook or generating a new release
manifest; otherwise, the manifest will omit the current app source.

Do not upload the contents of the ignored output directories or a local model
cache separately. `.gitignore` protects *untracked* files; if private data or
weights were already committed to a Git repository, remove them from Git's
index/history before publishing. The GitHub repository can be named
`Phyto-VLMChatbot`; no remote repository or publication is created by these
instructions. Before release, replace the placeholder author in
`CITATION.cff` and add the actual GitHub repository URL there.

## Citation and licence

Please cite [Laprade et al. (2024)](https://openaccess.thecvf.com/content/CVPR2024W/FGVC11/html/Laprade_HyperLeaf2024_-_A_Hyperspectral_Imaging_Dataset_for_Classification_and_Regression_CVPRW_2024_paper.html)
for HyperLeaf2024 and cite the accompanying PhytoSpectra-VLM publication when
available. The source-code licence is in `LICENSE`; the dataset and third-party
model are governed by their respective terms.
