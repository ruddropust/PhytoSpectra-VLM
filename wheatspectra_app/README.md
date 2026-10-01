# WheatSpectraChat on research_v3

This app keeps the original WheatSpectraChat React UI, upload flow, knowledge
answers, multi-turn routing, target-specific evidence panels, and fail-closed
answer contract. Its live image model is the completed `research_v3`
`stage42d_exact_v1` **seed 2026 / fold 0** PhytoCore + VLM bundle. The other
14 folds are evaluation runs; their aggregate is not a single deployment
checkpoint. The app does not claim validation on new plots, sites, seasons,
or devices.

The original `E:\Habib\PhytoChat\wheatspectra_app` is unchanged. No model
weights or datasets were copied into this app. The adapter reads them from
`research_v3/artifacts/stage42d_exact_v1` and checks the selected checkpoint,
visual bridge, VLM manifest, selected run report, and 15/15 PASS aggregate
before serving image inference. A missing or altered artifact fails closed;
the old research_v2 model is not silently substituted.

## Local setup

Use Python 3.11 with the repository dependencies and the extra backend
packages. From the `research_v3` directory:

```powershell
& 'E:\Habib\.ruddro\Scripts\python.exe' -m pip install -e '.[app]'
& 'E:\Habib\.ruddro\Scripts\python.exe' -m pip install -r '.\wheatspectra_app\backend\requirements.txt'
Set-Location '.\wheatspectra_app\frontend'
npm ci
```

The base Qwen2.5-1.5B-Instruct weights must already be available in the
local Hugging Face cache; the frozen VLM loader uses local files only. The
optional knowledge retriever also uses its locally cached embedding model.
If that model is absent, image inference and curated target definitions remain
available.

## Start the app

Open two terminals in `research_v3/wheatspectra_app`. Run
`start_backend.bat` in one and `start_frontend.bat` in the other. The backend
uses `E:\Habib\.ruddro\Scripts\python.exe` by default; on another machine,
set `PHYTO_PYTHON` to the Python 3.11 executable before launching. Open
<http://127.0.0.1:5173>. The API listens at <http://127.0.0.1:8000>.
Stop any earlier app using those ports before starting this one. The model
status endpoint should report `RESEARCH_V3_STAGE42D_EXACT`.

Upload a HyperLeaf-style unsigned 16-bit TIFF with shape
`204×48×352` or `48×352×204`. The upload route derives the leaf mask,
returns a preview, and prepares five PhytoCore predictions and target-specific
ROI visuals. The chat route uses the selected model's Qwen symbolic-plan
bundle; the verified answer values are resolved from PhytoCore, not copied
from free-form model text. Model-sensitive ROIs are not biological causal
evidence. For repeated app uploads, candidate windows are keyed by a hash
of the cube rather than the random session ID. Consequently, app ROIs need
not be identical to saved campaign ROIs, which were keyed by dataset image
identifiers. The status panel labels the active seed and fold.

The generated media, datasets, model weights, local configuration, and
frontend dependencies are ignored by Git. A GitHub clone needs the local
artifacts supplied separately before image inference will work.
