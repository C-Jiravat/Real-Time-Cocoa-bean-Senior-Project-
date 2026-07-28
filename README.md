# Cocoa Bean AI Platform

This repository is being rebuilt as a local-first FastAPI and React/Vite platform for cocoa-bean analysis. The product supports independent PTH and ONNX workflows, image analysis, browser-camera capture, Model Lab, benchmarking, HTML reports, durable local history, and later Supabase synchronization. The source of truth is [the PRD](docs/PRD_Cocoa_Bean_AI_Platform_EN.md).

## Repository status

SP-000 establishes the safe baseline only. `app/`, `backend/main.py`, and `backend/models.py` are legacy references, not the supported MVP backend: they contain import-time model loading and absolute Windows paths. New production code belongs under `backend/src/cocoa_platform/`; the canonical API and frontend modules will be added by their assigned tasks. Do not move, delete, or commit legacy weights, datasets, uploads, virtual environments, or generated reports.

See [the repository inventory](docs/REPOSITORY_INVENTORY.md) for what is present today and [the artifact manifest](docs/ARTIFACT_MANIFEST.md) for acquisition and checksum records.

## Bootstrap

Windows PowerShell, from the repository root:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r backend\requirements.txt
Copy-Item .env.example .env
```

Verify the canonical package boundary before starting feature work:

```powershell
.\.venv\Scripts\python -B -c "from pathlib import Path; import cocoa_platform; p=Path(cocoa_platform.__file__).resolve(); e=(Path.cwd()/'backend/src/cocoa_platform/__init__.py').resolve(); assert p == e, (p, e); print(p)"
```

This smoke test only imports the inert package marker. It does not import a legacy
module, load models, start an API, or validate inference behavior.

Install the optional PTH/ONNX runtime only when working on the model tasks:

```powershell
.\.venv\Scripts\python -m pip install -r backend\requirements-ml.txt
```

For NVIDIA GTX 1080 Ti acceptance, use the GPU compatibility instructions produced by the runtime/hardware tasks. The baseline is CPU-oriented and does not claim CUDA validation.

## Start commands

The current React/Vite reference frontend can be built or started with:

```powershell
Set-Location frontend
npm.cmd ci
npm.cmd run build
npm.cmd run dev
```

No canonical API exists at the SP-000 baseline. A later assigned task will provide
its supported start command. Until then, the legacy FastAPI prototype can be
inspected only (it is expected to fail to import without its historical model files):

```powershell
Set-Location backend
..\.venv\Scripts\python -m uvicorn main:app --reload --port 8000
```

## Local data and retention

Keep model files, datasets, uploaded images, and generated artifacts outside Git. Set `COCOA_DATA_ROOT`, `COCOA_MODEL_ROOT`, and `COCOA_ARTIFACT_ROOT` in `.env`. The final product retains datasets and result artifacts until the user explicitly deletes them; dataset edits create immutable versions. It must not persist original input or benchmark images to Supabase Storage.

## Contribution boundaries

- Follow the path ownership and integration gates in [the implementation backlog](docs/MULTI_AGENT_IMPLEMENTATION_TASKS.md).
- Do not change frozen contracts locally; create a contract-change task instead.
- PTH and ONNX benchmarks and GPU validation are independent runs. A GTX 1080 Ti result is valid only with recorded real-hardware evidence.
- Dataset ZIP ingestion must contain `images/` and YOLO-format `labels/`, and must be validated before extraction.
