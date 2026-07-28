# Repository Inventory

**Snapshot:** 2026-07-28 (SP-000)  
**Purpose:** distinguish canonical future production paths from legacy references and non-source local assets. This document inventories rather than relocates or deletes any existing file.

## Canonical target layout

| Path | Intended role | Status at snapshot |
| --- | --- | --- |
| `backend/src/cocoa_platform/__init__.py` | Canonical package marker | Present; inert SP-000 boundary marker with no imports or feature behavior |
| `backend/src/cocoa_platform/` | New FastAPI platform package | Reserved for later owned tasks; package content does not exist yet |
| `backend/tests/` | New backend automated tests | Reserved; created by later owned tasks |
| `frontend/src/` | React/Vite application | Existing reference frontend; later tasks own feature paths |
| `supabase/migrations/` | Supabase schema migrations | Reserved |
| `docs/` | PRD, contracts, implementation records | Active source-of-truth documentation |

## Existing references (read-only during migration)

| Path | Observed contents | Treatment |
| --- | --- | --- |
| `app/` | Flask-oriented CPU PTH/ONNX applications and benchmark scripts | Legacy reference; do not move or rewrite before feature-parity review |
| `backend/main.py`, `backend/models.py` | FastAPI prototype with import-time models and hard-coded historical paths | Legacy reference; not the supported canonical API |
| `frontend/` | Vite/React project with `package.json` and lockfile | Existing frontend reference; preserve outside explicit task ownership |
| `annotation/yolov12-main/` | Third-party/experiment YOLO source tree | Legacy research reference; review licensing and provenance before any promotion |
| root `*.py` files | Experimental checks, exports, analysis, and test scripts | Unowned legacy references; do not treat as production source |

## Local assets excluded from Git

| Category | Present examples | Policy |
| --- | --- | --- |
| Model weights | root `.pth`, `.pt`, `.onnx`, and `.onnx.data` files | Acquire from an approved registry/source; validate hash before use; never commit |
| Datasets | `classifier_dataset-defect/`, `defect_mix_new_old/`, `Test_Color/`, `test_data_defect/` and ZIP archives | Local-only; future importer validates a ZIP containing `images/` and YOLO `labels/` |
| Virtual environments | `.venv/`, `env_cocoa/`, `.conda/` | Recreate from manifests; never commit |
| Generated output | `runs/`, `adjusted_output/`, `aug_demo/`, `debug_crops/`, confusion matrices, root images | Store outside Git; publish only approved documentation/evidence |
| Archives and documents | root ZIP, PDF, PPTX, DOCX files | Local research/support material; not canonical source |

## Known baseline blockers

1. At the start of SP-000, the Git repository was on `master` with no commits. This inventory accompanies the reviewed SP-000 baseline commit.
2. The legacy FastAPI prototype imports `models.py`, which loads PTH/YOLO assets during module import through hard-coded historical paths. It is not part of the canonical import smoke test; the canonical marker has no imports.
3. Model/dataset source URLs, licenses, and role mapping for the existing root assets are not recorded. The listed checksums identify local files only; they do not establish provenance.
4. The final CPU/GPU dependency matrix and GTX 1080 Ti CUDA compatibility evidence are owned by later runtime/hardware tasks.
