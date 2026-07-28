# Cocoa Bean AI Platform — Multi-Agent Implementation Backlog

| Item | Value |
| --- | --- |
| Backlog version | 1.0 |
| Source of truth | `docs/PRD_Cocoa_Bean_AI_Platform_EN.md` version 1.1 |
| Repository | `D:\Chula\Senior_Project` |
| Target architecture | FastAPI + React/Vite + local registries + Supabase |
| Required GPU acceptance hardware | NVIDIA GeForce GTX 1080 Ti |

## 1. Purpose

This backlog divides the PRD into bounded tasks that multiple agents can implement without competing for the same files. Tasks are grouped into waves and integration gates. A task may start when the dependencies and gates explicitly listed on that task are complete; wave numbering is organizational and does not create an implicit dependency.

The current `app/` Flask applications and the current `backend/main.py` and `backend/models.py` FastAPI prototype are legacy references. New production code must be built under the canonical source directories defined below. Legacy files must not be moved or deleted until final feature-parity review.

## 2. Mandatory collaboration rules

1. Complete `SP-000` and Integration Gate `G0` before parallel implementation.
2. Each agent may modify only the paths assigned to its task.
3. Shared contracts, route tables, migrations, dependency manifests, lockfiles, and global frontend layout have one named owner.
4. If a frozen contract must change, stop the dependent task and open a contract-change task. Do not redefine the contract locally.
5. Do not commit model weights, datasets, uploaded files, reports, secrets, virtual environments, or generated runtime artifacts.
6. Do not move or rewrite legacy applications during feature development.
7. Every handoff must include changed files, commands executed, test results, assumptions, and unresolved blockers.
8. GPU work is accepted only with evidence from a real GTX 1080 Ti.
9. Agents sharing a working tree must work in non-overlapping paths and must not switch branches while another agent is active.
10. The Integration Agent merges or wires modules only after the relevant gate passes.
11. Sequential ownership transfer is allowed only when explicitly stated. `README.md` transfers from SP-000 to SP-080 after SP-070 acceptance.

## 3. Canonical target paths

```text
backend/
├── src/cocoa_platform/
│   ├── contracts/
│   ├── inference/
│   ├── grading/
│   ├── datasets/
│   ├── benchmark/
│   ├── runtime/
│   ├── model_lab/
│   ├── reports/
│   ├── persistence/
│   └── api/
└── tests/

frontend/src/
├── app/
├── lib/
└── features/

supabase/
├── migrations/
└── tests/
```

## 4. Wave 0 — Safe repository baseline

### SP-000 Repository bootstrap

- **Agent role:** Repository Steward
- **Dependencies:** None
- **Owned paths:** `.gitignore`, `.env.example`, `README.md`, `pyproject.toml`, `backend/requirements*.txt`, `backend/src/cocoa_platform/__init__.py`, `docs/REPOSITORY_INVENTORY.md`, `docs/ARTIFACT_MANIFEST.md`
- **Scope:** Establish the baseline repository, dependency entry points, artifact policy, and canonical source paths. Preserve current applications as references.
- **Deliverables:** Reviewed initial commit, repository inventory, model/dataset acquisition manifest with hashes, environment template, documented backend and frontend bootstrap commands.
- **Package-boundary constraint:** SP-000 may create only the inert root package marker `backend/src/cocoa_platform/__init__.py` under the canonical source root. It must not import or define contracts, APIs, runtimes, or feature behavior. No later task owns this file without an explicit ownership transfer.
- **Acceptance:**
  - Git tracks only intended source and documentation.
  - Secrets, virtual environments, uploads, datasets, weights, archives, and generated reports are excluded.
  - After installing `backend/requirements.txt`, import `cocoa_platform` succeeds, resolves to `backend/src/cocoa_platform/__init__.py`, and imports no legacy module or feature subpackage.
  - `npm.cmd run build` passes or a concrete pre-existing blocker is recorded.

### SP-005 MVP operational limits

- **Agent role:** Product Contract Facilitator
- **Dependencies:** SP-000
- **Owned paths:** `docs/contracts/OPERATIONAL_LIMITS.md`
- **Scope:** Convert every remaining PRD limit into an approved configuration value before feature agents implement upload, deletion, and performance behavior.
- **Required decisions:** Maximum compressed ZIP size, extracted size, file count, PTH/ONNX bundle size, deletion recovery policy, optional comparison warning thresholds, image-analysis latency budget, and Live Camera FPS target.
- **Acceptance:** Every limit has a value, unit, owner, intended configuration key, validation rule, and acceptance-test boundary. Unapproved values remain explicit blockers rather than agent-selected defaults. SP-010 converts this approved document into typed code.

### Gate G0 — Baseline safe

- A reviewed initial commit exists.
- Canonical production paths are approved.
- SP-005 operational limits are approved.
- Every next-wave agent has explicit path ownership.
- Legacy applications, root weights, and datasets are read-only references.

## 5. Wave 1 — Contracts and core correctness

### SP-010 Domain and API contracts

- **Agent role:** Contract Architect
- **Dependencies:** G0
- **Owned paths:** `backend/src/cocoa_platform/contracts/**`, `backend/tests/contracts/**`, `docs/contracts/API_CONTRACTS.md`, `docs/contracts/MODEL_CONTRACTS.md`, `docs/contracts/DATA_CONTRACTS.md`
- **Scope:** Define typed detection, classification, grading, model, runtime, dataset, benchmark, job, persistence, and error contracts.
- **Required decisions:**
  - Color: `0 Purple`, `1 Brown`.
  - Defect: `0 Normal`, `1 Germinate`, `2 Slaty / Hard as rock`, `3 Moldy`.
  - Stable internal defect keys: `normal`, `germinate`, `slaty_hard_as_rock`, `moldy`.
  - Every valid bean has Color top-1 and Defect top-1 with confidence.
  - PTH and ONNX benchmarks are separate run records.
- **Acceptance:**
  - Serialization round trips pass.
  - Invalid confidence, bounding boxes, class IDs, devices, and providers are rejected.
  - JSON examples and schemas exist for every public API result.
  - No dependent module defines duplicate enums or schemas.

### Gate G1A — Contracts frozen

- SP-010 schemas, enums, errors, examples, and version identifiers are reviewed.
- SP-005 operational limits are represented in the public contracts.
- Grade, inference, dataset, and persistence agents may now work in parallel without redefining shared contracts.

### SP-011 Grade engine

- **Agent role:** Domain Logic Engineer
- **Dependencies:** G1A
- **Owned paths:** `backend/src/cocoa_platform/grading/**`, `backend/tests/unit/grading/**`
- **Scope:** Implement grade calculation as pure domain logic.
- **Acceptance:**
  - `N` equals the number of unique YOLO bounding boxes after confidence filtering and NMS.
  - A downstream invalid crop or classifier failure does not reduce `N`; it marks the run incomplete.
  - A bean that is both Purple and Slaty / Hard as rock contributes once to `c2`.
  - `N = 0`, incomplete classification, exact threshold, and above-threshold cases are tested.
  - An incomplete run has `grade = null` and never displays a provisional grade.
  - Raw percentages are evaluated before display rounding.
  - Every result records `grade_standard_version` and per-bean contribution evidence.

### SP-012 Shared preprocessing and inference core

- **Agent role:** ML Runtime Engineer
- **Dependencies:** G1A
- **Owned paths:** `backend/src/cocoa_platform/inference/**`, `backend/tests/unit/inference/**`
- **Scope:** Build the canonical `YOLO → crop → Color ConvNeXt + Defect ConvNeXt` pipeline and shared preprocessing.
- **Acceptance:**
  - PTH Detector `.pt` and classifier `.pth` adapters work on CPU.
  - ONNX Detector and classifiers, including external data, work on CPUExecutionProvider.
  - Both classifiers return real top-1 confidence, not a placeholder.
  - Invalid crops and non-finite outputs use the shared incomplete-result contract.
  - No absolute model path exists in production code.
  - Explicit runtime selection never silently falls back.

### SP-013 Runtime Manager and Job Controller

- **Agent role:** Backend Concurrency Engineer
- **Dependencies:** G1A, SP-012
- **Owned paths:** `backend/src/cocoa_platform/runtime/**`, `backend/tests/unit/runtime/**`
- **Scope:** Implement transactional model-bundle activation, runtime capability reporting, job identity, cancellation, and stale-result protection.
- **Acceptance:**
  - A failed activation preserves the prior active bundle.
  - An old generation cannot overwrite a newer result.
  - Requested GPU failure returns an actionable error.
  - Runtime state reports model hashes, requested/actual device, provider, and relevant versions.
  - Loading is lazy and unload releases CPU/GPU resources.

### SP-014 Local Run Store and Persistence Port

- **Agent role:** Local Persistence Engineer
- **Dependencies:** G1A
- **Owned paths:** `backend/src/cocoa_platform/persistence/ports/**`, `backend/src/cocoa_platform/persistence/local/**`, `backend/tests/unit/persistence_local/**`
- **Scope:** Define the persistence interface used by APIs and implement the durable local run ledger, local artifact references, idempotency keys, retention state, and pending-sync state before Supabase exists.
- **Acceptance:**
  - Analysis and benchmark jobs write a durable local record before execution.
  - Restart recovery preserves `processing`, `completed`, `failed`, and `pending_sync` states.
  - Duplicate idempotency keys do not create duplicate runs.
  - History, delete, and sync-status APIs can operate against the local port before the Supabase adapter is installed.

### Gate G1 — Core contract locked

- Contracts are reviewed and versioned.
- Grade fixtures pass.
- PTH/CPU and ONNX/CPU use the same inference interface.
- The Runtime Manager and durable local persistence port pass their unit tests.
- Dependent agents may not modify frozen contracts directly.

## 6. Wave 2 — Dataset, benchmark, Model Lab, and reports

### SP-020 Dataset ZIP importer and Local Dataset Registry

- **Agent role:** Dataset Platform Engineer
- **Dependencies:** G1A
- **Owned paths:** `backend/src/cocoa_platform/datasets/**`, `backend/tests/unit/datasets/**`, `backend/tests/fixtures/dataset_archives/**`
- **Scope:** Import ZIP datasets with `images/` and YOLO-format `labels/`, validate them, create immutable versions, and retain them locally until explicit deletion.
- **Acceptance:**
  - An individual image larger than 50 MB is rejected.
  - Same-stem image/label pairing is enforced.
  - Coordinates outside `[0,1]`, wrong class IDs, corrupt images, missing labels, orphan labels, and duplicate normalized paths are rejected or reported according to the contract.
  - ZIP traversal, absolute paths, symlinks, decompression bombs, excessive file counts, and expanded-size violations are rejected.
  - Missing image/label pairs and orphan label files are rejected after their validation errors are reported.
  - Extraction streams to disk.
  - The canonical hash uses sorted relative paths and file contents.
  - Editing creates a new immutable version and hash.
  - Historical runs retain the old dataset reference after deletion.

### SP-021 End-to-end benchmark engine

- **Agent role:** ML Evaluation Engineer
- **Dependencies:** SP-010, SP-012, SP-020
- **Owned paths:** `backend/src/cocoa_platform/benchmark/**`, `backend/tests/unit/benchmark/**`, `backend/tests/fixtures/benchmark/**`
- **Scope:** Evaluate Detector-only and `YOLO → crop → selected ConvNeXt → one-to-one matching/evaluation` Color/Defect workflows for separate PTH and ONNX runs.
- **Acceptance:**
  - A prediction and ground truth participate in at most one match.
  - Correct, wrong-class, duplicate, false-positive, and false-negative golden fixtures return exact expected values.
  - Detector metrics are separated from end-to-end classification metrics.
  - Detector-only fixtures validate AP50 and 101-point mAP50-95.
  - False-positive, false-negative, duplicate, and wrong-class gallery artifacts are generated deterministically.
  - Run records include model hashes, dataset hash, class map, thresholds, preprocessing version, metric version, runtime, and timing.
  - PTH and ONNX generate independent results and reports.
  - Optional comparison refuses incompatible dataset hashes or experiment configurations.

### SP-022 Model Lab backend

- **Agent role:** Model Validation Engineer
- **Dependencies:** SP-010, SP-012, SP-013, SP-021
- **Owned paths:** `backend/src/cocoa_platform/model_lab/**`, `backend/tests/unit/model_lab/**`
- **Scope:** Implement PTH and ONNX candidate workspaces, inspection, validation, smoke testing, pipeline testing, benchmarking, temporary activation, and profile saving.
- **Acceptance:**
  - PTH supports recognized YOLO `.pt` and ConvNeXt `.pth` contracts without executing arbitrary uploaded code.
  - ONNX supports a self-contained graph or a safe graph-plus-external-data bundle.
  - External-data references cannot escape the candidate workspace.
  - Wrong role, class count, tensor contract, or architecture is rejected.
  - Candidate testing does not change the active bundle.
  - Failed temporary activation rolls back.
  - Hash and metadata are recorded before execution.
  - Inspection and smoke testing run in a bounded worker process with timeout, memory/output limits, termination cleanup, and adversarial resource-exhaustion tests.

### SP-023 HTML reporting

- **Agent role:** Reporting Engineer
- **Dependencies:** SP-010, SP-011, SP-021
- **Owned paths:** `backend/src/cocoa_platform/reports/**`, `backend/templates/reports/**`, `backend/tests/unit/reports/**`
- **Scope:** Produce deterministic Analysis, Benchmark, and optional Comparison HTML reports plus JSON and CSV exports.
- **Acceptance:**
  - Reports show model and dataset hashes, runtime, device/provider, grade version, configuration, and timestamp.
  - User-controlled strings are escaped.
  - A downloaded HTML report opens without the backend.
  - PTH and ONNX are never presented as one combined benchmark result.
  - Detector and end-to-end error galleries are linked or embedded in the applicable report.

### Gate G2 — Measurement trusted

- Gate G1 has passed.
- Dataset security and hashing tests pass.
- Grade and benchmark golden fixtures pass.
- Both Model Lab formats can be tested without replacing the active bundle.
- HTML reports render from frozen result contracts.

## 7. Wave 3 — FastAPI composition and live processing

### SP-030 FastAPI application

- **Agent role:** API Integration Engineer
- **Dependencies:** G2
- **Owned paths:** `backend/src/cocoa_platform/api/routes/system.py`, `backend/src/cocoa_platform/api/routes/runtime.py`, `backend/src/cocoa_platform/api/routes/analysis.py`, `backend/src/cocoa_platform/api/routes/datasets.py`, `backend/src/cocoa_platform/api/routes/benchmarks.py`, `backend/src/cocoa_platform/api/routes/model_lab.py`, `backend/src/cocoa_platform/api/routes/reports.py`, `backend/src/cocoa_platform/api/routes/history.py`, `backend/src/cocoa_platform/api/dependencies/core.py`, `backend/tests/api/rest/**`
- **Scope:** Implement typed REST routes for runtime, analysis, datasets, benchmark, Model Lab, reports, and local-first history/delete/sync status against frozen service and persistence ports.
- **Acceptance:**
  - OpenAPI contract tests pass.
  - Upload limits and structured validation errors are enforced.
  - Startup does not eagerly load every model.
  - CORS uses configuration rather than wildcard origins with credentials.
  - Original images and benchmark datasets are never sent to Supabase.
  - History, delete, and sync-status routes work against the durable local store before Supabase is configured.

### SP-031 Live Camera WebSocket

- **Agent role:** Real-Time Backend Engineer
- **Dependencies:** SP-013, SP-030
- **Owned paths:** `backend/src/cocoa_platform/api/live/**`, `backend/src/cocoa_platform/api/routes/live.py`, `backend/tests/api/live/**`
- **Scope:** Implement frame, result, progress, camera/session configuration, telemetry, rate-limit, backpressure, cancellation, and disconnect behavior.
- **Acceptance:**
  - Latest-frame-wins is enforced.
  - Oversized and malformed frames are rejected.
  - Disconnect releases jobs and runtime resources.
  - Stale results cannot replace newer frame results.
  - Live and still-image analysis use the same inference core.
  - A confirmed-snapshot message produces one gradeable and saveable analysis result; ordinary live frames are never persisted automatically.
  - The message contract supports selected camera-device metadata, resolution, target inference FPS, latency, processed FPS, and dropped-frame counts.

### SP-032 Backend API composition

- **Agent role:** Backend Integration Owner
- **Dependencies:** SP-030, SP-031
- **Owned paths:** `backend/src/cocoa_platform/api/app.py`, `backend/src/cocoa_platform/api/router.py`, `backend/src/cocoa_platform/api/composition/persistence_hook.py`, `backend/src/cocoa_platform/main.py`, `backend/tests/api/composition/**`
- **Scope:** Wire REST and WebSocket routes, lifecycle hooks, dependency factories, CORS configuration, and the canonical backend entry point without changing frozen endpoint contracts.
- **Acceptance:**
  - All route modules are mounted exactly once.
  - OpenAPI and WebSocket message contracts are frozen for frontend work.
  - The canonical start command works without importing legacy `backend/main.py` or eagerly loading every model.
  - The application uses `backend/src/cocoa_platform/api/composition/persistence_hook.py`, owned by SP-032, as an explicit provider extension point with the SP-014 local adapter as the default.

### Gate G3 — Backend complete on CPU

- One documented backend command starts the application.
- SP-032 composition tests pass.
- Image, Dataset ZIP, Benchmark, both Model Lab formats, reports, and Live WebSocket pass integration tests.
- PTH/CPU and ONNX/CPU pass end-to-end smoke tests.
- OpenAPI is frozen for frontend implementation.

## 8. Wave 4 — React frontend

### SP-040 Frontend foundation

- **Agent role:** Frontend Integrator
- **Dependencies:** G3
- **Owned paths:** `frontend/src/App.jsx`, `frontend/src/app/**`, `frontend/src/lib/**`, `frontend/src/main.jsx`, `frontend/src/styles/**`
- **Scope:** Create navigation, API/WebSocket clients, error boundaries, global layout, and an initial route registry. Final feature-route mounting belongs to SP-045.
- **Acceptance:** Frontend lint and build pass; feature modules use public exports; feature agents do not edit global layout or the route registry.

### SP-041 Image Analysis and Live Camera UI

- **Agent role:** Frontend Analysis Engineer
- **Dependencies:** SP-031, SP-040
- **Owned paths:** `frontend/src/features/analysis/**`, `frontend/src/features/live-camera/**`
- **Acceptance:**
  - Image validation enforces 50 MB.
  - The user can select model bundle and device and configure YOLO confidence and IoU within approved limits.
  - Per-bean Color and Defect top-1/confidence are shown.
  - Counts, percentages, runtime, timing, actual device, provider, annotations, and grade are visible.
  - Image results are saved only after an explicit action through the local-first persistence workflow.
  - Camera device, resolution, target FPS, Start, Pause, and Stop controls work.
  - Camera permission, absence, disconnect, cleanup, Canvas overlay, latency, processed FPS, and dropped-frame states work.
  - The user can explicitly confirm one live snapshot for grading and persistence; ordinary frames are not saved.
  - Annotated image, JSON, and CSV exports work for still-image and confirmed-snapshot results.

### SP-042 Dataset, Benchmark, and Comparison UI

- **Agent role:** Frontend Evaluation Engineer
- **Dependencies:** SP-020, SP-021, SP-023, SP-040
- **Owned paths:** `frontend/src/features/datasets/**`, `frontend/src/features/benchmark/**`, `frontend/src/features/comparison/**`
- **Acceptance:**
  - ZIP validation summary, class distribution, missing/orphan labels, version, and hash are visible.
  - PTH and ONNX runs are clearly separate.
  - Comparison is blocked or warned when conditions differ.
  - HTML reports can be downloaded.
  - Detector-only AP50/mAP50-95 and detector/end-to-end error galleries render separately.

### SP-043 Model Lab UI

- **Agent role:** Frontend Model Tools Engineer
- **Dependencies:** SP-022, SP-040
- **Owned paths:** `frontend/src/features/model-lab/**`
- **Acceptance:**
  - Separate PTH and ONNX tabs explain accepted files.
  - ONNX external-data bundles display their component files.
  - Inspect, test, benchmark, temporary activate, and save-profile actions are distinct.
  - Compare with Active is available when dataset and experiment conditions match and explains incompatible conditions when they do not.
  - Activation requires confirmation and shows rollback failures.

### SP-044 Dashboard and Runtime Management UI

- **Agent role:** Frontend Runtime Engineer
- **Dependencies:** SP-013, SP-040
- **Owned paths:** `frontend/src/features/dashboard/**`, `frontend/src/features/runtime/**`
- **Scope:** Implement health cards, latest-result summaries, model registry, active-bundle membership, capabilities, Validate, Warm-up, Activate, and Unload workflows.
- **Acceptance:** The Dashboard and Runtime pages show model hashes, bundle members, requested/actual device, provider, GPU availability, active grade-standard version, thresholds, read-only configuration source/status, actionable failures, and confirmation for disruptive runtime actions.

### SP-051 History UI

- **Agent role:** Frontend History Engineer
- **Dependencies:** SP-030, SP-040
- **Owned paths:** `frontend/src/features/history/**`
- **Scope:** Implement History against the local-first API contract; Supabase synchronization is an adapter concern and must not change the page contract.
- **Acceptance:** Filtering, details, delete confirmation, manual sync retry, sync states, signed/local downloads, expired-link handling, and source-dataset-unavailable history all work.

### SP-045 Frontend route integration

- **Agent role:** Frontend Integrator
- **Dependencies:** SP-041, SP-042, SP-043, SP-044, SP-051
- **Owned paths:** `frontend/src/App.jsx`, `frontend/src/app/routes/**`, `frontend/tests/integration/routes/**`
- **Scope:** Mount all completed feature modules and run cross-feature navigation and runtime-state integration tests. This is the ownership transfer of route wiring from SP-040.
- **Acceptance:** Every MVP page is reachable, direct navigation and error boundaries work, and frontend lint/build plus route integration tests pass.

### Gate G4 — User workflows complete

- Frontend lint and build pass.
- SP-045 route integration passes.
- Dashboard, Runtime Management, Image Analysis, Live Camera, Dataset ZIP, Detector/Color/Defect Benchmark, Comparison, both Model Lab tabs, and History pass browser smoke tests.
- Format, actual device, and provider appear on all primary pages.

## 9. Wave 5 — Supabase, offline durability, and history

### SP-050 Supabase schema and repository

- **Agent role:** Persistence Engineer
- **Dependencies:** G3
- **Owned paths:** `supabase/**`, `backend/src/cocoa_platform/persistence/supabase/**`, `backend/src/cocoa_platform/persistence/outbox_sync/**`, `backend/tests/integration/supabase/**`, `docs/SUPABASE_SETUP.md`
- **Scope:** Implement migrations, private Storage, Supabase repositories, signed URLs, synchronization from the SP-014 local ledger, retries, and deletion policies.
- **Acceptance:**
  - RLS is enabled and the service-role key remains backend-only.
  - Original input and dataset images are not uploaded.
  - Supabase outage does not lose an inference or benchmark result.
  - Retry is idempotent.
  - Dataset edits create versions rather than mutating hashes.
  - Explicit deletion preserves or marks historical references according to the PRD.

### SP-052 Supabase adapter integration

- **Agent role:** Persistence Integration Engineer
- **Dependencies:** SP-014, SP-030, SP-050
- **Owned paths:** `backend/src/cocoa_platform/api/composition/supabase_provider.py`, `backend/tests/api/history_persistence/**`
- **Scope:** Implement the Supabase provider loaded through the SP-032 persistence hook without editing application composition or changing REST/frontend contracts.
- **Acceptance:** The production application selects the Supabase provider through configuration and demonstrably uses it; local-only and Supabase-enabled configurations pass the same API contract tests; online sync, offline queue, retry, signed downloads, delete, and source-unavailable history paths work end to end.

### Gate G5 — Durable product

- Gate G4 has passed.
- Online persistence and offline retry pass.
- SP-052 adapter integration passes without an OpenAPI change.
- Signed private downloads pass.
- History browser tests from G4 continue to pass in Supabase-enabled mode.
- Retention, versioning, and deletion behavior match the PRD.

## 10. Wave 6 — GTX 1080 Ti validation

### SP-060 GPU compatibility and validation

- **Agent role:** GPU Validation Engineer
- **Dependencies:** G4, G5
- **Owned paths:** `backend/tests/hardware/**`, `docs/GPU_COMPATIBILITY.md`, `artifacts/validation/gpu/**`
- **Scope:** Validate all required flows on a real NVIDIA GeForce GTX 1080 Ti.
- **Acceptance:**
  - PTH uses PyTorch CUDA.
  - ONNX uses CUDAExecutionProvider.
  - Image Analysis, Live Camera, Benchmark, and both Model Lab tabs pass GPU smoke tests.
  - GPU, VRAM, driver, CUDA, cuDNN, PyTorch, ONNX Runtime, and provider versions are recorded.
  - Explicit GPU requests never silently fall back.
  - PTH and ONNX reports remain independent.
  - Mixed precision remains disabled until separately approved and validated.
  - PTH AMP/FP16 is compared only with the PTH FP32 baseline, and ONNX FP16 only with the ONNX FP32 baseline; neither comparison is an independent PTH-versus-ONNX benchmark pass criterion.

### Gate G6 — Hardware accepted

The following matrix passes with rerunnable evidence:

| Format | CPU | GTX 1080 Ti |
| --- | --- | --- |
| PTH | Pass | Pass using PyTorch CUDA |
| ONNX | Pass | Pass using CUDAExecutionProvider |

## 11. Wave 7 — Independent QA and release integration

### SP-070 End-to-end QA and security

- **Agent role:** Independent QA Agent
- **Dependencies:** G4, G5, G6
- **Owned paths:** `backend/tests/e2e/**`, `frontend/e2e/**`, `docs/TEST_EVIDENCE.md`
- **Scope:** Test fresh-checkout setup, primary user journeys, failure recovery, and security boundaries. Production defects are opened as separate fix tasks.
- **Acceptance:** Cover 50 MB image boundary, malicious ZIP, incompatible PTH/ONNX model, camera disconnect, stale job, Supabase outage/recovery, signed URLs, and the full CPU/GPU matrix.

### SP-080 Documentation and legacy retirement

- **Agent role:** Release Integrator
- **Dependencies:** SP-070
- **Owned paths:** `README.md`, `docs/release/**`, `legacy/**`, `docs/release/LEGACY_RELOCATION_MANIFEST.md`
- **Scope:** After the SP-070 acceptance gate, take ownership of `README.md` from SP-000, finalize operations documentation, and move obsolete applications only after feature parity is demonstrated.
- **Acceptance:**
  - A fresh Windows checkout follows the README successfully.
  - Backend and frontend each have one canonical start command.
  - Model and dataset acquisition and hashes are documented.
  - Production code no longer imports legacy modules.
  - Every MVP acceptance criterion links to current test evidence.

## 12. Critical path and safe parallel work

```text
SP-000 → SP-005 → G0 → SP-010 → G1A
G1A → SP-011
G1A → SP-012 → SP-013
G1A → SP-014
G1A → SP-020
SP-011 + SP-012 + SP-013 + SP-014 → G1
SP-012 + SP-020 → SP-021
SP-013 + SP-021 → SP-022
SP-011 + SP-021 → SP-023
G1 + SP-020 + SP-021 + SP-022 + SP-023 → G2
G2 → SP-030 → SP-031 → SP-032 → G3
G3 → SP-040
SP-040 → SP-041 / SP-042 / SP-043 / SP-044 / SP-051
SP-041 + SP-042 + SP-043 + SP-044 + SP-051 → SP-045 → G4
G3 → SP-050
SP-014 + SP-030 + SP-050 → SP-052
G4 + SP-050 + SP-052 → G5
G4 + G5 → SP-060 → G6
G4 + G5 + G6 → SP-070 → SP-080
```

After `G1A`, `SP-011`, `SP-012`, `SP-014`, and `SP-020` may run in parallel because their owned paths do not overlap.

After `G3`, SP-050 may run in parallel with frontend foundation work. After SP-040 creates the frontend shell, SP-041, SP-042, SP-043, SP-044, and SP-051 may run in parallel. The SP-040 owner returns as SP-045 to integrate route imports and global layout.

## 13. Standard agent handoff

Every completed task must report:

```text
Task ID:
Status:
Changed files:
Commands run:
Tests and results:
Artifacts/evidence:
Assumptions:
Known limitations:
Unresolved blockers:
Contract changes requested:
Recommended next task:
```
