# SupportLens — day-one starter

This repo contains working API contracts, raw-data readers, document-level BM25,
token budgets, a shared retrieval-risk gate and an Ollama probe. Classifier
artifacts, dense indexes and an end-to-end generator are **not trained/built yet**.
Banking requests return an explicit 503 until the separate Handoff-50 artifact is
available. Docs-QA currently returns evidence, labelled `retrieved`, without an
invented generated answer.

The original plan is frozen at [docs/plan-v1.0.md](docs/plan-v1.0.md). Execution
changes and the five technical guardrails are in [issues/README.md](issues/README.md).
Issue owners are planned assignments, not claims about work already done by people.

| Folder | Purpose |
|---|---|
| `src/supportlens/` | Contracts, raw readers, token budgets, retrieval, risk gate, API and generator probe support |
| `scripts/` | Download, inspect, audit, probe and smoke entry points |
| `configs/` | Source revisions, tokenizers, budgets and R1–R5 definitions |
| `issues/` | Owner/day tickets and technical backlog |
| `tests/` | Offline regressions and optional real-tokenizer integration check |
| `docs/` | Frozen plan, verification evidence and current status |

## Start on Windows

From this folder, with Python 3.11+ and network access for initial dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
```

The unit tests are offline. One real-tokenizer check skips until the small
tokenizer asset is downloaded; it does not download model weights automatically.

```powershell
.\.venv\Scripts\python.exe scripts/download_data.py --dataset multidoc2dial
.\.venv\Scripts\python.exe scripts/inspect_data.py
.\.venv\Scripts\python.exe scripts/fetch_tokenizers.py
.\.venv\Scripts\python.exe scripts/audit_token_budgets.py
```

The dataset downloader pins source revisions/checksums and writes SHA-256 manifests.
Use `--verify-only` to check existing data. `--preloaded` explicitly inventories
bytes acquired elsewhere; it never claims that a network download occurred.
Raw data/model assets are excluded from the starter archive; download them locally.

## Run the day-two retrieval slice

After `inspect_data.py` creates `data/processed/documents.jsonl`:

```powershell
$env:SUPPORTLENS_CORPUS = 'data/processed/documents.jsonl'
$env:SUPPORTLENS_LOG = 'runs/requests.jsonl'
.\.venv\Scripts\python.exe -m uvicorn supportlens.api:app --app-dir src --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs`. POST `/v1/assist`:

```json
{"mode":"docs_qa","question":"How do I change my address?","retriever":"R1"}
```

`GET /health` distinguishes process health from component readiness. `503` means
the requested artifact is absent; `422` means the request violates the contract.
No request can pass a gold document ID or silently switch Course-77/Handoff-50.

Optional UI, in another terminal after installing `.[ui]`:

```powershell
.\.venv\Scripts\python.exe -m streamlit run ui/app.py
```

The UI displays retrieved documents. Streaming generation through UI/API is an
assigned issue, not a feature claimed by this starter.

## Data tasks for Hưng and Cầu

```powershell
.\.venv\Scripts\python.exe scripts/download_data.py --dataset banking77
.\.venv\Scripts\python.exe scripts/download_data.py --dataset banking77_oos
.\.venv\Scripts\python.exe scripts/inspect_banking.py
```

OOS generic labels and original intent labels have different purposes. Keep
`label_original` for the 27-by-50 analysis. OOS-train files are available for audit
but must not enter the main classifier/reference bank. Audit flags are reported;
the starter does not silently modify official splits.

## Generator probe

Install/run Ollama separately and make `qwen2.5:1.5b` available. The probe does not
install Ollama or pull large weights. Download just its tokenizer:

```powershell
.\.venv\Scripts\python.exe scripts/fetch_tokenizers.py --include-generator
.\.venv\Scripts\python.exe scripts/probe_ollama.py
```

The probe sets `num_ctx=2048`, reserves 128 output tokens, packs at most three
evidence items within 600 **Qwen** tokens, and streams an explicit raw ChatML
prompt. It measures generator HTTP TTFT, not UI/end-to-end TTFT. It records
`prompt_eval_count` versus the supplied raw prompt token count; BOS/tokenizer
differences make this diagnostic, not proof of zero truncation. Overflow is
rejected before inference. Incomplete streams are labelled incomplete.

## Evaluation guardrails

- Encoder query = current question, then past turns newest-first; log truncation.
- Shared chunk text, including title, fits 220 encoder tokens within its 256 limit.
- Docs risk = annotated gold absent from top-5. This proxy does not establish
  answer factuality or general OOD detection. Tune a method/artifact-specific gate
  on dev using `scripts/fit_docs_gate.py`; never copy a threshold across score scales.
  Load a fitted gate with `SUPPORTLENS_DOCS_GATE=artifacts/docs_gate.json`.
- Tune LR `C` on dev before calibration and H1–H4 comparisons. No default `C=1`
  performance claim is embedded in this starter.
- Compare R4 vs R2 for dense training impact and R5 vs R3 if serving hybrid. Keep
  BM25/chunks/query policy/RRF configuration fixed in the latter comparison.
- A training smoke run determines throughput/ETA. Overnight compute is optional
  and recorded separately from active preparation, debugging and analysis time.

## Work and versioning

`issues/` holds 12 owner/day tickets plus a small later-work backlog. They are
ready to copy into GitHub; no remote repository or hosted GitHub issues are
created by this local starter. Plan-scope changes happen at gates 2, 5 and 8;
implementation bugs and measured findings are tracked directly in issues.

The local tag `plan-v1.0` freezes this plan snapshot; package version `0.1.0`
identifies the scaffold. See [docs/STATUS.md](docs/STATUS.md) for measured checks
and remaining gaps. The scripts/configs are the reproducibility interface;
sample inspection results are not benchmark results.
