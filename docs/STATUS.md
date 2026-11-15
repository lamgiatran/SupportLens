# Day-one verification — 15/11/2026

The plan snapshot is frozen. This report records implementation checks, not new
scope commitments or model-quality claims.

| Check | Observed result |
|---|---|
| Unit/API regression suite | 22 passed in 1.61 seconds |
| Raw MultiDoc2Dial documents | 488 across ssa/va/dmv/studentaid |
| Public train | 3,474 dialogues; 22,806 user-to-next-agent pairs |
| Public validation | 661 dialogues; 4,427 pairs |
| Train/validation dialogue-ID overlap | 0 |
| Token-budget smoke sample | First 200 train queries; 44 truncated histories; 0 current questions truncated |
| Chunk smoke sample | First 10 documents; 59 chunks; maximum 220 encoder tokens |
| Real-corpus API smoke | HTTP 200 through in-process ASGI TestClient; evidence from the 488-document corpus |
| BANKING77 | 10,003 train / 3,080 test; 77 labels |
| Normalized BANKING77 train/test text overlap | 6, flagged for owner review |
| BANKING77-OOS | 50 ID labels; original 27 ID-OOS intents preserved |
| Ollama probe | Unavailable / connection timeout; no measured generator or UI TTFT |

Evidence JSON/text is in [verification/](verification/). The raw archive was
fetched from the official dataset URL and ingested with the documented
`--preloaded` path because the command runner could not resolve external hosts.
Manifests mark this acquisition method; they do not pretend the Python downloader
performed the HTTP transfer. Wheel hashes were checked against PyPI metadata.

API tests were run in a local Python process whose event loop was available;
the restricted command runner could not create its normal Windows loopback
wakeup socket. The suite uses real FastAPI/Pydantic and ASGI TestClient, not a
substituted API implementation. No public server was deployed.

## Implemented and available

- Discriminated Pydantic request contracts and scope validation for two modes.
- Source-pinned, bounded downloader with SHA-256 manifests and allowlisted ZIP extraction.
- Raw document/dialogue readers; target responses are not retrieval input.
- Current-first, reverse-history encoder query builder and shared token-budgeted chunks.
- Document-level BM25 API slice, explicit readiness/missing-artifact responses.
- Tie-aware risk/coverage and dev-only Docs-QA threshold selection with artifact binding.
- Raw-prompt streaming Ollama probe with explicit context/output budgets and token-count diagnostics.
- 12 owner/day issues, R4/R5 backlog, minimal optional UI and run instructions.

## Still to implement or measure

- Passage-level indexes, audited full qrels, stable dialogue dev/held-out split.
- E1–E4, Handoff-50 model artifacts, C selection, calibration and benchmark results.
- Actual dense retriever training, re-encoding, R4/R5 comparisons and uncertainty.
- Generator connection to serving, stream through UI/API, generation cache and semantic citation review.
- UI runtime verification, human handoff quality, live-server latency and p95/TTFT on demo hardware.

Document-level retrieval is labelled `retrieved`; it is not presented as a generated
answer or as a successful passage benchmark. Gold-not-in-top-5 is a retrieval-risk
proxy. The tokenizer sample is sequential and small; its 22% history-truncation
rate is not a dataset-wide estimate. No model-quality metric is invented.
