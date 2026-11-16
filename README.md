# Natural Language Processing (CO3085)

SupportLens course project for class CC01, CLC_HK261, Semester 261, at Ho Chi Minh City University of Technology (HCMUT), VNU-HCM.

## Course and team

**Course:** Natural Language Processing (CO3085)  
**Instructor:** Trương Vĩnh Lân  
**Class:** CC01  
**Semester:** HK261

| Member | Student ID | Role | Responsibilities |
|---|---|---|---|
| [Trần Gia Lâm](https://github.com/lamgiatran) | 2352670 | Leader | Docs-QA data preparation, retrieval and generation pipeline, retriever fine-tuning when the core system is stable, evaluation, technical review and final integration. |
| [Nguyễn Hữu Cầu](https://github.com/cau26) | 2352129 | Member | Handoff-50 data audit, shared encoding and prediction pipeline, classifier training, calibration, confidence methods and out-of-scope analysis. |
| [Phan Trần Quốc Tuấn](https://github.com/tuanmin-27) | 2353273 | Member | API and UI integration, model loading, conversation state, streaming, error handling and end-to-end smoke testing. |
| [Nguyễn Tấn Hưng](https://github.com/Kun-05) | 2352439 | Member | BANKING77 preprocessing, BoW/TF-IDF classification baselines, controlled comparisons, error analysis and qrels review. |

## Project — SupportLens

SupportLens combines document question answering with evidence and confidence-based banking request triage. Users select a mode; both modes share the application interface, API and request logging.

| Component | Task | Methods | Evaluation |
|---|---|---|---|
| Docs-QA | Retrieve supporting passages and answer with citations or hand off when evidence is insufficient | BM25, frozen dense retrieval, hybrid RRF; local generation | Passage hit@3/hit@5, MRR, retrieval risk–coverage and citation review |
| Banking | Predict one of 50 supported intents and decide whether to accept or hand off | Frozen encoder with Logistic Regression; H1–H4 confidence methods | Classification results, risk–coverage, OOS false acceptance and error analysis |
| Course-77 study | Compare text representations and classifiers for the course requirements | BoW/TF-IDF with Logistic Regression and Linear SVM | Accuracy, macro-F1, confusion matrices and preprocessing ablation |

The document corpus and retrieval evaluation use MultiDoc2Dial. The Banking application uses Handoff-50; the 77-label study is a separate course experiment. A 77-label classifier is not used to claim that the remaining 27 intents were unseen.

## Work packages

Each member works in blocks of roughly two to three one-hour sessions. A first working module is handed over before the final model or threshold is selected, so integration can proceed in parallel.

| Package | Main outcome | Handoff |
|---|---|---|
| A — Prepare | Reviewed data, split rules and agreed input/output examples | Hưng sends qrels review to Lâm; Cầu completes the Banking data audit; Tuấn starts with the Docs API and agreed examples. |
| B — Integrate a first version | Baselines and a working Banking pipeline | Cầu sends a text-to-prediction pipeline to Tuấn; Tuấn checks standalone/API agreement. |
| C — Complete behavior | Dev-selected configuration, confidence thresholds and answer streaming | Cầu sends the selected Banking configuration; Lâm supplies generation/cache/deadline behavior; Tuấn integrates both modes. |
| D — Evaluate and deliver | Frozen evaluations, smoke tests, report sections and reproducible instructions | Members submit measured outputs and explanations; Lâm reviews and integrates the final submission. |

See [team tasks and handoffs](docs/TEAM_HANDOFF.md) and the [task board](issues/README.md) for inputs, expected outputs and acceptance conditions.

## Retrieval progress

Dev evaluation is available for BM25, frozen dense retrieval and hybrid RRF. On 2,213 dev queries, strict hit@3 is 0.4668, 0.4062 and 0.4803, respectively. The confidence interval for the hybrid–BM25 difference includes zero.

These are passage retrieval results on dev. They do not measure final answer quality. The configuration, context-packing audit and limitations are recorded in [the retrieval verification report](docs/VERIFICATION_v0.4.md).

Banking training and calibration, application integration and end-to-end generation are being developed through the assigned work packages. Retriever fine-tuning is the main extension after the core workflow is stable.

## Reproduction

Use Python 3.11 or later. Follow [the Docs-QA run guide](docs/RUN_DENSE.md) to install dependencies, prepare runtime assets and run the API/UI.

Data split IDs, experiment configurations, metrics and manifests are kept with the source. Raw datasets, pretrained weights and large vector caches are downloaded or rebuilt through the documented scripts.

The [experiment plan](docs/plan-v1.0.md) defines the data splits, model comparisons and evaluation procedure. Configuration and threshold selection use dev; held-out evaluation follows the final freeze.

## AI Usage Disclosure

 [AI_USAGE.md](docs/AI_USAGE.md).

## Submission

The final deliverables will include source code, a report, experiment outputs, run instructions and a demonstration. Complete the required LMS submission separately.
