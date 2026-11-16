# Third-party data and models

## MultiDoc2Dial

Docs-QA uses the **MultiDoc2Dial** dataset, documents and grounding annotations
published by the dataset authors. Source:

- [Dataset project](https://doc2dial.github.io/multidoc2dial/)
- [Paper](https://arxiv.org/abs/2109.12595)
- [IBM dataset card](https://huggingface.co/datasets/IBM/multidoc2dial)

The IBM dataset card declares Apache-2.0. A copy of that license is in
[docs/licenses/MultiDoc2Dial-Apache-2.0.txt](docs/licenses/MultiDoc2Dial-Apache-2.0.txt).
The exact archive URL, release and SHA-256 used by this project are recorded
in [configs/sources.json](configs/sources.json).

`data/processed/docs_qa/qrels_audit.jsonl` contains short source excerpts used
to inspect span-to-chunk mapping. These excerpts and grounding annotations are
attributed to MultiDoc2Dial and the originating document sources. Document IDs,
domain and offsets are retained for traceability. Our chunking, strict/lenient
rules and dialogue split are local experimental choices, not official dataset
baselines. The original source documents retain their applicable notices; this
file does not assign a new blanket license to every source website.

## BANKING77 and BANKING77-OOS

- BANKING77: [PolyAI task-specific datasets](https://github.com/PolyAI-LDN/task-specific-datasets),
  released under CC BY 4.0. See [the upstream license copy](docs/licenses/BANKING77-LICENSE.txt).
- BANKING77-OOS splits: [Few-Shot-Intent-Detection](https://github.com/jianguoz/Few-Shot-Intent-Detection).

Pinned revisions and individual source files are recorded in
`configs/sources.json`. Derived split inventories retain the original sample
text/labels with local IDs and audit flags. The Course-77 and Handoff-50
experiments are this group's configurations. Credit for the original data and
released OOS split belongs to their publishers.

## Pretrained encoder and generator

The frozen encoder is
[sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
The configured local generator is Qwen2.5-1.5B through Ollama. Weights are not
redistributed in Git. Refer to the upstream model cards and retain their
applicable licenses/notices when downloading or redistributing model assets.

These notices identify third-party material; they do not declare the dataset,
model weights and project source to be a single uniformly licensed work.
