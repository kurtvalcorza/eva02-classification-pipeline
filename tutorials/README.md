# Tutorials

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/kurtvalcorza/eva02-classification-pipeline)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/eva02-classification-pipeline/blob/main/tutorials/eva02_classification_colab.ipynb)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-timm%2Feva02__base__patch14__448-ffcc4d?style=flat)](https://huggingface.co/timm/eva02_base_patch14_448.mim_in22k_ft_in22k_in1k)
[![Upstream](https://img.shields.io/badge/Upstream-baaivision%2FEVA-181717?style=flat&logo=github&logoColor=white)](https://github.com/baaivision/EVA)
[![arXiv](https://img.shields.io/badge/arXiv-2303.11331-b31b1b.svg)](https://arxiv.org/abs/2303.11331)

Notebook specification: **DIMER Notebook Specification 1.0**

| Notebook | Profile | Capability | Default runtime | BYOD | Release status |
|---|---|---|---|---|---|
| `eva02_classification_colab.ipynb` | `TASK-INFERENCE` | EVA-02 Base 448 ImageNet-1k classification (1000 classes, argmax decision, rank-ordered top-5 softmax scores) on a synthetic in-code sample; `top_k_accuracy` scored only when the user supplies a ground-truth class index | CPU works but is slow (107 GMACs at 448 px); CUDA used automatically when available | single image file, gated off by default | **Candidate** — static checks pass; the clean-runtime execution row in `../docs/release-verification.md` is pending and must be recorded for the exact notebook revision before promotion |

## Conformance notes

- The notebook exercises `EVA02ClassificationPipeline` from the repository public API rather than reimplementing model loading; the pipeline pins the immutable upstream revision, stages missing snapshot files through the package's `stage_missing_files(..., allow_download=True)`, loads only from a digest-verified local snapshot (`verify_snapshot`), and executes no remote code. The notebook never calls `timm` or `huggingface_hub` directly.
- The default sample is synthetic (a 320 x 240 RGB gradient generated in code, deliberately non-square so the 448 x 448 squash-resize is visible); it has no ground truth, so no metric is reported on the default path and the run is plumbing/sanity evidence only. `top_k_accuracy` at `k=1`/`k=5` is computed only when the user sets `GROUND_TRUTH_INDEX` for a BYOD image, and is a single-image tutorial figure.
- §20.1 obligations: the label space (`NUM_CLASSES` = 1000, labels printed from the pipeline), the `argmax` decision rule, the uncalibrated softmax `score`, the absence of a shipped threshold, and the descending-score class ordering preserved in the exported CSV are all stated; the ground-truth index is range-checked against the fixed label space before scoring.
- `USE_BYOD` defaults to `False` so the sample path never opens an upload dialog.
- Recorded `SHOULD` deviation: EVAL11/EVAL12 (no majority-class baseline — it has no meaning for a single unlabelled image and cannot be scored without ground truth).
- `tools/validate_release_assets.py` performs source validation only. It does not satisfy the
  clean-runtime execution requirement; a release review must confirm that a recorded clean run in
  `docs/release-verification.md` matches the notebook revision under review before the status is
  promoted to `Release-grade`.
