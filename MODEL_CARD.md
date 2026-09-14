---
license: mit
model_card_spec: "1.1"
pipeline_tag: image-classification
task: "Image Classification"
base_model: timm/eva02_base_patch14_448.mim_in22k_ft_in22k_in1k
date_published: "2023-03-31"
date_published_source: "Hugging Face Hub repository creation date of the exact hosted checkpoint (`createdAt`, https://huggingface.co/api/models/timm/eva02_base_patch14_448.mim_in22k_ft_in22k_in1k)"
---

# EVA-02 Base patch14 448 mim_in22k_ft_in22k_in1k — Image Classification

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-timm%2Feva02__base__patch14__448.mim__in22k__ft__in22k__in1k-ffcc4d?style=flat)](https://huggingface.co/timm/eva02_base_patch14_448.mim_in22k_ft_in22k_in1k)
[![Upstream GitHub](https://img.shields.io/badge/Upstream%20GitHub-baaivision%2FEVA-181717?style=flat&logo=github&logoColor=white)](https://github.com/baaivision/EVA)
[![arXiv Paper](https://img.shields.io/badge/arXiv-2303.11331-b31b1b.svg)](https://arxiv.org/abs/2303.11331)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> [!WARNING]
> ⚠️ **Provided for research, training, and evaluation purposes only.** Model weights are redistributed unmodified under their upstream license, which controls your use, including any commercial use or redistribution; the accompanying code and notebooks are released under this repository's license. All of it is supplied **"as is"**, without warranty of any kind, and has not been validated for production, clinical, or safety-critical use. Running the notebooks downloads third-party weights and datasets governed by their own licenses and consumes compute on your own Colab/Kaggle account. To the maximum extent permitted by law, the maintainers of this repository and the DIMER platform accept no liability for any damages arising from their use. Hosting implies no affiliation with or endorsement by the original authors.

---

## Interactive Colab Tutorials

This pipeline provides a ready-to-run interactive Google Colab notebook that exercises the repository's public API end to end — bootstrap a fresh runtime, stage and verify the pinned upstream revision, validate an input, run the task, and inspect and export the outputs:

- **Task Inference Tutorial**:  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/eva02-classification-pipeline/blob/main/tutorials/eva02_classification_colab.ipynb) [`eva02_classification_colab.ipynb`](https://github.com/kurtvalcorza/eva02-classification-pipeline/blob/main/tutorials/eva02_classification_colab.ipynb)  
  *ImageNet-1k classification with the pinned EVA-02 Base 448 weights on a synthetic in-code sample: argmax decision plus rank-ordered top-5 softmax scores; CPU works but is slow at 448 px (107 GMACs); `top_k_accuracy` only when a ground-truth index is supplied.*

---

#### Description

`timm/eva02_base_patch14_448.mim_in22k_ft_in22k_in1k` is the EVA-02 Base vision transformer (Fang et al., arXiv:2303.11331): pre-trained on ImageNet-22k with masked image modelling using EVA-CLIP as the teacher, fine-tuned on ImageNet-22k and then on ImageNet-1k by the paper authors, converted to float32 and published in `timm` (upstream README), pinned here to revision `81063ecfe9c381a16a19d06f396d6c7011aa426a`. The network embeds a fixed 448×448 image as 32×32 = 1024 patches of 14 px plus a class token (1025 tokens of width 768, per the upstream README `forward_features` example), runs them through transformer blocks with SwiGLU MLPs, rotary position embeddings and an extra LayerNorm in the MLP (upstream README), mean-pools the tokens (`global_pool: "avg"` in the snapshot `config.json`) and applies a 1000-way `head`. Upstream reports 87.1 M parameters and 107.1 GMACs — roughly 24× the compute of the sibling ConvNeXt-Tiny at 224 px — which is the price of the 448-px input. Inference maps a normalised 3×448×448 tensor to 1000 logits in one forward pass; nothing is adapted or fine-tuned here. What this repository adds is packaging: the `EVA02ClassificationPipeline` class in `src/eva02_classification_pipeline/pipeline.py`, digest verification of the local snapshot (`verify_snapshot`), input validation, a fixed output contract and a `top_k_accuracy` helper.

#### Intended Use and Limitations

###### Primary Intended Uses

The task is single-label image classification: input one PIL image or a batch of up to `MAX_BATCH = 64` images; output, per image, the `top_k` (default 5) ImageNet-1k classes with their softmax scores plus the argmax label. Envisioned applications are the ones where accuracy is the constraint and a GPU is available — curated archive tagging, offline re-labelling of large collections, second-opinion classification behind a cheaper model, and an accuracy reference in the DIMER workbench (upstream reports 88.7 % top-1, the highest of the four timm classifiers in this set). In a larger system the pipeline is an inference component, not a decision engine; the 768-d pooled features are not exposed by this package (the DINOv2 sibling covers feature extraction).

###### Primary Intended Users

The intended users are machine-learning engineers, data scientists and application developers integrating a high-accuracy classifier into research prototypes, GPU-backed services, or the DIMER model workbench. The pipeline assumes its users understand that the label space is fixed to the 1000 ImageNet-1k classes, that a softmax score is not a calibrated probability, that a 107-GMAC model at 448 px is not an edge or CPU-latency candidate, and that any deployment on their own data needs a labelled evaluation set. It is not designed for hobbyist "point and trust" use.

###### Out-of-scope use cases

1. **Capability boundary:** not object detection, segmentation, multi-label tagging, OCR, or open-vocabulary classification; anything outside the 1000 ImageNet-1k classes cannot be named. The ImageNet-22k pre-training and intermediate fine-tuning vocabularies are not exposed — the final head answers only the 1k classes. Feature extraction is not exposed — use `dinov2-feature-extraction-pipeline`.
2. **Input boundary:** only PIL images are accepted (`TypeError` otherwise); any side above `MAX_IMAGE_SIDE = 4096` px or below 1 px is rejected; batches above 64 are rejected; every image is **squash-resized** to exactly 448×448 (`crop_mode: "squash"`, `crop_pct: 1.0`, `fixed_input_size: true` in the snapshot `config.json`), so aspect ratio is not preserved and strongly non-square images are distorted rather than cropped. Normalisation uses the CLIP mean/std from the snapshot, not the ImageNet ones. Non-RGB modes are converted to RGB; depth, multispectral and video inputs are unsupported.
3. **Decision boundary:** not for autonomous or high-impact decisions — content moderation takedowns, safety interlocks, medical or forensic triage — without a human reviewing the prediction and a locally measured error rate. Not for latency-bound or CPU-only paths; use the MobileNetV4 sibling there.

#### Factors

###### Groups

The pipeline is not human-centric: it is an object-centric classifier whose label space contains no person-identity, age, gender or skin-type categories. ImageNet-1k and the ImageNet-22k pre-training set nevertheless contain many images of people, ImageNet-22k in particular carries person-category synsets that were later flagged as offensive or non-imageable, and neither set is group-audited. Neither the upstream `timm` card nor this repository reports any group-level performance breakdown. The fairness audit therefore transfers to the operator: before deployment, measure `top_k_accuracy` on a labelled sample of your own data stratified by the groups that matter to your application, and treat any material gap as a blocker.

###### Instrumentation

ImageNet images were collected from web image searches (Deng et al., 2009) and are consumer camera photographs of varied, undocumented provenance — many makes of camera, lens and post-processing, mostly JPEG-encoded. The pipeline consumes decoded pixel arrays, so the instrument sits behind PIL: resolution, JPEG compression level, colour profile, white balance and sensor noise all reach the model as changed pixel statistics after the squash to 448 px; because the resize is a squash, the sensor's aspect ratio is itself an instrument factor — a 3:2 camera and a 16:9 camera present the same scene differently stretched. The pipeline does not detect drift, blur, over-exposure or a change of capture device; it only rejects non-image types and images outside the 1–4096 px side range.

###### Environment

Operating environment: Python 3.12 with `torch==2.14.0`, `torchvision==0.29.0`, `torchaudio==2.11.0`, `timm==1.0.29`, `pillow==11.3.0` (exact pins in `pyproject.toml`). CUDA is the intended device; `from_pretrained` picks `cuda:0` when available, else CPU, and runs in float32 on both (the checkpoint is 348 MB float32). On this repository's smoke run (claude-science WSL venv, RTX 5070 Ti 16 GB, one synthetic 256×256 image through `EVA02ClassificationPipeline.from_pretrained().predict`) loading the verified snapshot took 6.42 s and one 448-px prediction 0.81 s including transform and first-call CUDA warm-up — against 4.67 s / 1.47 s for the ConvNeXt-Tiny sibling on the same host, so the load is heavier and the warmed forward pass is not the bottleneck at batch 1. The CPU path was not measured and is expected to be tens of times slower than the 224-px siblings given 107 GMACs per image. Data environment: inputs are assumed to be natural photographs whose subject is one of the 1000 classes; line drawings, medical scans, satellite tiles, heavy occlusion or unusual viewpoints fall outside that assumption and degrade accuracy in ways the pipeline does not measure.

#### Metrics

###### Performance Measures

The only measure the code reports is `top_k_accuracy(predictions, targets, k)` in `pipeline.py`: the fraction of images whose target index appears among the first `k` predicted indices, for any `k` up to the requested `top_k`. It captures discrete correctness of the ranking, which suits a 1000-way single-label classifier where the operational question is "is the right class first, or at least in the shortlist". It says nothing about calibration or per-class behaviour, so a reader using top-1 alone cannot tell whether errors are near-misses (fixable by a shortlist) or confident mistakes. Upstream reports 88.692 % top-1 / 98.722 % top-5 on the ImageNet-1k validation set at 448 px for this checkpoint (upstream README comparison table; the `mim_in22k_ft_in1k` row without the intermediate 22k fine-tune is a different checkpoint at 88.23 %); this pipeline has not reproduced those numbers and reports no accuracy of its own. The public `evaluation_report(result, targets)` helper is the only reporting path: it emits a machine-readable report whose verdict is `sample-sanity` with `top_k_accuracy` at k=1 and k=5 when ground-truth indices are supplied, and `not-measurable` otherwise, stating in that case what labelled data would make the task measurable.

###### Decision thresholds

The default decision rule is `argmax` over the 1000 softmax scores, exposed as `DECISION_RULE = "argmax"` and reported as `predicted_index` / `predicted_label`; this is an implicit threshold of "highest score wins" with no minimum score. No acceptance threshold was set during development and none is shipped: the softmax score is uncalibrated, so any fixed cut-off would be arbitrary. A deployment that needs an abstain option must choose a score cut-off on its own labelled data, trading the cost of a wrong confident label (false positive) against the cost of an unanswered image (false negative) for its application.

###### Approaches to uncertainty and variability

This pipeline reports no accuracy number, so there is no estimation procedure or dispersion to state; the upstream figures cited above are single validation-set evaluations by the upstream author with no reported interval. Inference is deterministic given the same weights, device and library versions: there is no sampling, dropout and drop-path are disabled by `model.eval()`, and no seed is required; small numeric differences between CPU, GPU and attention-kernel choices can reorder near-tied classes, and the original float16/bfloat16 checkpoints (upstream README note) would differ from this float32 conversion in the low bits. The `score` field is a softmax over logits and is not calibrated; a caller who needs probabilities must fit a calibration map (for example temperature scaling) on their own labelled data.

#### Ethical considerations and biases

###### Data

Upstream states the checkpoint was pre-trained on ImageNet-22k with masked image modelling using an EVA-CLIP teacher, fine-tuned on ImageNet-22k and then ImageNet-1k (upstream README `datasets` and "Model Details"); the disclosure stops there — no per-image licensing, consent status or demographic composition is given for either set, the EVA-CLIP teacher itself was trained on web image-text pairs whose provenance this card cannot trace, and ImageNet is known to contain photographs of identifiable people scraped from the web, so the presence of personal data is not ruled out. This repository distributes code, tests and documentation; the 348 MB `model.safetensors` snapshot is git-ignored and staged locally under `weights/eva02-base-448/` with a manifest, and no sample data is shipped. The operator must audit the images they submit for personal, confidential or proprietary content; the pipeline performs no such check.

###### Human Life

The pipeline is not intended for decisions in health, safety, criminal justice, employment, credit, housing or any other domain central to human life, and it has not been validated or certified for any of them by anyone. Its only validation is the offline unit suite and the smoke run in this repository. Higher headline accuracy invites exactly the sensitive uses it is not validated for; where such a use is foreseeable — for example flagging images in a moderation queue — it is admissible only with a human reviewer on every consequential outcome, an independent domain evaluation on representative data, and whatever regulatory clearance the domain requires.

###### Mitigations

Implemented and inspectable in `src/eva02_classification_pipeline/pipeline.py`: (1) supply chain — `MODEL_REVISION` is a 40-hex commit; `verify_snapshot` re-hashes every file in `weights/eva02-base-448/dimer-base-manifest.json` (348 MB, so the check takes a few seconds and is part of the measured load time) and raises on the first size or SHA-256 mismatch before any weight is loaded; the Hub path is taken only with `allow_download=True` and then through timm's `hf-hub:<id>@<revision>` form; `trust_remote_code` is never enabled (timm executes no remote code). (2) Input integrity — `_validate` rejects non-PIL inputs, empty or over-size batches, and images outside 1–4096 px before the model runs, and the public `validate_inputs(images, top_k, names=...)` stage routes through the same private check so it raises exactly what `predict` raises while returning a machine-readable input manifest of the schema, ceilings, per-input observations and verdict. (3) Reproducibility — exact `==` dependency pins, `model.eval()`, deterministic preprocessing from the snapshot's `pretrained_cfg`, and `model_id`/`model_revision` in every result. (4) Refusals — no feature-map or training API is exposed; a missing snapshot with `allow_download=False` raises `FileNotFoundError`. No statistical mitigation (class re-balancing) is applied because the pipeline does not train.

###### Risks and harms

Overconfidence out of distribution: an unrelated image still yields a top-1 label — the smoke run labelled a synthetic colour gradient `screen, CRT screen` at score 0.013 — and the operator bears the harm when that label is acted on; a model with 88.7 % headline accuracy is more likely to be trusted without checking. Bias in the label space and training images: ImageNet's classes and their examples skew toward Western, web-scraped imagery, so objects common elsewhere are more often mislabelled; data subjects and third parties bear the harm when such labels feed downstream systems. Squash distortion: non-square subjects are stretched before classification, a failure mode the 224-px centre-crop siblings do not share. Cost: 107 GMACs per image makes an unbounded batch a denial-of-service vector on shared GPUs; `MAX_BATCH = 64` bounds one call but not the call rate. Automation bias applies as for the other classifiers. Likelihood under normal photographic use is moderate and rises sharply off-distribution; magnitude ranges from a wrong tag to a wrongly moderated image.

###### Use cases

The pipeline must not be used for surveillance, biometric or demographic profiling, or social scoring — its label space cannot do these, and adapting it to try would be a misuse. It must not support unlawful discrimination in employment, housing, credit, insurance, education or healthcare access, nor deceptive or manipulative applications such as fabricating evidence of what an image contains. Any use that violates the MIT terms of the upstream weights (which require preserving the copyright and permission notice) or the DIMER deployment terms is prohibited. The developers identify no further prohibited use beyond these because the model's output is a coarse object label.

## Immutable provenance

- Model: `timm/eva02_base_patch14_448.mim_in22k_ft_in22k_in1k`
- Revision: `81063ecfe9c381a16a19d06f396d6c7011aa426a`
- Snapshot manifest: `weights/eva02-base-448/dimer-base-manifest.json`, `totalBytes` 348498591
- `model.safetensors` SHA-256: `533937d6f9f8f8d4f50627ef0d00829a5015861a5b67e1c69c5a5e45b7dc2609` (348492484 bytes)
- `config.json` SHA-256: `d62e73f578eb363032000b6bfcc57764c6a5d10ee2562cbfe3f105d795e1c4be` (653 bytes)
- Weight format: SafeTensors, float32; loader `timm.create_model("eva02_base_patch14_448.mim_in22k_ft_in22k_in1k", pretrained=True, pretrained_cfg_overlay={"file": ...})`

## Input/output contract

- `EVA02ClassificationPipeline.from_pretrained(device=None, weights_dir=None, allow_download=False)`
- `predict(images, top_k=5)` — `images`: one `PIL.Image.Image` or a sequence of 1–64; sides 1–4096 px; any mode (converted to RGB); every image is squash-resized to 448×448. Returns `{"predictions": [{"predicted_index", "predicted_label", "top_k": [{"label", "index", "score"}, ...]}, ...], "top_k", "decision_rule", "device", "source", "model_id", "model_revision"}`; `score` is the softmax over 1000 classes.
- `top_k_accuracy(predictions, targets, k=1)` — accepts the `predictions` list above or plain index lists.
- `verify_snapshot(path=None)` — returns the manifest dict with `path`; raises `FileNotFoundError` / `ValueError`.

## Runtime

- Pins: `torch==2.14.0`, `torchvision==0.29.0`, `torchaudio==2.11.0`, `timm==1.0.29`, `huggingface-hub==0.36.2`, `safetensors==0.8.0`, `numpy==2.5.3`, `pillow==11.3.0`; Python 3.12.
- Precision: float32 on both CPU and CUDA; preprocessing squash-resize to 448×448, bicubic, CLIP mean `[0.4815, 0.4578, 0.4082]` / std `[0.2686, 0.2613, 0.2758]` from the snapshot `config.json`.
- Measured (claude-science WSL venv, RTX 5070 Ti 16 GB, `HF_HUB_OFFLINE=1`): device `cuda:0`, source `local-snapshot`, load 6.42 s, predict 0.81 s, total 7.23 s, top-1 on a synthetic 256×256 gradient image `screen, CRT screen` (index 782) at score 0.0127. CPU not measured.
- Tests: `pytest -q -o addopts= tests` — 11 passed, offline, no weights required; `ruff check src tests` clean.

## References

- Fang, Sun, Wang, Huang, Wang, Cao. EVA-02: A Visual Representation for Neon Genesis. 2023. https://arxiv.org/abs/2303.11331
- Sun et al. EVA-CLIP: Improved Training Techniques for CLIP at Scale. 2023. https://arxiv.org/abs/2303.15389
- Original weights and code: https://github.com/baaivision/EVA and https://huggingface.co/Yuxin-CV/EVA-02
- Wightman. PyTorch Image Models. https://github.com/huggingface/pytorch-image-models (doi:10.5281/zenodo.4414861)
- Deng et al. ImageNet: A large-scale hierarchical image database. CVPR 2009. https://doi.org/10.1109/CVPR.2009.5206848
- Upstream card: https://huggingface.co/timm/eva02_base_patch14_448.mim_in22k_ft_in22k_in1k
