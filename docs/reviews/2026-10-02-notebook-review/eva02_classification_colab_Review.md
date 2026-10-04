# EVA-02 Base 448 Classification E2E Notebook — Review

**Verdict: Needs revision**  
**Review date:** 3 October 2026 (relay batch of 2 October 2026)  
**Repository:** `kurtvalcorza/eva02-classification-pipeline`  
**Notebook:** `tutorials/eva02_classification_colab.ipynb`  
**Reviewed commit:** `8052cf07add9d0c5c4e99fd381eb97613e6d66e3` (`main`, confirmed with `gh api repos/kurtvalcorza/eva02-classification-pipeline/commits/main`)  
**Notebook Git blob:** `3f9b46469ce82e1e74b8ef76e5ec618bd4c5156d`. This is the blob executed in the recorded Kaggle Tesla T4 run of 2026-09-14 (commit `78f7f7c`); the notebook last changed in `78f7f7c`.  
**Finding prefix:** `EVA`  
**Framework:** Notebook Review Framework v1. **Requirements baseline:** NOTEBOOK_SPEC 2.2 (2026-09-26), `ml-worker` `origin/main`. The notebook declares 2.0.

## Executive assessment

The inference half of the notebook is careful and reproduces exactly. It carries `pipeline.py` verbatim, asserts the inline manifest against the module identity, stages and re-hashes the pinned 348 MB `timm/eva02_base_patch14_448.mim_in22k_ft_in22k_in1k` snapshot, validates a deliberately non-square synthetic gradient into an input manifest with a recorded rejection, classifies it with correct prose on argmax and uncalibrated softmax scores, and writes a `not-measurable` evaluation report that says what labelled data would be needed. A direct CPU run reproduced the Kaggle top-5 to four decimals.

The fine-tuning half, which the opening cell promises as a core outcome, does not support the conclusion it reports:

| Measure | This review (CPU, pins, defaults) | Kaggle T4 record (blob `3f9b4646`) |
|---|---|---|
| Code cells completed | 10/10 (158 s incl. 348 MB download) | 10/10 on pass 2 (pass 1 stopped at the install guard) |
| Gradient top-1 | `hair spray` 0.0179, same top-5 | identical |
| Fine-tune data | 26 train / 6 val (`frog`, `truck`), 1 epoch, batch 2 | identical |
| Fine-tuned held-out accuracy | **3/6 = 0.50** (every image predicted `truck`) | 5/6 = 0.833 |
| Majority baseline | 0.50 ("predicting 'truck'", a tie) | 0.50 (a tie) |
| Report verdict | **`success`**, lift +0.00 | `success`, lift +0.33 |
| Parameters changed by "head fine-tuning" | all 86.35 M (234/234 backbone tensors differ from the base) | not recorded |

Five problems stand in the way of `Ready for intended use`:

1. **No one-pass `Run all` (EVA-M1).** The recorded run stopped at the install cell's stale-module guard and passed only after a restart; the release record reports "10/10 ok code cells executed cleanly".
2. **The fine-tune result is a coin flip reported as success (EVA-M2).** Six validation images, one epoch, a hard-coded `'verdict': 'success'`, and no caveat; on CPU the fine-tuned model scores exactly the majority baseline and the report still says `success`.
3. **"Head fine-tuning" trains the whole network (EVA-M3).** The prose says a new head is trained; `fit` passes every parameter to AdamW and the exported artifact is the full 345 MB model.
4. **Guided layer largely absent (EVA-M4).** Declared `GUIDED`, but there is no audience, how-to-use, roadmap, glossary, prediction prompt, checkpoint or conclusion template, and the 511-line carried module is not labelled as infrastructure.
5. Two unresolved `MUST`s outside those Majors: the reload is never compared with the in-memory model (VER5, EVA-m1), and a BYOD dataset is fine-tuned before any image-size validation (DAT19, EVA-m3).

## 1. Review contract and evidence

| Item | Value |
|---|---|
| Declared profile / mode | `E2E` / `GUIDED` (metadata `dimer.notebook_profile` / `notebook_mode`, opening cell) |
| Declared spec | DIMER Notebook Specification **2.0** (metadata, opening cell, `NOTEBOOK_SOURCE`) |
| Spec baseline applied | NOTEBOOK_SPEC **2.2** |
| Intended audience | Not stated. Prerequisites: "basic Python and PIL image handling; what a softmax over class logits is and why it is not a calibrated probability" |
| Supported runtime | "Google Colab or Jupyter, Python 3.12"; CPU default, CUDA automatic |
| Promised outcomes | Pinned install; carried module; digest-verified snapshot; synthetic non-square sample validated into an input manifest; top-5 classification; `not-measurable`/`sample-sanity` evaluation report; digest-verified `Cleanlab/cifar-10-subset` download; "bounded in-kernel fine-tuning run of a new classification head"; export; fresh-boundary reload and held-out evaluation against a majority baseline; exports and provenance; BYOD single image and BYOD dataset `.zip` through the same cells |
| Generator | `tools/build_notebook.py` (`build_notebook.py/2`) + `tools/notebook_template.py`; recorded generating revision `6d64624` |
| Release status | `Candidate` (`STATUS.md`, `README.md`, `tutorials/README.md`, `docs/release-verification.md`) |

### Evidence actually obtained

- **Source inspection.** All 23 cells (10 code; cell 5 is the carried `pipeline.py`, 511 lines). Also read: the generator, template and `tools/validate_release_assets.py`; `src/eva02_classification_pipeline/pipeline.py` (`from_pretrained`, `fit`, `predict`, `validate_inputs`, `evaluation_report`); `README.md`, `STATUS.md`, `tutorials/README.md`, `docs/release-verification.md`. The repository has no `AGENTS.md` and no `docs/execution-evidence/`.
- **Documented execution evidence.** `docs/release-verification.md` row 2026-09-14 and its run summary (`.agent/backups/kaggle-pass-2026-09-14/.../run_summary.json`): Kaggle Tesla T4, **the reviewed blob**, clean model cache, image torch 2.10.0+cu128 / numpy 2.0.2; pass 1 `RuntimeError` in the install cell (`cuda-bindings 12.9.4 → 13.4.1; numpy 2.0.2 → 2.5.3`), `restarted_after_install_cell: true`, pass 2 10/10. No Colab run, no BYOD run.
- **Direct execution (this review).**
  - **Environment:** `run_probes.py`, Windows 11, 24 logical CPUs, CPU only (`CUDA_VISIBLE_DEVICES=-1`), Python 3.12, torch 2.14.0+cpu, torchvision 0.29.0+cpu, torchaudio 2.11.0+cpu, timm 1.0.29, safetensors 0.8.0, huggingface-hub 0.36.2, numpy 2.5.3, pillow 11.3.0 — the notebook's pins (CPU wheels), taken read-only from another pipeline repository's `.venv`. Nothing was installed.
  - **Install skipped:** cell 3 ran with `DIMER_NOTEBOOK_CI_PREINSTALLED=1`, the notebook's executor hook. Restart behaviour rests on the documented run.
  - **Clean assets (P1):** empty `HF_HOME` and working directory; cell 7 fetched all three manifest entries at the pinned revision and `verify_snapshot` passed; cell 17 downloaded and digest-checked the archive. P4–P6 reused P1's verified snapshot files to avoid re-downloading.
  - **Probes:** P0 static; P1 every code cell at defaults, then P3 in the same process (in-memory vs reloaded predictions; artifact tensors compared with the base); P2 the seeded subset's pixel-digest overlap; P4 BYOD dataset branch through a shimmed `google.colab.files.upload` (one compatible zip, seven incompatible); P5 BYOD single-image branch (text file, oversized image, cancelled upload); P6 cell 17 with a wrong expected archive digest.
- **Not verified:** a Colab run; the real Colab upload dialog; GPU numbers beyond the Kaggle record; a second seed or device for the fine-tune.
- **Learner observation:** none. No claim here is about measured learning effectiveness.

## 2. Separate judgments

- **Technical correctness:** inference path solid (P1 10/10, top-5 identical to Kaggle; snapshot digest-checked). Defects: the install pattern forces a restart (EVA-M1); a sample-archive digest mismatch is caught by a blanket `except Exception` and silently replaced by synthetic stripes (EVA-m2); BYOD dataset images bypass the ceilings until after training (EVA-m3).
- **Scientific validity:** the fine-tune evaluation is six images after 13 optimiser steps, with a tie-broken majority baseline and a fixed `success` verdict; its outcome flips between devices (0.50 on CPU, 0.833 on T4) and nothing tells the learner that (EVA-M2). The default subset happens to have no pixel copies across the split (P2), but the archive holds 100 identical original/darkened pairs and nothing checks for them (EVA-m4).
- **Promise fulfilment:** inference promises met. "Fine-tuning of a new classification head" is not what runs (EVA-M3); the reload is evaluated but not shown to reproduce the trained model (EVA-m1); the BYOD dataset path lacks validation and new-data inference (EVA-m3).
- **Learner experience:** accurate inference prose and a useful troubleshooting paragraph, but no guided layer (EVA-M4), and the closing interpretation never mentions the fine-tune result.
- **Spec conformance:** unresolved applicable MUSTs — RUN1, RUN10, ENV6, REL2 (EVA-M1); EVAL6, EVAL15 (EVA-M2); FT5, FT3, FT6 (EVA-M3); VER5 (EVA-m1); DAT13, DAT14, DAT19 (EVA-m3); SPL3 (EVA-m4). SHOULD deviations: EVAL10, ENV8 (EVA-M2); GDL1–GDL7, GDL9–GDL12, GDL14, UX8 (EVA-M4); VER4 (EVA-m1); UX10 (EVA-m2, EVA-m5); SPL10 (EVA-m4).

## 3. Promise and objective tracing

| Claim / objective | Implementation | Observable result | Learner interpretation | Status |
|---|---|---|---|---|
| One-pass `Run all` | cell 3 in-kernel `pip install` + stale-module guard | Kaggle pass 1 `RuntimeError`, restart, pass 2 10/10 | Section 1: "stops with a restart instruction" | **Not met** (EVA-M1) |
| Digest-verified pinned snapshot | cell 7 | 3/3 files fetched and verified, `source: local-snapshot` | clear | Met |
| Non-square synthetic sample validated into a manifest | cells 9, 11 | 320×240, aspect 1.333; manifest `accepted` + oversized-probe rejection | squash explained | Met |
| Top-5, argmax, uncalibrated scores | cell 13 | `hair spray` 0.0179 … `pick` 0.0147 | "expect a low top-1 score spread across unrelated classes" | Met |
| Evaluation report `not-measurable` | cell 15 | verdict and `needs` text | explained | Met |
| Digest-verified sample dataset | cell 17 | P6: wrong digest → warning, synthetic stripes, run continues | prose: "falls back … if air-gapped" | **Partly met** (EVA-m2) |
| "Fine-tuning of a new classification head" | cell 17 → `fit` | all 86.35 M parameters trained; 345 MB artifact | prose says head | **Not met** (EVA-M3) |
| Held-out evaluation against a majority baseline | cell 19 | CPU: 0.50 vs 0.50, verdict `success`; T4: 0.833 vs 0.50 | none; closing cell silent | **Not met as a conclusion** (EVA-M2) |
| Fresh-boundary reload | cell 19 | reloaded from files, `fine-tuned-artifact`; P3: 6/6 labels, max score diff 0.0 vs in-memory — but the notebook never compares | "verify artifact integrity" | Partly met (EVA-m1) |
| BYOD image through the same cells | cell 9 | P5: oversized → actionable `ValueError`; non-image → raw `UnidentifiedImageError`; cancel → `StopIteration` | limits stated | Partly met (EVA-m5) |
| BYOD dataset through the same validation/fine-tune/evaluate cells | cell 17 | P4: compatible zip 8/2 → fine-tune, reload, eval; no `validate_inputs`; no new-data inference | contract stated | Partly met (EVA-m3) |

| Learning objective (opening cell) | Learner activity | Evidence exercised |
|---|---|---|
| Install, read the module, resolve and verify the revision | run cells | printed identity and verified-file count |
| Generate and validate a non-square input | run cells | manifest and rejection finding printed |
| Read the argmax decision and uncalibrated scores | read output | prose guides the reading; no prompt to predict or explain |
| Execute in-kernel fine-tuning; export and fresh-reload | run cells | metrics printed; no prompt to interpret them; verdict fixed |
| Exercise BYOD; produce the evaluation report; export outputs | optional branches | no checkpoint |

The objectives are operations the code performs (GDL5); there is no learner-controlled Predict → Change → Run → Observe → Explain activity (GDL10). The "Next experiments" list is the only transfer prompt.

## 4. Journeys

| Journey | Basis | Result |
|---|---|---|
| **First-time learner** | Source inspection, all 23 cells | Inference stages are introduced with "Look for …" notes. Missing: audience, how-to-use, roadmap, task contract, glossary (EVA-02, MIM pre-training, patch 14, squash resize, CLIP normalisation, softmax, AdamW, cross-entropy, majority baseline, safetensors), predictions, checkpoints, conclusion template; the 511-line module cell is unlabelled and uncollapsed (EVA-M4). The fine-tune is described as head-only (EVA-M3) and its result is never interpreted (EVA-M2). |
| **Clean default** | Documented (Kaggle T4, reviewed blob) + direct (CPU, install skipped) | Kaggle: pass 1 failed at the install guard, pass 2 10/10 after a restart (EVA-M1). Direct: 10/10 in 158 s with a clean model cache and fresh downloads (cell 7 69 s, cell 17 72 s, cell 19 6 s); inference outputs equal the Kaggle record; fine-tune result differs (0.50 vs 0.833). Seven outputs written. No Colab run. |
| **Active learning** | Source inspection | No documented exercise changes a variable and states rerun scope. "Next experiments" suggests BYOD, `pipe.fit` on your own folder, a non-square photograph and a CPU/CUDA comparison, without predictions or rerun instructions. Not executed beyond the BYOD branches below. |
| **Reuse and recovery** | Direct (P4, P5, P6, shimmed upload) + source; real upload dialog not verified | BYOD dataset: compatible two-class zip (10 images) validated by class rule, split 8/2, fine-tuned, reloaded, evaluated (100 %, verdict `success`). Refusals: one class → "requires at least 2 distinct classes"; `../` member → "Security violation: illegal path"; no images → "No supported image files"; `.tar` → "must be a .zip file"; non-image `.png` → raw `UnidentifiedImageError`; 5000×4 images → fine-tuned, then rejected in cell 19 with the `MAX_IMAGE_SIDE` message; pre-split `val/` missing a class → accepted with one validation image, baseline 1.00, verdict `success`. BYOD image: 4200×50 → actionable `MAX_IMAGE_SIDE` error; text file → raw `UnidentifiedImageError`; cancelled upload → `StopIteration`. Sample digest mismatch → silent stripes fallback, verdict `success`. |

## 5. Findings

### Major

#### EVA-M1 — `Run all` needs a manual restart after the install cell, and the release record counts the restarted run

- **Cell/section:** cell 3, Section 1 (generator `tools/build_notebook.py` lines 48–70, the install-cell body; Section 1 prose in `tools/notebook_template.py`); the guard is enforced by `tools/validate_release_assets.py` lines 545–549; `docs/release-verification.md` Recorded executions row and Supported procedure; Troubleshooting paragraph (`tools/notebook_template.py` line 504).
- **Observed issue:** the cell `pip install`s eight pins into the running kernel, then raises `RuntimeError: Core dependencies changed while older modules were loaded … Restart the runtime, then rerun from the top.` when a loaded distribution changed. The opening cell promises that **Run all** in a fresh runtime installs the pins and completes every stage.
- **Consequence:** a learner selecting **Run all** on a stock Kaggle or Colab image hits an error in the first code cell and must restart and run again; RUN1, RUN10 and ENV6 forbid this. The release record's "PASSED — 10/10 ok code cells executed cleanly" omits the restart.
- **Evidence:** documented — Kaggle T4 run of blob `3f9b4646`: pass 1 `RuntimeError` naming `cuda-bindings` and `numpy`, `restarted_after_install_cell: true`, pass 2 10/10 (the workspace ledger records "1 restart after install cell"; the repository record does not). Source — P0: `pip_install_in_kernel: true`, `uses_uv: false`, restart instruction present.
- **Recommended correction:** adopt the fleet's **uv isolated-environment pattern**, which is how the capstone and newer workshop notebooks already run in one pass: the setup cell bootstraps uv, creates an isolated managed interpreter (`uv venv --managed-python --python 3.12.12 <ROOT>/env`), installs a hash-locked `requirements.txt` compiled with `uv pip compile` (`uv pip install --require-hashes --only-binary :all:`), and runs the pinned stages in that environment, so the kernel's preloaded NumPy/torch are never replaced and no restart can be required. Reference implementations on `main`: `ast-audio-classification-pipeline/tutorials/DIMER_Sound_Event_Classification_Workshop.ipynb` and `bioclip2-biodiversity-pipeline/tutorials/DIMER_Philippine_Biodiversity_Field_Survey_Capstone.ipynb`. Do not add another in-kernel install guard or loosen pins to dodge the restart. Implement it in `tools/build_notebook.py` (and drop the guard requirement in `tools/validate_release_assets.py`), regenerate, re-qualify with a one-pass hosted Run all, and correct the release record so a restart-dependent run is not reported as a `Run all` PASS.
- **Acceptance check:** a fresh Kaggle or Colab runtime completes every code cell in one pass with no restart and no error, recorded in `docs/release-verification.md` with the notebook blob id and `restarted: false`; `grep -n "Restart the runtime" tutorials/eva02_classification_colab.ipynb` returns nothing.
- **Spec:** RUN1, RUN10, ENV6, REL2.

#### EVA-M2 — The fine-tune evaluation cannot support a conclusion, and it reports `success` regardless of the result

- **Cell/section:** cell 17 (`SUBSET_PER_CLASS = 16`, `VALIDATION_SPLIT = 0.2`, `epochs=1`, `batch_size=2`), cell 19 (`'verdict': 'success'`, majority baseline by `max(set(val_labels), key=val_labels.count)`), closing Interpretation cell 22. Generator: `tools/notebook_template.py` lines 241, 349, 406 and the closing text from line 489.
- **Observed issue:** the held-out set is 6 images (3 per class) after 13 AdamW steps on 26 images. The report's verdict is the literal `'success'`, whatever the metrics. The majority baseline on a balanced split is a tie broken by set iteration order ("predicting 'truck'"). There is no untrained-head or zero-shot reference, no statement that 6 images cannot distinguish 5/6 from chance, and the closing section does not mention the fine-tune at all.
- **Consequence:** the learner is shown a "success" for a model that, on this review's CPU run, predicts `truck` for every image (accuracy 0.50 = baseline, lift +0.00, `frog` 0/3). On Kaggle T4 the same code and seed gave 5/6. Either way the learner is invited to conclude that fine-tuning worked, and is not told that the outcome flips between devices or that 6 images carry no statistical weight.
- **Evidence:** direct (P1/P3, CPU): history `train_loss 0.665, val_loss 0.667, val_accuracy 0.5`; report `{"verdict": "success", "accuracy": 0.5, "majority_baseline_accuracy": 0.5, "accuracy_lift_over_baseline": 0.0}`. Documented: Kaggle T4 5/6, lift +0.333. Direct (P4d): a BYOD pre-split zip with one validation image reported accuracy 1.00, baseline 1.00, verdict `success`. P6: a stripes fallback at baseline reported `success`. Source (P0): `eval_verdict_hard_coded_success: true`.
- **Recommended correction:** derive the verdict from the evidence (for example `sample-sanity` with the counts, never a fixed `success`; flag lift ≤ 0); enlarge the held-out split to a size the runtime allows on GPU (the full 400-image archive grouped by source pair, see EVA-m4) or report a simple interval with the counts; add an untrained-head or zero-shot ImageNet-mapping reference beside the majority baseline and report balanced accuracy, with a stated tie rule; state that the result varies by device and seed (ENV8); and add an interpretation paragraph for the fine-tune in the closing cell.
- **Acceptance check:** with the default run on CPU and on GPU, the cell 19 report contains no fixed `success` verdict, shows at least one non-trivial reference besides the majority class, states its sample size and variability, and the closing section interprets the fine-tune result; a run whose lift is ≤ 0 is labelled as such.
- **Spec:** EVAL6, EVAL15 (MUST); EVAL10, ENV8, DAT8.

#### EVA-M3 — "Head fine-tuning" trains every parameter of the network

- **Cell/section:** opening cell 0 ("100% in-kernel classification head fine-tuning", "head adaptation via `fit`"), cell 16 ("replaces the 1000-class head with a new linear classifier … initializes from the verified backbone"), `pipeline.py` `fit` (`torch.optim.AdamW(model.parameters(), …)`), cell 17. Generator: `tools/notebook_template.py` lines 67–69 and 215.
- **Observed issue:** `fit` creates the model with a 2-class head and optimises all parameters; nothing is frozen. The exported artifact is the full model. The fine-tune also seeds itself with `fit`'s own default `seed=20260910`, while the cell shows only `SEED = 42` (used for the split), and `weight_decay=0.01` is not shown.
- **Consequence:** the learner is taught that a new head was trained on a fixed backbone when 86 M backbone parameters were updated by 13 steps on 26 images; the trade-off the notebook implies (cheap head adaptation) is not the one demonstrated, and the recorded hyperparameters are incomplete.
- **Evidence:** direct (P3): 234 of 234 backbone tensors in the exported `model.safetensors` differ from the base snapshot (e.g. `blocks.0.attn.k_proj.weight`); artifact 345,422,632 bytes, 86,350,082 parameters. Source (P0): `fit_optimizer_over_all_parameters: true`, `fit_freezes_any_parameter: false`, `prose_states_trainable_count_or_full_finetune: false`.
- **Recommended correction:** decide which method the tutorial teaches. Either freeze the backbone in `fit` (an explicit `freeze_backbone=True` default, optimizer over trainable parameters only, head-only export with base identity for reload), or describe it as full fine-tuning. In both cases print the trainable/frozen parameter counts and record optimizer, learning rate, weight decay, epochs, batch size, the seed actually used, and precision in the cell output and result JSON.
- **Acceptance check:** cell 17 prints trainable and frozen parameter counts that match the prose; the seed printed is the one passed to `fit`; a probe comparing the exported weights with the base finds exactly the tensors the prose says were trained.
- **Spec:** FT5, FT3, FT6.

#### EVA-M4 — Declared `GUIDED`, but the guided layer is largely absent

- **Cell/section:** opening cells 0–1, every section boundary, cells 3 and 5, end of notebook. Generator: `tools/notebook_template.py` and `tools/build_notebook.py` section assembly.
- **Observed issue:** no intended-learner statement, no **How to use this notebook**, no roadmap, no Input → Model → Output task contract, no glossary, no prediction before classification or fine-tuning, no interpretation checkpoint with a sample answer, no conclusion template. Cell 5 (511 lines) and cell 3 (53 lines) carry no **Infrastructure** label and no `cellView: form`. A Troubleshooting paragraph exists but omits the dataset download, fallback, out-of-memory and BYOD dataset errors.
- **Consequence:** a self-paced learner gets accurate inference prose but no help predicting, checking or stating a conclusion, and the carried module dominates the scroll.
- **Evidence:** source inspection; P0 `guided_markers` (how_to_use, roadmap, glossary, check_your_reasoning, what_to_notice, conclusion, infrastructure, intended learner all false; `predict` is a false positive from the `predict()` API name), `cellView_form_cells: []`.
- **Recommended correction:** add the GDL layer in the template per NOTEBOOK_SPEC 2.2: audience and how-to-use, roadmap, task contract, glossary, a prediction before Sections 6 and 9, "What to notice" after each principal stage, collapsible checkpoint answers, one Predict → Change one thing → Run → Observe → Explain activity with its rerun scope (for example epochs or `SUBSET_PER_CLASS`, once EVA-M2 is fixed), troubleshooting for the fine-tune stage, and a conclusion scaffold; title cells 3 and 5 `# @title Infrastructure: …` with `cellView: form`.
- **Acceptance check:** each of GDL1–GDL7, GDL9–GDL12 and GDL14 maps to a named cell in a checklist added to `tutorials/README.md`, and cells 3 and 5 carry `cellView: form` with an Infrastructure title.
- **Spec:** GDL1–GDL7, GDL9–GDL12, GDL14, UX8.

### Minor

#### EVA-m1 — The reload is evaluated but never shown to reproduce the trained model

- **Cell/section:** cell 18 prose ("verify artifact integrity across an isolation boundary"), cell 19; `fine_tuned_pipe` from cell 17 is never used again.
- **Observed issue:** cell 19 loads the artifact and scores the validation images, but does not compare those outputs with the in-memory model's, so "loading succeeded" and "the artifact reproduces the model" are not distinguished.
- **Consequence:** a broken export (wrong head, wrong labels, missing tensors under `strict=True` exceptions aside) would still print an accuracy; the learner has no equivalence evidence.
- **Evidence:** direct (P3): the two do agree — 6/6 labels, max score difference 0.0 — so this is a missing check, not a defect in the artifact. Source (P0): `reload_compared_to_in_memory: false`.
- **Recommended correction:** predict the validation images with `fine_tuned_pipe` and `reloaded_pipe`, assert identical labels and scores within a printed tolerance, and record the comparison in the report.
- **Acceptance check:** cell 19 prints an equivalence line with its tolerance and raises if it fails.
- **Spec:** VER5 (MUST), VER4.

#### EVA-m2 — A sample-archive digest failure silently becomes a synthetic-stripes fine-tune

- **Cell/section:** cell 17, the `try … except Exception` around the download (`tools/notebook_template.py` around line 264); cell 16 prose.
- **Observed issue:** the SHA-256 mismatch `ValueError` is raised inside the `try`, so a verification failure is caught by the same blanket handler as a network error; the run prints a one-line warning and fine-tunes on 12 synthetic stripe images.
- **Consequence:** a tampered or changed archive does not stop the run; the fine-tune, the report (`success`) and the exports describe a different dataset, and a reviewer reading only the summary would not notice.
- **Evidence:** direct (P6): wrong expected digest → "falling back to deterministic synthetic dataset", `dataset_source: synthetic stripes fallback`, accuracy 0.50 = baseline, verdict `success`.
- **Recommended correction:** let a digest mismatch raise with a corrective message; if an offline fallback is kept, restrict it to network errors, label every downstream output (report, result JSON, closing text) as the fallback, and do not report it as a fine-tune result on the tutorial data.
- **Acceptance check:** with a wrong expected digest, cell 17 stops with an error naming the expected and actual digest; with the network blocked, the report and result JSON carry `dataset_source: synthetic stripes fallback` and a non-success verdict.
- **Spec:** MOD8 (by analogy for data), RUN5, UX10.

#### EVA-m3 — The BYOD dataset branch skips validation and new-data inference

- **Cell/section:** cell 17 BYOD branch, cell 19; cell 0 promises it "enters the same validation, seeded split, in-kernel fine-tuning, export, fresh-reload and held-out evaluation cells".
- **Observed issue:** archive images are never passed through `validate_inputs`, so the 4096 px ceiling is applied only when cell 19 predicts, after training; a non-image member raises a raw `UnidentifiedImageError`; a pre-split archive whose `val/` lacks a class is accepted (one validation image, baseline 1.00); there is no new-data inference after evaluation; there is no limit on archive or member size.
- **Consequence:** a user can spend a full fine-tune on data that is then rejected, gets an unhelpful error on the commonest mistake, and is told `success` on a one-image evaluation.
- **Evidence:** direct (P4): 5000×4 images fine-tuned (9 s), then cell 19 `ValueError: image side outside 1..MAX_IMAGE_SIDE=4096 px: (5000, 4)`; non-image member → `UnidentifiedImageError: cannot identify image file <_io.BytesIO …>`; pre-split missing class → 2 train / 1 val, verdict `success`. Good refusals: one class, `../` path, no images, non-zip.
- **Recommended correction:** run `validate_inputs` (or the same checks with the member name) on every archive image before splitting; wrap decode errors in a `ValueError` naming the member and accepted formats; require every class in both splits with a minimum validation count; add inference on held-back or uploaded new images; cap archive size and image count.
- **Acceptance check:** an archive with an oversized image or a non-image member is rejected in cell 17 before any training with a message naming the member and the failed rule; a pre-split archive missing a validation class is rejected; the branch ends with predictions on new images.
- **Spec:** DAT13, DAT14, DAT19 (MUST), VAL1.

#### EVA-m4 — The random split is not described as assuming independence, and duplicates are not checked

- **Cell/section:** cell 17 (per-class random split), cell 16 prose; `Cleanlab/cifar-10-subset` archive layout.
- **Observed issue:** the archive holds `original_images/` and `darkened_images/` copies of the same 100 thumbnails per class; 100 of these pairs are pixel-identical. The split treats every file as independent and nothing checks for copies across train and validation. The prose does not say that a random split assumes independent images.
- **Consequence:** with the default seed and 16 per class there happens to be no overlap, but any change to `SUBSET_PER_CLASS`, the seed or the validation share can put copies on both sides (the EfficientNet-B0 sibling notebook, which splits the full archive, has 25 of 60 held-out images that are exact copies of training images), and the learner has no warning.
- **Evidence:** direct (P2): 100 pixel-identical pairs; default subset 26/6, 0 exact copies and 0 counterparts of validation images in train, 3 of 6 validation images from `darkened_images`.
- **Recommended correction:** group each original/darkened pair (or each pixel digest) on one side of the split, or read only `original_images/`; add a pixel-digest disjointness assertion; state the independence assumption and how grouping protects it.
- **Acceptance check:** for any `SUBSET_PER_CLASS` from 2 to 200, no validation image has a pixel digest equal to a training image, enforced by an assertion in the notebook, and cell 16 names the assumption.
- **Spec:** SPL3 (MUST), SPL5, SPL10.

#### EVA-m5 — The BYOD single-image branch fails with raw exceptions on a non-image or a cancelled upload

- **Cell/section:** cell 9 BYOD branch (`next(iter(uploaded))`, `Image.open(...)`).
- **Observed issue:** a text file named `.png` raises `UnidentifiedImageError` with no corrective action; an empty upload raises a bare `StopIteration`. The oversized case is handled well by `validate_inputs` in cell 11.
- **Consequence:** unhelpful errors on the commonest BYOD mistakes.
- **Evidence:** direct (P5, shimmed upload).
- **Recommended correction:** check that exactly one file was uploaded and wrap decode in a `ValueError` naming the file and accepted formats.
- **Acceptance check:** both cases produce a `ValueError` that names the condition and the next action.
- **Spec:** DAT19, UX10.

#### EVA-m6 — Release documentation does not describe the notebook it certifies

- **Cell/section:** `docs/release-verification.md` (Executor paths, Supported procedure step 5, Recorded executions, Current status), `tutorials/README.md` table, cell 16 prose.
- **Observed issue:** the procedure's default-path checklist lists no dataset download, fine-tune, reload or fine-tuned evaluation stage; the Current status paragraph says both "No clean-runtime execution … has been recorded yet" and "clean GPU execution evidence is now recorded below" (the row is above); the executor table names a "Kaggle CPU kernel" while the run was a T4; the Run-all column in `tutorials/README.md` says "verified"; cell 16 says artifacts are "written atomically" while `fit` writes them in place with `save_file`.
- **Consequence:** a release reviewer following the procedure would not check the stages that carry this review's Majors.
- **Evidence:** source inspection; P0 `atomic_write_claimed: true`, `module_writes_atomically: false`.
- **Recommended correction:** regenerate the procedure from the notebook's actual stages, fix the status paragraph and executor table, and either write atomically (temp file + `os.replace`) or drop the claim.
- **Acceptance check:** every principal stage of the notebook appears in the procedure; the status paragraph is self-consistent; `atomically` appears only if the code uses an atomic rename.
- **Spec:** REL2, REL8.

### Suggestions

- **EVA-S1 — Use the ImageNet head as a zero-shot reference.** ImageNet-1k has frog (`tree frog`, `bullfrog`, `tailed frog`) and truck classes; a grouped-mass zero-shot score on the same 6 (or more) images would show what fine-tuning adds.
- **EVA-S2 — Declare the current spec.** The notebook and docs declare NOTEBOOK_SPEC 2.0; regenerate against 2.2 when the template is revised.
- **EVA-S3 — Drop `mode='RGB'` in `Image.fromarray`.** The Kaggle run logged Pillow's deprecation warning ("will be removed in Pillow 13 (2026-10-15)") for cell 9; the stripes fallback in cell 17 uses the same call.
- **EVA-S4 — Show the misclassified validation images** with their scores, turning cell 19's table into an interpretation activity.

## 6. Readiness

**Needs revision.** Open Majors EVA-M1 to EVA-M4; unresolved MUSTs RUN1, RUN10, ENV6, REL2 (EVA-M1), EVAL6, EVAL15 (EVA-M2), FT3, FT5, FT6 (EVA-M3), VER5 (EVA-m1), DAT13, DAT14, DAT19 (EVA-m3), SPL3 (EVA-m4). Remaining gates after the fixes: a one-pass hosted Run all of the regenerated blob, a fine-tune evaluation whose verdict follows from its numbers on both CPU and GPU, and hosted BYOD positive and negative runs.

## 7. Verified versus inferred

- **Verified by direct execution (CPU, install skipped):** the default path (10/10, inference identical to Kaggle), the CPU fine-tune outcome and fixed `success` verdict, full-network training and reload equivalence (P3), the split overlap counts (P2), the BYOD dataset and image branches with eleven inputs (P4, P5), and the digest-mismatch fallback (P6).
- **Verified from documented execution:** the install-cell restart and the T4 fine-tune result 5/6.
- **Inferred from source:** Colab upload-dialog behaviour, behaviour on a hosted 2-vCPU runtime (this host has 24 logical CPUs, so its 158 s is not a hosted CPU figure), and the effect of the proposed fixes.
- **Most likely to be wrong:** EVA-M2's framing of the CPU result. The 0.50 outcome comes from one CPU run at the notebook's seed; a different CPU or thread count might land elsewhere. The finding rests on the fixed verdict and the 6-image evaluation, which hold whatever the number.

Probe ZIP: `eva02_classification_colab_Review_Probes.zip` (`run_probes.py`, `results.json`, `source_manifest.json`).
