"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.2 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
module, and the model pin/stage/verify cells are produced by the generator from repository
sources so they cannot drift from the package.

Review fixes (Notebook Review Framework v1, review PR #9, EVA-M1..M4 / EVA-m1..m6): the runtime is the fleet's uv
isolated environment (no in-kernel install, no restart); the fine-tuning method is stated as it runs (
head-only fine-tuning by default, full fine-tuning as the Section 12 activity) and its configuration, trainable set and dataset provenance are
printed and exported; the held-out verdict is computed from the counts with a 95 % Wilson interval against the
majority baseline instead of a hard-coded "success"; the dataset archive is validated before training; the air-gapped
fallback works; the reload is compared with the in-memory model; and the guided layer (who it is for, how to use,
roadmap, predictions, worked answers, a change-one-thing activity, troubleshooting, glossary, conclusion) is added.

Code cells are written with single braces and escaped by ``_py`` for the generator's ``str.format`` pass; ``@STEM@``
becomes the output stem.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering


def _py(code: str) -> str:
    """Escape a code cell for the generator's ``str.format`` pass; ``@STEM@`` stands for ``{stem}``."""
    return code.replace("{", "{{").replace("}", "}}").replace("@STEM@", "{stem}")


_SAMPLE_CODE = _py(
    """import hashlib
import io
import os

import numpy as np
from PIL import Image, UnidentifiedImageError

USE_BYOD = False  # @param {type:"boolean"}
BYOD_IMAGE_PATH = ''  # @param {type:"string"}
GROUND_TRUTH_INDEX = -1  # @param {type:"integer"}
SAMPLE_WIDTH = 320
SAMPLE_HEIGHT = 240

if USE_BYOD:
    if BYOD_IMAGE_PATH:
        image_name = os.path.basename(BYOD_IMAGE_PATH)
        with open(BYOD_IMAGE_PATH, 'rb') as handle:
            image_bytes = handle.read()
    else:
        from google.colab import files
        uploaded = files.upload()
        if len(uploaded) != 1:
            raise ValueError(f'Upload exactly one image file (got {len(uploaded)}). Outside Colab, set BYOD_IMAGE_PATH to a file path instead.')
        image_name, image_bytes = next(iter(uploaded.items()))
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as exc:
        raise ValueError(f'{image_name}: not a decodable image ({type(exc).__name__}). Use a PNG, JPEG, WebP or BMP file and run this cell again.') from None
    sample_kind = 'BYOD'
else:
    # Deterministic synthetic gradient, deliberately non-square so the squash to 448x448 is visible.
    red = np.tile(np.linspace(0.0, 255.0, SAMPLE_WIDTH), (SAMPLE_HEIGHT, 1))
    green = np.tile(np.linspace(0.0, 255.0, SAMPLE_HEIGHT), (SAMPLE_WIDTH, 1)).T
    blue = (red + green) / 2.0
    array = np.rint(np.stack([red, green, blue], axis=-1)).astype(np.uint8)
    image = Image.fromarray(array, mode='RGB')
    image_name = f'synthetic_gradient_{SAMPLE_WIDTH}x{SAMPLE_HEIGHT}.png'
    sample_kind = 'synthetic'

if GROUND_TRUTH_INDEX != -1 and not 0 <= GROUND_TRUTH_INDEX < NUM_CLASSES:
    raise ValueError(f'GROUND_TRUTH_INDEX must be -1 (unknown) or an ImageNet-1k class index in 0..{NUM_CLASSES - 1}, got {GROUND_TRUTH_INDEX}.')
ground_truth = None if GROUND_TRUTH_INDEX == -1 else GROUND_TRUTH_INDEX
sample_sha256 = hashlib.sha256(np.asarray(image.convert('RGB')).tobytes()).hexdigest()
aspect_ratio = round(image.size[0] / image.size[1], 3)
print({'sample_kind': sample_kind, 'name': image_name, 'mode': image.mode, 'size': image.size, 'aspect_ratio': aspect_ratio, 'rgb_sha256': sample_sha256, 'ground_truth_index': ground_truth})"""
)

_VALIDATE_CODE = _py(
    """import json
import os

os.makedirs('outputs', exist_ok=True)
print({'ceilings': {'NUM_CLASSES': NUM_CLASSES, 'MAX_IMAGE_SIDE': MAX_IMAGE_SIDE, 'MAX_BATCH': MAX_BATCH}})
input_manifest = validate_inputs(image, top_k=5, names=[image_name])
# Demonstrate rejection on an input that breaks a ceiling; the finding is recorded, not swallowed.
try:
    validate_inputs(Image.new('RGB', (MAX_IMAGE_SIDE + 1, 8)))
except ValueError as exc:
    input_manifest['findings'].append({'input': 'oversized-probe', 'verdict': 'rejected', 'message': str(exc)})
with open('outputs/@STEM@_input_manifest.json', 'w', encoding='utf-8') as handle:
    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)
print(json.dumps(input_manifest, indent=2))"""
)

_CLASSIFY_CODE = _py(
    """result = pipe.predict(image, top_k=5)
prediction = result['predictions'][0]
print({'decision_rule': result['decision_rule'], 'predicted_index': prediction['predicted_index'], 'predicted_label': prediction['predicted_label'], 'device': result['device'], 'source': result['source']})
for rank, item in enumerate(prediction['top_k'], start=1):
    print(f"{rank:>2}. index {item['index']:>4}  score {item['score']:.4f}  {item['label']}")"""
)

_EVALUATE_CODE = _py(
    """targets = None if ground_truth is None else [ground_truth]
report = evaluation_report(result, targets, sample_kind=sample_kind)
with open('outputs/@STEM@_evaluation_report.json', 'w', encoding='utf-8') as handle:
    json.dump(report, handle, indent=2, ensure_ascii=False)
print(json.dumps(report, indent=2))
if report['verdict'] == 'not-measurable':
    print('No ground-truth class index was supplied, so top_k_accuracy is not computed; the prediction above is sanity evidence only.')"""
)

_DATA_CODE = _py(
    """import hashlib
import os
import urllib.request

USE_BYOD_DATASET = False  # @param {type:"boolean"}
BYOD_DATASET_PATH = ''  # @param {type:"string"}
SPLIT_SEED = 42  # @param {type:"integer"}
SAMPLE_DATASET_URL = 'https://huggingface.co/datasets/Cleanlab/cifar-10-subset/resolve/bb5a7aabf1d14d2d1e3e49d0d8f917bda3622f75/CIFAR-10-subset.zip'
SAMPLE_DATASET_SHA256 = '66f90a4f87d865e8eb653b62f10e754684075a32314177de76832349d4b1fb19'
VALIDATION_SPLIT = 0.2
SUBSET_PER_CLASS = 100  # images per class kept from the tutorial dataset (50 photo pairs: 80 train + 20 held out)

if USE_BYOD_DATASET:
    if BYOD_DATASET_PATH:
        zip_name = os.path.basename(BYOD_DATASET_PATH)
        with open(BYOD_DATASET_PATH, 'rb') as handle:
            zip_bytes = handle.read()
    else:
        from google.colab import files
        uploaded = files.upload()
        if len(uploaded) != 1:
            raise ValueError(f'Upload exactly one dataset .zip archive (got {len(uploaded)}). Outside Colab, set BYOD_DATASET_PATH instead.')
        zip_name, zip_bytes = next(iter(uploaded.items()))
    if not zip_name.lower().endswith('.zip'):
        raise ValueError(f'BYOD archive must be a .zip file, got {zip_name}')
    dataset = load_image_zip(zip_bytes, seed=SPLIT_SEED, validation_split=VALIDATION_SPLIT, source=f'user upload: {zip_name}')
    dataset_kind = 'BYOD'
else:
    zip_bytes, download_error = None, None
    try:
        request = urllib.request.Request(SAMPLE_DATASET_URL, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(request, timeout=30) as response:
            zip_bytes = response.read()
    except OSError as exc:  # no network, DNS failure, HTTP error, timeout
        download_error = exc
    if zip_bytes is not None:
        actual_sha = hashlib.sha256(zip_bytes).hexdigest()
        if actual_sha != SAMPLE_DATASET_SHA256:
            raise ValueError(f'The tutorial dataset from {SAMPLE_DATASET_URL} has SHA-256 {actual_sha}, expected {SAMPLE_DATASET_SHA256}: refusing to train on it. Run this cell again; if it repeats, the download is being altered.')
        dataset = load_image_zip(zip_bytes, seed=SPLIT_SEED, validation_split=VALIDATION_SPLIT, subset_per_class=SUBSET_PER_CLASS, source='Cleanlab/cifar-10-subset @ bb5a7aab (MIT)')
        dataset_kind = 'tutorial dataset'
    else:
        print(f'Warning: the tutorial dataset could not be downloaded from {SAMPLE_DATASET_URL} ({download_error}); falling back to the deterministic synthetic stripes dataset (2 classes, 8 training and 4 held-out images). Results below then describe stripes, not photographs.')
        dataset = synthetic_stripes_dataset()
        dataset_kind = 'synthetic fallback'

CUSTOM_CLASSES = dataset['classes']
train_images, train_targets = dataset['train_images'], dataset['train_targets']
val_images, val_targets = dataset['val_images'], dataset['val_targets']
dataset_manifest = dataset['manifest']
# Two held-out images that share a file name with a training image of the same class may be copies of one picture
# (the tutorial archive stores each photo twice: original_images/ and darkened_images/).
train_keys = {(t, os.path.basename(i)) for t, i in zip(train_targets, dataset['train_ids'])}
shared_names = [i for t, i in zip(val_targets, dataset['val_ids']) if (t, os.path.basename(i)) in train_keys]
# A random split assumes the held-out images are independent of the training images; a pixel-identical copy on both
# sides breaks that (the model could score it from memory). Check it by pixel digest, whatever the file names.
train_digests = {hashlib.sha256(img.convert('RGB').tobytes()).hexdigest() for img in train_images}
held_out_pixel_copies = [i for img, i in zip(val_images, dataset['val_ids']) if hashlib.sha256(img.convert('RGB').tobytes()).hexdigest() in train_digests]
if held_out_pixel_copies and dataset_kind != 'BYOD':
    raise RuntimeError(f'{len(held_out_pixel_copies)} held-out image(s) are pixel-identical copies of training images, e.g. {held_out_pixel_copies[:3]}: the split is not independent.')
if held_out_pixel_copies:
    print(f'Warning: {len(held_out_pixel_copies)} held-out image(s) in your archive are pixel-identical copies of training images (e.g. {held_out_pixel_copies[:3]}); they inflate the held-out accuracy. Remove the duplicates and run this cell again.')
dataset_provenance = {
    'kind': dataset_kind,
    'source': dataset_manifest['source'],
    'sha256': dataset_manifest['sha256'],
    'classes': CUSTOM_CLASSES,
    'per_class': dataset_manifest['per_class'],
    'split_seed': dataset_manifest.get('seed'),
    'subset_per_class': dataset_manifest.get('subset_per_class'),
    'validation_split': dataset_manifest.get('validation_split'),
    'held_out_sharing_a_file_name_with_train': len(shared_names),
    'held_out_pixel_copies_in_train': len(held_out_pixel_copies),
}
print({'dataset': dataset_kind, 'source': dataset_manifest['source'], 'sha256': dataset_manifest['sha256'], 'layout': dataset_manifest['layout'], 'image_size_px': list(train_images[0].size)})
print(f"{'class':<32}{'train':>6}{'held out':>10}")
for name, counts in dataset_manifest['per_class'].items():
    print(f"{name:<32}{counts['train']:>6}{counts['val']:>10}")
print({'train_images': len(train_images), 'held_out_images': len(val_images), 'held_out_sharing_a_file_name_with_train': len(shared_names), 'held_out_pixel_copies_in_train': len(held_out_pixel_copies), 'skipped_files': dataset_manifest['skipped'][:10]})"""
)

_FIT_CODE = _py(
    """TRAINABLE = 'head'  # @param ["all", "head"]
EPOCHS = 1  # @param {type:"integer"}
BATCH_SIZE = 4  # @param {type:"integer"}
LEARNING_RATE = 1e-4  # @param {type:"number"}
WEIGHT_DECAY = 0.01  # @param {type:"number"}
TRAIN_SEED = 20260910  # @param {type:"integer"}

if TRAINABLE not in ('all', 'head'):
    raise ValueError(f"TRAINABLE must be 'all' (full fine-tuning) or 'head' (head only), got {TRAINABLE!r}")
train_backbone = TRAINABLE == 'all'
planned = trainable_parameter_counts(len(CUSTOM_CLASSES), train_backbone=train_backbone)
print({'method': planned['method'], 'trainable_parameters': planned['trainable_parameters'], 'frozen_parameters': planned['frozen_parameters'], 'epochs': EPOCHS, 'batch_size': BATCH_SIZE, 'learning_rate': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY, 'seed': TRAIN_SEED})

ft_output_dir = 'outputs/@STEM@_finetuned'
fine_tuned_pipe, train_meta = pipe.fit(
    train_images=train_images,
    train_targets=train_targets,
    val_images=val_images,
    val_targets=val_targets,
    class_names=CUSTOM_CLASSES,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    seed=TRAIN_SEED,
    train_backbone=train_backbone,
    weights_dir=WEIGHTS_DIR,
    output_dir=ft_output_dir,
    provenance=dataset_provenance,
)
finetune_config = train_meta['config']
if (finetune_config['trainable_parameters'], finetune_config['frozen_parameters']) != (planned['trainable_parameters'], planned['frozen_parameters']):
    raise RuntimeError(f'fit trained a different parameter set than the one printed above: {finetune_config}')
print({'fine_tuning': 'complete', 'method': finetune_config['method'], 'device': train_meta['device'], 'classes': CUSTOM_CLASSES, 'train_samples': finetune_config['train_samples'], 'val_samples': finetune_config['val_samples']})
for row in train_meta['history']:
    print({key: round(value, 4) if isinstance(value, float) else value for key, value in row.items()})"""
)

_RELOAD_CODE = _py(
    """import csv
import hashlib

EQUIVALENCE_TOLERANCE = 1e-4  # largest allowed softmax-score difference between the reloaded and in-memory model

reloaded_pipe = EVA02ClassificationPipeline.from_pretrained(weights_dir=ft_output_dir)

# Fresh-boundary check: the reloaded artifact must reproduce the in-memory model on every held-out image.
predicted_indices, predicted_scores, label_agreement, max_score_difference = [], [], 0, 0.0
for img in val_images:
    reloaded = reloaded_pipe.predict(img)['predictions'][0]
    in_memory = fine_tuned_pipe.predict(img)['predictions'][0]
    label_agreement += reloaded['predicted_index'] == in_memory['predicted_index']
    in_memory_scores = {item['index']: item['score'] for item in in_memory['top_k']}
    for item in reloaded['top_k']:
        max_score_difference = max(max_score_difference, abs(item['score'] - in_memory_scores.get(item['index'], 0.0)))
    predicted_indices.append(reloaded['predicted_index'])
    predicted_scores.append(reloaded['top_k'][0]['score'])
equivalence = {'images': len(val_images), 'label_agreement': label_agreement, 'max_score_difference': max_score_difference, 'tolerance': EQUIVALENCE_TOLERANCE}
equivalence['equivalent'] = label_agreement == len(val_images) and max_score_difference <= EQUIVALENCE_TOLERANCE
print({'fresh_boundary_source': reloaded_pipe.source, **equivalence})
if not equivalence['equivalent']:
    raise RuntimeError(f'The reloaded artifact does not reproduce the in-memory model ({equivalence}). Re-run Section 9; if it repeats, the export is broken and must not be used.')
artifact_sha256 = {}
for name in ('model.safetensors', 'model-config.json'):
    with open(os.path.join(ft_output_dir, name), 'rb') as handle:
        artifact_sha256[name] = hashlib.sha256(handle.read()).hexdigest()

# Held-out evaluation of the reloaded artifact, with the uncertainty that n images allow.
finetuned_eval_report = finetune_evaluation_report(val_targets, predicted_indices, CUSTOM_CLASSES, item_ids=dataset['val_ids'], scores=predicted_scores, sample_kind=dataset_kind)
finetuned_eval_report.update({'fresh_boundary_source': reloaded_pipe.source, 'equivalence': equivalence, 'artifact_sha256': artifact_sha256, 'fine_tuning_method': finetune_config['method']})
with open('outputs/@STEM@_finetuned_evaluation_report.json', 'w', encoding='utf-8') as handle:
    json.dump(finetuned_eval_report, handle, indent=2, ensure_ascii=False)
with open('outputs/@STEM@_validation_predictions.csv', 'w', encoding='utf-8', newline='') as handle:
    writer = csv.writer(handle)
    writer.writerow(['sample_index', 'item_id', 'true_label', 'predicted_label', 'correct', 'confidence'])
    for k, (item_id, target, predicted, score) in enumerate(zip(dataset['val_ids'], val_targets, predicted_indices, predicted_scores)):
        writer.writerow([k, item_id, CUSTOM_CLASSES[target], CUSTOM_CLASSES[predicted], target == predicted, f'{score:.6f}'])

metrics = finetuned_eval_report['metrics']
low, high = metrics['accuracy_wilson_95']
print(f"Held-out images: {finetuned_eval_report['n']} ({dataset_kind}) - a tutorial metric, not a benchmark")
print(f"Accuracy:        {finetuned_eval_report['correct']}/{finetuned_eval_report['n']} = {metrics['accuracy']:.1%}   95% Wilson interval {low:.1%} to {high:.1%}")
print(f"Majority-class baseline: {metrics['majority_baseline_accuracy']:.1%} (always answering '{metrics['majority_baseline_class']}')")
print(f"Verdict: {finetuned_eval_report['verdict']}; comparison to baseline: {finetuned_eval_report['comparison_to_baseline']}")
for name, counts in finetuned_eval_report['per_class'].items():
    print(f"  {name:<32}{counts['correct']}/{counts['total']} correct")
print({'misclassified': finetuned_eval_report['misclassified']})

run_history = globals().get('run_history', [])
run_history.append({'method': finetune_config['method'], 'trainable_parameters': finetune_config['trainable_parameters'], 'epochs': finetune_config['epochs'], 'learning_rate': finetune_config['learning_rate'], 'split_seed': dataset_provenance['split_seed'], 'dataset': dataset_kind, 'final_val_loss': round(train_meta['history'][-1]['val_loss'], 4), 'correct': finetuned_eval_report['correct'], 'n': finetuned_eval_report['n'], 'wilson_95': [round(low, 3), round(high, 3)], 'comparison_to_baseline': finetuned_eval_report['comparison_to_baseline']})"""
)

_EXPORT_CODE = _py(
    """import csv

payload = {
    'prediction': result,
    'evaluation_report': report,
    'input_manifest': input_manifest,
    'fine_tuning': {
        'config': finetune_config,
        'dataset': dataset_manifest,
        'history': train_meta['history'],
        'classes': CUSTOM_CLASSES,
        'evaluation': finetuned_eval_report,
        'artifact_dir': ft_output_dir,
        'artifact_sha256': artifact_sha256,
    },
    'sample': {'kind': sample_kind, 'name': image_name, 'size': list(image.size), 'aspect_ratio': aspect_ratio, 'rgb_sha256': sample_sha256, 'ground_truth_index': ground_truth},
    'notebook_source': NOTEBOOK_SOURCE,
    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],
    'model_id': MODEL_ID,
    'model_revision': MODEL_REVISION,
    'model_license': MODEL_LICENSE,
    'runtime': {
        'python': platform.python_version(),
        'torch': torch.__version__,
        'timm': timm.__version__,
        'device': pipe.device,
    },
}
with open('outputs/@STEM@_result.json', 'w', encoding='utf-8') as handle:
    json.dump(payload, handle, indent=2, ensure_ascii=False)
with open('outputs/@STEM@_top_k.csv', 'w', encoding='utf-8', newline='') as handle:
    writer = csv.writer(handle)
    writer.writerow(['image', 'rank', 'index', 'label', 'score'])
    for rank, item in enumerate(prediction['top_k'], start=1):
        writer.writerow([image_name, rank, item['index'], item['label'], f"{item['score']:.6f}"])
print({'outputs': sorted(os.listdir('outputs')), 'finetuned': sorted(os.listdir(ft_output_dir))})
print(['outputs/@STEM@_finetuned/model.safetensors', 'outputs/@STEM@_finetuned/model-config.json', 'outputs/@STEM@_validation_predictions.csv', 'outputs/@STEM@_finetuned_evaluation_report.json'])"""
)

_ACTIVITY_CODE = _py(
    """# Section 10 adds one row per fine-tuning run in this session; this cell only prints them.
columns = ['method', 'trainable_parameters', 'epochs', 'learning_rate', 'split_seed', 'final_val_loss', 'correct', 'n', 'wilson_95', 'comparison_to_baseline']
print(' | '.join(columns))
for row in run_history:
    print(' | '.join(str(row[column]) for column in columns))
if len(run_history) == 1:
    print("One run so far. Set TRAINABLE = 'all' in Section 9, select that cell and choose Runtime > Run after; this table then shows both runs.")"""
)

TEMPLATE = {
    "package": "eva02_classification_pipeline",
    "repo_name": "eva02-classification-pipeline",
    "stem": "eva02_classification",
    "notebook_name": "eva02_classification_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "infrastructure_labels": True,
    "collapse_model_cell": True,
    "isolated_runtime": True,
    # The fleet's uv isolated-environment mechanism (ast-audio-classification-pipeline / bioclip2-biodiversity-pipeline):
    # managed CPython, a size- and SHA-256-verified uv wheel, and a lock compiled from the pyproject pins with
    # `uv pip compile pyproject.toml --python-version 3.12 --python-platform x86_64-manylinux_2_28 --generate-hashes
    # --only-binary :all: -o tutorials/requirements-colab.lock.txt`.
    "managed_python": "3.12.12",
    "uv": {
        "version": "0.12.15",
        "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "bytes": 20081404,
        "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
    },
    "lock": "tutorials/requirements-colab.lock.txt",
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime builds an isolated environment from the hash-locked pins (the "
        "kernel's own packages are left alone, so no restart is needed), stages and digest-verifies the pinned snapshot, "
        "generates the deterministic synthetic sample image, validates it into an input manifest, classifies it with the "
        "pretrained ImageNet-1k head, writes the zero-shot evaluation report, downloads and digest-verifies the "
        "`Cleanlab/cifar-10-subset` tutorial dataset and validates a seeded, pair-grouped split of 100 images per class "
        "(160 training, 40 held out; both copies of a photo stay on one side), fine-tunes a new two-class head on the frozen "
        "pretrained backbone (**head-only fine-tuning**, one epoch, AdamW and "
        "cross-entropy), exports the artifact (`model.safetensors` + `model-config.json`), reloads it across a fresh "
        "boundary and checks it reproduces the in-memory model, evaluates it on the 40 held-out images with a 95 % interval "
        "against the majority-class baseline, and writes outputs and provenance. No repository clone, DIMER worker or "
        "service, credential, upload dialog or configuration edit is required (NOTEBOOK_SPEC 2.2 §5). Measured times: "
        "the previous notebook version (one image of inference and a 26-image fine-tune) took 209.4 s on a Kaggle Tesla T4 "
        "(2026-09-14, after a restart of the install cell); this version's default path took 238 s of cell time in a "
        "local Windows CPU check (2026-10-04, 24 threads, files pre-staged; 140 s of it the head-only fine-tuning on 160 "
        "images and 89 s the reload check and held-out evaluation), and the optional full fine-tuning of Section 12 took "
        "318 s there. A GPU is much faster; no GPU run of this version is recorded yet. Building the isolated environment "
        "(PyTorch with its CUDA libraries) and downloading the 348 MB checkpoint come on top of these and usually take a "
        "few minutes (an estimate)."
    ),
    "byod": (
        "Two optional branches, both off by default and never part of the default path. `USE_BYOD = True` in Section 4 "
        "takes one image (upload dialog on Colab, or a file path in `BYOD_IMAGE_PATH` on any runtime) through the same "
        "validation, classification, evaluation-report and export cells as the synthetic sample. `USE_BYOD_DATASET = True` "
        "in Section 8 takes a `.zip` of class folders (upload, or `BYOD_DATASET_PATH`) through the same validation, seeded "
        "split, fine-tuning, export, fresh-reload and held-out evaluation cells as the tutorial dataset; Section 8 refuses "
        "a bad archive (named file or class, and the rule) before any training. After the default run, set the field, "
        "select that section's cell and choose **Runtime → Run after**. The archive contract, the limits and the privacy "
        "guidance are in the Prerequisites; uploads stay inside this runtime."
    ),
    "pipeline_class": "EVA02ClassificationPipeline",
    "weights_key": "eva02-base-448",
    "runtime_imports": ["torch", "timm"],
    "title": "EVA-02 Base 448 — DIMER image classification tutorial (standalone)",
    "badges": [
        (
            "GitHub",
            "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/kurtvalcorza/eva02-classification-pipeline",
        ),
        (
            "Open In Colab",
            "https://colab.research.google.com/assets/colab-badge.svg",
            "https://colab.research.google.com/github/kurtvalcorza/eva02-classification-pipeline/blob/main/tutorials/eva02_classification_colab.ipynb",
        ),
        (
            "Hugging Face",
            "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-timm%2Feva02__base__patch14__448-ffcc4d?style=flat",
            "https://huggingface.co/timm/eva02_base_patch14_448.mim_in22k_ft_in22k_in1k",
        ),
        (
            "Upstream",
            "https://img.shields.io/badge/Upstream-baaivision%2FEVA-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/baaivision/EVA",
        ),
        ("arXiv", "https://img.shields.io/badge/arXiv-2303.11331-b31b1b.svg", "https://arxiv.org/abs/2303.11331"),
    ],
    "capability": "ImageNet-1k single-label image classification (1000 classes) and in-kernel fine-tuning on custom classes (head-only by default, or full) using the pinned EVA-02 Base patch-14 448 px weights",
    "intro": (
        "At inference the pipeline **squash-resizes every input to a fixed 448 x 448** (aspect ratio is not preserved — "
        "a non-square image is stretched, not cropped), normalises with the CLIP mean/std from the snapshot config, runs "
        "one forward pass of the vision transformer, and applies a softmax over 1000 logits. "
        "**In-kernel fine-tuning:** this notebook demonstrates both zero-shot base inference on ImageNet-1k classes and "
        "fine-tuning on custom classes with PyTorch AdamW and cross-entropy loss. `fit` replaces the 1000-class head with "
        "a new linear classifier sized to your classes and, by default, trains **only that classifier** with the "
        "pretrained backbone frozen (head-only fine-tuning: 1,538 parameters); with `TRAINABLE = 'all'` it trains "
        "every parameter, the backbone included (full fine-tuning: 86.35 M parameters). The notebook prints the trainable and frozen parameter counts before training. "
        "The carried pipeline module adds snapshot verification, input and dataset validation, `fit`, a held-out "
        "evaluation report with its uncertainty, and the `top_k_accuracy`, `validate_inputs` and `evaluation_report` "
        "helpers. The default inference sample is a synthetic image generated in code; its prediction is demonstration "
        "(plumbing) evidence, not a production-quality or benchmark claim. The fine-tuning result is measured on **40 "
        "held-out images** (20 photos, each in two copies), so it is a tutorial metric with a wide 95 % interval, not evidence that the method works "
        "in general.\n\n"
        "**Who this is for.** A learner who knows basic Python and has met the idea of an image classifier, and wants to "
        "see how a pretrained network is used, adapted to new classes and checked honestly. No prior experience with "
        "EVA-02, vision transformers, timm or fine-tuning is assumed; each term is explained where it is first used and "
        "again in the **Glossary** at the end. No GPU is required, but this is a large model: on CPU the default path "
        "takes minutes (a GPU is used automatically).\n\n"
        "**Input → Model → Output.**\n\n"
        "| | ImageNet-1k inference (Sections 4–7) | Fine-tuning (Sections 8–11) |\n"
        "|---|---|---|\n"
        "| Input | one RGB image (any size up to 4096 px), squash-resized to 448×448 (aspect ratio not kept, nothing cropped) | a labelled image set: class folders, split into training and held-out images |\n"
        "| Model | EVA-02 Base vision-transformer backbone (patch 14, 448 px) + the pretrained 1000-class head | the same backbone + a new head sized to your classes, trained with AdamW |\n"
        "| Output | the argmax class and a ranked top-5 with softmax scores (not probabilities of being right) | an artifact (`model.safetensors` + `model-config.json`) and held-out accuracy with a 95 % interval and the majority-class baseline |\n\n"
        "**How to use this notebook.** Choose a runtime (Colab, Kaggle or Linux Jupyter; a GPU is faster, CPU works), then "
        "**Runtime → Run all**. Sections 1–3 are **infrastructure** — the isolated environment, the carried code and the "
        "model verification — and can be run without study; their code is collapsed. The learning path starts in Section 4. "
        "Form fields (`# @param`) are the only values meant to be edited; the defaults reproduce the default path. Each "
        "learner section states what it does and asks you to **predict** before it runs; the next section opens with "
        "**What to notice** and a collapsible **Check your reasoning** block with a worked answer. Each optional experiment "
        "names the field to change and the cells to re-run. Section 12 is a **change-one-thing activity**. "
        "**Troubleshooting**, a **Glossary** and a **Conclusion** template are at the end. Your notes are optional.\n\n"
        "**Roadmap:** *core concepts* — 4 the sample → 5 validate it → 6 classify with the pretrained model → 7 the "
        "evaluation report; *evaluation practice* — 8 the fine-tuning data and its split → 9 fine-tune → 10 reload, "
        "check equivalence and evaluate on held-out images → 11 export outputs and provenance (*engineering*) → 12 "
        "**change one thing: train every layer** → conclude."
    ),
    "learning_objectives": (
        "by the end you should be able to (1) explain *image → squash-resize to 448×448 → EVA-02 → softmax → argmax and top-5* "
        "and say why the score is not a probability of being right (Section 6); (2) explain why a prediction on the "
        "synthetic gradient is `not-measurable` and what would make it measurable (Section 7); (3) identify how many "
        "parameters head-only and full fine-tuning train, from the printed counts (Sections 9 and 12); (4) interpret a held-out "
        "accuracy together with its sample size, its 95 % interval and the majority-class baseline, and decide whether it "
        "supports a claim (Section 10); (5) check that an exported artifact reproduces the evaluated model (Section 10); "
        "and (6) predict, measure and explain how full fine-tuning compares with head-only fine-tuning on the same split, "
        "including why it can collapse to one class (Section 12)."
    ),
    "exclusions": (
        "object detection, segmentation, multi-label tagging, OCR, open-vocabulary classification, feature/embedding "
        "extraction (the DINOv2 sibling covers that). The base label space is fixed to the 1000 ImageNet-1k classes; an "
        "image whose subject is outside that space still receives a label unless adapted via in-kernel fine-tuning. The fine-tuning sections demonstrate the "
        "procedure on a tiny sample; they do not establish that fine-tuning improves accuracy on any task."
    ),
    "prerequisites": [
        "- **Learner:** basic Python and Colab or Jupyter familiarity; what an image classifier does. The notebook explains the softmax over class logits (and why it is not a calibrated probability), fine-tuning (full vs head-only), AdamW, cross-entropy, a held-out split, the majority-class baseline and a confidence interval where they are first used; the Glossary repeats them.",
        "- **Runtime:** a fresh **Linux x86_64** runtime — Google Colab, Kaggle or Linux Jupyter. Section 1 builds its own Python 3.12.12 environment from a hash-locked list of manylinux wheels, so the kernel's own Python version does not matter, and a Windows or macOS kernel is not supported (Section 1 stops with that message). The default path runs on CPU and uses CUDA automatically when available; inference and training are float32. The locked install (PyTorch 2.14.0 with its CUDA libraries) and the 348 MB checkpoint are the largest downloads. This is the heaviest of the DIMER timm classifiers (107 GMACs per 448 px image). **CPU works but is slow**: in a local Windows CPU check (24 threads, files pre-staged) this version's default path took 238 s of cell time, 140 s of it the head-only fine-tuning, and the optional full fine-tuning of Section 12 took 318 s; a small hosted CPU runtime will be slower. A GPU runtime is recommended. The previous notebook version took 209.4 s on a Kaggle Tesla T4 (2026-09-14). The isolated install and the downloads add a few minutes (an estimate).",
        "- **Data (default path):** the inference sample is a deterministic 320 x 240 RGB gradient generated in code (no ground truth), deliberately **not square** so that the squash to 448 x 448 is visible in the printed aspect ratio. The fine-tuning sections download one dataset: [`Cleanlab/cifar-10-subset`](https://huggingface.co/datasets/Cleanlab/cifar-10-subset) at commit `bb5a7aab` (MIT licence), a 986,707-byte `.zip` from `huggingface.co`, refused unless its SHA-256 is `66f90a4f…`. It holds 400 CIFAR-10 images of 32×32 px (upscaled to 448 px by the model's preprocessing) in two classes, `frog` and `truck`, each stored twice: an original and a darkened copy. The notebook keeps 100 per class — 50 photos, each with both copies — and holds out 20 per class (10 photos), always keeping a photo's two copies on the same side of the split. If the download fails, Section 8 falls back to a synthetic two-class stripes dataset and says so.",
        "- **BYOD image (Section 4):** one image file decodable by Pillow (PNG, JPEG, WebP, BMP and similar), any colour mode, longest side at most 4096 px; optionally its ImageNet-1k class index (0–999). It is squash-resized to 448 x 448 regardless of its aspect ratio.",
        "- **BYOD dataset (Section 8):** one `.zip` of at most 200 MB and 2,000 images (`.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`; longest side at most 4096 px) in **either** `<class>/<image>` folders (split here: per class, seeded, 20 % held out, at least 2 images per class) **or** `train/<class>/…` plus `val/<class>/…` (also `valid/`, `validation/`) used as given. At least 2 and at most 100 classes; every image must be inside a class folder; every validation class must also exist in `train/`; other files are skipped and listed. Section 8 checks all of this and decodes every image before any training, and names the file or class it refuses. Validation is structural, not semantic: nothing checks that an image shows its folder's class.",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded inputs remain in the notebook runtime; this pipeline does not send them to a third-party inference API.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Generate the synthetic sample or optional BYOD\n\n"
                "The default sample is **synthetic**: a deterministic 320 x 240 RGB gradient built in code (red ramps left to "
                "right, green top to bottom, blue is their mean), so it needs no download and its SHA-256 is printed for the "
                "record. It is deliberately **not square** so that the squash to 448 x 448 is visible in the printed aspect "
                "ratio. A gradient is not a photograph of any ImageNet class, so it has **no ground truth**: whatever label "
                "the model returns is a sanity check that the input contract, preprocessing and forward pass work, not a "
                "correctness measurement. BYOD is optional and disabled by default; when enabled, upload one image file (or "
                "set `BYOD_IMAGE_PATH` outside Colab) and, if you know its ImageNet-1k class index (0–999), set "
                "`GROUND_TRUTH_INDEX` so the evaluation step can compute `top_k_accuracy`. Leave it at `-1` when the label is "
                "unknown. A file that is not a decodable image is refused with its name. Look for a dictionary naming the "
                "sample kind, its size, aspect ratio and digest, and whether a ground-truth index was supplied."
            ),
            "code": _SAMPLE_CODE,
        },
        {
            "md": (
                "## 5. Validate the input → input manifest\n\n"
                "`validate_inputs` is the pipeline's public validation stage: it applies exactly the checks `predict` "
                "applies — type, batch size 1..`MAX_BATCH`, image side 1..`MAX_IMAGE_SIDE` px, `top_k` 1..`NUM_CLASSES` — "
                "and returns an **input manifest** naming the schema and ceilings, each input's observed mode and size, "
                "and the verdict. The manifest is written to `outputs/{stem}_input_manifest.json`. To show what rejection "
                "looks like, the cell also validates a deliberately oversized image and records the pipeline's own error "
                "message as a finding. **What the pipeline changes about your image:** it converts to RGB and "
                "squash-resizes to exactly 448 x 448 (`crop_mode: \"squash\"`, `crop_pct: 1.0` in the snapshot config) — "
                "nothing is cropped or dropped, but a non-square image is distorted in proportion to the aspect ratio "
                "printed above. The notebook itself does not resize, crop, or subsample.\n\n"
                "**Predict before running:** the oversized probe is 4097 px wide. Will the manifest's verdict for *your* "
                "image change because of it?"
            ),
            "code": _VALIDATE_CODE,
        },
        {
            "md": (
                "**What to notice:** `verdict: accepted` for the sample, and one finding that names the ceiling the probe "
                "broke (`MAX_IMAGE_SIDE=4096`).\n\n"
                "<details><summary>Check your reasoning</summary>No. The probe is validated separately and its rejection is "
                "recorded as a finding; the sample's own verdict stays `accepted`. A rejected input never reaches the model: "
                "`predict` raises the same error.</details>\n\n"
                "## 6. Classify\n\n"
                "`predict` returns, per image, `predicted_index`/`predicted_label` and a `top_k` list of `{{label, index, "
                "score}}` entries **ordered by descending score** — rank position is the class ordering, and the exported "
                "files preserve it. The decision rule is `argmax` over the 1000 softmax scores (`decision_rule` in the "
                "result); the pipeline ships no acceptance threshold, and `score` is a softmax over uncalibrated logits, "
                "**not a calibrated probability**. A deployment that needs an abstain option must choose its own score "
                "cut-off on its own labelled data — downstream calibration is the caller's responsibility. Inference is "
                "deterministic given the same weights, device and library versions (no sampling, `model.eval()`); CPU and "
                "CUDA kernel choices can reorder near-tied classes. On a CPU runtime this is a large model (107 GMACs per "
                "image): expect seconds per image, not milliseconds.\n\n"
                "**Predict before running:** the gradient shows no object. Will the top-1 score be high (above 0.5) or low? "
                "Will the five classes be related to each other?"
            ),
            "code": _CLASSIFY_CODE,
        },
        {
            "md": (
                "**What to notice:** a low top-1 score spread across unrelated classes.\n\n"
                "<details><summary>Check your reasoning</summary>Low and unrelated. In a local Windows CPU check of this "
                "version the top-1 was `hair spray` with score 0.0179, followed by `maraca`, `velvet`, `screen, CRT screen` "
                "and `pick` at about 0.015 each (the model card's smoke run labelled the gradient `screen, CRT screen` at "
                "0.013; another device can reorder such near-ties). A softmax always sums to 1 over the 1000 classes, so the "
                "model must name *something* even when nothing matches; a low, flat top-5 is how that looks. It is not a "
                "probability that the image shows hair spray.</details>\n\n"
                "## 7. Evaluate → evaluation report\n\n"
                "`evaluation_report` is the pipeline's public evaluation stage and always produces a report. When a "
                "ground-truth class index was supplied in Section 4 it carries `top_k_accuracy` (the repository's metric "
                "helper) at k=1 and k=5 with the verdict `sample-sanity` — a single-image tutorial metric with no dispersion "
                "estimate. On the synthetic default sample no metric exists, so the verdict is `not-measurable` and the "
                "report states what would make the task measurable: labelled photographs with ImageNet-1k class indices, "
                "for example a held-out sample of your own data scored against its majority-class baseline, or the "
                "ImageNet-1k validation set (whose upstream 88.692 % top-1 / 98.722 % top-5 at 448 px is quoted by the "
                "model card, not measured here). The report is written to `outputs/{stem}_evaluation_report.json`.\n\n"
                "**Predict before running:** which verdict will the report give for the gradient, and why?"
            ),
            "code": _EVALUATE_CODE,
        },
        {
            "md": (
                "**What to notice:** `verdict: not-measurable`, an empty `metrics` list, and a `needs` field saying what "
                "labelled data would be required.\n\n"
                "<details><summary>Check your reasoning</summary>`not-measurable`: accuracy compares a prediction with a "
                "true label, and a gradient has none. Supplying a photograph and its index through BYOD switches the "
                "verdict to `sample-sanity` — still one image, so still not a measurement of the model.</details>\n\n"
                "## 8. Fine-tuning data: acquire, validate and split\n\n"
                "From here the notebook adapts the model to **new classes**. That needs labelled images, split into a "
                "**training** set the model learns from and a **held-out** set it never sees during training, so the "
                "evaluation in Section 10 measures something the model could not have memorised.\n\n"
                "- **Default (`USE_BYOD_DATASET = False`):** downloads [`Cleanlab/cifar-10-subset`](https://huggingface.co/datasets/Cleanlab/cifar-10-subset) "
                "(MIT licence, 986,707 bytes) and refuses it unless its SHA-256 matches. It holds 400 CIFAR-10 images of "
                "**32×32 px** in two classes, `frog` and `truck`; each photo is stored twice, as an original and a darkened "
                "copy. The split is **pair-grouped**: images of one class with the same file name (`original_images/frog/"
                "image_7.png` and `darkened_images/frog/image_7.png`) form one group, and a group always lands on one side, "
                "so a held-out photo's darkened twin is never in training (otherwise the model could score it by memory). "
                "With `SPLIT_SEED`, `SUBSET_PER_CLASS = 100` images (50 groups) per class are drawn and 20 % of the groups "
                "(10 per class) are held out: **160 training and 40 held-out images**. The model's preprocessing upscales "
                "every image to 448 px. A random split **assumes the held-out images are independent** of the training "
                "images; the grouping protects that assumption against the twin copies, and the cell also compares every "
                "held-out image with every training image by pixel digest and stops if any is an exact copy. It prints "
                "`held_out_sharing_a_file_name_with_train` and `held_out_pixel_copies_in_train`, both 0 here. If the download fails, the cell says so and falls back to a synthetic two-class stripes dataset "
                "(8 training, 4 held-out images); the results below then describe stripes.\n"
                "- **Bring Your Own Data (`USE_BYOD_DATASET = True`):** upload a `.zip` (or set `BYOD_DATASET_PATH`) of "
                "`<class>/<image>` folders, split here per class with `SPLIT_SEED`, or of `train/<class>/` and "
                "`val/<class>/` folders used as given. `load_image_zip` decodes every image, checks it against "
                "`MAX_IMAGE_SIDE`, and refuses — before any training, naming the file or class and the rule — an image "
                "outside a class folder, a validation class missing from `train/`, a class with fewer than 2 image groups, fewer "
                "than 2 classes, an undecodable or oversized image, or an archive over its limits (see the Prerequisites). "
                "Other files are skipped and listed.\n\n"
                "**Predict before running:** with 20 held-out images per class, by how many percentage points does one "
                "wrong prediction move the held-out accuracy? And how many independent photos are those 40 images?"
            ),
            "code": _DATA_CODE,
        },
        {
            "md": (
                "**What to notice:** the dataset digest, `image_size_px: [32, 32]`, 80 training and 20 held-out images per "
                "class, `held_out_sharing_a_file_name_with_train: 0` and `held_out_pixel_copies_in_train: 0`.\n\n"
                "<details><summary>Check your reasoning</summary>By 2.5 points: 40 held-out images, so each one is "
                "1/40 of the accuracy. But they are only 20 photos, each held out with its darkened copy, and the two copies "
                "are usually classified alike, so the 40 images carry roughly the evidence of 20 independent photos. Keep "
                "that in mind when Section 10 prints an interval computed as if all 40 were independent.</details>\n\n"
                "## 9. Fine-tune\n\n"
                "`pipe.fit(...)` builds the network from the verified snapshot, replaces the 1000-class head with a new "
                "linear classifier sized to the target classes (`len(class_names)`, here 2), and trains with "
                "`torch.optim.AdamW` (Adam with decoupled weight decay) on the **cross-entropy** loss between the softmax "
                "and the true class. Each run starts again from the pretrained snapshot. The form fields are the whole "
                "training configuration:\n\n"
                "- `TRAINABLE`: `'head'` (default) is **head-only fine-tuning** — the pretrained backbone is frozen and "
                "only the new classifier is trained (1,538 parameters for 2 classes); `'all'` is **full fine-tuning** — "
                "every parameter, the backbone included, is updated (about 86.35 M parameters; Section 12).\n"
                "- `EPOCHS` (passes over the training images), `BATCH_SIZE`, `LEARNING_RATE`, `WEIGHT_DECAY` and "
                "`TRAIN_SEED`, which seeds PyTorch (head initialisation, shuffling) and Python's `random` (the training "
                "augmentation), so a CPU run with the same settings repeats exactly; GPU kernels can still differ "
                "slightly.\n\n"
                "The cell prints the trainable and frozen parameter counts **before** training and checks that `fit` "
                "trained exactly that set. The configuration, the counts and the dataset provenance (source, SHA-256, "
                "classes, split) are written into `model-config.json` with the artifact and into the result file. No "
                "external worker, CLI subprocess or unpinned dependency is involved. The per-epoch history reports "
                "training loss, validation loss and validation accuracy on the held-out images.\n\n"
                "**Predict before running:** after one epoch on 160 images, will the validation accuracy be above the 50 % "
                "that always answering one class would get? How many parameters will the default run train?"
            ),
            "code": _FIT_CODE,
        },
        {
            "md": (
                "**What to notice:** `head-only fine-tuning` with `trainable_parameters` 1,538 (768 features × 2 classes + "
                "2 biases) and `frozen_parameters` 86,348,544 (the whole pretrained backbone), then one history row with "
                "the training loss, the validation loss and the validation accuracy.\n\n"
                "<details><summary>Check your reasoning</summary>Very likely, and by a wide margin. In a local Windows CPU "
                "check of this notebook version (torch 2.14.0, defaults, 2026-10-04) one epoch gave training loss 0.38, "
                "validation loss 0.17 and validation accuracy 38/40. The frozen backbone already describes images well "
                "(it was pretrained on ImageNet-22k and fine-tuned for ImageNet-1k), so a linear head on its features only "
                "has to learn which direction separates frogs from trucks — 1,538 numbers that 160 images can pin down. "
                "With `SPLIT_SEED` 0, 1 and 2 the same check scored 40/40, 38/40 and 40/40. A GPU run can differ by an image "
                "or two.</details>\n\n"
                "## 10. Fresh-boundary reload, equivalence check and held-out evaluation\n\n"
                "Loading a file is not proof that it holds the model you evaluated. "
                "`EVA02ClassificationPipeline.from_pretrained` loads `model.safetensors` and `model-config.json` from "
                "`outputs/{stem}_finetuned` as a fresh deployment would (architecture rebuilt, weights restored with "
                "`strict=True`, class names from the config); the cell then predicts every held-out image with **both** the "
                "reloaded and the in-memory model and requires the same labels and a softmax-score difference within "
                "`EQUIVALENCE_TOLERANCE` (`equivalent: True`). It stops with an explanation otherwise. The artifact's "
                "SHA-256 digests are recorded.\n\n"
                "The held-out evaluation is a **tutorial metric on a handful of images, not a benchmark**. "
                "`finetune_evaluation_report` reports the counts, the accuracy with its **95 % Wilson interval**, the "
                "**majority-class baseline** (the accuracy of always answering the most common held-out class) and per-class "
                "counts and misclassified images. Its verdict is always `sample-sanity`; `comparison_to_baseline` is "
                "`above-baseline` only when the whole interval lies above the baseline, `below-baseline` when it lies "
                "below, and otherwise `indistinguishable-from-baseline`. Files: "
                "`outputs/{stem}_validation_predictions.csv` and `outputs/{stem}_finetuned_evaluation_report.json`.\n\n"
                "Because the 40 held-out images are 20 photos in two copies, the interval treats correlated images as "
                "independent and is somewhat **too narrow**; read it as a lower bound on the uncertainty.\n\n"
                "**Predict before running:** if the model gets 26 of the 40 held-out images right (65 %), does the 95 % "
                "interval exclude the 50 % baseline? What about 30 of 40?"
            ),
            "code": _RELOAD_CODE,
        },
        {
            "md": (
                "**What to notice:** `equivalent: True` with a score difference near 0; then `n`, the interval, the "
                "baseline and the comparison word, not just the accuracy, and which images were misclassified.\n\n"
                "<details><summary>Check your reasoning</summary>26/40 = 65 % has a 95 % Wilson interval of about 49.5 % to "
                "78 %: its lower end sits just under the baseline, so `indistinguishable-from-baseline`. 30/40 = 75 % gives "
                "about 60 % to 86 %: `above-baseline`. In the local CPU check of this version the default run scored 38/40 "
                "(interval about 83 % to 99 %, `above-baseline`; the two wrong images are the original and the darkened copy "
                "of one frog photo, both called trucks) and the reload matched exactly (difference 0.0); with `SPLIT_SEED` "
                "0, 1 and 2 it scored 40/40, 38/40 and 40/40. "
                "That is consistent evidence that the adapted head separates these two classes on this data — but on 20 "
                "photos of one archive, not a claim about frogs and trucks in general.</details>\n\n"
                "## 11. Export outputs and provenance\n\n"
                "*Engineering and reproducibility.* Machine-readable JSON preserves the full prediction (argmax decision and "
                "the rank-ordered top-k scores), the evaluation report, the fine-tuning configuration (method, trainable and "
                "frozen parameter counts, epochs, batch size, learning rate, weight decay, seed), the dataset manifest "
                "(source, SHA-256, classes, split sizes, skipped files), the training history, the held-out evaluation with "
                "its interval and the equivalence check, the artifact digests, the sample identity and digest, the "
                "notebook's source (repository, revision, embedded module digest, generator), the model identifier, the "
                "immutable model revision, and the runtime identity (Python, `torch`, `timm`, device). The rank-ordered "
                "top-k table and the held-out predictions are written as CSV files so class ordering survives downstream "
                "use. The fine-tuned artifact is in `outputs/{stem}_finetuned/`. No credentials are recorded."
            ),
            "code": _EXPORT_CODE,
        },
        {
            "md": (
                "**What to notice:** six files in `outputs/` plus the two-file artifact directory, and in "
                "`outputs/{stem}_result.json` a `fine_tuning.config` block you could rerun from.\n\n"
                "## 12. Your turn — change one thing: train every layer\n\n"
                "**Predict → Change one thing → Run → Observe → Explain.**\n\n"
                "1. **Predict:** with every layer trainable (`TRAINABLE = 'all'`, 86,350,082 trainable parameters instead "
                "of 1,538), will one epoch at the same learning rate do better or worse than the head-only run on the 40 "
                "held-out images? Write your guess down.\n"
                "2. **Change one thing:** in Section 9 set `TRAINABLE = 'all'` and nothing else.\n"
                "3. **Run:** select the Section 9 cell and choose **Runtime → Run after** (it re-runs Sections 9–12; the "
                "artifact and result files are overwritten with the full fine-tuning run, which `fine_tuning.config.method` "
                "records). It is slower: 318 s instead of 140 s for the training in the local CPU check; on a GPU, if memory runs out, lower `BATCH_SIZE`.\n"
                "4. **Observe:** this cell prints one row per run in this session — method, trainable parameters, final "
                "validation loss, the held-out count with its interval and the comparison to the baseline.\n"
                "5. **Explain:** why can training more parameters give a *worse* model here, and does the held-out set have "
                "enough images to tell the two methods apart?\n\n"
                "<details><summary>Check your reasoning</summary>Worse, in the local CPU check of this notebook version: "
                "one epoch of full fine-tuning ended with validation loss 0.54 (head-only: 0.17) and the held-out count "
                "fell from 38/40 to **30/40, every error the same mistake: 10 of the 20 frogs called trucks** (truck 20/20, "
                "frog 10/20; interval about 60 % to 86 %, still `above-baseline`). That is the start of a **collapse** "
                "towards one class: with all 86.35 M parameters free, 40 AdamW steps at learning rate 1e-4 on batches of 4 "
                "tiny, upscaled images moved the pretrained backbone away from the features that made the task easy, and "
                "the cheapest way to lower the loss was to lean towards one answer. It can go all the way: in the review of "
                "the previous notebook version, a CPU run of full fine-tuning (one epoch on 26 images) called every held-out "
                "image a truck, and in the ConvNeXt sibling notebook full fine-tuning called all 40 held-out images frogs. "
                "On a GPU your numbers may differ. Can the held-out set tell the methods apart? Only just: the intervals "
                "(about 83–99 % vs 60–86 %) overlap slightly, and they are too narrow because the copies are correlated, so "
                "one run is suggestive, not decisive; the higher validation loss and the one-sided errors point the same "
                "way. More parameters need more data or a gentler schedule: a smaller `LEARNING_RATE` (e.g. 1e-5) is the "
                "natural next experiment for full fine-tuning.</details>"
            ),
            "code": _ACTIVITY_CODE,
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "*Evaluation practice.* The predicted label is the argmax of a softmax over the fixed 1000-class ImageNet-1k label "
        "space (or the custom class names after fine-tuning); the `score` values are uncalibrated softmax outputs, not "
        "probabilities of correctness, and the pipeline ships no threshold. On the synthetic gradient the label is "
        "meaningless by construction and the evaluation report says `not-measurable`; a `top_k_accuracy` value shown for a "
        "single BYOD image is 0 or 1 and says nothing about the error rate on a domain, camera, or class distribution. "
        "Every input is squash-resized to 448 x 448, so strongly non-square subjects are distorted before "
        "classification.\n\n"
        "The **fine-tuning result** is a held-out accuracy on **40 images** — 20 photos (10 frogs, 10 trucks, 32×32 px "
        "upscaled), each with its darkened copy — from one seeded, pair-grouped split and one training run. Read it "
        "together with its 95 % interval and the 50 % majority-class baseline printed in Section 10, remembering that "
        "the interval treats the 40 correlated images as independent and is therefore too narrow. Different split "
        "seeds or devices move the count: a local CPU check of this version scored 38/40 at the default seed and 40/40, "
        "38/40 and 40/40 at split seeds 0, 1 and 2 (head-only), while full fine-tuning (Section 12) fell to 30/40, every "
        "error a frog called a truck. It shows that the fine-tuning, export and reload path runs "
        "and produces a working two-class classifier on this sample; it does **not** show how well fine-tuning works on "
        "frogs and trucks, on CIFAR-10, or on your data. The default run trains only the 1,538-parameter head, so the "
        "frozen ImageNet features decide what the classifier can see; full fine-tuning of all 86.35 M parameters on 160 "
        "images is unstable with these settings, as Section 12 shows. The pipeline does not detect out-of-distribution "
        "inputs (drawings, scans, satellite tiles), blur, or capture-device drift, and it provides no features, "
        "detection, segmentation, multi-label, OCR, or open-vocabulary capability.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline module, carried in this notebook, "
        "can acquire and digest-verify the pinned model, validate the demonstrated input, execute the public pipeline "
        "path, perform in-kernel fine-tuning, reload an artifact that reproduces the in-memory model, and emit the shown "
        "machine-readable outputs in the tested runtime — without the repository being reachable. It does **not** "
        "establish benchmark superiority, deployment calibration, safety for high-consequence decisions, or production "
        "fitness on an unseen domain.\n\n"
        "**Next experiments** (each starts after the default Run all):\n\n"
        "1. **A photograph with a known class:** in Section 4 set `USE_BYOD = True` and `GROUND_TRUTH_INDEX` to its "
        "ImageNet-1k index, select the Section 4 cell and choose **Runtime → Run after**; Section 7 switches to "
        "`sample-sanity` with `top_k_accuracy` at k=1 and k=5. (Run after also repeats the fine-tuning in Sections 9–11.)\n"
        "2. **Another split:** in Section 8 change `SPLIT_SEED`, then **Run after** from Section 8; Section 12 shows how "
        "the held-out count moves.\n"
        "3. **Your own classes:** in Section 8 set `USE_BYOD_DATASET = True` (and upload, or set `BYOD_DATASET_PATH`), "
        "then **Run after** from Section 8.\n"
        "4. **Gentler full fine-tuning:** after Section 12, keep `TRAINABLE = 'all'` and lower `LEARNING_RATE` (e.g. "
        "1e-5) in Section 9, then **Run after** from Section 9; compare its row with the full fine-tuning one.\n"
        "5. **See the squash:** upload a strongly non-square photograph through BYOD (Section 4), then a square crop of "
        "it, and compare the top-5 lists.\n\n"
        "## Troubleshooting\n\n"
        "- **Section 1 stops with \"needs a Linux x86_64 runtime\".** The locked environment is built from manylinux "
        "wheels; use Colab, Kaggle or a Linux Jupyter server.\n"
        "- **Section 1 fails to download `uv`, Python or a package.** The runtime needs `pypi.org`, "
        "`files.pythonhosted.org` and the python-build-standalone release host. Re-run the cell; a size or SHA-256 "
        "mismatch is refused on purpose.\n"
        "- **\"The isolated environment's Python process exited\".** Usually out of memory. Restart the session and choose "
        "**Run all** again; on a small CPU runtime lower `BATCH_SIZE` in Section 9.\n"
        "- **Section 3 reports a size or SHA-256 mismatch.** A snapshot file was altered or truncated; delete "
        "`weights/eva02-base-448/model.safetensors` and run Section 3 again.\n"
        "- **Section 8 prints \"falling back to the deterministic synthetic stripes dataset\".** `huggingface.co` was "
        "unreachable. The run continues on stripes; re-run Section 8 onward once the network is back. A SHA-256 mismatch "
        "of the downloaded archive stops the cell instead.\n"
        "- **Section 8 refuses a BYOD archive.** The message names the file or class and the rule (class folders, "
        "matching `train/` and `val/` classes, at least 2 images per class, decodable images of at most 4096 px, the "
        "archive limits). Fix the archive and run Section 8 again.\n"
        "- **Section 4 refuses a BYOD image** as not decodable: convert it to PNG or JPEG. Outside Colab there is no upload "
        "dialog; use `BYOD_IMAGE_PATH` / `BYOD_DATASET_PATH`.\n"
        "- **Section 10 stops with \"does not reproduce the in-memory model\".** The export or reload is broken; re-run "
        "Section 9. Do not use that artifact.\n"
        "- **CUDA out of memory.** Lower `BATCH_SIZE` (full fine-tuning at 448 px needs the most memory), or switch the "
        "runtime to CPU.\n"
        "- **Sections 6, 9 or 10 are very slow.** Expected on a CPU runtime for this 107-GMAC model (see the "
        "Prerequisites); switch to a GPU runtime, or use the MobileNetV4 sibling notebook for CPU latency.\n\n"
        "## Glossary\n\n"
        "- **Vision transformer / EVA-02:** a vision transformer cuts the image into patches (here 14 x 14 px, so a "
        "448 px image is 32 x 32 patches) and processes them with attention layers. EVA-02 (Fang et al., 2023) is such "
        "a transformer; EVA-02 Base has about 86 M parameters. This checkpoint was pretrained by masked image modelling "
        "on ImageNet-22k and fine-tuned on ImageNet-22k and then ImageNet-1k.\n"
        "- **Logits / softmax:** the network's raw class scores; the softmax turns them into positive numbers summing to "
        "1. A softmax score is a ranking signal, not a calibrated probability of being right.\n"
        "- **Argmax / top-k:** the class with the highest score / the k highest-scoring classes in order.\n"
        "- **Squash resize:** the image is resized to exactly 448 x 448 without keeping its aspect ratio, so nothing is "
        "cropped but a non-square image is stretched.\n"
        "- **Head:** the final linear layer mapping features to class logits. **Backbone:** everything before it.\n"
        "- **Full fine-tuning / head-only fine-tuning:** training every parameter / training only a new head with the "
        "backbone frozen.\n"
        "- **AdamW:** the Adam optimiser with decoupled weight decay. **Cross-entropy:** the loss that penalises a low "
        "softmax score for the true class. **Epoch:** one pass over the training images.\n"
        "- **Held-out (validation) set:** labelled images kept out of training and used only to evaluate.\n"
        "- **Majority-class baseline:** the accuracy of always answering the most common class of the evaluated images.\n"
        "- **95 % Wilson interval:** a range for the true accuracy that is consistent with the observed count at the "
        "95 % level; with few images it is wide.\n"
        "- **Fresh-boundary reload / equivalence:** loading the exported files into a new pipeline and checking that it "
        "predicts what the in-memory model predicted.\n\n"
        "## Conclusion (your notes)\n\n"
        "Fill in from the numbers this run printed; keep each claim to what the evidence shows.\n\n"
        "- **Task:** what was classified, into which classes, from which data?\n"
        "- **Principal result:** the held-out count, `n`, the 95 % interval, and the fine-tuning method.\n"
        "- **Baseline / reference:** the majority-class baseline, and `comparison_to_baseline`.\n"
        "- **Uncertainty or failure mode:** what would change with another split seed, another device, or more images? "
        "Which images were misclassified?\n"
        "- **Limitations:** what does this run *not* show (see Interpretation and limits)?\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/eva02-classification-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/eva02-classification-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/eva02-classification-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/baaivision/EVA\n"
        "- EVA-02 paper: https://arxiv.org/abs/2303.11331\n"
        "- timm documentation: https://huggingface.co/docs/timm\n"
        "- Tutorial dataset: https://huggingface.co/datasets/Cleanlab/cifar-10-subset\n"
        "- Wilson, E. B. (1927), Probable inference, the law of succession, and statistical inference, JASA 22(158)"
    ),
}
