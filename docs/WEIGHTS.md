# Weight provenance and DIMER hosting

- Upstream: `timm/eva02_base_patch14_448.mim_in22k_ft_in22k_in1k`
- Immutable revision: `81063ecfe9c381a16a19d06f396d6c7011aa426a`
- Weight format: SafeTensors, float32 (`model.safetensors`, 348492484 bytes); upstream notes the original EVA-02 checkpoints are float16/bfloat16 and were converted to float32 for `timm`.
- Upstream weight license: MIT (`license: mit` in the snapshot README front matter and `pretrained_cfg.license` in `config.json`)
- Local snapshot: `weights/eva02-base-448/` with `dimer-base-manifest.json` (per-file bytes + SHA-256, `totalBytes` 348498591); the Git repository does not vendor the checkpoint.
- Load-time check: `verify_snapshot()` in `src/eva02_classification_pipeline/pipeline.py` re-hashes every manifest entry and refuses on any mismatch.
- DIMER hosting: MIT permits use, modification, distribution and commercial use provided the copyright and permission notice are preserved; DIMER may mirror the pinned checkpoint in its model store with that notice. Original weights: https://github.com/baaivision/EVA and https://huggingface.co/Yuxin-CV/EVA-02.
- Loader trust boundary: `timm==1.0.29` built-in `eva02_base_patch14_448` architecture; weights loaded from a file path via `pretrained_cfg_overlay`; no remote code is executed. Hub download is opt-in and pinned to the revision above.
