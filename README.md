# SignAvatar Generative Project

This project has two model paths:

- **V1 baseline** in `src/model`: directly maps `gloss_id -> SMPL-X motion`.
- **V2 SignVAE-inspired pipeline** in `src/model_v2`: trains a Motion VQ-VAE, then trains an autoregressive gloss-to-motion-token generator.

V2 is inspired by the SignAvatars paper's SignVAE / Sign-VQVAE idea. It is not an exact official reproduction because official SignVAE training code is not publicly available in the public repository.

## Milestone 1: Dataset Setup and Verification

This repository contains utilities to prepare and validate the dataset before any modeling or training work begins.

### Directory layout

Place files as follows:

- `data/raw/`:
  - put the dataset zip file here (e.g. `wlasl_pkls_cropFalse_defult_shape.zip`)
  - put `WLASL_v0.3.json` here
- `human_models/smplx/`:
  - put the SMPL-X model files here, such as `SMPLX_NEUTRAL.npz`, `SMPLX_MALE.npz`, `SMPLX_FEMALE.npz`

The project will also create these directories automatically when importing configuration:

- `data/annotations/`
- `data/processed/`
- `outputs/`
- `checkpoints/`

## How to run

Run from the project root.

### 1. Extract dataset

```bash
python3 -m src.data.extract_dataset
```

This will:
- automatically locate the first `.zip` file in `data/raw/`
- extract contents into `data/annotations/`
- skip extraction if `.pkl` files already exist in `data/annotations/`

### 2. Verify dataset

```bash
python3 -m src.data.verify_dataset
```

This will:
- scan `data/annotations/` recursively for `.pkl` files
- open a few `.pkl` files safely
- inspect available keys and attempt to read the `smplx` key
- save a JSON verification report to `data/processed/dataset_verification_report.json`

### 3. Build index

```bash
python3 -m src.data.build_index
```

This will:
- read `data/raw/WLASL_v0.3.json` if present
- build mapping from `video_id` to `gloss`
- map `.pkl` files by filename stem to `video_id`
- save the index to `data/processed/index.json`

## Milestone 2: Dataset Preparation

### 4. Prepare data splits and normalization

```bash
python3 -m src.data.prepare_data
```

This will:
- read `data/processed/index.json`
- build `data/vocab.json` with special tokens `<PAD>` and `<UNK>`
- build `data/splits.json` with train/val/test partitions and path-to-gloss mapping
- compute global normalization stats on the training split
- save `data/normalization_stats.json`
- save a summary to `data/processed/prepare_data_summary.json`

For the improved 10-gloss experiment, select the 10 most frequent glosses and resample all clips to 60 frames:

```bash
python3 -m src.data.prepare_data --seq-len 60 --max-glosses 10 --selection-mode top_count --sequence-mode resample
```

Notebook equivalent:

```python
from src.data.prepare_data import prepare_data

summary = prepare_data(
    seq_len=60,
    max_glosses=10,
    selection_mode="top_count",
    sequence_mode="resample",
)
```

For V2, prepare 80-frame sequences, filter very short clips, canonicalize camera translation, and fix body-shape betas:

```bash
python3 -m src.data.prepare_data --seq-len 80 --max-glosses 10 --selection-mode top_count --sequence-mode resample --min-seq-len 20 --canonicalize-camera --fix-betas
```

This additionally saves:

```text
outputs/v2_data_summary.json
```

### 5. Load dataset with PyTorch

```python
from src.data.dataset import SignAvatarDataset

train_ds = SignAvatarDataset(split="train")
sample = train_ds[0]
print(sample["gloss"])
print(sample["gloss_id"])
print(sample["motion"].shape)
```

## Milestone 3: Model Architecture and Loss

### 6. Test model architecture

```bash
python3 -m src.model.architecture
```

This will:
- create a `SignMotionGenerator` model with `vocab_size=12`
- run a forward pass with dummy input
- verify output shape `(4, 60, 182)`
- compute loss with dummy target
- print loss components

## Milestone 4: Training Loop

### 7. Train the model

```bash
python3 -m src.model.train
```

This will:
- automatically detect GPU/MPS/CPU
- load train/val datasets
- create `SignMotionGenerator` with config from vocab
- train for configurable epochs
- save checkpoints:
  - `checkpoints/best_model.pth` (best validation loss)
  - `checkpoints/epoch_XX.pth` (periodic saves)
  - `checkpoints/final_model.pth` (final model)
- save full training history to `outputs/logs/training_history.json`
- use AdamW optimizer with gradient clipping
- use ReduceLROnPlateau scheduler
- print progress with tqdm

### 8. Train with custom parameters

```python
from src.model.train import train_model

history = train_model(
    epochs=20,
    batch_size=8,
    learning_rate=1e-4,
    save_every=5
)
```

### 9. Resume training from checkpoint

```python
from src.model.train import train_model

history = train_model(
    epochs=50,
    resume_from="checkpoints/best_model.pth"
)
```

## Milestone 5: Inference from Gloss Text

Milestone 5 loads the trained neural model and generates SMPL-X motion directly from a gloss. It does not load or replay `.pkl` files, and it does not include rendering or Gradio yet.

### 10. Generate motion from the command line

```bash
python3 -m src.model.inference --text "about"
```

This will:
- load `checkpoints/best_model.pth`
- load `data/vocab.json`
- map the input text to a gloss ID
- run neural model inference
- denormalize with `data/normalization_stats.json`
- save generated motion to `outputs/generated/generated_smplx.npy`
- save metadata to `outputs/generated/generated_metadata.json`

To see supported glosses:

```bash
python3 -m src.model.inference --list-glosses
```

### 11. Use inference from a notebook

```python
from src.model.inference import generate_motion, list_available_glosses

print(list_available_glosses())

motion, metadata = generate_motion("about")

print(motion.shape)
print(metadata)
```

Expected motion shape:

```text
(60, 182)
```

Optional custom checkpoint:

```python
motion, metadata = generate_motion(
    "about",
    checkpoint_path="checkpoints/final_model.pth"
)
```

## Milestone 6: Render Generated Motion

Milestone 6 renders the generated SMPL-X `.npy` sequence into an MP4 animation. The default renderer uses `outputs/generated/generated_smplx.npy` from Milestone 5 and does not load or replay a dataset `.pkl` file.

### 12. Render the generated MP4

```bash
python3 -m src.renderer.smplx_renderer
```

Expected input:

```text
outputs/generated/generated_smplx.npy
```

Expected output:

```text
outputs/videos/generated_animation.mp4
```

### 13. Test only the first frame

Before rendering the full animation, test SMPL-X loading and headless rendering with:

```bash
python3 -m src.renderer.smplx_renderer --first-frame
```

Expected output:

```text
outputs/videos/generated_first_frame.png
```

### 14. Test renderer with a neutral zero pose

To verify the renderer independently from generated motion:

```bash
python3 -m src.renderer.smplx_renderer --zero-pose
```

Expected output:

```text
outputs/videos/zero_pose_test.png
```

### 15. Use rendering from a notebook

```python
from src.renderer.smplx_renderer import render_generated_motion, render_first_frame, render_zero_pose

zero_path = render_zero_pose()
print(zero_path)

frame_path = render_first_frame()
print(frame_path)

video_path = render_generated_motion()
print(video_path)
```

Optional short render:

```bash
python3 -m src.renderer.smplx_renderer --max-frames 10
```

Orientation debugging:

```bash
python3 -m src.renderer.smplx_renderer --orientation-grid
python3 -m src.renderer.smplx_renderer --zero-pose-grid
```

These save:

```text
outputs/videos/orientation_grid.png
outputs/videos/zero_pose_orientation_grid.png
```

Default rendering applies a view-layer correction only:

```text
--force-flip-vertical
--view-yaw 0
--view-pitch 0
--view-roll 0
```

## Milestone 7: Demo Video Generation

Milestone 7 generates motion and renders demo videos for one or more available glosses. It uses model inference only and does not load or replay dataset `.pkl` files.

### 16. Generate one gloss

```bash
python3 -m src.demo.generate_demo_videos --gloss about
```

### 17. Generate the first 5 glosses

```bash
python3 -m src.demo.generate_demo_videos --limit 5
```

### 18. Generate all glosses

```bash
python3 -m src.demo.generate_demo_videos
```

For faster test renders:

```bash
python3 -m src.demo.generate_demo_videos --limit 3 --max-frames 20
```

To avoid rerendering videos that already exist:

```bash
python3 -m src.demo.generate_demo_videos --skip-existing
```

Outputs are saved as:

```text
outputs/generated/{safe_gloss}_smplx.npy
outputs/generated/{safe_gloss}_metadata.json
outputs/videos/{safe_gloss}_animation.mp4
outputs/demo/demo_index.json
```

Example:

```text
a lot -> outputs/videos/a_lot_animation.mp4
```

Notebook usage:

```python
from src.demo.generate_demo_videos import generate_demo_videos

results = generate_demo_videos(limit=5)
print(results)
```

Display one MP4 in Colab:

```python
from IPython.display import Video

Video("outputs/videos/about_animation.mp4", embed=True, width=512, height=512)
```

## Notes

- Milestone 5 only covers inference and `.npy` motion generation.
- Milestone 6 only covers rendering generated `.npy` motion to image/video.
- Milestone 7 only covers batch demo video generation from model inference.
- Gradio is intentionally not included yet.
- All paths are resolved relative to the project root using `pathlib`.

## Troubleshooting

- **Positional encoding device mismatch**: The positional encoding is registered as a buffer using `register_buffer()`, so it automatically moves with the model to CPU/GPU.
- **ReduceLROnPlateau verbose argument**: The `verbose` argument was removed for compatibility with current PyTorch versions in Colab.
- **Checkpoint missing**: Confirm `checkpoints/best_model.pth` exists, or pass `checkpoint_path` / `--checkpoint`.
- **Unknown gloss**: Unknown text falls back to `<UNK>`. Run `list_available_glosses()` or `python3 -m src.model.inference --list-glosses` to see available glosses.
- **SMPL-X model missing**: Confirm `SMPLX_NEUTRAL.npz` or `SMPLX_NEUTRAL.pkl` exists under `human_models/smplx/`.
- **Avatar upside down**: The renderer applies `--force-flip-vertical` by default as a view-layer correction. To inspect the unflipped render, pass `--no-force-flip-vertical`.
- **Avatar back side visible**: The renderer applies `--view-yaw 0` by default after comparing the generated-motion orientation grid. Use `--orientation-grid` or `--zero-pose-grid` to compare yaw and flip settings visually.
- **Avatar appears as dotted points**: The final renderer uses SMPL-X faces with pyrender/trimesh, or a face-based OpenCV triangle rasterizer if pyrender fails. Point-cloud rendering is debug-only and is not used in the default generated render path.
- **Colab EGL/OpenGL errors**: The renderer sets `PYOPENGL_PLATFORM=egl` before importing pyrender. In Colab, install EGL packages if full mesh rendering fails:

```bash
apt-get update && apt-get install -y libegl1-mesa-dev libgles2-mesa-dev
```

If EGL still fails, the module falls back to an OpenCV solid triangle render that still uses SMPL-X mesh faces.

## Model Improvement Experiment: Weighted Loss and Resampling

This experiment improves sign specificity for the current 10-gloss setup by emphasizing hands and using temporal resampling instead of zero padding.

### Why weighted loss?

The original loss treats all 182 SMPL-X dimensions equally. For sign language generation, hand motion matters more than shape, expression, or source camera translation. `WeightedSignMotionLoss` increases left/right hand weights and ignores betas/translation during loss computation.

SMPL-X part weights:

```text
root/global: 0.5
body: 1.0
left_hand: 4.0
right_hand: 4.0
jaw: 0.2
betas: 0.0
expression: 0.1
translation: 0.0
```

Temporal weights:

```text
velocity_weight = 0.7
acceleration_weight = 0.2
```

### Why resampling?

Padding short clips with zeros can teach the model artificial stillness or collapse near the end of a sequence. `sequence_mode="resample"` linearly interpolates each motion sequence to exactly 60 frames, preserving the whole sign over the model's fixed output length.

### Backup old checkpoints safely

Before training the improved model:

```bash
python3 -m src.utils.backup_checkpoints
```

This creates:

```text
backups/checkpoints_before_weighted_loss_<timestamp>/
```

Then it cleans `checkpoints/` so new improved checkpoints are stored cleanly. It preserves `.gitkeep` if present.

### Train improved model on 10 glosses

Small smoke test:

```python
from src.model.train import train_model

history = train_model(
    epochs=2,
    batch_size=4,
    learning_rate=3e-5,
    loss_type="weighted",
    sequence_mode="resample",
    save_every=1
)
```

Full improved run:

```python
from src.model.train import train_model

history = train_model(
    epochs=300,
    batch_size=8,
    learning_rate=3e-5,
    dropout=0.05,
    loss_type="weighted",
    sequence_mode="resample",
    resume_from=None,
    save_every=25
)
```

If you are rebuilding the 10-gloss training set first, run:

```python
from src.data.prepare_data import prepare_data

summary = prepare_data(
    seq_len=60,
    max_glosses=10,
    selection_mode="top_count",
    sequence_mode="resample",
)
```

### Evaluate per-part metrics

```bash
python3 -m src.evaluation.evaluate_model --split test --checkpoint checkpoints/best_model.pth
```

Evaluation output:

```text
outputs/evaluation/evaluation_metrics.json
```

The evaluation report includes total weighted loss plus per-part MSE for root, body, hands, jaw, betas, expression, and translation.

## Version 2: SignVAE-Inspired Pipeline

V2 is inspired by the SignAvatars paper's SignVAE / Sign-VQVAE direction, but it is not an exact official reproduction. The public SignAvatars repository does not currently include official SignVAE training code, so this project implements a paper-inspired VQ-VAE plus autoregressive token generation pipeline.

V2 keeps the original V1 baseline in `src/model` untouched. New code lives under `src/model_v2`.

V2 flow:

```text
Stage 1:
SMPL-X motion -> Motion VQ-VAE encoder -> vector quantizer -> decoder -> reconstructed SMPL-X motion

Stage 2:
gloss_id -> causal Transformer token generator -> motion token ids

Stage 3:
text/gloss -> token generator -> VQ-VAE decoder -> generated SMPL-X motion -> renderer
```

Default V2 shapes:

```text
motion: (80, 182)
latent code indices: (20,)
codebook size: 512
latent dim: 256
```

### Prepare V2 data

```bash
python -m src.data.prepare_data --max-glosses 10 --selection-mode top_count --sequence-mode resample --seq-len 80 --min-seq-len 20 --canonicalize-camera --fix-betas
```

This writes V2 summary statistics to:

```text
outputs/v2_data_summary.json
```

### Train Motion VQ-VAE

```bash
python scripts/train_v2_vqvae.py
```

Smoke test:

```bash
python3 scripts/train_v2_vqvae.py --epochs 2 --batch-size 8 --save-every 1
```

The VQ-VAE learns:

```text
SMPL-X motion sequence -> discrete motion tokens -> reconstructed SMPL-X motion
```

Checkpoints are saved in `checkpoints_v2/`, and reconstruction samples are saved in:

```text
outputs/v2/reconstructions/
```

Expected VQ-VAE artifacts:

```text
checkpoints_v2/vqvae_best.pth
checkpoints_v2/vqvae_final.pth
outputs/v2/logs/vqvae_training_history.json
outputs/v2/logs/vqvae_test_metrics.json
```

### Check Reconstruction Videos

Reconstruction `.npy` files are compatible with the existing renderer. V2 generated motion keeps shape `(80, 182)`.

### VQ-VAE Debugging Workflow

Before training the token generator seriously, verify that the tokenizer is using the codebook and reconstructing real motion well.

Step A: reconstruction-only autoencoder test:

```bash
python scripts/train_v2_vqvae.py \
  --epochs 100 \
  --batch-size 16 \
  --learning-rate 2e-4 \
  --seq-len 80 \
  --disable-quantization
```

Step B: EMA VQ-VAE with a smaller codebook:

```bash
python scripts/train_v2_vqvae.py \
  --epochs 500 \
  --batch-size 16 \
  --learning-rate 2e-4 \
  --seq-len 80 \
  --codebook-size 128 \
  --latent-dim 256 \
  --quantizer-type ema \
  --vq-loss-weight 0.25 \
  --vq-warmup-epochs 50
```

Step C: evaluate code usage and reconstruction:

```bash
python scripts/evaluate_v2.py --mode vqvae --split train
python scripts/evaluate_v2.py --mode vqvae --split test
```

Evaluation writes code usage histograms to:

```text
outputs/v2/evaluation/vqvae_code_usage_train.json
outputs/v2/evaluation/vqvae_code_usage_test.json
```

Step D: render original vs reconstructed comparisons:

```bash
python scripts/render_v2_reconstructions.py --split test --num-samples 10
```

Useful tokenizer diagnostics:

```text
perplexity
unique_codes
active_code_pct
dead_code_count
code_usage_entropy
top_10_codes
vq_loss
recon_loss
left_hand_mse
right_hand_mse
```

### Train Token Generator

```bash
python scripts/train_v2_token_generator.py
```

Smoke test:

```bash
python3 scripts/train_v2_token_generator.py --epochs 2 --batch-size 8
```

This freezes the trained VQ-VAE and trains:

```text
gloss_id -> autoregressive Transformer -> motion token sequence
```

Expected token-generator artifacts:

```text
checkpoints_v2/token_generator_best.pth
checkpoints_v2/token_generator_final.pth
outputs/v2/logs/token_generator_training_history.json
```

### Generate One V2 Motion

```python
from src.model_v2.inference_v2 import generate_motion_v2

motion, metadata = generate_motion_v2("about")
print(motion.shape)
print(metadata)
```

Expected output shape:

```text
(80, 182)
```

### Generate V2 Demo

```bash
python scripts/generate_v2_demo.py
```

Generate one gloss without rendering:

```bash
python3 scripts/generate_v2_demo.py --gloss about --no-render
```

Generated motions are saved in `outputs/v2/generated/`, and videos are saved in:

```text
outputs/v2/videos/
```

### Evaluate V2

```bash
python3 scripts/evaluate_v2.py --split test
```

Outputs are saved in:

```text
outputs/v2/evaluation/
```

### V2 Smoke-Test Checklist

```bash
python3 -m src.data.prepare_data --seq-len 80 --max-glosses 10 --selection-mode top_count --sequence-mode resample --min-seq-len 20 --canonicalize-camera --fix-betas
python3 scripts/train_v2_vqvae.py --epochs 2 --batch-size 8 --save-every 1
python3 scripts/train_v2_token_generator.py --epochs 2 --batch-size 8
python3 scripts/generate_v2_demo.py --gloss about --no-render
```

The smoke test should produce:

```text
checkpoints_v2/vqvae_best.pth
checkpoints_v2/token_generator_best.pth
outputs/v2/generated/about_v2_smplx.npy
```
