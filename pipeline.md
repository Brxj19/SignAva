# SignAvatar Pipeline and Performance Report

This document explains the complete SignAvatar generative pipeline and summarizes the model performance metrics calculated from the current project artifacts.

The project now contains two model paths:

- **V1 baseline**: direct `gloss_id -> SMPL-X motion` generation in `src/model`.
- **V2 SignVAE-inspired pipeline**: Motion VQ-VAE plus autoregressive motion-token generation in `src/model_v2`.

V2 is inspired by the SignAvatars paper's SignVAE / Sign-VQVAE idea, but it is not an exact official reproduction because official SignVAE training code is not publicly available in the public repository.

## 1. Pipeline Overview

The project converts a gloss text input into a generated SMPL-X motion sequence and then renders that sequence as a video.

V1 pipeline:

```text
Raw WLASL/SignAvatar data
        |
        v
Dataset extraction and verification
        |
        v
Index, splits, vocabulary, normalization stats
        |
        v
SignMotionGenerator training
        |
        v
Checkpointed neural model
        |
        v
Text gloss -> gloss id -> generated SMPL-X motion
        |
        v
SMPL-X mesh rendering
        |
        v
MP4 sign animation
```

V2 pipeline:

```text
Raw WLASL/SignAvatar data
        |
        v
Dataset extraction, verification, index
        |
        v
V2 preparation: resample to 80, filter short clips, canonicalize camera, fix betas
        |
        v
Stage 1: Motion VQ-VAE
SMPL-X motion -> encoder -> vector quantizer -> decoder -> reconstructed SMPL-X motion
        |
        v
Stage 2: Token generator
gloss_id -> autoregressive Transformer -> motion token ids
        |
        v
Stage 3: V2 inference
text/gloss -> token ids -> VQ-VAE decoder -> generated SMPL-X motion
        |
        v
Existing SMPL-X renderer
        |
        v
MP4 sign animation
```

The final demo pipeline uses generated neural motion only. It does not load or replay real dataset `.pkl` files during inference, rendering, or demo video generation.

## 2. Dataset Pipeline

### Dataset Inputs

Required raw files:

```text
data/raw/wlasl_pkls_cropFalse_defult_shape.zip
data/raw/WLASL_v0.3.json
```

Required SMPL-X files:

```text
human_models/smplx/SMPLX_NEUTRAL.npz
human_models/smplx/SMPLX_NEUTRAL.pkl
human_models/smplx/SMPLX_MALE.npz
human_models/smplx/SMPLX_MALE.pkl
human_models/smplx/SMPLX_FEMALE.npz
human_models/smplx/SMPLX_FEMALE.pkl
```

### Preparation Stages

1. `src.data.extract_dataset`
   Extracts annotation `.pkl` files from the raw zip into `data/annotations/`.

2. `src.data.verify_dataset`
   Checks that `.pkl` files contain an `smplx` entry and records shapes in `data/processed/dataset_verification_report.json`.

3. `src.data.build_index`
   Builds a video-id to gloss mapping and saves `data/processed/index.json`.

4. `src.data.prepare_data`
   Builds the training vocabulary, train/validation/test splits, and normalization statistics.

For V2, `src.data.prepare_data` also supports:

```text
seq_len = 80
sequence_mode = resample
min_seq_len = 20
max_seq_len = optional
selection_mode = top_count or all
canonicalize_camera = true
fix_betas = true
```

V2 camera canonicalization sets SMPL-X dimensions `179:182` to zero. Fixed betas replace dimensions `159:169` with a consistent value, currently zeros.

### Current V2 Smoke Dataset Summary

Calculated from the V2 smoke preparation command:

```bash
python3 -m src.data.prepare_data --seq-len 80 --max-glosses 10 --selection-mode top_count --sequence-mode resample --min-seq-len 20 --canonicalize-camera --fix-betas
```

| Item | Value |
|---|---:|
| Total selected samples before filtering | 134 |
| Kept samples | 133 |
| Skipped too short | 1 |
| Skipped too long | 0 |
| Selected gloss count | 10 |
| Vocabulary size | 12 |
| Train samples | 90 |
| Validation samples | 23 |
| Test samples | 20 |
| Motion sequence length | 80 frames |
| Motion dimension | 182 |
| Latent token length | 20 |
| Camera canonicalized | true |
| Betas fixed | true |

V2 smoke gloss vocabulary:

```text
about
accident
africa
again
ago
all
always
approve
argue
arrive
```

V2 summary output:

```text
outputs/v2_data_summary.json
```

### Earlier V1 Dataset Summary

Calculated from:

```text
data/vocab.json
data/splits.json
data/processed/prepare_data_summary.json
data/processed/dataset_verification_report.json
```

| Item | Value |
|---|---:|
| Total `.pkl` files detected | 1000 |
| Selected gloss count | 10 |
| Vocabulary size | 12 |
| Special tokens | `<PAD>`, `<UNK>` |
| Train samples | 53 |
| Validation samples | 10 |
| Test samples | 10 |
| Motion sequence length | 60 frames |
| Motion dimension | 182 |

Current gloss vocabulary:

```text
a
a lot
abdomen
able
about
above
accent
accept
accident
accomplish
```

## 3. Motion Representation

V1 motion sequence shape:

```text
(60, 182)
```

V2 motion sequence shape:

```text
(80, 182)
```

V2 latent token sequence shape for `seq_len=80` and `downsample_factor=4`:

```text
(20,)
```

The 182-D SMPL-X parameter layout used by the renderer is:

| Slice | Meaning | Size |
|---|---|---:|
| `0:3` | global/root orientation | 3 |
| `3:66` | body pose | 63 |
| `66:111` | left hand PCA pose | 45 |
| `111:156` | right hand PCA pose | 45 |
| `156:159` | jaw pose | 3 |
| `159:169` | shape/betas | 10 |
| `169:179` | expression | 10 |
| `179:182` | camera translation from source format | 3 |

For generated rendering, translation is kept stable in the renderer view layer. Orientation fixes are also applied only in the renderer/view layer, not by modifying generated motion arrays.

## 4. V1 Model Architecture

Model:

```text
SignMotionGenerator
```

Current architecture configuration from the full training run:

| Hyperparameter | Value |
|---|---:|
| Vocabulary size | 12 |
| Sequence length | 60 |
| Motion dimension | 182 |
| Transformer dimension | 256 |
| Attention heads | 8 |
| Transformer decoder layers | 4 |
| Dropout | 0.1 |
| Trainable parameters | 4,278,966 |

The model maps a gloss id to a full sequence:

```text
gloss_id -> SignMotionGenerator -> (1, 60, 182)
```

After inference, the batch dimension is removed:

```text
(1, 60, 182) -> (60, 182)
```

## 4b. V2 Model Architecture

V2 lives under:

```text
src/model_v2/
```

### Stage 1: Motion VQ-VAE

Main files:

```text
src/model_v2/vector_quantizer.py
src/model_v2/vqvae.py
src/model_v2/losses.py
src/model_v2/train_vqvae.py
```

Default configuration:

| Hyperparameter | Value |
|---|---:|
| Sequence length | 80 |
| Motion dimension | 182 |
| Latent dimension | 256 |
| Codebook size | 512 |
| Downsample factor | 4 |
| Latent token length | 20 |
| Commitment beta | 0.25 |

Motion VQ-VAE flow:

```text
(batch, 80, 182)
        |
        v
Conv1D temporal encoder
        |
        v
(batch, 20, 256)
        |
        v
VectorQuantizer nearest-code lookup
        |
        v
code_indices: (batch, 20)
quantized: (batch, 20, 256)
        |
        v
ConvTranspose1D temporal decoder
        |
        v
reconstruction: (batch, 80, 182)
```

### Stage 2: Autoregressive Token Generator

Main files:

```text
src/model_v2/token_generator.py
src/model_v2/train_token_generator.py
```

Default configuration:

| Hyperparameter | Value |
|---|---:|
| Token length | 20 |
| Codebook size | 512 |
| Transformer dimension | 256 |
| Attention heads | 8 |
| Layers | 4 |
| Dropout | 0.1 |

Token generator flow:

```text
gloss_id + previous motion tokens
        |
        v
GPT-style causal Transformer
        |
        v
logits: (batch, 20, 512)
```

At inference:

```text
gloss_id -> autoregressive tokens -> VQ-VAE decoder -> (80, 182)
```

### V2 Losses

V2 reconstruction loss uses SMPL-X part weights:

| Slice | Part | Weight |
|---|---|---:|
| `0:3` | root/global | 0.2 |
| `3:66` | body | 1.2 |
| `66:111` | left hand | 4.0 |
| `111:156` | right hand | 4.0 |
| `156:159` | jaw | 0.2 |
| `159:169` | betas | 0.0 |
| `169:179` | expression | 0.1 |
| `179:182` | camera translation | 0.0 |

Total V2 VQ-VAE loss:

```text
recon_loss
+ 0.7 * velocity_loss
+ 0.2 * acceleration_loss
+ vq_loss
+ 0.2 * root_velocity_loss
```

## 5. V1 Training Pipeline

Training module:

```text
src.model.train
```

Training uses:

- `SignAvatarDataset`
- `SignMotionGenerator`
- `SignMotionLoss`
- `AdamW`
- `ReduceLROnPlateau`
- gradient clipping
- CUDA/MPS/CPU auto device selection

Loss components recorded during training:

- total loss
- MSE
- velocity loss
- acceleration loss

The validation metric used for best-model selection is validation loss.

## 5b. V2 Training Pipeline

### Train Motion VQ-VAE

Command:

```bash
python3 scripts/train_v2_vqvae.py
```

Smoke command:

```bash
python3 scripts/train_v2_vqvae.py --epochs 2 --batch-size 8 --save-every 1
```

Saved artifacts:

```text
checkpoints_v2/vqvae_best.pth
checkpoints_v2/vqvae_final.pth
checkpoints_v2/vqvae_epoch_XXX.pth
outputs/v2/logs/vqvae_training_history.json
outputs/v2/logs/vqvae_test_metrics.json
outputs/v2/reconstructions/
```

### Train Token Generator

Command:

```bash
python3 scripts/train_v2_token_generator.py
```

Smoke command:

```bash
python3 scripts/train_v2_token_generator.py --epochs 2 --batch-size 8
```

Saved artifacts:

```text
checkpoints_v2/token_generator_best.pth
checkpoints_v2/token_generator_final.pth
outputs/v2/logs/token_generator_training_history.json
```

Token-generator metrics:

```text
cross_entropy
token_accuracy
top5_token_accuracy
per-gloss token accuracy
```

## 6. Performance Metrics

Metrics below are calculated from the full 50-epoch training log:

```text
outputs/logs (1)/training_history.json
```

### Best Validation Epoch

| Metric | Value |
|---|---:|
| Epoch index | 20 |
| Epoch number | 21 |
| Train loss | 0.749592 |
| Validation loss | 0.722210 |
| Train MSE | 0.699054 |
| Validation MSE | 0.693129 |
| Train velocity loss | 0.070477 |
| Validation velocity loss | 0.043101 |
| Train acceleration loss | 0.152991 |
| Validation acceleration loss | 0.075309 |
| Learning rate | 0.000100 |

### Final Epoch

| Metric | Value |
|---|---:|
| Epoch index | 49 |
| Epoch number | 50 |
| Train loss | 0.697148 |
| Validation loss | 0.740301 |
| Train MSE | 0.649413 |
| Validation MSE | 0.712062 |
| Train velocity loss | 0.066791 |
| Validation velocity loss | 0.042001 |
| Train acceleration loss | 0.143394 |
| Validation acceleration loss | 0.072380 |
| Learning rate | 0.00000625 |

### Improvement From Epoch 1 to Best Validation Epoch

| Metric | Epoch 1 | Best Epoch | Improvement |
|---|---:|---:|---:|
| Train loss | 1.227731 | 0.749592 | 38.94% |
| Validation loss | 0.937931 | 0.722210 | 23.00% |
| Train MSE | 1.136955 | 0.699054 | 38.52% |
| Validation MSE | 0.892146 | 0.693129 | 22.31% |
| Train velocity loss | 0.120917 | 0.070477 | 41.71% |
| Validation velocity loss | 0.064005 | 0.043101 | 32.66% |
| Train acceleration loss | 0.303173 | 0.152991 | 49.54% |
| Validation acceleration loss | 0.137835 | 0.075309 | 45.36% |

### Generalization Gap

| Checkpoint point | Validation loss - Train loss |
|---|---:|
| Best validation epoch | -0.027382 |
| Final epoch | 0.043153 |

Interpretation:

- The model improved substantially during the 50-epoch run.
- Best validation performance occurred at epoch 21.
- Final training loss continued to decrease, while final validation loss was slightly worse than the best epoch. This suggests mild overfitting after the best validation checkpoint.
- The dataset is small, so validation metrics may vary noticeably with split composition.

## 7. Checkpoint Artifact Note

The full 50-epoch artifacts are present with duplicate names:

```text
checkpoints/best_model (1).pth
checkpoints/final_model (1).pth
checkpoints/epoch_049.pth
outputs/logs (1)/training_history.json
```

The currently named `checkpoints/best_model.pth` appears to be a 1-epoch smoke-test checkpoint:

| File | Epoch | Train loss | Validation loss | Config epochs |
|---|---:|---:|---:|---:|
| `checkpoints/best_model.pth` | 0 | 1.174367 | 0.871329 | 1 |
| `checkpoints/best_model (1).pth` | 20 | 0.749592 | 0.722210 | 50 |
| `checkpoints/final_model.pth` | 0 | 1.174367 | 0.871329 | 1 |
| `checkpoints/final_model (1).pth` | 49 | 0.697148 | 0.740301 | 50 |
| `checkpoints/epoch_049.pth` | 49 | 0.697148 | 0.740301 | 50 |

For best full-run inference, use:

```bash
python3 -m src.model.inference --text "about" --checkpoint "checkpoints/best_model (1).pth"
```

Or rename/copy the full-run checkpoint to:

```text
checkpoints/best_model.pth
```

so default inference uses it.

## 8. V1 Inference Pipeline

Inference module:

```text
src.model.inference
```

Steps:

1. Load checkpoint.
2. Read saved model config and vocabulary size.
3. Rebuild `SignMotionGenerator`.
4. Load model weights.
5. Load `data/vocab.json`.
6. Convert text to gloss id.
7. Run neural inference with `torch.no_grad()`.
8. Denormalize generated motion using `data/normalization_stats.json`.
9. Save `.npy` motion and metadata.

Default command:

```bash
python3 -m src.model.inference --text "about"
```

Output:

```text
outputs/generated/generated_smplx.npy
outputs/generated/generated_metadata.json
```

Current generated default motion statistics:

| Metric | Value |
|---|---:|
| Shape | `(60, 182)` |
| dtype | `float32` |
| Minimum value | -3.450571 |
| Maximum value | 20.110413 |
| Mean | 0.084432 |
| Standard deviation | 1.339296 |

## 8b. V2 Inference Pipeline

Inference module:

```text
src.model_v2.inference_v2
```

Steps:

1. Load `data/vocab.json`.
2. Convert text/gloss to `gloss_id`.
3. Load `checkpoints_v2/token_generator_best.pth`.
4. Load `checkpoints_v2/vqvae_best.pth`.
5. Autoregressively generate 20 motion token ids.
6. Decode token ids through the VQ-VAE decoder.
7. Denormalize with `data/normalization_stats.json`.
8. Save V2 `.npy` motion and metadata.

Notebook command:

```python
from src.model_v2.inference_v2 import generate_motion_v2

motion, metadata = generate_motion_v2("about")
print(motion.shape)
print(metadata)
```

Output:

```text
outputs/v2/generated/about_v2_smplx.npy
outputs/v2/generated/about_v2_metadata.json
```

Expected generated V2 motion:

| Metric | Value |
|---|---:|
| Shape | `(80, 182)` |
| Latent tokens generated | 20 |
| Codebook size | 512 |

## 9. Rendering Pipeline

Renderer module:

```text
src.renderer.smplx_renderer
```

Steps:

1. Load generated `.npy` motion.
2. Parse each `(182,)` frame into SMPL-X parameters.
3. Run the SMPL-X body model.
4. Extract mesh vertices and faces.
5. Apply renderer-only view correction.
6. Render a solid mesh.
7. Save MP4 video.

Default corrected orientation flags:

```text
--force-flip-vertical --view-yaw 0 --view-pitch 0 --view-roll 0
```

Render command:

```bash
python3 -m src.renderer.smplx_renderer --force-flip-vertical --view-yaw 0
```

Output:

```text
outputs/videos/generated_animation.mp4
```

Current video validation:

| Video group | Count | Frames | FPS | Resolution |
|---|---:|---:|---:|---|
| Demo gloss videos | 10 | 60 each | 20.0 | 512x512 |

Generated demo videos currently present:

```text
outputs/videos/a_animation.mp4
outputs/videos/a_lot_animation.mp4
outputs/videos/abdomen_animation.mp4
outputs/videos/able_animation.mp4
outputs/videos/about_animation.mp4
outputs/videos/above_animation.mp4
outputs/videos/accent_animation.mp4
outputs/videos/accept_animation.mp4
outputs/videos/accident_animation.mp4
outputs/videos/accomplish_animation.mp4
```

The current demo index reports:

| Status | Count |
|---|---:|
| done | 10 |

## 10. V1 Demo Generation Pipeline

Demo module:

```text
src.demo.generate_demo_videos
```

For each selected gloss:

1. Generate motion with the trained model.
2. Save gloss-specific `.npy`.
3. Save gloss-specific metadata.
4. Render gloss-specific MP4.
5. Write an entry to `outputs/demo/demo_index.json`.

Command for one gloss:

```bash
python3 -m src.demo.generate_demo_videos --gloss about --force-flip-vertical --view-yaw 0
```

Command for all glosses:

```bash
python3 -m src.demo.generate_demo_videos --force-flip-vertical --view-yaw 0
```

Outputs:

```text
outputs/generated/{safe_gloss}_smplx.npy
outputs/generated/{safe_gloss}_metadata.json
outputs/videos/{safe_gloss}_animation.mp4
outputs/demo/demo_index.json
```

## 10b. V2 Demo Generation Pipeline

Demo script:

```text
scripts/generate_v2_demo.py
```

For each selected gloss:

1. Generate motion token ids with the V2 token generator.
2. Decode motion with the VQ-VAE decoder.
3. Save gloss-specific V2 `.npy`.
4. Save gloss-specific V2 metadata.
5. Optionally render with the existing SMPL-X renderer.
6. Write `outputs/v2/demo_index.json`.

Command for one gloss without rendering:

```bash
python3 scripts/generate_v2_demo.py --gloss about --no-render
```

Command for one gloss with rendering:

```bash
python3 scripts/generate_v2_demo.py --gloss about
```

Outputs:

```text
outputs/v2/generated/{safe_gloss}_v2_smplx.npy
outputs/v2/generated/{safe_gloss}_v2_metadata.json
outputs/v2/videos/{safe_gloss}_v2_animation.mp4
outputs/v2/demo_index.json
```

## 11. How Metrics Were Calculated

Training metrics were calculated from:

```text
outputs/logs (1)/training_history.json
```

Best epoch:

```python
best = min(history, key=lambda row: row["val_loss"])
```

Improvement percentage:

```python
improvement = (epoch_1_metric - best_epoch_metric) / epoch_1_metric * 100
```

Generalization gap:

```python
gap = val_loss - train_loss
```

Video metrics were checked with OpenCV:

```python
cap = cv2.VideoCapture(video_path)
frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
fps = cap.get(cv2.CAP_PROP_FPS)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
```

## 12. Limitations

- The project is currently trained on 10 glosses, which is a small vocabulary.
- The model is evaluated primarily with reconstruction-style motion losses, not human perceptual quality metrics.
- No separate quantitative test-set evaluation report has been generated yet.
- The validation set has only 10 samples, so validation metrics should be interpreted carefully.
- Rendering quality depends on SMPL-X model availability and headless OpenGL/EGL behavior.
- If the default checkpoint path points to a smoke-test checkpoint, inference quality may be worse than the full 50-epoch model.
- V2 is paper-inspired, not an official SignVAE reproduction.
- The current V2 checkpoints from smoke testing are only 2-epoch sanity-check artifacts and are not quality models.
- V2 token generation quality depends strongly on VQ-VAE reconstruction quality and codebook usage.

## 13. Recommended Next Evaluation Step

Add a test-set evaluation script that:

1. Loads `checkpoints/best_model (1).pth`.
2. Runs inference for each test sample gloss.
3. Compares generated motion against normalized/denormalized test motion.
4. Reports test MSE, velocity loss, acceleration loss, and total motion loss.
5. Saves a machine-readable report such as:

```text
outputs/evaluation/test_metrics.json
```

This would make the performance report more complete than train/validation history alone.

For V2, the next useful evaluation step is:

```bash
python3 scripts/evaluate_v2.py --split test
```

This evaluates VQ-VAE reconstruction and token-generator accuracy, saving reports under:

```text
outputs/v2/evaluation/
```
