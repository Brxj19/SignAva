# SignAvatar Project Runbook

This runbook lists the commands to run the project from dataset preparation to inference, rendering, and demo video generation.

Run commands from the project root:

```bash
cd /path/to/SignAvatar_Generative_Project
```

In Colab, after mounting Drive:

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
cd /content/drive/MyDrive/FYP/SignAvatar_Generative_Project
```

## 1. Environment Setup

### Local

```bash
python3 -m pip install -r requirements.txt
```

### Colab

Install Python requirements:

```bash
pip install -r requirements.txt
```

Install EGL/OpenGL packages for headless rendering:

```bash
apt-get update && apt-get install -y libegl1-mesa-dev libgles2-mesa-dev
```

Optional, if pyrender/pyglet has issues in Colab:

```bash
pip install pyglet==2.0.7
```

## 2. Required Files

Dataset files:

```text
data/raw/wlasl_pkls_cropFalse_defult_shape.zip
data/raw/WLASL_v0.3.json
```

SMPL-X model files:

```text
human_models/smplx/SMPLX_NEUTRAL.npz
human_models/smplx/SMPLX_NEUTRAL.pkl
human_models/smplx/SMPLX_MALE.npz
human_models/smplx/SMPLX_MALE.pkl
human_models/smplx/SMPLX_FEMALE.npz
human_models/smplx/SMPLX_FEMALE.pkl
```

If Google Drive times out while reading SMPL-X files, mark `human_models/smplx/` as available offline or copy those files to local Colab storage.

## 3. Data Preparation

Extract dataset:

```bash
python3 -m src.data.extract_dataset
```

Verify dataset:

```bash
python3 -m src.data.verify_dataset
```

Build index:

```bash
python3 -m src.data.build_index
```

Prepare splits, vocab, and normalization stats:

```bash
python3 -m src.data.prepare_data
```

For the improved 10-gloss weighted-loss experiment, use the top 10 glosses by sample count and resample every sequence to 60 frames:

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
summary
```

Expected outputs:

```text
data/processed/index.json
data/processed/dataset_verification_report.json
data/vocab.json
data/splits.json
data/normalization_stats.json
```

## 4. Model Architecture Check

```bash
python3 -m src.model.architecture
```

Expected model output shape:

```text
(batch, 60, 182)
```

## 5. Training

Train:

```bash
python3 -m src.model.train
```

Expected outputs:

```text
checkpoints/best_model.pth
checkpoints/final_model.pth
outputs/logs/training_history.json
```

Resume training from checkpoint:

```python
from src.model.train import train_model

history = train_model(
    epochs=50,
    resume_from="checkpoints/best_model.pth"
)
```

## 6. Inference: Generate SMPL-X Motion From Text

List available glosses:

```bash
python3 -m src.model.inference --list-glosses
```

Generate motion for one gloss:

```bash
python3 -m src.model.inference --text "about"
```

Expected outputs:

```text
outputs/generated/generated_smplx.npy
outputs/generated/generated_metadata.json
```

Expected motion shape:

```text
(60, 182)
```

Notebook usage:

```python
from src.model.inference import generate_motion, list_available_glosses

print(list_available_glosses())
motion, metadata = generate_motion("about")
print(motion.shape)
print(metadata)
```

## 7. Render Generated Motion

Render the latest generated motion:

```bash
python3 -m src.renderer.smplx_renderer --force-flip-vertical --view-yaw 0 --view-pitch 0 --view-roll 0
```

Expected input:

```text
outputs/generated/generated_smplx.npy
```

Expected output:

```text
outputs/videos/generated_animation.mp4
```

Render only the first frame:

```bash
python3 -m src.renderer.smplx_renderer --first-frame --force-flip-vertical --view-yaw 0
```

Expected output:

```text
outputs/videos/generated_first_frame.png
```

Render a neutral zero pose test:

```bash
python3 -m src.renderer.smplx_renderer --zero-pose --force-flip-vertical --view-yaw 0
```

Expected output:

```text
outputs/videos/zero_pose_test.png
```

Short render for testing:

```bash
python3 -m src.renderer.smplx_renderer --max-frames 20 --force-flip-vertical --view-yaw 0
```

## 8. Orientation Debugging

Generate orientation grids:

```bash
python3 -m src.renderer.smplx_renderer --orientation-grid
python3 -m src.renderer.smplx_renderer --zero-pose-grid
```

Expected outputs:

```text
outputs/videos/orientation_grid.png
outputs/videos/zero_pose_orientation_grid.png
```

Default corrected orientation flags:

```text
--force-flip-vertical --view-yaw 0 --view-pitch 0 --view-roll 0
```

If the avatar is still showing the back side in your environment, compare `orientation_grid.png` and rerun with the yaw that looks correct:

```bash
python3 -m src.renderer.smplx_renderer --first-frame --force-flip-vertical --view-yaw 180
python3 -m src.renderer.smplx_renderer --force-flip-vertical --view-yaw 180
```

If the avatar is upside down, make sure you are not disabling the flip:

```bash
python3 -m src.renderer.smplx_renderer --force-flip-vertical --view-yaw 0
```

Do not use:

```bash
--no-force-flip-vertical
```

unless you are debugging orientation.

## 9. Demo Video Generation

Generate one gloss:

```bash
python3 -m src.demo.generate_demo_videos --gloss about --force-flip-vertical --view-yaw 0
```

Generate selected glosses:

```bash
python3 -m src.demo.generate_demo_videos --gloss about --gloss accident --force-flip-vertical --view-yaw 0
```

Generate first 5 glosses:

```bash
python3 -m src.demo.generate_demo_videos --limit 5 --force-flip-vertical --view-yaw 0
```

Generate all glosses:

```bash
python3 -m src.demo.generate_demo_videos --force-flip-vertical --view-yaw 0
```

Fast test render:

```bash
python3 -m src.demo.generate_demo_videos --limit 3 --max-frames 20 --force-flip-vertical --view-yaw 0
```

Skip videos that already exist:

```bash
python3 -m src.demo.generate_demo_videos --skip-existing --force-flip-vertical --view-yaw 0
```

Important: do not use `--skip-existing` when fixing old upside-down or back-facing videos. Re-run without it so the old MP4s are overwritten.

Demo outputs:

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

results = generate_demo_videos(
    glosses=["about"],
    force_flip_vertical=True,
    view_yaw_degrees=0,
)
print(results)
```

## 10. Display Videos in Colab

Display one generated demo video:

```python
from IPython.display import Video

Video(
    "outputs/videos/about_animation.mp4",
    embed=True,
    width=512,
    height=512,
)
```

Display the default renderer output:

```python
from IPython.display import Video

Video(
    "outputs/videos/generated_animation.mp4",
    embed=True,
    width=512,
    height=512,
)
```

## 11. Common Full Workflows

### Full Local Workflow

```bash
python3 -m pip install -r requirements.txt
python3 -m src.data.extract_dataset
python3 -m src.data.verify_dataset
python3 -m src.data.build_index
python3 -m src.data.prepare_data --seq-len 60 --max-glosses 10 --selection-mode top_count --sequence-mode resample
python3 -m src.model.architecture
python3 -m src.model.train
python3 -m src.model.inference --text "about"
python3 -m src.renderer.smplx_renderer --first-frame --force-flip-vertical --view-yaw 0
python3 -m src.renderer.smplx_renderer --force-flip-vertical --view-yaw 0
python3 -m src.demo.generate_demo_videos --gloss about --force-flip-vertical --view-yaw 0
```

### Full Colab Workflow

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
cd /content/drive/MyDrive/FYP/SignAvatar_Generative_Project
apt-get update && apt-get install -y libegl1-mesa-dev libgles2-mesa-dev
pip install -r requirements.txt
python3 -m src.data.extract_dataset
python3 -m src.data.verify_dataset
python3 -m src.data.build_index
python3 -m src.data.prepare_data --seq-len 60 --max-glosses 10 --selection-mode top_count --sequence-mode resample
python3 -m src.model.train
python3 -m src.model.inference --text "about"
python3 -m src.renderer.smplx_renderer --first-frame --force-flip-vertical --view-yaw 0
python3 -m src.renderer.smplx_renderer --force-flip-vertical --view-yaw 0
python3 -m src.demo.generate_demo_videos --gloss about --force-flip-vertical --view-yaw 0
```

## 12. Troubleshooting

### Checkpoint missing

```text
checkpoints/best_model.pth
```

Run training first or pass a custom checkpoint to inference:

```bash
python3 -m src.model.inference --text "about" --checkpoint checkpoints/final_model.pth
```

### Unknown gloss

List valid glosses:

```bash
python3 -m src.model.inference --list-glosses
```

### Avatar upside down

Use:

```bash
--force-flip-vertical
```

Example:

```bash
python3 -m src.demo.generate_demo_videos --gloss about --force-flip-vertical --view-yaw 0
```

### Avatar back side visible

Create the orientation grid:

```bash
python3 -m src.renderer.smplx_renderer --orientation-grid
```

Then test yaw values:

```bash
python3 -m src.demo.generate_demo_videos --gloss about --force-flip-vertical --view-yaw 0
python3 -m src.demo.generate_demo_videos --gloss about --force-flip-vertical --view-yaw 180
```

### Old bad videos still show

Delete or overwrite old videos. Do not use `--skip-existing`.

```bash
python3 -m src.demo.generate_demo_videos --gloss about --force-flip-vertical --view-yaw 0
```

### Google Drive timeout on SMPL-X files

If you see `TimeoutError: [Errno 60] Operation timed out` while loading `SMPLX_NEUTRAL.npz`, make the folder available offline or copy it to local disk. In Colab, copying to `/content` is usually faster than rendering directly from Drive.

### Colab EGL/OpenGL issue

Run:

```bash
apt-get update && apt-get install -y libegl1-mesa-dev libgles2-mesa-dev
```

Then restart the runtime and rerun setup if needed.

## 13. Safety Notes

- Inference and demo generation use the trained neural model.
- Final render uses generated `.npy` motion.
- Final demo generation does not load or replay dataset `.pkl` files.
- Orientation fixes are applied only in the renderer/view layer.
- Do not modify `outputs/generated/generated_smplx.npy` to fix camera orientation.
