import argparse
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import numpy as np
import torch
from tqdm import tqdm

from src.config import OUTPUT_DIR, SMPLX_MODEL_DIR
from src.renderer.generated_motion import DEFAULT_GENERATED_MOTION_PATH, load_generated_motion


DEFAULT_VIDEO_DIR = OUTPUT_DIR / "videos"
DEFAULT_VIDEO_PATH = DEFAULT_VIDEO_DIR / "generated_animation.mp4"
DEFAULT_FIRST_FRAME_PATH = DEFAULT_VIDEO_DIR / "generated_first_frame.png"
DEFAULT_ZERO_POSE_PATH = DEFAULT_VIDEO_DIR / "zero_pose_test.png"
DEFAULT_ORIENTATION_GRID_PATH = DEFAULT_VIDEO_DIR / "orientation_grid.png"
DEFAULT_ZERO_POSE_GRID_PATH = DEFAULT_VIDEO_DIR / "zero_pose_orientation_grid.png"


def rotation_matrix_x(degrees: float) -> np.ndarray:
    radians = np.radians(degrees)
    c, s = np.cos(radians), np.sin(radians)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float32)


def rotation_matrix_y(degrees: float) -> np.ndarray:
    radians = np.radians(degrees)
    c, s = np.cos(radians), np.sin(radians)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)


def rotation_matrix_z(degrees: float) -> np.ndarray:
    radians = np.radians(degrees)
    c, s = np.cos(radians), np.sin(radians)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


class SMPLXRenderer:
    def __init__(
        self,
        model_dir: str | Path | None = None,
        gender: str = "neutral",
        image_size: tuple[int, int] = (512, 512),
        fps: int = 20,
        device: str | torch.device | None = None,
        force_flip_vertical: bool = True,
        flip_vertical: bool = False,
        view_yaw_degrees: float = 0.0,
        view_pitch_degrees: float = 0.0,
        view_roll_degrees: float = 0.0,
        debug_point_fallback: bool = False,
    ):
        self.model_dir = Path(model_dir) if model_dir is not None else SMPLX_MODEL_DIR
        self.gender = gender.lower()
        self.image_size = image_size
        self.fps = fps
        self.device = torch.device(device) if device is not None else self._select_device()
        self.force_flip_vertical = force_flip_vertical
        self.flip_vertical = flip_vertical
        self.view_yaw_degrees = view_yaw_degrees
        self.view_pitch_degrees = view_pitch_degrees
        self.view_roll_degrees = view_roll_degrees
        self.debug_point_fallback = debug_point_fallback
        self.smplx_model = self._load_smplx_model()
        self.faces = np.asarray(self.smplx_model.faces)
        self._logged_render_mode = False
        self._logged_solid_fallback = False

    def _select_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _load_smplx_model(self) -> Any:
        if self.gender not in {"neutral", "male", "female"}:
            raise ValueError("gender must be one of: neutral, male, female")
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"SMPL-X model directory not found: {self.model_dir}. "
                "Expected model files under human_models/smplx/."
            )
        model_path, model_ext = self._resolve_smplx_model_file()

        try:
            import smplx
        except ImportError as exc:
            raise ImportError("Missing dependency 'smplx'. Install requirements.txt before rendering.") from exc

        try:
            model = smplx.create(
                model_path=str(model_path),
                model_type="smplx",
                gender=self.gender,
                ext=model_ext,
                use_face_contour=False,
                flat_hand_mean=True,
                use_pca=True,
                num_pca_comps=45,
                num_betas=10,
                num_expression_coeffs=10,
            )
        except Exception as exc:
            sync_hint = ""
            if isinstance(exc, TimeoutError):
                sync_hint = (
                    " The model file timed out while reading. If this project is in Google Drive/iCloud, "
                    "mark human_models/smplx as available offline or copy the SMPL-X files to local disk."
                )
            raise RuntimeError(
                f"Failed to load SMPL-X model from {model_path}. "
                "Confirm SMPLX_NEUTRAL/MALE/FEMALE .npz or .pkl files are present."
                f"{sync_hint}"
            ) from exc

        return model.to(self.device).eval()

    def _resolve_smplx_model_file(self) -> tuple[Path, str]:
        model_name = f"SMPLX_{self.gender.upper()}"
        for ext in ("npz", "pkl"):
            direct_path = self.model_dir / f"{model_name}.{ext}"
            if direct_path.exists():
                return direct_path, ext

        nested_dir = self.model_dir / "smplx"
        for ext in ("npz", "pkl"):
            nested_path = nested_dir / f"{model_name}.{ext}"
            if nested_path.exists():
                return nested_path, ext

        raise FileNotFoundError(
            f"No SMPL-X {self.gender} model file found in {self.model_dir}. "
            f"Expected {model_name}.npz or {model_name}.pkl."
        )

    def parse_smplx_frame(self, frame: np.ndarray) -> dict[str, torch.Tensor]:
        frame = np.asarray(frame, dtype=np.float32).reshape(-1)
        if frame.shape[0] != 182:
            raise ValueError(f"Expected one SMPL-X frame with 182 values, got {frame.shape[0]}.")

        # SignAvatars/WLASL 182-D layout from the original rendering notebook:
        # 0:3 global/root orient, 3:66 body pose, 66:111 left hand PCA,
        # 111:156 right hand PCA, 156:159 jaw, 159:169 betas,
        # 169:179 expression, 179:182 camera translation.
        arrays = {
            "global_orient": frame[0:3],
            "body_pose": frame[3:66],
            "left_hand_pose": frame[66:111],
            "right_hand_pose": frame[111:156],
            "jaw_pose": frame[156:159],
            "betas": frame[159:169],
            "expression": frame[169:179],
            # The old demo preserved camera translation separately and rendered
            # meshes at the origin. Keep transl zero for stable generated demos.
            "transl": np.zeros(3, dtype=np.float32),
        }

        return {
            key: torch.from_numpy(value.reshape(1, -1)).to(self.device)
            for key, value in arrays.items()
        }

    def motion_to_vertices(self, motion: np.ndarray, max_frames: int | None = None) -> list[dict[str, np.ndarray]]:
        motion = np.asarray(motion, dtype=np.float32)
        if motion.ndim != 2 or motion.shape[1] != 182:
            raise ValueError(f"Expected motion shape (T, 182), got {motion.shape}.")

        if max_frames is not None:
            if max_frames <= 0:
                raise ValueError("max_frames must be positive when provided.")
            motion = motion[:max_frames]

        mesh_frames = []
        with torch.no_grad():
            for frame in tqdm(motion, desc="Generating SMPL-X meshes", leave=False):
                params = self.parse_smplx_frame(frame)
                output = self.smplx_model(**params)
                mesh_frames.append(
                    {
                        "vertices": output.vertices[0].detach().cpu().numpy(),
                        "joints": output.joints[0].detach().cpu().numpy(),
                    }
                )
        return mesh_frames

    def zero_pose_mesh_frame(self) -> dict[str, np.ndarray]:
        params = {
            "global_orient": torch.zeros((1, 3), dtype=torch.float32, device=self.device),
            "body_pose": torch.zeros((1, 63), dtype=torch.float32, device=self.device),
            "left_hand_pose": torch.zeros((1, 45), dtype=torch.float32, device=self.device),
            "right_hand_pose": torch.zeros((1, 45), dtype=torch.float32, device=self.device),
            "jaw_pose": torch.zeros((1, 3), dtype=torch.float32, device=self.device),
            "betas": torch.zeros((1, 10), dtype=torch.float32, device=self.device),
            "expression": torch.zeros((1, 10), dtype=torch.float32, device=self.device),
            "transl": torch.zeros((1, 3), dtype=torch.float32, device=self.device),
        }
        with torch.no_grad():
            output = self.smplx_model(**params)
        return {
            "vertices": output.vertices[0].detach().cpu().numpy(),
            "joints": output.joints[0].detach().cpu().numpy(),
        }

    def render_generated_motion(
        self,
        motion_path: str | Path | None = None,
        output_path: str | Path | None = None,
        max_frames: int | None = None,
    ) -> Path:
        motion = load_generated_motion(motion_path)
        mesh_frames = self.motion_to_vertices(motion, max_frames=max_frames)

        video_path = Path(output_path) if output_path is not None else DEFAULT_VIDEO_PATH
        video_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_video(mesh_frames, video_path)
        return video_path

    def render_first_frame(
        self,
        motion_path: str | Path | None = None,
        output_path: str | Path | None = None,
    ) -> Path:
        motion = load_generated_motion(motion_path)
        mesh_frames = self.motion_to_vertices(motion[:1], max_frames=1)

        image_path = Path(output_path) if output_path is not None else DEFAULT_FIRST_FRAME_PATH
        image_path.parent.mkdir(parents=True, exist_ok=True)
        frame = self._render_mesh_frame(mesh_frames[0])
        if not cv2.imwrite(str(image_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"Failed to write first-frame image to {image_path}.")
        return image_path

    def render_zero_pose(self, output_path: str | Path | None = None) -> Path:
        image_path = Path(output_path) if output_path is not None else DEFAULT_ZERO_POSE_PATH
        image_path.parent.mkdir(parents=True, exist_ok=True)
        frame = self._render_mesh_frame(self.zero_pose_mesh_frame())
        if not cv2.imwrite(str(image_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"Failed to write zero-pose image to {image_path}.")
        return image_path

    def render_orientation_grid(
        self,
        motion_path: str | Path | None = None,
        output_path: str | Path | None = None,
    ) -> Path:
        motion = load_generated_motion(motion_path)
        mesh_frame = self.motion_to_vertices(motion[:1], max_frames=1)[0]
        image_path = Path(output_path) if output_path is not None else DEFAULT_ORIENTATION_GRID_PATH
        return self._write_orientation_grid(mesh_frame, image_path)

    def render_zero_pose_orientation_grid(self, output_path: str | Path | None = None) -> Path:
        image_path = Path(output_path) if output_path is not None else DEFAULT_ZERO_POSE_GRID_PATH
        return self._write_orientation_grid(self.zero_pose_mesh_frame(), image_path)

    def _write_orientation_grid(self, mesh_frame: dict[str, np.ndarray], output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        variants = [
            (0.0, False),
            (0.0, True),
            (180.0, False),
            (180.0, True),
            (90.0, True),
            (-90.0, True),
        ]

        original = (
            self.view_yaw_degrees,
            self.view_pitch_degrees,
            self.view_roll_degrees,
            self.force_flip_vertical,
            self.flip_vertical,
            self._logged_render_mode,
            self._logged_solid_fallback,
        )

        cells = []
        try:
            for yaw, force_flip in variants:
                self.view_yaw_degrees = yaw
                self.view_pitch_degrees = 0.0
                self.view_roll_degrees = 0.0
                self.force_flip_vertical = force_flip
                self.flip_vertical = False
                self._logged_render_mode = True
                self._logged_solid_fallback = True
                frame = self._render_mesh_frame(mesh_frame)
                label = f"yaw={int(yaw)} flip={force_flip}"
                cells.append(self._add_label(frame, label))
        finally:
            (
                self.view_yaw_degrees,
                self.view_pitch_degrees,
                self.view_roll_degrees,
                self.force_flip_vertical,
                self.flip_vertical,
                self._logged_render_mode,
                self._logged_solid_fallback,
            ) = original

        row1 = np.concatenate(cells[:3], axis=1)
        row2 = np.concatenate(cells[3:], axis=1)
        grid = np.concatenate([row1, row2], axis=0)
        if not cv2.imwrite(str(output_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"Failed to write orientation grid to {output_path}.")
        return output_path

    def _add_label(self, frame: np.ndarray, label: str) -> np.ndarray:
        labeled = frame.copy()
        cv2.rectangle(labeled, (0, 0), (labeled.shape[1], 34), (255, 255, 255), -1)
        cv2.putText(
            labeled,
            label,
            (10, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        return labeled

    def _write_video(self, mesh_frames: list[dict[str, np.ndarray]], video_path: Path) -> None:
        if not mesh_frames:
            raise RuntimeError("No vertices were generated, so no video can be rendered.")

        width, height = self.image_size
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"OpenCV could not open MP4 writer for {video_path}.")

        try:
            for mesh_frame in tqdm(mesh_frames, desc="Rendering video", leave=False):
                rgb = self._render_mesh_frame(mesh_frame)
                writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()

    def _render_mesh_frame(self, mesh_frame: dict[str, np.ndarray]) -> np.ndarray:
        vertices = self._apply_view_transform(mesh_frame["vertices"])
        joints = mesh_frame["joints"]
        flip_applied = self._should_flip_vertical(joints)
        self._log_render_mode(vertices, flip_applied)
        try:
            frame = self._render_vertices_with_pyrender(vertices)
        except Exception as exc:
            if not self._logged_solid_fallback:
                print(
                    "pyrender solid mesh rendering failed. Using OpenCV triangle rasterizer "
                    "with SMPL-X faces for this render."
                )
                self._logged_solid_fallback = True
            frame = self._render_vertices_with_opencv_solid_mesh(vertices)

            if self.debug_point_fallback:
                print("Debug point-cloud fallback is enabled, but solid triangle fallback was used first.")

        if flip_applied:
            frame = np.flipud(frame)
        return np.ascontiguousarray(frame)

    def _should_flip_vertical(self, joints: np.ndarray) -> bool:
        if self.force_flip_vertical:
            return True
        if not self.flip_vertical:
            return False
        if joints.shape[0] <= 15:
            return True

        head_y = float(joints[15, 1])
        feet_y = float(np.mean(joints[[10, 11], 1]))
        # Enable a view-layer correction only when this frame would render
        # upside down with the current projection. This keeps generated root
        # orientation untouched while making the default zero pose upright.
        return head_y > feet_y

    def _view_rotation_matrix(self) -> np.ndarray:
        rx = rotation_matrix_x(self.view_pitch_degrees)
        ry = rotation_matrix_y(self.view_yaw_degrees)
        rz = rotation_matrix_z(self.view_roll_degrees)
        return rz @ ry @ rx

    def _apply_view_transform(self, vertices: np.ndarray) -> np.ndarray:
        rotation = self._view_rotation_matrix()
        return vertices.astype(np.float32) @ rotation.T

    def _log_render_mode(self, vertices: np.ndarray, flip_applied: bool) -> None:
        if self._logged_render_mode:
            return
        print(f"Render mode: solid SMPL-X mesh")
        print(f"vertices shape: {vertices.shape}")
        print(f"faces shape: {self.faces.shape}")
        print(f"solid mesh rendering: True")
        print(f"force vertical flip: {self.force_flip_vertical}")
        print(f"auto vertical flip enabled: {self.flip_vertical}")
        print(f"vertical flip applied: {flip_applied}")
        print(f"view_yaw_degrees: {self.view_yaw_degrees}")
        print(f"view_pitch_degrees: {self.view_pitch_degrees}")
        print(f"view_roll_degrees: {self.view_roll_degrees}")
        self._logged_render_mode = True

    def _render_vertices_with_pyrender(self, vertices: np.ndarray) -> np.ndarray:
        try:
            import pyrender
            import trimesh
        except ImportError as exc:
            raise ImportError("Missing pyrender/trimesh. Install requirements.txt before rendering.") from exc

        width, height = self.image_size
        # pyrender's camera looks along its local -Z axis, which makes the raw
        # SMPL-X view appear opposite to the simple OpenCV projection. Apply a
        # backend-only yaw correction so view_yaw_degrees has the same visual
        # meaning in both renderers.
        render_vertices = vertices @ rotation_matrix_y(180.0).T
        mesh = trimesh.Trimesh(vertices=render_vertices, faces=self.faces, process=False)

        scene = pyrender.Scene(bg_color=[255, 255, 255, 255], ambient_light=[0.25, 0.25, 0.25])
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            roughnessFactor=0.65,
            baseColorFactor=[0.55, 0.66, 0.86, 1.0],
        )
        mesh_node = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)
        scene.add(mesh_node)

        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        camera_pose = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, -0.1],
                [0.0, 0.0, 1.0, 3.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        scene.add(camera, pose=camera_pose)
        scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.0), pose=camera_pose)

        renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
        try:
            color, _ = renderer.render(scene)
            return color[:, :, :3].astype(np.uint8)
        finally:
            renderer.delete()

    def _render_vertices_with_opencv_solid_mesh(self, vertices: np.ndarray) -> np.ndarray:
        width, height = self.image_size
        canvas = np.full((height, width, 3), 255, dtype=np.uint8)

        projected, depths = self._project_vertices(vertices)
        face_vertices = projected[self.faces]
        face_depths = depths[self.faces].mean(axis=1)
        order = np.argsort(face_depths)[::-1]

        light_dir = np.array([0.2, -0.5, 1.0], dtype=np.float32)
        light_dir /= np.linalg.norm(light_dir)
        base_color = np.array([120, 145, 215], dtype=np.float32)

        for face_index in order:
            tri = face_vertices[face_index]
            if (
                np.all(tri[:, 0] < 0)
                or np.all(tri[:, 0] >= width)
                or np.all(tri[:, 1] < 0)
                or np.all(tri[:, 1] >= height)
            ):
                continue

            vertex_ids = self.faces[face_index]
            v0, v1, v2 = vertices[vertex_ids]
            normal = np.cross(v1 - v0, v2 - v0)
            norm = np.linalg.norm(normal)
            if norm <= 1e-8:
                continue
            normal = normal / norm
            shade = 0.45 + 0.55 * max(float(np.dot(normal, light_dir)), 0.0)
            color = np.clip(base_color * shade, 35, 235).astype(np.uint8).tolist()
            cv2.fillConvexPoly(canvas, tri.astype(np.int32), color, lineType=cv2.LINE_AA)

        return canvas

    def _project_vertices(self, vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        width, height = self.image_size
        centered = vertices.astype(np.float32).copy()
        centered -= centered.mean(axis=0, keepdims=True)

        xy = centered[:, [0, 1]]
        xy[:, 1] *= -1.0
        scale = np.percentile(np.linalg.norm(xy, axis=1), 99)
        if not np.isfinite(scale) or scale <= 1e-6:
            scale = 1.0

        xy = xy / (scale * 2.4)
        xy[:, 0] = (xy[:, 0] + 0.5) * width
        xy[:, 1] = (xy[:, 1] + 0.52) * height
        return xy.astype(np.int32), centered[:, 2]

    def _render_vertices_with_opencv(self, vertices: np.ndarray) -> np.ndarray:
        if not self.debug_point_fallback:
            raise RuntimeError("Point-cloud rendering is debug-only. Enable debug_point_fallback=True to use it.")

        width, height = self.image_size
        canvas = np.full((height, width, 3), 255, dtype=np.uint8)

        points = vertices[:, [0, 1]].copy()
        points[:, 1] *= -1.0
        center = points.mean(axis=0, keepdims=True)
        points -= center
        scale = np.percentile(np.linalg.norm(points, axis=1), 98)
        if not np.isfinite(scale) or scale <= 1e-6:
            scale = 1.0
        points = points / (scale * 2.2)
        points[:, 0] = (points[:, 0] + 0.5) * width
        points[:, 1] = (points[:, 1] + 0.52) * height
        points = points.astype(np.int32)

        valid = (
            (points[:, 0] >= 0)
            & (points[:, 0] < width)
            & (points[:, 1] >= 0)
            & (points[:, 1] < height)
        )
        for x, y in points[valid][::3]:
            cv2.circle(canvas, (int(x), int(y)), 1, (90, 110, 170), -1, lineType=cv2.LINE_AA)
        return canvas


def render_generated_motion(
    motion_path: str | Path | None = None,
    output_path: str | Path | None = None,
    max_frames: int | None = None,
) -> Path:
    renderer = SMPLXRenderer()
    return renderer.render_generated_motion(
        motion_path=motion_path,
        output_path=output_path,
        max_frames=max_frames,
    )


def render_first_frame(
    motion_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> Path:
    renderer = SMPLXRenderer()
    return renderer.render_first_frame(motion_path=motion_path, output_path=output_path)


def render_zero_pose(output_path: str | Path | None = None) -> Path:
    renderer = SMPLXRenderer()
    return renderer.render_zero_pose(output_path=output_path)


def render_orientation_grid(
    motion_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> Path:
    renderer = SMPLXRenderer()
    return renderer.render_orientation_grid(motion_path=motion_path, output_path=output_path)


def render_zero_pose_orientation_grid(output_path: str | Path | None = None) -> Path:
    renderer = SMPLXRenderer()
    return renderer.render_zero_pose_orientation_grid(output_path=output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render generated SMPL-X motion to MP4.")
    parser.add_argument("--motion-path", type=str, default=str(DEFAULT_GENERATED_MOTION_PATH))
    parser.add_argument("--output-path", type=str, default=str(DEFAULT_VIDEO_PATH))
    parser.add_argument("--first-frame", action="store_true", help="Render only the first frame PNG.")
    parser.add_argument("--zero-pose", action="store_true", help="Render a neutral SMPL-X zero-pose PNG.")
    parser.add_argument("--orientation-grid", action="store_true", help="Render generated first-frame orientation grid.")
    parser.add_argument("--zero-pose-grid", action="store_true", help="Render zero-pose orientation grid.")
    parser.add_argument("--max-frames", type=int, default=None, help="Render at most this many frames.")
    parser.add_argument("--gender", type=str, default="neutral", choices=["neutral", "male", "female"])
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--image-size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), default=(512, 512))
    parser.add_argument("--force-flip-vertical", dest="force_flip_vertical", action="store_true", default=True)
    parser.add_argument("--no-force-flip-vertical", dest="force_flip_vertical", action="store_false")
    parser.add_argument("--no-flip-vertical", action="store_true", help="Disable final vertical frame flip.")
    parser.add_argument("--view-yaw", type=float, default=0.0)
    parser.add_argument("--view-pitch", type=float, default=0.0)
    parser.add_argument("--view-roll", type=float, default=0.0)
    parser.add_argument("--debug-point-fallback", action="store_true", help="Allow debug point-cloud fallback.")
    args = parser.parse_args()

    renderer = SMPLXRenderer(
        gender=args.gender,
        image_size=tuple(args.image_size),
        fps=args.fps,
        force_flip_vertical=args.force_flip_vertical,
        flip_vertical=not args.no_flip_vertical and not args.force_flip_vertical,
        view_yaw_degrees=args.view_yaw,
        view_pitch_degrees=args.view_pitch,
        view_roll_degrees=args.view_roll,
        debug_point_fallback=args.debug_point_fallback,
    )
    if args.orientation_grid:
        output_path = renderer.render_orientation_grid(
            motion_path=args.motion_path,
            output_path=DEFAULT_ORIENTATION_GRID_PATH if args.output_path == str(DEFAULT_VIDEO_PATH) else args.output_path,
        )
    elif args.zero_pose_grid:
        output_path = renderer.render_zero_pose_orientation_grid(
            output_path=DEFAULT_ZERO_POSE_GRID_PATH if args.output_path == str(DEFAULT_VIDEO_PATH) else args.output_path,
        )
    elif args.zero_pose:
        output_path = renderer.render_zero_pose(
            output_path=DEFAULT_ZERO_POSE_PATH if args.output_path == str(DEFAULT_VIDEO_PATH) else args.output_path,
        )
    elif args.first_frame:
        output_path = renderer.render_first_frame(
            motion_path=args.motion_path,
            output_path=DEFAULT_FIRST_FRAME_PATH if args.output_path == str(DEFAULT_VIDEO_PATH) else args.output_path,
        )
    else:
        output_path = renderer.render_generated_motion(
            motion_path=args.motion_path,
            output_path=args.output_path,
            max_frames=args.max_frames,
        )
    print(output_path)


if __name__ == "__main__":
    main()
