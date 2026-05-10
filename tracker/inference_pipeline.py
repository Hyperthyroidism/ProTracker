from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from models.protracker import ProTracker
from tracker.track_manager import TrackManager
from tracker.prompt_generator import PromptGenerator


class ProTrackerInferencePipeline:
    """
    Complete inference pipeline for ProTracker.

    This pipeline connects all major modules:

        1. Video loading
        2. YOLO11 vessel detection
        3. Graph-based cascaded prompt optimization
        4. Target-aware prompt refinement
        5. SAM2-based segmentation and tracking
        6. MOT-format result saving
        7. Visualization output

    It is called by run.py and test.py.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Args:
            config: Configuration dictionary loaded from yaml file.
        """
        self.config = config

        tracking_cfg = config.get("tracking", {})
        sam2_cfg = config.get("sam2", {})
        visualization_cfg = config.get("visualization", {})

        self.result_dir = Path(tracking_cfg.get("result_dir", "results/test"))
        self.result_dir.mkdir(parents=True, exist_ok=True)

        self.use_sam2_mask = tracking_cfg.get("use_sam2_mask", True)
        self.save_mot_results = tracking_cfg.get("save_mot_results", True)

        self.visualization_enabled = visualization_cfg.get("enabled", True)
        self.show_box = visualization_cfg.get("show_box", True)
        self.show_mask = visualization_cfg.get("show_mask", True)
        self.show_track_id = visualization_cfg.get("show_track_id", True)
        self.show_confidence = visualization_cfg.get("show_confidence", False)
        self.line_width = visualization_cfg.get("line_width", 2)

        self.protracker = ProTracker(config)

        self.track_manager = TrackManager(
            association_threshold=tracking_cfg.get("association_threshold", 0.5),
            max_lost_frames=tracking_cfg.get("max_lost_frames", 30),
            min_track_length=tracking_cfg.get("min_track_length", 3),
        )

        self.prompt_generator = PromptGenerator(
            prompt_type=sam2_cfg.get("prompt_type", "box"),
            use_mask_prompt=sam2_cfg.get("use_mask_prompt", True),
        )

    @staticmethod
    def read_video_frames(video_path: str) -> List[np.ndarray]:
        """
        Read all frames from a video.

        Args:
            video_path: Input video path.

        Returns:
            List of video frames in BGR format.
        """
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        frames = []

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frames.append(frame)

        cap.release()

        return frames

    @staticmethod
    def get_video_info(video_path: str) -> Dict[str, Any]:
        """
        Get basic video information.

        Args:
            video_path: Input video path.

        Returns:
            Video information dictionary.
        """
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        cap.release()

        return {
            "fps": fps,
            "width": width,
            "height": height,
            "frame_count": frame_count,
        }

    def process_frames_before_sam2(
        self,
        frames: List[np.ndarray],
    ) -> List[Dict[str, Any]]:
        """
        Process frames before SAM2 propagation.

        This step performs vessel detection, graph-based prompt optimization,
        target-aware prompt refinement, and track management frame by frame.

        Args:
            frames: Video frame list.

        Returns:
            Frame-level result list.
        """
        frame_results = []

        for frame_id, frame in enumerate(frames):
            detections = self.protracker.detect_frame(
                frame=frame,
                frame_id=frame_id,
            )

            optimized_prompts = self.protracker.optimize_prompts(
                detections=detections,
                frame_id=frame_id,
                track_manager=self.track_manager,
            )

            assigned_prompts = self.track_manager.update(
                detections=optimized_prompts,
                frame_id=frame_id,
            )

            image_shape = frame.shape[:2]
            track_histories = self.track_manager.get_histories()

            refined_prompts = self.protracker.refine_prompts(
                prompts=assigned_prompts,
                image_shape=image_shape,
                track_histories=track_histories,
            )

            sam2_prompts = self.prompt_generator(
                items=refined_prompts,
                image_shape=image_shape,
            )

            frame_result = {
                "frame_id": frame_id,
                "detections": detections,
                "optimized_prompts": optimized_prompts,
                "assigned_prompts": assigned_prompts,
                "refined_prompts": refined_prompts,
                "sam2_prompts": sam2_prompts,
            }

            frame_results.append(frame_result)

            if frame_id % 20 == 0:
                print(
                    f"[Frame {frame_id:05d}] "
                    f"detections={len(detections)}, "
                    f"prompts={len(sam2_prompts)}"
                )

        return frame_results

    def select_initial_prompts(
        self,
        frame_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Select initial prompts for SAM2.

        By default, this function selects the first frame that contains valid prompts.

        Args:
            frame_results: Frame-level results.

        Returns:
            Dictionary containing initial frame index and prompt list.
        """
        for result in frame_results:
            prompts = result.get("sam2_prompts", [])

            if len(prompts) > 0:
                return {
                    "initial_frame_idx": int(result["frame_id"]),
                    "prompts": prompts,
                }

        return {
            "initial_frame_idx": 0,
            "prompts": [],
        }

    def run_sam2(
        self,
        video_path: str,
        initial_frame_idx: int,
        prompts: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Run SAM2 tracking with selected prompts.

        Args:
            video_path: Input video path.
            initial_frame_idx: Initial frame index.
            prompts: Prompt list.

        Returns:
            SAM2 segmentation results.
        """
        if len(prompts) == 0:
            print("No valid prompt found. SAM2 tracking is skipped.")
            return []

        segments = self.protracker.run_sam2_tracking(
            video_path=video_path,
            prompts=prompts,
            initial_frame_idx=initial_frame_idx,
        )

        return segments

    @staticmethod
    def mask_to_box(mask: np.ndarray) -> Optional[np.ndarray]:
        """
        Convert binary mask to bounding box.

        Args:
            mask: Binary mask.

        Returns:
            Bounding box in xyxy format. If mask is empty, return None.
        """
        if mask is None:
            return None

        mask = np.asarray(mask)

        if mask.ndim > 2:
            mask = np.squeeze(mask)

        ys, xs = np.where(mask > 0)

        if len(xs) == 0 or len(ys) == 0:
            return None

        x1 = float(np.min(xs))
        y1 = float(np.min(ys))
        x2 = float(np.max(xs))
        y2 = float(np.max(ys))

        return np.array([x1, y1, x2, y2], dtype=np.float32)

    @staticmethod
    def xyxy_to_xywh(box: np.ndarray) -> np.ndarray:
        """
        Convert xyxy box to xywh box.

        Args:
            box: Bounding box in xyxy format.

        Returns:
            Bounding box in xywh format.
        """
        x1, y1, x2, y2 = box
        return np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)

    def build_results_from_sam2(
        self,
        segments: List[Dict[str, Any]],
    ) -> List[List[float]]:
        """
        Convert SAM2 mask results into MOT-format tracking results.

        Args:
            segments: SAM2 video segmentation results.

        Returns:
            MOT-format results:
            [frame_id, track_id, x, y, w, h, confidence, class_id, visibility]
        """
        mot_results = []

        for frame_result in segments:
            frame_id = int(frame_result["frame_id"])
            masks = frame_result.get("masks", {})

            for track_id, mask in masks.items():
                box_xyxy = self.mask_to_box(mask)

                if box_xyxy is None:
                    continue

                x, y, w, h = self.xyxy_to_xywh(box_xyxy)

                mot_line = [
                    float(frame_id + 1),
                    float(track_id),
                    float(x),
                    float(y),
                    float(w),
                    float(h),
                    1.0,
                    0.0,
                    1.0,
                ]

                mot_results.append(mot_line)

        return mot_results

    def build_results_from_prompts(
        self,
        frame_results: List[Dict[str, Any]],
    ) -> List[List[float]]:
        """
        Convert refined prompts into MOT-format tracking results.

        This function is used as a fallback when SAM2 is not available.

        Args:
            frame_results: Frame-level results.

        Returns:
            MOT-format results.
        """
        mot_results = []

        for frame_result in frame_results:
            frame_id = int(frame_result["frame_id"])

            prompts = frame_result.get("refined_prompts", [])

            for prompt in prompts:
                if "bbox_xywh" not in prompt:
                    continue

                x, y, w, h = prompt["bbox_xywh"]
                track_id = int(prompt.get("track_id", -1))

                if track_id < 0:
                    continue

                mot_line = [
                    float(frame_id + 1),
                    float(track_id),
                    float(x),
                    float(y),
                    float(w),
                    float(h),
                    float(prompt.get("confidence", 1.0)),
                    float(prompt.get("class_id", 0)),
                    1.0,
                ]

                mot_results.append(mot_line)

        return mot_results

    @staticmethod
    def save_mot_results(
        mot_results: List[List[float]],
        save_path: str,
    ) -> None:
        """
        Save MOT-format results.

        Args:
            mot_results: MOT-format result list.
            save_path: Output txt path.
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w", encoding="utf-8") as f:
            for line in mot_results:
                line_str = ",".join(
                    [
                        str(int(value)) if idx in [0, 1, 7] else f"{value:.6f}"
                        for idx, value in enumerate(line)
                    ]
                )
                f.write(line_str + "\n")

    @staticmethod
    def get_color(track_id: int) -> tuple:
        """
        Generate a deterministic color for a track id.

        Args:
            track_id: Track identity.

        Returns:
            BGR color tuple.
        """
        np.random.seed(track_id)
        color = np.random.randint(0, 255, size=3).tolist()
        return int(color[0]), int(color[1]), int(color[2])

    def visualize_results(
        self,
        frames: List[np.ndarray],
        mot_results: List[List[float]],
        output_video_path: str,
        fps: float = 25.0,
    ) -> None:
        """
        Visualize tracking results and save output video.

        Args:
            frames: Video frame list.
            mot_results: MOT-format results.
            output_video_path: Output video path.
            fps: Video FPS.
        """
        if len(frames) == 0:
            return

        output_video_path = Path(output_video_path)
        output_video_path.parent.mkdir(parents=True, exist_ok=True)

        height, width = frames[0].shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(output_video_path),
            fourcc,
            fps,
            (width, height),
        )

        results_by_frame: Dict[int, List[List[float]]] = {}

        for line in mot_results:
            frame_id = int(line[0]) - 1
            results_by_frame.setdefault(frame_id, []).append(line)

        for frame_id, frame in enumerate(frames):
            vis_frame = frame.copy()

            for line in results_by_frame.get(frame_id, []):
                _, track_id, x, y, w, h, conf, _, _ = line

                track_id = int(track_id)
                color = self.get_color(track_id)

                x1 = int(x)
                y1 = int(y)
                x2 = int(x + w)
                y2 = int(y + h)

                if self.show_box:
                    cv2.rectangle(
                        vis_frame,
                        (x1, y1),
                        (x2, y2),
                        color,
                        self.line_width,
                    )

                if self.show_track_id:
                    if self.show_confidence:
                        text = f"ID {track_id} {conf:.2f}"
                    else:
                        text = f"ID {track_id}"

                    cv2.putText(
                        vis_frame,
                        text,
                        (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )

            writer.write(vis_frame)

        writer.release()

    def run(
        self,
        video_path: str,
        output_dir: str,
    ) -> Dict[str, Any]:
        """
        Run the complete ProTracker inference pipeline.

        Args:
            video_path: Input video path.
            output_dir: Output directory.

        Returns:
            Result dictionary.
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Running ProTracker on video: {video_path}")

        video_info = self.get_video_info(str(video_path))
        frames = self.read_video_frames(str(video_path))

        print(
            f"Video loaded: {len(frames)} frames, "
            f"resolution={video_info['width']}x{video_info['height']}, "
            f"fps={video_info['fps']:.2f}"
        )

        self.track_manager.reset()

        frame_results = self.process_frames_before_sam2(frames)

        initial_info = self.select_initial_prompts(frame_results)
        initial_frame_idx = initial_info["initial_frame_idx"]
        initial_prompts = initial_info["prompts"]

        print(
            f"Initial SAM2 prompts selected from frame {initial_frame_idx}, "
            f"number of prompts: {len(initial_prompts)}"
        )

        segments = self.run_sam2(
            video_path=str(video_path),
            initial_frame_idx=initial_frame_idx,
            prompts=initial_prompts,
        )

        if len(segments) > 0 and self.use_sam2_mask:
            mot_results = self.build_results_from_sam2(segments)
        else:
            mot_results = self.build_results_from_prompts(frame_results)

        result_txt_path = output_dir / f"{video_path.stem}.txt"

        if self.save_mot_results:
            self.save_mot_results_to_file(
                mot_results=mot_results,
                save_path=str(result_txt_path),
            )

        output_video_path = output_dir / f"{video_path.stem}_vis.mp4"

        if self.visualization_enabled:
            self.visualize_results(
                frames=frames,
                mot_results=mot_results,
                output_video_path=str(output_video_path),
                fps=video_info["fps"] if video_info["fps"] > 0 else 25.0,
            )

        result = {
            "video_path": str(video_path),
            "output_dir": str(output_dir),
            "result_txt": str(result_txt_path),
            "output_video": str(output_video_path),
            "num_frames": len(frames),
            "num_results": len(mot_results),
            "initial_frame_idx": initial_frame_idx,
            "num_initial_prompts": len(initial_prompts),
        }

        print("Inference finished.")
        print(f"MOT result saved to: {result_txt_path}")

        if self.visualization_enabled:
            print(f"Visualization saved to: {output_video_path}")

        return result

    def save_mot_results_to_file(
        self,
        mot_results: List[List[float]],
        save_path: str,
    ) -> None:
        """
        Wrapper for saving MOT-format results.

        Args:
            mot_results: MOT-format results.
            save_path: Output txt path.
        """
        self.save_mot_results(
            mot_results=mot_results,
            save_path=save_path,
        )