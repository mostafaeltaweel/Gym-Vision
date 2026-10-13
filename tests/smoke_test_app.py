"""Smoke test for Gym Vision AI modules and end-to-end video processing."""

import os
from pathlib import Path
import cv2
from src.pose_detector import PoseDetector
from src.pullup_analyzer import PullUpAnalyzer
from src.visualizer import GymVisualizer


def test_smoke_end_to_end():
    sample_video = "tests/sample_workout.mp4"
    assert os.path.exists(sample_video), "Sample video does not exist."

    cap = cv2.VideoCapture(sample_video)
    detector = PoseDetector()
    analyzer = PullUpAnalyzer()

    processed_frames = 0
    while cap.isOpened() and processed_frames < 30:
        ret, frame = cap.read()
        if not ret:
            break
        norm_lms, pixel_lms, _ = detector.find_pose(frame)
        if pixel_lms:
            frame = GymVisualizer.draw_skeleton(frame, pixel_lms)
            phase, rep = analyzer.update(norm_lms)
            frame = GymVisualizer.draw_hud(
                frame,
                "Pull-Ups",
                analyzer.total_reps,
                "Status",
                "Testing",
                phase.value,
                "Smoke test",
            )
        processed_frames += 1

    cap.release()
    detector.close()
    assert processed_frames > 0, "No frames were processed."
    print(f"Successfully processed {processed_frames} frames in smoke test.")


if __name__ == "__main__":
    test_smoke_end_to_end()
