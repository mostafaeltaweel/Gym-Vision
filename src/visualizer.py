"""Visualization & HUD Overlay Module for Gym Vision AI."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .pose_detector import Point


class GymVisualizer:
    """Renders biomechanical skeletons, telemetry gauges, and HUD overlays."""

    SKELETON_PAIRS = [
        # Torso & Arms
        ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
        ("LEFT_SHOULDER", "LEFT_ELBOW"),
        ("LEFT_ELBOW", "LEFT_WRIST"),
        ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
        ("RIGHT_ELBOW", "RIGHT_WRIST"),
        ("LEFT_SHOULDER", "LEFT_HIP"),
        ("RIGHT_SHOULDER", "RIGHT_HIP"),
        ("LEFT_HIP", "RIGHT_HIP"),
        # Legs
        ("LEFT_HIP", "LEFT_KNEE"),
        ("LEFT_KNEE", "LEFT_ANKLE"),
        ("RIGHT_HIP", "RIGHT_KNEE"),
        ("RIGHT_KNEE", "RIGHT_ANKLE"),
    ]

    # Colors (BGR)
    COLOR_NEON_GREEN = (50, 255, 50)
    COLOR_CYAN = (255, 220, 0)
    COLOR_ORANGE = (0, 165, 255)
    COLOR_RED = (50, 50, 255)
    COLOR_WHITE = (255, 255, 255)
    COLOR_DARK_BG = (20, 20, 20)

    @classmethod
    def draw_skeleton(
        cls,
        frame: np.ndarray,
        pixel_landmarks: Dict[str, Point],
        color: Tuple[int, int, int] = (0, 255, 200),
        thickness: int = 3,
    ) -> np.ndarray:
        """Draw pose skeleton connections and joint nodes."""
        h, w, _ = frame.shape

        # Draw connection lines
        for start_name, end_name in cls.SKELETON_PAIRS:
            p1 = pixel_landmarks.get(start_name)
            p2 = pixel_landmarks.get(end_name)
            if p1 and p2 and p1.visibility > 0.4 and p2.visibility > 0.4:
                pt1 = (int(p1.x), int(p1.y))
                pt2 = (int(p2.x), int(p2.y))
                # Outer shadow
                cv2.line(frame, pt1, pt2, (0, 0, 0), thickness + 2)
                # Main line
                cv2.line(frame, pt1, pt2, color, thickness)

        # Draw joint keypoints
        for name, p in pixel_landmarks.items():
            if p.visibility > 0.4:
                pt = (int(p.x), int(p.y))
                cv2.circle(frame, pt, 6, (0, 0, 0), -1)
                cv2.circle(frame, pt, 4, cls.COLOR_WHITE, -1)

        return frame

    @classmethod
    def draw_hud(
        cls,
        frame: np.ndarray,
        exercise_name: str,
        reps_count: int,
        primary_metric_label: str,
        primary_metric_value: str,
        phase_label: str,
        status_feedback: str,
        is_good_form: bool = True,
    ) -> np.ndarray:
        """Render a modern HUD card at top of frame."""
        h, w, _ = frame.shape
        hud_h = int(min(120, h * 0.18))
        hud_w = min(w - 20, 680)
        x1, y1 = 10, 10
        x2, y2 = x1 + hud_w, y1 + hud_h

        # Semi-transparent dark background card
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), cls.COLOR_DARK_BG, -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Border colored according to status
        border_color = cls.COLOR_NEON_GREEN if is_good_form else cls.COLOR_ORANGE
        cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, 2)

        # Left block: Exercise Title & Phase
        cv2.putText(
            frame,
            f"{exercise_name.upper()}",
            (x1 + 15, y1 + 30),
            cv2.FONT_HERSHEY_DUPLEX,
            0.75,
            cls.COLOR_WHITE,
            2,
            cv2.LINE_AA,
        )

        phase_color = cls.COLOR_CYAN if phase_label != "IDLE" else (180, 180, 180)
        cv2.putText(
            frame,
            f"Phase: {phase_label}",
            (x1 + 15, y1 + 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            phase_color,
            1,
            cv2.LINE_AA,
        )

        # Center block: Reps Banner
        rep_box_x = x1 + 220
        cv2.putText(
            frame,
            "REPS",
            (rep_box_x, y1 + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            str(reps_count),
            (rep_box_x, y1 + 70),
            cv2.FONT_HERSHEY_DUPLEX,
            1.4,
            border_color,
            3,
            cv2.LINE_AA,
        )

        # Right block: Active Angle / Metric
        metric_x = x1 + 350
        cv2.putText(
            frame,
            primary_metric_label.upper(),
            (metric_x, y1 + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            primary_metric_value,
            (metric_x, y1 + 65),
            cv2.FONT_HERSHEY_DUPLEX,
            1.1,
            cls.COLOR_WHITE,
            2,
            cv2.LINE_AA,
        )

        # Bottom Sub-banner: Status Feedback text
        cv2.putText(
            frame,
            f">> {status_feedback}",
            (x1 + 15, y2 - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            border_color,
            1,
            cv2.LINE_AA,
        )

        return frame

    @classmethod
    def draw_joint_angle(
        cls,
        frame: np.ndarray,
        vertex_point: Point,
        angle_val: float,
        label: str = "",
        color: Tuple[int, int, int] = (0, 255, 255),
    ) -> np.ndarray:
        """Render angle text badge directly beside a joint."""
        px, py = int(vertex_point.x), int(vertex_point.y)
        text = f"{int(angle_val)} deg" if not label else f"{label}: {int(angle_val)} deg"

        # Background tag
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (px + 10, py - th - 5), (px + 15 + tw, py + 5), (0, 0, 0), -1)
        cv2.rectangle(frame, (px + 10, py - th - 5), (px + 15 + tw, py + 5), color, 1)
        cv2.putText(
            frame,
            text,
            (px + 12, py),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
        return frame
