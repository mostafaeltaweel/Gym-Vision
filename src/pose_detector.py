"""Pose Detector module using MediaPipe Pose for gym exercise analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class Point:
    x: float  # Normalized 0..1 or pixel coordinate
    y: float
    z: float = 0.0
    visibility: float = 1.0


class PoseDetector:
    """Wrapper around MediaPipe Pose estimation with angle calculation and smoothing."""

    # Common MediaPipe landmark index aliases
    LANDMARKS = {
        "NOSE": 0,
        "LEFT_EYE_INNER": 1,
        "LEFT_EYE": 2,
        "LEFT_EYE_OUTER": 3,
        "RIGHT_EYE_INNER": 4,
        "RIGHT_EYE": 5,
        "RIGHT_EYE_OUTER": 6,
        "LEFT_EAR": 7,
        "RIGHT_EAR": 8,
        "MOUTH_LEFT": 9,
        "MOUTH_RIGHT": 10,
        "LEFT_SHOULDER": 11,
        "RIGHT_SHOULDER": 12,
        "LEFT_ELBOW": 13,
        "RIGHT_ELBOW": 14,
        "LEFT_WRIST": 15,
        "RIGHT_WRIST": 16,
        "LEFT_PINKY": 17,
        "RIGHT_PINKY": 18,
        "LEFT_INDEX": 19,
        "RIGHT_INDEX": 20,
        "LEFT_THUMB": 21,
        "RIGHT_THUMB": 22,
        "LEFT_HIP": 23,
        "RIGHT_HIP": 24,
        "LEFT_KNEE": 25,
        "RIGHT_KNEE": 26,
        "LEFT_ANKLE": 27,
        "RIGHT_ANKLE": 28,
        "LEFT_HEEL": 29,
        "RIGHT_HEEL": 30,
        "LEFT_FOOT_INDEX": 31,
        "RIGHT_FOOT_INDEX": 32,
    }

    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        smooth_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        import mediapipe as mp

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.prev_landmarks: Optional[Dict[str, Point]] = None
        self.alpha = 0.7  # Smoothing coefficient

    def find_pose(self, frame: np.ndarray) -> Tuple[Optional[Dict[str, Point]], Optional[Dict[str, Point]], any]:
        """Process a frame and return normalized and pixel landmark dictionaries.
        
        Returns:
            (norm_landmarks, pixel_landmarks, raw_results)
        """
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        if not results.pose_landmarks:
            return None, None, results

        norm_landmarks: Dict[str, Point] = {}
        pixel_landmarks: Dict[str, Point] = {}

        for name, idx in self.LANDMARKS.items():
            lm = results.pose_landmarks.landmark[idx]
            norm_landmarks[name] = Point(x=lm.x, y=lm.y, z=lm.z, visibility=lm.visibility)
            pixel_landmarks[name] = Point(
                x=lm.x * w,
                y=lm.y * h,
                z=lm.z * w,
                visibility=lm.visibility,
            )

        # Apply smoothing
        if self.prev_landmarks is not None:
            for name in norm_landmarks:
                norm_landmarks[name].x = self.alpha * norm_landmarks[name].x + (1 - self.alpha) * self.prev_landmarks[name].x
                norm_landmarks[name].y = self.alpha * norm_landmarks[name].y + (1 - self.alpha) * self.prev_landmarks[name].y

        self.prev_landmarks = norm_landmarks
        return norm_landmarks, pixel_landmarks, results

    @staticmethod
    def calculate_angle(a: Point | Tuple[float, float], b: Point | Tuple[float, float], c: Point | Tuple[float, float]) -> float:
        """Calculate angle between points a, b, c with b as the vertex.
        
        Returns angle in degrees (0 to 180).
        """
        ax, ay = (a.x, a.y) if isinstance(a, Point) else a
        bx, by = (b.x, b.y) if isinstance(b, Point) else b
        cx, cy = (c.x, c.y) if isinstance(c, Point) else c

        v1 = (ax - bx, ay - by)
        v2 = (cx - bx, cy - by)

        dot = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
        mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

        if mag1 * mag2 == 0:
            return 0.0

        cosine = dot / (mag1 * mag2)
        cosine = max(-1.0, min(1.0, cosine))  # Clamp for float precision
        angle = math.degrees(math.acos(cosine))
        return float(angle)

    @staticmethod
    def midpoint(p1: Point, p2: Point) -> Point:
        """Calculate midpoint between two points."""
        return Point(
            x=(p1.x + p2.x) / 2.0,
            y=(p1.y + p2.y) / 2.0,
            z=(p1.z + p2.z) / 2.0,
            visibility=min(p1.visibility, p2.visibility),
        )

    @staticmethod
    def distance(p1: Point, p2: Point) -> float:
        """Calculate Euclidean distance between two points."""
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

    def close(self):
        self.pose.close()
