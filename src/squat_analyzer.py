"""Squat Biomechanical Rep & Depth Form Analyzer."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from .pose_detector import Point, PoseDetector


class SquatPhase(str, Enum):
    STANDING = "STANDING"
    DESCENDING = "DESCENDING"
    BOTTOM = "BOTTOM"
    ASCENDING = "ASCENDING"


class SquatDepth(str, Enum):
    FULL = "FULL (BELOW PARALLEL)"
    PARALLEL = "PARALLEL"
    SHALLOW = "SHALLOW"


@dataclass
class SquatRepDetail:
    rep_number: int
    depth: SquatDepth
    feedback: str
    min_knee_angle: float
    max_knee_angle: float
    descent_time: float
    ascent_time: float
    total_time: float
    start_frame: int
    bottom_frame: int
    end_frame: int


class SquatAnalyzer:
    """Biomechanical counter and depth evaluator for Squats."""

    def __init__(
        self,
        standing_knee_threshold: float = 160.0,
        parallel_knee_threshold: float = 90.0,
        deep_knee_threshold: float = 80.0,
        fps: float = 30.0,
    ):
        self.standing_knee_threshold = standing_knee_threshold
        self.parallel_knee_threshold = parallel_knee_threshold
        self.deep_knee_threshold = deep_knee_threshold
        self.fps = fps

        self.phase = SquatPhase.STANDING
        self.total_reps = 0
        self.full_depth_reps = 0
        self.shallow_reps = 0
        self.rep_history: List[SquatRepDetail] = []

        self.current_rep_start_frame = 0
        self.current_rep_start_time = 0.0
        self.current_bottom_frame = 0
        self.current_bottom_time = 0.0
        self.min_knee_angle = 180.0
        self.max_knee_angle = 0.0

        self.knee_angle_series: List[float] = []
        self.frames_series: List[int] = []
        self.frame_idx = 0

    def reset(self):
        self.phase = SquatPhase.STANDING
        self.total_reps = 0
        self.full_depth_reps = 0
        self.shallow_reps = 0
        self.rep_history.clear()
        self.knee_angle_series.clear()
        self.frames_series.clear()
        self.frame_idx = 0

    def update(self, landmarks: Dict[str, Point], timestamp: Optional[float] = None) -> Tuple[SquatPhase, Optional[SquatRepDetail]]:
        self.frame_idx += 1
        t = timestamp if timestamp is not None else (self.frame_idx / self.fps)

        # Detect better visible side
        left_vis = (landmarks.get("LEFT_HIP", Point(0, 0, 0, 0)).visibility +
                    landmarks.get("LEFT_KNEE", Point(0, 0, 0, 0)).visibility)
        right_vis = (landmarks.get("RIGHT_HIP", Point(0, 0, 0, 0)).visibility +
                     landmarks.get("RIGHT_KNEE", Point(0, 0, 0, 0)).visibility)
        side = "LEFT" if left_vis >= right_vis else "RIGHT"

        hip = landmarks.get(f"{side}_HIP")
        knee = landmarks.get(f"{side}_KNEE")
        ankle = landmarks.get(f"{side}_ANKLE")

        if not all(p and p.visibility > 0.4 for p in [hip, knee, ankle]):
            return self.phase, None

        knee_angle = PoseDetector.calculate_angle(hip, knee, ankle)
        self.knee_angle_series.append(knee_angle)
        self.frames_series.append(self.frame_idx)

        completed_rep: Optional[SquatRepDetail] = None

        if self.phase == SquatPhase.STANDING:
            if knee_angle < self.standing_knee_threshold - 10.0:
                self.phase = SquatPhase.DESCENDING
                self.current_rep_start_frame = self.frame_idx
                self.current_rep_start_time = t
                self.min_knee_angle = knee_angle
                self.max_knee_angle = knee_angle

        elif self.phase == SquatPhase.DESCENDING:
            self.min_knee_angle = min(self.min_knee_angle, knee_angle)

            # Reached lowest point when angle begins turning back upwards
            if knee_angle > self.min_knee_angle + 10.0 and self.min_knee_angle < 120.0:
                self.phase = SquatPhase.ASCENDING
                self.current_bottom_frame = self.frame_idx
                self.current_bottom_time = t

        if self.phase == SquatPhase.ASCENDING:
            self.max_knee_angle = max(self.max_knee_angle, knee_angle)

            # Lockout at top standing position
            if knee_angle >= self.standing_knee_threshold:
                self.total_reps += 1
                descent_dur = max(0.1, self.current_bottom_time - self.current_rep_start_time)
                ascent_dur = max(0.1, t - self.current_bottom_time)
                total_dur = descent_dur + ascent_dur

                if self.min_knee_angle <= self.deep_knee_threshold:
                    depth = SquatDepth.FULL
                    feedback = "Great depth! Well below parallel."
                    self.full_depth_reps += 1
                elif self.min_knee_angle <= self.parallel_knee_threshold:
                    depth = SquatDepth.PARALLEL
                    feedback = "Solid parallel depth achieved."
                    self.full_depth_reps += 1
                else:
                    depth = SquatDepth.SHALLOW
                    feedback = f"Shallow squat: knee reached only {int(self.min_knee_angle)}°. Aim for 90° or lower."
                    self.shallow_reps += 1

                completed_rep = SquatRepDetail(
                    rep_number=self.total_reps,
                    depth=depth,
                    feedback=feedback,
                    min_knee_angle=round(self.min_knee_angle, 1),
                    max_knee_angle=round(self.max_knee_angle, 1),
                    descent_time=round(descent_dur, 2),
                    ascent_time=round(ascent_dur, 2),
                    total_time=round(total_dur, 2),
                    start_frame=self.current_rep_start_frame,
                    bottom_frame=self.current_bottom_frame,
                    end_frame=self.frame_idx,
                )
                self.rep_history.append(completed_rep)

                # Reset
                self.phase = SquatPhase.STANDING
                self.min_knee_angle = 180.0
                self.max_knee_angle = 0.0

        return self.phase, completed_rep
