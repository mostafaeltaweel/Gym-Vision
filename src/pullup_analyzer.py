"""Pull-Up and Chin-Up Biomechanical Rep & Form Analyzer."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from .pose_detector import Point, PoseDetector


class RepGrade(str, Enum):
    FULL = "FULL"
    PARTIAL = "PARTIAL"


class PullUpPhase(str, Enum):
    IDLE = "IDLE"
    HANG = "HANG"
    ASCENT = "ASCENT"
    PEAK = "PEAK"
    DESCENT = "DESCENT"


@dataclass
class RepDetail:
    rep_number: int
    grade: RepGrade
    feedback: str
    bottom_elbow_angle: float
    top_elbow_angle: float
    head_clearance: float  # Positive means chin/head above hands
    ascent_time: float
    descent_time: float
    total_time: float
    start_frame: int
    peak_frame: int
    end_frame: int


class PullUpAnalyzer:
    """Biomechanical counter and form grader for Pull-ups and Chin-ups."""

    def __init__(
        self,
        hang_angle_threshold: float = 150.0,
        chin_clearance_ratio: float = 0.05,
        elbow_peak_threshold: float = 90.0,
        fps: float = 30.0,
    ):
        self.hang_angle_threshold = hang_angle_threshold
        self.chin_clearance_ratio = chin_clearance_ratio
        self.elbow_peak_threshold = elbow_peak_threshold
        self.fps = fps

        # State tracking
        self.phase = PullUpPhase.IDLE
        self.total_reps = 0
        self.full_reps = 0
        self.partial_reps = 0
        self.rep_history: List[RepDetail] = []

        # Current rep tracking variables
        self.current_rep_start_frame = 0
        self.current_rep_start_time = 0.0
        self.current_peak_frame = 0
        self.current_peak_time = 0.0
        self.min_elbow_angle = 180.0
        self.max_elbow_angle = 0.0
        self.max_head_clearance = -1.0
        self.achieved_hang = False
        self.achieved_top = False

        # Time series telemetry
        self.displacement_series: List[float] = []
        self.elbow_series: List[float] = []
        self.frames_series: List[int] = []

        self.frame_idx = 0

    def reset(self):
        """Reset state for a new session."""
        self.phase = PullUpPhase.IDLE
        self.total_reps = 0
        self.full_reps = 0
        self.partial_reps = 0
        self.rep_history.clear()
        self.displacement_series.clear()
        self.elbow_series.clear()
        self.frames_series.clear()
        self.frame_idx = 0

    def update(self, landmarks: Dict[str, Point], timestamp: Optional[float] = None) -> Tuple[PullUpPhase, Optional[RepDetail]]:
        """Process a frame's landmarks and update biomechanical rep state."""
        self.frame_idx += 1
        t = timestamp if timestamp is not None else (self.frame_idx / self.fps)

        # Check required landmarks visibility
        required = [
            "LEFT_SHOULDER", "RIGHT_SHOULDER",
            "LEFT_ELBOW", "RIGHT_ELBOW",
            "LEFT_WRIST", "RIGHT_WRIST",
            "LEFT_HIP", "RIGHT_HIP",
        ]
        if not all(name in landmarks and landmarks[name].visibility > 0.4 for name in required):
            return self.phase, None

        # 1. Calculate joint angles
        left_elbow = PoseDetector.calculate_angle(
            landmarks["LEFT_SHOULDER"], landmarks["LEFT_ELBOW"], landmarks["LEFT_WRIST"]
        )
        right_elbow = PoseDetector.calculate_angle(
            landmarks["RIGHT_SHOULDER"], landmarks["RIGHT_ELBOW"], landmarks["RIGHT_WRIST"]
        )
        avg_elbow = (left_elbow + right_elbow) / 2.0
        self.elbow_series.append(avg_elbow)

        # 2. Torso and Hands coordinates (in normalized Y: 0 is top, 1 is bottom)
        hands_y = (landmarks["LEFT_WRIST"].y + landmarks["RIGHT_WRIST"].y) / 2.0
        shoulders_y = (landmarks["LEFT_SHOULDER"].y + landmarks["RIGHT_SHOULDER"].y) / 2.0
        hips_y = (landmarks["LEFT_HIP"].y + landmarks["RIGHT_HIP"].y) / 2.0
        torso_y = (shoulders_y + hips_y) / 2.0

        # Torso length for scale invariance
        torso_len = abs(hips_y - shoulders_y)
        if torso_len < 0.05:
            torso_len = 0.2

        # Relative height displacement: Higher is further up towards/above bar
        # hands_y - shoulders_y: when pulling up, shoulders get closer to or above hands
        displacement = (hands_y - shoulders_y) / torso_len
        self.displacement_series.append(displacement)
        self.frames_series.append(self.frame_idx)

        # Head clearance relative to hands (positive means head/chin above hands)
        head_y = landmarks["NOSE"].y
        head_clearance = (hands_y - head_y) / torso_len

        completed_rep: Optional[RepDetail] = None

        # 3. State Machine transitions
        if self.phase == PullUpPhase.IDLE:
            # Check if athlete is in dead hang (elbows extended)
            if avg_elbow >= self.hang_angle_threshold:
                self.phase = PullUpPhase.HANG
                self.achieved_hang = True
                self.max_elbow_angle = avg_elbow

        elif self.phase == PullUpPhase.HANG:
            self.max_elbow_angle = max(self.max_elbow_angle, avg_elbow)
            if avg_elbow >= self.hang_angle_threshold:
                self.achieved_hang = True

            # If athlete begins pulling up (elbow flexes below 140° and displacement increases)
            if avg_elbow < 140.0:
                self.phase = PullUpPhase.ASCENT
                self.current_rep_start_frame = self.frame_idx
                self.current_rep_start_time = t
                self.min_elbow_angle = avg_elbow
                self.max_head_clearance = head_clearance

        elif self.phase == PullUpPhase.ASCENT:
            self.min_elbow_angle = min(self.min_elbow_angle, avg_elbow)
            self.max_head_clearance = max(self.max_head_clearance, head_clearance)

            # Check if reached peak: elbow flexion peak or head clearing bar
            if head_clearance >= self.chin_clearance_ratio or avg_elbow <= self.elbow_peak_threshold:
                self.achieved_top = True

            # When elbow angle starts opening again after reaching peak flexion
            if avg_elbow > self.min_elbow_angle + 15.0 and self.min_elbow_angle < 110.0:
                self.phase = PullUpPhase.DESCENT
                self.current_peak_frame = self.frame_idx
                self.current_peak_time = t

        if self.phase == PullUpPhase.DESCENT:
            self.max_elbow_angle = max(self.max_elbow_angle, avg_elbow)

            # Rep closes when arms extend back towards dead hang (> 145°) or bottom plateau
            if avg_elbow >= 145.0 or (avg_elbow >= self.hang_angle_threshold - 10.0):
                # Rep complete!
                self.total_reps += 1
                ascent_dur = max(0.1, self.current_peak_time - self.current_rep_start_time)
                descent_dur = max(0.1, t - self.current_peak_time)
                total_dur = ascent_dur + descent_dur

                # Determine Grade & Feedback
                feedbacks = []
                passed_bottom = self.achieved_hang or (self.max_elbow_angle >= self.hang_angle_threshold)
                passed_top = self.achieved_top or (self.max_head_clearance >= self.chin_clearance_ratio)

                if not passed_bottom:
                    feedbacks.append(f"Elbows didn't reach dead hang ({int(self.max_elbow_angle)}° < {int(self.hang_angle_threshold)}°)")
                if not passed_top:
                    feedbacks.append(f"Chin didn't clear the bar ({self.max_head_clearance:.2f} < {self.chin_clearance_ratio:.2f})")

                if passed_bottom and passed_top:
                    grade = RepGrade.FULL
                    feedback_str = "Full Range of Motion! Clean rep."
                    self.full_reps += 1
                else:
                    grade = RepGrade.PARTIAL
                    feedback_str = "Partial Rep: " + "; ".join(feedbacks)
                    self.partial_reps += 1

                completed_rep = RepDetail(
                    rep_number=self.total_reps,
                    grade=grade,
                    feedback=feedback_str,
                    bottom_elbow_angle=round(self.max_elbow_angle, 1),
                    top_elbow_angle=round(self.min_elbow_angle, 1),
                    head_clearance=round(self.max_head_clearance, 2),
                    ascent_time=round(ascent_dur, 2),
                    descent_time=round(descent_dur, 2),
                    total_time=round(total_dur, 2),
                    start_frame=self.current_rep_start_frame,
                    peak_frame=self.current_peak_frame,
                    end_frame=self.frame_idx,
                )
                self.rep_history.append(completed_rep)

                # Reset for next rep
                self.phase = PullUpPhase.HANG
                self.achieved_hang = (avg_elbow >= self.hang_angle_threshold)
                self.achieved_top = False
                self.max_elbow_angle = avg_elbow
                self.min_elbow_angle = 180.0
                self.max_head_clearance = -1.0

        return self.phase, completed_rep
