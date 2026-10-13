"""Deadlift Biomechanical Rep & Back Posture Form Analyzer."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from .pose_detector import Point, PoseDetector


class DeadliftPhase(str, Enum):
    SETUP = "SETUP"
    LIFTING = "LIFTING"
    LOCKOUT = "LOCKOUT"
    LOWERING = "LOWERING"


class BackForm(str, Enum):
    STRAIGHT = "STRAIGHT"
    ROUNDED = "ROUNDED"


@dataclass
class DeadliftRepDetail:
    rep_number: int
    back_form: BackForm
    lockout_achieved: bool
    feedback: str
    min_hip_angle: float
    max_hip_angle: float
    lift_time: float
    lower_time: float
    total_time: float
    start_frame: int
    lockout_frame: int
    end_frame: int


class DeadliftAnalyzer:
    """Biomechanical counter and back-rounding detector for Deadlifts."""

    def __init__(
        self,
        lockout_hip_threshold: float = 165.0,
        hinge_start_threshold: float = 135.0,
        spine_curvature_threshold: float = 25.0,  # Max allowable spinal deviation
        fps: float = 30.0,
    ):
        self.lockout_hip_threshold = lockout_hip_threshold
        self.hinge_start_threshold = hinge_start_threshold
        self.spine_curvature_threshold = spine_curvature_threshold
        self.fps = fps

        self.phase = DeadliftPhase.SETUP
        self.total_reps = 0
        self.straight_reps = 0
        self.rounded_reps = 0
        self.rep_history: List[DeadliftRepDetail] = []

        # Current rep metrics
        self.current_rep_start_frame = 0
        self.current_rep_start_time = 0.0
        self.current_lockout_frame = 0
        self.current_lockout_time = 0.0
        self.min_hip_angle = 180.0
        self.max_hip_angle = 0.0
        self.max_back_curvature = 0.0

        # Time series telemetry
        self.hip_angle_series: List[float] = []
        self.back_curvature_series: List[float] = []
        self.frames_series: List[int] = []

        self.frame_idx = 0

    def reset(self):
        self.phase = DeadliftPhase.SETUP
        self.total_reps = 0
        self.straight_reps = 0
        self.rounded_reps = 0
        self.rep_history.clear()
        self.hip_angle_series.clear()
        self.back_curvature_series.clear()
        self.frames_series.clear()
        self.frame_idx = 0

    def update(self, landmarks: Dict[str, Point], timestamp: Optional[float] = None) -> Tuple[DeadliftPhase, Optional[DeadliftRepDetail]]:
        self.frame_idx += 1
        t = timestamp if timestamp is not None else (self.frame_idx / self.fps)

        # Use whichever side has better visibility (left or right side-on view)
        left_vis = (landmarks.get("LEFT_HIP", Point(0, 0, 0, 0)).visibility +
                    landmarks.get("LEFT_KNEE", Point(0, 0, 0, 0)).visibility)
        right_vis = (landmarks.get("RIGHT_HIP", Point(0, 0, 0, 0)).visibility +
                     landmarks.get("RIGHT_KNEE", Point(0, 0, 0, 0)).visibility)
        side = "LEFT" if left_vis >= right_vis else "RIGHT"

        shoulder = landmarks.get(f"{side}_SHOULDER")
        hip = landmarks.get(f"{side}_HIP")
        knee = landmarks.get(f"{side}_KNEE")
        ankle = landmarks.get(f"{side}_ANKLE")
        ear = landmarks.get(f"{side}_EAR")

        if not all(p and p.visibility > 0.4 for p in [shoulder, hip, knee, ankle]):
            return self.phase, None

        # 1. Hip hinge angle (Shoulder - Hip - Knee)
        hip_angle = PoseDetector.calculate_angle(shoulder, hip, knee)
        knee_angle = PoseDetector.calculate_angle(hip, knee, ankle)
        self.hip_angle_series.append(hip_angle)

        # 2. Back curvature estimate (deviation between Ear-Shoulder line and Shoulder-Hip line)
        if ear and ear.visibility > 0.4:
            neck_back_angle = PoseDetector.calculate_angle(ear, shoulder, hip)
            curvature = abs(180.0 - neck_back_angle)
        else:
            curvature = 0.0

        self.back_curvature_series.append(curvature)
        self.frames_series.append(self.frame_idx)

        completed_rep: Optional[DeadliftRepDetail] = None

        # 3. State Machine transitions
        if self.phase == DeadliftPhase.SETUP:
            # At bottom hinge position (hips flexed below threshold)
            if hip_angle < self.hinge_start_threshold:
                self.min_hip_angle = min(self.min_hip_angle, hip_angle)
                self.phase = DeadliftPhase.LIFTING
                self.current_rep_start_frame = self.frame_idx
                self.current_rep_start_time = t
                self.max_back_curvature = curvature

        elif self.phase == DeadliftPhase.LIFTING:
            self.max_hip_angle = max(self.max_hip_angle, hip_angle)
            self.max_back_curvature = max(self.max_back_curvature, curvature)

            # Check for lockout (hips and knees fully extended)
            if hip_angle >= self.lockout_hip_threshold and knee_angle >= 160.0:
                self.phase = DeadliftPhase.LOCKOUT
                self.current_lockout_frame = self.frame_idx
                self.current_lockout_time = t

        elif self.phase == DeadliftPhase.LOCKOUT:
            # Athlete begins descending / lowering barbell
            if hip_angle < self.lockout_hip_threshold - 10.0:
                self.phase = DeadliftPhase.LOWERING

        elif self.phase == DeadliftPhase.LOWERING:
            # Rep completes when returned to bottom setup hinge
            if hip_angle < self.hinge_start_threshold:
                self.total_reps += 1
                lift_dur = max(0.1, self.current_lockout_time - self.current_rep_start_time)
                lower_dur = max(0.1, t - self.current_lockout_time)
                total_dur = lift_dur + lower_dur

                # Form assessment
                is_rounded = self.max_back_curvature > self.spine_curvature_threshold
                lockout_ok = self.max_hip_angle >= self.lockout_hip_threshold

                if is_rounded:
                    back_form = BackForm.ROUNDED
                    feedback = f"Warning: Excessive back rounding detected ({int(self.max_back_curvature)}° curvature). Keep lats engaged."
                    self.rounded_reps += 1
                else:
                    back_form = BackForm.STRAIGHT
                    feedback = "Excellent flat neutral back through the entire pull."
                    self.straight_reps += 1

                completed_rep = DeadliftRepDetail(
                    rep_number=self.total_reps,
                    back_form=back_form,
                    lockout_achieved=lockout_ok,
                    feedback=feedback,
                    min_hip_angle=round(self.min_hip_angle, 1),
                    max_hip_angle=round(self.max_hip_angle, 1),
                    lift_time=round(lift_dur, 2),
                    lower_time=round(lower_dur, 2),
                    total_time=round(total_dur, 2),
                    start_frame=self.current_rep_start_frame,
                    lockout_frame=self.current_lockout_frame,
                    end_frame=self.frame_idx,
                )
                self.rep_history.append(completed_rep)

                # Reset for next rep
                self.phase = DeadliftPhase.SETUP
                self.min_hip_angle = 180.0
                self.max_hip_angle = 0.0
                self.max_back_curvature = 0.0

        return self.phase, completed_rep
