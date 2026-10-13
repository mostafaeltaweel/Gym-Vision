"""Unit tests for Gym Vision AI biomechanics analyzers."""

import pytest
from src.pose_detector import Point, PoseDetector
from src.pullup_analyzer import PullUpAnalyzer, PullUpPhase, RepGrade
from src.deadlift_analyzer import DeadliftAnalyzer, BackForm, DeadliftPhase
from src.squat_analyzer import SquatAnalyzer, SquatDepth, SquatPhase


def test_angle_calculation_right_angle():
    a = Point(x=0.0, y=1.0)
    b = Point(x=0.0, y=0.0)  # vertex
    c = Point(x=1.0, y=0.0)
    angle = PoseDetector.calculate_angle(a, b, c)
    assert abs(angle - 90.0) < 1e-4


def test_angle_calculation_straight_line():
    a = Point(x=0.0, y=1.0)
    b = Point(x=0.0, y=0.0)
    c = Point(x=0.0, y=-1.0)
    angle = PoseDetector.calculate_angle(a, b, c)
    assert abs(angle - 180.0) < 1e-4


def test_angle_calculation_acute():
    a = Point(x=0.0, y=1.0)
    b = Point(x=0.0, y=0.0)
    c = Point(x=1.0, y=1.0)
    angle = PoseDetector.calculate_angle(a, b, c)
    assert abs(angle - 45.0) < 1e-4


def test_pullup_analyzer_full_rep():
    analyzer = PullUpAnalyzer(hang_angle_threshold=150.0, chin_clearance_ratio=0.05, fps=30.0)
    
    # Simulate a full rep trajectory:
    # 1. Dead hang: arms extended (160°), hands high, torso low
    hang_lms = {
        "LEFT_SHOULDER": Point(x=0.45, y=0.4, visibility=0.9),
        "RIGHT_SHOULDER": Point(x=0.55, y=0.4, visibility=0.9),
        "LEFT_ELBOW": Point(x=0.45, y=0.25, visibility=0.9),
        "RIGHT_ELBOW": Point(x=0.55, y=0.25, visibility=0.9),
        "LEFT_WRIST": Point(x=0.45, y=0.1, visibility=0.9),
        "RIGHT_WRIST": Point(x=0.55, y=0.1, visibility=0.9),
        "LEFT_HIP": Point(x=0.45, y=0.7, visibility=0.9),
        "RIGHT_HIP": Point(x=0.55, y=0.7, visibility=0.9),
        "NOSE": Point(x=0.5, y=0.35, visibility=0.9),
    }
    
    # Update with hang
    analyzer.update(hang_lms, timestamp=0.1)
    assert analyzer.phase == PullUpPhase.HANG

    # 2. Pull up to peak: elbows flexed (80°), head cleared hands
    peak_lms = {
        "LEFT_SHOULDER": Point(x=0.45, y=0.18, visibility=0.9),
        "RIGHT_SHOULDER": Point(x=0.55, y=0.18, visibility=0.9),
        "LEFT_ELBOW": Point(x=0.40, y=0.22, visibility=0.9),
        "RIGHT_ELBOW": Point(x=0.60, y=0.22, visibility=0.9),
        "LEFT_WRIST": Point(x=0.45, y=0.1, visibility=0.9),
        "RIGHT_WRIST": Point(x=0.55, y=0.1, visibility=0.9),
        "LEFT_HIP": Point(x=0.45, y=0.48, visibility=0.9),
        "RIGHT_HIP": Point(x=0.55, y=0.48, visibility=0.9),
        "NOSE": Point(x=0.5, y=0.05, visibility=0.9),  # Head clears hands (0.05 < 0.1)
    }
    analyzer.update(peak_lms, timestamp=1.0)
    assert analyzer.phase in [PullUpPhase.ASCENT, PullUpPhase.PEAK]

    # Open arms slightly to trigger peak -> descent
    peak_open_lms = dict(peak_lms)
    peak_open_lms["LEFT_ELBOW"] = Point(x=0.40, y=0.24, visibility=0.9)
    peak_open_lms["RIGHT_ELBOW"] = Point(x=0.60, y=0.24, visibility=0.9)
    analyzer.update(peak_open_lms, timestamp=1.5)

    # 3. Return to dead hang
    _, completed_rep = analyzer.update(hang_lms, timestamp=2.5)
    assert completed_rep is not None
    assert analyzer.total_reps == 1
    assert completed_rep.grade == RepGrade.FULL


def test_squat_analyzer_full_depth():
    analyzer = SquatAnalyzer(standing_knee_threshold=160.0, parallel_knee_threshold=90.0, fps=30.0)

    # Standing position (knee 170°)
    standing_lms = {
        "LEFT_HIP": Point(x=0.5, y=0.4, visibility=0.9),
        "LEFT_KNEE": Point(x=0.5, y=0.65, visibility=0.9),
        "LEFT_ANKLE": Point(x=0.5, y=0.9, visibility=0.9),
    }
    analyzer.update(standing_lms, timestamp=0.1)
    assert analyzer.phase == SquatPhase.STANDING

    # Squat bottom (knee angle ~ 85°)
    bottom_lms = {
        "LEFT_HIP": Point(x=0.4, y=0.65, visibility=0.9),
        "LEFT_KNEE": Point(x=0.55, y=0.66, visibility=0.9),
        "LEFT_ANKLE": Point(x=0.5, y=0.9, visibility=0.9),
    }
    analyzer.update(bottom_lms, timestamp=1.2)
    assert analyzer.phase in [SquatPhase.DESCENDING, SquatPhase.BOTTOM]

    # Begin ascending
    ascend_lms = {
        "LEFT_HIP": Point(x=0.45, y=0.55, visibility=0.9),
        "LEFT_KNEE": Point(x=0.52, y=0.65, visibility=0.9),
        "LEFT_ANKLE": Point(x=0.5, y=0.9, visibility=0.9),
    }
    analyzer.update(ascend_lms, timestamp=1.8)

    # Return to standing
    _, completed_rep = analyzer.update(standing_lms, timestamp=2.5)
    assert completed_rep is not None
    assert analyzer.total_reps == 1
    assert completed_rep.depth in [SquatDepth.FULL, SquatDepth.PARALLEL]
