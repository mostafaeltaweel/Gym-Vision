"""Gym Vision AI - Streamlit Web Application."""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.deadlift_analyzer import BackForm, DeadliftAnalyzer, DeadliftPhase
from src.pose_detector import PoseDetector
from src.pullup_analyzer import PullUpAnalyzer, PullUpPhase, RepGrade
from src.squat_analyzer import SquatAnalyzer, SquatDepth, SquatPhase
from src.visualizer import GymVisualizer

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Gym Vision AI - Form & Reps Tracker",
    page_icon="🏋️‍♂️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #1e2130;
        border: 1px solid #2e334d;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        margin-bottom: 12px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #00d26a;
    }
    .metric-label {
        font-size: 14px;
        color: #9aa0a6;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 16px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_sidebar():
    st.sidebar.title("🏋️‍♂️ Gym Vision AI")
    st.sidebar.caption("Biomechanical Computer Vision Workout Tracker")
    st.sidebar.markdown("---")

    exercise = st.sidebar.selectbox(
        "Select Exercise",
        ["Pull-Ups / Chin-Ups", "Deadlift", "Squats"],
        index=0,
    )

    mode = st.sidebar.radio(
        "Input Mode",
        ["Video File Upload", "Live Webcam"],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Biomechanics Thresholds")

    params = {}
    if exercise == "Pull-Ups / Chin-Ups":
        params["hang_angle"] = st.sidebar.slider(
            "Dead-Hang Elbow Angle (deg)",
            min_value=140.0,
            max_value=175.0,
            value=150.0,
            step=1.0,
            help="Minimum elbow angle required at the bottom to count as a full dead hang.",
        )
        params["chin_clearance"] = st.sidebar.slider(
            "Chin Clearance Threshold",
            min_value=0.0,
            max_value=0.20,
            value=0.05,
            step=0.01,
            help="Torso fraction the head must clear above the hands at the peak.",
        )
        params["elbow_peak"] = st.sidebar.slider(
            "Top Elbow Flexion (deg)",
            min_value=60.0,
            max_value=110.0,
            value=90.0,
            step=1.0,
            help="Maximum elbow angle at the peak of the pull.",
        )

    elif exercise == "Deadlift":
        params["lockout_hip"] = st.sidebar.slider(
            "Lockout Hip Angle (deg)",
            min_value=155.0,
            max_value=178.0,
            value=165.0,
            step=1.0,
            help="Hip extension angle required to complete lockout.",
        )
        params["curvature_thresh"] = st.sidebar.slider(
            "Max Back Curvature Tolerance (deg)",
            min_value=15.0,
            max_value=40.0,
            value=25.0,
            step=1.0,
            help="Deviation from neutral spine before flagging rounded back.",
        )

    elif exercise == "Squats":
        params["standing_knee"] = st.sidebar.slider(
            "Standing Knee Angle (deg)",
            min_value=150.0,
            max_value=175.0,
            value=160.0,
            step=1.0,
        )
        params["parallel_knee"] = st.sidebar.slider(
            "Parallel Depth Angle (deg)",
            min_value=80.0,
            max_value=100.0,
            value=90.0,
            step=1.0,
            help="Knee angle threshold for parallel depth.",
        )

    return exercise, mode, params


def process_video_file(video_path: str, exercise: str, params: dict, status_placeholder, progress_bar, preview_placeholder):
    """Processes a video file frame-by-frame and produces an annotated video and telemetry data."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error opening video file.")
        return None, None, None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize Detector & Analyzer
    detector = PoseDetector(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    if exercise == "Pull-Ups / Chin-Ups":
        analyzer = PullUpAnalyzer(
            hang_angle_threshold=params.get("hang_angle", 150.0),
            chin_clearance_ratio=params.get("chin_clearance", 0.05),
            elbow_peak_threshold=params.get("elbow_peak", 90.0),
            fps=fps,
        )
    elif exercise == "Deadlift":
        analyzer = DeadliftAnalyzer(
            lockout_hip_threshold=params.get("lockout_hip", 165.0),
            spine_curvature_threshold=params.get("curvature_thresh", 25.0),
            fps=fps,
        )
    else:  # Squats
        analyzer = SquatAnalyzer(
            standing_knee_threshold=params.get("standing_knee", 160.0),
            parallel_knee_threshold=params.get("parallel_knee", 90.0),
            fps=fps,
        )

    # Temporary output file
    output_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    output_path = output_temp.name
    output_temp.close()

    # OpenCV VideoWriter with mp4v codec
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    recent_feedback = "Ready to start"
    is_good = True

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        t = frame_idx / fps

        # Pose Estimation
        norm_lms, pixel_lms, _ = detector.find_pose(frame)

        if pixel_lms:
            # Draw Skeleton
            GymVisualizer.draw_skeleton(frame, pixel_lms)

            # Update Biomechanics
            if exercise == "Pull-Ups / Chin-Ups":
                phase, rep = analyzer.update(norm_lms, timestamp=t)
                if rep:
                    recent_feedback = rep.feedback
                    is_good = (rep.grade == RepGrade.FULL)

                left_e = PoseDetector.calculate_angle(pixel_lms["LEFT_SHOULDER"], pixel_lms["LEFT_ELBOW"], pixel_lms["LEFT_WRIST"])
                GymVisualizer.draw_joint_angle(frame, pixel_lms["LEFT_ELBOW"], left_e, label="Elbow")

                GymVisualizer.draw_hud(
                    frame,
                    exercise_name="Pull-Ups",
                    reps_count=analyzer.total_reps,
                    primary_metric_label="Elbow Angle",
                    primary_metric_value=f"{int(left_e)} deg",
                    phase_label=phase.value,
                    status_feedback=recent_feedback,
                    is_good_form=is_good,
                )

            elif exercise == "Deadlift":
                phase, rep = analyzer.update(norm_lms, timestamp=t)
                if rep:
                    recent_feedback = rep.feedback
                    is_good = (rep.back_form == BackForm.STRAIGHT)

                hip_val = analyzer.hip_angle_series[-1] if analyzer.hip_angle_series else 180.0
                back_val = analyzer.back_curvature_series[-1] if analyzer.back_curvature_series else 0.0

                GymVisualizer.draw_hud(
                    frame,
                    exercise_name="Deadlift",
                    reps_count=analyzer.total_reps,
                    primary_metric_label="Spine Flexion",
                    primary_metric_value=f"{int(back_val)} deg",
                    phase_label=phase.value,
                    status_feedback=recent_feedback,
                    is_good_form=(back_val <= params.get("curvature_thresh", 25.0)),
                )

            else:  # Squats
                phase, rep = analyzer.update(norm_lms, timestamp=t)
                if rep:
                    recent_feedback = rep.feedback
                    is_good = (rep.depth != SquatDepth.SHALLOW)

                knee_val = analyzer.knee_angle_series[-1] if analyzer.knee_angle_series else 180.0

                GymVisualizer.draw_hud(
                    frame,
                    exercise_name="Squats",
                    reps_count=analyzer.total_reps,
                    primary_metric_label="Knee Angle",
                    primary_metric_value=f"{int(knee_val)} deg",
                    phase_label=phase.value,
                    status_feedback=recent_feedback,
                    is_good_form=is_good,
                )

        out.write(frame)

        # Update progress in UI
        if frame_idx % 15 == 0 or frame_idx == total_frames:
            prog = min(1.0, frame_idx / total_frames)
            progress_bar.progress(prog)
            status_placeholder.text(f"Analyzing frame {frame_idx} of {total_frames} ({int(prog * 100)}%)...")
            # Show live thumbnail every 30 frames
            if frame_idx % 30 == 0:
                rgb_preview = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                preview_placeholder.image(rgb_preview, caption=f"Processing Preview (Frame {frame_idx})", use_container_width=True)

    cap.release()
    out.release()
    detector.close()

    progress_bar.progress(1.0)
    status_placeholder.success("Processing complete!")

    return output_path, analyzer, fps


def main():
    exercise, mode, params = render_sidebar()

    st.title("🏋️‍♂️ Gym Vision AI")
    st.markdown(
        f"**Active Exercise:** `{exercise}` | **Mode:** `{mode}` | **Powered by MediaPipe Pose Engine**"
    )

    tab_video, tab_webcam, tab_guide = st.tabs(["📁 Video Analysis", "📹 Live Webcam Tracker", "📖 Biomechanics Guide"])

    # ------------------ TAB 1: VIDEO ANALYSIS ------------------
    with tab_video:
        uploaded_file = st.file_uploader(
            "Upload Exercise Clip (MP4, MOV, AVI)",
            type=["mp4", "mov", "avi"],
            help="Film side-on or front-on with the athlete and bar/weights clearly visible.",
        )

        if uploaded_file is not None:
            col_orig, col_proc = st.columns(2)
            with col_orig:
                st.subheader("Source Video")
                st.video(uploaded_file)

            # Save uploaded video to temp
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tfile:
                tfile.write(uploaded_file.read())
                temp_video_path = tfile.name

            if st.button("🚀 Run Biomechanics & Rep Analysis", type="primary"):
                progress_bar = st.progress(0.0)
                status_placeholder = st.empty()
                preview_placeholder = st.empty()

                start_time = time.time()
                output_video_path, analyzer, fps = process_video_file(
                    temp_video_path, exercise, params, status_placeholder, progress_bar, preview_placeholder
                )
                elapsed = time.time() - start_time

                if output_video_path and os.path.exists(output_video_path):
                    with col_proc:
                        st.subheader("Annotated Video & HUD")
                        st.video(output_video_path)
                        with open(output_video_path, "rb") as vf:
                            st.download_button(
                                label="⬇️ Download Annotated Video",
                                data=vf,
                                file_name=f"annotated_{uploaded_file.name}",
                                mime="video/mp4",
                            )

                    st.markdown("---")
                    st.subheader("📊 Session Executive Summary")

                    # Metrics row
                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.metric("Total Reps", analyzer.total_reps)
                    with m2:
                        if exercise == "Pull-Ups / Chin-Ups":
                            st.metric("Full ROM Reps", analyzer.full_reps)
                        elif exercise == "Deadlift":
                            st.metric("Straight Back Reps", analyzer.straight_reps)
                        else:
                            st.metric("Full Depth Reps", analyzer.full_depth_reps)
                    with m3:
                        if exercise == "Pull-Ups / Chin-Ups":
                            st.metric("Partial Reps", analyzer.partial_reps)
                        elif exercise == "Deadlift":
                            st.metric("Rounded Back Reps", analyzer.rounded_reps)
                        else:
                            st.metric("Shallow Reps", analyzer.shallow_reps)
                    with m4:
                        rate = (
                            round((analyzer.full_reps / analyzer.total_reps) * 100, 1)
                            if exercise == "Pull-Ups / Chin-Ups" and analyzer.total_reps > 0
                            else (
                                round((analyzer.straight_reps / analyzer.total_reps) * 100, 1)
                                if exercise == "Deadlift" and analyzer.total_reps > 0
                                else (
                                    round((analyzer.full_depth_reps / analyzer.total_reps) * 100, 1)
                                    if analyzer.total_reps > 0
                                    else 0.0
                                )
                            )
                        )
                        st.metric("Form Success Rate", f"{rate}%")

                    # Reps Breakdown Table
                    st.subheader("📋 Rep-by-Rep Breakdown")
                    if analyzer.rep_history:
                        rep_records = []
                        for r in analyzer.rep_history:
                            if exercise == "Pull-Ups / Chin-Ups":
                                rep_records.append({
                                    "Rep #": r.rep_number,
                                    "Grade": r.grade.value,
                                    "Bottom Elbow (deg)": r.bottom_elbow_angle,
                                    "Top Elbow (deg)": r.top_elbow_angle,
                                    "Head Clearance": r.head_clearance,
                                    "Ascent (s)": r.ascent_time,
                                    "Descent (s)": r.descent_time,
                                    "Total Time (s)": r.total_time,
                                    "Feedback": r.feedback,
                                })
                            elif exercise == "Deadlift":
                                rep_records.append({
                                    "Rep #": r.rep_number,
                                    "Back Form": r.back_form.value,
                                    "Lockout": "Yes" if r.lockout_achieved else "No",
                                    "Min Hip (deg)": r.min_hip_angle,
                                    "Lockout Hip (deg)": r.max_hip_angle,
                                    "Lift Time (s)": r.lift_time,
                                    "Lower Time (s)": r.lower_time,
                                    "Total Time (s)": r.total_time,
                                    "Feedback": r.feedback,
                                })
                            else:  # Squats
                                rep_records.append({
                                    "Rep #": r.rep_number,
                                    "Depth": r.depth.value,
                                    "Min Knee (deg)": r.min_knee_angle,
                                    "Descent (s)": r.descent_time,
                                    "Ascent (s)": r.ascent_time,
                                    "Total Time (s)": r.total_time,
                                    "Feedback": r.feedback,
                                })

                        df_reps = pd.DataFrame(rep_records)
                        st.dataframe(df_reps, use_container_width=True)

                        # CSV Export
                        csv_data = df_reps.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="⬇️ Export Rep Metrics (CSV)",
                            data=csv_data,
                            file_name=f"{exercise.lower().replace(' ', '_')}_metrics.csv",
                            mime="text/csv",
                        )
                    else:
                        st.info("No completed reps detected in this clip.")

                    # Interactive Plotly Visualizations
                    st.subheader("📈 Biomechanical Telemetry Over Time")
                    if exercise == "Pull-Ups / Chin-Ups" and analyzer.elbow_series:
                        time_axis = [f / fps for f in analyzer.frames_series]
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=time_axis,
                            y=analyzer.elbow_series,
                            mode="lines",
                            name="Elbow Flexion Angle",
                            line=dict(color="#00d26a", width=2),
                        ))
                        # Add dead-hang threshold line
                        fig.add_hline(
                            y=params.get("hang_angle", 150.0),
                            line_dash="dash",
                            line_color="yellow",
                            annotation_text="Dead-Hang Threshold",
                        )
                        fig.update_layout(
                            title="Elbow Angle (Dead-Hang vs Pull Peak)",
                            xaxis_title="Time (seconds)",
                            yaxis_title="Elbow Angle (degrees)",
                            template="plotly_dark",
                            height=380,
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    elif exercise == "Deadlift" and analyzer.hip_angle_series:
                        time_axis = [f / fps for f in analyzer.frames_series]
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=time_axis,
                            y=analyzer.hip_angle_series,
                            mode="lines",
                            name="Hip Hinge Angle",
                            line=dict(color="#00e5ff", width=2),
                        ))
                        fig.add_trace(go.Scatter(
                            x=time_axis,
                            y=analyzer.back_curvature_series,
                            mode="lines",
                            name="Spine Curvature (Deviation)",
                            line=dict(color="#ff5252", width=2),
                        ))
                        fig.add_hline(
                            y=params.get("curvature_thresh", 25.0),
                            line_dash="dash",
                            line_color="red",
                            annotation_text="Back Rounding Cutoff",
                        )
                        fig.update_layout(
                            title="Hip Hinge & Back Curvature",
                            xaxis_title="Time (seconds)",
                            yaxis_title="Degrees",
                            template="plotly_dark",
                            height=380,
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    elif exercise == "Squats" and analyzer.knee_angle_series:
                        time_axis = [f / fps for f in analyzer.frames_series]
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=time_axis,
                            y=analyzer.knee_angle_series,
                            mode="lines",
                            name="Knee Flexion Angle",
                            line=dict(color="#a855f7", width=2),
                        ))
                        fig.add_hline(
                            y=params.get("parallel_knee", 90.0),
                            line_dash="dash",
                            line_color="#00e5ff",
                            annotation_text="Parallel Depth (90°)",
                        )
                        fig.update_layout(
                            title="Knee Angle & Squat Depth Progression",
                            xaxis_title="Time (seconds)",
                            yaxis_title="Knee Angle (degrees)",
                            template="plotly_dark",
                            height=380,
                        )
                        st.plotly_chart(fig, use_container_width=True)

    # ------------------ TAB 2: LIVE WEBCAM TRACKER ------------------
    with tab_webcam:
        st.subheader("📹 Real-Time Live Webcam Workout")
        st.info("Allow camera permissions when prompted. Position camera 2-3 meters away so your entire body and exercise equipment are in frame.")

        col_cam_ctrl, col_cam_feed = st.columns([1, 3])

        with col_cam_ctrl:
            st.markdown("#### Live Session Control")
            run_cam = st.toggle("🔴 Start Live Camera Tracker", value=False)
            cam_idx = st.number_input("Camera Index", min_value=0, max_value=5, value=0)

            live_reps_ph = st.empty()
            live_phase_ph = st.empty()
            live_feedback_ph = st.empty()

        with col_cam_feed:
            feed_ph = st.empty()

            if run_cam:
                cap = cv2.VideoCapture(int(cam_idx))
                if not cap.isOpened():
                    st.error("Unable to open webcam. Please verify your camera connection or index.")
                else:
                    detector = PoseDetector(min_detection_confidence=0.5, min_tracking_confidence=0.5)
                    if exercise == "Pull-Ups / Chin-Ups":
                        analyzer = PullUpAnalyzer(
                            hang_angle_threshold=params.get("hang_angle", 150.0),
                            chin_clearance_ratio=params.get("chin_clearance", 0.05),
                            elbow_peak_threshold=params.get("elbow_peak", 90.0),
                        )
                    elif exercise == "Deadlift":
                        analyzer = DeadliftAnalyzer(
                            lockout_hip_threshold=params.get("lockout_hip", 165.0),
                            spine_curvature_threshold=params.get("curvature_thresh", 25.0),
                        )
                    else:
                        analyzer = SquatAnalyzer(
                            standing_knee_threshold=params.get("standing_knee", 160.0),
                            parallel_knee_threshold=params.get("parallel_knee", 90.0),
                        )

                    feedback_text = "Perform your rep"
                    is_good = True

                    while run_cam and cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            st.warning("Webcam feed disconnected.")
                            break

                        # Flip for mirror view
                        frame = cv2.flip(frame, 1)
                        norm_lms, pixel_lms, _ = detector.find_pose(frame)

                        if pixel_lms:
                            GymVisualizer.draw_skeleton(frame, pixel_lms)
                            if exercise == "Pull-Ups / Chin-Ups":
                                phase, rep = analyzer.update(norm_lms)
                                if rep:
                                    feedback_text = rep.feedback
                                    is_good = (rep.grade == RepGrade.FULL)

                                left_e = PoseDetector.calculate_angle(pixel_lms["LEFT_SHOULDER"], pixel_lms["LEFT_ELBOW"], pixel_lms["LEFT_WRIST"])
                                GymVisualizer.draw_hud(
                                    frame,
                                    "Pull-Ups",
                                    analyzer.total_reps,
                                    "Elbow Angle",
                                    f"{int(left_e)} deg",
                                    phase.value,
                                    feedback_text,
                                    is_good,
                                )

                            elif exercise == "Deadlift":
                                phase, rep = analyzer.update(norm_lms)
                                if rep:
                                    feedback_text = rep.feedback
                                    is_good = (rep.back_form == BackForm.STRAIGHT)

                                back_val = analyzer.back_curvature_series[-1] if analyzer.back_curvature_series else 0.0
                                GymVisualizer.draw_hud(
                                    frame,
                                    "Deadlift",
                                    analyzer.total_reps,
                                    "Spine Flexion",
                                    f"{int(back_val)} deg",
                                    phase.value,
                                    feedback_text,
                                    is_good,
                                )

                            else:
                                phase, rep = analyzer.update(norm_lms)
                                if rep:
                                    feedback_text = rep.feedback
                                    is_good = (rep.depth != SquatDepth.SHALLOW)

                                knee_val = analyzer.knee_angle_series[-1] if analyzer.knee_angle_series else 180.0
                                GymVisualizer.draw_hud(
                                    frame,
                                    "Squats",
                                    analyzer.total_reps,
                                    "Knee Angle",
                                    f"{int(knee_val)} deg",
                                    phase.value,
                                    feedback_text,
                                    is_good,
                                )

                            # Update side cards
                            live_reps_ph.metric("Live Reps", analyzer.total_reps)
                            live_phase_ph.metric("Current Phase", phase.value)
                            live_feedback_ph.info(f"Coach: {feedback_text}")

                        # Render frame to Streamlit
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        feed_ph.image(rgb_frame, channels="RGB", use_container_width=True)

                    cap.release()
                    detector.close()

    # ------------------ TAB 3: BIOMECHANICS GUIDE ------------------
    with tab_guide:
        st.subheader("📖 Biomechanics & Scientific Standards")
        st.markdown(
            """
            ### 1. Pull-Ups & Chin-Ups
            * **Dead-Hang Requirement**:
              At the bottom of each rep, the arms must reach full extension. A anatomical straight arm reads ~165-175° on pose models due to deltoid landmarks. 
              Our default threshold is **150°** to account for shoulder anatomy.
            * **Bar Clearance**:
              The chin/head must cross vertically above the hands/bar line. Stopping short results in a **Partial Rep** classification.
            * **Rep Cadence**:
              Controlled ascent and descent phases are timed independently to monitor concentric and eccentric velocity.

            ---

            ### 2. Deadlift
            * **Hip-Hinge Mechanics**:
              The movement is initiated and concluded through the hips rather than spinal flexion.
            * **Spine Neutrality (Straight vs Rounded Back)**:
              The cervical and thoracic spine must remain aligned with the lumbar spine. 
              Deviations greater than **25°** between the head-shoulder line and shoulder-hip line trigger a **Rounded Back** warning to prevent spinal injuries.
            * **Lockout**:
              Achieving full hip extension (>= 165°) with shoulders set back.

            ---

            ### 3. Squats
            * **Parallel Depth**:
              A valid squat requires the crease of the hip to descend at least level with or below the top of the patella (knee angle <= 90°).
            * **Depth Grading**:
              - **Full**: Knee angle <= 80° (Below parallel).
              - **Parallel**: Knee angle between 80° and 90°.
              - **Shallow**: Knee angle > 90° (Insufficient depth).
            """
        )


if __name__ == "__main__":
    main()
