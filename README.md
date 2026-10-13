# 🏋️‍♂️ Gym Vision AI - Exercise Form & Rep Counter Web App

**Gym Vision AI** is an interactive, computer-vision-powered biomechanical workout tracker built with **Streamlit** and **MediaPipe Pose**. It provides real-time rep counting, Range of Motion (ROM) evaluation, joint angle gauges, and posture injury prevention warnings for key strength exercises.

---

## 🌟 Key Features

1. **Multi-Exercise Biomechanics**:
   - **Pull-Ups & Chin-Ups**: 
     - Automatic rep counting (armed at dead hang, confirmed at peak chin clearance).
     - Full vs Partial ROM grading: checks dead-hang elbow extension (>= 150°) and chin clearing the bar.
     - Ascent, descent, and total pull duration timing per rep.
   - **Deadlifts**:
     - Hip hinge kinematics & lockout detection.
     - Spine curvature monitoring to flag **Straight Back** vs **Rounded Back** to prevent lumbar injuries.
   - **Squats**:
     - Parallel and below-parallel depth detection.
     - Real-time knee angle progression curves.

2. **Dual Operation Modes**:
   - **Video File Analysis**: Upload `.mp4`, `.mov`, or `.avi` files. Features frame-by-frame processing, high-visibility skeleton overlay, HUD banner, interactive Plotly telemetry charts, and CSV data export.
   - **Live Webcam Mode**: Real-time workout coach with live HUD overlay, audio/visual cues, and live rep tally.

3. **100% Local & Offline**:
   - Powered by pre-trained on-device MediaPipe Pose estimation.
   - Zero API keys required, zero latency, zero recurring cloud costs.

---

## 🚀 Quick Start

### 1. Launch the Web Application

Double-click `run_app.bat` or run the following in PowerShell:

```powershell
.\run_app.ps1
```

Or manually activate the virtual environment and start Streamlit:

```bash
# Windows
.venv\Scripts\activate
streamlit run app.py
```

The application will open automatically in your browser at `http://localhost:8501`.

---

## 📁 Project Architecture

```
d:\Antigravity\GYM App\
├── app.py                     # Main Streamlit web application
├── requirements.txt           # Python package dependencies
├── run_app.bat                # 1-click Windows batch launcher
├── run_app.ps1                # 1-click PowerShell launcher
├── README.md                  # Project documentation
├── src/
│   ├── __init__.py
│   ├── pose_detector.py       # MediaPipe Pose wrapper & angle geometry
│   ├── pullup_analyzer.py     # Pull-up/Chin-up rep counting & ROM engine
│   ├── deadlift_analyzer.py   # Deadlift hinge & spine curvature analyzer
│   ├── squat_analyzer.py      # Squat depth & knee angle analyzer
│   └── visualizer.py          # HUD overlays, skeletons, & gauges
└── tests/
    ├── test_analyzers.py      # Unit tests for kinematics and state machines
    └── generate_test_video.py # Synthetic video generator for smoke testing
```

---

## 🧪 Running Unit Tests

```bash
.venv\Scripts\python.exe -m pytest tests/
```
