"""Utility to generate a synthetic gym workout clip for automated testing."""

from pathlib import Path
import cv2
import numpy as np


def generate_sample_clip(output_path: str = "tests/sample_workout.mp4", duration_sec: int = 3, fps: int = 30):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = duration_sec * fps
    for i in range(total_frames):
        # Create dark gym background
        frame = np.full((height, width, 3), 35, dtype=np.uint8)

        # Pull-up bar
        cv2.line(frame, (150, 80), (490, 80), (180, 180, 180), 8)

        # Vertical cycle motion: 0 to 1 and back
        cycle = np.sin((i / total_frames) * 2 * np.pi)
        y_offset = int(cycle * 60)

        # Draw a synthetic human stick figure performing pull-up
        hands_y = 80
        head_y = 130 - y_offset
        shoulders_y = 170 - y_offset
        hips_y = 290 - y_offset
        knees_y = 380 - y_offset
        ankles_y = 440 - y_offset

        # Head
        cv2.circle(frame, (320, head_y), 25, (220, 200, 180), -1)
        # Torso
        cv2.line(frame, (320, shoulders_y), (320, hips_y), (50, 150, 220), 8)
        # Arms
        cv2.line(frame, (250, hands_y), (270, (hands_y + shoulders_y) // 2), (220, 200, 180), 5)
        cv2.line(frame, (270, (hands_y + shoulders_y) // 2), (320, shoulders_y), (220, 200, 180), 5)
        cv2.line(frame, (390, hands_y), (370, (hands_y + shoulders_y) // 2), (220, 200, 180), 5)
        cv2.line(frame, (370, (hands_y + shoulders_y) // 2), (320, shoulders_y), (220, 200, 180), 5)
        # Legs
        cv2.line(frame, (320, hips_y), (300, knees_y), (40, 80, 180), 6)
        cv2.line(frame, (300, knees_y), (290, ankles_y), (40, 80, 180), 6)
        cv2.line(frame, (320, hips_y), (340, knees_y), (40, 80, 180), 6)
        cv2.line(frame, (340, knees_y), (350, ankles_y), (40, 80, 180), 6)

        out.write(frame)

    out.release()
    return output_path


if __name__ == "__main__":
    generate_sample_clip()
