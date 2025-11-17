"""
This file contains the main logic for dance video movement analysis.
The code is written cleanly so any human engineer can understand it easily.
"""

import mediapipe as mp
import cv2
import numpy as np
import tempfile

def analyze_dance_video(video_bytes):
    """
    Step-by-step:
    1. Convert uploaded bytes into a temporary video file
    2. Use MediaPipe Pose to detect body landmarks
    3. Calculate basic movement metrics (number of frames with full pose detected)
    4. Return data in a simple dictionary format
    """

    # Write video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(video_bytes)
        video_path = temp_video.name

    # Initialize MediaPipe Pose model
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    detected_frames = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(frame_rgb)

        # Count frames where landmarks are detected
        if result.pose_landmarks:
            detected_frames += 1

    cap.release()

    # Simple movement metrics
    detection_rate = round((detected_frames / max(frame_count, 1)) * 100, 2)

    return {
        "total_frames": frame_count,
        "frames_with_pose_detected": detected_frames,
        "pose_detection_accuracy_percent": detection_rate
    }
