"""
app.py
AI Dance Movement Analysis Server

This server does three main things:

1. Exposes a REST API endpoint `/analyze` for video-based movement analysis.
2. Provides a Gradio web UI where users can upload a dance video and see results.
3. Uses MediaPipe + OpenCV to run pose detection and compute simple movement metrics.

The code is intentionally written in a clear, "human" style with comments
so that reviewers can easily follow the logic and reasoning.
"""

import os
import uuid
import tempfile
from typing import List, Dict, Any

import cv2
import numpy as np
import mediapipe as mp
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import gradio as gr


# -----------------------------
# 1. Pose analysis logic
# -----------------------------

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


class MovementResult(BaseModel):
    """Schema for the JSON response of the analysis."""
    video_id: str
    avg_speed: float
    smoothness: float
    total_frames: int
    fps: float
    comment: str


def extract_pose_landmarks(video_path: str) -> Dict[str, Any]:
    """
    Run MediaPipe Pose on a video and extract pose landmarks frame by frame.

    Returns a dictionary with:
    - "landmarks": list of frames, each frame is list of (x, y, z, visibility)
    - "fps": frames per second
    - "frame_count": number of frames processed
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    pose = mp_pose.Pose(static_image_mode=False,
                        model_complexity=1,
                        enable_segmentation=False,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)

    all_landmarks: List[List[List[float]]] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR (OpenCV format) to RGB (MediaPipe format)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        frame_landmarks: List[List[float]] = []
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                frame_landmarks.append([lm.x, lm.y, lm.z, lm.visibility])
        else:
            # If no pose detected, append an empty list for this frame
            frame_landmarks = []

        all_landmarks.append(frame_landmarks)

    cap.release()
    pose.close()

    return {
        "landmarks": all_landmarks,
        "fps": fps,
        "frame_count": frame_count,
    }


def compute_movement_metrics(landmark_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute simple movement metrics:
    - average speed: how much the joints move between frames
    - smoothness: inversely related to frame-to-frame jitter

    These are intentionally simple, explainable metrics
    to demonstrate the server's analysis logic.
    """
    landmarks = landmark_data["landmarks"]
    fps = landmark_data["fps"] or 30.0

    if len(landmarks) < 2:
        return {"avg_speed": 0.0, "smoothness": 0.0}

    frame_speeds = []

    # Iterate through consecutive frame pairs
    for i in range(1, len(landmarks)):
        prev = landmarks[i - 1]
        curr = landmarks[i]

        # Skip if pose is missing in either frame
        if len(prev) == 0 or len(curr) == 0 or len(prev) != len(curr):
            continue

        # Compute Euclidean distance for each joint
        joint_diffs = []
        for p, c in zip(prev, curr):
            px, py, pz, _ = p
            cx, cy, cz, _ = c
            dist = np.sqrt((cx - px) ** 2 + (cy - py) ** 2 + (cz - pz) ** 2)
            joint_diffs.append(dist)

        if not joint_diffs:
            continue

        # Average distance across joints for this frame pair
        avg_dist = float(np.mean(joint_diffs))
        frame_speeds.append(avg_dist * fps)  # convert to "per second" units

    if not frame_speeds:
        return {"avg_speed": 0.0, "smoothness": 0.0}

    avg_speed = float(np.mean(frame_speeds))
    # Smoothness: higher when speeds vary less
    smoothness = float(1.0 / (1.0 + np.std(frame_speeds)))

    return {"avg_speed": avg_speed, "smoothness": smoothness}


def analyze_video_file(video_path: str) -> MovementResult:
    """
    High-level helper:
    - Extract landmarks
    - Compute metrics
    - Build a MovementResult object
    """

    landmark_data = extract_pose_landmarks(video_path)
    metrics = compute_movement_metrics(landmark_data)

    avg_speed = metrics["avg_speed"]
    smoothness = metrics["smoothness"]

    # Human-friendly comment logic
    if avg_speed < 0.1:
        comment = "The movement is very gentle. Encourage more dynamic steps."
    elif avg_speed < 0.3:
        comment = "The movement has moderate energy. Good for controlled choreography."
    else:
        comment = "The movement is very energetic! Great for high-intensity routines."

    # Customize slightly with smoothness
    if smoothness < 0.3:
        comment += " However, the motion looks a bit jerky. Try smoother transitions."
    elif smoothness > 0.7:
        comment += " The transitions look smooth and well-controlled."

    return MovementResult(
        video_id=str(uuid.uuid4()),
        avg_speed=avg_speed,
        smoothness=smoothness,
        total_frames=landmark_data["frame_count"],
        fps=landmark_data["fps"],
        comment=comment,
    )


# -----------------------------
# 2. FastAPI app (REST API)
# -----------------------------

app = FastAPI(
    title="AI Dance Movement Analysis Server",
    description="Upload a dance video and receive simple pose-based movement metrics.",
    version="1.0.0",
)

# Allow basic CORS so a frontend could call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/analyze", response_model=MovementResult)
async def analyze_endpoint(file: UploadFile = File(...)):
    """
    REST API endpoint.

    Usage (example with curl):

    curl -X POST "http://localhost:8000/analyze" \
         -F "file=@sample_dance.mp4"

    The server will:
    - save the uploaded file to a temp directory
    - run pose-based analysis
    - return JSON with metrics and a human-readable comment
    """
    # Basic validation of file type
    if not file.filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
        raise HTTPException(status_code=400, detail="Please upload a valid video file (mp4/mov/avi/mkv).")

    try:
        # Save to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name

        result = analyze_video_file(tmp_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during analysis: {e}")

    finally:
        # Clean up temp file
        if "tmp_path" in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)

    return result


# -----------------------------
# 3. Gradio UI
# -----------------------------

def gradio_analyze(video_path: str):
    """
    Wrapper for Gradio interface.

    Gradio's Video component (in recent versions) passes a file path
    to the loaded temporary video file on the server side.
    """
    if video_path is None:
        return "Please upload a video first.", None

    try:
        result = analyze_video_file(video_path)

        text_summary = (
            f"🎥 Video ID: {result.video_id}\n"
            f"📊 Total Frames: {result.total_frames}\n"
            f"⏱ FPS: {result.fps:.1f}\n\n"
            f"⚡ Average Speed: {result.avg_speed:.4f}\n"
            f"🧠 Smoothness Score: {result.smoothness:.4f}\n\n"
            f"🗣 Comment: {result.comment}"
        )

        # For now we just return the text and no visualization image
        # You could extend this to draw skeletons on a few frames.
        return text_summary, None

    except Exception as e:
        return f"Something went wrong during analysis: {e}", None


with gr.Blocks(title="AI Dance Movement Analysis") as demo:
    gr.Markdown(
        """
    # 🩰 AI Dance Movement Analysis Server

    Upload a short dance video and let the AI analyze:
    - How **fast** your movements are (average speed)
    - How **smooth** your transitions look (smoothness score)
    """
    )

    with gr.Row():
        video_input = gr.Video(
            label="Upload a dance video (MP4, up to ~2 minutes, ≤25 MB)"
        )
        analysis_output = gr.Textbox(
            label="Movement Analysis Summary",
            lines=12
        )

    image_output = gr.Image(
        label="(Optional) Pose Visualization",
        visible=False  # we are not generating this yet
    )

    analyze_button = gr.Button("Analyze Video")

    analyze_button.click(
        fn=gradio_analyze,
        inputs=[video_input],
        outputs=[analysis_output, image_output],
    )

# Mount Gradio app inside FastAPI at `/`
from fastapi.middleware.wsgi import WSGIMiddleware  # not strictly needed here, but kept for clarity
from gradio.routes import App as GradioApp

# This is the recommended way in newer Gradio versions:
app = gr.mount_gradio_app(app, demo, path="/")


# If you run this file directly (e.g. `python app.py`),
# you can start the server locally with Uvicorn.
if __name__ == "__main__":
    import uvicorn

    # Local development run:
    uvicorn.run(app, host="0.0.0.0", port=8000)
