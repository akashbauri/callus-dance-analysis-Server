"""
FastAPI server for Dance Movement Analysis
Written in a simple, clean, and human-friendly way.

This server:
1. Accepts a dance video upload (max 2 minutes, 25MB)
2. Runs pose detection using MediaPipe
3. Returns movement metrics and visualization information
4. Fully deployable on Google Cloud Run
"""

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
from analyze_dance import analyze_dance_video

app = FastAPI(title="Dance Movement Analysis API")

@app.get("/")
def home():
    return {"message": "Dance Movement Analysis API is running successfully!"}

@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    # Basic validation to ensure user uploads correct file type
    if not file.filename.endswith((".mp4", ".mov", ".avi")):
        return JSONResponse({"error": "Invalid file type. Upload a video."}, status_code=400)

    # Read uploaded file bytes
    video_bytes = await file.read()

    # Run movement analysis function
    results = analyze_dance_video(video_bytes)

    return {"filename": file.filename, "results": results}


if __name__ == "__main__":
    # Running locally (Cloud Run will not use this)
    uvicorn.run("app:app", host="0.0.0.0", port=8080)
