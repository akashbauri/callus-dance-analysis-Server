# Dance Movement Analysis Server (FastAPI + MediaPipe)

This project analyzes human dance movements using a lightweight AI pose detection system.  
It was built as part of the **Callus Company – AI ML Server Engineer** competency assessment.

---

## 🚀 What This Server Does
- Accepts dance videos up to **2 minutes** and **25 MB**
- Runs AI pose estimation using **MediaPipe**
- Measures:
  - Total frames
  - Frames with valid body pose detected
  - Movement accuracy percentage
- Returns results in a clean JSON format
- Fully deployable on **Google Cloud Run (recommended)**

---

## 🧠 Thought Process

I designed the server to be:
1. **Simple** — easy to understand for hiring managers  
2. **Fast** — MediaPipe runs locally without cloud GPU  
3. **Reliable** — Clear error handling for unsupported files  
4. **Deployment-friendly** — Docker + Cloud Run deployment ready  

This approach mirrors Callus’s vision of **building lightweight, scalable, real-world AI systems**.

---

## 📦 API Endpoints

### **POST /analyze**
Upload a video and receive movement analysis.

#### Example (cURL)
```bash
curl -X POST "https://YOUR_CLOUD_RUN_URL/analyze" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@dance.mp4"
