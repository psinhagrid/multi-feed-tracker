# Multi-Feed Tracker

A computer vision system for real-time object detection, tracking, and cross-camera person re-identification with AI activity captioning.

---

## Demo 3 — Cross-Camera Re-ID + Activity Intelligence

### What it does
Select a person in one video feed. The system re-identifies that person in a second feed from a different camera, tracks them with a bounding box, and uses a local vision-language model to caption their actions in real time.

### Retail use case
Track a shopper across store cameras automatically. The system logs which shelves they visit, what they pick up, and what they put back — producing structured behavioural data without staff involvement.

```
Customer journey — 14:23 to 14:31
  [CAM 2] Entered store, walked toward produce
  [CAM 5] Examined item at shelf 3, returned it, picked up second item
  [CAM 8] Picked up item at shelf 10 immediately
  [CAM 11] Spent 4 min at clearance shelf, no pickup
  [CAM 3] Proceeded to checkout
```

This data feeds directly into shelf layout optimisation, footfall analysis, and loss prevention.

### Technical implementation

**Re-identification pipeline**
- Upload video 1 (reference camera). Draw a bounding box around the target person.
- YOLOv8-seg runs segmentation to extract a masked crop of the person.
- [fast-reid](https://github.com/JDAI-CV/fast-reid) (ResNet50, Market1501 weights) extracts a 2048-dim embedding from the masked crop, with a hue histogram computed from the torso region as a secondary feature.
- Upload video 2 (target camera). YOLO detects all people every N frames. Each detection is compared to the reference embedding via cosine similarity and hue histogram distance.
- The highest-similarity detection above threshold is locked as the target.

**Tracking between detections**
- OpenCV CSRT tracker maintains the bounding box between YOLO detection intervals (~15 frames).
- Smart reinit logic distinguishes between YOLO missing the target vs. detecting a different person using IoU overlap and similarity gap thresholds.
- Adaptation mechanism: after 3 consecutive high-confidence solo detections, the reference embedding is updated to account for cross-camera domain shift (different lighting, angle).

**Activity captioning**
- Every 1 second, the tracked person's bounding box crop is saved (224×224 RGB).
- A sliding buffer of 8 frames (8 seconds of temporal context) is maintained.
- Every 3 seconds, the buffer is sent to [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) running locally on Apple Silicon MPS.
- The model returns a 1–2 sentence description of the action sequence observed across the frames.
- Captions stream word-by-word into the UI with a loading progress bar during model warm-up.
- Only frames where the green (matched) box is active are buffered — no captions generated from unconfirmed detections.

**Stack:** FastAPI · YOLOv8 · fast-reid (ResNet50) · OpenCV CSRT · Qwen2.5-VL-3B (transformers, MPS) · React

---

## Demo 2 — Single-Camera Object Tracker

### What it does
Draw a box around any object in a video. The system tracks it continuously, surviving occlusion, scale change, and crowded scenes.

### Use cases
- Security operator locks onto a suspect in a crowd without losing them between frames
- Sports broadcast follows a specific player through an entire play
- Retail analytics tracks which shelves a customer visits and in what order
- Drone or vehicle operations maintain a moving target lock across a wide scene

### Technical implementation

- User draws a bounding box in the React UI. Coordinates are sent to the backend.
- OpenCV CSRT (Discriminative Correlation Filter with Channel and Spatial Reliability) initialises a tracker on that region.
- Every frame, CSRT predicts the new bounding box using learned appearance and spatial features.
- Every N frames (configurable), YOLOv8 re-runs detection. If a detection overlaps the current tracker position (IoU > threshold), the tracker is re-initialised on the current frame to prevent drift.
- Claude Vision API labels the tracked object on first detection (e.g. "person in blue jacket") and the label persists on the overlay.
- Background threading: detection runs in a daemon thread so CSRT updates never block video streaming.

**Stack:** FastAPI · YOLOv8 · OpenCV CSRT · Claude Vision API · React

---

## Demo 1 — Text-Based Object Search

### What it does
Describe what you're looking for in plain English. The system finds it across video frames without any retraining or labelling.

### Use cases
- Retail: *"person wearing store uniform"* for staff tracking across hours of footage
- Safety: *"worker without hard hat"* for automated compliance checks
- Warehouse: *"forklift near racking"* for incident review
- Security: *"person with large backpack near exit"* in seconds, not hours
- Smart CCTV: *"red jacket"* or *"person crouching"* — any natural language description

### Technical implementation

- User types a description in the React UI. The query is sent to the backend.
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) (Tiny variant) performs open-vocabulary detection — it grounds the text query to regions in the frame without task-specific fine-tuning.
- Detections above the confidence threshold are drawn with bounding boxes and returned as a stream.
- Claude Vision API generates a concise label for each detected region, displayed on the overlay.
- Detection interval and confidence threshold are configurable from the UI.

**Stack:** FastAPI · Grounding DINO · Claude Vision API · React

---

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
./start_app.sh
```

- **Frontend:** http://localhost:8000
- **Backend API:** http://localhost:8080
- **API docs:** http://localhost:8080/docs

**Re-ID setup** (required for Demo 3):
```bash
git clone https://github.com/JDAI-CV/fast-reid.git
cd fast-reid && pip install -r requirements.txt && python setup.py develop && cd ..
# Place Market1501 ResNet50 weights at fast-reid/model.pth
```

**Activity captioning setup** (required for Demo 3):
```bash
pip install hf-xet
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-VL-3B-Instruct', ignore_patterns=['*.gguf'])"
```
Model (~7 GB) downloads once and is cached. Loads into MPS memory on first backend start (~70 s), then stays ready for the session.

---

## Architecture

```
frontend/          React + Vite + shadcn/ui
app.py             FastAPI — video streaming, Re-ID pipeline, captioning
detection/         Grounding DINO wrapper
re_id/             fast-reid embedding extractor + cosine matcher
utils/             CSRT tracker factory, device selection
```
