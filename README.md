# Pushkaralu Line Counter (People-Count)

Real-time CCTV people counter with 2D virtual tripwire gate, YOLOv8 person detection, ByteTrack multi-object tracking, and zero-latency live browser monitoring.

![Pushkaralu Line Counter](https://raw.githubusercontent.com/abdulaleemarshad1979/People-Count/main/templates/preview.png)

## Key Features

- **Decoupled 30 FPS Pipeline**: RTSP video capture, asynchronous AI inference, and MJPEG web streaming run on independent threads for smooth playback and responsiveness.
- **Pure 2D Virtual Tripwire**:
  - Interactive drag-and-drop handles (`A` and `B`) to align the gate across any doorway, corridor, or walkway.
  - Mathematically guaranteed 2D line segment intersection counting with trajectory history memory.
  - Symmetrical **IN** and **OUT** direction tracking with live occupancy display (`OCC`).
  - **1-Click Flip Direction**: Instantly swap IN and OUT directions via the UI.
- **Enhanced Person Tracking**:
  - Fine-tuned ByteTrack parameters optimized for overlapping and occluded crowds.
  - Torso/chest anchor tracking immune to office desks, chairs, and foreground clutter.
  - Spatial Re-ID memory survives temporary tracker drops and ID switches.
- **Interactive Web Interface**:
  - Zero-plugin browser dashboard accessible from desktop, tablet, or phone.
  - Real-time HUD with IN, OUT, and OCC counters, Active Tracks, and FPS indicators.
  - Live confidence tuning slider and doorway presets.

## Architecture

```
                                  ┌────────────────────────┐
                                  │  Threaded RTSP Capture │ (Pulls raw camera stream)
                                  └───────────┬────────────┘
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      ▼                                               ▼
         ┌────────────────────────┐                      ┌────────────────────────┐
         │ Async AI Worker        │                      │ Display & MJPEG Loop   │
         │ (YOLOv8 + ByteTrack)   │                      │ (Overlay & 30 FPS web) │
         └────────────┬───────────┘                      └────────────┬───────────┘
                      │                                               │
                      ▼                                               ▼
         ┌────────────────────────┐                      ┌────────────────────────┐
         │ 2D Virtual Tripwire    │                      │ Interactive Web UI     │
         │ (IN / OUT / Occupancy) │                      │ (Canvas + Live Stream) │
         └────────────────────────┘                      └────────────────────────┘
```

## Quick Start

### 1. Requirements & Setup

```bash
# Clone the repository
git clone https://github.com/abdulaleemarshad1979/People-Count.git
cd People-Count

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Camera

Create a launch script or export environment variables:

```bash
export ICSEE_CAMERA_IP="192.168.1.111"
export ICSEE_CAMERA_USER="admin"
export ICSEE_CAMERA_PASSWORD="your_password"
export ICSEE_CAMERA_PORT="554"
export ICSEE_CAMERA_CHANNEL="1"
export ICSEE_CAMERA_STREAM="0"  # 0 for Main HD stream, 1 for Sub stream

# Launch the counter
python3 app.py
```

Open your browser at `http://127.0.0.1:5000`.

## Configuration Options

| Environment Variable | Default | Description |
|---|---|---|
| `ICSEE_CAMERA_IP` | `192.168.1.10` | IP address of the RTSP/iCSee camera |
| `ICSEE_CAMERA_USER` | `admin` | Camera RTSP username |
| `ICSEE_CAMERA_PASSWORD` | `""` | Camera RTSP password |
| `ICSEE_CAMERA_STREAM` | `0` | `0` = Main stream (HD), `1` = Sub stream |
| `RTSP_URL` | `""` | Direct RTSP URL override or webcam index (`0`) |
| `YOLO_MODEL` | `yolov8n.pt` | YOLOv8 model (`yolov8n.pt`, `yolov8s.pt`, etc.) |
| `YOLO_CONFIDENCE` | `0.22` | Person detection confidence threshold |
| `YOLO_IMGSZ` | `480` | Input image size for inference |
| `WEB_PORT` | `5000` | HTTP port for the web dashboard |
| `WEB_HOST` | `127.0.0.1` | HTTP bind address (`0.0.0.0` for local network access) |
