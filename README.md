# Pushkaralu Line Counter (People-Count)

Real-time CCTV people counter with 2D virtual tripwire gate, YOLOv8 person detection, tuned ByteTrack multi-object tracking, robust anti-jitter crossing engine, and zero-latency live browser monitoring.

![Pushkaralu Line Counter](https://raw.githubusercontent.com/abdulaleemarshad1979/People-Count/main/templates/preview.png)

## Key Features

- **Decoupled 30 FPS Pipeline**: RTSP video capture, asynchronous AI inference, and MJPEG web streaming run on independent threads for smooth playback and responsiveness.
- **Robust Anti-Jitter Line Counter (`counter.py`)**:
  - **Dead-Band Hysteresis**: A person is counted only after being stably outside a dead-band on Side A for $N$ consecutive frames, then becoming stably confirmed on Side B. Box vibrations or hovering near the line cannot satisfy this condition.
  - **Strict Line Segment Intersection**: Ensures the anchor path actually cuts through the virtual tripwire segment, rejecting people who walk around its ends.
  - **Ghost Track Elimination**: Detections living fewer than 3 frames are rejected.
  - **Track Handoff**: If ByteTrack drops an ID and issues a new one near the line, the new ID inherits the previous track's state and history, preventing double counts.
  - **Duplicate Count Suppression**: Re-identification switches at the line for vanished tracks are automatically suppressed.
  - **Live In-Span Occupancy (`OCC`)**: Live count of visible, confirmed persons currently on the IN side and within the line's span. Decrements immediately when someone exits view or steps back.
- **Tuned ByteTrack Configuration (`bytetrack_surveillance.yaml`)**:
  - Correct `match_thresh: 0.80` (replaces over-restrictive `0.30` which previously caused frequent ID switches at CCTV frame rates).
  - High and low detection thresholds configured for seamless track continuity through occlusions.
- **Interactive Web Interface**:
  - Zero-plugin browser dashboard accessible from desktop, tablet, or phone.
  - Real-time HUD with IN, OUT, and OCC counters, Active Tracks, and FPS indicators.
  - Interactive handles (`A` and `B`) to drag the tripwire anywhere in the frame.
  - 1-click **Flip Direction** button and real-time confidence tuning slider.
- **Audit Logging**:
  - Automatically records every crossing event to `count_events.csv` with timestamp, track ID, and direction.

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
         │ LineCounter Engine     │                      │ Interactive Web UI     │
         │ (Dead-band, Handoff)   │                      │ (Canvas + Live Stream) │
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

### 2. Run Simulation Tests

Verify the anti-jitter and crossing engine across 11 simulated test cases (including stationary jitter, pacing, ID handoff, ghost tracks, and batch walk simulations):

```bash
python3 test_counter.py
```

### 3. Configure Camera & Launch

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
| `YOLO_CONFIDENCE` | `0.30` | Person detection confidence threshold |
| `YOLO_IMGSZ` | `480` | Input image size for inference |
| `COUNT_BAND_RATIO` | `0.12` | Dead-band half-width as fraction of person box height |
| `COUNT_CONFIRM_FRAMES`| `2` | Consecutive frames needed to confirm a side |
| `COUNT_MIN_HITS` | `3` | Track must be seen this many times before it can count |
| `EVENT_LOG` | `count_events.csv` | Path to CSV file for crossing audit events |
| `WEB_PORT` | `5000` | HTTP port for the web dashboard |
| `WEB_HOST` | `127.0.0.1` | HTTP bind address (`0.0.0.0` for local network access) |

## Best Practices for Camera & Tripwire Setup

1. **Tripwire Position**: Place the counting line across the middle of the entrance, hallway, or walkway where people are in full view for several frames before and after crossing. Avoid placing the line at the very top or bottom frame edge where people suddenly appear/disappear.
2. **Span Width**: Drag points `A` and `B` across the full width of the passable opening. Persons walking outside this segment will not trigger counts.
3. **Invert Direction**: If the green **IN** arrow points in the opposite direction of entry, click the **Flip Direction** button in the web UI.
