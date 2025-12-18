# Person Re-Identification System

A real-time person tracking and re-identification system that remembers people even after they leave the scene. Built with YOLO, ByteTracker, and OSNet for production-ready performance.

## 📖 What is this?

Ever wondered how security systems can track the same person across multiple cameras or remember someone who walked away and came back? That's Re-Identification (Re-ID), and this project does exactly that.
Here's what makes it special:

* **Detects and tracks multiple people simultaneously**
* **Remembers people even after they leave the frame**
* **Re-identifies them when they come back** (even minutes later!)
* **Handles real-world challenges**: crowds, occlusions, different distances
* **Works in real-time** on consumer hardware

**Perfect for:**

* Security and surveillance systems
* Retail analytics (unique visitor counting)
* Event monitoring and crowd management
* Any scenario where you need to track people over time

## ✨ Key Features

### 🔍 Smart Detection & Tracking

* **YOLOv11-pose** detects people with 17 body keypoints
* **ByteTracker** keeps track IDs stable through occlusions
* **Automatic scale classification** (NEAR/MID/FAR based on distance)
* **Quality gate** filters out blurry or partial detections

### 🎭 Advanced Re-Identification

* **OSNet neural network** extracts unique 512-D "fingerprints" for each person
* **Smart Memory Bank** remembers people for 30 seconds to 5 minutes based on:

  * How long they were visible
  * Detection confidence
  * Quality of their appearance
  * How many times they reappeared
* **Multi-Scale Prototypes**: separate embeddings for near/mid/far distances
* **Hungarian Matching** considers similarity, time, space, and quality

### 🛡️ Production-Ready Robustness

* **MFSS (Moving Frame Similarity Score)**: smooths out frame-to-frame noise
* **K-Window Confirmations**: needs 4 out of 8 frames to confirm a match
* **Anti-Teleport**: rejects physically impossible matches (person can't teleport!)
* **Adaptive Lock**: prevents ID flipping right after re-identification
* **Error Penalty**: learns from mistakes with decaying penalties
* **Smart Updates**: only updates embeddings when quality is good

### 📊 Intelligent Buffering

* Collects 10 best-quality embeddings per person
* Automatically rejects outliers
* Checks for diversity (different angles/distances/lighting)
* Uses **medoid** (not average) for robustness

---

## 🏗️ How It Works

### System Flow

```
┌──────────────────────────────────────────────────────────────┐
│                      VIDEO INPUT                             │
│                 (MP4 file or stream)                         │
└──────────────────┬───────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────┐
│                DETECTOR THREAD (async)                      │
│              YOLOv11-pose detection                          │
│  • Finds people in each frame                                │
│  • Extracts 17 body keypoints                                │
│  • Classifies scale (NEAR/MID/FAR)                          │
└──────────────────┬───────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────┐
│                  BYTETRACKER                                  │
│            Kalman Filter tracking                            │
│  • Assigns stable track IDs                                  │
│  • Predicts motion                                           │
│  • Handles occlusions                                        │
│  • States: ACTIVE, PENDING, TEMP_LOST, LOST                 │
└──────────────────┬───────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────┐
│              RE-ID EMBEDDER THREAD (async)                    │
│                   OSNet extraction                           │
│  • Crops shoulder-to-ankles region                           │
│  • Quality gate filtering                                    │
│  • Extracts 512-D embeddings                                 │
│  • Buffers best samples                                      │
└──────────────────┬───────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────┐
│                   RE-IDENTIFIER                               │
│            Identity matching & management                     │
│  • Hungarian matching (multi-factor cost)                    │
│  • MFSS temporal smoothing                                   │
│  • K-window confirmations                                    │
│  • Anti-teleport validation                                  │
│  • Updates Identity Bank                                     │
└──────────────────┬───────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────┐
│                  IDENTITY BANK                                │
│               Long-term memory                               │
│  • Stores multi-scale prototypes                             │
│  • Dynamic TTL (30s-5min)                                    │
│  • Health decay & LRU eviction                               │
│  • Persistent colors per person                              │
└──────────────────┬───────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────┐
│                  VISUALIZATION                                │
│  • Bounding boxes with persistent colors                     │
│  • Person IDs (P01, P02, ...)                                │
│  • Statistics overlay                                         │
└──────────────────────────────────────────────────────────────┘
```

---

## 🚀 Getting Started

### Prerequisites

* Python 3.8 or higher
* pip package manager
* (Optional) CUDA-capable GPU for faster processing

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/dwSize-PE/PersonDetection.git
   cd PersonDetection
   ```

2. Create and activate virtual environment:

   ```bash
   # Create virtual environment
   python -m venv venv

   # Activate it
   # On Windows:
   venv\Scripts\activate

   # On Linux/Mac:
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Install ByteTracker module:

   ```bash
   pip install -e ./app/tracker/bytetrack
   ```

5. Add your video:

   * Place your video in the `data` folder.
   * The file must be named: `data/video.mp4`

   Example:

   ```bash
   cp /path/to/your/video.mp4 data/video.mp4
   ```

> Note: All required models (YOLO and OSNet) are already included in the `models/` folder. No additional downloads needed!

### Running the System

Start the detection and tracking:

```bash
python start.py
```

**What you'll see:**

* Real-time video playback with detection boxes
* Persistent person IDs (P01, P02, P03...)
* Tracking statistics in the top-left corner

Press `ESC` to exit.

---

## Troubleshooting

* **"No module named 'cython_bbox'"**

  ```bash
  pip install -e ./app/tracker/bytetrack
  ```

* **"Video file not found"**

  * Make sure your video is located at `data/video.mp4`
  * Check that the file name is exactly `video.mp4` or edit `VIDEO_PATH` in `app/stream.py` to point to your file.

* **Low FPS / Slow performance**

  * Use a smaller video resolution (720p instead of 1080p)
  * Enable GPU if available (requires CUDA)
  * Close other applications to free up resources

* **"RuntimeError: CUDA out of memory"**

  * Edit `app/osnet/osnet_model.py` and change the device parameter to force CPU usage:

  ```python
  osnet = OsNetEmbedder(device="cpu")  # Force CPU usage
  ```

## License

This project is licensed under the **Educational and Non-Commercial Use License**. For more details, please refer to the [LICENSE.TXT](LICENSE.TXT) file.
