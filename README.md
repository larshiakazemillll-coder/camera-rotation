<p align="center">
  <img src="ChatGPT Image Nov 18, 2025, 02_05_34 PM.png" width="500" />
</p>

# 🎭 AI-Powered PTZ Face Recognition & Tracking System 🤖

<div align="center">

![AI Face Recognition](https://img.shields.io/badge/AI-Face%20Recognition-blue)
![PTZ Control](https://img.shields.io/badge/PTZ-Auto%20Tracking-green)
![Real-time](https://img.shields.io/badge/Real--time-30%20FPS-orange)
![OpenVINO](https://img.shields.io/badge/OpenVINO-Optimized-purple)

**"Your AI security guard that never blinks! 👁️‍🗨️"**

*Automatically finds people, zooms in on their faces, and identifies them in real-time*

</div>

## 🚀 What Does This System Do?

Imagine having a smart security camera that can:
- **Automatically rotate** left and right searching for people
- **Detect humans** using advanced YOLOv8 AI
- **Zoom in intelligently** on detected faces
- **Recognize identities** by comparing with a database
- **Operate 24/7** without human intervention

This is exactly what this system does! It's like having a robotic security guard with super-human vision! 🦸‍♂️

## 🎯 System Overview

### The AI Dream Team 🤖

| Component | Role | Superpower |
|-----------|------|------------|
| **YOLOv8** | People Detective 👥 | Finds humans in the camera view |
| **RetinaFace** | Face Spotter 🎯 | Precisely locates faces in zoomed images |
| **ArcFace** | Identity Verifier 🔍 | Extracts unique face features |
| **FAISS** | Memory Bank 🧠 | Fast face database searching |
| **Dahua PTZ** | Robotic Eye 🤖 | Moves, zooms, and tracks automatically |

### How It Works: The Magic Pipeline ✨

```
1. 🔄 Camera rotates left → right continuously
2. 👥 YOLOv8 detects people in real-time
3. ⏹️ Camera stops when person detected
4. 🔍 Zooms in on upper body/face region
5. 📸 Captures high-resolution face image
6. 🎯 RetinaFace extracts face landmarks
7. 🔢 ArcFace converts face to numbers (embeddings)
8. 🗄️ FAISS searches database for matches
9. 🆔 Returns identity with confidence score
10. 🔄 Resets and continues searching
```

## ⚡ Quick Start

### Prerequisites 🛠️

```bash
# Core dependencies
pip install openvino-dev[onnx]
pip install ultralytics  # YOLOv8
pip install opencv-python
pip install faiss-cpu
pip install requests
pip install scikit-learn
pip install scikit-image
```

### Hardware Requirements 💻

- **Camera**: Dahua PTZ IP Camera (or compatible)
- **Processing**: CPU with OpenVINO support (Intel recommended)
- **Network**: Stable connection to face database API
- **Storage**: For temporary image processing

### Configuration ⚙️

```python
# Update these in main() function
DAHUA_IP = "192.168.7.108"        # Your camera IP
PTZ_USERNAME = "admin"            # Camera login
PTZ_PASSWORD = "Aa123456!"        # Camera password
DB_IP = "192.168.7.40:80"         # Face database API
RTSP_URL = "rtsp://admin:Aa123456!@192.168.7.108:554/cam/realmonitor?channel=1&subtype=0"
```

### Run the System 🏃‍♂️

```bash
# Start the intelligent surveillance system
python app.py
```

## 🏗️ System Architecture

### Multi-Threaded Design 🧵

The system uses three specialized threads for optimal performance:

```python
# 1. Frame Reader Thread 📹
- Continuously reads RTSP stream
- Maintains 30 FPS frame rate
- Manages frame buffer queue

# 2. Detection Thread 🔍
- Runs YOLOv8 person detection
- Processes frames from queue
- Triggers PTZ control when people found

# 3. PTZ Control Thread 🤖
- Manages camera movement patterns
- Handles zoom and focus operations
- Coordinates face recognition process
```

### Core Components Deep Dive 🔬

#### 1. Face Recognition Engine (`FaceRecognition` Class)

```python
class FaceRecognition:
    def __init__(self, db_url, retinaface_model_path):
        self.detector = RetinaFace()      # Face detection
        self.recModel = ArcFaceONNX()     # Feature extraction
        self.index_db = faiss.Index()     # Fast similarity search
        self.l2_normalizer = Normalizer() # Feature normalization
    
    def detect_and_recognize(self, img):
        # 1. Detect faces with RetinaFace
        # 2. Align and crop faces
        # 3. Extract embeddings with ArcFace
        # 4. Search database with FAISS
        # 5. Return matches and confidence
```

#### 2. PTZ Controller (`DahuaPTZController` Class)

```python
class DahuaPTZController:
    def move_directly(self, x, y, zoom):
        # Move camera to specific coordinates
        # x, y: normalized position (0-1)
        # zoom: zoom level (0-1)
    
    def move_left/right(self, speed, duration):
        # Smooth camera rotation
```

#### 3. YOLOv8 Person Detection

```python
def detect_person_yolov8(frame, model):
    # Uses YOLOv8-nano for fast inference
    # Only detects 'person' class (class 0)
    # Configurable confidence threshold
```

## 🎨 Intelligent Zoom Algorithm

The system doesn't just zoom randomly - it uses smart calculations:

```python
# Smart zoom coordinates calculation
x_center = (x1 + x2) / 2                    # Person center X
y_center = (y1 + y2/3) / 2                  # Focus on upper body (face region)
face_area = (x2 - x1) * (y2/3 - y1)         # Calculate face area
arg3 = face_area / (frame_h * frame_w)      # Normalized zoom level
```

This ensures the camera zooms precisely to capture the face area for optimal recognition!

## 📊 Performance Optimization

### Model Selection 🚀

| Model | Purpose | Speed | Accuracy |
|-------|---------|-------|----------|
| **YOLOv8-nano** | Person detection | ⚡⚡⚡⚡ | ✅✅ |
| **RetinaFace** | Face detection | ⚡⚡⚡ | ✅✅✅ |
| **ArcFace-128** | Feature extraction | ⚡⚡ | ✅✅✅✅ |

### Threading Strategy 🧵

- **Frame Reader**: Never blocks detection
- **Detection**: Runs independently of PTZ movements  
- **PTZ Control**: Smooth camera operations without freezing

### OpenVINO Acceleration ⚡

All models optimized with OpenVINO for:
- 2-3x faster inference
- Lower CPU usage
- Better resource management

## 🔧 Configuration Options

### Detection Sensitivity 🎛️

```python
# Adjust based on your environment
YOLO_CONFIDENCE = 0.5      # Lower = more detections, Higher = fewer false positives
FACE_CONFIDENCE = 0.6      # Face detection threshold
RECOGNITION_THRESHOLD = 0.7 # Face matching confidence
```

### Camera Behavior 📡

```python
# PTZ movement parameters
SCAN_SPEED = 50            # Camera rotation speed (1-100)
SCAN_DURATION = 1          # Movement duration per step
ZOOM_DWELL_TIME = 5        # How long to stay zoomed (seconds)
```

## 🗂️ Project Structure

```
ai-ptz-face-system/
├── 🧠 Models/
│   ├── yolov8n.pt                 # Person detection
│   ├── retinaface_mnet025_v2.onnx # Face detection
│   └── rec_mobile_128.xml/bin     # Face recognition
├── 🔧 Core/
│   ├── app.py                     # Main application
│   ├── retinaface.py              # Face detection wrapper
│   ├── arcface_onnx.py           # Feature extraction
│   └── face_align.py              # Face alignment utils
├── 🎥 Outputs/
│   └── zoomed_frames.jpg          # Captured face images
└── 📚 Utils/
    └── nms.py                     # Non-maximum suppression
```

## 🌟 Key Features

### ✅ Automatic People Search
- Continuous left-right scanning pattern
- Real-time person detection
- Instant movement stopping on detection

### ✅ Intelligent Zoom
- Smart coordinate calculation
- Upper-body focus for better face capture
- Adaptive zoom levels based on distance

### ✅ High-Accuracy Recognition
- State-of-the-art face detection (RetinaFace)
- Robust feature extraction (ArcFace)
- Fast database matching (FAISS)

### ✅ Robust Error Handling
- Network retry mechanisms
- Frame read recovery
- PTZ command verification

### ✅ Real-time Performance
- Multi-threaded architecture
- Optimized model inference
- Efficient resource usage

## 🚨 Troubleshooting Guide

### Common Issues & Solutions 🔧

```python
# Problem: No faces detected after zoom
# Solution: Adjust zoom coordinates in ptz_control_thread
y2_new = y1 + height / 3  # Try 2.5 or 2 for different face positions

# Problem: Too many false person detections  
# Solution: Increase YOLO confidence threshold
persons = detect_person_yolov8(frame, yolo_model, conf_threshold=0.7)

# Problem: Camera movement too jerky
# Solution: Adjust movement parameters
ptz.move_right(speed=30, duration=2)  # Slower, longer movements

# Problem: Face recognition not working
# Solution: Check database connection and face alignment
```

### Performance Tuning 🎯

```python
# For better accuracy (slower):
YOLO_MODEL = "yolov8m.pt"  # Medium model
FACE_CONFIDENCE = 0.5      # More face detections

# For faster performance:
YOLO_MODEL = "yolov8n.pt"  # Nano model  
FACE_CONFIDENCE = 0.7      # Fewer, higher-quality detections
```

## 🎯 Use Cases

### 🏢 Security & Surveillance
- Automated perimeter monitoring
- Access control systems
- Intruder detection and identification

### 🏫 Smart Facilities
- Conference room occupancy
- Visitor management
- Automated attendance systems

### 🏢 Corporate Campuses
- Employee tracking (opt-in)
- Restricted area monitoring
- After-hours security

### 🎪 Event Management
- Crowd monitoring
- VIP tracking
- Security automation

## 🔮 Future Enhancements

- [ ] **Multiple Camera Support** - Networked surveillance
- [ ] **Face Anti-Spoofing** - Prevent photo/video attacks  
- [ ] **Behavior Analysis** - Anomaly detection
- [ ] **Cloud Integration** - Centralized management
- [ ] **Mobile Alerts** - Real-time notifications
- [ ] **Historical Analytics** - Movement pattern analysis

## 📊 Sample Output

When running, you'll see real-time information:

```
🎯 Detection: Person found at coordinates [320, 180, 480, 420]
🔍 Zooming: Focusing on face region...
📸 Captured: High-resolution face image saved
🎭 Recognition: Match found - ID: john_doe (Confidence: 94.2%)
🔄 Resetting: Camera returning to scan pattern...
```

## 🤝 Contributing

We welcome improvements! Areas for contribution:
- Performance optimization
- Additional camera support
- Enhanced recognition algorithms
- Better error handling
- Documentation improvements

## 📜 License

This project is intended for research and legitimate security applications. Please ensure compliance with local privacy laws and regulations when deploying.

---

<div align="center">

**Built with ❤️ using cutting-edge AI technologies**

*"Making the world safer, one face at a time! 🌟"*

</div>

---

### 🚀 Ready to Deploy?

Set up your camera, configure the IP addresses, and launch the system to experience autonomous intelligent surveillance! Your AI security guard is waiting to go on duty! 🎭🔒
