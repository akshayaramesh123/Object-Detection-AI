# Object Detection AI

An AI-powered object detection web app built using Streamlit and YOLOv8.
Users can upload an image and detect multiple objects with bounding boxes and confidence scores.

## Features
- Upload image files (JPG, PNG)
- Detect multiple objects
- Show object names
- Display confidence scores
- Draw bounding boxes on detected objects
- Download detected image

## Technologies Used
- Python
- Streamlit
- Ultralytics YOLOv8
- OpenCV
- Pillow
- NumPy
- Pandas

## How It Works
1. User uploads an image
2. YOLOv8 model processes the image
3. Objects are detected
4. Bounding boxes and labels are displayed
5. Detected image and object list are shown

## Installation

```bash
pip install streamlit ultralytics opencv-python pillow numpy pandas
