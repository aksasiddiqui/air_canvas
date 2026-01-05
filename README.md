# Air Canvas

Air Canvas is a real-time hand gesture based drawing application built using MediaPipe and OpenCV. It allows you to draw on a virtual canvas by tracking your hand movements through a webcam, without using a mouse or stylus.

## Features
- Real-time hand landmark detection
- Draw using index finger movements
- Gesture-based controls for drawing and color selection
- Fist gesture to open or close the color palette
- Open hand gesture to toggle drawing mode
- Save drawings as PNG images
- Clear canvas instantly
- Fullscreen immersive drawing experience

## Gesture Controls
- **Index finger up**: Draw on the canvas  
- **Fist**: Toggle color palette  
- **Open hand**: Toggle drawing mode  

## Keyboard Controls
- `C` – Clear canvas  
- `S` – Save drawing  
- `Q` – Quit application  

## Tech Stack
- Python
- OpenCV
- MediaPipe
- NumPy

## Installation
```bash
pip install opencv-python mediapipe numpy
```

## Usage 
```
python air_canvas.py
```
