import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ultralytics import YOLO
import sys
import os
from pathlib import Path
from PIL import Image
import yaml
from datetime import datetime
import time
import base64
import glob
import warnings
import json
import math
warnings.filterwarnings("ignore")

# Add ByteTrack to sys.path
BYTE_TRACK_DIR = Path(r"C:\Users\iampr\Desktop\Drone intern\ByteTrack")
sys.path.append(str(BYTE_TRACK_DIR))
try:
    from yolox.tracker.byte_tracker import BYTETracker
except ImportError:
    st.error("ByteTrack not found in ByteTrack/. Ensure yolox/tracker/byte_tracker.py exists.")
    st.stop()

# Paths
BASE_DIR = Path(r"C:\Users\iampr\Desktop\Drone intern")
MODEL_PATH = BASE_DIR / "best_drone_model.pt"
DATA_YAML = BASE_DIR / "data.yaml"
TEMP_INPUT_DIR = BASE_DIR / "temp_input"

# Vehicle weights for pollution score
VEHICLE_WEIGHTS = {2: 1.0, 3: 0.3, 5: 2.5, 7: 3.0}  # car=2, motorcycle=3, bus=5, truck=7

# Load class names from data.yaml
try:
    with open(DATA_YAML, 'r') as f:
        data_config = yaml.safe_load(f)
        CLASS_NAMES = data_config.get('names', {})
except FileNotFoundError:
    CLASS_NAMES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
    st.warning("data.yaml not found. Using default class names.")

# Streamlit page config
st.set_page_config(page_title="DroneSync Traffic AI", layout="wide", initial_sidebar_state="expanded")

# Glassmorphism CSS and JavaScript for particle effects
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css');
    .main {
        background: linear-gradient(135deg, #0a0a23, #1c1c3f);
        color: #e0e0ff;
        font-family: 'Inter', sans-serif;
        position: relative;
        overflow: hidden;
    }
    .particle-bg {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: 0;
    }
    .sidebar .sidebar-content {
        background: rgba(28, 28, 63, 0.7);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(0, 255, 136, 0.2);
        padding: 20px;
    }
    .nav-item {
        display: flex;
        align-items: center;
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 8px;
        transition: all 0.3s ease;
        cursor: pointer;
        color: #e0e0ff;
        font-size: 16px;
        font-weight: 600;
    }
    .nav-item:hover {
        background: rgba(0, 255, 136, 0.15);
        box-shadow: 0 0 15px rgba(0, 255, 136, 0.3);
        transform: translateX(5px);
    }
    .nav-item.active {
        background: rgba(0, 255, 136, 0.25);
        box-shadow: 0 0 20px rgba(0, 255, 136, 0.5);
    }
    .nav-icon {
        margin-right: 12px;
        font-size: 18px;
    }
    .sub-nav-item {
        padding: 8px 16px 8px 40px;
        font-size: 14px;
        color: #b0b0ff;
        transition: all 0.3s ease;
    }
    .sub-nav-item:hover {
        color: #e0e0ff;
        background: rgba(0, 255, 136, 0.1);
    }
    .stButton>button {
        background: linear-gradient(45deg, #00ff88, #00b4d8);
        color: #0a0a23;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        box-shadow: 0 4px 15px rgba(0, 255, 136, 0.4);
        transition: all 0.3s ease;
        font-weight: 600;
    }
    .stButton>button:hover {
        background: linear-gradient(45deg, #00b4d8, #00ff88);
        box-shadow: 0 6px 20px rgba(0, 255, 136, 0.6);
        transform: translateY(-2px);
    }
    .stButton>button:disabled {
        background: #4a4a6a;
        box-shadow: none;
        cursor: not-allowed;
    }
    .glass-card {
        background: rgba(28, 28, 63, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 255, 136, 0.2);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px rgba(0, 255, 136, 0.2);
        transition: transform 0.3s ease;
        position: relative;
        z-index: 1;
    }
    .glass-card:hover {
        transform: translateY(-5px);
    }
    .metric-card {
        background: rgba(28, 28, 63, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        color: #e0e0ff;
        box-shadow: 0 4px 15px rgba(0, 255, 136, 0.2);
        margin: 10px 0;
    }
    .congestion-meter {
        background: linear-gradient(90deg, #00ff88, #ff00ff);
        height: 20px;
        border-radius: 10px;
        transition: width 0.5s ease;
        box-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
    }
    .radial-meter {
        width: 150px;
        height: 150px;
        margin: auto;
        position: relative;
    }
    .video-container {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(0, 255, 136, 0.2);
        background: rgba(10, 10, 35, 0.9);
    }
    .control-panel {
        display: flex;
        gap: 10px;
        padding: 10px;
        background: rgba(28, 28, 63, 0.7);
        border-radius: 8px;
        box-shadow: 0 4px 15px rgba(0, 255, 136, 0.2);
        margin-top: 10px;
    }
    .loader {
        border: 6px solid #00ff88;
        border-top: 6px solid #00b4d8;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        animation: spin 1s linear infinite;
        margin: auto;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    .toast {
        position: fixed;
        top: 20px;
        right: 20px;
        background: rgba(0, 255, 136, 0.9);
        color: #0a0a23;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 4px 15px rgba(0, 255, 136, 0.4);
        z-index: 1000;
        animation: slideIn 0.5s ease, fadeOut 0.5s ease 2.5s forwards;
    }
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    @keyframes fadeOut {
        from { opacity: 1; }
        to { opacity: 0; }
    }
    .stSlider .st-bq {
        background: rgba(28, 28, 63, 0.7);
        border: 1px solid #00ff88;
        border-radius: 8px;
    }
    .stSlider .st-br {
        background: #00ff88;
    }
    .stSelectbox .st-c3 {
        background: rgba(28, 28, 63, 0.7);
        border: 1px solid #00ff88;
        border-radius: 8px;
        color: #e0e0ff;
    }
    .stSelectbox .st-c4 {
        background: #00ff88;
    }
    .plotly-chart {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(0, 255, 136, 0.2);
    }
    </style>
    <script>
    // Particle background animation
    function createParticleCanvas() {
        const canvas = document.createElement('canvas');
        canvas.className = 'particle-bg';
        document.querySelector('.main').prepend(canvas);
        const ctx = canvas.getContext('2d');
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
        const particles = [];
        const numParticles = 100;
        for (let i = 0; i < numParticles; i++) {
            particles.push({
                x: Math.random() * canvas.width,
                y: Math.random() * canvas.height,
                radius: Math.random() * 2 + 1,
                vx: Math.random() * 0.5 - 0.25,
                vy: Math.random() * 0.5 - 0.25,
                color: 'rgba(0, 255, 136, 0.5)'
            });
        }
        function animateParticles() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            particles.forEach(p => {
                p.x += p.vx;
                p.y += p.vy;
                if (p.x < 0 || p.x > canvas.width) p.vx *= -1;
                if (p.y < 0 || p.y > canvas.height) p.vy *= -1;
                ctx.beginPath();
                ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
                ctx.fillStyle = p.color;
                ctx.fill();
                particles.forEach(other => {
                    const dx = p.x - other.x;
                    const dy = p.y - other.y;
                    const distance = Math.sqrt(dx * dx + dy * dy);
                    if (distance < 100) {
                        ctx.beginPath();
                        ctx.moveTo(p.x, p.y);
                        ctx.lineTo(other.x, other.y);
                        ctx.strokeStyle = 'rgba(0, 255, 136, 0.1)';
                        ctx.stroke();
                    }
                });
            });
            requestAnimationFrame(animateParticles);
        }
        animateParticles();
        window.addEventListener('resize', () => {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
        });
    }
    document.addEventListener('DOMContentLoaded', createParticleCanvas);
    // Toast notification
    function showToast(message, type = 'success') {
        const toast = document.createElement('div');
        toast.className = 'toast';
        toast.style.background = type === 'success' ? 'rgba(0, 255, 136, 0.9)' : 'rgba(255, 0, 255, 0.9)';
        toast.innerText = message;
        document.body.appendChild(toast);
        setTimeout(() => toast.remove(), 3000);
    }
    </script>
""", unsafe_allow_html=True)

# Initialize session state
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'paused' not in st.session_state:
    st.session_state.paused = False
if 'vehicle_count' not in st.session_state:
    st.session_state.vehicle_count = 0
if 'density' not in st.session_state:
    st.session_state.density = 0.0
if 'pollution' not in st.session_state:
    st.session_state.pollution = 0.0
if 'detections' not in st.session_state:
    st.session_state.detections = []
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 'Live Detection'
if 'active_sub_tab' not in st.session_state:
    st.session_state.active_sub_tab = None
if 'playback_speed' not in st.session_state:
    st.session_state.playback_speed = 1.0
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True
if 'show_labels' not in st.session_state:
    st.session_state.show_labels = True
if 'show_heatmap' not in st.session_state:
    st.session_state.show_heatmap = True
if 'frame_index' not in st.session_state:
    st.session_state.frame_index = 0
if 'cap' not in st.session_state:
    st.session_state.cap = None
if 'video_path' not in st.session_state:
    st.session_state.video_path = None
if 'movement_trails' not in st.session_state:
    st.session_state.movement_trails = {}
if 'zone_counts' not in st.session_state:
    st.session_state.zone_counts = {}

# Backend Functions
@st.cache_resource
def load_model():
    """Load YOLOv8 model."""
    try:
        model = YOLO(MODEL_PATH)
        if torch.cuda.is_available():
            model.to('cuda')
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

@st.cache_resource
def load_tracker():
    """Initialize ByteTrack."""
    try:
        return BYTETracker(frame_rate=30, track_thresh=0.5, match_thresh=0.8)
    except Exception as e:
        st.error(f"Failed to load ByteTrack: {str(e)}")
        return None

def detect_objects(frame, model, tracker, conf=0.3):
    """Run YOLOv8 detection and ByteTrack."""
    results = model(frame, conf=conf, classes=[2, 3, 5, 7])
    detections = []
    boxes = results[0].boxes
    if boxes is not None:
        dets = np.hstack((boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()[:, None], boxes.cls.cpu().numpy()[:, None]))
        tracks = tracker.update(dets)
        for track in tracks:
            x1, y1, x2, y2, track_id, conf, cls = track[:7]
            detections.append({
                'track_id': int(track_id),
                'class': int(cls),
                'class_name': CLASS_NAMES.get(int(cls), "Unknown"),
                'bbox': [x1, y1, x2, y2],
                'conf': conf,
                'center': [(x1 + x2) / 2, (y1 + y2) / 2]
            })
    frame_rgb = cv2.cvtColor(results[0].plot(labels=st.session_state.show_labels), cv2.COLOR_BGR2RGB)
    return frame_rgb, detections

def draw_movement_trails(frame, detections):
    """Draw movement trails for tracked vehicles."""
    for det in detections:
        track_id = det['track_id']
        center = det['center']
        if track_id not in st.session_state.movement_trails:
            st.session_state.movement_trails[track_id] = []
        st.session_state.movement_trails[track_id].append(center)
        st.session_state.movement_trails[track_id] = st.session_state.movement_trails[track_id][-10:]  # Keep last 10 points
        for i in range(1, len(st.session_state.movement_trails[track_id])):
            pt1 = tuple(map(int, st.session_state.movement_trails[track_id][i-1]))
            pt2 = tuple(map(int, st.session_state.movement_trails[track_id][i]))
            alpha = i / len(st.session_state.movement_trails[track_id])
            color = (0, int(255 * alpha), int(136 * alpha))
            cv2.line(frame, pt1, pt2, color, 2)
    return frame

def calculate_density(detections, frame_area):
    """Calculate traffic density."""
    return len(detections) / (frame_area / 1e6) if frame_area else 0

def calculate_pollution(detections):
    """Calculate pollution score."""
    score = 0
    for det in detections:
        cls = det['class']
        score += VEHICLE_WEIGHTS.get(cls, 1.0)
    return score

def draw_heatmap(detections, frame_shape, sigma=50):
    """Generate congestion heatmap."""
    h, w = frame_shape[:2]
    heatmap = np.zeros((h, w))
    for det in detections:
        x, y = int(det['center'][0]), int(det['center'][1])
        if 0 <= x < w and 0 <= y < h:
            for i in range(-sigma, sigma + 1):
                for j in range(-sigma, sigma + 1):
                    if 0 <= x + i < w and 0 <= y + j < h:
                        heatmap[y + j, x + i] += np.exp(-(i**2 + j**2) / (2 * sigma**2))
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

def assign_zone(center, frame_shape):
    """Assign detection to a zone based on position."""
    h, w = frame_shape[:2]
    x, y = center
    zone_x = int(x / (w / 4))  # Divide frame into 4x4 grid
    zone_y = int(y / (h / 4))
    return f"Zone-{zone_x}-{zone_y}"

def save_uploaded_file(uploaded_file):
    """Save uploaded file to temp_input."""
    TEMP_INPUT_DIR.mkdir(exist_ok=True)
    file_path = TEMP_INPUT_DIR / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(file_path)

@st.cache_data
def plot_vehicle_count(df):
    """Line chart of vehicle count."""
    fig = px.line(df, x='Timestamp', y='Vehicles', title='Vehicle Count Over Time',
                  color_discrete_sequence=['#00ff88'])
    fig.update_layout(paper_bgcolor='rgba(28,28,63,0.7)', plot_bgcolor='rgba(28,28,63,0.7)',
                      font=dict(color='#e0e0ff'), margin=dict(l=20, r=20, t=50, b=20))
    return fig

@st.cache_data
def plot_pollution(df):
    """Bar chart of pollution score."""
    fig = px.bar(df, x='Timestamp', y='Pollution', title='Pollution Score Over Time',
                 color_discrete_sequence=['#ff00ff'])
    fig.update_layout(paper_bgcolor='rgba(28,28,63,0.7)', plot_bgcolor='rgba(28,28,63,0.7)',
                      font=dict(color='#e0e0ff'), margin=dict(l=20, r=20, t=50, b=20))
    return fig

@st.cache_data
def plot_density(df):
    """Line chart of traffic density."""
    fig = px.line(df, x='Timestamp', y='Density', title='Traffic Density (Vehicles/km²)',
                  color_discrete_sequence=['#00b4d8'])
    fig.update_layout(paper_bgcolor='rgba(28,28,63,0.7)', plot_bgcolor='rgba(28,28,63,0.7)',
                      font=dict(color='#e0e0ff'), margin=dict(l=20, r=20, t=50, b=20))
    return fig

@st.cache_data
def plot_zone_counts(df):
    """Bar chart of zone congestion."""
    zone_counts = df.groupby('Zone')['Vehicles'].sum().sort_values(ascending=False).head(10)
    fig = px.bar(x=zone_counts.index, y=zone_counts.values, title='Top Congested Zones',
                 labels={'x': 'Zone', 'y': 'Vehicle Count'}, color=zone_counts.index,
                 color_discrete_sequence=['#00ff88', '#00b4d8', '#ff00ff', '#ff9900', '#9900ff'])
    fig.update_layout(paper_bgcolor='rgba(28,28,63,0.7)', plot_bgcolor='rgba(28,28,63,0.7)',
                      font=dict(color='#e0e0ff'), margin=dict(l=20, r=20, t=50, b=20))
    return fig

@st.cache_data
def plot_3d_vehicle_distribution(df):
    """3D scatter plot of vehicle positions."""
    if 'Center_X' not in df or 'Center_Y' not in df:
        return None
    fig = go.Figure(data=[
        go.Scatter3d(
            x=df['Center_X'],
            y=df['Center_Y'],
            z=df['Frame'],
            mode='markers',
            marker=dict(
                size=5,
                color=df['Vehicles'],
                colorscale='Viridis',
                opacity=0.8
            ),
            text=df['Class_Name']
        )
    ])
    fig.update_layout(
        scene=dict(
            xaxis_title='X Position',
            yaxis_title='Y Position',
            zaxis_title='Frame',
            bgcolor='rgba(28,28,63,0.7)'
        ),
        paper_bgcolor='rgba(28,28,63,0.7)',
        font=dict(color='#e0e0ff'),
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

# UI Components
def sidebar():
    """Render glassmorphism sidebar with collapsible menus."""
    with st.sidebar:
        st.markdown('<div style="text-align: center; margin-bottom: 20px;"><h2>🚁 DroneSync AI</h2></div>', unsafe_allow_html=True)
        tabs = [
            ("Live Detection", "fa-video"),
            ("Upload & Run", "fa-upload"),
            ("Analytics", "fa-chart-line"),
            ("Export", "fa-download"),
            ("Settings", "fa-cog"),
            ("Help", "fa-question-circle")
        ]
        for tab_name, icon in tabs:
            active = tab_name == st.session_state.active_tab
            st.markdown(
                f'<div class="nav-item {"active" if active else ""}" onclick="window.location.hash=\'{tab_name.lower().replace(" ", "-")}\'">'
                f'<i class="fa {icon} nav-icon"></i>{tab_name}</div>',
                unsafe_allow_html=True
            )
            if active:
                st.session_state.active_tab = tab_name
            if tab_name == "Analytics" and active:
                sub_tabs = ["Vehicle Stats", "Pollution Trends", "Zone Analysis", "3D Distribution"]
                for sub_tab in sub_tabs:
                    sub_active = st.session_state.active_sub_tab == sub_tab
                    st.markdown(
                        f'<div class="sub-nav-item {"active" if sub_active else ""}" onclick="window.location.hash=\'{sub_tab.lower().replace(" ", "-")}\'">'
                        f'{sub_tab}</div>',
                        unsafe_allow_html=True
                    )
                    if sub_active:
                        st.session_state.active_sub_tab = sub_tab
        st.markdown('<hr style="border-color: rgba(0, 255, 136, 0.2); margin: 20px 0;">', unsafe_allow_html=True)
        st.session_state.dark_mode = st.checkbox("🌙 Dark Mode", value=True)
        st.session_state.show_labels = st.checkbox("🏷️ Show Labels", value=True)
        st.session_state.show_heatmap = st.checkbox("🔥 Show Heatmap", value=True)

def radial_congestion_meter(value, max_value=50):
    """Render radial congestion meter."""
    percentage = min(value / max_value, 1.0) * 100
    angle = percentage * 3.6
    st.markdown(f"""
        <div class="radial-meter">
            <svg viewBox="0 0 100 100">
                <circle cx="50" cy="50" r="45" fill="none" stroke="#4a4a6a" stroke-width="10"/>
                <circle cx="50" cy="50" r="45" fill="none" stroke="#00ff88"
                        stroke-width="10" stroke-dasharray="{angle*2.83},283"
                        transform="rotate(-90 50 50)"/>
                <text x="50" y="55" text-anchor="middle" fill="#e0e0ff" font-size="20">{percentage:.1f}%</text>
            </svg>
        </div>
    """, unsafe_allow_html=True)

# Tab Functions
def live_detection():
    """Live Detection tab."""
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🚁 Live Drone Feed")
    col1, col2 = st.columns([3, 1])
    with col1:
        video_file = st.file_uploader("Upload Video (.mp4, .avi)", type=["mp4", "avi"], key="live_video")
        frame_placeholder = st.empty()
        heatmap_placeholder = st.empty()
    with col2:
        conf_threshold = st.slider("Confidence", 0.1, 0.9, 0.3, 0.05, key="live_conf")
        st.session_state.playback_speed = st.slider("Playback Speed", 0.5, 2.0, 1.0, 0.1, key="live_speed")
        col2_1, col2_2, col2_3 = st.columns(3)
        with col2_1:
            if st.button("▶️ Play", disabled=st.session_state.processing and not st.session_state.paused):
                st.session_state.processing = True
                st.session_state.paused = False
                st.session_state.frame_index = 0
                st.session_state.detections = []
                st.session_state.movement_trails = {}
                st.session_state.zone_counts = {}
        with col2_2:
            if st.button("⏸️ Pause", disabled=not st.session_state.processing or st.session_state.paused):
                st.session_state.paused = True
        with col2_3:
            if st.button("🛑 Stop", disabled=not st.session_state.processing):
                st.session_state.processing = False
                st.session_state.paused = False
        frame_slider = st.slider("Frame", 0, 1000, st.session_state.frame_index, key="frame_slider")

    if video_file and st.session_state.processing and not st.session_state.paused:
        if st.session_state.video_path != video_file.name:
            st.session_state.video_path = save_uploaded_file(video_file)
            st.session_state.cap = cv2.VideoCapture(st.session_state.video_path)
        model = load_model()
        tracker = load_tracker()
        if not model or not tracker or not st.session_state.cap:
            st.session_state.processing = False
            st.error("Model, tracker, or video failed to load.")
            return
        frame_area = st.session_state.cap.get(3) * st.session_state.cap.get(4)
        total_frames = int(st.session_state.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = st.session_state.cap.get(cv2.CAP_PROP_FPS)
        frame_time = 1 / (fps * st.session_state.playback_speed)

        st.session_state.cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.frame_index)
        ret, frame = st.session_state.cap.read()
        if not ret:
            st.session_state.processing = False
            st.session_state.cap.release()
            st.session_state.cap = None
            st.success("Video processing completed.")
            st.markdown('<script>showToast("Video processing completed.", "success");</script>', unsafe_allow_html=True)
            return
        frame_rgb, detections = detect_objects(frame, model, tracker, conf_threshold)
        frame_rgb = draw_movement_trails(frame_rgb, detections)
        st.session_state.vehicle_count = len(detections)
        st.session_state.density = calculate_density(detections, frame_area)
        st.session_state.pollution = calculate_pollution(detections)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for det in detections:
            zone = assign_zone(det['center'], frame.shape)
            st.session_state.zone_counts[zone] = st.session_state.zone_counts.get(zone, 0) + 1
            st.session_state.detections.append({
                'Timestamp': timestamp,
                'Frame': st.session_state.frame_index,
                'Vehicles': st.session_state.vehicle_count,
                'Density': st.session_state.density,
                'Pollution': st.session_state.pollution,
                'Zone': zone,
                'Class_Name': det['class_name'],
                'Center_X': det['center'][0],
                'Center_Y': det['center'][1]
            })
        frame_placeholder.image(frame_rgb, channels="RGB", caption=f"Frame {st.session_state.frame_index}")
        if st.session_state.show_heatmap and detections:
            heatmap = draw_heatmap(detections, frame.shape)
            heatmap_placeholder.image(heatmap, caption="Congestion Heatmap", use_column_width=True)
        st.session_state.frame_index = min(st.session_state.frame_index + 1, total_frames - 1)
        time.sleep(frame_time)
        st.rerun()

    if st.session_state.frame_index != frame_slider:
        st.session_state.frame_index = frame_slider
        st.rerun()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("🚗 Vehicles", st.session_state.vehicle_count)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("📏 Density (veh/km²)", f"{st.session_state.density:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("💨 Pollution Score", f"{st.session_state.pollution:.1f}")
        st.markdown('</div>', unsafe_allow_html=True)
    radial_congestion_meter(st.session_state.density)
    st.markdown('</div>', unsafe_allow_html=True)

def upload_and_run():
    """Upload & Run tab."""
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📤 Full Video Analysis")
    video_file = st.file_uploader("Upload Video (.mp4, .avi)", type=["mp4", "avi"], key="upload_run")
    conf_threshold = st.slider("Confidence", 0.1, 0.9, 0.3, 0.05, key="upload_conf")
    progress_bar = st.progress(0.0)
    status_placeholder = st.empty()
    frame_placeholder = st.empty()
    heatmap_placeholder = st.empty()

    if video_file and st.button("🚀 Run Analysis", disabled=st.session_state.processing):
        model = load_model()
        tracker = load_tracker()
        if not model or not tracker:
            st.error("Model or tracker failed to load.")
            return
        video_path = save_uploaded_file(video_file)
        cap = cv2.VideoCapture(video_path)
        frame_area = cap.get(3) * cap.get(4)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        st.session_state.detections = []
        st.session_state.movement_trails = {}
        st.session_state.zone_counts = {}
        frame_count = 0
        st.session_state.processing = True

        while cap.isOpened() and st.session_state.processing:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb, detections = detect_objects(frame, model, tracker, conf_threshold)
            frame_rgb = draw_movement_trails(frame_rgb, detections)
            density = calculate_density(detections, frame_area)
            pollution = calculate_pollution(detections)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for det in detections:
                zone = assign_zone(det['center'], frame.shape)
                st.session_state.zone_counts[zone] = st.session_state.zone_counts.get(zone, 0) + 1
                st.session_state.detections.append({
                    'Timestamp': timestamp,
                    'Frame': frame_count,
                    'Vehicles': len(detections),
                    'Density': density,
                    'Pollution': pollution,
                    'Zone': zone,
                    'Class_Name': det['class_name'],
                    'Center_X': det['center'][0],
                    'Center_Y': det['center'][1]
                })
            frame_count += 1
            progress = frame_count / total_frames
            progress_bar.progress(progress)
            status_placeholder.text(f"Processing frame {frame_count}/{total_frames}")
            frame_placeholder.image(frame_rgb, channels="RGB", caption=f"Frame {frame_count}")
            if st.session_state.show_heatmap and detections:
                heatmap = draw_heatmap(detections, frame.shape)
                heatmap_placeholder.image(heatmap, caption="Congestion Heatmap", use_column_width=True)
        cap.release()
        st.session_state.processing = False
        progress_bar.progress(1.0)
        status_placeholder.text("Analysis completed!")
        st.markdown('<script>showToast("Analysis completed.", "success");</script>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

def analytics():
    """Analytics tab with sub-menus."""
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📊 Traffic Insights")
    if not st.session_state.detections:
        st.info("No data available. Run an analysis first.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    df = pd.DataFrame(st.session_state.detections)
    sub_tab = st.session_state.active_sub_tab or "Vehicle Stats"
    col1, col2 = st.columns(2)

    if sub_tab == "Vehicle Stats":
        with col1:
            st.plotly_chart(plot_vehicle_count(df), use_container_width=True, cls="plotly-chart")
        with col2:
            st.plotly_chart(plot_density(df), use_container_width=True, cls="plotly-chart")
        st.markdown('<hr style="border-color: rgba(0, 255, 136, 0.2); margin: 20px 0;">', unsafe_allow_html=True)
        st.subheader("Vehicle Class Breakdown")
        class_counts = df['Class_Name'].value_counts()
        fig = px.pie(values=class_counts.values, names=class_counts.index, title='Vehicle Class Distribution',
                     color_discrete_sequence=['#00ff88', '#00b4d8', '#ff00ff', '#ff9900'])
        fig.update_layout(paper_bgcolor='rgba(28,28,63,0.7)', font=dict(color='#e0e0ff'))
        st.plotly_chart(fig, use_container_width=True, cls="plotly-chart")
    elif sub_tab == "Pollution Trends":
        with col1:
            st.plotly_chart(plot_pollution(df), use_container_width=True, cls="plotly-chart")
        with col2:
            st.subheader("Pollution by Vehicle Type")
            pollution_by_class = df.groupby('Class_Name')['Pollution'].sum()
            fig = px.bar(x=pollution_by_class.index, y=pollution_by_class.values, title='Pollution by Vehicle Type',
                         color=pollution_by_class.index, color_discrete_sequence=['#ff00ff', '#9900ff', '#ff9900'])
            fig.update_layout(paper_bgcolor='rgba(28,28,63,0.7)', font=dict(color='#e0e0ff'))
            st.plotly_chart(fig, use_container_width=True, cls="plotly-chart")
    elif sub_tab == "Zone Analysis":
        with col1:
            st.plotly_chart(plot_zone_counts(df), use_container_width=True, cls="plotly-chart")
        with col2:
            st.subheader("Zone Density Map")
            zone_density = df.groupby('Zone')['Density'].mean()
            fig = px.bar(x=zone_density.index, y=zone_density.values, title='Average Density by Zone',
                         color=zone_density.index, color_discrete_sequence=['#00b4d8', '#00ff88', '#ff00ff'])
            fig.update_layout(paper_bgcolor='rgba(28,28,63,0.7)', font=dict(color='#e0e0ff'))
            st.plotly_chart(fig, use_container_width=True, cls="plotly-chart")
    elif sub_tab == "3D Distribution":
        fig = plot_3d_vehicle_distribution(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True, cls="plotly-chart")
        else:
            st.warning("3D distribution data unavailable.")
    st.markdown('<hr style="border-color: rgba(0, 255, 136, 0.2); margin: 20px 0;">', unsafe_allow_html=True)
    st.subheader("Summary Statistics")
    st.dataframe(df.describe(), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

def export():
    """Export tab."""
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📁 Export Analytics")
    if st.session_state.detections:
        df = pd.DataFrame(st.session_state.detections)
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        href = f'<a href="data:file/csv;base64,{b64}" download="traffic_analytics_{timestamp}.csv">📥 Download CSV</a>'
        st.markdown(href, unsafe_allow_html=True)
        st.markdown('<hr style="border-color: rgba(0, 255, 136, 0.2); margin: 20px 0;">', unsafe_allow_html=True)
        st.subheader("Data Preview")
        st.dataframe(df, use_container_width=True)
        if st.button("📊 Download All Plots"):
            plots = {
                "vehicle_count.html": plot_vehicle_count(df),
                "pollution.html": plot_pollution(df),
                "density.html": plot_density(df),
                "zone_counts.html": plot_zone_counts(df)
            }
            for name, fig in plots.items():
                fig.write_html(name)
                with open(name, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                    href = f'<a href="data:text/html;base64,{b64}" download="{name}">Download {name.replace(".html", "")}</a>'
                    st.markdown(href, unsafe_allow_html=True)
            st.markdown('<script>showToast("Plots downloaded.", "success");</script>', unsafe_allow_html=True)
    else:
        st.info("No data available. Run an analysis first.")
    st.markdown('</div>', unsafe_allow_html=True)

def settings():
    """Settings tab."""
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("⚙️ Configuration")
    col1, col2 = st.columns(2)
    with col1:
        st.slider("Model Confidence", 0.1, 0.9, 0.3, 0.05, key="global_conf")
        st.session_state.playback_speed = st.slider("Default Playback Speed", 0.5, 2.0, 1.0, 0.1, key="global_speed")
        st.session_state.show_labels = st.checkbox("Show Bounding Box Labels", value=st.session_state.show_labels, key="global_labels")
        st.session_state.show_heatmap = st.checkbox("Enable Congestion Heatmap", value=st.session_state.show_heatmap, key="global_heatmap")
    with col2:
        st.multiselect("Vehicle Classes", list(CLASS_NAMES.values()), default=list(CLASS_NAMES.values()), key="global_classes")
        st.selectbox("Heatmap Style", ["Jet", "Hot", "Viridis"], key="heatmap_style")
        st.number_input("Heatmap Sigma", 10, 100, 50, 5, key="heatmap_sigma")
    if st.button("💾 Save Settings"):
        st.markdown('<script>showToast("Settings saved.", "success");</script>', unsafe_allow_html=True)
    st.markdown('<hr style="border-color: rgba(0, 255, 136, 0.2); margin: 20px 0;">', unsafe_allow_html=True)
    st.subheader("System Info")
    st.write(f"Model Path: {MODEL_PATH}")
    st.write(f"ByteTrack Path: {BYTE_TRACK_DIR}")
    st.write(f"Streamlit Version: {st.__version__}")
    st.markdown('</div>', unsafe_allow_html=True)

def help_tab():
    """Help tab."""
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("❓ Help & Documentation")
    st.markdown("""
        ### Welcome to DroneSync Traffic AI
        This application provides real-time traffic analysis using drone footage, powered by YOLOv8 and ByteTrack.

        #### Getting Started
        1. **Live Detection**: Upload a video to view real-time vehicle detection with movement trails and heatmaps.
        2. **Upload & Run**: Process an entire video for detailed analytics.
        3. **Analytics**: Explore vehicle counts, pollution trends, zone congestion, and 3D distributions.
        4. **Export**: Download frame-wise data as CSV and visualization plots.
        5. **Settings**: Configure confidence, playback speed, and visualization options.
        6. **Help**: Refer to this guide for assistance.

        #### Features
        - **Real-Time Detection**: YOLOv8 detects vehicles (cars, motorcycles, buses, trucks) with ByteTrack for multi-object tracking.
        - **Analytics**: Vehicle counts, traffic density, pollution scores, and zone-based congestion.
        - **Visualizations**: Interactive Plotly charts, dynamic heatmaps, and movement trails.
        - **Export**: CSV files with timestamped data and downloadable HTML plots.
        - **UI**: Glassmorphism design with particle animations, responsive layout, and dark/light mode.

        #### Troubleshooting
        - **Model Loading Error**: Ensure `best_drone_model.pt` exists at `C:\Users\iampr\Desktop\Drone intern`.
        - **ByteTrack Error**: Verify `ByteTrack/yolox/tracker/byte_tracker.py` is present.
        - **Video Issues**: Only `.mp4` and `.avi` formats are supported.
        - **Performance**: For GPU acceleration, install `torch` with CUDA support.

        #### Contact
        For support, contact the development team via the project repository or email.

        ### Keyboard Shortcuts
        - **Play/Pause**: Spacebar
        - **Stop**: Esc
        - **Next Frame**: Right Arrow
        - **Previous Frame**: Left Arrow
    """)
    st.markdown('<hr style="border-color: rgba(0, 255, 136, 0.2); margin: 20px 0;">', unsafe_allow_html=True)
    st.subheader("Sample Data")
    if st.button("📂 Load Sample Analytics"):
        sample_data = [
            {'Timestamp': '2025-06-15 12:27:00', 'Frame': i, 'Vehicles': np.random.randint(0, 20),
             'Density': np.random.uniform(0, 50), 'Pollution': np.random.uniform(0, 100),
             'Zone': f"Zone-{i%4}-{i%4}", 'Class_Name': np.random.choice(list(CLASS_NAMES.values())),
             'Center_X': np.random.uniform(0, 1280), 'Center_Y': np.random.uniform(0, 720)}
            for i in range(100)
        ]
        st.session_state.detections = sample_data
        st.markdown('<script>showToast("Sample data loaded.", "success");</script>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Main App
def main():
    sidebar()
    tab_map = {
        'Live Detection': live_detection,
        'Upload & Run': upload_and_run,
        'Analytics': analytics,
        'Export': export,
        'Settings': settings,
        'Help': help_tab
    }
    tab_map[st.session_state.active_tab]()
    # JavaScript for keyboard shortcuts
    st.markdown("""
        <script>
        document.addEventListener('keydown', (e) => {
            if (e.key === ' ') {
                e.preventDefault();
                document.querySelector('[data-testid="stButton"] button').click();
            } else if (e.key === 'Escape') {
                document.querySelectorAll('[data-testid="stButton"] button')[2]?.click();
            } else if (e.key === 'ArrowRight') {
                const slider = document.querySelector('[data-testid="stSlider"] input');
                if (slider) slider.value = parseInt(slider.value) + 1;
            } else if (e.key === 'ArrowLeft') {
                const slider = document.querySelector('[data-testid="stSlider"] input');
                if (slider) slider.value = parseInt(slider.value) - 1;
            }
        });
        </script>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()