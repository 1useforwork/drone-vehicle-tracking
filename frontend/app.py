import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import time
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Vehicle Detection & Tracking System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .detection-box {
        border: 2px solid #28a745;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []
if 'tracking_data' not in st.session_state:
    st.session_state.tracking_data = []
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0

def main():
    st.markdown('<h1 class="main-header">🚗 Vehicle Detection & Tracking System</h1>', unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model Selection
        st.subheader("Model Settings")
        model_options = [
            "YOLOv8-nano + ByteTrack",
            "YOLOv8-small + BoT-SORT v1.3",
            "YOLOv8-medium + DeepSORT",
            "YOLOv8-large + OCSORT",
            "YOLOv5-small + StrongSORT"
        ]
        selected_model = st.selectbox("Select Detection Model", model_options, index=1)
        
        # Input Mode Selection
        st.subheader("Input Mode")
        input_mode = st.radio(
            "Choose input source:",
            ["Upload Video", "Live Stream", "Webcam"],
            help="Select how you want to provide video input"
        )
        
        # Detection Settings
        st.subheader("Detection Parameters")
        confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
        nms_threshold = st.slider("NMS Threshold", 0.1, 1.0, 0.4, 0.05)
        
        # Vehicle Classes
        st.subheader("Vehicle Classes")
        vehicle_classes = st.multiselect(
            "Select vehicle types to detect:",
            ["Car", "Truck", "Bus", "Motorcycle", "Bicycle", "Van"],
            default=["Car", "Truck", "Bus"]
        )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Video Processing")
        
        if input_mode == "Upload Video":
            uploaded_file = st.file_uploader(
                "Choose a video file",
                type=['mp4', 'avi', 'mov', 'mkv'],
                help="Upload a video file for vehicle detection"
            )
            
            if uploaded_file is not None:
                # Save uploaded file temporarily
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded_file.read())
                
                # Process video
                process_uploaded_video(tfile.name, selected_model, confidence_threshold, vehicle_classes)
                
        elif input_mode == "Live Stream":
            st.info("🔴 Live Stream Mode")
            stream_url = st.text_input("Enter stream URL (RTMP/HTTP):", placeholder="rtmp://example.com/stream")
            
            col_start, col_stop = st.columns(2)
            with col_start:
                start_stream = st.button("▶️ Start Stream", use_container_width=True)
            with col_stop:
                stop_stream = st.button("⏹️ Stop Stream", use_container_width=True)
            
            if start_stream and stream_url:
                process_live_stream(stream_url, selected_model, confidence_threshold, vehicle_classes)
                
        else:  # Webcam
            st.info("📷 Webcam Mode")
            col_start, col_stop = st.columns(2)
            with col_start:
                start_webcam = st.button("▶️ Start Webcam", use_container_width=True)
            with col_stop:
                stop_webcam = st.button("⏹️ Stop Webcam", use_container_width=True)
            
            if start_webcam:
                process_webcam(selected_model, confidence_threshold, vehicle_classes)
    
    with col2:
        st.subheader("📊 Real-time Metrics")
        
        # Display current model info
        st.markdown(f"""
        <div class="metric-card">
            <h4>🤖 Current Model</h4>
            <p><strong>{selected_model}</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Performance metrics
        display_performance_metrics()
        
        # Detection statistics
        display_detection_stats()
    
    # Bottom section - Analytics
    st.markdown("---")
    st.subheader("📈 Analytics Dashboard")
    
    analytics_tab1, analytics_tab2, analytics_tab3 = st.tabs(["Detection History", "Tracking Performance", "Vehicle Counts"])
    
    with analytics_tab1:
        display_detection_history()
    
    with analytics_tab2:
        display_tracking_performance()
    
    with analytics_tab3:
        display_vehicle_counts()

def process_uploaded_video(video_path, model_name, confidence, vehicle_classes):
    """Process uploaded video file"""
    st.info(f"Processing video with {model_name}...")
    
    # Placeholder for actual video processing
    video_placeholder = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Simulate video processing
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    frame_count = 0
    detections = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Simulate detection processing
        frame_detections = simulate_detection(frame, vehicle_classes, confidence)
        detections.extend(frame_detections)
        
        # Update progress
        frame_count += 1
        progress = frame_count / total_frames
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {frame_count}/{total_frames}")
        
        # Display processed frame (every 10th frame to avoid overwhelming)
        if frame_count % 10 == 0:
            processed_frame = draw_detections(frame, frame_detections)
            video_placeholder.image(processed_frame, channels="BGR", use_column_width=True)
        
        # Break after processing some frames for demo
        if frame_count > 100:
            break
    
    cap.release()
    os.unlink(video_path)  # Clean up temporary file
    
    # Update session state
    st.session_state.detection_history.extend(detections)
    st.session_state.frame_count += frame_count
    
    st.success(f"✅ Video processed! Detected {len(detections)} vehicles across {frame_count} frames.")

def process_live_stream(stream_url, model_name, confidence, vehicle_classes):
    """Process live stream"""
    st.info(f"🔴 Processing live stream with {model_name}...")
    
    # Placeholder for live stream processing
    stream_placeholder = st.empty()
    metrics_placeholder = st.empty()
    
    # Simulate live stream processing
    for i in range(50):  # Simulate 50 frames
        # Create dummy frame
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Simulate detection
        frame_detections = simulate_detection(dummy_frame, vehicle_classes, confidence)
        
        # Draw detections
        processed_frame = draw_detections(dummy_frame, frame_detections)
        
        # Update display
        stream_placeholder.image(processed_frame, channels="BGR", use_column_width=True)
        
        # Update metrics
        with metrics_placeholder.container():
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Frame", i+1)
            with col2:
                st.metric("Detections", len(frame_detections))
            with col3:
                st.metric("FPS", f"{20 + np.random.randint(-5, 5)}")
        
        time.sleep(0.1)  # Simulate processing time

def process_webcam(model_name, confidence, vehicle_classes):
    """Process webcam input"""
    st.info(f"📷 Processing webcam with {model_name}...")
    st.warning("Webcam processing would be implemented here with actual camera access.")

def simulate_detection(frame, vehicle_classes, confidence):
    """Simulate vehicle detection on a frame"""
    height, width = frame.shape[:2]
    detections = []
    
    # Simulate random detections
    num_detections = np.random.randint(0, 6)
    
    for _ in range(num_detections):
        if np.random.random() > confidence:
            continue
            
        # Random bounding box
        x1 = np.random.randint(0, width - 100)
        y1 = np.random.randint(0, height - 100)
        x2 = x1 + np.random.randint(50, 150)
        y2 = y1 + np.random.randint(50, 100)
        
        # Random vehicle class
        vehicle_class = np.random.choice(vehicle_classes)
        conf = np.random.uniform(confidence, 1.0)
        track_id = np.random.randint(1, 100)
        
        detections.append({
            'bbox': [x1, y1, x2, y2],
            'class': vehicle_class,
            'confidence': conf,
            'track_id': track_id,
            'timestamp': datetime.now()
        })
    
    return detections

def draw_detections(frame, detections):
    """Draw detection bounding boxes on frame"""
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        label = f"{det['class']} #{det['track_id']} ({det['confidence']:.2f})"
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    return frame

def display_performance_metrics():
    """Display model performance metrics"""
    st.markdown("### 🎯 Performance Metrics")
    
    # Simulated metrics
    metrics = {
        'mAP@50': 0.847,
        'mAP@95': 0.623,
        'IDF1': 0.751,
        'MOTA': 0.689,
        'FPS': 24.3
    }
    
    for metric, value in metrics.items():
        if metric == 'FPS':
            st.metric(metric, f"{value:.1f}")
        else:
            st.metric(metric, f"{value:.3f}")

def display_detection_stats():
    """Display current detection statistics"""
    st.markdown("### 🚗 Detection Stats")
    
    if st.session_state.detection_history:
        total_detections = len(st.session_state.detection_history)
        vehicle_counts = {}
        
        for det in st.session_state.detection_history:
            vehicle_class = det['class']
            vehicle_counts[vehicle_class] = vehicle_counts.get(vehicle_class, 0) + 1
        
        st.metric("Total Detections", total_detections)
        
        for vehicle_type, count in vehicle_counts.items():
            st.metric(f"{vehicle_type}s", count)
    else:
        st.info("No detections yet. Start processing video to see statistics.")

def display_detection_history():
    """Display detection history table"""
    if st.session_state.detection_history:
        # Convert to DataFrame
        df_data = []
        for det in st.session_state.detection_history[-100:]:  # Show last 100
            df_data.append({
                'Timestamp': det['timestamp'].strftime('%H:%M:%S'),
                'Vehicle Type': det['class'],
                'Track ID': det['track_id'],
                'Confidence': f"{det['confidence']:.3f}",
                'Bbox': f"({det['bbox'][0]}, {det['bbox'][1]}, {det['bbox'][2]}, {det['bbox'][3]})"
            })
        
        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No detection history available.")

def display_tracking_performance():
    """Display tracking performance charts"""
    if st.session_state.detection_history:
        # Create sample tracking data
        track_ids = [det['track_id'] for det in st.session_state.detection_history]
        unique_tracks = len(set(track_ids))
        
        # Track duration chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(st.session_state.detection_history))),
            y=[det['confidence'] for det in st.session_state.detection_history],
            mode='lines+markers',
            name='Confidence Over Time'
        ))
        fig.update_layout(
            title='Detection Confidence Over Time',
            xaxis_title='Detection Index',
            yaxis_title='Confidence Score'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.metric("Unique Tracks", unique_tracks)
    else:
        st.info("No tracking data available.")

def display_vehicle_counts():
    """Display vehicle count analytics"""
    if st.session_state.detection_history:
        # Vehicle type distribution
        vehicle_counts = {}
        for det in st.session_state.detection_history:
            vehicle_class = det['class']
            vehicle_counts[vehicle_class] = vehicle_counts.get(vehicle_class, 0) + 1
        
        # Pie chart
        fig = px.pie(
            values=list(vehicle_counts.values()),
            names=list(vehicle_counts.keys()),
            title="Vehicle Type Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Bar chart
        fig_bar = px.bar(
            x=list(vehicle_counts.keys()),
            y=list(vehicle_counts.values()),
            title="Vehicle Counts by Type"
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No vehicle count data available.")

if __name__ == "__main__":
    main()