import cv2
from ultralytics import YOLO

# Load your trained model
model = YOLO("runs/detect/train9/weights/best.pt")  # Use your best.pt path

# Use the full absolute path
video_path = r"C:\Users\iampr\Desktop\Drone intern\datasets\traffic drone shot of a city.mp4"
cap = cv2.VideoCapture(video_path)

# Output writer (same resolution as input)
out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30,
                      (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame)

    # Draw bounding boxes on the frame
    annotated_frame = results[0].plot()

    # Save to output video
    out.write(annotated_frame)

    # Optional: Show window
    cv2.imshow("YOLOv8 Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
