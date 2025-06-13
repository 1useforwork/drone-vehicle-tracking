import cv2

def draw_boxes(frame, detections, track_ids, class_names):
    for det, track_id in zip(detections, track_ids):
        x1, y1, x2, y2, _, cls = map(int, det)
        label = f"{class_names[cls]} ID:{track_id}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    return frame