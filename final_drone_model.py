from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion
import numpy as np
import cv2

model1 = YOLO("yolov8m.pt")
model2 = YOLO("yolov8s-visdrone.pt")
model3 = YOLO("yolov8x.pt")

def run(img_path):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    results = [m(img)[0] for m in [model1, model2, model3]]

    boxes, scores, labels = [], [], []
    for r in results:
        if r.boxes is None:
            continue
        b = r.boxes.xyxy.cpu().numpy()
        s = r.boxes.conf.cpu().numpy()
        l = r.boxes.cls.cpu().numpy()
        boxes.append([[x1 / w, y1 / h, x2 / w, y2 / h] for x1, y1, x2, y2 in b])
        scores.append(list(s))
        labels.append(list(l))

    b_fused, s_fused, l_fused = weighted_boxes_fusion(
        boxes, scores, labels, iou_thr=0.5, skip_box_thr=0.001
    )
    b_fused = np.array(b_fused)
    b_fused[:, [0, 2]] *= w
    b_fused[:, [1, 3]] *= h

    for i, bb in enumerate(b_fused):
        x1, y1, x2, y2 = map(int, bb)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            f"{model1.names[int(l_fused[i])]} {s_fused[i]:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    cv2.imwrite("final_output.jpg", img)

run("C:\\Users\\iampr\\Desktop\\Drone intern\\DroneData\\images\\train\\0006.jpg")
