from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

VEHICLE_CLASSES = {"car", "bus", "truck", "motorcycle"}

def detect_vehicles(frame):
    results = model(frame, conf=0.20, verbose=False)

    counts = {
        "car": 0,
        "bus": 0,
        "truck": 0,
        "motorcycle": 0
    }

    annotated_frame = frame.copy()

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])

            if cls_name in VEHICLE_CLASSES:
                counts[cls_name] += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated_frame,
                    f"{cls_name} {conf:.2f}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

    density = sum(counts.values())
    return annotated_frame, counts, density