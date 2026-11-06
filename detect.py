import cv2
from ultralytics import YOLO

VIDEO_PATH = "temple.mp4"
MODEL_PATH = "yolov8n.pt"

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Could not open temple.mp4")
    exit()

print("Video opened successfully!")
print("Starting person detection...")

while True:

    ret, frame = cap.read()

    if not ret:
        print("Video processing completed.")
        break

    results = model(
        frame,
        classes=[0],       # Person only
        conf=0.15,         # Lower confidence for small/partially visible people
        imgsz=1280,        # Higher resolution
        iou=0.55,
        verbose=False
    )

    result = results[0]

    annotated_frame = result.plot(
        labels=True,
        boxes=True
    )

    person_count = len(result.boxes)

    cv2.putText(
        annotated_frame,
        f"People Detected: {person_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    cv2.imshow(
        "YOLOv8 - Person Detection",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("Detection program closed.")