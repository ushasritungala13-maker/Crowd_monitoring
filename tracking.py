import cv2
from ultralytics import YOLO

from people_count import get_current_people_count
from people_count import get_tracking_ids

from density_estimation import calculate_density

from zone_classification import classify_zone
from zone_classification import get_alert


# ==========================================
# SETTINGS
# ==========================================

VIDEO_PATH = "temple.mp4"
MODEL_PATH = "yolov8n.pt"
TRACKER_PATH = "my_botsort.yaml"


# ==========================================
# LOAD YOLO MODEL
# ==========================================

model = YOLO(MODEL_PATH)


# ==========================================
# OPEN VIDEO
# ==========================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():

    print("ERROR: Could not open temple.mp4")

    exit()


print("Video opened successfully!")
print("Starting complete crowd monitoring...")


# ==========================================
# UNIQUE IDs SEEN THROUGHOUT VIDEO
# ==========================================

all_seen_ids = set()


# ==========================================
# PROCESS VIDEO
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:

        print("Video processing completed.")

        break


    # ======================================
    # YOLO + BOT-SORT
    # ======================================

    results = model.track(

        frame,

        persist=True,

        tracker=TRACKER_PATH,

        classes=[0],

        conf=0.20,

        imgsz=1280,

        iou=0.60,

        verbose=False
    )


    result = results[0]


    # ======================================
    # CURRENT PEOPLE
    # ======================================

    current_people = get_current_people_count(
        result
    )


    # ======================================
    # CURRENT TRACKING IDs
    # ======================================

    current_ids = get_tracking_ids(
        result
    )


    # ======================================
    # ADD IDs TO TOTAL UNIQUE IDs
    # ======================================

    for track_id in current_ids:

        all_seen_ids.add(track_id)


    unique_ids_seen = len(
        all_seen_ids
    )


    # ======================================
    # DENSITY
    # ======================================

    density = calculate_density(
        current_people
    )


    # ======================================
    # ZONE
    # ======================================

    zone = classify_zone(
        density
    )


    # ======================================
    # ALERT
    # ======================================

    alert = get_alert(
        zone
    )


    # ======================================
    # DRAW PERSON BOXES + IDs
    # ======================================

    output = result.plot(
        labels=True,
        boxes=True
    )


    # ======================================
    # INFORMATION PANEL
    # ======================================

    # Current People

    cv2.putText(

        output,

        f"Current People: {current_people}",

        (20, 40),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.85,

        (0, 255, 255),

        2
    )


    # Unique IDs Seen

    cv2.putText(

        output,

        f"Unique IDs Seen: {unique_ids_seen}",

        (20, 75),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.75,

        (255, 255, 255),

        2
    )


    # Density

    cv2.putText(

        output,

        f"Density: {density}",

        (20, 110),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.75,

        (255, 255, 255),

        2
    )


    # Zone

    cv2.putText(

        output,

        f"Zone: {zone}",

        (20, 145),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.75,

        (255, 255, 255),

        2
    )


    # Alert

    cv2.putText(

        output,

        f"Alert: {alert}",

        (20, 180),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.75,

        (0, 255, 255),

        2
    )


    # ======================================
    # SHOW VIDEO
    # ======================================

    cv2.imshow(

        "AI Crowd Monitoring System",

        output
    )


    # ======================================
    # TERMINAL OUTPUT
    # ======================================

    print(

        f"Current People: {current_people} | "

        f"Unique IDs Seen: {unique_ids_seen} | "

        f"Density: {density} | "

        f"Zone: {zone} | "

        f"Alert: {alert}"
    )


    # ======================================
    # QUIT
    # ======================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ==========================================
# CLEANUP
# ==========================================

cap.release()

cv2.destroyAllWindows()


print()
print("==========================================")
print("CROWD MONITORING COMPLETED")
print("==========================================")

print(
    "Total Unique IDs Seen:",
    len(all_seen_ids)
)