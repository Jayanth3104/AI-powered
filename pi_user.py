# test_yolo_picam.py
import cv2
import torch

# ---------------------------
# 1️⃣ Load your YOLOv5 model
# ---------------------------
print("🔄 Loading YOLOv5 model...")
model = torch.hub.load(
    'yolov5',        # YOLOv5 repo
    'custom',                     # custom model
    path='yolov5/runs/train/ambulance-detection3/weights/best.pt',  # your trained weights
    source='local'                # load from local repo
)
model.eval()
print("✅ Model loaded!")

# ---------------------------
# 2️⃣ Connect to Pi camera stream
# ---------------------------
pi_stream_url = "http://192.168.1.6:5001/video_feed"  # replace with your Pi IP
cap = cv2.VideoCapture(pi_stream_url)

if not cap.isOpened():
    print("❌ Cannot open Pi camera stream")
    exit()

print("🔄 Connected to Pi camera stream...")

# ---------------------------
# 3️⃣ Run YOLOv5 detection
# ---------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # Convert BGR to RGB for YOLOv5
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform inference
    results = model(img_rgb)

    # Render results on frame
    results.render()  # updates results.imgs with boxes and labels
    annotated_frame = results.ims[0]  # get the first (and only) frame

    # Convert RGB back to BGR for OpenCV display
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

    # Display
    cv2.imshow("YOLOv5 PiCam Detection", annotated_frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
