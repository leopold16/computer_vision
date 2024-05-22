import cv2
import time
import numpy as np
from roboflow import Roboflow

# Initialize Roboflow
rf = Roboflow(api_key="b965xH5DvvHRHYj8a9xh")
project = rf.workspace().project("human-counter-btofl")
model = project.version("1").model

# Path to the pre-recorded video file
video_path = '/Users/leopold/venv/sample2.mp4'

# Initialize the video capture object
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Set the desired frame rate for processing
frame_rate = 1  # 1 frame per second
prev = 0  # Previous time frame was captured

# Get the frames per second (fps) of the video
fps = cap.get(cv2.CAP_PROP_FPS)

# Calculate the number of frames to skip to achieve the desired frame rate
frames_to_skip = int(fps / frame_rate)

# Define the points of the polygon
polygon_points = np.array([[454, 361],[447, 501],[630,507],[628, 342],[456,358]])

# Initialize list to keep track of trackers
trackers = []

# Set to keep track of unique customer IDs
customer_count = 0

while True:
    time_elapsed = time.time() - prev

    if time_elapsed > 1.0 / frame_rate:
        prev = time.time()

        # Read the next frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame or end of video.")
            break

        # Convert the frame to RGB (if necessary)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform object detection
        predictions = model.predict(rgb_frame, confidence=40, overlap=30).json()

        # Draw the polygon on the frame
        cv2.polylines(frame, [polygon_points], isClosed=True, color=(0, 255, 255), thickness=2)

        # Create a mask image that contains the polygon
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [polygon_points], 255)

        # Initialize or update trackers
        new_trackers = []
        for tracker, bbox in trackers:
            success, newbox = tracker.update(frame)
            if success:
                new_trackers.append((tracker, newbox))
                p1 = (int(newbox[0]), int(newbox[1]))
                p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
                cv2.putText(frame, f'Customer', (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        for prediction in predictions['predictions']:
            x, y, w, h = prediction['x'], prediction['y'], prediction['width'], prediction['height']
            class_name = prediction['class']

            if class_name == 'customer':
                # Check if the center of the bounding box is within the polygon
                center_x = x
                center_y = y
                if cv2.pointPolygonTest(polygon_points, (center_x, center_y), False) >= 0:
                    # Add a new tracker
                    bbox = (int(x - w / 2), int(y - h / 2), int(w), int(h))
                    tracker = cv2.TrackerKCF_create()
                    tracker.init(frame, bbox)
                    new_trackers.append((tracker, bbox))
                    customer_count += 1

        trackers = new_trackers

        # Display the frame with detected objects
        cv2.imshow('Detected Frame with Polygon', frame)

        # Skip the next frames_to_skip - 1 frames
        for _ in range(frames_to_skip - 1):
            cap.grab()

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Print the total number of customers detected
print(f'Total number of customers entered the polygon: {customer_count}')

# Release the video capture object
cap.release()
cv2.destroyAllWindows()