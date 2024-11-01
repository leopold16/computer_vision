import cv2
import time
import numpy as np
import subprocess
from roboflow import Roboflow
import threading

# Initialize Roboflow
rf = Roboflow(api_key="b965xH5DvvHRHYj8a9xh")
project = rf.workspace().project("human-counter-btofl")
model = project.version("3").model

# Entry polygon points (this is for live stream, and may need to be adjusted for other formats)
entry_polygon_points = np.array([
    [1906, 1856], 
    [1905, 1477], 
    [1709, 1468], 
    [1546, 871], 
    [907, 903], 
    [947, 1992], 
    [1905, 1862]
])

store_polygon_points = np.array([
    [1553, 853], 
    [1725, 1632], 
    [2197, 1560], 
    [2211, 723], 
    [1862, 664], 
    [1726, 373], 
    [1512, 374], 
    [1615, 732], 
    [1552, 860]
])

# Initialize list to keep track of trackers and displayed customers
trackers = []
displayed_customers = {}
start_times = {}

# Set to keep track of unique customers who entered the store
unique_customer_ids = set()

# Counter to keep track of store entries
store_entry_count = 0

# Set to keep track of recent detections
recent_detections = []

# Time window in seconds to ignore new detections
time_window = 0.25  # Adjust as needed

# Flag to indicate if fetching is done
fetch_done = False

def is_overlapping(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    if x1 < x2 + w2 and x1 + w1 > x2 and y1 < y2 + h2 and y1 + h1 > y2:
        return True
    return False

def fetch_latest_image():
    global fetch_done
    while not fetch_done:
        result = subprocess.run(
            ["curl", "-o", "image.jpg", "ftp://aibot:Food1234!@178.62.8.167/files/image.jpg"],
            capture_output=True
        )
        if result.returncode != 0:
            print("Error: Could not fetch the latest image.")
            fetch_done = True
            return None
        time.sleep(0.1)  # Sleep a bit to prevent overloading the server

fetch_thread = threading.Thread(target=fetch_latest_image)
fetch_thread.start()

while True:
    # Read the image into OpenCV
    frame = cv2.imread("image.jpg")
    if frame is None:
        print("Error: Could not read the image.")
        continue

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform object detection
    try:
        predictions = model.predict(rgb_frame, confidence=20, overlap=50).json()
    except Exception as e:
        print(f"Error during model prediction: {e}")
        continue

    # Draw the polygons on the frame
    cv2.polylines(frame, [entry_polygon_points], isClosed=True, color=(0, 255, 255), thickness=2)
    cv2.polylines(frame, [store_polygon_points], isClosed=True, color=(255, 0, 0), thickness=2)

    # Create masks for the polygons
    entry_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    store_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(entry_mask, [entry_polygon_points], 255)
    cv2.fillPoly(store_mask, [store_polygon_points], 255)

    # Initialize or update trackers
    new_trackers = []
    for tracker, bbox, start_time in trackers:
        success, newbox = tracker.update(frame)
        if success:
            new_trackers.append((tracker, newbox, start_time))
            p1 = (int(newbox[0]), int(newbox[1]))
            p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))

            elapsed_time = time.time() - start_time
            displayed_customers[tracker] = elapsed_time
            label = f'{elapsed_time:.1f}s'

            # Check if the customer enters the store polygon
            center_x = int(newbox[0] + newbox[2] / 2)
            center_y = int(newbox[1] + newbox[3] / 2)
            if cv2.pointPolygonTest(store_polygon_points, (center_x, center_y), False) >= 0 and tracker not in unique_customer_ids:
                unique_customer_ids.add(tracker)
                store_entry_count += 1

            # Display the bounding box and time
            cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
            cv2.putText(frame, label, (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    current_time = time.time()

    for prediction in predictions['predictions']:
        x, y, w, h = prediction['x'], prediction['y'], prediction['width'], prediction['height']
        class_name = prediction['class']

        # Draw bounding box for all detected objects
        bbox = (int(x - w / 2), int(y - h / 2), int(w), int(h))
        p1 = (int(x - w / 2), int(y - h / 2))
        p2 = (int(x + w / 2), int(y + h / 2))
        if class_name == 'customer':
            color = (0, 255, 0)  # Green for customers
            label = 'Customer'
        elif class_name == 'worker':
            color = (0, 0, 255)  # Red for workers
            label = 'Staff'
        else:
            color = (255, 0, 0)  # Blue for other classes
            label = class_name

        # Always display the bounding box for customers
        cv2.rectangle(frame, p1, p2, color, 2)
        cv2.putText(frame, label, (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        if class_name == 'customer':
            # Check if the center of the bounding box is within the entry polygon
            center_x = x
            center_y = y
            if cv2.pointPolygonTest(entry_polygon_points, (center_x, center_y), False) >= 0:
                # Check if this detection is within the time window of recent detections
                if not any(
                    abs(center_x - rx) < w and abs(center_y - ry) < h and (current_time - rt) < time_window
                    for rx, ry, rt in recent_detections
                ):
                    # Check for overlapping boxes
                    if not any(is_overlapping(bbox, t[1]) for t in trackers):
                        # Add a new tracker
                        tracker = cv2.TrackerCSRT_create()
                        tracker.init(frame, bbox)
                        new_trackers.append((tracker, bbox, time.time()))
                        start_times[tracker] = time.time()

                        # Add this detection to recent detections
                        recent_detections.append((center_x, center_y, current_time))

                        # Remove old detections outside the time window
                        recent_detections = [(rx, ry, rt) for rx, ry, rt in recent_detections if (current_time - rt) < time_window]

    trackers = new_trackers

    # Display customer count, store entry count, and entry percentage
    customer_count = len(trackers)
    if customer_count > 0:
        store_entry_percentage = (store_entry_count / customer_count) * 100
    else:
        store_entry_percentage = 0

    cv2.putText(frame, f'Total customers: {customer_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f'Conversions: {store_entry_count}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f'Conversion %: {store_entry_percentage:.2f}%', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the frame with detected objects
    cv2.imshow('Detected Frame with Polygons', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        fetch_done = True
        break

# Print the total number of customers detected and the percentage that entered the store
print(f'Total number of customers detected: {customer_count}')
print(f'Total number of customers entered the store: {store_entry_count}')
print(f'Percentage of customers who entered the store: {store_entry_percentage:.2f}%')

# Release the video capture object
cv2.destroyAllWindows()
