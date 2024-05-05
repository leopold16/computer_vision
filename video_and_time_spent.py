#pip install inference
#pip install pafy
#pip install youtube-dl
#pip install supervision as sv

import cv2
import numpy as np
from inference import get_model
import supervision as sv
from google.colab.patches import cv2_imshow


# Load a pre-trained YOLOv8n model
model = get_model(model_id="yolov8n-640")

# Open a video file or capture device
video = cv2.VideoCapture('/content/Man Walking Through the City - 4k Ultra HD Stock Footage.mp4')
fps = video.get(cv2.CAP_PROP_FPS)  # Frame rate of the video

# Check if video opened successfully
if not video.isOpened():
    print("Error: Could not open video.")
    exit()

# Initialize tracking and timekeeping
trackers = {}
visibility_times = {}
tracker_dormancy = {}  # Tracker for dormancy period

frame_count = 0
max_dormant_frames = 10  # Allow tracker to be dormant for up to 10 frames

# Annotators for bounding boxes and labels
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

while True:
    ret, frame = video.read()
    if not ret:
        break
    frame_count += 1

    results = model.infer(frame)
    if results and hasattr(results[0], 'predictions'):
        detections = results[0].predictions
    else:
        print("No detections found or incorrect result structure.")
        break

    current_trackers = {}
    used_trackers = set()

    # Update trackers with detections
    for detection in detections:
        if detection.class_name == 'person':
            x = detection.x
            y = detection.y
            w = detection.width
            h = detection.height
            center = (int(x + w / 2), int(y + h / 2))
            found = False
            for tid, (last_pos, _) in trackers.items():
                if np.linalg.norm(np.array(center) - np.array(last_pos)) < 50:
                    found = True
                    current_trackers[tid] = (center, frame_count)
                    visibility_times[tid].append(frame_count)
                    used_trackers.add(tid)
                    break
            if not found:
                new_id = len(trackers) + 1
                current_trackers[new_id] = (center, frame_count)
                visibility_times[new_id] = [frame_count]

    # Check for dormant trackers and reactivate if within threshold
    for tid, (last_pos, last_frame) in trackers.items():
        if tid not in used_trackers and (frame_count - last_frame <= max_dormant_frames):
            current_trackers[tid] = (last_pos, last_frame)  # Carry forward dormant tracker

    trackers = current_trackers


    # Convert results for visualization
    detections = sv.Detections.from_inference(results[0].dict(by_alias=True, exclude_none=True))

    # Annotate the frame with inference results
    annotated_frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

    # Convert the annotated frame back to BGR for OpenCV display
    annotated_frame = cv2.cvtColor(np.array(annotated_frame), cv2.COLOR_RGB2BGR)

    # Display the frame
    cv2_imshow(annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  

video.release()
cv2.destroyAllWindows()

# Calculate the average visibility time
total_time = 0
count = 0
for times in visibility_times.values():
    if times:
        duration = (max(times) - min(times) + 1) / fps
        total_time += duration
        count += 1

average_time = total_time / count if count else 0
print(f"Average visibility time for humans: {average_time:.2f} seconds")
