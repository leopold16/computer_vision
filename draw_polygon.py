import cv2
import numpy as np

# List to store points
polygon_points = []

# Mouse callback function
def draw_polygon(event, x, y, flags, param):
    global polygon_points, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        polygon_points.append((x, y))
        drawing = True

    if event == cv2.EVENT_MOUSEMOVE and drawing:
        temp_image = img.copy()
        if len(polygon_points) > 1:
            cv2.polylines(temp_image, [np.array(polygon_points)], isClosed=False, color=(0, 255, 255), thickness=2)
        cv2.imshow("Image", temp_image)

    if event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if len(polygon_points) > 1:
            cv2.polylines(img, [np.array(polygon_points)], isClosed=False, color=(0, 255, 255), thickness=2)
        cv2.imshow("Image", img)

def save_polygon_points(polygon_points, filename):
    with open(filename, 'w') as f:
        for point in polygon_points:
            f.write(f'{point[0]},{point[1]}\n')

# Load an image
img = cv2.imread('latest_image.jpg')
if img is None:
    print("Error: Could not load image.")
    exit()

drawing = False

# Create a window and bind the function to window
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", draw_polygon)

while True:
    cv2.imshow("Image", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Press 'q' to quit
        break
    elif key == ord('s'):  # Press 's' to save the polygon points
        save_polygon_points(polygon_points, 'polygon_points.txt')
        print("Polygon points saved to polygon_points.txt")
        break

cv2.destroyAllWindows()
