import numpy as np
import cv2
from canny_edge import canny_edge_detector

# Structuring Element
def get_structuring_element(shape, ksize):
    return np.ones(ksize, dtype=np.uint8)

# Morphological Close Operation
def morphology_ex(image, operation, kernel):

    if operation == 'close':
        dilated = cv2.dilate(image, kernel)
        closed = cv2.erode(dilated, kernel)
        return closed
    
    return image

# Find Contours
def find_contours(image):
    contours = []
    visited = np.zeros_like(image, dtype=bool)

    def dfs(x, y):

        stack = [(x, y)]
        contour = []

        while stack:
            cx, cy = stack.pop()
            if visited[cy, cx] or image[cy, cx] == 0:
                continue

            visited[cy, cx] = True
            contour.append((cx, cy))

            for nx, ny in [(cx-1, cy), (cx+1, cy), (cx, cy-1), (cx, cy+1)]:
                if 0 <= nx < image.shape[1] and 0 <= ny < image.shape[0] and not visited[ny, nx]:
                    stack.append((nx, ny))

        return contour

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):

            if image[y, x] and not visited[y, x]:
                contour = dfs(x, y)
                if contour:
                    contours.append(np.array(contour))

    return contours

# Bounding Rect
def bounding_rect(contour):

    x_coords = contour[:, 0]
    y_coords = contour[:, 1]

    x = np.min(x_coords)
    y = np.min(y_coords)
    w = np.max(x_coords) - x
    h = np.max(y_coords) - y

    return x, y, w, h

# Custom Rectangle
def custom_rectangle(image, pt1, pt2, color, thickness):

    x1, y1 = pt1
    x2, y2 = pt2

    image[y1:y1+thickness, x1:x2] = color
    image[y2:y2+thickness, x1:x2] = color
    image[y1:y2, x1:x1+thickness] = color
    image[y1:y2, x2:x2+thickness] = color

    return image

def detect_vehicles(image_path, output_path):
    image = cv2.imread(image_path)
    edges = canny_edge_detector(image, 50, 150)

    if edges.dtype != np.uint8:
        edges = (edges * 255).astype(np.uint8)

    kernel = get_structuring_element(cv2.MORPH_RECT, (5, 5))
    closed = morphology_ex(edges, 'close', kernel)

    contours = find_contours(closed)

    vehicle_count = 0
    for contour in contours:
        if len(contour) < 6000:  # Filter out small contours
            continue

        x, y, w, h = bounding_rect(contour)
        if w * h < 20000:  # Filter based on area to reduce false positives
            continue

        image = custom_rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        vehicle_count += 1

    cv2.putText(image, f'Vehicle Count: {vehicle_count}', (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 6, cv2.LINE_AA)
    cv2.imwrite(output_path, image)

    print(f'Number of vehicles detected: {vehicle_count}')

input_path = r'C:\Users\Hp\OneDrive\Desktop\Vehicle\Input Image\car3.jpg'
output_path = r'C:\Users\Hp\OneDrive\Desktop\Vehicle\Output Image\output_car3.jpg'

detect_vehicles(input_path, output_path)
