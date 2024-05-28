## Vehicle Detection and Counting using Image Processing

This project provides a simple algorithm to detect and count vehicles in an image using image processing techniques. The implementation uses OpenCV, a popular computer vision library in Python, to perform various image processing tasks.

## Output Image
![output_image](https://github.com/Naieem-55/vehicle_count_in_image/assets/60366614/faa4ab3b-8133-45af-b44b-2ed28cca30a1)


## Features
- Convert the input image to grayscale.
- Apply Gaussian blur to reduce noise.
- Perform edge detection using the Canny edge detector.
- Use morphological operations to enhance the detected edges.
- Find and filter contours to detect vehicles.
- Draw bounding boxes around detected vehicles.
- Count the number of vehicles and overlay the count on the image.
- Save the processed image with the vehicle count to a specified directory.

## Prerequisites
- Python 3.x
- OpenCV

## Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/Naieem-55/vehicle_count_in_image
    cd Vehicle_Count
    ```

2. Install the required Python packages:
    ```sh
    pip install opencv-python
    ```




