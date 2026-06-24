import cv2
import numpy as np
import glob
import os

def fit_lane(contour, img_shape):
    points = contour.reshape(-1, 2)
    x = points[:, 0]
    y = points[:, 1]

    try:
        # Fit a 3rd degree polynomial for S-curves and better accuracy
        poly = np.polyfit(y, x, 3)
    except Exception as e:
        poly = None

    return poly, points

def process_image(image_path, output_dir):
    img = cv2.imread(image_path)
    if img is None:
        return

    img_shape = img.shape
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Relaxed Blue mask bounds
    lower_blue = np.array([90, 50, 20])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Use vertical rectangular kernel to avoid bridging unrelated horizontal lanes
    kernel = np.ones((9, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    left_lanes = []
    right_lanes = []

    center_x = img_shape[1] / 2

    for c in contours:
        area = cv2.contourArea(c)
        if area < 100:  # Relaxed area threshold to catch smaller lanes like 000008 corner
            continue

        points = c.reshape(-1, 2)
        max_y = np.max(points[:, 1])
        min_y = np.min(points[:, 1])

        # Discard horizontal noisy lines
        if max_y - min_y < 15:
            continue

        # Determine left or right based on the x-coordinate at the bottom-most part of the contour
        bottom_pts = points[points[:, 1] >= max_y - 10]
        bottom_x = np.mean(bottom_pts[:, 0])

        poly, points = fit_lane(c, img_shape)
        if poly is None:
            continue

        if bottom_x < center_x:
            left_lanes.append((area, poly, points, bottom_x))
        else:
            right_lanes.append((area, poly, points, bottom_x))

    # Draw lanes
    output_img = img.copy()

    def draw_lane(lane_info, color):
        poly = lane_info[1]
        points = lane_info[2]

        min_y = np.min(points[:, 1])
        max_y = np.max(points[:, 1])

        y_vals = np.linspace(min_y, max_y, int(max_y - min_y + 1))
        x_vals = np.polyval(poly, y_vals)

        # Clip x values so OpenCV doesn't overflow
        x_vals = np.clip(x_vals, -10000, 10000)

        pts = np.vstack((x_vals, y_vals)).astype(np.int32).T
        cv2.polylines(output_img, [pts], isClosed=False, color=color, thickness=3)

    # Right lane: largest area in the right half
    if right_lanes:
        right_lanes.sort(key=lambda x: x[0], reverse=True)
        draw_lane(right_lanes[0], (0, 255, 0)) # Green

    # Left lane: largest area in the left half
    if left_lanes:
        left_lanes.sort(key=lambda x: x[0], reverse=True)
        draw_lane(left_lanes[0], (0, 0, 255)) # Red

    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(image_path)
    cv2.imwrite(os.path.join(output_dir, filename), output_img)
    print(f"Processed {filename}")

if __name__ == '__main__':
    input_dir = '/tmp/mushr_frames'
    output_dir = 'output_lanes'

    images = glob.glob(os.path.join(input_dir, '*.png'))
    for img_path in sorted(images):
        process_image(img_path, output_dir)
