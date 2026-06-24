import glob
import json
import os

import cv2

# Configuration
IMAGE_DIR = "/tmp/mushr_frames"
OUTPUT_JSON = os.path.join(IMAGE_DIR, "labels.json")
MAX_POINTS = 6

# Global lists to store points for the current frame
left_points = []
right_points = []
redraw_needed = True


def mouse_callback(event, x, y, flags, param):
    global left_points, right_points, redraw_needed

    # Left Click -> Left Lane
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(left_points) < MAX_POINTS:
            left_points.append([x, y])
            redraw_needed = True
            print(f"Added Left Point {len(left_points)}/{MAX_POINTS}: ({x}, {y})")

    # Right Click -> Right Lane
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(right_points) < MAX_POINTS:
            right_points.append([x, y])
            redraw_needed = True
            print(f"Added Right Point {len(right_points)}/{MAX_POINTS}: ({x}, {y})")


def main():
    global left_points, right_points, redraw_needed

    # 1. Load existing progress if available
    labels_data = {}
    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON, "r") as f:
            try:
                labels_data = json.load(f)
                print(
                    f"Loaded existing progress. {len(labels_data)} images already labeled."
                )
            except json.JSONDecodeError:
                print("Found labels.json but it was empty/corrupted. Starting fresh.")

    # 2. Find and sort all raw_*.png images
    search_path = os.path.join(IMAGE_DIR, "raw_*.png")
    image_paths = sorted(glob.glob(search_path))

    if not image_paths:
        print(
            f"No images found matching 'raw_*.png' in {IMAGE_DIR}. Please check the path."
        )
        return

    print("\n--- CONTROLS ---")
    print("• Left Click  : Mark Left Lane (Red)")
    print("• Right Click : Mark Right Lane (Blue)")
    print("• 'C' Key     : Clear current selections")
    print("• Spacebar    : Save and Move to Next Image")
    print("• ESC Key     : Save progress and Exit")
    print("----------------\n")

    # Replace the old cv2.namedWindow line with this:
    cv2.namedWindow(
        "Lane Labeler",
        flags=cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL,
    )
    cv2.setMouseCallback("Lane Labeler", mouse_callback)

    for idx, img_path in enumerate(image_paths):
        filename = os.path.basename(img_path)

        # Skip if already labeled
        if filename in labels_data:
            continue

        print(f"[{idx + 1}/{len(image_paths)}] Labeling: {filename}")

        # Read the raw image
        img_original = cv2.imread(img_path)
        if img_original is None:
            print(f"Failed to load image: {img_path}")
            continue

        # Reset points for the new image
        left_points = []
        right_points = []
        redraw_needed = True

        while True:
            if redraw_needed:
                # Always draw on a fresh copy of the original frame
                img_display = img_original.copy()

                # Draw Left points (Red) and connect them with lines
                for i, pt in enumerate(left_points):
                    cv2.circle(img_display, tuple(pt), 4, (0, 0, 255), -1)
                    cv2.putText(
                        img_display,
                        str(i + 1),
                        (pt[0] + 5, pt[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 0, 255),
                        1,
                    )
                if len(left_points) > 1:
                    for i in range(len(left_points) - 1):
                        cv2.line(
                            img_display,
                            tuple(left_points[i]),
                            tuple(left_points[i + 1]),
                            (0, 0, 180),
                            1,
                        )

                # Draw Right points (Blue) and connect them with lines
                for i, pt in enumerate(right_points):
                    cv2.circle(img_display, tuple(pt), 4, (255, 0, 0), -1)
                    cv2.putText(
                        img_display,
                        str(i + 1),
                        (pt[0] + 5, pt[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 0, 0),
                        1,
                    )
                if len(right_points) > 1:
                    for i in range(len(right_points) - 1):
                        cv2.line(
                            img_display,
                            tuple(right_points[i]),
                            tuple(right_points[i + 1]),
                            (180, 0, 0),
                            1,
                        )

                cv2.imshow("Lane Labeler", img_display)
                redraw_needed = False

            # Wait for keypress
            key = cv2.waitKey(20) & 0xFF

            # 'c' key to clear current points
            if key == ord("c") or key == ord("C"):
                left_points = []
                right_points = []
                redraw_needed = True
                print("Cleared points for this frame.")

            # Spacebar to save and advance
            elif key == ord(" "):
                # Padding logic: If less than 6 points, fill remaining with [0, 0]
                final_left = list(left_points)
                while len(final_left) < MAX_POINTS:
                    final_left.append([0, 0])

                final_right = list(right_points)
                while len(final_right) < MAX_POINTS:
                    final_right.append([0, 0])

                # Flatten arrays or keep as pairs? Storing as flattened list
                # (x1, y1, x2, y2...) makes it incredibly easy to pipe into a PyTorch tensor.
                flattened_left = [coord for pt in final_left for coord in pt]
                flattened_right = [coord for pt in final_right for coord in pt]

                # Store mapping
                labels_data[filename] = {
                    "left_lane": flattened_left,
                    "right_lane": flattened_right,
                }

                # Incremental save so you never lose progress
                with open(OUTPUT_JSON, "w") as f:
                    json.dump(labels_data, f, indent=4)

                break  # Break out of inner loop to load next image

            # ESC key to exit program completely
            elif key == 27:
                print("Exiting and saving current progress...")
                cv2.destroyAllWindows()
                return

    print("\nAll images labeled successfully!")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
