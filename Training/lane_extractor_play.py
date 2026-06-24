import glob
import os

import cv2
import torch
import torch.nn as nn
from torchvision import transforms


# --- 1. RE-DECLARE THE TINY NETWORK (WITH SIGMOID FIX) ---
class TinyLaneNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.regressor = nn.Sequential(
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 24),
            nn.Sigmoid(),  # Crucial constraint matches fixed training
        )

    def forward(self, x):
        x = self.features(x)
        return self.regressor(x)


def main():
    IMAGE_DIR = "/tmp/mushr_frames"
    MODEL_PATH = "tiny_lane_net.pth"

    # Check for model weights
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Weights file '{MODEL_PATH}' not found. Please re-train first.")
        return

    # Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyLaneNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"Loaded model onto {device}. Processing images...")

    # Image preprocessing pipeline (must match training)
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Grab all raw images
    image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "raw_*.png")))
    if not image_paths:
        print(f"No raw images found in {IMAGE_DIR}")
        return

    print("\n--- CONTROLS ---")
    print("• Any Key : Move to next frame")
    print("• ESC Key : Close Visualizer")
    print("----------------\n")

    for img_path in image_paths:
        # 1. Load original image and save original dimensions
        orig_img = cv2.imread(img_path)
        if orig_img is None:
            continue
        orig_h, orig_w, _ = orig_img.shape

        # 2. Preprocess for the neural network
        rgb_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        tensor_img = (
            transform(rgb_img).unsqueeze(0).to(device)
        )  # Add batch dimension -> [1, 3, 128, 128]

        # 3. Model Inference
        with torch.no_grad():
            predictions = model(tensor_img).squeeze(0).cpu().numpy()  # Shape: [24]

        # Split predictions back into left and right lanes
        left_lane = predictions[0:12]
        right_lane = predictions[12:24]

        # 4. Rescale and Draw Left Lane (Red Circles)
        for i in range(0, 12, 2):
            norm_x, norm_y = left_lane[i], left_lane[i + 1]
            # Ignore placeholder points close to zero
            if norm_x > 0.01 or norm_y > 0.01:
                real_x = int(norm_x * orig_w)
                real_y = int(norm_y * orig_h)
                cv2.circle(orig_img, (real_x, real_y), 5, (0, 0, 255), -1)

        # 5. Rescale and Draw Right Lane (Blue Circles)
        for i in range(0, 12, 2):
            norm_x, norm_y = right_lane[i], right_lane[i + 1]
            if norm_x > 0.01 or norm_y > 0.01:
                real_x = int(norm_x * orig_w)
                real_y = int(norm_y * orig_h)
                cv2.circle(orig_img, (real_x, real_y), 5, (255, 0, 0), -1)

        # Display output overlay
        cv2.imshow("Model Output Validation", orig_img)

        # Wait for keypress; ESC exits loop
        key = cv2.waitKey(0) & 0xFF
        if key == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
