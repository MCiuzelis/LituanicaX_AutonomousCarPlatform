import json
import os

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# --- 1. STABILIZED POLYNOMIAL LOSS FUNCTION ---
class PolyLoss(nn.Module):
    def __init__(self, y_anchor_weight=0.15):
        super().__init__()
        self.y_anchor_weight = y_anchor_weight

    def forward(self, preds, targets):
        B = preds.size(0)
        preds = preds.view(B, 2, 6, 2)
        targets = targets.view(B, 2, 6, 2)

        total_loss = 0.0
        count = 0

        for b in range(B):
            for lane in range(2):
                pred_lane = preds[b, lane]
                target_lane = targets[b, lane]

                valid_mask = (target_lane[:, 0] != 0) | (target_lane[:, 1] != 0)
                gt_points = target_lane[valid_mask]
                num_points = gt_points.size(0)

                if num_points < 2:
                    total_loss += torch.mean((pred_lane - target_lane) ** 2)
                    count += 1
                    continue

                # Cap the polynomial degree to 3 (Cubic) to prevent wild oscillations
                deg = min(num_points - 1, 3)

                gt_x = gt_points[:, 0]
                gt_y = gt_points[:, 1]

                # Build Vandermonde Matrix [num_points, deg + 1]
                V = torch.stack([gt_y**i for i in range(deg + 1)], dim=1)

                try:
                    # Cleaned up: lstsq natively handles rectangular matrices without manual ridge adjustments
                    coefs = torch.linalg.lstsq(V, gt_x.unsqueeze(1)).solution.squeeze(1)
                except (
                    RuntimeError
                ):  # FIXED: Changed from RuntimeException to RuntimeError
                    total_loss += torch.mean((pred_lane - target_lane) ** 2)
                    count += 1
                    continue

                pred_x = pred_lane[valid_mask, 0]
                pred_y = pred_lane[valid_mask, 1]

                expected_x = torch.zeros_like(pred_x)
                for i in range(deg + 1):
                    expected_x += coefs[i] * (pred_y**i)

                dist_to_poly = torch.mean((pred_x - expected_x) ** 2)
                y_anchor = torch.mean((pred_y - gt_y) ** 2)

                total_loss += dist_to_poly + (self.y_anchor_weight * y_anchor)
                count += 1

        return total_loss / count


# --- 2. CUSTOM DATASET LOADER ---
class LaneDataset(Dataset):
    def __init__(self, json_path, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        with open(json_path, "r") as f:
            self.data = json.load(f)
        self.img_names = list(self.data.keys())

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)

        image = cv2.imread(img_path)
        h, w, _ = image.shape
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        left_target = self.data[img_name]["left_lane"]
        right_target = self.data[img_name]["right_lane"]

        norm_left = [
            val / w if i % 2 == 0 else val / h for i, val in enumerate(left_target)
        ]
        norm_right = [
            val / w if i % 2 == 0 else val / h for i, val in enumerate(right_target)
        ]

        target = torch.tensor(norm_left + norm_right, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, target


# --- 3. LIGHTWEIGHT PERCEPTION NETWORK ---
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
            nn.Linear(64 * 16 * 16, 128), nn.ReLU(), nn.Linear(128, 24), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        return self.regressor(x)


# --- 4. MAIN TRAINING LOOP ---
def train():
    data_dir = "/tmp/mushr_frames"
    json_path = os.path.join(data_dir, "labels.json")
    num_epochs = 1000

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = LaneDataset(json_path=json_path, img_dir=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyLaneNet().to(device)
    criterion = PolyLoss(y_anchor_weight=0.15)

    optimizer = optim.Adam(model.parameters(), lr=2e-4)

    print(f"Starting training on device: {device} with {len(dataset)} samples...")

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model.forward(images)

            loss = criterion(outputs, targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_loss += loss.item() * images.size(0)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] - Average Loss: {epoch_loss / len(dataset):.6f}"
        )

    torch.save(model.state_dict(), "tiny_lane_net.pth")
    print("Training finished! Model saved as tiny_lane_net.pth")


if __name__ == "__main__":
    train()
