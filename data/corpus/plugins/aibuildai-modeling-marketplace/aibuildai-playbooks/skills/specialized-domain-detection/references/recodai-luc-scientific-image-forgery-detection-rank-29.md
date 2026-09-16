# Recod.ai/LUC - Scientific Image Forgery Detection

Competition: recodai-luc-scientific-image-forgery-detection
Rank: #29
Source: https://www.kaggle.com/c/recodai-luc-scientific-image-forgery-detection/writeups/recod-ai-luc-scientific-image-forgery-detection

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import pandas as pd
import os
from torchvision import transforms
from typing import Tuple, List, Dict

# --- Configuration and Constants ---
DATA_ROOT = './'
VALIDATION_IMAGES_DIR = os.path.join(DATA_ROOT, 'train_images') # Uses the same image dir as validation images are taken from training data
VALIDATION_MASKS_CSV = os.path.join(DATA_ROOT, 'val_masks.csv') # File created by the training script
MODEL_WEIGHTS_PATH = 'forgery_detection_model.pth' # Path to your best trained model

IMG_HEIGHT = 256 
IMG_WIDTH = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16

# --- Utility Functions (RLE and IoU Metric) ---

def rle_decode(mask_rle: str, shape: Tuple[int, int]) -> np.ndarray:
    """Decodes a run-length encoded mask to a binary array."""
    if mask_rle == 'authentic' or not mask_rle:
        return np.zeros(shape, dtype=np.uint8)
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

def calculate_iou(preds: np.ndarray, targets: np.ndarray) -> float:
    """Calculates Mean Intersection over Union (IoU) across flattened arrays."""
    intersection = np.sum(preds * targets)
    union = np.sum(preds) + np.sum(targets) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou

# --- Model Architecture (MUST match training script) ---

class NoiseResidual(nn.Module):
    def __init__(self, kernel_size=5):
        super(NoiseResidual, self).__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2

    def forward(self, x):
        channels = x.shape[1]
        pool = nn.AvgPool2d(kernel_size=self.kernel_size, stride=1, padding=self.pad).to(x.device)
        smooth_x = x.clone()
        for i in range(channels):
            smooth_x[:, i:i+1, :, :] = pool(x[:, i:i+1, :, :])
        residual = x - smooth_x
        return torch.cat([x, residual], dim=1)

def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )

class ForgeryDetectionUNet(nn.Module):
    def __init__(self, in_channels=6, out_channels=1):
        super(ForgeryDetectionUNet, self).__init__()
        self.noise_preprocessor = NoiseResidual(kernel_size=5)
        self.enc1 = conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = conv_block(256, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = conv_block(128, 64)
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x):
        x = self.noise_preprocessor(x)
        e1 = self.enc1(x); p1 = self.pool1(e1)
        e2 = self.enc2(p1); p2 = self.pool2(e2)
        e3 = self.enc3(p2); p3 = self.pool3(e3)
        b = self.bottleneck(p3)
        u3 = self.upconv3(b); u3 = torch.cat((u3, e3), dim=1); d3 = self.dec3(u3)
        u2 = self.upconv2(d3); u2 = torch.cat((u2, e2), dim=1); d2 = self.dec2(u2)
        u1 = self.upconv1(d2); u1 = torch.cat((u1, e1), dim=1); d1 = self.dec1(u1)
        out = self.out_conv(d1)
        return torch.sigmoid(out)

# --- Validation Data Loader (Must match ForgeryDataset structure) ---

class ValidationForgeryDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: str):
        self.df = df
        self.img_dir = img_dir
        self.case_ids = df['case_id'].unique().tolist()
        self.mask_map: Dict[str, List[str]] = self._create_mask_map(df)
        
    def _create_mask_map(self, df: pd.DataFrame) -> dict:
        mask_map = {}
        for case_id, group in df.groupby('case_id'):
            mask_map[case_id] = group['annotation'].tolist()
        return mask_map

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        case_id = self.case_ids[idx]
        img_path = os.path.join(self.img_dir, f"{case_id}.png")
        
        image = cv2.imread(img_path)
        if image is None:
             raise FileNotFoundError(f"Image not found: {img_path}")
             
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        masks_rle = self.mask_map.get(case_id, [])
        combined_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        for rle in masks_rle:
            if rle != 'authentic':
                mask = rle_decode(rle, (image.shape[0], image.shape[1]))
                combined_mask = np.maximum(combined_mask, mask) 

        image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
        mask = cv2.resize(combined_mask, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
        
        image_tensor = transforms.ToTensor()(image.astype(np.float32) / 255.0)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
        
        return image_tensor, mask_tensor

# --- Main Optimization Function ---

def optimize_threshold(model_path: str, val_df_path: str, val_img_dir: str):
    print("--- Starting Threshold Optimization ---")
    
    # 1. Load Model and Data
    model = ForgeryDetectionUNet(in_channels=6, out_channels=1).to(DEVICE)
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    except FileNotFoundError:
        print(f"ERROR: Model weights not found at {model_path}. Please run forgery_training_template.py first.")
        return

    model.eval()
    
    try:
        val_df = pd.read_csv(val_df_path)
    except FileNotFoundError:
        print(f"ERROR: Validation mask CSV not found at {val_df_path}. Run training first.")
        return

    val_dataset = ValidationForgeryDataset(val_df, val_img_dir)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Collect all predictions and ground truth masks
    all_preds = []
    all_targets = []
    
    print(f"Generating predictions for {len(val_dataset)} validation images...")

    with torch.no_grad():
        for images, masks in val_dataloader:
            images = images.to(DEVICE)
            outputs = model(images)
            
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(masks.cpu().numpy())

    raw_preds = np.concatenate(all_preds, axis=0).flatten()
    raw_targets = np.concatenate(all_targets, axis=0).flatten()
    
    # 3. Perform Threshold Sweep (0.10 to 0.90 in steps of 0.05)
    thresholds = np.arange(0.1, 0.95, 0.05)
    best_iou = 0.0
    best_threshold = 0.5
    
    print("\nStarting threshold sweep...")
    print("-" * 35)

    for T in thresholds:
        binary_preds = (raw_preds > T).astype(np.uint8)
        current_iou = calculate_iou(binary_preds, raw_targets)
        
        print(f"Threshold {T:.2f}: IoU = {current_iou:.6f}")
        
        if current_iou > best_iou:
            best_iou = current_iou
            best_threshold = T

    # 4. Report Results
    print("-" * 35)
    print(f"Optimization Complete!")
    print(f"🎉 Best IoU found: {best_iou:.6f}")
    print(f"🔑 Optimal Threshold: {best_threshold:.2f}")
    print("-" * 35)
    print("ACTION REQUIRED: Update OPTIMAL_THRESHOLD in generate_submission.py with this value!")
    return best_threshold


if __name__ == '__main__':
    optimize_threshold(
        model_path=MODEL_WEIGHTS_PATH, 
        val_df_path=VALIDATION_MASKS_CSV,
        val_img_dir=VALIDATION_IMAGES_DIR
    )
