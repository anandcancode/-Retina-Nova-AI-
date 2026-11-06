"""
Generate sample retinal fundus images for all 5 stages so users can immediately test the upload pipeline.
Saves to:
- static/images/samples/
- dataset/samples/
"""

import os
import cv2
from train import create_sample_fundus_image

SAMPLE_DIR_STATIC = "static/images/samples"
SAMPLE_DIR_DATASET = "dataset/samples"

os.makedirs(SAMPLE_DIR_STATIC, exist_ok=True)
os.makedirs(SAMPLE_DIR_DATASET, exist_ok=True)

stages = [
    (0, "stage0_normal.png", "Normal Retina (No DR)"),
    (1, "stage1_mild_dr.png", "Mild NPDR (Microaneurysms)"),
    (2, "stage2_moderate_dr.png", "Moderate NPDR (Exudates & Hemorrhages)"),
    (3, "stage3_severe_dr.png", "Severe NPDR (Multiquadrant Hemorrhages)"),
    (4, "stage4_proliferative_dr.png", "Proliferative DR (Neovascularization)")
]

for stage_idx, filename, label in stages:
    raw_img, enh_img = create_sample_fundus_image(stage_idx, size=(400, 400))
    # Convert RGB to BGR for cv2 saving
    raw_bgr = cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR)
    
    path_static = os.path.join(SAMPLE_DIR_STATIC, filename)
    path_dataset = os.path.join(SAMPLE_DIR_DATASET, filename)
    
    cv2.imwrite(path_static, raw_bgr)
    cv2.imwrite(path_dataset, raw_bgr)
    print(f"Generated sample fundus image: {path_static} ({label})")

print("Sample retinal fundus images generated successfully!")

