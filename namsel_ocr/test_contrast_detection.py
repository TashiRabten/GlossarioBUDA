#!/usr/bin/env python3
import cv2 as cv
import numpy as np

def test_first_syllable_contrast():
    # Load the original image
    img = cv.imread('debug_01_raw_paragraph.png', cv.IMREAD_GRAYSCALE)
    if img is None:
        print("Could not load image")
        return
    
    # Position of first syllable based on our analysis (around x=30, y=25)
    # Extract region around first syllable with different padding sizes
    x, y = 30, 25
    median_height = 14  # From our previous analysis
    
    for padding_factor in [0.5, 1.0, 1.5, 2.0]:
        padding = int(median_height * padding_factor)
        print(f"\n=== Padding factor: {padding_factor}, padding: {padding} pixels ===")
        
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img.shape[1], x + 20 + padding)  # Assume ~20px wide syllable
        y2 = min(img.shape[0], y + median_height + padding)
        
        region = img[y1:y2, x1:x2]
        if region.size > 0:
            std_contrast = np.std(region.astype(np.float32))
            mean_intensity = np.mean(region)
            
            print(f"  Region: ({x1},{y1})-({x2},{y2}), size: {region.shape}")
            print(f"  Std contrast: {std_contrast:.1f}")
            print(f"  Mean intensity: {mean_intensity:.1f}")
            print(f"  Would be flagged as low-contrast: {std_contrast < 55.0}")
            
            # Save the region for inspection
            filename = f"debug_first_syllable_padding_{padding_factor:.1f}.png"
            cv.imwrite(filename, region)
            print(f"  Saved: {filename}")

def test_all_syllables_contrast():
    # Load the original image
    img = cv.imread('debug_01_raw_paragraph.png', cv.IMREAD_GRAYSCALE)
    if img is None:
        print("Could not load image")
        return
    
    # Approximate positions of the three སྐྱེ syllables
    syllables = [
        (30, 25, "First སྐྱེ"),
        (130, 25, "Middle སྐྱེད"), 
        (240, 25, "Third སྐྱེ")
    ]
    
    median_height = 14
    padding = int(median_height * 0.5)  # Same as in the OCR system
    
    print("=== Syllable Contrast Comparison ===")
    for x, y, label in syllables:
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img.shape[1], x + 20 + padding)
        y2 = min(img.shape[0], y + median_height + padding)
        
        region = img[y1:y2, x1:x2]
        if region.size > 0:
            std_contrast = np.std(region.astype(np.float32))
            print(f"{label}: contrast={std_contrast:.1f}, low-contrast={std_contrast < 55.0}")

if __name__ == "__main__":
    test_first_syllable_contrast()
    test_all_syllables_contrast()