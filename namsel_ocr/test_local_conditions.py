#!/usr/bin/env python3
import cv2 as cv
import numpy as np

def analyze_local_region(img, x, y, size=50):
    """Extract and analyze a local region around a syllable"""
    h, w = img.shape
    
    # Extract region
    x1 = max(0, x - size//2)
    y1 = max(0, y - size//2) 
    x2 = min(w, x + size//2)
    y2 = min(h, y + size//2)
    
    region = img[y1:y2, x1:x2]
    
    # Calculate statistics
    mean_intensity = np.mean(region)
    std_intensity = np.std(region)
    min_intensity = np.min(region)
    max_intensity = np.max(region)
    contrast = max_intensity - min_intensity
    
    return {
        'region': region,
        'coords': (x1, y1, x2, y2),
        'mean': mean_intensity,
        'std': std_intensity,
        'min': min_intensity,
        'max': max_intensity,
        'contrast': contrast
    }

def test_syllable_regions():
    # Load the original image
    img = cv.imread('debug_01_raw_paragraph.png', cv.IMREAD_GRAYSCALE)
    if img is None:
        print("Could not load image")
        return
    
    # Expected positions of the three སྐྱེ syllables based on our analysis
    # These are approximate - we'll refine based on what we see
    syllable_positions = [
        (30, 25, "First སྐྱེ (fragmented)"),
        (130, 25, "Middle སྐྱེད (intact)"), 
        (240, 25, "Third སྐྱེ (fragmented?)")
    ]
    
    print("=== Local Region Analysis ===")
    
    for i, (x, y, label) in enumerate(syllable_positions):
        print(f"\n{label}:")
        stats = analyze_local_region(img, x, y, size=60)
        
        print(f"  Position: ({x}, {y})")
        print(f"  Mean intensity: {stats['mean']:.1f}")
        print(f"  Std deviation: {stats['std']:.1f}")
        print(f"  Contrast: {stats['contrast']:.1f}")
        print(f"  Min/Max: {stats['min']}/{stats['max']}")
        
        # Save the local region for visual inspection
        region_filename = f"debug_local_region_{i}_{label.replace(' ', '_').replace('(', '').replace(')', '')}.png"
        cv.imwrite(region_filename, stats['region'])
        print(f"  Saved region: {region_filename}")

def test_adaptive_threshold_locally():
    """Test different adaptive threshold parameters on each region"""
    img = cv.imread('debug_01_raw_paragraph.png', cv.IMREAD_GRAYSCALE)
    if img is None:
        print("Could not load image")
        return
        
    positions = [(30, 25), (130, 25), (240, 25)]
    labels = ["first", "middle", "third"]
    
    # Test different block sizes and C values
    block_sizes = [15, 21, 25, 31]
    c_values = [2, 4, 6]
    
    for i, ((x, y), label) in enumerate(zip(positions, labels)):
        print(f"\n=== Testing {label} syllable at ({x}, {y}) ===")
        
        # Extract larger region for thresholding
        size = 80
        x1 = max(0, x - size//2)
        y1 = max(0, y - size//2)
        x2 = min(img.shape[1], x + size//2)
        y2 = min(img.shape[0], y + size//2)
        
        region = img[y1:y2, x1:x2]
        
        for block_size in block_sizes:
            for c in c_values:
                try:
                    # Apply adaptive threshold
                    threshold = cv.adaptiveThreshold(
                        region, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv.THRESH_BINARY_INV, block_size, c
                    )
                    
                    # Count contours
                    contours = cv.findContours(threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2]
                    num_contours = len(contours)
                    
                    # Save result
                    filename = f"debug_threshold_{label}_block{block_size}_c{c}.png"
                    cv.imwrite(filename, threshold)
                    
                    print(f"  Block={block_size}, C={c}: {num_contours} contours -> {filename}")
                    
                except Exception as e:
                    print(f"  Block={block_size}, C={c}: ERROR - {e}")

if __name__ == "__main__":
    test_syllable_regions()
    test_adaptive_threshold_locally()