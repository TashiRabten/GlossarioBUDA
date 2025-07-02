#!/usr/bin/env python3

import sys
import os
sys.path.append('/mnt/c/Users/tashi.TASHI-LENOVO/APPS/GlossarioBUDA/namsel_ocr')

import segment
from page_elements2 import PageElements
import numpy as np

def debug_space_insertion(image_path):
    """Debug space insertion logic for a specific image"""
    print(f"=== DEBUGGING SPACE INSERTION FOR {image_path} ===")
    
    # Load image and process
    page = PageElements()
    page.set_image(image_path)
    
    # Get segmentation
    segmentation = segment.segment_chars(page)
    
    # Now trace through the space insertion logic step by step
    print("\n=== TRACING SPACE INSERTION LOGIC ===")
    
    tsek_mean = segmentation.final_box_info.tsek_mean if hasattr(segmentation, 'final_box_info') else 10
    print(f"tsek_mean = {tsek_mean}")
    
    for line_num, vectors in enumerate(segmentation.vectors):
        print(f"\nLine {line_num}: {len(vectors)} vectors")
        new_boxes = segmentation.new_boxes[line_num]
        print(f"Line {line_num}: {len(new_boxes)} boxes")
        
        # Show all boxes
        for i, box in enumerate(new_boxes):
            print(f"  Box {i}: {box}")
        
        # Check gaps between consecutive boxes
        for i in range(len(new_boxes) - 1):
            current_box = new_boxes[i]
            next_box = new_boxes[i + 1]
            
            # Calculate gap: next_box_left - (current_box_left + current_box_width)
            gap = next_box[0] - (current_box[0] + current_box[2])
            
            print(f"  Gap {i}->{i+1}: {gap} pixels")
            print(f"    Current box: x={current_box[0]}, w={current_box[2]}, right_edge={current_box[0] + current_box[2]}")
            print(f"    Next box: x={next_box[0]}")
            print(f"    Gap >= 2*tsek_mean ({2*tsek_mean})? {gap >= 2*tsek_mean}")
            print(f"    Gap >= 1.5*tsek_mean ({1.5*tsek_mean})? {gap >= 1.5*tsek_mean}")
            
            if gap >= 2*tsek_mean:
                print(f"    -> SPACE INSERTED (2*tsek_mean threshold)")
            elif gap >= 1.5*tsek_mean:
                print(f"    -> SPACE INSERTED (1.5*tsek_mean threshold)")
            else:
                print(f"    -> No space inserted")

if __name__ == "__main__":
    debug_space_insertion("paragraph2.png")