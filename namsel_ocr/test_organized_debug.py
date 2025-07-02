#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('.')

from namsel import *
from page_elements2 import PageElements
from classify import load_cls
import cv2

# Load classifier
fast_cls = load_cls('logistic-cls')

def test_organized_debug():
    print('=== Testing organized debug system with paragraph.png ===')
    
    # Expected first line
    expected_first = "སྐྱེ་དངོས་ཀྱིས་སྐྱེད་པའམ་སྐྱེ་དངོས་ནས་"
    print(f'Expected first syllable: {expected_first[:3]} (skyed)')
    
    # Test only paragraph.png with new organized debug system
    img_file = '../paragraph.png'
    print(f'Processing: {img_file}')
    
    try:
        # Load image
        img = cv2.imread(img_file, 0)
        if img is None:
            print(f"Could not load {img_file}")
            return
            
        print(f'[DEBUG] Loaded image: {img_file}, shape: {img.shape}')
        
        # Process with PageElements to trigger organized debug output
        page_elem = PageElements(img, fast_cls, flpath=img_file)
        
        print(f'[DEBUG] Found {len(page_elem.get_boxes())} character boxes')
        print(f'[DEBUG] Check the generated debug_normalized_paragraph_charXX_*.png files')
        print(f'[DEBUG] First 5 character boxes:')
        for i in range(min(5, len(page_elem.get_boxes()))):
            cbox = page_elem.get_boxes()[i]
            print(f'  Char {i:02d}: x={cbox[0]}, y={cbox[1]}, w={cbox[2]}, h={cbox[3]}')
        
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_organized_debug()