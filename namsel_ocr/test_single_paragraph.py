#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('.')

from namsel import *
import cv2

def test_single_paragraph():
    print('=== Testing paragraph.png ONLY ===')
    
    # Expected first line
    expected_first = "སྐྱེ་དངོས་ཀྱིས་སྐྱེད་པའམ་སྐྱེ་དངོས་ནས་"
    print(f'Expected first syllable: {expected_first[:3]} (skyed)')
    
    # Test paragraph.png with character-level debugging
    img_file = '../paragraph.png'
    print(f'Processing: {img_file}')
    
    try:
        # Load image
        img = cv2.imread(img_file, 0)
        if img is None:
            print(f"Could not load {img_file}")
            return
            
        print(f'[DEBUG] Loaded image: {img_file}, shape: {img.shape}')
        
        # Create segmenter and recognizer instances
        char_boxes, lines_chars, small_cc_lines_chars = segmenter_9_13(img_file, img)
        print(f'[DEBUG] Found {len(char_boxes)} character boxes')
        
        # Run recognition
        recognized_text = recognize_chars_9_13(img_file, char_boxes, lines_chars, 
                                             small_cc_lines_chars, recognizer='hmm', 
                                             postprocess=False, viterbi_postprocessing=False)
        
        print(f'Actual output: "{recognized_text.strip()}"')
        print(f'Expected:      "{expected_first}"')
        
        # Show first few characters comparison
        actual_chars = list(recognized_text.strip())
        expected_chars = list(expected_first)
        
        print('\nFirst 10 characters:')
        for i in range(min(len(actual_chars), len(expected_chars), 10)):
            match = "✓" if actual_chars[i] == expected_chars[i] else "✗"
            print(f'  {i}: "{actual_chars[i]}" vs "{expected_chars[i]}" {match}')
            
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_single_paragraph()