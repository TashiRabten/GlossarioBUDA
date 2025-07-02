#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('.')

from namsel import *
import cv2

def test_paragraph_first_char():
    print("=== Testing paragraph.png first character ===")
    
    # Expected first syllable should be "སྐྱེ" (skyed)
    expected_first = "སྐྱེ་དངོས་ཀྱིས་སྐྱེད་པའམ་སྐྱེ་དངོས་ནས་"
    print(f"Expected first syllable: {expected_first[:3]} (skyed)")
    
    # Process the image
    segmenter = Segmenter()
    recognizer = Recognizer()
    
    # Load and preprocess
    img_file = '../paragraph.png'
    img = cv2.imread(img_file, 0)
    
    if img is None:
        print(f"Could not load {img_file}")
        return
        
    # Get page elements
    pg_lst = segmenter.page_elements(img, img_file)
    
    if not pg_lst:
        print("No page elements found")
        return
        
    # Get the first page
    page = pg_lst[0]
    
    # Find the first line
    if not page.lines_chars:
        print("No lines found")
        return
        
    first_line_chars = page.lines_chars[0]
    print(f"First line has {len(first_line_chars)} characters")
    
    # Look at the first few characters
    for i, char_idx in enumerate(first_line_chars[:5]):
        if i >= len(page.new_boxes):
            break
            
        box = page.new_boxes[i]
        print(f"Character {i}: box={box}")
        
        # Check if there are cached predictions
        if hasattr(page, 'char_prds') and i < len(page.char_prds):
            pred_class = page.char_prds[i]
            if pred_class in page.class_to_char:
                pred_char = page.class_to_char[pred_class]
                print(f"  Predicted: class {pred_class} = '{pred_char}'")
            else:
                print(f"  Predicted: class {pred_class} (unknown)")
        
    # Run recognition to get the actual output
    recognized_text = recognizer.recognize_image(img_file, viterbi_postprocessing=False)
    print(f"\nActual output: '{recognized_text.strip()}'")
    print(f"Expected:      '{expected_first}'")
    
    # Check if they match at the beginning
    actual_chars = list(recognized_text.strip())
    expected_chars = list(expected_first)
    
    print("\nCharacter-by-character comparison:")
    for i in range(min(len(actual_chars), len(expected_chars), 10)):
        match = "✓" if actual_chars[i] == expected_chars[i] else "✗"
        print(f"  Pos {i}: '{actual_chars[i]}' vs '{expected_chars[i]}' {match}")

if __name__ == '__main__':
    test_paragraph_first_char()