#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('.')

from namsel import *

def test_paragraph_only():
    print('=== Testing paragraph.png ONLY ===')
    
    # Expected first line
    expected_first = "སྐྱེ་དངོས་ཀྱིས་སྐྱེད་པའམ་སྐྱེ་དངོས་ནས་"
    print(f'Expected first syllable: {expected_first[:3]} (skyed)')
    
    # Process only paragraph.png
    img_file = '../paragraph.png'
    
    # Run recognition - this will generate debug normalized images
    output = ocr_line(img_file, viterbi_postprocessing=False)
    
    print(f'Actual output: "{output.strip()}"')
    print(f'Expected:      "{expected_first}"')
    
    # Show first few characters comparison
    actual_chars = list(output.strip())
    expected_chars = list(expected_first)
    
    print('\nFirst 10 characters:')
    for i in range(min(len(actual_chars), len(expected_chars), 10)):
        match = "✓" if actual_chars[i] == expected_chars[i] else "✗"
        print(f'  {i}: "{actual_chars[i]}" vs "{expected_chars[i]}" {match}')

if __name__ == '__main__':
    test_paragraph_only()