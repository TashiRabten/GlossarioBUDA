#!/usr/bin/env python3

import cv2
import os
import sys
from namsel import PageRecognizer
from config_manager import Config, default_config

def test_paragraph2():
    print("=== Testing paragraph2.png ===")
    
    config = Config()
    page_recognizer = PageRecognizer('../paragraph2.png', config)
    
    result = page_recognizer.recognize_page(text=True)
    
    print(f"Final output: '{result}'")
    print(f"Length: {len(result)} characters")
    
    # Count syllables vs tshegs
    syllables = 0
    tshegs = 0
    for char in result:
        if char == '་':
            tshegs += 1
        elif char != '\n' and char != ' ':
            syllables += 1
            
    print(f"Syllables: {syllables}, Tshegs: {tshegs}, Total: {syllables + tshegs}")
    
    # Show first 20 characters
    print(f"First 20 chars: {repr(result[:20])}")

if __name__ == "__main__":
    test_paragraph2()