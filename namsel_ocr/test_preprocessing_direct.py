#!/usr/bin/env python3
"""
Test current preprocessing status - run directly from namsel_ocr directory
"""
import sys
import os

# Import namsel directly
from namsel import PageRecognizer
from config_manager import Config, default_config

# Test tsheg-la-tsheg.png
print('=== Testing tsheg-la-tsheg.png ===')
try:
    config = Config()
    recognizer = PageRecognizer('../tsheg-la-tsheg.png', config)
    results = recognizer.recognize_page(text=True)
    print('Output:', repr(results))
    
    # Get contours from page elements
    recognizer.get_page_elements()
    if hasattr(recognizer, 'page_elements') and recognizer.page_elements:
        print('Number of contours found:', len(recognizer.page_elements.contours))
    else:
        print('No page elements found')
    if results and results.strip() == "":
        print('❌ BROKEN: No output produced')
    elif results and "ལ་" in results:
        print('✅ WORKING: Contains expected ལ་')
    else:
        print('⚠️  ISSUE: Wrong output')
except Exception as e:
    print('Error:', e)
    import traceback
    traceback.print_exc()

print()

# Test paragraph2.png  
print('=== Testing paragraph2.png ===')
try:
    config = Config()
    recognizer = PageRecognizer('../paragraph2.png', config)
    results = recognizer.recognize_page(text=True)
    print('Output (first 50 chars):', repr(results[:50] if results else None))
    
    # Get contours from page elements
    recognizer.get_page_elements()
    if hasattr(recognizer, 'page_elements') and recognizer.page_elements:
        contour_count = len(recognizer.page_elements.contours)
        print('Number of contours found:', contour_count)
        if contour_count > 100:
            print('❌ OVER-SEGMENTATION: Too many contours detected')
        elif contour_count < 80:
            print('✅ SEGMENTATION IMPROVED: Reasonable contour count')  
        else:
            print('⚠️  SEGMENTATION STILL HIGH: Some improvement needed')
    else:
        print('No page elements found')
except Exception as e:
    print('Error:', e)
    import traceback
    traceback.print_exc()

print()

# Test paragraph.png  
print('=== Testing paragraph.png ===')
try:
    config = Config()
    recognizer = PageRecognizer('../paragraph.png', config)
    results = recognizer.recognize_page(text=True)
    print('Full Output:')
    print(repr(results))
    print('\nOutput (clean):')
    print(results.strip() if results else "None")
    
    # Expected first line
    expected_first = "སྐྱེ་དངོས་ཀྱིས་སྐྱེད་པའམ་སྐྱེ་དངོས་ནས་"
    print(f'\nExpected first line: {expected_first}')
    
    # Check first character
    if results:
        lines = results.strip().split('\n')
        if lines and lines[0]:
            current_first_char = lines[0][0]
            expected_first_char = expected_first[0]  # 'ས'
            print(f'\nFirst character analysis:')
            print(f'Current: "{current_first_char}"')
            print(f'Expected: "{expected_first_char}" (from སྐྱེ)')
            if current_first_char != expected_first_char:
                print(f'❌ MISMATCH: Got "{current_first_char}" instead of "{expected_first_char}"')
                print('This suggests the first syllable "སྐྱེ" is being misrecognized')
    
    # Get contours from page elements
    recognizer.get_page_elements()
    if hasattr(recognizer, 'page_elements') and recognizer.page_elements:
        contour_count = len(recognizer.page_elements.contours)
        print(f'Number of contours found: {contour_count}')
    else:
        print('No page elements found')
        
except Exception as e:
    print('Error:', e)
    import traceback
    traceback.print_exc()