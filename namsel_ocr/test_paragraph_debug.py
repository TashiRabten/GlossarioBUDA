#!/usr/bin/env python3
import sys
import warnings
warnings.filterwarnings("ignore")
print("ignoring all warnings")

# Test paragraph.png with same approach as test_preprocessing_direct.py
print("=== Testing paragraph.png ===")

from PIL import Image
import sys
import page_elements2
import numpy as np

# Load and process paragraph.png
img = Image.open('../paragraph.png')
img_arr = np.array(img)

print(f"[IMG DEBUG] paragraph.png: shape={img_arr.shape}")

# Initialize page elements processor like in test_preprocessing_direct.py
shapes = page_elements2.PageElements(img, conf={'viterbi_postprocessing': False,
                                                'recognizer': 'hmm',
                                                'postprocess': False,
                                                'line_segmenter': 'stochastic',
                                                'clear_hr': True,
                                                'detect_o': False})

# Get the text output
try:
    result = shapes.get_text()
    print(f"CURRENT OUTPUT:")
    print(repr(result))
    print(f"\nCURRENT OUTPUT (clean):")
    print(result.strip())
    
    # Expected
    expected = "སྐྱེ་དངོས་ཀྱིས་སྐྱེད་པའམ་སྐྱེ་དངོས་ནས་"
    print(f"\nEXPECTED FIRST LINE:")
    print(expected)
    
    # Find the issue with first syllable
    current_lines = result.strip().split('\n')
    if current_lines:
        first_line = current_lines[0]
        print(f"\nFIRST CHARACTER ANALYSIS:")
        if first_line:
            first_char = first_line[0]
            print(f"Current first char: '{first_char}'")
            print(f"Expected first char: 'ས' (from སྐྱེ)")
            if first_char == 'ང':
                print("❌ CONFIRMED: First syllable detected as 'ང' instead of 'ས' from 'སྐྱེ'")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    
print("\n=== Now look for normalized images ===")
print("The debug normalized images should show the first syllable normalization")
print("Look for the character that's being classified as 'ང' but should be 'སྐྱེ'")