#!/usr/bin/env python3

import sys
import warnings
warnings.filterwarnings("ignore")
print("ignoring all warnings")

# Test paragraph.png and trace the first syllable
print("=== Testing paragraph.png - First Syllable Analysis ===")

from PIL import Image
import namsel

# Load the image
img = Image.open("../paragraph.png")

# Initialize the page recognizer
pr = namsel.PageRecognizer(img, conf=namsel.default_config)

# Process and get detailed debug info
try:
    result = pr.get_text()
    print("CURRENT OUTPUT:")
    print(repr(result))
    print("\nCURRENT OUTPUT (clean):")
    print(result.strip())
    
    # Expected output from paragraph.txt (first line)
    expected_first_line = "སྐྱེ་དངོས་ཀྱིས་སྐྱེད་པའམ་སྐྱེ་དངོས་ནས་"
    print(f"\nEXPECTED FIRST LINE:")
    print(expected_first_line)
    
    # Compare first characters
    current_lines = result.strip().split('\n')
    if current_lines:
        first_line = current_lines[0]
        print(f"\nFIRST LINE COMPARISON:")
        print(f"Current:  '{first_line}'")
        print(f"Expected: '{expected_first_line}'")
        
        # Character by character analysis
        print(f"\nCHARACTER ANALYSIS:")
        print(f"First char - Current: '{first_line[0] if first_line else 'NONE'}' vs Expected: '{expected_first_line[0]}'")
        if first_line and first_line[0] != expected_first_line[0]:
            print(f"❌ MISMATCH: Got '{first_line[0]}' (should be 'ས' from སྐྱེ)")
        
except Exception as e:
    print(f"Error processing: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Check recent normalized images for first syllable ===")
print("Look for the most recent debug_normalized_*.png files that correspond to the first character/syllable")
print("The first syllable 'སྐྱེ' is being misclassified - need to find its normalized image")