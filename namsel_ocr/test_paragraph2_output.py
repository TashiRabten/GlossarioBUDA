#!/usr/bin/env python3

import sys
import warnings
warnings.filterwarnings("ignore")
print("ignoring all warnings")

# Test just paragraph2.png and extract the final output
print("=== Testing paragraph2.png ONLY ===")

# Test the functionality
from PIL import Image
import namsel

# Load the image
img = Image.open("../paragraph2.png")

# Initialize the page recognizer
pr = namsel.PageRecognizer(img, conf=namsel.default_config)

# Process and get the text
try:
    result = pr.get_text()
    print("CURRENT OUTPUT:")
    print(repr(result))
    print("\nCURRENT OUTPUT (clean):")
    print(result.strip())
    
    # Expected output from paragraph2.txt
    expected = "དྲན་པ་མི་གསལ་བའམ་བརྗེད་ངས་སམ།\nརྣམ་གཡེང་ཅན་གྱི་རང་བཞིན།"
    print("\nEXPECTED OUTPUT:")
    print(expected)
    
    # Line by line comparison
    current_lines = result.strip().split('\n')
    expected_lines = expected.split('\n')
    
    print("\n=== LINE-BY-LINE COMPARISON ===")
    for i, (current, expected_line) in enumerate(zip(current_lines, expected_lines)):
        print(f"Line {i+1}:")
        print(f"  Current:  {repr(current)}")
        print(f"  Expected: {repr(expected_line)}")
        print(f"  Match: {current == expected_line}")
        
        if current != expected_line:
            print(f"  Char differences:")
            for j, (c, e) in enumerate(zip(current, expected_line)):
                if c != e:
                    print(f"    Position {j}: got '{c}' expected '{e}'")
    
except Exception as e:
    print(f"Error processing: {e}")
    import traceback
    traceback.print_exc()