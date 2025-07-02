#!/usr/bin/env python3
"""
Test paragraph.png specifically 
"""
import sys
import os
import warnings
warnings.filterwarnings("ignore")

# Import namsel directly
from namsel import PageRecognizer
from config_manager import Config, default_config

# Test paragraph.png (larger image)
print('=== Testing data/out/paragraph.png ===')
try:
    config = Config()
    recognizer = PageRecognizer('data/out/paragraph.png', config)
    results = recognizer.recognize_page(text=True)
    print('Output:', repr(results))
    
    # Print first 100 chars for comparison
    if results:
        print('Output (first 100 chars):', repr(results[:100]))
    
    # Expected from paragraph.txt
    expected = """སྐྱེ་དངོས་ཀྱིས་སྐྱེད་པའམ་སྐྱེ་དངོས་ནས་
བྱུང་བ་མིན་མོད། སྐྱེ་དངོས་ཤིག་གམ་
དེའི་ཁོར་ཡུག་ལ་ཤུགས་རྐྱེན་སྤྲོད་སྲིད་
པའི་རྐྱེན་ནམ། ཁྱད་ཆོས། ཡང་ན་བརྒྱུད་
རིམ་ཞིག་ལ་གོ།"""
    
    print('Expected (first line):', expected.split('\n')[0])
    
    if results:
        actual_first_line = results.split('\n')[0] if '\n' in results else results
        print('Actual (first line):', actual_first_line)
        
        if actual_first_line.strip() == expected.split('\n')[0].strip():
            print('✅ PERFECT MATCH!')
        else:
            print('❌ MISMATCH')
            
except Exception as e:
    print('Error:', e)
    import traceback
    traceback.print_exc()