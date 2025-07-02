# Namsel OCR Debug Session Report

## Problem Summary
Namsel OCR has correct contour detection (60 contours → 30 main characters) but produces wrong final text output with:
1. **Character misrecognition**: Syllables (པ, མ, ི, ག) being replaced by tshegs (་)
2. **Missing characters**: Correctly recognized syllables disappearing from final output
3. **Wrong tsheg positions**: Real tshegs appearing in wrong positions

## Test Environment

### Test Script
```bash
python3 test_preprocessing_direct.py
```

### Test Images
- **paragraph2.png**: Main problematic image (2 lines, 44 expected chars)
- **test-1.png**: Simpler test case (1 line, verification image)
- **tsheg-la-tsheg.png**: Working correctly (reference)

### Expected vs Actual Outputs

**paragraph2.png:**
- Expected: `དྲན་པ་མི་གསལ་བའམ་བརྗེད་ངས་སམ། རྣམ་གཡེང་ཅན་གྱི་རང་བཞིན།` (44 chars)
- Actual: `དྲན་་ག་ལ་འ་བརྗེ་ངསསམ་རྗེདངས་ཿལྟཱི་། རྣམ་ཡེངཅན་རངབ་` (48+ chars)

**test-1.png:**  
- Expected: Should contain `སྐྱ` and proper tsheg positions
- Actual: `ནི་ངོས་་` (missing `སྐྱ`, wrong tsheg positions)

## OCR Pipeline - 7 Phases Analysis

### Phase 1: Image Loading & Preprocessing ✅ WORKING
**Status**: Correct
**Files**: `namsel.py`
**Classes/Methods**:
- `PageRecognizer.__init__()` - Image loading
- `PageRecognizer.recognize_page()` - Main entry point
- Scale-invariant preprocessing functions

**Debug Info**: 
```
[DEBUG] Saved raw image: debug_01_raw_paragraph2.png
[PREPROCESSING] Applying scale-invariant preprocessing
[DEBUG] Saved preprocessed image: debug_02_preprocessed_paragraph2.png
```
**Findings**: Image quality is good, no distortion or preprocessing issues.

### Phase 2: Contour Detection ✅ WORKING  
**Status**: Correct
**Files**: `page_elements2.py`
**Classes/Methods**:
- `PageElements.__init__()` - Main contour detection
- `PageElements._contours()` - OpenCV contour finding
- `PageElements.get_boxes()` - Bounding box extraction
- `PageElements._boxes()` - Box calculation from contours

**Debug Info**:
```
[CHAR DEBUG] Found 60 characters
[CHAR DEBUG] Char 0: x=239, y=66, w=4, h=25
[CHAR DEBUG] Saved 60/60 character images as debug_extracted_char_XXX.png
```
**Key Files**: `debug_char_*.png`, `debug_contour_letter_*.png`
**Findings**: 60 contours detected correctly, character extraction quality is good.

### Phase 3: Document Analysis ✅ WORKING
**Status**: Correct  
**Files**: `page_elements2.py`
**Classes/Methods**:
- `PageElements._analyze_document_type()` - SINGLE-LINE vs MULTILINE detection
- `PageElements.__init__()` - Position filtering logic
- Font-aware filtering with adaptive thresholds

**Debug Info**:
```
[DOCUMENT ANALYSIS] Document type: MULTILINE
[DOCUMENT ANALYSIS] Y-coordinate range: 8-75, spread: 67
[HIERARCHY DEBUG] Filtering small contour X: area=Y, dims=WxH
```
**Findings**: Correctly identifies MULTILINE documents, proper filtering ranges.

### Phase 4: Line Detection ✅ WORKING
**Status**: Correct
**Files**: `line_breaker.py`
**Classes/Methods**:
- `LineCut.__init__()` - Main line detection
- `LineCut.find_line_breaks()` - Horizontal line separation
- `LineCut.get_line_chars()` - Character assignment to lines
- Character sorting by x-coordinate within lines

**Debug Info**:
```
[LINECUT DEBUG] Line 0: 17 chars: [list of indices]
[LINECUT DEBUG] Line 1: 13 chars: [list of indices]  
[VECTOR DEBUG] Line 0: 17 vectors, 17 new_boxes
[VECTOR DEBUG] Line 1: 13 vectors, 13 new_boxes
```
**Findings**: 
- Correctly assigns 30 main characters (17+13) to 2 lines
- Characters processed in correct left-to-right order
- Total: 60 contours → 30 main character vectors

### Phase 5: Character Segmentation ✅ WORKING
**Status**: Correct
**Files**: `segment.py`
**Classes/Methods**:
- `segment_chars()` - Main segmentation function
- `Segmenter.construct_vector_set_stochastic()` - Vector creation
- `PechaCharSegmenter` - Character combination and feature extraction
- Feature extraction and normalization (346-dimensional vectors)

**Debug Info**:
```
[VECTOR DEBUG] Line 0, vector 0: shape=(346,), box=[12, 16, 13, 21]
[VECTOR DEBUG] Line 0, vector 1: shape=(346,), box=[27, 16, 13, 24]
[SEGMENTER DEBUG] Using segmenter type: stochastic
```
**Findings**: Feature vectors (346 dimensions) generated correctly for each character.

### Phase 6: Character Recognition 🔄 PARTIALLY WORKING
**Status**: Mixed - Individual recognition correct, but pipeline mixing
**Files**: `recognize.py`, `page_elements2.py`
**Classes/Methods**:

#### Main Character Pipeline:
- `recognize_chars_hmm()` - HMM-based character recognition
- `predict_proba()` - Classifier predictions
- Cache system for storing predictions

#### Punctuation Pipeline:
- `PageElements` tsheg detection logic
- Small contour processing in `page_elements2.py`
- Adaptive threshold detection for punctuation

**Debug Info**:
```
[CHAR DEBUG] Character 0: classified as class 50 = 'ད' (confidence: 0.857)
[CHAR DEBUG] Character 1: classified as class 455 = 'སྐྱ' (confidence: 0.849)  
[CHAR DEBUG] Character 4: classified as class 510 = '་' (confidence: 0.545)
[PUNCT DEBUG] Detected TSHEG at i=4, box=(82, 23, 4, 5)
```

**Two Separate Pipelines Identified**:

#### Main Character Pipeline:
- Processes 30 main characters through vector recognition
- High confidence predictions (0.8-0.9) for syllables
- Results stored in cache system

#### Punctuation Pipeline:  
- Processes small contours separately for tsheg detection
- Lower confidence predictions (0.3-0.5) for tshegs
- Uses different insertion logic

**Findings**: Individual character recognition is mostly correct, but the two pipelines are getting mixed up.

### Phase 7: Results Assembly ❌ BROKEN - CORE ISSUE
**Status**: Critical failure in cache mapping system
**Files**: `recognize.py`, `namsel.py`
**Classes/Methods**:
- `recognize_chars_hmm()` - Main assembly function
- Cache mapping system (lines with `[CACHE MAPPING]`)
- `PageRecognizer.extract_lines()` - Final text construction
- Tsheg insertion logic (multiple locations in `recognize.py`)

**Debug Info**:
```
[CACHE MAPPING] Vector 0 -> original chars [8, 1] -> using char 8
[CACHE MAPPING] Vector 2 -> original chars [9] -> using char 9
[CACHE MAPPING] Vector 3 -> original chars [5] -> using char 5
[ASSEMBLY DEBUG] Line 0, pos 0: adding 'ནི' (char 8)
[ASSEMBLY DEBUG] Line 0, pos 1: adding '་' (char 6)
```

**Critical Problems**:

1. **Cache Mapping Corruption**: 
   - Character 1 (`'སྐྱ'`) recognized correctly but **completely missing** from final output
   - Vector-to-character mappings are scrambled
   - Cache system losing track of character assignments

2. **Pipeline Mixing**:
   - Main character results getting overwritten by punctuation pipeline
   - Tsheg insertion logic interfering with syllable positions

3. **Assembly Order Issues**:
   ```
   [ASSEMBLY DEBUG] Line 0, pos 0: adding 'ནི' (char 8)
   [ASSEMBLY DEBUG] Line 0, pos 1: adding '་' (char 6)  
   [ASSEMBLY DEBUG] Line 0, pos 2: adding 'ངོ' (char 9)
   ```
   Characters appear in wrong positions due to mapping errors.

## Key Code Locations

### Space Insertion Logic (FIXED):
- **File**: `recognize.py` lines 780, 1228, 1492
- **Fix Applied**: Increased threshold from 1.5/2.0*tsek_mean to 8.0*tsek_mean
- **Status**: ✅ Extra spaces eliminated

### Character Order Processing:
- **File**: `line_breaker.py` - Line assignment and sorting
- **File**: `segment.py` - Vector creation from line assignments  
- **File**: `recognize.py` - Cache mapping and final assembly

### Cache Mapping System:
- **File**: `recognize.py` around lines with `[CACHE MAPPING]` debug
- **Issue**: Vector-to-character correspondence broken
- **Impact**: Characters lost or mapped to wrong positions

## Debug Commands for Quick Verification

```bash
# Run full test
python3 test_preprocessing_direct.py

# Check character detection order  
python3 test_preprocessing_direct.py 2>/dev/null | grep "CHAR DEBUG.*Char.*:"

# Check vector processing order
python3 test_preprocessing_direct.py 2>/dev/null | grep "VECTOR DEBUG] Line"

# Check cache mapping issues
python3 test_preprocessing_direct.py 2>&1 | grep "CACHE MAPPING"

# Check final assembly
python3 test_preprocessing_direct.py 2>&1 | grep "ASSEMBLY DEBUG"

# Check character classifications
python3 test_preprocessing_direct.py 2>&1 | grep "Character.*classified"
```

## Root Cause Analysis

**Primary Issue**: The cache mapping system in Phase 7 (Results Assembly) is corrupted.

**Evidence**:
1. Individual character recognition is correct (Phase 6)
2. Vector generation is correct (Phase 5) 
3. Cache retrieval loses/scrambles character assignments (Phase 7)
4. Two separate pipelines (main chars vs punctuation) interfere with each other

**Next Steps**: 
1. Fix cache mapping logic to preserve character-to-position correspondence
2. Separate punctuation pipeline interference from main character pipeline
3. Ensure recognized characters appear in correct final positions

## Status Summary
- ✅ Phases 1-5: Working correctly
- 🔄 Phase 6: Recognition correct, but pipeline separation issues  
- ❌ Phase 7: Critical cache mapping and assembly failures

The architecture is sound - the issue is in the final assembly logic where correctly recognized characters get lost or scrambled.

---

# LATEST UPDATE: Character Segmentation Issue Identified

## Problem: First Syllable Segmentation Failure

**Date**: 2025-07-02  
**Status**: ✅ ROOT CAUSE IDENTIFIED - Character segmentation breaking compound syllables

### Debug System Implementation
Successfully implemented organized debug system:
- **Before**: Random numbered files like `debug_normalized_7795.png` mixed from all sources
- **After**: Organized naming like `debug_normalized_paragraph_char146_x27_y19.png`
- **Features**: Source file identification, character index, position coordinates

### First Syllable Analysis Results

**Expected**: `སྐྱེ་དངོས་ཀྱིས་སྐྱེད་པའམ་སྐྱེ་དངོས་ནས་` (paragraph.txt)  
**Actual**: Characters starting with fragments instead of complete "སྐྱེ" syllable

**Key Finding**: The first character at position (x=27, y=19):
- **Contour file**: `debug_contour_letter_line0_char0_27_19_14x32.png`
- **Normalized file**: `debug_normalized_paragraph_char146_x27_y19.png` 
- **Content**: Just a stroke fragment, not complete "སྐྱེ" syllable

### Root Cause: Compound Syllable Fragmentation

The compound Tibetan syllable "སྐྱེ" (skyed) is being incorrectly segmented into multiple separate character boxes:

| Character | Position | Content | Should Be |
|-----------|----------|---------|-----------|
| char146 | x=27, y=19 | Stroke fragment | Part of "སྐྱེ" |
| char145 | x=58, y=19 | "ས" component | Part of "སྐྱེ" |
| char144 | x=93, y=19 | "སྐྱ" component | Part of "སྐྱེ" |
| char142 | x=131, y=19 | "སྐྱེ" component | Part of "སྐྱེ" |

**Critical Issue**: Instead of one unified "སྐྱེ" syllable, the system creates 4 separate character boxes, each containing fragments that get classified individually.

### Technical Solution Required

**Problem Location**: Character segmentation/contour detection phase in `page_elements2.py`
**Fix Needed**: Modify segmentation algorithm to:
1. Detect compound syllables as single units
2. Prevent splitting of connected Tibetan character components  
3. Keep consonant clusters and vowel marks together
4. Process complete syllables rather than individual strokes

### Files Modified for Debug System
```python
# feature_extraction.py - Line 86
def normalize_and_extract_features(arr, debug_source=None, debug_char_idx=None, debug_position=None):
    # Creates organized debug images: debug_normalized_[source]_char[XX]_[position].png

# page_elements2.py - Lines 616, 874  
cbox = self.get_boxes()[i]
debug_pos = f"x{cbox[0]}_y{cbox[1]}"
features = normalize_and_extract_features(tmparr, debug_source=self.flpath, debug_char_idx=i, debug_position=debug_pos)
```

### Updated Status Summary
- ✅ Phases 1-5: Working correctly
- 🔄 Phase 6: Recognition correct, but pipeline separation issues  
- ❌ Phase 7: Critical cache mapping and assembly failures
- ❌ **NEW**: Phase 2 (Contour Detection) - Breaking compound syllables into fragments

### Next Steps Priority
1. **HIGH**: Fix character segmentation to preserve compound syllables in Phase 2
2. **MEDIUM**: Test segmentation fix on all three images (tsheg-la-tsheg.png, paragraph.png, paragraph2.png)  
3. **MEDIUM**: Fix Phase 7 cache mapping issues (previous priority)
4. **LOW**: Clean up debug images after validation

**Status**: Ready for segmentation algorithm modification to fix compound syllable detection.

---

# LATEST UPDATE: Specific Syllable Fragmentation Pattern Analysis

**Date**: 2025-07-02  
**Status**: 🔍 PATTERN IDENTIFIED - Specific syllables fragment while identical ones remain intact

## Observed Problem Pattern

### Test Case: paragraph.png
**Expected first line**: `སྐྱེ་དངོས་ཀྱིས་སྐྱེད་པའམ་སྐྱེ་དངོས་ནས་`  
**Actual first line**: `ནི་དངོས་ཀྱིས་སྐྱེད་པའམ་ནི་དངོས་ནས་`  

### Critical Observation: Inconsistent Behavior
The same syllable "སྐྱེ" shows different segmentation results:
- **Position 1**: "སྐྱེ" → fragments to "ནི" ❌ BROKEN
- **Position 6**: "སྐྱེད" → correctly detected ✅ WORKING  
- **Position 11**: "སྐྱེ" → fragments to "ནི" ❌ BROKEN

## Character Detection Results
Generated debug files show:
```
[CHAR DEBUG] Character 143: classified as class 56 = 'ནི' (confidence: 0.586)
[CHAR DEBUG] Character 144: classified as class 458 = 'སྐྱེ' (confidence: 0.646)  
[CHAR DEBUG] Character 151: classified as class 56 = 'ནི' (confidence: 0.575)
```

**Character boxes detected**:
- Character 143: `debug_char_143_212_19_10x7.png` - 10x7 pixels (very small fragment)
- Character 144: `debug_char_144_131_19_14x32.png` - 14x32 pixels (full syllable)
- Character 151: `debug_char_151_26_19_11x7.png` - 11x7 pixels (very small fragment)

## Local Contrast Analysis Results

### Test Script Created: `test_local_conditions.py`
```python
# Analyzes local image conditions around each syllable position
# Tests different adaptive threshold parameters on each region
```

### Findings from Local Analysis:
```
First སྐྱེ (fragmented):
  Mean intensity: 240.1, Std deviation: 50.8
  Contrast: 239.0, Min/Max: 16/255

Middle སྐྱེད (intact):  
  Mean intensity: 230.1, Std deviation: 64.6
  Contrast: 239.0, Min/Max: 16/255

Third སྐྱེ (fragmented):
  Mean intensity: 229.7, Std deviation: 66.6
  Contrast: 239.0, Min/Max: 16/255
```

### Contour Count Analysis:
When testing different adaptive threshold parameters:
- **First syllable**: 5 contours (under-segmented) ← Missing parts
- **Middle syllable**: 9 contours (properly segmented) ← Working correctly  
- **Third syllable**: 11 contours (over-segmented) ← Too many parts

## Generated Debug Files for Investigation

### Local Region Images:
- `debug_local_region_0_First_སྐྱེ_fragmented.png`
- `debug_local_region_1_Middle_སྐྱེད_intact.png` 
- `debug_local_region_2_Third_སྐྱེ_fragmented?.png`

### Threshold Testing Results:
- `debug_threshold_first_block25_c4.png` - Shows merged character parts
- `debug_threshold_middle_block25_c4.png` - Shows well-separated components
- `debug_threshold_third_block25_c4.png` - Shows over-fragmented parts

### Character Fragment Images:
- `debug_normalized_1037.png` - Fragment classified as "ནི" (first syllable)
- `debug_normalized_8356.png` - Complete syllable classified as "སྐྱེ" (correct detection)
- `debug_normalized_4515.png` - Fragment classified as "ནི" (third syllable)

## Scale-Adaptive Thresholding Implementation

### Current System Status:
```
[ADAPTIVE THRESHOLD] Block size: 25, C: 3
[CONTRAST ANALYSIS] Character regions: 110
[CONTRAST ANALYSIS] Low-contrast characters: 3/110
[CONTRAST ADAPTIVE] Low contrast detected, applying C reduction: -1
```

## Assembly Debug Pattern:
```
[ASSEMBLY DEBUG] Line 0, pos 0: adding 'ནི' (from k=[26, 19, 15, 32, ...])
[ASSEMBLY DEBUG] Line 0, pos 6: adding 'སྐྱེད' (correctly preserved)
[ASSEMBLY DEBUG] Line 0, pos 11: adding 'ནི' (should be 'སྐྱེ')
```

## Questions for Further Investigation

**Claude Opus** - Based on these observations, we need to understand why identical syllables fragment differently. To solve this problem, please tell me what specific classes, methods, or code sections you need to examine. Consider requesting:

1. **Contour Detection Classes**: Which classes in `page_elements2.py` handle the initial contour detection?
2. **Adaptive Thresholding Logic**: What methods control the OpenCV adaptive threshold parameters?
3. **Character Segmentation Pipeline**: How does the system decide to split or merge character components?
4. **Local Processing Methods**: Are there any methods that process image regions differently based on position?
5. **Syllable Assembly Logic**: Which classes are responsible for reconstructing compound syllables?

**What classes/methods do you need to see to diagnose why the first and third "སྐྱེ" syllables lose their main components while the middle "སྐྱེད" syllable remains intact?**

## Files Available for Analysis
- Source images: `debug_01_raw_paragraph.png`
- Preprocessed: `debug_02_preprocessed_paragraph.png` 
- Post-threshold: `debug_03_before_segmentation_paragraph.png`
- Character extractions: `debug_char_*.png` (152 files)
- Normalized features: `debug_normalized_*.png`
- Local regions: `debug_local_region_*.png`
- Threshold tests: `debug_threshold_*.png`
- Test scripts: `test_local_conditions.py`, `test_contrast_detection.py`

**Status**: Pattern documented - Ready for targeted class analysis to solve differential syllable fragmentation.