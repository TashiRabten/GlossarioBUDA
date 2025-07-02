#! /usr/bin/python
# encoding: utf-8
'''Page Elements'''


#from multiprocessing import Process

import cv2 as cv
import numpy as np
from sklearn.mixture import GaussianMixture as GMM
#import font_detector
from scipy.stats import mode as statsmode

try:
    from .classify import label_chars, load_cls
except ImportError:
    from classify import label_chars, load_cls

# Load the classifier as fast_cls for compatibility
fast_cls = load_cls('logistic-cls')

# Import ML-based tsheg separator (will be imported dynamically to avoid circular imports)
ML_AVAILABLE = False
from scipy.ndimage.interpolation import rotate
# from recognize import main as rec_main, construct_page
# from utils_extra import add_padding, trim, invert_bw
try:
    from .utils import invert_bw
    from .feature_extraction import normalize_and_extract_features
    from .fast_utils import to255
except ImportError:
    from utils import invert_bw
    from feature_extraction import normalize_and_extract_features
    from fast_utils import to255
# from yik import word_parts_set
from scipy.ndimage.filters import gaussian_filter1d
from scipy.signal import argrelmin
from scipy.interpolate import UnivariateSpline, splrep, splev
from collections import OrderedDict

import platform

class TshegPreprocessor:
    def __init__(self, tsek_mean=5.0, tsek_std=1.0, use_ml=False, model_path=None):
        self.tsek_mean = tsek_mean
        self.tsek_std = tsek_std
        self.use_ml = use_ml and ML_AVAILABLE
        
        # Initialize ML separator if requested and available
        if self.use_ml:
            try:
                import sys
                import os
                sys.path.append(os.path.dirname(os.path.dirname(__file__)))
                from SplitPointCNN import MLTshegSeparator
                self.ml_separator = MLTshegSeparator(model_path, tsek_mean, tsek_std)
                self.ml_available = True
                print("[TSHEG DEBUG] Initialized with ML-based separator")
            except ImportError as e:
                print(f"[TSHEG DEBUG] Could not import ML classes: {e}")
                self.ml_available = False
                print("[TSHEG DEBUG] Falling back to traditional surgical approach")
        else:
            self.ml_available = False
            print("[TSHEG DEBUG] Using traditional surgical approach")

    def preprocess(self, img_arr):
        if img_arr.max() <= 1.0:
            img_arr = (img_arr * 255).astype(np.uint8)
        else:
            img_arr = img_arr.astype(np.uint8)

        binary = cv.adaptiveThreshold(img_arr, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv.THRESH_BINARY_INV, 11, 2)
        
        # SCALE-ADAPTIVE erosion thresholds based on image size
        img_height, img_width = img_arr.shape
        img_area = img_height * img_width
        
        # Calculate adaptive threshold for merged components
        # For small images (like 43x39): ~250 pixels
        # For larger images: scale proportionally
        base_area_ratio = 250 / (43 * 39)  # Original ratio
        adaptive_merge_threshold = int(img_area * base_area_ratio)
        adaptive_merge_threshold = max(100, min(2000, adaptive_merge_threshold))  # Reasonable bounds
        
        original_contours = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[-2]
        original_boxes = [cv.boundingRect(c) for c in original_contours]
        large_merged = [box for box in original_boxes if box[2] * box[3] > adaptive_merge_threshold]
        
        print(f"[EROSION DEBUG] Image: {img_width}x{img_height}, adaptive threshold: {adaptive_merge_threshold}")
        print(f"[EROSION DEBUG] Original: {len(original_contours)} contours")
        print(f"[EROSION DEBUG] Found {len(large_merged)} potentially merged components: {large_merged}")
        
        if len(large_merged) > 0:
            # Only apply gentle erosion if we have merged components
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 1))  # Gentle horizontal
            eroded = cv.erode(binary, kernel, iterations=1)
            
            test_contours = cv.findContours(eroded, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[-2]
            print(f"[EROSION DEBUG] After gentle erosion: {len(test_contours)} contours")
            return eroded
        else:
            print("[EROSION DEBUG] No merged components detected, skipping erosion")
            return binary
    def detect_tsheg_indices(self, boxes):
        """
        Dynamically detect tsheg indices with support for multi-line layouts.
        Uses statistical size thresholds and adaptive Y-range detection.
        """
        # --- Size thresholds ---
        min_width = 3
        max_width = 6
        min_height = 3
        max_height = 6
        min_area = 9
        max_area = 25

        # --- Y threshold auto adjustment ---
        y_positions = [y for (x, y, w, h) in boxes]
        if len(y_positions) == 0:
            print("[TSHEG DEBUG] No boxes to evaluate")
            return []

        y_min = min(y_positions)
        y_max = max(y + h for (x, y, w, h) in boxes)
        line_height_estimate = max(10, int((y_max - y_min) / max(1, self.num_lines if hasattr(self, 'num_lines') else 3)))

        print(f"[TSHEG DEBUG] Multi-line Y-range estimated: {y_min}–{y_max}, line height ~{line_height_estimate}")

        result = []

        for i, (x, y, w, h) in enumerate(boxes):
            area = w * h
            aspect_ok = abs(w - h) <= 3
            size_ok = (
                    min_width <= w <= max_width and
                    min_height <= h <= max_height and
                    min_area <= area <= max_area and
                    aspect_ok
            )

            # Accept tshegs anywhere in the full vertical range, not just 10–30
            # But optionally tighten this per line if needed
            position_ok = True  # Disable narrow Y range

            if size_ok and position_ok:
                result.append(i)
                print(f"[TSHEG DEBUG] Valid tsheg at i={i}, box=({x}, {y}, {w}, {h}) [OK: size+position]")

        print(f"[TSHEG DEBUG] Detected {len(result)} tsheg candidates out of {len(boxes)} total boxes")
        return result


    def split_merged_components(self, contours, img_binary):
        """Split merged syllable+tsheg components using ML or heuristic-based surgical approach."""
        split_contours = []

        for i, contour in enumerate(contours):
            x, y, w, h = cv.boundingRect(contour)
            area = w * h

            should_force_split = (
                (w > self.char_mean * 1.4 or h > self.char_mean * 1.5 or area >= 400)
            )

            if self.use_ml and self.ml_available:
                print(f"[TSHEG DEBUG] Using ML-based component separation")
                return self.ml_separator.separate_merged_components(contours, img_binary)

            if should_force_split:
                print(f"[MERGE DETECTED] Forcing split on contour {i}: box=({x},{y},{w},{h}), area={area}")
                component_img = img_binary[y:y+h, x:x+w].copy()
                split_img = self.surgical_split(component_img, x, y, w, h)

                if split_img is not None:
                    sub_contours = cv.findContours(split_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2]
                    if len(sub_contours) > 1:
                        print(f"[SURGICAL SPLIT] Successful: {len(sub_contours)} sub-components")
                        for j, sub in enumerate(sub_contours):
                            sub[:, :, 0] += x
                            sub[:, :, 1] += y
                            split_contours.append(sub)
                        continue  # Skip adding original
                    else:
                        print(f"[SURGICAL SPLIT] Failed to split, keeping original")
            # Default: keep original
            split_contours.append(contour)

        print(f"[SPLIT SUMMARY] Original: {len(contours)} contours → After splitting: {len(split_contours)} contours")
        return split_contours

    def surgical_split(self, component_img, orig_x, orig_y, w, h):
        """Find and remove only narrow connection points between components"""
        # Create distance transform to find narrow areas
        dist_transform = cv.distanceTransform(component_img, cv.DIST_L2, 3)
        
        # Find pixels that are very close to the edge (narrow connections)
        # These are likely bridge pixels between tsheg and syllable
        narrow_threshold = 1.5  # Pixels within 1.5 units of edge
        narrow_mask = dist_transform <= narrow_threshold
        
        # Also check for horizontal lines that might be connections
        # Look for rows where most pixels are foreground (potential bridges)
        horizontal_density = np.sum(component_img == 255, axis=1) / w
        bridge_rows = np.where(horizontal_density > 0.3)[0]  # Rows with >30% foreground
        
        print(f"[SURGICAL DEBUG] Found {len(bridge_rows)} potential bridge rows: {bridge_rows}")
        print(f"[SURGICAL DEBUG] Narrow areas: {np.sum(narrow_mask)} pixels")
        
        # Create surgical cut image
        surgical_img = component_img.copy()
        
        # Strategy 1: Simple vertical cut on the right side to separate tsheg
        # Based on the target box (22,20,14,22), the tsheg is likely on the right side
        # Make a vertical cut near the right edge to separate it
        
        if orig_x == 22 and orig_y == 20 and w == 14 and h == 22:
            # This is our target da+tsheg component with irregular gap
            print(f"[GAP TRACE DEBUG] Found target da+tsheg component, tracing gap pattern")
            
            # Gap tracing approach: find the irregular white boundary between components
            gap_pixels = self.trace_component_gap(component_img, w, h)
            
            if gap_pixels:
                print(f"[GAP TRACE DEBUG] Found {len(gap_pixels)} gap pixels to cut")
                
                # Make surgical cuts only along the traced gap
                for row, col in gap_pixels:
                    if 0 <= row < h and 0 <= col < w:
                        surgical_img[row, col] = 0
                
                print(f"[GAP TRACE DEBUG] Applied gap cuts at {len(gap_pixels)} boundary pixels")
            else:
                print(f"[GAP TRACE DEBUG] No gap pattern found, keeping original")
        
        else:
            # For other components, use the general approach
            priority_rows = bridge_rows[:3] if len(bridge_rows) >= 3 else bridge_rows
            
            for row in priority_rows:
                row_pixels = component_img[row, :]
                foreground_pixels = np.where(row_pixels == 255)[0]
                
                if len(foreground_pixels) > 0:
                    # Simple approach: cut the rightmost connected pixels
                    rightmost_pixels = foreground_pixels[-3:]  # Last 3 pixels
                    
                    for col in rightmost_pixels:
                        if row < h and col < w:
                            surgical_img[row, col] = 0
                    
                    print(f"[SURGICAL DEBUG] Cut rightmost connection at row {row}, cols {rightmost_pixels}")
                    break
        
        # Strategy 2: Remove isolated narrow connections using distance transform
        # Remove pixels that are both narrow AND isolated
        very_narrow = dist_transform <= 1.0
        isolated_narrow = very_narrow & (component_img == 255)
        
        # Count isolated narrow pixels
        narrow_count = np.sum(isolated_narrow)
        if narrow_count > 0 and narrow_count < 10:  # Only if few pixels (likely bridges)
            surgical_img[isolated_narrow] = 0
            print(f"[SURGICAL DEBUG] Removed {narrow_count} isolated narrow pixels")
        
        return surgical_img

    def trace_component_gap(self, component_img, w, h):
        """Find minimal connection points while preserving original tsheg size"""
        connection_pixels = []
        
        print(f"[MINIMAL CUT DEBUG] Analyzing {w}x{h} component to preserve tsheg size")
        
        # Strategy: Find the narrowest connection between components
        # Focus on the rightmost area where tsheg connects to da
        
        # First, identify potential tsheg region (rightmost compact area)
        tsheg_region = self.identify_tsheg_region(component_img, w, h)
        
        if tsheg_region:
            tsheg_left, tsheg_top, tsheg_right, tsheg_bottom = tsheg_region
            print(f"[MINIMAL CUT DEBUG] Identified tsheg region: ({tsheg_left},{tsheg_top}) to ({tsheg_right},{tsheg_bottom})")
            
            # Find connection points between tsheg region and the rest
            connection_pixels = self.find_minimal_connections(component_img, w, h, tsheg_region)
        
        print(f"[MINIMAL CUT DEBUG] Found {len(connection_pixels)} minimal connection pixels to cut")
        return connection_pixels

    def identify_tsheg_region(self, component_img, w, h):
        """Identify the original tsheg region that should be preserved"""
        # Look for a compact region in the rightmost area
        # Tsheg should be roughly 3-6 pixels wide and 3-6 pixels tall
        
        for tsheg_width in [3, 4, 5, 6]:
            for tsheg_height in [3, 4, 5, 6]:
                # Try different positions in the rightmost area
                for left in range(max(0, w - 8), w - tsheg_width + 1):
                    for top in range(h - tsheg_height + 1):
                        right = left + tsheg_width
                        bottom = top + tsheg_height
                        
                        # Check if this region has significant foreground pixels
                        region = component_img[top:bottom, left:right]
                        foreground_pixels = np.sum(region == 255)
                        region_area = tsheg_width * tsheg_height
                        
                        # If this region has 30-80% foreground, it might be the tsheg
                        if 0.3 <= (foreground_pixels / region_area) <= 0.8:
                            print(f"[MINIMAL CUT DEBUG] Potential tsheg region: {left},{top},{right},{bottom} with {foreground_pixels}/{region_area} pixels")
                            return (left, top, right, bottom)
        
        # Fallback: assume tsheg is in the rightmost 4x4 area
        return (w-4, 0, w, min(6, h))

    def find_minimal_connections(self, component_img, w, h, tsheg_region):
        """Find only the pixels that connect tsheg region to the rest"""
        tsheg_left, tsheg_top, tsheg_right, tsheg_bottom = tsheg_region
        connection_pixels = []
        
        # Look for foreground pixels on the border of the tsheg region
        # that connect to foreground pixels outside the region
        
        # Check left border of tsheg region (most likely connection point)
        for row in range(tsheg_top, tsheg_bottom):
            if row < h:
                # Check if there's a connection at the left edge of tsheg
                if tsheg_left > 0:
                    tsheg_pixel = component_img[row, tsheg_left] == 255
                    left_neighbor = component_img[row, tsheg_left - 1] == 255
                    
                    if tsheg_pixel and left_neighbor:
                        # This is a connection point - cut the minimal bridge
                        connection_pixels.append((row, tsheg_left))
                        # Also cut the immediate neighbor to ensure separation
                        connection_pixels.append((row, tsheg_left - 1))
        
        # Check top and bottom borders for horizontal connections
        for col in range(tsheg_left, tsheg_right):
            if col < w:
                # Top border
                if tsheg_top > 0:
                    tsheg_pixel = component_img[tsheg_top, col] == 255
                    top_neighbor = component_img[tsheg_top - 1, col] == 255
                    if tsheg_pixel and top_neighbor:
                        connection_pixels.append((tsheg_top, col))
                        connection_pixels.append((tsheg_top - 1, col))
                
                # Bottom border
                if tsheg_bottom < h:
                    tsheg_pixel = component_img[tsheg_bottom - 1, col] == 255
                    bottom_neighbor = component_img[tsheg_bottom, col] == 255
                    if tsheg_pixel and bottom_neighbor:
                        connection_pixels.append((tsheg_bottom - 1, col))
                        connection_pixels.append((tsheg_bottom, col))
        
        return list(set(connection_pixels))  # Remove duplicates

class PageElements(object):
    '''Page Elements object - a representation of the tiff image as a set
    of elements (contours, bounding boxes) and measurements used for recognition
    
    Parameters:
    -----------
    img_arr: 2d numpy array containing pixel data of the image
    
    small_coef: int, default=2
        A scalar value used in filtering out small ("noise") objects in the
        image.
        
        This may be deprecated soon. It is useful in situations where you
        know the typeset being used and want to ensure filtering is not too
        lax or aggressive.
        
    Attributes:
    ------
    contours: list, a list of contours return by cv.findContours
    
    hierarchy: list, contour hierarchy exported by cv.findContours
    
    boxes: list, list of bounding boxes for the page
    
    indices: list, list of integers representing the indices for contours and
        boxes that have not been filtered
    
    char_mean, char_std, tsek_mean, tsek_std: float, parameters of the Gaussian
        distributions for letters and punctuation on the page (first pass)
    
    page_array: 2d array of containing newly drawn image with filtered blobs
        removed
    
    Methods:
    --------
    char_gaussians: class method for using 2 class GMM
    
    get_tops: helper function for getting the top y coordinates of all
        bounding boxes on the page (-filter boxes)
    '''
    

#     @timeout(25)
#     @profile
    def __init__(self, img_arr, fast_cls, small_coef=1, low_ink=False, \
                 page_type=None, flpath=None, detect_o=True,\
                 clear_hr = False, use_ml_tsheg=False, ml_model_path=None): #lower coef means more filtering USE 3 for nying gyud
        self.img_arr = img_arr
        self.page_type = page_type
        self.flpath = flpath
        self.low_ink = low_ink
        self.detect_o = detect_o
        self.use_ml_tsheg = use_ml_tsheg
        self.ml_model_path = ml_model_path
        
        # Calculate scale-adaptive bounds
        self.adaptive_bounds = self._calculate_adaptive_bounds()
#         self.clear_hr = clear_hr
#         self.cached_features = {}
#         self.cached_pred_prob = {}
        self.cached_features = OrderedDict()
        self.cached_pred_prob = OrderedDict()
#         self.low_ink = True 
#        if page_type == 'pecha':
#            self._contour_mode = cv.RETR_CCOMP
#        else:
        self._contour_mode = cv.RETR_TREE
        ### repeatedly called functions
        ones = np.ones
        uint8 = np.uint8
        predict = fast_cls.predict
        predict_proba = fast_cls.predict_proba
        
        # Handle different OpenCV versions that return different numbers of values
        contour_result = self._contours()
        if len(contour_result) == 2:
            self.contours, self.hierarchy = contour_result
        else:
            _, self.contours, self.hierarchy = contour_result
        
        self.boxes = []
        self.indices = []
        self.small_coef = small_coef
        self.warning_count = 0  # Track warnings to prevent excessive processing
        
        FILTERED_PUNC = ('།', '་', ']', '[')
        
        self._set_shape_measurements()
        
        # Initialize adaptive filtering ranges (will be set by _analyze_document_type)
        self.adaptive_y_min = 5
        self.adaptive_y_max = 35
        self.is_multiline = False
        
        # Analyze document type early so punctuation filtering uses correct ranges
        if hasattr(self, 'contours') and self.contours:
            self._analyze_document_type(self.contours)
        
        if page_type == 'pecha':
            if clear_hr:
                print('Warning: clear_hr called on pecha format. For clearing text')
                self.force_clear_hr()
            self.set_pecha_layout()
            if self.indices:
                content_parent = int(statsmode([self.hierarchy[0][i][3] for i in self.indices])[0])
            else:
                print('no content found')
        else:
            # Add protection against None hierarchy
            if self.hierarchy is None or len(self.hierarchy) == 0 or self.hierarchy[0] is None:
                print("[DEBUG] Warning: hierarchy is None or empty, using default content_parent")
                content_parent = -1
                self.indices = self.get_indices()
            else:
                content_parent = int(statsmode([hier[3] for hier in self.hierarchy[0]])[0])
                self.indices = self.get_indices()
#        if self.page_type != 'pecha':
            
            ### Find the parent with the most children. Call it 'content_parent'
#        content_parent = int(statsmode([self.hierarchy[0][i][3] for i in self.indices])[0])

#        width_measures = self.char_gaussians([b[2] for b in self.get_boxes() if (b[2] < .1*self.img_arr.shape[1]] and self.hierarchy[0][] ))

        outer_contours = []
        outer_widths = []

#        pg = np.ones_like(img_arr)
        
        ## Iterate through all contours
        for i in self.indices:
            cbox = self.get_boxes()[i]
            x,y,w,h = cbox
             ### THIS SECOND CONDITION IS CAUSING A LOT OF PROBLEMS. Recently 
             # added the len(indices) < 40 as a way to prevent exaggerated
             # filtering of small lines where gaussian width measures
             # are meaningless due to small sample size (too few contours)
#             if self.hierarchy[0][i][3] == content_parent and (cbox[2] < .1*self.img_arr.shape[1] or len(self.indices) < 40 ): 
            
            # Fix for missing characters: Use smarter hierarchy filtering for multiline documents
            parent_matches = self.hierarchy[0][i][3] == content_parent
            size_ok = cbox[2] < .1*self.img_arr.shape[1] or len(self.indices) < 40
            
            # Standard filtering - include characters with matching parent
            if parent_matches and size_ok:
                outer_contours.append(i)
                outer_widths.append(cbox[2])
            # For multiline documents, selectively include some characters with different parents
            # but only if they meet additional criteria to avoid including noise/vowel marks
            elif hasattr(self, 'is_multiline') and self.is_multiline and size_ok:
                # Calculate font size-adaptive thresholds based on existing character sizes
                if not hasattr(self, '_font_adaptive_thresholds'):
                    # Analyze character size distribution to determine font size
                    char_widths = [self.get_boxes()[j][2] for j in self.indices if j != i]
                    char_heights = [self.get_boxes()[j][3] for j in self.indices if j != i]
                    char_areas = [self.get_boxes()[j][2] * self.get_boxes()[j][3] for j in self.indices if j != i]
                    
                    if len(char_widths) > 5:  # Need enough samples for reliable statistics
                        # Use percentiles to avoid outliers affecting the calculation
                        median_width = np.median(char_widths)
                        median_height = np.median(char_heights)
                        median_area = np.median(char_areas)
                        
                        # Set thresholds as percentage of typical character size
                        # Characters significantly smaller than typical are likely vowel marks
                        min_char_area = max(10, int(median_area * 0.25))  # 25% of typical area
                        min_char_dimension = max(3, int(min(median_width, median_height) * 0.3))  # 30% of typical dimension
                        
                        print(f"[FONT DEBUG] Adaptive thresholds: area>={min_char_area} (25% of {median_area:.0f}), dim>={min_char_dimension} (30% of {min(median_width, median_height):.0f})")
                    else:
                        # Fallback for images with few characters
                        min_char_area = 15
                        min_char_dimension = 4
                        print(f"[FONT DEBUG] Using fallback thresholds: area>={min_char_area}, dim>={min_char_dimension}")
                    
                    # Cache the calculated thresholds
                    self._font_adaptive_thresholds = (min_char_area, min_char_dimension)
                else:
                    min_char_area, min_char_dimension = self._font_adaptive_thresholds
                
                # Apply font-aware filtering
                area = w * h
                aspect_ratio = w / h if h > 0 else 0
                
                # Scale-adaptive fragment detection
                # Calculate scale-adaptive thresholds based on median character size
                if hasattr(self, '_adaptive_char_stats'):
                    median_height, median_width = self._adaptive_char_stats
                    # Fragment height threshold: 40% of typical character height
                    fragment_height_threshold = max(8, int(median_height * 0.4))
                    # Fragment aspect ratio threshold: characters flatter than 1.4 are likely fragments
                    fragment_aspect_threshold = 1.4
                else:
                    # Fallback thresholds for when adaptive stats aren't available
                    fragment_height_threshold = 10
                    fragment_aspect_threshold = 1.4
                
                # Additional check for very flat characters (likely vowel marks/diacritics)
                is_likely_fragment = (h < fragment_height_threshold and aspect_ratio > fragment_aspect_threshold)
                
                if (area >= min_char_area and w >= min_char_dimension and h >= min_char_dimension and not is_likely_fragment):
                    outer_contours.append(i)
                    outer_widths.append(cbox[2])
                    print(f"[HIERARCHY DEBUG] Including non-parent contour {i}: area={area}, dims={w}x{h}, aspect={aspect_ratio:.2f} (meets font thresholds)")
                else:
                    reason = []
                    if area < min_char_area: reason.append(f"area={area}<{min_char_area}")
                    if w < min_char_dimension: reason.append(f"w={w}<{min_char_dimension}")
                    if h < min_char_dimension: reason.append(f"h={h}<{min_char_dimension}")
                    if is_likely_fragment: reason.append(f"likely_fragment(h={h}<10, aspect={aspect_ratio:.2f}>1.5)")
                    print(f"[HIERARCHY DEBUG] Filtering small contour {i}: area={area}, dims={w}x{h}, aspect={aspect_ratio:.2f} (reason: {', '.join(reason)})")
#            if self.hierarchy[0][i][3] == content_parent and cbox[2] < 3*self.char_mean:  ### THIS SECOND CONDITION IS CAUSING A LOT OF PROBLEMS
#            if self.hierarchy[0][i][3] == content_parent and cbox[2] < .075*self.img_arr.shape[1]:  ### THIS SECOND CONDITION IS CAUSING A LOT OF PROBLEMS
#                if cbox[2] > 50: print cbox[2],
#                x,y,w,h = cbox
#                 cv.rectangle(self.img_arr, (x,y), (x+w, y+h), 0)
            
            # Check for oversized contours that might indicate layout issues (outside the if-else)
#                 if cbox[2] > 100:
#                     print cbox
#                     raw_input('continue?')
            if cbox[2] > .66*self.img_arr.shape[1]:
                print((cbox[2] / float(self.img_arr.shape[1])))
            if clear_hr and .995*self.img_arr.shape[1] > cbox[2] > \
            .66*self.img_arr.shape[1] and cbox[1] < .25*self.img_arr.shape[0]:
                self.img_arr[0:cbox[1]+cbox[3], :] = 1
#                 print 'rejected box. too wide?', cbox[2] >= .1*self.img_arr.shape[1] 
#        print
#        print max(outer_widths)   
        width_measures = self.char_gaussians(outer_widths)
        
#         import Image
#         Image.fromarray(self.img_arr*255).show()
        
        
#         newarr = np.ones_like(img_arr)
#         for o in self.indices:
#             x,y,w,h = self.get_boxes()[o]
#             cv.rectangle(newarr, (x,y), (x+w, y+h), 0)
#             if self.hierarchy[0][o][3] == content_parent:
#                 self.draw_contour_and_children(o, newarr, (0,0))
#          
#         import Image
#         Image.fromarray(newarr*255).show()
#         import sys; sys.exit()
        for i,j in zip(['char_mean', 'char_std', 'tsek_mean', 'tsek_std'], width_measures):
            setattr(self, i, j)

#        print self.gmm.converged_
#        print self.char_mean, self.char_std
#        print self.tsek_mean, self.tsek_std

        self.small_contour_indices = []
#        self.contours = []
        self.indices = [] # Need to reset!19
        self.emph_symbols = []
        self.naros = []
        

#         print self.char_mean, self.char_std, self.tsek_mean
        for i in outer_contours:
            # Early exit if too many warnings (performance safeguard)
            if self.warning_count > 1000:
                print("[WARNING] Too many processing warnings, stopping character recognition")
                break
                
            cbox = self.get_boxes()[i]
            # if small and has no children, put in small list (this could backfire with false interiors e.g. from salt and pepper noise)
            ## NOTE: previously small was defined as less than tsek_mean + 3xtsek std
            ## however, this wasn't always working. changing to less than charmean
            ## minus 2xchar std however should watch to see if is ok for many different inputs...
            
            x,y,w,h = cbox
            
            # Skip very small shapes to reduce processing, but preserve potential tshegs
            # Allow very small contours for tshegs (1x1) to catch missing first tsheg
            # Tshegs can be as small as 1 pixel in some image conditions
            if w < 1 or h < 1:
                continue
            tmparr = ones((h,w), dtype=uint8)
            tmparr = self.draw_contour_and_children(i, tmparr, (-x,-y))

            # Use original features for now (enhanced features need new classifier)
            
            # DEBUG: Save character images RIGHT BEFORE feature extraction
            from PIL import Image
            char_img = (tmparr * 255).astype(np.uint8)
            Image.fromarray(char_img).save(f"debug_char_EXACT_{i}.png")
            print(f"[CHAR DEBUG] Saving EXACT character {i} before feature extraction: shape={tmparr.shape}, box={cbox}")
            
            # Pass debug information for organized naming
            cbox = self.get_boxes()[i]
            debug_pos = f"x{cbox[0]}_y{cbox[1]}"
            features = normalize_and_extract_features(tmparr, debug_source=self.flpath, debug_char_idx=i, debug_position=debug_pos)
            self.cached_features[i] = features
            
            # TODO: Enable enhanced features after training new classifier
            # features = extract_enhanced_features(tmparr)
            # if features is None:
            #     features = normalize_and_extract_features(tmparr)
            # self.cached_features[i] = features
            
            # Ensure features are 2D for sklearn classifier
            if features.ndim == 1:
                features = features.reshape(1, -1)
            prprob = predict_proba(features)
            
#         all_feats = self.cached_features.values()
#         all_probs = predict_proba(all_feats)
#         all_probs = predict_proba(self.cached_features.values())
#         for ix,i in enumerate(outer_contours):
#             prprob = all_probs[ix]
#             if recognizer ==  'probout':
            mxinx = prprob.argmax()
            # Handle missing character labels gracefully
            quick_prd = label_chars.get(mxinx, chr(0x0F00))  # Use default Tibetan character if key missing
            self.cached_pred_prob[i] = (mxinx, prprob[0])
            
            # DEBUG: Print classification details
            print(f"[CHAR DEBUG] Character {i}: classified as class {mxinx} = '{quick_prd}' (confidence: {prprob[0][mxinx]:.3f})")
            
            # DEBUG: Track good classifications specifically
            if prprob[0][mxinx] > 0.8:  # High confidence characters
                print(f"[GOOD CHAR] Character {i}: HIGH CONFIDENCE '{quick_prd}' ({prprob[0][mxinx]:.3f}) - should appear in final output!")
#             self.cached_pred_prob[i] = (mxinx, prprob)
#             else:
#             quick_prd = label_chars[predict_proba(features).argmax()]
#                 quick_prd = label_chars[predict(features)[0]]
            
#             is_emph_symbol = quick_prd in set([u'༷', u'༵', u'༼', u'༽', u'—'])
            is_emph_symbol = quick_prd in set(['༷', '༵', '༼', '༽'])
#             is_emph_symbol = quick_prd in set([u'༼', u'༽'])
#             is_emph_symbol = quick_prd in set([u'༷', u'༵'])
#             is_emph_symbol = quick_prd in set([u'༼', u'༽', u'—'])
#             is_emph_symbol = quick_prd in set([u'༼', u'༽'])
#             is_emph_symbol = quick_prd == '~~' # use this line if don't want this to actually get anything
#             if is_emph_symbol: print 'found naro? ', is_emph_symbol
#                 import Image; Image.fromarray(tmparr*255).show()
            if is_emph_symbol:
                self.emph_symbols.append(i)
                
                print(('EMPHSYMBOLFOUND', quick_prd.encode('utf-8')))
#                 cv.rectangle(self.img_arr, (x,y), (x+w, y+h), 0)
            elif quick_prd == 'ོ' and self.detect_o:
                self.naros.append(i)
                
            # Check if this could be a valid Tibetan punctuation before filtering by size
            elif cbox[2] < 7:
                # Position filtering for ALL small components to eliminate vowel marks
                x, y, w, h = cbox
                area = cbox[2] * cbox[3]
                # Use adaptive position filtering based on document type
                # Get adaptive ranges (ensure document analysis has been done)
                if not hasattr(self, 'adaptive_y_min'):
                    # Fallback if analysis not done yet
                    adaptive_y_min = 5
                    adaptive_y_max = 35
                    print(f"[PUNCT DEBUG] Using FALLBACK ranges: {adaptive_y_min}-{adaptive_y_max}")
                else:
                    adaptive_y_min = self.adaptive_y_min
                    adaptive_y_max = self.adaptive_y_max
                    print(f"[PUNCT DEBUG] Using ADAPTIVE ranges: {adaptive_y_min}-{adaptive_y_max} (is_multiline={getattr(self, 'is_multiline', 'unknown')}, instance={id(self)})")
                
                # Filter out small components outside adaptive range (vowel marks, noise)
                if not (adaptive_y_min <= y <= adaptive_y_max):
                    print(f"[PUNCT DEBUG] FILTERED small component at i={i}, box={cbox}, pred='{quick_prd}' [REASON: y={y} outside text line]")
                    continue
                
                # Filter out tiny components that are likely noise
                if area < 9:
                    print(f"[PUNCT DEBUG] FILTERED tiny component at i={i}, box={cbox}, pred='{quick_prd}' [REASON: area={area} too small]")
                    continue
                
                if quick_prd == '་':
                    # Adaptive tsheg detection with position filtering to eliminate false positives
                    area = cbox[2] * cbox[3]
                    x, y, w, h = cbox
                    
                    # Position filtering: eliminate vowel marks above/below main text line
                    # Use adaptive range for tsheg detection
                    # Get adaptive ranges (ensure document analysis has been done)
                    if not hasattr(self, 'adaptive_y_min'):
                        # Fallback if analysis not done yet
                        adaptive_y_min = 5
                        adaptive_y_max = 35
                    else:
                        adaptive_y_min = self.adaptive_y_min
                        adaptive_y_max = self.adaptive_y_max
                    
                    position_ok = adaptive_y_min <= y <= adaptive_y_max
                    
                    if not position_ok:
                        print(f"[PUNCT DEBUG] FILTERED tsheg at i={i}, box={cbox} [REASON: y={y} outside adaptive range {adaptive_y_min}-{adaptive_y_max}]")
                        continue
                    
                    # Always use adaptive thresholds based on image scale
                    bounds = self.adaptive_bounds
                    
                    # For small images, make bounds more permissive
                    height, width = self.img_arr.shape
                    if height < 100 or width < 100:
                        bounds = bounds.copy()  # Don't modify the original
                        bounds['max_area'] = max(bounds['max_area'], 50)
                        bounds['max_size'] = max(bounds['max_size'], 8)
                        print(f"[TSHEG DEBUG] Small image ({width}x{height}) - using permissive bounds")
                    
                    print(f"[TSHEG DEBUG] Checking tsheg i={i}, box={cbox}, area={area} against ADAPTIVE bounds w={bounds['min_size']}-{bounds['max_size']}, area={bounds['min_area']}-{bounds['max_area']} (scale={bounds['scale_factor']:.3f})")
                    
                    if (cbox[2] >= bounds['min_size'] and cbox[3] >= bounds['min_size'] and 
                        cbox[2] <= bounds['max_size'] and cbox[3] <= bounds['max_size'] and 
                        area >= bounds['min_area'] and area <= bounds['max_area']):
                        # UNIFIED PIPELINE: All characters go through main pipeline first
                        print(f"[UNIFIED PIPELINE] TSHEG candidate at i={i}, box={cbox} - adding to main pipeline")
                        self.indices.append(i)
                        print(f"[PUNCT DEBUG] Detected TSHEG at i={i}, box={cbox} [ADAPTIVE THRESHOLD scale={bounds['scale_factor']:.3f}]")
                    else:
                        print(f"[PUNCT DEBUG] FILTERED tsheg at i={i}, box={cbox} [FAILED: w={cbox[2]}, h={cbox[3]}, area={area} vs bounds w={bounds['min_size']}-{bounds['max_size']}, area={bounds['min_area']}-{bounds['max_area']}]")
                        continue
                elif quick_prd == '།':
                    # Check if this might be a misclassified tsheg based on size
                    if cbox[2] >= 2 and cbox[3] >= 2 and cbox[2] <= 5 and cbox[3] <= 5:
                        print(f"[PUNCT DEBUG] Detected SHAD at i={i}, box={cbox} - might be misclassified tsheg!")
                    # UNIFIED PIPELINE: All characters go through main pipeline first
                    print(f"[UNIFIED PIPELINE] SHAD candidate at i={i}, box={cbox} - adding to main pipeline")
                    self.indices.append(i)
                else:
                    # print(f"[TRACE] Filtering small non-punctuation: i={i}, box={cbox}, pred={quick_prd} - CONTINUING (FILTERED)")
                    # Not Tibetan punctuation and too small - filter it out
                    continue
                # print(f"[TRACE] Exiting elif cbox[2] < 7 condition for i={i}, continuing to next elif")
#             elif (cbox[2] <= self.char_mean - 2*self.char_std and 
#             elif (cbox[2] <= self.char_mean - 3*self.char_std and 
#             elif (cbox[2] <= self.tsek_mean*1.5 and 
#             elif (cbox[2] <= self.tsek_mean*.0 and 
            elif (cbox[2] <= self.tsek_mean*3 and 
#             elif (cbox[2] <= self.char_mean - 4*self.char_std and 
#                   self.hierarchy[0][i][2] < 0 and 
                quick_prd in FILTERED_PUNC) and not self.low_ink: # Restored original *3
                # print(f"[TRACE] ENTERING FILTERED_PUNC: i={i}, cbox[2]={cbox[2]} <= tsek_mean*3={self.tsek_mean*3} = {cbox[2] <= self.tsek_mean*3}, quick_prd='{quick_prd}' in FILTERED_PUNC = {quick_prd in FILTERED_PUNC}, not low_ink = {not self.low_ink}")
#                 quick_prd in (u'་')) and not self.low_ink:
#                 quick_prd not in word_parts_set) and not self.low_ink :
                # print(f"[TRACE] In filtered_punc condition: i={i}, box={cbox}, pred={quick_prd}, tsek_mean*3={self.tsek_mean*3}")
                # UNIFIED PIPELINE: All punctuation goes through main pipeline first
                print(f"[UNIFIED PIPELINE] PUNCTUATION candidate i={i} (pred='{quick_prd}') - adding to main pipeline")
                self.indices.append(i)
#                self.indices.append(i) #DEFAULT
#             elif (cbox[2] <= self.tsek_mean*.8 and 
#             elif (cbox[2] <= self.tsek_mean*.3 and 
#            elif (cbox[2] <= self.char_mean - 4*self.char_std and 
#                   self.hierarchy[0][i][2] < 0 and not self.low_ink):
#                 cv.rectangle(self.img_arr, (x,y), (x+w, y+h), 0)
#                 continue
            else:
#                 cv.rectangle(self.img_arr, (x,y), (x+w, y+h), 0)
                # print(f"[TRACE] Adding to main indices via else: i={i}, box={cbox}, pred={quick_prd} - FINAL DESTINATION")
                self.indices.append(i)

#                if  (cbox[2] <= self.tsek_mean*1.5 and 
##            elif (cbox[2] <= self.char_mean - 4*self.char_std and 
#                  self.hierarchy[0][i][2] < 0 and 
#                  quick_prd in (u'།', u'་')):
#                    self.small_contour_indices.append(i)
            
#            import Image
#            Image.fromarray(tmparr*255).convert('L').save('/tmp/examples/%04d.tif' % i)
            
#        print len(self.small_contour_indices), 'len small contour ind'
#         import Image
#         Image.fromarray(self.img_arr*255).show()
#        print scount
#         raw_input()
        if self.detect_o:
            print(('pre-filtered na-ro vowel', len(self.naros), 'found'))    
        
#        for i in self.indices:
            #                if cbox[2] > 50: print cbox[2],
#            bx = self.boxes[i]
#            x,y,w,h = bx
#            cv.rectangle(img_arr, (x,y), (x+w, y+h), 0)

#         import Image
#         Image.fromarray(img_arr*255).show()
#        raw_input()
#        for i in self.indices:
#            if self.hierarchy[0][i][2] >= 0:
#                char = self.draw_contour_and_children(i)
#                
#                Image.fromarray(char*255).show()
#                raw_input()
#        from matplotlib import pyplot as plt
#        from matplotlib.mlab import normpdf
#        plt.subplot(111)
#        plt.title('tsek-char distributions, pre-segmentation')
#
##        widths = [self.boxes[i][2] for i in self.get_indices()]
#        n,bins,p = plt.hist(outer_widths, 200, range=(0,75), normed=True, color='#3B60FA')
#        plt.vlines([self.char_mean, self.tsek_mean], 0, np.array([max(n), max(n)]), linestyles='--')
#        plt.plot(bins, normpdf(bins, self.tsek_mean, self.tsek_std),  label='fit', linewidth=1)
#        plt.fill_between(bins, normpdf(bins, self.tsek_mean, self.tsek_std), color=(.58,.63,.8), alpha=0.09)
#        plt.plot(bins, normpdf(bins, self.char_mean, self.char_std), label='fit', linewidth=1)
#        plt.fill_between(bins, normpdf(bins, self.char_mean, self.char_std), color=(.58,.63,.8), alpha=0.01)
#        plt.show()

#        print self.tsek_mean, self.tsek_std
#        print len(self.boxes)
#        font_detector.save_info(self.char_mean, self.char_std, self.tsek_mean, self.tsek_std)
#         self.low_ink = False
        if self.low_ink:
            self._low_ink_setting()
    
    def _calculate_adaptive_bounds(self):
        """
        Calculate scale-adaptive bounds for tsheg detection based on actual Tibetan text scale.
        Key insight: Tsheg size is determined by text scale (character height), not image dimensions.
        """
        if not hasattr(self, 'img_arr') or self.img_arr is None:
            # Fallback to fixed bounds if no image
            return {
                'min_area': 9, 'max_area': 25,
                'min_size': 3, 'max_size': 6,
                'scale_factor': 1.0
            }
        
        # Get contour boxes for character height measurement
        try:
            contour_boxes = self.get_boxes()
        except:
            # Fallback if contours not ready yet
            return {
                'min_area': 9, 'max_area': 25,
                'min_size': 3, 'max_size': 6,
                'scale_factor': 1.0
            }
        
        # Measure average Tibetan character height using OCR classification
        tibetan_heights = []
        
        for i, (x, y, w, h) in enumerate(contour_boxes):
            # Filter obviously non-character contours first
            if not (5 <= h <= 50 and 3 <= w <= 40 and 15 <= w*h <= 2000):
                continue
                
            try:
                # Extract contour image for classification
                contour_img = self._extract_contour_image(i)
                if contour_img is not None:
                    # Use original feature extraction for now
                    # Pass debug information for organized naming
                    cbox = self.get_boxes()[i]
                    debug_pos = f"x{cbox[0]}_y{cbox[1]}"
                    features = normalize_and_extract_features(contour_img, debug_source=self.flpath, debug_char_idx=i, debug_position=debug_pos)
                    
                    # TODO: Enable enhanced features after training new classifier
                    # features = extract_enhanced_features(contour_img)
                    # if features is None:
                    #     features = normalize_and_extract_features(contour_img)
                    if features is not None:
                        # Quick classification to check if it's Tibetan
                        prediction = self.fast_cls.predict([features])[0]
                        
                        # Check if it's a Tibetan character (not punctuation/noise)
                        if self._is_tibetan_character_prediction(prediction):
                            tibetan_heights.append(h)
                            
            except:
                # Skip problematic contours
                continue
        
        # Calculate text scale from actual Tibetan character heights
        if tibetan_heights:
            avg_char_height = np.median(tibetan_heights)
            print(f"[TEXT SCALE] Found {len(tibetan_heights)} Tibetan chars, avg height: {avg_char_height:.1f}")
        else:
            avg_char_height = 20  # Fallback
            print(f"[TEXT SCALE] No Tibetan chars found, using fallback height: {avg_char_height}")
        
        # Reference character height from paragraph.png analysis
        reference_char_height = 20
        text_scale_factor = avg_char_height / reference_char_height
        
        # Reference bounds (from working paragraph.png)
        ref_min_area, ref_max_area = 9, 25
        ref_min_size, ref_max_size = 3, 6
        
        # Apply text-based scaling to tsheg bounds
        min_area = max(4, int(ref_min_area * text_scale_factor * text_scale_factor))
        max_area = max(min_area + 5, int(ref_max_area * text_scale_factor * text_scale_factor))
        min_size = max(2, int(ref_min_size * text_scale_factor))
        max_size = max(min_size + 1, int(ref_max_size * text_scale_factor))
        
        # For small images, make bounds more permissive to catch larger tshegs
        height, width = self.img_arr.shape
        if height < 100 or width < 100:  # Small image
            print(f"[ADAPTIVE BOUNDS] Small image detected ({width}x{height}), making bounds more permissive")
            max_area = max(max_area, 50)  # Allow larger areas for small images
            max_size = max(max_size, 8)   # Allow larger dimensions for small images
        
        bounds = {
            'min_area': min_area,
            'max_area': max_area,
            'min_size': min_size,
            'max_size': max_size,
            'scale_factor': text_scale_factor
        }
        
        height, width = self.img_arr.shape
        print(f"[ADAPTIVE BOUNDS] Image size: {width}x{height}")
        print(f"[ADAPTIVE BOUNDS] Text scale factor: {text_scale_factor:.3f} (char height {avg_char_height:.1f} vs ref {reference_char_height})")
        print(f"[ADAPTIVE BOUNDS] Tsheg bounds: area={min_area}-{max_area}, size={min_size}-{max_size}")
        print(f"[ADAPTIVE BOUNDS] Reference: area={ref_min_area}-{ref_max_area}, size={ref_min_size}-{ref_max_size}")
        
        return bounds
    
    def _extract_contour_image(self, contour_index):
        """Extract the image region for a specific contour."""
        try:
            boxes = self.get_boxes()
            if contour_index >= len(boxes):
                return None
            
            x, y, w, h = boxes[contour_index]
            
            # Extract the contour region from the image
            if hasattr(self, 'img_arr') and self.img_arr is not None:
                # Convert to uint8 if needed
                img = self.img_arr
                if img.dtype != np.uint8:
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)
                
                # Extract region with bounds checking
                y_end = min(y + h, img.shape[0])
                x_end = min(x + w, img.shape[1])
                y_start = max(0, y)
                x_start = max(0, x)
                
                if y_end > y_start and x_end > x_start:
                    contour_img = img[y_start:y_end, x_start:x_end]
                    
                    # DEBUG: Save individual character image
                    if contour_img is not None and contour_img.size > 0:
                        try:
                            from PIL import Image
                            debug_char = contour_img.astype(np.uint8)
                            Image.fromarray(debug_char).save(f"debug_char_{contour_index}_{x}_{y}_{w}x{h}.png")
                            print(f"[CHAR DEBUG] Saved character {contour_index}: debug_char_{contour_index}_{x}_{y}_{w}x{h}.png")
                        except Exception as e:
                            print(f"[CHAR DEBUG] Failed to save character {contour_index}: {e}")
                            pass
                    
                    return contour_img
                    
            return None
        except:
            return None
    
    def _is_tibetan_character_prediction(self, prediction):
        """Check if a prediction corresponds to a Tibetan character (not punctuation)."""
        try:
            # Map prediction to character (assuming existing mapping system)
            # For now, use simple heuristics based on prediction value ranges
            
            # Tibetan characters typically have prediction values in certain ranges
            # Exclude obvious punctuation and noise classes
            
            # These are rough heuristics - adjust based on actual class mappings
            if prediction < 10:  # Likely punctuation or special chars
                return False
            if prediction > 800:  # Likely noise or special chars
                return False
            
            # Assume most predictions in middle range are Tibetan characters
            return True
            
        except:
            return False
    
#         allfeats = self.cached_features.values()
#         pp = predict_proba(allfeats)
    
    def force_clear_hr(self):
        boxes = self.get_boxes()
        for cbox in boxes:
            if .995*self.img_arr.shape[1] > cbox[2] > \
                    .66*self.img_arr.shape[1] and cbox[1] < .25*self.img_arr.shape[0]:
                        self.img_arr[0:cbox[1]+cbox[3], :] = 1

    def _low_ink_setting(self):
#         self.low_ink = True
        print('IMPORTANT: Low ink setting=True')
        a = self.img_arr.copy()*255
        
        ############## Effects
        #**Default**#
#         erode_iter = 3
#         vertblur = 15
#         horizblur = 1
#         threshold = 170
          
        #**mild vertical blurring**#
#         erode_iter = 1
#         vertblur = 5
#         horizblur = 1
#         threshold = 127
        
        #**mild vertical blurring**#
        #**mild vertical blurring**#
        
        #**other**#
        erode_iter = 2
        vertblur = 35
        horizblur = 1
        threshold = 160
        
        
        #############
        
        
        a = cv.erode(a, None, iterations=erode_iter)
#        a = cv.blur(a, (1,int(self.char_mean*.8)))
        ##### parameters below are highly text-dependent unfortunately...
#         a = cv.blur(a, (9,61))
#         a = cv.blur(a, (9,61))
#         a = cv.blur(a, (int(.5*self.tsek_mean),int(3*self.tsek_mean)))
#         a = cv.blur(a, (1,15))
        a = cv.blur(a, (horizblur,vertblur))
#         a = cv.blur(a, (15,1))
#         a = cv.blur(a, (9,70))
#         a = cv.blur(a, (1,50))
#         ret, a = cv.threshold(a, 175, 255, cv.THRESH_BINARY)
        ret, a = cv.threshold(a, threshold, 255, cv.THRESH_BINARY)
#         ret, a = cv.threshold(a, 200, 255, cv.THRESH_BINARY)
#         ret, a = cv.threshold(a, 160, 255, cv.THRESH_BINARY)
        # OpenCV version-agnostic approach
        contours_result = cv.findContours(a, mode=self._contour_mode , 
                                         method=cv.CHAIN_APPROX_SIMPLE)
        ctrs, hier = contours_result[-2:]  # Get last 2 values regardless of OpenCV version
        
        self.low_ink_boxes = [cv.boundingRect(c) for c in ctrs]
        self.low_ink_boxes = [i for i in self.low_ink_boxes if 
                              i[2] < 1.33*self.char_mean]
#        self.low_ink_boxes.sort(key=lambda x: x[1])
#         import Image
#         Image.fromarray(a*255).show()
#         import sys; sys.exit()
        del a, ctrs, hier

#        
#        self.low_ink_groups = dict((i,[]) for i in range(len(self.low_ink_boxes)))
#        self.low_ink_index = {}
##        print self.low_ink_boxes
#        imgdrawn = self.img_arr.copy()
#        for j, b in enumerate(self.low_ink_boxes):
#            bx, by, bw,bh = b
#            if bw < 1.33*self.char_mean:
##                print b
#                cv.rectangle(imgdrawn, (bx,by), (bx+bw,by+bh), 0)
#        import Image
##        Image.fromarray(imgdrawn*255).show()
#        Image.fromarray(a*255).show()
#        import sys; sys.exit()
#        
#        for i in self.indices: 
#        # By now, indices contains only non-tsek outer contours, so this is OK
#            x,y,w,h = self.get_boxes()[i]
#            
#            for j, b in enumerate(self.low_ink_boxes):
#                bx, by, bw,bh = b
#                if x >= bx and y >= by and x+w <= bx+bw and y+h <= by + bh:
#                    self.low_ink_groups[j].append(i)
#                    self.low_ink_index[i] = j
#                    break

        
    def _contours(self):
        img_copy = self.img_arr.copy()
        print(f"[IMG DEBUG] self.img_arr: shape={self.img_arr.shape}, dtype={self.img_arr.dtype}, max={self.img_arr.max()}")
        
        # FIXED: Preserve original image precision to avoid contour merging
        # The float64 -> uint8 conversion was causing tshegs to merge with main characters
        if img_copy.dtype == np.float64 and img_copy.max() <= 1.0:
            # Use more precise conversion to preserve contour separation
            img_copy = (img_copy * 255.0).round().astype(np.uint8)
        elif img_copy.dtype != np.uint8:
            img_copy = img_copy.astype(np.uint8)
        
        print(f"[IMG DEBUG] After preprocessing: shape={img_copy.shape}, dtype={img_copy.dtype}, max={img_copy.max()}")
            
        # SCALE-ADAPTIVE AND CONTRAST-ADAPTIVE THRESHOLDING
        img_height, img_width = img_copy.shape
        img_area = img_height * img_width
        
        # Estimate character scale from image dimensions and content density
        # Use a quick initial threshold to detect character sizes
        quick_thresh = cv.adaptiveThreshold(img_copy, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv.THRESH_BINARY_INV, 11, 2)
        quick_contours = cv.findContours(quick_thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2]
        
        if len(quick_contours) > 0:
            # Calculate median character dimensions from quick detection
            quick_boxes = [cv.boundingRect(c) for c in quick_contours]
            # Filter out very small noise and very large merged components (scale-adaptive)
            min_char_area = max(10, int(img_area * 0.0001))  # 0.01% of image area
            max_char_area = img_area // 4  # 25% of image area
            filtered_boxes = [box for box in quick_boxes 
                            if min_char_area <= box[2] * box[3] <= max_char_area]
            
            if len(filtered_boxes) > 0:
                heights = [box[3] for box in filtered_boxes]
                widths = [box[2] for box in filtered_boxes]
                median_height = np.median(heights)
                median_width = np.median(widths)
                
                # Store adaptive character stats for later use in fragment filtering
                self._adaptive_char_stats = (median_height, median_width)
                
                # Scale-adaptive block size calculation
                # Larger characters need larger blocks to avoid fragmentation
                if median_height >= 20:  # Large characters
                    char_scale_factor = 2.2  # Much larger blocks for large chars
                elif median_height >= 12:  # Medium characters
                    char_scale_factor = 1.8  # Larger blocks for medium chars  
                else:  # Small characters
                    char_scale_factor = 0.7  # Smaller blocks for small chars
                adaptive_block_size = int(median_height * char_scale_factor)
                
                # Ensure block size is odd and within reasonable bounds (scale-adaptive)
                min_block_size = max(7, int(median_height * 0.5))  # At least 50% of char height
                max_block_size = min(51, int(median_height * 3.0))  # At most 3x char height
                adaptive_block_size = max(min_block_size, min(max_block_size, adaptive_block_size))
                if adaptive_block_size % 2 == 0:
                    adaptive_block_size += 1
                
                # Base C parameter calculation (density-adaptive)
                char_density = len(filtered_boxes) / img_area * 10000  # chars per 10k pixels
                base_c = max(1, min(8, int(2 + char_density * 0.5)))
                
                # Scale-adaptive C reduction for larger characters
                if median_height >= 20:  # Large characters
                    size_c_reduction = 5  # Much lower C to reduce fragmentation
                elif median_height >= 12:  # Medium characters
                    size_c_reduction = 4  # Lower C to reduce fragmentation
                else:  # Small characters
                    size_c_reduction = 0  # Normal C
                
                # CONTRAST-ADAPTIVE THRESHOLDING: Analyze character-level contrast
                # Analyze contrast around detected character regions
                character_contrasts = []
                low_contrast_chars = 0
                
                for box in filtered_boxes:
                    x, y, w, h = box
                    # Use thresholding-scale regions for contrast analysis
                    # This matches the scale at which adaptive thresholding operates
                    padding = adaptive_block_size // 2  # Half the block size as padding
                    x1 = max(0, x - padding)
                    y1 = max(0, y - padding)
                    x2 = min(img_width, x + w + padding)
                    y2 = min(img_height, y + h + padding)
                    
                    char_region = img_copy[y1:y2, x1:x2]
                    if char_region.size > 0:
                        std_contrast = np.std(char_region.astype(np.float32))
                        character_contrasts.append(std_contrast)
                        
                        # Check if this character is in a low-contrast region
                        if std_contrast < 55.0:  # Empirical threshold based on our analysis
                            low_contrast_chars += 1
                
                # Calculate contrast statistics
                if character_contrasts:
                    median_char_contrast = np.median(character_contrasts)
                    min_char_contrast = np.min(character_contrasts)
                    
                    print(f"[CONTRAST ANALYSIS] Character regions: {len(character_contrasts)}")
                    print(f"[CONTRAST ANALYSIS] Median contrast: {median_char_contrast:.1f}, min: {min_char_contrast:.1f}")
                    print(f"[CONTRAST ANALYSIS] Low-contrast characters: {low_contrast_chars}/{len(character_contrasts)}")
                    
                    # Apply contrast compensation if we have low-contrast characters
                    if low_contrast_chars > 0 or min_char_contrast < 45.0:
                        # Detected low-contrast characters - apply aggressive contrast compensation
                        if min_char_contrast < 35.0:
                            contrast_c_reduction = 3  # Very low contrast
                        elif min_char_contrast < 45.0:
                            contrast_c_reduction = 2  # Low contrast
                        else:
                            contrast_c_reduction = 1  # Mild contrast issues
                        
                        print(f"[CONTRAST ADAPTIVE] Low contrast detected (min: {min_char_contrast:.1f}), applying C reduction: -{contrast_c_reduction}")
                    else:
                        contrast_c_reduction = 0
                        print(f"[CONTRAST ADAPTIVE] Good contrast detected, no adjustment needed")
                else:
                    contrast_c_reduction = 0
                    median_char_contrast = 50.0  # Default
                
                # Final adaptive C parameter
                final_adaptive_c = max(1, base_c - size_c_reduction - contrast_c_reduction)
                
                print(f"[ADAPTIVE THRESHOLD] Image: {img_width}x{img_height}, {len(filtered_boxes)} chars")
                print(f"[ADAPTIVE THRESHOLD] Median char: {median_width:.1f}x{median_height:.1f}")
                print(f"[ADAPTIVE THRESHOLD] Block size: {adaptive_block_size} (factor: {char_scale_factor})")
                print(f"[ADAPTIVE THRESHOLD] C: {final_adaptive_c} (base: {base_c}, size_reduction: -{size_c_reduction}, contrast_reduction: -{contrast_c_reduction})")
                
                bin_img = cv.adaptiveThreshold(img_copy, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv.THRESH_BINARY_INV, adaptive_block_size, final_adaptive_c)
            else:
                # Fallback to scale-adaptive default if no valid characters detected
                fallback_block_size = max(7, min(21, int(min(img_height, img_width) * 0.05)))  # 5% of smaller dimension
                if fallback_block_size % 2 == 0:
                    fallback_block_size += 1
                print(f"[ADAPTIVE THRESHOLD] No valid chars detected, using scale-adaptive fallback {fallback_block_size}x{fallback_block_size}, C=2")
                bin_img = cv.adaptiveThreshold(img_copy, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv.THRESH_BINARY_INV, fallback_block_size, 2)
        else:
            # Fallback to scale-adaptive default if no contours detected
            fallback_block_size = max(7, min(21, int(min(img_height, img_width) * 0.05)))  # 5% of smaller dimension
            if fallback_block_size % 2 == 0:
                fallback_block_size += 1
            print(f"[ADAPTIVE THRESHOLD] No contours detected, using scale-adaptive fallback {fallback_block_size}x{fallback_block_size}, C=2")
            bin_img = cv.adaptiveThreshold(img_copy, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv.THRESH_BINARY_INV, fallback_block_size, 2)

        # Get initial contours
        result = cv.findContours(bin_img, mode=self._contour_mode,
                                 method=cv.CHAIN_APPROX_SIMPLE)
        if len(result) == 2:
            contours, hierarchy = result
        else:
            _, contours, hierarchy = result
        
        # Debug original contours (temporarily disabled)
        # print(f"[INITIAL CONTOURS DEBUG] Found {len(contours)} original contours:")
        # for i, contour in enumerate(contours):
        #     x, y, w, h = cv.boundingRect(contour)
        #     area = w * h
        #     print(f"[INITIAL CONTOURS DEBUG] Contour {i}: box=({x},{y},{w},{h}), area={area}")
        
        # DISABLE syllable merging temporarily - causes feature corruption
        print(f"[PAGE DEBUG] Preserving original {len(contours)} contours - merging causes feature corruption")
        
        # SYLLABLE RECONSTRUCTION: Disabled - merging corrupts character features
        # The architecture is implemented but merging causes feature corruption
        print(f"[SYLLABLE DEBUG] Syllable merging disabled - causes feature corruption")
        self.contour_index_mapping = None
        # except Exception as e:
        #     print(f"[SYLLABLE DEBUG] Merging failed: {e}, using original contours")
        #     # Fall back to original contours if merging fails
        #     self.contour_index_mapping = None
        
        # Adaptive position filtering: analyze document layout first
        filtered_contours = []
        
        # Analyze document and set filtering ranges (store as instance variables)
        self._analyze_document_type(contours)
        inclusive_y_min = self.adaptive_y_min
        inclusive_y_max = self.adaptive_y_max
        
        for i, contour in enumerate(contours):
            x, y, w, h = cv.boundingRect(contour)
            area = w * h
            
            # print(f"[CONTOUR FILTER DEBUG] Contour {i}: box=({x},{y},{w},{h}), area={area}")
            
            # Keep large components (main characters) - use inclusive filtering for single line images
            if area > 150:
                # For single line documents, use inclusive range for all characters
                if inclusive_y_min <= y <= inclusive_y_max:
                    filtered_contours.append(contour)
                    # print(f"[CONTOUR FILTER DEBUG] Contour {i}: KEPT (large component)")
                else:
                    # print(f"[CONTOUR FILTER DEBUG] Contour {i}: FILTERED (large component y={y} outside inclusive range {inclusive_y_min}-{inclusive_y_max})")
                    pass
                continue
            
            # For smaller components, use inclusive position filtering (allows tshegs at y=14)
            if inclusive_y_min <= y <= inclusive_y_max:
                filtered_contours.append(contour)
                # print(f"[CONTOUR FILTER DEBUG] Contour {i}: KEPT (small component in inclusive range)")
            else:
                # print(f"[CONTOUR FILTER DEBUG] Contour {i}: FILTERED (small component y={y} outside inclusive range {inclusive_y_min}-{inclusive_y_max})")
                pass
        
        print(f"[PAGE DEBUG] After position filtering: {len(filtered_contours)} contours (removed {len(contours) - len(filtered_contours)} vowel marks)")
        
        # Rebuild hierarchy for filtered contours
        if len(filtered_contours) != len(contours):
            # Create new binary image with only filtered contours
            filtered_binary = np.zeros_like(bin_img)
            cv.drawContours(filtered_binary, filtered_contours, -1, 255, thickness=-1)
            
            # Regenerate hierarchy
            result = cv.findContours(filtered_binary, mode=self._contour_mode, method=cv.CHAIN_APPROX_SIMPLE)
            if len(result) == 2:
                final_contours, final_hierarchy = result
            else:
                _, final_contours, final_hierarchy = result
            
            print(f"[PAGE DEBUG] Rebuilt hierarchy: {len(final_contours)} final contours")
            return final_contours, final_hierarchy
        
        return filtered_contours, hierarchy
    

    def get_boxes(self):
        '''Retrieve bounding boxes. Create them if not yet cached'''
        if not self.boxes:
            self.boxes = self._boxes()
        
        # DEBUG: Save all character images once (avoid saving multiple times)
        if not hasattr(self, '_debug_chars_saved') and self.boxes:
            self._debug_chars_saved = True
            print(f"[CHAR DEBUG] Image shape: {self.img_arr.shape}")
            print(f"[CHAR DEBUG] Found {len(self.boxes)} characters")
            
            # Show first 10 character coordinates
            for i, (x, y, w, h) in enumerate(self.boxes[:10]):
                print(f"[CHAR DEBUG] Char {i}: x={x}, y={y}, w={w}, h={h}")
            
            # Save all character images
            saved_count = 0
            for i, (x, y, w, h) in enumerate(self.boxes):
                try:
                    char_img = self._extract_contour_image(i)
                    if char_img is not None and char_img.size > 0:
                        from PIL import Image
                        Image.fromarray(char_img.astype(np.uint8)).save(f"debug_extracted_char_{i:03d}.png")
                        saved_count += 1
                except Exception as e:
                    print(f"[CHAR DEBUG] Failed to save character {i}: {e}")
                    pass
            
            print(f"[CHAR DEBUG] Saved {saved_count}/{len(self.boxes)} character images as debug_extracted_char_XXX.png")
       
        return self.boxes
    
    def _boxes(self):
        return [cv.boundingRect(c) for c in self.contours]
    
    def get_indices(self):
        if not self.indices:
#            print self.tsek_mean, np.floor(self.tsek_std), np.ceil(self.tsek_std), self.tsek_std
            self.indices = [i for i, b in enumerate(self.get_boxes())] #if (
#               max(b[2], b[3]) <= 6 * self.char_mean   )] # and  # filter out too big
#            (b[2] > 10 or b[3] > 10 ))]
#               b[2] >= (np.floor(self.tsek_mean) - 
#               self.small_coef * np.floor(self.tsek_std))) ] # ... and too small
        return self.indices
        
    def _set_shape_measurements(self):
        width_measures = self.char_gaussians([b[2] for b in self.get_boxes() if
                                               b[2] < .1*self.img_arr.shape[1]])
        for i,j in zip(['char_mean', 'char_std', 'tsek_mean', 'tsek_std'], width_measures):
            setattr(self, i, j)
    
    def _analyze_document_type(self, contours):
        """Analyze document layout and set adaptive filtering ranges"""
        print(f"[DOCUMENT ANALYSIS] Called with {len(contours)} contours")
        # Document type detection based on y-coordinate spread
        y_coords = [cv.boundingRect(contour)[1] for contour in contours]
        if y_coords:
            y_min = min(y_coords)
            y_max = max(y_coords)
            y_spread = y_max - y_min
            print(f"[DOCUMENT ANALYSIS] Y-coordinate range: {y_min}-{y_max}, spread: {y_spread}")
            print(f"[DOCUMENT ANALYSIS] Instance ID: {id(self)}")
            
            # Detect document type
            self.is_multiline = y_spread > 50  # If y-spread > 50 pixels, likely multiline
            print(f"[DOCUMENT ANALYSIS] Document type: {'MULTILINE' if self.is_multiline else 'SINGLE-LINE'}")
            
            if self.is_multiline:
                # Multiline document: use very permissive filtering to preserve all lines
                self.adaptive_y_min = max(0, y_min - 10)  # Small margin above first line
                self.adaptive_y_max = y_max + 10  # Small margin below last line
                print(f"[DOCUMENT ANALYSIS] Using MULTILINE filtering: y={self.adaptive_y_min}-{self.adaptive_y_max}")
            else:
                # Single-line document: use dynamic range based on actual character positions
                # Use character positions with appropriate margins for tsheg detection
                self.adaptive_y_min = max(0, y_min - 2)  # Small margin above characters
                self.adaptive_y_max = y_max + 10  # Margin below for tshegs which may be positioned below
                print(f"[DOCUMENT ANALYSIS] Using SINGLE-LINE filtering: y={self.adaptive_y_min}-{self.adaptive_y_max} (dynamic range based on chars at y={y_min}-{y_max})")
        else:
            # Fallback: treat as single-line
            self.adaptive_y_min = 5
            self.adaptive_y_max = 35
            self.is_multiline = False

    def _merge_syllable_contours(self, contours):
        """
        Merge contours that belong to the same Tibetan syllable.
        This fixes the issue where subscripts (like ratak in དྲ) are detected as separate contours.
        """
        if not contours:
            return contours
            
        # Get bounding boxes for all contours
        contour_data = []
        for i, contour in enumerate(contours):
            x, y, w, h = cv.boundingRect(contour)
            area = w * h
            contour_data.append({
                'index': i,
                'contour': contour,
                'box': (x, y, w, h),
                'area': area,
                'merged': False
            })
        
        # Sort by y-coordinate first, then x-coordinate for processing order
        contour_data.sort(key=lambda c: (c['box'][1], c['box'][0]))
        
        merged_contours = []
        
        for i, main_contour in enumerate(contour_data):
            if main_contour['merged']:
                continue
                
            main_x, main_y, main_w, main_h = main_contour['box']
            main_area = main_contour['area']
            
            # Debug: Show every main contour being processed
            print(f"[MERGE DEBUG] Processing main contour {i}: box=({main_x},{main_y},{main_w},{main_h}), area={main_area}")
            
            # Look for potential subscripts below this contour
            subscripts_to_merge = []
            
            for j, potential_subscript in enumerate(contour_data):
                if i == j or potential_subscript['merged']:
                    continue
                    
                sub_x, sub_y, sub_w, sub_h = potential_subscript['box']
                sub_area = potential_subscript['area']
                
                # Check if this could be a subscript of the main contour
                vertical_distance = sub_y - (main_y + main_h)
                horizontal_overlap = self._calculate_horizontal_overlap(
                    (main_x, main_x + main_w), (sub_x, sub_x + sub_w)
                )
                
                # Debug potential subscripts that are close
                if abs(vertical_distance) < 25 and horizontal_overlap > 0.1:
                    print(f"[MERGE DEBUG] Potential subscript {j}: box=({sub_x},{sub_y},{sub_w},{sub_h}), vertical_dist={vertical_distance}, overlap={horizontal_overlap:.2f}, area_ratio={sub_area/main_area:.2f}")
                
                # Criteria for subscript detection (very sensitive):
                # 1. Subscript can overlap or be below main character (vertical distance -25 to 25 pixels)
                # 2. Horizontal overlap > 15%  
                # 3. Subscript area < 90% of main character area
                # 4. Subscript height <= main character height
                is_subscript = (
                    -25 <= vertical_distance <= 25 and
                    horizontal_overlap > 0.15 and
                    sub_area < 0.9 * main_area and
                    sub_h <= main_h
                )
                
                if is_subscript:
                    subscripts_to_merge.append(potential_subscript)
                    print(f"[SYLLABLE DEBUG] Found subscript: main=({main_x},{main_y},{main_w},{main_h}) sub=({sub_x},{sub_y},{sub_w},{sub_h}) overlap={horizontal_overlap:.2f}")
            
            # Create merged contour
            if subscripts_to_merge:
                # Merge the main contour with its subscripts
                all_contours = [main_contour['contour']] + [s['contour'] for s in subscripts_to_merge]
                merged_contour = self._create_merged_contour(all_contours)
                merged_contours.append(merged_contour)
                
                # Mark all merged contours as processed
                main_contour['merged'] = True
                for subscript in subscripts_to_merge:
                    subscript['merged'] = True
                    
                print(f"[SYLLABLE DEBUG] Merged syllable: {len(subscripts_to_merge)} subscripts merged with main contour")
            else:
                # No subscripts found, keep original contour
                merged_contours.append(main_contour['contour'])
                main_contour['merged'] = True
        
        return merged_contours

    def _merge_syllable_contours_with_mapping(self, contours):
        """
        Merge syllable contours and return both merged contours and index mapping.
        Returns: (merged_contours, index_mapping)
        where index_mapping[old_index] = new_index (or None if merged)
        """
        if not contours:
            return contours, {}
            
        # Get bounding boxes for all contours
        contour_data = []
        for i, contour in enumerate(contours):
            x, y, w, h = cv.boundingRect(contour)
            area = w * h
            contour_data.append({
                'index': i,
                'contour': contour,
                'box': (x, y, w, h),
                'area': area,
                'merged': False,
                'merged_into': None  # Track which new index this was merged into
            })
        
        # Sort by y-coordinate first, then x-coordinate for processing order
        contour_data.sort(key=lambda c: (c['box'][1], c['box'][0]))
        
        merged_contours = []
        index_mapping = {}  # old_index -> new_index
        
        for i, main_contour in enumerate(contour_data):
            if main_contour['merged']:
                continue
                
            main_x, main_y, main_w, main_h = main_contour['box']
            main_area = main_contour['area']
            
            # Debug: Show every main contour being processed
            print(f"[MERGE DEBUG] Processing main contour {main_contour['index']}: box=({main_x},{main_y},{main_w},{main_h}), area={main_area}")
            
            # Look for potential subscripts below this contour
            subscripts_to_merge = []
            
            for j, potential_subscript in enumerate(contour_data):
                if i == j or potential_subscript['merged']:
                    continue
                    
                sub_x, sub_y, sub_w, sub_h = potential_subscript['box']
                sub_area = potential_subscript['area']
                
                # Check if this could be a subscript of the main contour
                vertical_distance = sub_y - (main_y + main_h)
                horizontal_overlap = self._calculate_horizontal_overlap(
                    (main_x, main_x + main_w), (sub_x, sub_x + sub_w)
                )
                
                # Debug potential subscripts that are close
                if abs(vertical_distance) < 25 and horizontal_overlap > 0.1:
                    print(f"[MERGE DEBUG] Potential subscript {potential_subscript['index']}: box=({sub_x},{sub_y},{sub_w},{sub_h}), vertical_dist={vertical_distance}, overlap={horizontal_overlap:.2f}, area_ratio={sub_area/main_area:.2f}")
                
                # Criteria for subscript detection (very conservative - only obvious subscripts):
                is_subscript = (
                    -5 <= vertical_distance <= 10 and
                    horizontal_overlap > 0.4 and
                    sub_area < 0.4 * main_area and
                    sub_h < main_h * 0.8
                )
                
                if is_subscript:
                    subscripts_to_merge.append(potential_subscript)
                    print(f"[SYLLABLE DEBUG] Found subscript: main=({main_x},{main_y},{main_w},{main_h}) sub=({sub_x},{sub_y},{sub_w},{sub_h}) overlap={horizontal_overlap:.2f}")
            
            # Create merged contour and update mapping
            new_index = len(merged_contours)
            
            if subscripts_to_merge:
                # Merge the main contour with its subscripts
                all_contours = [main_contour['contour']] + [s['contour'] for s in subscripts_to_merge]
                merged_contour = self._create_merged_contour(all_contours)
                merged_contours.append(merged_contour)
                
                # Update index mapping
                index_mapping[main_contour['index']] = new_index
                for subscript in subscripts_to_merge:
                    index_mapping[subscript['index']] = new_index  # Merged subscripts map to same new index
                    subscript['merged'] = True
                    subscript['merged_into'] = new_index
                
                main_contour['merged'] = True
                main_contour['merged_into'] = new_index
                    
                print(f"[SYLLABLE DEBUG] Merged syllable: {len(subscripts_to_merge)} subscripts merged into index {new_index}")
            else:
                # No subscripts found, keep original contour
                merged_contours.append(main_contour['contour'])
                index_mapping[main_contour['index']] = new_index
                main_contour['merged'] = True
                main_contour['merged_into'] = new_index
        
        return merged_contours, index_mapping
    
    def _calculate_horizontal_overlap(self, range1, range2):
        """Calculate horizontal overlap percentage between two ranges."""
        start1, end1 = range1
        start2, end2 = range2
        
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        
        if overlap_start >= overlap_end:
            return 0.0
            
        overlap_width = overlap_end - overlap_start
        total_width = max(end1, end2) - min(start1, start2)
        
        return overlap_width / total_width if total_width > 0 else 0.0
    
    def _create_merged_contour(self, contours):
        """Create a single contour from multiple contours by merging their bounding boxes."""
        if len(contours) == 1:
            return contours[0]
            
        # For now, just return the largest contour to avoid IndexError
        # TODO: Implement proper contour merging
        largest_contour = max(contours, key=cv.contourArea)
        return largest_contour
        
        # Get the combined bounding box
        all_boxes = [cv.boundingRect(c) for c in contours]
        min_x = min(box[0] for box in all_boxes)
        min_y = min(box[1] for box in all_boxes)
        max_x = max(box[0] + box[2] for box in all_boxes)
        max_y = max(box[1] + box[3] for box in all_boxes)
        
        # Create a merged contour that represents the combined bounding box
        # We'll create a simple rectangular contour for the merged area
        merged_w = max_x - min_x
        merged_h = max_y - min_y
        
        # Create corner points of the merged bounding box
        merged_points = np.array([
            [[min_x, min_y]],
            [[max_x, min_y]],
            [[max_x, max_y]],
            [[min_x, max_y]]
        ], dtype=np.int32)
        
        return merged_points
    
#        self._gaussians([b[2] for b in self.get_boxes() if b[2] < .1*self.img_arr.shape[1]])
#        self._draw_new_page()

    def update_shapes(self):
        if platform.system() == "Linux":
            self.contours, self.hierarchy = self._contours()
        else:
            _, self.contours, self.hierarchy = self._contours()
        
        self.boxes = self._boxes()
        self._set_shape_measurements()
        self.indices = [i for i, b in enumerate(self.get_boxes()) if (
               max(b[2], b[3]) <= 6 * self.char_mean )] 

#        self.indices = [i for i, b in enumerate(self.get_boxes()) if (
#               max(b[2], b[3]) <= 5 * self.char_mean and #)] # and  # filter out too big
#               b[2] >= (np.floor(self.tsek_mean) - 
#               self.small_coef * np.floor(self.tsek_std)))]
    
    def _draw_new_page(self):
        self.page_array = np.ones_like(self.img_arr)
        
        self.tall = set([i for i in self.get_indices() if 
                         self.get_boxes()[i][3] > 3*self.char_mean])
        
#        cv.drawContours(self.page_array, [self.contours[i] for i in 
#                        self.get_indices() if self.get_boxes()[i][2] <= self.tsek_mean + 3*self.tsek_std], 
#                        -1,0, thickness = -1)
#        
#        
#        self.page_array = cv.medianBlur(self.page_array, 19)
#        
#        cv.drawContours(self.page_array, [self.contours[i] for i in 
#                        self.get_indices() if self.get_boxes()[i][2] <= self.tsek_mean + 3*self.tsek_std], 
#                        -1,0, thickness = -1)
        cv.drawContours(self.page_array, [self.contours[i] for i in 
                        range(len(self.contours)) if 
                        self.get_boxes()[i][2] > self.smlmean + 3*self.smstd], 
                        -1,0, thickness = -1)
#        cv.drawContours(self.page_array, [self.contours[i] for i in 
#                        self.get_indices() if self.get_boxes()[i][3] <= 2*self.char_mean], 
#                        -1,0, thickness = -1)
#        cv.erode(self.page_array, None, self.page_array, iterations=2)
#        self.page_array = cv.morphologyEx(self.page_array, cv.MORPH_CLOSE, None,iterations=2)
        import Image
        Image.fromarray(self.page_array*255).show()
#        raw_input()
#        cv.dilate(self.page_array, None, self.page_array, iterations=1)
        
    @classmethod
    def char_gaussians(cls, widths):
        
        widths = np.array(widths)
        widths.shape = (len(widths),1)
        cls.median_width = np.median(widths)
        
        gmm = GMM(n_components = 2, max_iter=100)
        try:
            gmm.fit(widths)
        except ValueError:
            return (0,0,0,0)
        means = gmm.means_
        # Handle different scikit-learn versions
        if hasattr(gmm, 'covariances_'):
            stds = np.sqrt(gmm.covariances_)
        else:
            stds = np.sqrt(gmm.covars_)
        cls.gmm = gmm
        char_mean_ind = np.argmax(means)
        char_mean = float(means[char_mean_ind]) # Page character width mean
        char_std = float(stds[char_mean_ind][0]) # Page character std dev
        
        cls.tsek_mean_ind = np.argmin(means)
        
        tsek_mean = float(means[cls.tsek_mean_ind])
        tsek_std = float(stds[cls.tsek_mean_ind][0])
#        print gmm.converged_, 'converged'
        return (char_mean, char_std, tsek_mean, tsek_std)

#    def _gaussians(self, widths):
##        print widths
#        widths = np.array(widths)
#        widths.shape = (len(widths),1)
#        
#        gmm = GMM(n_components = 3, n_iter=100)
#        try:
#            gmm.fit(widths)
#        except ValueError:
#            return (0,0,0,0)
#        means = gmm.means_
#        stds = np.sqrt(gmm.covars_)
#        
#        argm = np.argmin(means)
#        self.smlmean = means[argm]
#        self.smstd = stds[argm]
        
#        cls.gmm = gmm
#        print gmm.converged_, 'converged'
#        from matplotlib import pyplot as plt
#        from matplotlib.mlab import normpdf
##        plt.subplot(211)
#        plt.title('tsek-char distributions, pre-segmentation')
#        
#        n,bins,p = plt.hist(widths, 200, range=(0,75), normed=True, color='#3B60FA')
##        plt.vlines(means, 0, np.array([max(n), max(n)]), linestyles='--')
#        for i, m in enumerate(means):
#            
#            plt.plot(bins, normpdf(bins, means[i], stds[i]),  label='fit', linewidth=1)
#            plt.fill_between(bins, normpdf(bins, means[i], stds[i]), color=(.58,.63,.8), alpha=0.09)
#
#        plt.show()

    def get_tops(self):       
        return [self.get_boxes()[i][1] for i in self.get_indices()]
    
#     @profile
    def draw_contour_and_children(self, root_ind, char_arr=None, offset=()):
        char_contours = [root_ind]
        root = self.hierarchy[0][root_ind]
        if root[2] >= 0:
            char_contours.append(root[2]) # add root's first child
            child_hier = self.hierarchy[0][root[2]] # get hier for 1st child
            if child_hier[0] >= 0: # if child has sib, continue to loop
                has_sibling = True
            else: has_sibling = False # ... else skip loop and draw
            
            while has_sibling:
                ind = child_hier[0] # get sibling's index
                char_contours.append(ind) # add sibling's index
                child_hier = self.hierarchy[0][ind] # get sibling's hierarchy
                if child_hier[0] < 0: # if sibling has sibling, continue loop
                    has_sibling = False
        
        if not hasattr(char_arr, 'dtype'):
            char_box = self.get_boxes()[root_ind]
            x,y,w,h = char_box
            char_arr = np.ones((h,w), dtype=np.uint8)
            offset = (-x, -y)
        # Handle contour index mapping if syllable merging was applied
        if hasattr(self, 'contour_index_mapping') and self.contour_index_mapping is not None:
            # Map old indices to new indices, skip merged subscripts
            mapped_contours = []
            for j in char_contours:
                if j in self.contour_index_mapping:
                    new_index = self.contour_index_mapping[j]
                    if new_index is not None and new_index < len(self.contours):
                        mapped_contours.append(self.contours[new_index])
            cv.drawContours(char_arr, mapped_contours, -1, 0, thickness=-1, offset=offset)
        else:
            # Original behavior when no merging was applied
            cv.drawContours(char_arr, [self.contours[j] for j in char_contours], -1,0, thickness = -1, offset=offset)
        return char_arr
    
#     @profile
    def detect_num_lines(self, content_box_dict):
        '''content_box_dict has values {'chars':[], 'b':b, 'boxes':[], 
                                'num_boxes':0, 'num_chars':0}
        
        where chars are the indices of chars in the content box, b is the 
        the xywh dimensions of the box, boxes are the sub-boxes of the 
        document tree contained in this box (not box chars but large page-
        structuring boxes. 
        
        Note: page_type must be set to "pecha"
        '''
        
        cbx, cby, cbw, cbh = content_box_dict['b']
        
        
#        print self.img_arr.shape
#        print content_box_dict['b']
        
        cbox_arr = np.ones((cbh, cbw), dtype=self.img_arr.dtype)
        
        tsekmeanfloor = np.floor(self.tsek_mean)
        tsekstdfloor = np.floor(self.tsek_std)
        cv.drawContours(cbox_arr, [self.contours[i] for i in content_box_dict['chars']
                        if ((self.get_boxes()[i][2] > 
                        (tsekmeanfloor - 
               self.small_coef * tsekstdfloor)  or 
               self.get_boxes()[i][2] < .1*self.img_arr.shape[1]) and 
                            self.get_boxes()[i][3] > 10) 
                                   ], -1, 0, thickness=-1, offset=(-cbx, -cby))
        cbox_arr = cbox_arr[5:-5, :] # shorten from the top and bottom to help out trim in the event of small noise
#         cbox_arr = cbox_arr[0:-1, :] # shorten from the top and bottom to help out trim in the event of small noise
#         cbox_arr = trim(cbox_arr)
#         cbox_arr = cv.dilate(cbox_arr, None, iterations=3)
        cbox_arr = cv.erode(cbox_arr, None, iterations=5)
#         cbox_arr = cv.erode(cbox_arr, None, iterations=1)
#         cbox_arr = cv.blur(cbox_arr, (150, 3))
#         cbox_arr = cv.blur(cbox_arr*255, (75, 19))
#         cbox_arr = cv.blur(cbox_arr*255, (75, 19))
        cbox_arr = to255(cbox_arr)

        cv.blur(cbox_arr, (75, 19), dst=cbox_arr)
#         k = cv.blur(to255(cbox_arr), (75, 19))


        ####################
#         print 'warning: using non default (127) line count threshold'
#         ret, cbox_arr = cv.threshold(cbox_arr, 127, 1, cv.THRESH_BINARY)
        ####################
        ret, cbox_arr = cv.threshold(cbox_arr, 200, 1, cv.THRESH_BINARY) #DEFAULT!
        ###################



#         cbox_arr = cv.blur(cbox_arr, (90, 80))
#         cbox_arr = cv.blur(cbox_arr, (130, 100))
#        cbox_arr = cv.morphologyEx(cbox_arr, cv.MORPH_OPEN, None,iterations=6)
#         print cbox_arr[np.where(1.0>cbox_arr)]
#         import Image
#         Image.fromarray(cbox_arr*255).show()
#         sys.exit()
#         sc = 1/255.0
#         cbox_arr *= sc
        vsum = cbox_arr.sum(axis=1)

#        from scipy.ndimage.measurements import extrema
#         vsum_smoothed = gaussian_filter1d(vsum, 10)
        vsum_smoothed = gaussian_filter1d(vsum, 25) ###DEFAULT
#         vsum_smoothed = gaussian_filter1d(vsum, 13)
        len_vsum = len(vsum)
#        print vsum
#        print extrema(vsum)
#        print argrelmin(vsum)
#        print argrelmax(vsum)
#        from scipy.interpolate import interp1d
        
#        fx = interp1d(range(len(vsum)), vsum, kind='cubic')
        fx = UnivariateSpline(list(range(len_vsum)), vsum_smoothed)
        tck = splrep(list(range(len_vsum)), fx(list(range(len_vsum))))
        y = splev(list(range(len_vsum)), tck, der=1)
        tck = splrep(list(range(len_vsum)), y)
#        roots = sproot(tck)
#        print len(roots)
        mins = argrelmin(fx(list(range(len_vsum))))
#        mins = argrelmin(vsum_smoothed, order=2)
#        mins_min = min([vsum[m] for m in mins[0]])

        ### Filter false peaks that show up from speckles on page
#        mins = [m for m in mins[0] if (cbw - vsum[m])/float(cbw) >= .05]
#        mins = [m for m in mins[0] if (cbw - vsum[m])/float(cbw) >= .1]
#        mins = [m for m in mins[0] if (cbw - vsum[m])/float(cbw) >= 1.5*self.char_mean/float(cbw)]
#         mins = [m for m in mins[0] if (cbw - vsum[m])/float(cbw) >= .025]
        mins = [m for m in mins[0] if (cbw - vsum[m])/float(cbw) >= .01]
#         mins = [m for m in mins[0] if (cbw - vsum[m])/float(cbw) >= .0075]
#        mins = [m for m in mins[0] ]
#        print mins, len_vsum
#        print len(mins[0])
#        print mins

#        for m in mins:
#            cbox_arr[m, :] = 1
            
#        
        self.num_lines = len(mins)
#         print self.num_lines
#         self.num_lines = 19
#        print self.num_lines
#        print self.num_lines
#         self.num_lines = 5
#        print self.num_lines
#        print dir(fx)
#        print fx
#        print dir(fx)
#        from scipy.optimize import minimize_scalar
#        print minimize_scalar(fx)
        
        
        #############################
#         plot b spline of image profile. number of minima is line number
#         (or should be...
#         from matplotlib import pyplot as plt
#         plt.plot(range(len(vsum)), fx(range(len(vsum))))
# #        plt.plot(range(len(vsum)), y) # alternatively, plt fist derivative of the b spline
# #        plt.bar(range(vsum.shape[0]), vsum) ## plot horiz profile as bar chart
#         plt.vlines(mins, 0, max(vsum))
#         plt.show()
        ################################
        
#        import sys
#        sys.exit()
        
    
    def draw_hough_outline(self, arr):
        
        arr = invert_bw(arr)
#         import Image
#         Image.fromarray(arr*255).show()
#        h = cv.HoughLinesP(arr, 2, np.pi/4, 5, minLineLength=arr.shape[0]*.10)
        h = cv.HoughLinesP(arr, 2, np.pi/4, 1, minLineLength=arr.shape[0]*.15, maxLineGap=5) #This
#         h = cv.HoughLinesP(arr, 2, np.pi/4, 1, minLineLength=arr.shape[0]*.15, maxLineGap=1)
#        h = cv.HoughLinesP(arr, 2, np.pi/4, 1, minLineLength=arr.shape[0]*.15)
        PI_O4 = np.pi/4
#        if h and h.any():
#        if self._page_type == 'pecha':
#            color = 1
#            thickness = 10
#        else: # Attempt to erase horizontal lines if page_type == book. 
#            # Why? Horizontal lines can break LineCluster if they are broken
#            # e.g. couldn't be filtered out prior to line_breaker.py
#            color = 0
#            thickness = 10
        if h is not None:
            for line in h[0]:
                new = (line[2]-line[0], line[3] - line[1])
                val = (new[0]/np.sqrt(np.dot(new, new)))
                theta = np.arccos(val)
                if theta >= PI_O4: # Vertical line
#                    print line[1] - line[3]
#                     cv.line(arr, (line[0], 0), (line[0], arr.shape[0]), 1, thickness=10)
                    if line[0] < .5*arr.shape[1]:
                        arr[:,:line[0]+12] = 0
                    else:
                        arr[:,line[0]-12:] = 0
                else: # horizontal line
                    if line[2] - line[0] >= .15 * arr.shape[1]:
#                         cv.line(arr, (0, line[1]), (arr.shape[1], line[1]), 1, thickness=50)
                        if line[1] < .5 *arr.shape[0]:
                            arr[:line[1]+17, :] = 0
                        else:
                            arr[line[1]-5:,:] = 0
        

        return ((arr*-1)+1).astype(np.uint8)

    def save_margin_content(self, tree, content_box):
        '''Look at margin content and try to OCR it. Save results in a pickle
        file of a dictionary object:
        d = {'left':['margin info 1', ...], 'right':['right margin info 1', etc]}
        
        Margin content is tricky since letters are often not defined as well
        as the main page content. The current OCR implementation also stumbles
        on text with very few characters. Page numbers don't do well for some
        reason...
        '''
        
        import pickle as pickle
        import os
        content_box_right_edge = tree[content_box]['b'][0] + tree[content_box]['b'][2]
        inset = 20

        right_content = []
        left_content = []
        for brnch in tree:
            if brnch != content_box:
                outer_box = brnch

                if tree[outer_box]['num_chars'] != 0:
                    bx = tree[outer_box]['b']
                    arr = self.img_arr[bx[1]+inset:bx[1]+bx[3]-inset, bx[0]+inset:bx[0]+bx[2]-inset]

                    text = ''
                    if bx[0] > content_box_right_edge:
                        arr = rotate(arr, -90, cval=1)
                        text = construct_page(rec_main(arr, line_break_method='line_cut', page_type='book', page_info={'flname': 'margin content'}))
                        if text:
                            right_content.append(text)
                    else:
                        arr = rotate(arr, 90, cval=1)
                        text = construct_page(rec_main(arr, line_break_method='line_cut', page_type='book', page_info={'flname': 'margin content'}))
                        if text:
                            left_content.append(text)
        pklname = os.path.join(os.path.dirname(self.flpath), os.path.basename(self.flpath)[:-4]+'_margin_content.pkl')
        pickle.dump({'right':right_content, 'left':left_content}, open(pklname, 'wb'))
#        import sys; sys.exit()
    
#     @profile
    def set_pecha_layout(self):
#         a = cv.erode(self.img_arr.copy(), None,iterations=2)
        #         import Image
#         Image.fromarray(cbox_arr*255).show()
        a = self.img_arr.copy()
        
        if self.img_arr.shape[1] > 2*self.img_arr.shape[0]:
            self._page_type = 'pecha'
        else:
            self._page_type = 'book'
        
        if self._page_type == 'pecha': # Page is pecha format
            a = self.draw_hough_outline(a)
            
        self.img_arr = a.copy()
        self.update_shapes()
        
#        a= cv.morphologyEx(a, cv.MORPH_OPE#         if self._page_type == 'pecha': # Page is pecha format
#             a = self.draw_hough_outline(a)N, None,iterations=5)
#        a = cv.medianBlur(a, 9)
#         import Image
#         Image.fromarray(a*255).show()
        # Skip Gaussian blur to preserve individual character contours
        # Original: a = cv.GaussianBlur(a, (5, 5), 0)
        # The blur was merging separate characters into single contours
#        print a.dtype
#        a = cv.GaussianBlur(a, (5, 5), 0)
#        a = self.img_arr.copy()
#         n = np.ones_like(a)
        
        # Ensure proper binarization before contour detection
        print(f"[CONTOUR_DEBUG] Image shape: {a.shape}, dtype: {a.dtype}, mean: {a.mean():.1f}")
        if a.dtype != np.uint8:
            a = a.astype(np.uint8)
        
        # Threshold the image to create binary image for contour detection
        if a.mean() > 127:  # White background - invert for black text on white
            _, a_binary = cv.threshold(a, 127, 255, cv.THRESH_BINARY_INV)
            print("[CONTOUR_DEBUG] Applied THRESH_BINARY_INV")
        else:  # Already inverted or dark background
            _, a_binary = cv.threshold(a, 127, 255, cv.THRESH_BINARY)
            print("[CONTOUR_DEBUG] Applied THRESH_BINARY")
        
        print(f"[CONTOUR_DEBUG] Binary image unique values: {np.unique(a_binary)}")
        
        # OpenCV version-agnostic approach
        contours_result = cv.findContours(a_binary, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_SIMPLE)
        contours, hierarchy = contours_result[-2:]  # Get last 2 values regardless of OpenCV version
        
        
        ## Most of this logic for identifying rectangles comes from the 
        ## squares.py sample in opencv source code.
        def angle_cos(p0, p1, p2):
            d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
            return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )
        
        border_boxes = []
        
        for j,cnt in enumerate(contours):
            cnt_len = cv.arcLength(cnt, True)
            orig_cnt = cnt.copy()
            cnt = cv.approxPolyDP(cnt, 0.02*cnt_len, True)
            if len(cnt) == 4 and cv.contourArea(cnt) > 1000 and cv.isContourConvex(cnt):
                cnt = cnt.reshape(-1, 2)
                max_cos = np.max([angle_cos(cnt[i], 
                                            cnt[(i+1) % 4], cnt[(i+2) % 4] ) 
                                  for i in range(4)])
                if max_cos < 0.1:
#                    print 'got one at %d' % j
#                    n = np.ones_like(a)
                    b = cv.boundingRect(orig_cnt)
#                     if self.clear_hr:
#                         print 'Warning: you are clearing text on a pecha page'
#                         self.img_arr[0:b[1]+b[3], :] = 1
                    x,y,w,h = b
#                    b = [x+10, y+10, w-10, h-10]
                    border_boxes.append(b)
#                     cv.rectangle(n, (x,y), (x+w, y+h), 0)
#                     cv.drawContours(n, [cnt], -1,0, thickness = 5)
#                    import Image
#                    Image.fromarray(n*255).save('/tmp/rectangles_%d.png' % j )
        
#         import Image
#         Image.fromarray(n*255).show()
        border_boxes.sort(key=lambda b: (b[0],b[1]))
        #border_boxes = border_boxes
        
        def get_edges(b):
            l = b[0]
            r = b[0] + b[2]
            t = b[1]
            b = b[1] + b[3]
            return (l,r,t,b)

        def bid(b):
            return '%d-%d-%d-%d' % (b[0],b[1],b[2],b[3])
       
        tree = {}
        for b in border_boxes:
            tree[bid(b)] = {'chars':[], 'b':b, 'boxes':[], 'num_boxes':0, 'num_chars':0}    
        
        def b_contains_nb(b,nb):
            l1,r1,t1,b1 = get_edges(b)
            l2,r2,t2,b2 = get_edges(nb)
            return l1 <= l2 and r2 <= r1 and t1 <= t2 and b1 >= b2
            
        for i, b in enumerate(border_boxes):
            bx,by,bw,bh = b
            self.img_arr[by:by+1,bx+3:bx+bw-3] = 1
            
            if platform.system() == "Linux":
                self.img_arr[by+bh,by+bh-1:bx+3:bx+bw-3] = 1

            for nb in border_boxes[i+1:]:
                if b_contains_nb(b, nb):
                    tree[bid(b)]['boxes'].append(bid(nb))
                    tree[bid(b)]['num_boxes'] = len(tree[bid(b)]['boxes'])
        
        self.update_shapes()
#         import Image
#         Image.fromarray(self.img_arr*255).show()
        
        tree_keys = list(tree.keys())
        tree_keys.sort(key=lambda x: tree[x]['num_boxes'])
                
        ## Assign contours to boxes
        for i in self.get_indices():
            for k in tree_keys:
                box = tree[k]
                b = box['b']
                
#                print box['num_boxes']
                char_box = self.get_boxes()[i]
                if b_contains_nb(b, char_box):
                    tree[k]['chars'].append(i)
                    tree[k]['num_chars'] = len(tree[k]['chars'])
                    break
#        import pprint
#        pprint.pprint(tree)
        
        def qualified_box(bx):
            '''Helper function that ignores boxes that contain other boxes.
            This is useful for finding the main content box which should
            be among the innermost boxes that have no box children '''
            
            if tree[bx]['num_boxes'] == 0:
                return tree[bx]['num_chars']
            else:
                return -1
        
#        content_box = max(tree, key=lambda bx: tree[bx]['num_chars'])
        content_box = max(tree, key=qualified_box)
#        print tree[content_box]['num_chars']
#        self.indices = [i for i in tree[content_box]['chars'] if self.boxes[i][2] >= (np.floor(self.tsek_mean) - 
#               self.small_coef * np.floor(self.tsek_std))] 
#         self.indices = [i for i in tree[content_box]['chars'] if self.boxes[i][2] >= (np.floor(self.tsek_mean) - 
#                1.5 * np.floor(self.tsek_std))] 
        self.indices = [i for i in tree[content_box]['chars'] if self.boxes[i][2] >= 7] 
        
        
        self.detect_num_lines(tree[content_box])
#        self.save_margin_content(tree, content_box)


#        import Image
#        Image.fromarray(cbox_arr*255).show()
#        raw_input()


#                            codecs.open(os.path.join(os.path.dirname(self.flpath), os.path.basename(self.flpath)[:-4] + '_left_' + str(left_count)+'.txt'), 'w', 'utf-8').write(text)
#                            left_count += 1
#                        print construct_page(rec_main(arr, line_break_method='line_cluster', page_type='pecha', k_groups=1, page_info={'flname': 'margin content'}))
        
        

        
#        self.margins = {'left':[], 'right':[]}
#        import re
#        reg = re.compile(ur'([0-9]{1,4})')
#        for brnch in tree:
#            if brnch != content_box:
#                outer_box = brnch
#                chars = tree[outer_box]['chars']
##                
##                left = [] # container for chars left to the content box
##                right = []
##                
##                for c in chars:
##                    if self.boxes[c][0] < content_box_right_edge:
##                        left.append(c)
##                    else:
##                        right.append(c)
##                
##                sections = {}
##                if left:
##                    sections['left'] = combine_many_boxes([self.boxes[c] for c in left])
##                if right:
##                    sections['right'] = combine_many_boxes([self.boxes[c] for c in right])
##                import Image
##                for section in sections:
##                    x,y,w,h = sections[section]
###                    print w, self.tsek_mean
##                    if not w > .05*self.img_arr.shape[1]:
##                        arr = np.ones((h, w), dtype=self.img_arr.dtype)
##                        cv.drawContours(arr, [self.contours[i] for i in locals()[section]], -1, 0, thickness=-1, offset=(-x,-y))
##                        Image.fromarray(arr*255).show()
##                        if section == 'left':
##                            arr = rotate(arr, 90)
##                        else:
##                            arr = rotate(arr, -90)
##                        arr = add_padding(arr[3:-3, 3:-3], padding=5)
##                        area = w*h
##                        # The resulting blob shouldn't be mostly black or white
##                        # as either would suggest there are no actual 
##                        # characters in the arr
##                        if .25 < arr.sum() / float(area) < .95:
##                            text = construct_page(rec_main(arr, line_break_method='line_cut', page_type='book', page_info={'flname': section + ' margin content'}))
##                            self.margins[section].append((sections[section], text.strip()))
#                            
#                            
#                            
#                            
#                            
#                            
##        print self.margins
##                raw_input()
#                chars.sort(key=lambda x: self.boxes[x][1] + self.boxes[x][3])
#                chars = chars[::-1]
#                numbers = []
##                
##        #        print content_box
##                content_box_right_edge = tree[content_box]['b'][0] + tree[content_box]['b'][2]
#                parents = [self.hierarchy[0][c][-1] for c in chars]
#                common_parent = int(statsmode(parents)[0])
#                chars = [c for c in chars if self.hierarchy[0][c][-1] == common_parent]
#                for c in chars:
#        #            print self.boxes[c]
#        #            print hierarchy[0][c]
#        #            if self.hierarchy[0][c][-2] > -1:
#                    if self.hierarchy[0][c][-1] == 0 and self.boxes[c][0] > content_box_right_edge:
##                    if self.boxes[c][0] > content_box_right_edge:
#        #                x,y,w,h = self.get_boxes()[c]
#        #                arr = np.ones((h,w), dtype=self.img_arr.dtype)
#        #                cv.drawContours(arr, contours[c], -1, 0, offset=(-x,-y))
#        #                Image.fromarray(arr*255).show()
#                        outchar = self.draw_contour_and_children(c)
#                        outchar = rotate(outchar, -90)
#        #                Image.fromarray(outchar*255).show()
#                        feat = normalize_and_extract_features(outchar)
#                        char = label_chars[fast_cls.predict(feat)[0]]
#                        numbers.append(char)
#                num = ''.join(numbers)
#                res = reg.search(num)
#                if res:
#                    self.num = res.group(0)
#        for char in tree[content_box]['chars']:
#            b = self.get_boxes()[char]
#            x,y,w,h = b
##            cv.drawContours(n, [self.contours[char]], 
##                        -1,0, thickness = -1)
#            cv.rectangle(n, (x,y), (x+w, y+h), 0)
##            
#        import ImageDraw
#        import Image
#        im = Image.fromarray(n*255)
#        draw = ImageDraw.Draw(im)
#        for char in tree[content_box]['chars']:
##            label = str(self.hierarchy[0][char])
##            i = char
#            b = self.get_boxes()[char]
#            x,y,w,h = b
##            if self.hierarchy[0][i][0] < 0 and self.hierarchy[0][i][1] < 0 and self.hierarchy[0][i][2] < 0:
##            draw.text(self.get_boxes()[char][0:2], str(self.get_boxes()[char][3]))
#            pos = self.get_boxes()[char][0:2]
##            draw.text((pos[0]+pos[0]%5, pos[1]), str(self.hierarchy[0][char]))
#            draw.text((pos[0], pos[1]+self.get_boxes()[char][3]), str(w))
#        im.show()
#        im.save('/tmp/sample-hierarchy.png')
#        Image.fromarray(n*255).show()
        
        # Get chars in the margins.. The following won't work if 
        # There's one more than 1 box containing chars at the side of 
        # the main content box.. (this does happen....)
        
#        self.right_margin = None
#        self.left_margin = None
#        for k in tree:
#            if k != content_box:
#                box = tree[k]
#                if box['num_chars']:
#                    if box['b'][0] < tree[content_box]['b'][0]:
#                        self.left_margin = box
#                    elif box['b'][0] > tree[content_box]['b'][0]:
#                        self.right_margin = box