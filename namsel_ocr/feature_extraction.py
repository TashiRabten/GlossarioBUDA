import pickle as pickle
from cv2 import GaussianBlur
from cv2 import HuMoments, moments, GaussianBlur
try:
    from .fast_utils import fnormalize, scale_transform
except ImportError:
    from fast_utils import fnormalize, scale_transform
import numpy as np
import joblib
from sklearn.mixture import GaussianMixture  # Updated import
try:
    from .sobel_features import sobel_features
    from .transitions import transition_features
    from .zernike_moments import zernike_features
except ImportError:
    from sobel_features import sobel_features
    from transitions import transition_features
    from zernike_moments import zernike_features
import os
try:
    from .utils import local_file
except ImportError:
    from utils import local_file
import platform

SCALER_PATH = 'zernike_scaler-latest'
scaler_full_path = local_file(SCALER_PATH)
if os.path.exists(scaler_full_path):
    scaler = joblib.load(scaler_full_path)
    transform = scaler.transform
    try:
        sc_o_std = 1.0/scaler.scale_
    except AttributeError:
        sc_o_std = 1.0/scaler.std_
    sc_mean = scaler.mean_
    SCALER_DEFINED = True
else:
    SCALER_DEFINED = False

FEAT_SIZE = 346
hstack = np.hstack

NORM_SIZE = 32
ARR_SHAPE = (NORM_SIZE, NORM_SIZE)
x3 = np.empty(NORM_SIZE*2, dtype=np.uint8)
newarr = np.empty(ARR_SHAPE, dtype=np.uint8)

magnitude = np.empty(ARR_SHAPE, np.double)
direction = np.empty(ARR_SHAPE, np.double)
sx = np.empty(ARR_SHAPE, np.double)
sy = np.empty(ARR_SHAPE, np.double)
# Use np.intp consistently for Cython compatibility
# This matches the DTYPE_t definition in sobel_features.pyx
x2 = np.zeros((192), dtype=np.intp)

# Suppress NumPy warnings during feature loading
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        if platform.system() == "Windows":
            D = pickle.load(open(local_file(r'features\D_matrix.pkl'), 'rb'))
            Bpqk = pickle.load(open(local_file(r'features\Bpqk17.pkl'), 'rb'))
            Ipi = pickle.load(open(local_file(r'features\Ipi32.pkl'), 'rb'))
        else:
            D = pickle.load(open(local_file('features/D_matrix.pkl'), 'rb'))
            Bpqk = pickle.load(open(local_file('features/Bpqk17.pkl'), 'rb'))
            Ipi = pickle.load(open(local_file('features/Ipi32.pkl'), 'rb'))
        # print("[DEBUG] Successfully loaded all feature files")
    except Exception as e:
        print(f"Warning: Could not load feature files: {e}")
        # Create dummy data with correct dimensions
        D = np.eye(100)
        Bpqk = np.zeros((18, 18, 32))
        Ipi = np.zeros((32, 32))

Ipi = np.array(Ipi, Ipi.dtype, order='F')
deg = 17
Mpqs = np.zeros((deg+1, deg+1), np.double, order='F')
Rpq = np.empty((deg+1, deg+1), complex)
ws = np.array([1, -1j, -1, 1j], complex)
Zpq = np.empty((90), np.double)
Yiq = np.zeros((deg+1, NORM_SIZE), np.double, order='F')


def normalize_and_extract_features(arr, debug_source=None, debug_char_idx=None, debug_position=None):
    global newarr, x3, Zpq
    
    try:
        # Log input properties for debugging
        if hasattr(arr, 'shape'):
            # Input has shape attribute, proceed with processing
            pass
        else:
            # Input doesn't have shape attribute, return None
            return None
        
        newarr = newarr.astype(np.uint8)
        
        # Call fnormalize with error handling
        try:
            fnormalize(arr, newarr)
            #print(f"[DEBUG] fnormalize: SUCCESS, output shape={newarr.shape}, min={newarr.min()}, max={newarr.max()}")
            
            # DEBUG: Save normalized image with organized naming
            from PIL import Image
            try:
                # Check the range and convert properly
                print(f"[FEATURE DEBUG] newarr shape: {newarr.shape}, dtype: {newarr.dtype}, min: {newarr.min()}, max: {newarr.max()}")
                
                # Ensure proper conversion to visible range
                if newarr.max() <= 1.0:
                    # If values are in 0-1 range, scale to 0-255
                    debug_img = (newarr * 255).astype(np.uint8)
                elif newarr.max() <= 255:
                    # Values already in 0-255 range
                    debug_img = newarr.astype(np.uint8)
                else:
                    # Scale down if values are too high
                    debug_img = ((newarr / newarr.max()) * 255).astype(np.uint8)
                
                print(f"[FEATURE DEBUG] debug_img min: {debug_img.min()}, max: {debug_img.max()}")
                
                # Create organized filename
                if debug_source and debug_char_idx is not None:
                    # Extract just the filename without path and extension
                    import os
                    source_name = os.path.splitext(os.path.basename(debug_source))[0]
                    if debug_position:
                        filename = f"debug_normalized_{source_name}_char{debug_char_idx:02d}_{debug_position}.png"
                    else:
                        filename = f"debug_normalized_{source_name}_char{debug_char_idx:02d}.png"
                else:
                    # Fallback to random ID if debug info not provided
                    import random
                    debug_id = random.randint(1000, 9999)
                    filename = f"debug_normalized_{debug_id}.png"
                
                Image.fromarray(debug_img).save(filename)
                print(f"[FEATURE DEBUG] Saved normalized image: {filename}")
            except Exception as e:
                print(f"[FEATURE DEBUG] Failed to save debug image: {e}")
                pass  # Don't fail feature extraction for debug issues
                
        except Exception as e:
            #print(f"[DEBUG] fnormalize: FAILED with error: {e}")
            # Fallback: copy arr to newarr directly
            if arr.shape == newarr.shape:
                newarr[:] = arr.astype(np.uint8)
            else:
                #print(f"[DEBUG] Shape mismatch for fallback: arr.shape={arr.shape}, newarr.shape={newarr.shape}")
                return None
        
        # Call extract_features with validation
        result = extract_features(newarr)
        
        if result is None:
            #print(f"[DEBUG] extract_features returned None")
            return None
        elif hasattr(result, 'shape'):
           # print(f"[DEBUG] extract_features: SUCCESS, result shape={result.shape}, non-zero={np.count_nonzero(result)}")
            if result.shape[0] != FEAT_SIZE:
                print(f"[DEBUG] WARNING: Feature vector has wrong size {result.shape[0]}, expected {FEAT_SIZE}")
        else:
            print(f"[DEBUG] extract_features: Unexpected result type={type(result)}")
            return None
            
        return result
        
    except Exception as e:
        print(f"[DEBUG] normalize_and_extract_features: FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def extract_features(arr, scale=True):  # Re-enable scaling but need to fix corruption
    global x3, Zpq
    # print(f"[DEBUG] extract_features: Input array shape={arr.shape}, dtype={arr.dtype}")
    
    # Transition features
    transition_features(arr, x3)
    #print(f"[DEBUG] Transition features: x3 shape={x3.shape}, non-zero count={np.count_nonzero(x3)}")
    
    arr = arr.astype(np.double)
    Yiq.fill(0.0)
    
    # Zernike features
    try:
        zernike_features(arr, D, Bpqk, Ipi, Mpqs, Rpq, Yiq, ws, Zpq)
        #print(f"[DEBUG] Zernike features: SUCCESS, Zpq shape={Zpq.shape}, non-zero count={np.count_nonzero(Zpq)}")
    except ValueError as e:
        # Handle dimension mismatch by filling with zeros
        # Only print warning occasionally to avoid spam
        if hasattr(zernike_features, 'warning_count'):
            zernike_features.warning_count += 1
        else:
            zernike_features.warning_count = 1
        if zernike_features.warning_count <= 5:
            print(f"Warning: Zernike feature extraction failed: {e}")
        Zpq.fill(0.0)
        print(f"[DEBUG] Zernike features: FAILED, filled with zeros")
    
    # Sobel features  
    GaussianBlur(arr, ksize=(5, 5), sigmaX=1, dst=newarr)
    x2.fill(0)
    try:
        sobel_features(arr, magnitude, direction, sx, sy, x2)
        #print(f"[DEBUG] Sobel features: SUCCESS, x2 shape={x2.shape}, non-zero count={np.count_nonzero(x2)}")
    except (ValueError, TypeError) as e:
        # Handle dtype mismatch by recreating x2 with correct dtype and retrying
        if hasattr(sobel_features, 'warning_count'):
            sobel_features.warning_count += 1
        else:
            sobel_features.warning_count = 1
        if sobel_features.warning_count <= 3:
            print(f"Warning: Sobel feature extraction failed: {e}")
            
        # Try alternative dtypes without global (x2 is already module-level)
        # Note: np.long was removed in NumPy 1.20+, use int instead
        alt_dtypes = [np.int64, np.int32, np.intp]
        if hasattr(np, 'long'):
            alt_dtypes.append(np.long)
        else:
            alt_dtypes.append(int)  # Fallback to built-in int
        
        for alt_dtype in alt_dtypes:
            try:
                x2_temp = np.zeros((192), dtype=alt_dtype)
                sobel_features(arr, magnitude, direction, sx, sy, x2_temp)
               # print(f"[DEBUG] Sobel features: SUCCESS with {alt_dtype}, x2 shape={x2_temp.shape}, non-zero count={np.count_nonzero(x2_temp)}")
                # Replace the global x2 with the working version
                x2[:] = x2_temp[:]
                break
            except (ValueError, TypeError):
                continue
        else:
            # If all dtypes fail, fill with zeros
            x2.fill(0.0)
            print(f"[DEBUG] Sobel features: FAILED with all dtypes, filled with zeros")
    
    # Combine features
    x1 = hstack((Zpq, x2, x3))
    #print(f"[DEBUG] Combined features: x1 shape={x1.shape}, non-zero count={np.count_nonzero(x1)}")
    
    if scale:
        if not SCALER_DEFINED:
            # Only print warning occasionally to avoid spam
            if not hasattr(extract_features, 'scaler_warning_count'):
                extract_features.scaler_warning_count = 0
            if extract_features.scaler_warning_count < 5:
                print("Warning: Scaler not defined, skipping scaling")
                extract_features.scaler_warning_count += 1
            # Skip scaling instead of raising error
        else:
           # print(f"[DEBUG] Before scaling: non-zero count={np.count_nonzero(x1)}")
            # CRITICAL FIX: Check dimension compatibility before scaling
            if len(x1) != len(sc_mean):
                print(f"[DEBUG] DIMENSION MISMATCH: features={len(x1)}, scaler={len(sc_mean)}")
                print(f"[DEBUG] Skipping scaling due to dimension mismatch")
            else:
                scale_transform(x1, sc_mean, sc_o_std, FEAT_SIZE)
                #print(f"[DEBUG] After scaling: non-zero count={np.count_nonzero(x1)}")
    
    #print(f"[DEBUG] Final features: shape={x1.shape}, non-zero count={np.count_nonzero(x1)}")
    return x1


def invert_binary_image(arr):
    '''
    Invert a binary image so that zero-pixels are considered as background.
    This is assumed by various functions in OpenCV and other libraries.
    
    Parameters:
    -----------
    arr: 2D numpy array containing only 1s and 0s
    
    Returns:
    --------
    2d inverted array
    '''
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr, np.uint8)  # Ensure input is a numpy array
    
    if np.max(arr) == 255:
        return (arr / -255) + 1
    else:
        return (arr * -1) + 1


def get_zernike_moments(arr):
    if arr.shape != (32, 32):
        arr.shape = (32, 32)
    
    zernike_features(arr, D, Bpqk, Ipi, Mpqs, Rpq, Yiq, ws, Zpq)
    return Zpq


def get_hu_moments(arr):
    arr = invert_binary_image(arr)
    if arr.shape != (32, 32):
        arr.shape = (32, 32)
    m = moments(arr.astype(np.float64), binaryImage=True)
    hu = HuMoments(m)
    return hu.flatten()
