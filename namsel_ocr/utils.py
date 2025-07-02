from numpy import vstack, hstack, ones
import numpy as np
from cv2 import findContours, boundingRect, RETR_TREE, CHAIN_APPROX_SIMPLE, resize, INTER_CUBIC
import os

from numpy.random import binomial
import uuid
from random import choice

interp = INTER_CUBIC

urlsafechars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
def random_seq(length=15):
    return ''.join([choice(urlsafechars) for i in range(length)])

def normalize(a):
    '''Normalize raw character image array data into 32x32 matrix with an
    aspect ratio equal to the sqrt of the original aspect ratio.
    
    Parameters:
    ----------
    
    a: numpy 2d array
    
    Returns:
    --------
    Normalized 2d numpy array
    '''
    a = a.astype(np.uint8)
    h, w = a.shape
    h = float(h)
    w = float(w)
    L = 32
    sm = np.argmin([h,w])
    bg = np.argmax([h,w])
    R1 = [h,w][sm]/[h,w][bg]

    R2 = np.sqrt(R1) 

        
    if sm == 0:
        H2 = L*R2
        W2 = L
    else:
        H2 = L
        W2 = L*R2
    
    alpha = W2 / w
    beta = H2 / h

    a = resize(a, (0,0), fy=beta, fx=alpha, interpolation=interp)

    smn = a.shape[sm]
    offset = np.floor((L - smn) / 2.)
    c = np.ones((L,L), dtype=np.uint8)

    if (L - smn) % 2 == 1:
        start = offset+1
        end = offset
    else:
        start = end = offset
        
    if sm == 0:
#            print c[start:L-end, :].shape, a.shape
        c[start:L-end, :] = a
    else:
#            print c[:,start:L-end].shape, a.shape
        c[:,start:L-end] = a

    return c


def check_for_overlap(box1, box2, thresh = .77):
    if box1[0] == -1 or box2[0] == -1:
        return False
    
    x,y,w,h = box1[:4]
    xx,yy,ww,hh = box2[:4]
    r = x + w
    rr = xx + ww
    overlap = float(max(rr,r) - min(x, xx) - abs(rr-r) - abs(xx-x))/float(min(w, ww))
    if overlap >= thresh:
        return True
    return False

def add_padding(arr, padding=3):
    '''Add padding to an array to avoid problems with contour extraction
    including the image edges as a contour.
    
    Arguments: arr - the array to be padded, padding - padding amount in pixels
    '''
    
    arr = vstack((ones((padding, arr.shape[1]), dtype=arr.dtype), arr))
    arr = vstack((arr, ones((padding, arr.shape[1]), dtype=arr.dtype)))
    arr = hstack((ones((arr.shape[0],padding), dtype=arr.dtype), arr))
    arr = hstack((arr, ones((arr.shape[0],padding), dtype=arr.dtype)))
    return arr

def trim(arr, sides='trbl', new_offset=False):
    '''Remove empty white space from the edges of a matrix
    '''
    top=0
    bottom = len(arr)-1
    left = 0
    right = arr.shape[1]
    offset = {'top':0, 'bottom':0, 'right':0, 'left':0}
    if 't' in sides:
        for i, row in enumerate(arr):
            if not row.all():
                top = i
                offset['top'] = i
                break
    if 'b' in sides:
        for i in range(bottom, 0, -1):
            if not arr[i].all():
                offset['bottom'] = -(bottom-i)
                bottom = i
                break
    
    if 'l' in sides:
        for i, row in enumerate(arr.transpose()):
            if not row.all():
                left = i
                offset['left'] = i
                break
    
    if 'r' in sides:
        for i in range(right-1, 0, -1):
            if not arr.transpose()[i].all():
                offset['right'] = -(right-i)
                right = i
                break
    
#    print bottom, top, left, right
    if not new_offset:
        return arr[top:bottom, left:right]
    else:
        return arr[top:bottom, left:right], offset


def local_file(local_file_name):
    return os.path.join(os.path.dirname(__file__), local_file_name)

def invert_bw(arr):
    '''
    Invert black and white
     
    '''
    arr = arr.copy()
     
    return ((arr*-1)+1).astype(np.uint8)

def create_unique_id():
    return str(uuid.uuid4())

def clear_area_in_boxes(arr, boxes):
    for b in boxes:
        x,y,w,h = b
        arr[y:y+h,x:x+w] = 1
    return arr

def remove_small_contours(arr, wthresh=5, hthresh=5):
    area = arr.size
    arr = add_padding(arr)
    
    # Handle different OpenCV versions that return different numbers of values
    contour_result = findContours(arr.copy(), mode=RETR_TREE, method=CHAIN_APPROX_SIMPLE)
    if len(contour_result) == 2:
        contours, hier = contour_result
    else:
        _, contours, hier = contour_result
    rects = [boundingRect(c) for c in contours]
    
    for rect in rects:
        if rect[2] <= wthresh and rect[3] <= hthresh:
            arr[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]] = 1
    return arr


