# encoding: utf-8
try:
    from .classify import load_cls, label_chars
    from .fast_utils import fnormalize, ftrim, gausslogprob, fadd_padding
    from .feature_extraction import normalize_and_extract_features
    from .page_elements2 import PageElements
except ImportError:
    from classify import load_cls, label_chars
    from fast_utils import fnormalize, ftrim, gausslogprob, fadd_padding
    from feature_extraction import normalize_and_extract_features
    from page_elements2 import PageElements
import cv2 as cv
from numpy import floor, uint8, ones, argmax, hstack, mean, std, \
    ceil, load
import numpy as np
from random import gauss
import joblib
from tempfile import mkdtemp
try:
    from .transitions import horizontal_transitions
    from .utils import add_padding, local_file, check_for_overlap
    from .viterbi_cython import viterbi_cython
except ImportError:
    from transitions import horizontal_transitions
    from utils import add_padding, local_file, check_for_overlap
    from viterbi_cython import viterbi_cython

import platform

cls = load_cls('logistic-cls')

## commonly called functions
GaussianBlur = cv.GaussianBlur
predict_log_proba = cls.predict_log_proba
boundingRect = cv.boundingRect
char_gaussians = PageElements.char_gaussians

trans_p = load(open(local_file('stack_bigram_logprob32.npz'),'rb'))
trans_p = trans_p[trans_p.files[0]].transpose()
start_p = load(open(local_file('stack_start_logprob32.npz'),'rb'))
start_p = start_p[start_p.files[0]]
n_states = trans_p.shape[0]


def combine_many_boxes(bxs):
    '''Return the largest bounding box using max height and width from all boxes

    bxs is a list of boxes

    returns (x,y,w,h) of the new box
    '''

    if not bxs:
        raise ValueError('No boxes specified')

    new_top = min([b[1] for b in bxs])
    new_bottom = max([b[1]+b[3] for b in bxs])
    new_left = min([b[0] for b in bxs])
    new_right = max([b[0]+b[2] for b in bxs])
    return (new_left, new_top, new_right-new_left, new_bottom-new_top)

def box_attrs(b):
    led = b[0] # left edge
    red = b[0] + b[2] # right edge
    top = b[1]
    bottom = b[1] + b[3]
    return led, red, top, bottom, b[2], b[3]

#@profile
def image_to_32x16_vector(im):
    a = ones((32,16), dtype=uint8)
    h,w = im.shape
    s = min(32.0/h, 16.0/w)
    b = cv.resize(im, (0,0), fy=s, fx=s, interpolation=cv.INTER_AREA)
    a[0:b.shape[0], 0:b.shape[1]] = b
    return a.flatten()


def normalized_scale(arr):
    crss_denom = float(max(horizontal_transitions(arr)))
    if crss_denom > 0:
        crossings_val = (arr.shape[1])/crss_denom
        scale =  35.0 / crossings_val
    else:
        scale = 1.0

    return scale

class CombineBoxesForPage(object):
    def __init__(self, line_info):
        self.widths = []
        self.final_boxes = []
        self.final_indices = []
        self.line_width_means = []
        self.larger_font_lines = []
        self.line_info = line_info
        self.transitions = []
        self.hangoff = line_info.shapes.conf['combine_hangoff']

        for i in range(self.line_info.k):
            initial_boxes = len(line_info.lines_chars[i]) if i < len(line_info.lines_chars) else 0
            initial_chars = line_info.lines_chars[i] if i < len(line_info.lines_chars) else []
            print(f"[DEBUG] Line {i}: Starting with {initial_boxes} boxes before combination: {initial_chars}")
            self.combine_for_line_ind(self.line_info, lineind=i)
            final_boxes = len(self.final_boxes[i]) if i < len(self.final_boxes) else 0
            print(f"[DEBUG] Line {i}: {final_boxes} boxes after combination")


        ###########IMPORTANT USE THIS NORMALLY

        self.char_mean = mean(self.widths)
        self.char_std = std(self.widths)
        self.tsek_mean = line_info.shapes.tsek_mean
        self.tsek_std = line_info.shapes.tsek_std

    ###########################################

    def _low_ink_sort(self, line_info, x):
        if len(x) == 2:
            return x[0][0]
        else:
            return line_info.get_box(x)[0]


    def li_combine_for_line_ind(self, line_info, lineind=None):
        fli = [] # Final Line indices
        flb = []
        line_widths = []

        line = line_info.lines_chars[lineind]

        lib = self.line_info.low_ink_boxes[lineind]

        shapes = line_info.shapes

        if line:
            top = 1000000 # arbitrary high number
            bottom = 0
            for k in line:
                j = line_info.get_box(k)
                if j[1] < top:
                    top = j[1]
                if j[1] + j[3] > bottom:
                    bottom = j[1] + j[3]

            firstbox = line_info.get_box(line[0])
            lastbox = line_info.get_box(line[-1])
            whitespace = []
            for p, c in enumerate(line):
                if p + 1 < len(line):
                    ab = line_info.get_box(c)
                    nab = line_info.get_box(line[p+1])
                    ws_diff = nab[0] - (ab[0]+ab[2])
                    #                    if ws_diff  > 10*shapes.char_mean:
                    whitespace.append(ws_diff)

            sum_whitespace = sum(whitespace)

            ln_arr = shapes.img_arr[top:bottom,firstbox[0]:lastbox[0]+lastbox[2]].copy()

            # Convert to uint8 for horizontal_transitions compatibility
            ln_arr_uint8 = (ln_arr * 255).astype(np.uint8) if ln_arr.dtype != np.uint8 else ln_arr
            crss_denom = float(max(horizontal_transitions(ln_arr_uint8)))
            if crss_denom > 0:
                crossings_val = (ln_arr.shape[1]-sum_whitespace)/crss_denom
                scale =  35.0 / crossings_val
            else:
                scale = 1.0
            self.transitions.append(scale)

        else:
            return []

        line = iter(line)
        low_ink_segmentation = {}
        not_intr = []
        for i in line:
            b = line_info.get_box(i)
            led, red, top, bottom, w, h = box_attrs(b)
            for p, bx in enumerate(lib):
                ledn, ren, topn, bottomn, wn, hn = box_attrs(bx)
                is_interior = (led >= ledn-5 and red <= ren+5) #and not bottom < topn
                if is_interior:
                    cur = low_ink_segmentation.get(p,[])
                    cur.append(i)
                    low_ink_segmentation[p] = cur
                    break
                else: continue
            else:
                not_intr.append([i])

        all_li_seg = list(low_ink_segmentation.values())
        all_li_seg.extend(not_intr)
        for j in all_li_seg:
            fli.append(j)
            x,y,w,h = combine_many_boxes([line_info.get_box(i) for i in j])
            flb.append([x,y,w,h])
            self.widths.append(w*scale)

        fliflb = list(zip(fli, flb))
        fliflb.sort(key=lambda x: x[1][0])

        fli = [i[0] for i in fliflb]
        flb = [i[1] for i in fliflb]

        self.final_indices.append(fli)
        self.final_boxes.append(flb)

    def combine_for_line_ind(self, line_info, lineind=None):
        fli = [] # Final Line indices
        flb = []
        line_widths = []
        line = line_info.lines_chars[lineind]

        line.sort(key=lambda x: line_info.get_box(x)[0])

        shapes = line_info.shapes

        # line, line info, shapes
        if line:
            top = 1000000 # arbitrary high number
            bottom = 0
            for k in line:
                j = line_info.get_box(k)
                if j[1] < top:
                    top = j[1]
                if j[1] + j[3] > bottom:
                    bottom = j[1] + j[3]

            firstbox = line_info.get_box(line[0])
            lastbox = line_info.get_box(line[-1])
            whitespace = np.zeros(len(line), dtype=int)
            for p, c in enumerate(line):
                if p + 1 < len(line):
                    ab = line_info.get_box(c)
                    nab = line_info.get_box(line[p+1])
                    ws_diff = nab[0] - (ab[0]+ab[2])
                    whitespace[p] = ws_diff

            sum_whitespace = whitespace.sum()

            ln_arr = shapes.img_arr[top:bottom,firstbox[0]:lastbox[0]+lastbox[2]].copy()

            try:
                for inx in line_info.small_cc_lines_chars[lineind]:
                    box = line_info.get_box(inx)
                    ln_arr[box[1] - top:box[1]+box[3]-top, box[0]-firstbox[0]:box[0]+box[2]-firstbox[0]] = 1
            except:
                pass
            #             Remove small contours when calculating scale
            #             for k in line_info.small_cc_lines_chars[lineind]:
            #                 x, y, w, h = shapes.get_boxes()[k]
            #                 x = x - firstbox[0]
            #                 y = y - top
            #                 ln_arr[y:y+h, x:x+w] = 1

            # Convert to uint8 for horizontal_transitions compatibility
            ln_arr_uint8 = (ln_arr * 255).astype(np.uint8) if ln_arr.dtype != np.uint8 else ln_arr
            crss_denom = float(max(horizontal_transitions(ln_arr_uint8)))
            if crss_denom > 0:
                crossings_val = (ln_arr.shape[1]-sum_whitespace )/crss_denom
                scale =  35.0 / crossings_val
            else:
                scale = 1.0

            self.transitions.append(scale)

        else:
            return []

        line = iter(line)

        # Initialize the current box and its attrs#         BREAKWIDTH = 3.0
        try:
            cur_ind = [next(line)]
        except StopIteration:
            return []

        # cb is current box, b is the next box
        cb = line_info.get_box(cur_ind[0])

        led, red, top, bottom, w, h = box_attrs(cb)




        # Loop through box, combine and close along the way
        for i in line:
            b = line_info.get_box(i)

            ledn, ren, topn, bottomn, wn, hn = box_attrs(b)

            # If left edge of next box doesn't overlap cur box
            # separate as 2 different chars
            is_interior = ((ledn >= led and ren <= red) or (ledn <= led and ren >= red)) and not bottom < topn
            if not isinstance(i, str):
                is_interior = line_info.shapes.hierarchy[0][i][0] < 0 and line_info.shapes.hierarchy[0][i][1] < 0 and line_info.shapes.hierarchy[0][i][2] < 0 # i.e. it has no peers at its place in the tree... and no children
            else:
                is_interior = False # Its a string, meaning it is the result of a horizontal cut and likely not an interior

            if ledn > red: #or (not is_interior and in_tsek_pop(shapes, wn, topn, top, bottomn, bottom, i) and not hn > 1.5*shapes.tsek_mean):
                # FIXED: Uncommented tsheg detection to preserve tshegs as separate characters for vector creation

                fli.append(cur_ind)
                x,y,w,h = combine_many_boxes([line_info.get_box(j) for j in cur_ind])
                flb.append((x,y,w,h))
                self.widths.append(w*scale)
                cur_ind = [i]
                cb = b
                led, red, top, bottom, w, h = box_attrs(cb)

            #                continue
            else: # There is overlap
                # one box is completely enveloped by the other
                if is_interior:
                    #                print 'enveloped'

                    cur_ind.append(i)
                    bxs = [line_info.get_box(j) for j in cur_ind]
                    bxs.append(b)
                    cb = combine_many_boxes(bxs)
                    led, red, top, bottom, w, h = box_attrs(cb)

                elif ((float(min(wn, w)) - abs((red - ledn))) / float(min(wn, w))) < self.hangoff: # amount hanging off end is 30 %
                    cur_ind.append(i)
                    bxs = [line_info.get_box(j) for j in cur_ind]
                    bxs.append(b)
                    cb = combine_many_boxes(bxs)
                    led, red, top, bottom, w, h = box_attrs(cb)

                # The overlap is incidental / boxes are not related
                else:
                    #                print 'incidental overlap'
                    fli.append(cur_ind)
                    x,y,w,h = combine_many_boxes([line_info.get_box(j) for j in cur_ind])
                    flb.append((x,y,w,h))
                    self.widths.append(w*scale)

                    cur_ind = [i]
                    cb = b
                    led, red, top, bottom, w, h = box_attrs(cb)
        fli.append(cur_ind)
        x,y,w,h = combine_many_boxes([line_info.get_box(j) for j in cur_ind])
        flb.append((x,y,w,h))
        line_widths.append(w)
        self.widths.append(w*scale)

        if shapes.low_ink:
            lib = self.line_info.low_ink_boxes[lineind]
            low_ink_segmentation = {}
            not_intr = []
            for d, box in enumerate(flb):
                #                b = line_info.get_box(i)
                led, red, top, bottom, w, h = box_attrs(box)
                for p, bx in enumerate(lib):
                    ledn, ren, topn, bottomn, wn, hn = box_attrs(bx)
                    is_interior = (led >= ledn-15 and red <= ren+15) #and not bottom < topn
                    if is_interior:


                        ###  This attempts to remove noise
                        ### that doesn't fall into blurred low ink
                        ### box but does get combined according to normal
                        ### combination rules
                        #                         for inx in fli[d]:
                        #                             tb = line_info.get_box(inx)
                        #                             if tb[1] >= topn and tb[1] + tb[3] <= bottomn:
                        #                                 cur = low_ink_segmentation.get(p,[])
                        # #                                 cur.extend(fli[d])
                        #                                 cur.append(inx)
                        #                                 low_ink_segmentation[p] = cur


                        cur = low_ink_segmentation.get(p,[])
                        cur.extend(fli[d])
                        low_ink_segmentation[p] = cur
                        break
                    else: continue
                else:
                    #                print 'fail'
                    not_intr.append(fli[d])
                    #NO! do something


            all_li_seg = list(low_ink_segmentation.values())
            all_li_seg.extend(not_intr)
            newfli = []
            newflb = []
            for j in all_li_seg:
                newfli.append(j)
                x,y,w,h = combine_many_boxes([line_info.get_box(i) for i in j])
                newflb.append([x,y,w,h])
                self.widths.append(w*scale)

            fli = newfli
            flb = newflb

            fliflb = list(zip(fli, flb))
            fliflb.sort(key=lambda x: x[1][0])

            fli = [i[0] for i in fliflb]
            flb = [i[1] for i in fliflb]

        self.final_indices.append(fli)
        self.final_boxes.append(flb)
        self.line_width_means.append(mean(line_widths))
        if not line:
            print(flb)

def in_tsek_pop(shapes, width, topn, top, bottomn, baseline,cur_ind):
    '''Determine whether a box is of approx tsek-width'''
    if shapes.tsek_mean - 3*shapes.tsek_std <= width <= shapes.tsek_mean+shapes.tsek_std and topn - .5*shapes.tsek_std <= baseline <= bottomn + .5*shapes.tsek_std:# and h>=0:

        return True
    else:
        return False

class Segmenter(object):
    def __init__(self, line_info, break_resolution = 6, draw_outlines=True):
        self.line_info = line_info
        self.draw_outlines = draw_outlines
        self.break_window_resolution = break_resolution
        self.breakwidth = line_info.shapes.conf['break_width']
        self.cached_features = line_info.shapes.cached_features

        segmenter_type = line_info.shapes.conf['segmenter']
        print(f"[SEGMENTER DEBUG] Using segmenter type: {segmenter_type}")
        
        if segmenter_type == 'experimental':
            print(f"[SEGMENTER DEBUG] Calling construct_vector_set_experimental()")
            self.construct_vector_set_experimental()
        elif segmenter_type == 'stochastic':
            print(f"[SEGMENTER DEBUG] Calling construct_vector_set_stochastic()")
            self.construct_vector_set_stochastic()
        elif segmenter_type == 'simple':
            print(f"[SEGMENTER DEBUG] Calling construct_vector_set_simple()")
            self.construct_vector_set_simple()
        else:
            print(f"[SEGMENTER DEBUG] Unknown segmenter type: {segmenter_type}, defaulting to simple")
            self.construct_vector_set_simple()


    def _min_variance_breakwidth(self):
        widths = self.final_box_info.widths
        char_mean = self.final_box_info.char_mean
        char_std = self.final_box_info.char_std
        ws = [1.75, 2.5, 2.75, 3.0, 3.6, 4.0, 8.0]

        new_widths = [[] for i in range(len(ws))]
        for wd in widths:
            for i, w in enumerate(ws):
                if wd >= char_mean + w*char_std :

                    splits = int(floor(float(wd)/(char_mean-char_std)))
                    for u in range(splits):
                        new_widths[i].append(char_mean)
                    else:
                        new_widths[i].append(wd)
                else:
                    new_widths[i].append(wd)

        best_var_arg = np.argmin([np.var(wnews) for wnews in new_widths])
        return ws[best_var_arg]



    def _sample_widths_method(self, chars, letter, letter_box, oo_scale_l, line_num=None):
        x, y, w, h = letter_box

        ################default
        cur_mean = self.final_box_info.char_mean*.97
        cur_std = .295*self.final_box_info.char_std
        print(f"[SEGMENT DEBUG] Breaking {chars} chars: cur_mean={cur_mean:.1f}, cur_std={cur_std:.1f}, box_width={w}")
        #################
        best_prob = -np.inf

        if chars > 1:
            letter = cv.dilate(letter.copy(), None, iterations=1)

            padding_amount = 3

            for n in range(15):

                widths = [gauss(cur_mean, cur_std) for i in range(chars)]
                prev = 0
                vecs = []
                wdthprobs = 0
                boxes = []
                for i, val in enumerate(widths):
                    if i == chars - 1:
                        end = letter.shape[1]
                    else:
                        end = prev+val
                    wdthprobs += gausslogprob(cur_mean, cur_std, end-prev)

                    s = fadd_padding(letter[:,int(prev):int(end)], padding_amount)

                    # OpenCV version-agnostic approach for findContours
                    # OpenCV 3.x returns (image, contours, hierarchy)
                    # OpenCV 4.x returns (contours, hierarchy)
                    contours_result = cv.findContours(s.copy(), mode=cv.RETR_TREE , method=cv.CHAIN_APPROX_NONE)
                    ctrs, hier = contours_result[-2:]  # Get last 2 values regardless of OpenCV version

                    bounding = list(map(boundingRect, ctrs))
                    for k, b in enumerate(bounding):
                        if (b[2] < 23 or b[3] < 23) and hier[0][k][3] == 0:
                            s[b[1]-1:b[1]+b[3]+1,b[0]-1:b[0]+b[2]+1] = 1
                    s = s[padding_amount:-padding_amount, padding_amount:-padding_amount]
                    s, ofst = ftrim(s, new_offset=True)

                    if 0 not in s.shape:
                        nnbox = [x+(prev + ofst['left'])*oo_scale_l, y + (ofst['top']*oo_scale_l), s.shape[1]*oo_scale_l, s.shape[0]*oo_scale_l]
                        if line_num is not None:
                            naro = self.line_info.check_naro_overlap(line_num, nnbox)
                            if naro != False:

                                naro_box = self.line_info.get_box(naro)
                                nnbox = combine_many_boxes([nnbox, naro_box])
                                ss = cv.resize(s, dsize=(0,0), fx=oo_scale_l, fy=oo_scale_l)
                                ss = np.vstack((ones((nnbox[3]-ss.shape[0], ss.shape[1]), dtype=ss.dtype), ss))
                                ss = hstack((ss,ones((ss.shape[0],nnbox[2] - ss.shape[1]), dtype=ss.dtype)))

                                cv.drawContours(ss, [self.line_info.get_contour(naro)], -1,0, thickness = -1, offset=(-naro_box[0],-naro_box[1]))
                                s = ss
                        vecs.append(normalize_and_extract_features(s))
                        boxes.append(nnbox)
                    else:
                        break
                    prev += val
                if not vecs: continue
                xn = len(vecs)

                vecs = np.array(vecs).reshape(xn, 346) # 346 is len(vecs[0])

                probs = predict_log_proba(vecs)
                probs = probs.astype(np.float32)

                if n%10 == 0 and n != 0:

                    cur_mean = self.final_box_info.char_mean*(.97-(3*n/1000.0))

                prob, prds = viterbi_cython(xn, n_states, start_p, trans_p, probs)
                prob = prob + wdthprobs
                if prob > best_prob:
                    best_prob = prob
                    best_prd = prds
                    best_boxes = boxes
        else:
            best_boxes = [letter_box]
            features = normalize_and_extract_features(letter)
            probs = predict_log_proba(features.reshape(1, -1))
            amx = probs[0].argmax()
            try:
                startprob = start_p[amx]
            except IndexError:
                startprob = 1e-10
            best_prob = probs[0][amx] + gausslogprob(cur_mean, cur_std, letter_box[2]/oo_scale_l) + startprob
            best_prd = [amx]

        final_prob = best_prob
        res = []
        for i, val in enumerate(best_prd):
            best_boxes[i] = [int(np.round(k)) for k in best_boxes[i]]
            best_boxes[i].extend([float(np.exp(final_prob)),label_chars[val]])
            res.append(best_boxes[i])

        return (final_prob, res)


    def _detach_tsek(self, letter):
        # 1. check if detach makes sense: i.e. will chopping off end result in
        # something that looks and acts like a tsek, in size and position
        # 2. isolate the tsek-part, create a new bounding box for it
        # update the parent box with new dimensions

        #         tsek_part = letter[:, letter.shape[1]-tsek_mean:]


        pass

    def construct_vector_set_stochastic(self):
        # separate attached tsek
        # note this may note go here exactly, but somewhere in this function
        if self.line_info.shapes.conf.get('detach_tsek'):
            self._detach_tsek()

        final_box_info = CombineBoxesForPage(self.line_info)

        self.final_box_info = final_box_info
        final_boxes = final_box_info.final_boxes

        final_indices = final_box_info.final_indices
        scales = final_box_info.transitions

        self.vectors = [[] for i in range(self.line_info.k)]
        self.new_boxes = [[] for i in range(self.line_info.k)] #


        BREAKWIDTH = self.breakwidth

        for l in range(len(final_indices)): # for each line
            try:
                scale_l = scales[l]
                oo_scale_l = 1.0/scale_l
            except:
                print(('ERROR AT ', l, len(scales)))
                raise
            try:
                lb = list(range(len(final_indices[l])))
            except IndexError:
                continue

            segmented = 0
            for i in lb: # for each line box

                ## New draw, takes into account tree hierarchy of contours
                x, y, w, h = final_boxes[l][i]
                letter = ones((h,w), dtype=uint8)
                lindices = final_indices[l][i]
                len_lindices = len(lindices)
                for k in lindices:
                    if not isinstance(k, str):
                        letter = self.line_info.shapes.draw_contour_and_children(k, char_arr=letter, offset=(-x,-y))
                    else:
                        cv.drawContours(letter, [self.line_info.get_contour(k)], -1,0, thickness = -1, offset=(-x,-y))

                if w*scale_l >= 1 and h*scale_l >= 1:
                    letter = cv.resize(letter, dsize=(0,0), fx=scale_l, fy=scale_l)
                
                # DEBUG: Save the draw_contour_and_children result to see what vectors are using
                try:
                    from PIL import Image
                    debug_letter = letter.astype(np.uint8)
                    if debug_letter.max() <= 1:
                        debug_letter = (debug_letter * 255).astype(np.uint8)
                    Image.fromarray(debug_letter).save(f"debug_contour_letter_line{l}_char{i}_{x}_{y}_{w}x{h}.png")
                    print(f"[CONTOUR DEBUG] Saved contour letter: debug_contour_letter_line{l}_char{i}_{x}_{y}_{w}x{h}.png")
                except Exception as e:
                    print(f"[CONTOUR DEBUG] Failed to save contour letter: {e}")
                    pass

                # Check if character is too large and needs segmentation
                # Use more conservative threshold to avoid over-segmentation
                conservative_breakpoint = final_box_info.char_mean + (BREAKWIDTH + 0.5)*final_box_info.char_std
                print(f"[SEGMENT DEBUG] Character analysis: width={letter.shape[1]:.1f}, conservative_breakpoint={conservative_breakpoint:.1f}")
                
                if letter.shape[1] >= conservative_breakpoint: # if a box is too large, break it
                    sw = w*scale_l
                    sh = h*scale_l
                    chars = sw // (final_box_info.char_mean - 1.5*final_box_info.char_std)# important, floor division
                    chars = min(chars, 4)
                    print(f"[SEGMENT DEBUG] Large character detected: sw={sw:.1f}, calculated_chars={chars}")
                    
                    # Additional check: only segment if we're confident it contains multiple characters
                    if chars > 1.5:  # More conservative - only segment if clearly > 1.5 characters


                        w = sw
                        h = sh

                        all_choices = []

                        for chars in range(int(chars),0,-1):
                            #                             if l == 1:
                            if self.line_info.shapes.detect_o:
                                line_num = l
                            else:
                                line_num = None
                            all_choices.append(self._sample_widths_method(chars, letter, final_boxes[l][i], oo_scale_l, line_num=line_num))

                        ## Append complete recognization results to vector list

                        mx = max(all_choices)
                        print(f"[LARGE CHAR DEBUG] Line {l}, char {i}: Segmenting into {len(mx[-1])} parts")
                        for v_idx, v in enumerate(mx[-1]):
                            print(f"[LARGE CHAR DEBUG] Line {l}, char {i}, part {v_idx}: appending v={v[:6]}...")
                            self.new_boxes[l].append(v)
                            self.vectors[l].append(v)
                            self.line_info.shapes.img_arr[v[1]:v[1]+v[3], v[0]+v[2]] = 1


                    else:
                        self.new_boxes[l].append([x,y, w, h])
                        if len_lindices == 1:
                            # FORCE FRESH FEATURE EXTRACTION - DISABLE CACHE
                            print(f"[VECTOR DEBUG] FORCING FRESH feature extraction for line{l}_char{i}, letter.shape={letter.shape}")
                            vect = normalize_and_extract_features(letter)
                            # try:
                            #     vect = self.cached_features[lindices[0]]
                            #     print(f"[VECTOR DEBUG] Using CACHED features for line{l}_char{i}, lindices={lindices[0]}")
                            # except: #FIXME: should really check key used
                            #     print(f"[VECTOR DEBUG] FRESH feature extraction for line{l}_char{i}, letter.shape={letter.shape}")
                            #     vect = normalize_and_extract_features(letter)
                        else:
                            print(f"[VECTOR DEBUG] MULTIPLE indices feature extraction for line{l}_char{i}, letter.shape={letter.shape}")
                            vect = normalize_and_extract_features(letter)
                        
                        if vect is not None:
                            print(f"[VECTOR DEBUG] Vector created: line{l}_char{i}, vect.shape={vect.shape if hasattr(vect, 'shape') else type(vect)}")
                        else:
                            print(f"[VECTOR DEBUG] Vector is None for line{l}_char{i}")
                        self.vectors[l].append(vect)

                else:
                    self.new_boxes[l].append([x,y, w, h])
                    if len_lindices == 1:
                        # FORCE FRESH FEATURE EXTRACTION - DISABLE CACHE
                        print(f"[VECTOR DEBUG ALT] FORCING FRESH feature extraction for line{l}_char{i}, letter.shape={letter.shape}")
                        vect = normalize_and_extract_features(letter)
                        # try:
                        #     vect = self.cached_features[lindices[0]]
                        #     print(f"[VECTOR DEBUG ALT] Using CACHED features for line{l}_char{i}, lindices={lindices[0]}")
                        # except KeyError:
                        #     print(f"[VECTOR DEBUG ALT] FRESH feature extraction for line{l}_char{i}, letter.shape={letter.shape}")
                        #     vect = normalize_and_extract_features(letter)
                    else:
                        print(f"[VECTOR DEBUG ALT] MULTIPLE indices feature extraction for line{l}_char{i}, letter.shape={letter.shape}")
                        vect = normalize_and_extract_features(letter)
                    
                    if vect is not None:
                        print(f"[VECTOR DEBUG ALT] Vector created: line{l}_char{i}, vect.shape={vect.shape if hasattr(vect, 'shape') else type(vect)}")
                    else:
                        print(f"[VECTOR DEBUG ALT] Vector is None for line{l}_char{i}")
                    self.vectors[l].append(vect)


        if not any(self.vectors):
            print('no vectors')
            return
        else:
            if self.line_info.shapes.detect_o:

                for i, line in enumerate(self.new_boxes):

                    used_boxes = set()
                    for n in self.line_info.line_naros[i]:
                        if n in used_boxes:
                            continue
                        box = self.line_info.get_box(n)
                        x,y,w,h = box
                        for k, box1 in enumerate(line):
                            assert isinstance(box1, (list, tuple)), 'error - {}-{}-{}'.format(str(box1), i, k)
                            assert isinstance(box, (list, tuple)), box
                            try:
                                overlap = check_for_overlap(box1, box)
                            except:

                                print((i, k, box1, 'BOX problem'))
                            if overlap:
                                used_boxes.add(n)

                                try:
                                    nbox = list(combine_many_boxes([box, box1]))
                                except:
                                    print((nbox, 'slkfjlkfj'))
                                    raise
                                if isinstance(self.vectors[i][k], str):
                                    self.vectors[i][k] += 'ོ'
                                    nbox = box1
                                    nbox[-1] = self.vectors[i][k]
                                elif isinstance(self.vectors[i][k], list):
                                    if not self.vectors[i][k][-1][-1] == 'ོ':
                                        pchar = self.vectors[i][k][-1] + 'ོ'
                                        self.vectors[i][k][-1] = pchar
                                    nbox = self.vectors[i][k]
                                else:
                                    probs = cls.predict_log_proba(self.vectors[i][k])
                                    mx = np.argmax(probs)
                                    prob = probs[0][mx]
                                    ch = label_chars[mx] + 'ོ'
                                    self.vectors[i][k] = ch
                                    nbox.append(prob)
                                    nbox.append(ch)
                                self.new_boxes[i][k] = nbox


    def construct_vector_set_experimental(self):

        NINF = -np.inf

        final_box_info = CombineBoxesForPage(self.line_info)

        self.final_box_info = final_box_info
        final_boxes = final_box_info.final_boxes

        final_indices = final_box_info.final_indices
        scales = final_box_info.transitions

        self.vectors = [[] for i in range(self.line_info.k)]
        self.new_boxes = [[] for i in range(self.line_info.k)] #
        cur_mean = self.final_box_info.char_mean
        cur_std = self.final_box_info.char_std
        BREAKWIDTH = self.breakwidth
        rbfcls = self.line_info.rbfcls
        for l in range(len(final_indices)): # for each line
            try:
                scale_l = scales[l]
            except:
                print(('ERROR AT ', l, len(scales)))
                raise
            char_mean_int = floor(final_box_info.char_mean)
            char_std_int = ceil(final_box_info.char_std)

            try:
                lb = list(range(len(final_indices[l])))
            except IndexError:
                print('index error')
                continue

            segmented = 0
            for i in lb: # for each line box

                ## New draw, takes into account tree hierarchy of contours
                x, y, w, h = final_boxes[l][i]
                letter = ones((h,w), dtype=uint8)
                for k in final_indices[l][i]:
                    if not isinstance(k, str):
                        letter = self.line_info.shapes.draw_contour_and_children(k, char_arr=letter, offset=(-x,-y))
                    else:
                        cv.drawContours(letter, [self.line_info.get_contour(k)], -1,0, thickness = -1, offset=(-x,-y))

                # DEBUG: Save the contour-drawn image BEFORE scaling
                print(f"[CONTOUR DEBUG] Line {l}, char {i}: contours={final_indices[l][i]}, box=({x},{y},{w},{h})")
                print(f"[CONTOUR DEBUG] letter.shape={letter.shape}, dtype={letter.dtype}, min={letter.min()}, max={letter.max()}")
                try:
                    from PIL import Image
                    # Convert to visible format for saving
                    if letter.max() <= 1:
                        debug_img = (letter * 255).astype(np.uint8)
                    else:
                        debug_img = letter.astype(np.uint8)
                    
                    Image.fromarray(debug_img).save(f"debug_vector_char_L{l}_C{i}_contours_{'-'.join(map(str, final_indices[l][i]))}.png")
                    print(f"[CONTOUR DEBUG] Saved: debug_vector_char_L{l}_C{i}_contours_{'-'.join(map(str, final_indices[l][i]))}.png")
                except Exception as e:
                    print(f"[CONTOUR DEBUG] Failed to save: {e}")

                letter = cv.resize(letter, dsize=(0,0), fx=scale_l, fy=scale_l)
                
                # DEBUG: Save the contour-drawn image AFTER scaling
                try:
                    if letter.max() <= 1:
                        debug_img_scaled = (letter * 255).astype(np.uint8)
                    else:
                        debug_img_scaled = letter.astype(np.uint8)
                    
                    Image.fromarray(debug_img_scaled).save(f"debug_vector_char_L{l}_C{i}_SCALED_contours_{'-'.join(map(str, final_indices[l][i]))}.png")
                    print(f"[CONTOUR DEBUG] Saved scaled: debug_vector_char_L{l}_C{i}_SCALED_contours_{'-'.join(map(str, final_indices[l][i]))}.png")
                except Exception as e:
                    print(f"[CONTOUR DEBUG] Failed to save scaled: {e}")
                if letter.shape[1] >= (final_box_info.char_mean + BREAKWIDTH*final_box_info.char_std): # if a box is too large, break it
                    #
                    segmented += 1
                    sw = w*scale_l
                    sh = h*scale_l
                    vsum = letter.sum(axis=0)
                    chars = sw // (final_box_info.char_mean - 1.5*final_box_info.char_std)# important, floor division

                    if 10.0 > chars > 1.0: # Assume chars-to-be-broken don't span > 10
                        #                     if chars:
                        w = sw
                        h = sh

                        best_box_dim = []
                        best_prob = 0.0
                        best_seq = None
                        ## Iterate through a range of variable chars if
                        ## chars is greater than 2. This allows potential
                        ## breaks for chars-1, chars-2
                        #                         all_choices = []

                        for chars in range(int(chars),1,-1):

                            for z in range(0,21,2):
                                segs = []
                                prev_breakline = 0
                                for pos in range(int(chars-1)):
                                    if char_mean_int - z >= 0:

                                        upper_range = [int(np.round((pos+1)*(char_mean_int-z))), int(np.round((pos+1)*(char_mean_int+z)))]
                                        vsum_range = vsum[upper_range[0]:upper_range[1]]

                                        if vsum_range.any():
                                            breakline = int(np.round((pos+1)*(char_mean_int-z) + argmax(vsum_range)))
                                        else:
                                            breakline = None

                                        if breakline:
                                            sg = letter[:,prev_breakline:breakline]

                                            prev_breakline = breakline
                                        else:
                                            sg = letter[:,int(np.round(pos*(char_mean_int-z))):int(np.round((pos+1)*(char_mean_int-z)))]
                                            prev_breakline = int(np.round((pos+1)*(char_mean_int-z)))

                                        segs.append(sg)

                                segs.append(letter[:,int(np.round((chars-1)*(char_mean_int-z))):])



                                segs = [fadd_padding(sg, 2) for sg in segs]
                                # OpenCV version-agnostic approach for list comprehension
                                seg_ctrs = [cv.findContours(sg.copy(), mode=cv.RETR_CCOMP, method=cv.CHAIN_APPROX_SIMPLE)[-2:] for sg in segs]
                                try:
                                    seg_bxs = [[cv.boundingRect(k) for k in sgc[0]] for sgc in seg_ctrs]
                                except:
                                    print(sgc)
                                    raise

                                bxs = []
                                nsegs = []

                                prev_w = 0
                                for zi, ltb in enumerate(seg_bxs):
                                    seg = segs[zi]
                                    for b in ltb:
                                        if b[2] < (final_box_info.tsek_mean + 4*final_box_info.tsek_std) or b[3] < final_box_info.tsek_mean + 4*final_box_info.tsek_std:
                                            seg[b[1]-1:b[1]+b[3]+1,b[0]-1:b[0]+b[2]+1] = True
                                    seg, ofst = ftrim(seg, new_offset=True)
                                    bx = [x+prev_w+(ofst['left']/scale_l), y + (ofst['top']/scale_l), seg.shape[1]/scale_l, seg.shape[0]/scale_l]
                                    prev_w += seg.shape[1]/scale_l
                                    bxs.append(bx)
                                    nsegs.append(seg)


                                xt = [normalize_and_extract_features(sg) for sg in nsegs if 0 not in sg.shape]
                                prd_probs = cls.predict_log_proba(xt)
                                prd_probs = prd_probs.astype(np.float32)

                                prob, prds = viterbi_cython(prd_probs.shape[0], n_states, start_p, trans_p, prd_probs)
                                prob = np.exp(prob)

                                if prob > best_prob:
                                    best_prob = prob
                                    best_seq = prds
                                    best_box_dim = bxs
                                    best_xt = xt

                        if not best_box_dim:
                            best_prob = prob
                            best_seq = prds
                            best_box_dim = bxs
                            best_xt = xt

                        for u in range(len(best_seq)):
                            self.vectors[l].append(label_chars[best_seq[u]])
                            best_box = best_box_dim[u]
                            best_box = [int(np.round(ii)) for ii in best_box]
                            best_box.append(best_prob)
                            best_box.append(label_chars[best_seq[u]])
                            self.new_boxes[l].append(best_box)

                            try:
                                self.line_info.shapes.img_arr[best_box[1]:best_box[1]+best_box[3], best_box[0]+best_box[2]] = 1
                            except:

                                pass


                    else:
                        self.new_boxes[l].append([x,y, w, h])
                        vect = normalize_and_extract_features(letter)
                        self.vectors[l].append(vect)

                else:
                    self.new_boxes[l].append([x,y, w, h])
                    vect = normalize_and_extract_features(letter)
                    self.vectors[l].append(vect)

        if not any(self.vectors):
            print('no vectors')
            return
        else:
            if self.line_info.shapes.detect_o:

                for i, l in enumerate(self.new_boxes):
                    for n in self.line_info.line_naros[i]:
                        box = self.line_info.get_box(n)
                        x,y,w,h = box
                        r0 = x+w
                        for k, b in enumerate(l):
                            if ((b[2] + w) - abs(b[0] - x) - abs((b[0]+b[2]) - r0)) / (2*float(min(w, b[2]))) > .8:
                                try:
                                    nbox = list(combine_many_boxes([box, b]))
                                except:
                                    print((nbox[3]))
                                    raise
                                if isinstance(self.vectors[i][k], str):
                                    self.vectors[i][k] += 'ོ'
                                    nbox = b
                                    nbox[-1] = self.vectors[i][k]
                                else:
                                    probs = cls.predict_log_proba(self.vectors[i][k])
                                    mx = np.argmax(probs)
                                    prob = probs[0][mx]
                                    mx = rbfcls.predict(self.vectors[i][k])[0]
                                    ch = label_chars[mx] + 'ོ'
                                    self.vectors[i][k] = ch
                                    nbox.append(prob)
                                    nbox.append(ch)
                                self.new_boxes[i][k] = nbox


    def construct_vector_set_simple(self):
        self.too_big = [[] for i in range(self.line_info.k)]
        self.too_big_box = [[] for i in range(self.line_info.k)]
        self.too_big_loc = []
        char_mean = self.line_info.shapes.char_mean
        for i in range(self.line_info.k):
            line = self.line_info.lines_chars[i]
            for j, c in enumerate(line):
                x,y,w,h = self.line_info.get_box(c)
                if w > 1.75*char_mean or h > 2.5*char_mean:
                    letter = ones((h,w), dtype=uint8)
                    if not isinstance(c, str):
                        letter = self.line_info.shapes.draw_contour_and_children(c, char_arr=letter, offset=(-x,-y))
                    else:
                        cv.drawContours(letter, [self.line_info.get_contour(c)], -1,0, thickness = -1, offset=(-x,-y))

                    self.too_big[i].append(letter)
                    self.too_big_loc.append((i, j))
                    self.too_big_box[i].append([x,y,w,h])

        for loc in self.too_big_loc:
            self.line_info.lines_chars[loc[0]][loc[1]] = 'xx'

        for k in self.line_info.lines_chars:
            self.line_info.lines_chars[k] = [xx for xx in self.line_info.lines_chars[k] if xx != 'xx']

        final_box_info = CombineBoxesForPage(self.line_info)
        scales = final_box_info.transitions
        self.final_box_info = final_box_info
        final_boxes = final_box_info.final_boxes
        char_mean = self.final_box_info.char_mean
        final_indices = final_box_info.final_indices
        self.vectors = [[] for i in range(self.line_info.k)]
        self.new_boxes = [[] for i in range(self.line_info.k)] #
        for l in range(self.line_info.k): # for each line
            try:
                lb = list(range(len(final_indices[l])))
            except IndexError:
                continue

            try:
                scale_l = scales[l]
                oo_scale_l = 1.0/scale_l
            except:
                print(('ERROR AT ', l, len(scales)))
                raise

            for ii, i in enumerate(lb): # for each line box

                ## New draw, takes into account tree hierarchy of contours
                x, y, w, h = final_boxes[l][i]
                letter = ones((h,w), dtype=uint8)
                for k in final_indices[l][i]:
                    if not isinstance(k, str):
                        letter = self.line_info.shapes.draw_contour_and_children(k, char_arr=letter, offset=(-x,-y))
                    else:
                        cv.drawContours(letter, [self.line_info.get_contour(k)], -1,0, thickness = -1, offset=(-x,-y))

                letter = cv.resize(letter, dsize=(0,0), fx=scale_l, fy=scale_l)
                self.new_boxes[l].append([x,y, w, h])
                vect = normalize_and_extract_features(letter)
                self.vectors[l].append(vect)

        if not any(self.vectors):
            print('no vectors')
            return            
        
