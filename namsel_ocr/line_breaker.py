# encoding: utf-8
'''Line breaking'''
from numpy import  array, float64, argmax, argmin, uint8, ones, floor, mean, std, where, argsort
import cv2 as cv
try:
    from .utils import  check_for_overlap
    from .fast_utils import ftrim, fadd_padding
except ImportError:
    from utils import  check_for_overlap
    from fast_utils import ftrim, fadd_padding
import sys
from bisect import bisect, bisect_right
try:
    from .feature_extraction import normalize_and_extract_features
    from .classify import load_cls, label_chars
except ImportError:
    from feature_extraction import normalize_and_extract_features
    from classify import load_cls, label_chars

cls = load_cls('logistic-cls')

class LineCut(object):
    '''Line Cutting object - breaks lines in a page where lines are separated
    by empty whitespace
    
    Parameters:
    --------------------
    shapes: page_element object, (see page_elements.py)
    
    thresh_scale: float, default=.9995
        A threshold value for determining the breakline in the event that
        there is black pixel noise between lines. Should be set high to avoid
        setting line breaks through characters themselves. 
    
    Attributes:
    -----------
    lines_chars: list of lists, length=number of lines on page. Each sub-list
        contains the indices for the bounding boxes/contours assigned to
        its corresponding line.
    
    line_indices: list of int, indices of breaklines with respect to page_array
    
    baselines: list of int, the index of the baseline for each line where
        baseline here is usually a line that goes through all the thick "head"
        (Tibetan: mgo) parts found on most Tibetan letters
    
    Methods:
    --------
    
    get_box: return the bounding box for a given index
    get_contour: return the contour for a given index
    
    The get_box, get_contour methods are mostly here for API compatibility with
    LineCluster.
    '''

    def __init__(self, shapes, thresh_scale=.9995):
        self.shapes = shapes
        self.baselines = []

        ### Inflate chars to avoid vowels breaking off their lines
        inflated = shapes.img_arr.copy()
        INFL_ITER = shapes.conf['line_cut_inflation']
        inflated = cv.erode(inflated, None, iterations=INFL_ITER)

        self.vsum = inflated.sum(axis=1)

        ### Determine line indices
        vsum_max = self.vsum.max()
        threshold =  vsum_max * thresh_scale
        self.line_indices = []
        for i, s in enumerate(self.vsum):
            if s < threshold and self.vsum[i - 1] >= threshold:
                self.line_indices.append(i - 1)
        li = self.line_indices
        self.k = len(self.line_indices)
        ### Calculate heights of lines. Split out lines that are too tall
        diffs = [li[i+1] - li[i] - len(where(self.vsum[li[i]:li[i+1]] == vsum_max)[0]) for i in range(len(li[:-2]))]

        too_tall = []
        for i in range(len(diffs)):
            otherdiffs = diffs[:i] + diffs[i+1:]
            odmean = mean(otherdiffs)
            odstd = std(otherdiffs)
            if diffs[i] > 2*odstd + odmean:
                #TODO: use better criteria for above condition
                too_tall.append(i)
        
        if shapes.conf['stop_line_cut']:
            if len(too_tall) > 0:
                return
#        
        ### Separate lines deemed too tall
#         added_lines = 0
#         for i in too_tall:
#             i += added_lines
#             padding = floor(diffs[i]*.25)
#             top_bound = self.line_indices[i] + padding
#             bottom_bound = self.line_indices[i+1] - padding
# #            print top_bound, bottom_bound
#             extra_line = top_bound + self.shapes.img_arr[top_bound:bottom_bound].sum(axis=1).argmax()
#             insert_point = bisect(self.line_indices, extra_line)
#             self.line_indices.insert(insert_point, extra_line)
#             added_lines += 1
        
#        li = self.line_indices                                                                                              
#        diffs = [li[i+1] - li[i] for i in range(len(li[:-2]))]
        
        ############## Draw line breaks on original page
#        for l in self.line_indices:
#            self.shapes.img_arr[l] = 0
#        import Image
#        Image.fromarray(self.shapes.img_arr*255).show()
        ##############

        self.assign_char_indices()
        
#        self._remove_small_noise()
        
    def get_baselines(self):
        
        if not self.baselines:
            for i, k in enumerate(self.line_indices[:-1]):
                vsum_vals = self.vsum[k:self.line_indices[i+1]]
                if vsum_vals.any():
                    self.baselines.append(k +\
                        argmin(vsum_vals))
            self.baselines.append(self.line_indices[-1]+\
                    argmin(self.vsum[self.line_indices[-1]:]))

        return self.baselines
            
    def assign_char_indices(self):
        '''
        Notes:
        ------
        Complexity: 
        nlogn for sorting
        linear for zip and index extraction
        bisect is log n
        '''
        char_tops = list(zip(self.shapes.get_tops(), self.shapes.get_indices()))
        char_tops.sort(key=lambda x: x[0])
        sorted_indices = [i[1] for i in char_tops]
        
        # Debug: Show what indices we're working with and their box sizes
        print(f"[LINECUT DEBUG] Processing {len(sorted_indices)} total indices")
        for idx in sorted_indices:
            box = self.shapes.get_boxes()[idx]
            print(f"[LINECUT DEBUG] Index {idx}: box={box}, w={box[2]}, h={box[3]}")
        _line_insert_indxs = []
        _line_insert_indxs.extend([bisect(char_tops, (i - 1,))
                                   for i in self.line_indices])
        self.lines_chars = []
        if not _line_insert_indxs: 
            # No line breaks found - treat whole page as single line
            print("[LINECUT] No line breaks detected, treating as single line")
            self.lines_chars = [sorted_indices]
            print(f"[LINECUT DEBUG] Single line assigned {len(sorted_indices)} characters: {sorted_indices}")
            self.k = 1
            return
        for i, l in enumerate(_line_insert_indxs[:-1]):
            self.lines_chars.append(sorted_indices[l:_line_insert_indxs[i+1]])
        
        self.lines_chars.append(sorted_indices[_line_insert_indxs[-1]:])
        
        # Debug: Show final line assignment
        print(f"[LINECUT DEBUG] Final lines_chars assignment:")
        for i, line_chars in enumerate(self.lines_chars):
            print(f"[LINECUT DEBUG] Line {i}: {len(line_chars)} chars: {line_chars}")
        
        
        self.k = len(self.line_indices)
        
        ## Small contour insertion with improved line assignment
        print(f"[LINECUT DEBUG] Processing {len(self.shapes.small_contour_indices)} small_contour_indices: {self.shapes.small_contour_indices}")
        cctops = [self.shapes.get_boxes()[i][1] for i in self.shapes.small_contour_indices]
        char_tops = list(zip(cctops, self.shapes.small_contour_indices))
        char_tops.sort(key=lambda x: x[0])
        sorted_indices = [i[1] for i in char_tops]
        print(f"[LINECUT DEBUG] sorted_indices after y-coord sort: {sorted_indices}")
        
        # Get the y-coordinates of actual text lines (main characters) for better assignment
        main_line_y_coords = []
        for line_chars in self.lines_chars:
            if line_chars:  # If this line has characters
                line_y_coords = [self.shapes.get_boxes()[idx][1] for idx in line_chars]
                avg_y = sum(line_y_coords) / len(line_y_coords)
                main_line_y_coords.append(avg_y)
            else:
                main_line_y_coords.append(float('inf'))  # No characters in this line
        
        # Assign small characters to lines based on proximity to main text lines
        self.small_cc_lines_chars = [[] for _ in range(len(self.lines_chars))]
        
        for y_coord, char_idx in char_tops:
            # Find the closest main text line
            best_line = 0
            min_distance = float('inf')
            
            for line_idx, main_y in enumerate(main_line_y_coords):
                if main_y != float('inf'):  # Line has characters
                    distance = abs(y_coord - main_y)
                    if distance < min_distance:
                        min_distance = distance
                        best_line = line_idx
            
            # Assign the small character to the closest line
            if best_line < len(self.small_cc_lines_chars):
                self.small_cc_lines_chars[best_line].append(char_idx)
                print(f"[LINECUT DEBUG] Assigned small char {char_idx} to line {best_line} (distance={min_distance:.1f})")
        
        # Filter out empty lines to match the original logic
        print(f"[LINECUT DEBUG] Before filtering: small_cc_lines_chars = {self.small_cc_lines_chars}")
        self.small_cc_lines_chars = [self.small_cc_lines_chars[i] for i in range(len(self.lines_chars)) if self.lines_chars[i]]
        print(f"[LINECUT DEBUG] After filtering: small_cc_lines_chars = {self.small_cc_lines_chars}")

        if self.shapes.detect_o:
            cctops = [self.shapes.get_boxes()[i][1] for i in self.shapes.naros]
            char_tops = list(zip(cctops, self.shapes.naros))
            char_tops.sort(key=lambda x: x[0])
            sorted_indices = [i[1] for i in char_tops]
            _line_insert_indxs = []
            _line_insert_indxs.extend([bisect_right(char_tops, (i - 1,))
                           for i in self.line_indices])

            if not _line_insert_indxs: sys.exit()
            
            self.line_naros = []
            for i, l in enumerate(_line_insert_indxs[:-1]):
                self.line_naros.append(sorted_indices[l:_line_insert_indxs[i+1]])
            
            self.line_naros.append(sorted_indices[_line_insert_indxs[-1]:])
            
            self.line_naros  = [self.line_naros[i] for i in range(len(self.lines_chars)) if self.lines_chars[i]]
        
        if self.shapes.low_ink:
            
            cctops = [lib[1] for lib in self.shapes.low_ink_boxes]
            char_tops = list(zip(cctops, self.shapes.low_ink_boxes))
            char_tops.sort(key=lambda x: x[0])
            sorted_indices = [i[1] for i in char_tops]
            _line_insert_indxs = []
            _line_insert_indxs.extend([bisect_right(char_tops, (i - 1,))
                                       for i in self.line_indices])
            
            self.low_ink_boxes = []
            if not _line_insert_indxs: sys.exit()
            for i, l in enumerate(_line_insert_indxs[:-1]):
                self.low_ink_boxes.append(sorted_indices[l:_line_insert_indxs[i+1]])
            
            self.low_ink_boxes.append(sorted_indices[_line_insert_indxs[-1]:])
            
            self.low_ink_boxes= [self.low_ink_boxes[i] for i in range(len(self.lines_chars)) if self.lines_chars[i]]
        
#
#        for c in small_cc:
#            t = self.shapes.get_boxes()[c][1]
#            for i, line in enumerate(self.lines_chars):
#                if i+1 < len(self.lines_chars):
#                    if self.lines_chars[i+1]:
#                        next_topmost =  self.shapes.get_boxes()[self.lines_chars[i+1][0]][1]
#                        if topmost < t and t < next_topmost:
#                            self.small_cc_lines_chars[i].append(c)
#                            break
#                        topmost = next_topmost
#        
        
    def _remove_small_noise(self):
        global_tsek_mean = self.shapes.tsek_mean
        global_tsek_std = self.shapes.tsek_std
        global_char_mean = self.shapes.char_mean
        global_char_std = self.shapes.char_std
        for l in self.lines_chars:
            widths = [self.shapes.get_boxes()[i][2] for i in l]
            char_mean, char_std, tsek_mean, tsek_std = self.shapes.char_gaussians(widths)
            for i in l: 
                if self.shapes.get_boxes()[i][2] < (global_tsek_mean - 
                   1 * global_tsek_std) and not char_mean < global_char_mean -.5* global_char_std:
                    l.remove(i)
#                elif len(l) < 4 and global_tsek_mean - global_tsek_std <= self.shapes.get_boxes()[i][2] <= global_tsek_mean + global_tsek_std:
#                    l.remove(i)
        
    # These two functions are for API compatability with LineCluster
    def get_box(self, ind):
        return self.shapes.get_boxes()[ind]
    
    def get_contour(self, ind):
        return self.shapes.contours[ind]

class LineCluster(object):
    '''Line Cluster object - breaks lines by clustering according to tops of
    bounding boxes. Useful in cases where it drawing a straight line between
    page lines is difficult. Requires you know how many lines are on the page
    beforehand.
    
    Parameters:
    --------------------
    shapes: page_element object, (see page_elements.py)
    
    k: int, required
        number of lines on the page
    
    Attributes:
    -----------
    lines_chars: list of lists, length=number of lines on page. Each sub-list
        contains the indices for the bounding boxes/contours assigned to
        its corresponding line.
    
    line_indices: list of int, indices of breaklines with respect to page_array
    
    baselines: list of int, the index of the baseline for each line where
        baseline here is usually a line that goes through all the thick "head"
        (Tibetan: mgo) parts found on most Tibetan letters
    
    Methods:
    --------
    
    get_box: return the bounding box for a given index
    get_contour: return the contour for a given index
    
    The get_box, get_contour methods are mostly here for API compatibility with
    LineCluster.
    
    Notes:
    ------
    This code is messy and in progress. Some of the logic in the end contains
    hardcoded values particular to the Nyingma Gyudbum which is obviously 
    useless for general cases.
    '''
    def __init__(self, shapes, k):
        from sklearn.cluster import KMeans
        self.shapes = shapes
        self.k = k
        self.page_array = shapes.img_arr

        if shapes.conf['line_cluster_pos'] == 'top':
            tops = array(shapes.get_tops(), dtype=float64)
        elif shapes.conf['line_cluster_pos'] == 'center':
            tops = array(
                 [t[1] + .5*shapes.char_mean for t in shapes.get_boxes() if t[3] > 2* shapes.tsek_mean],
                    dtype=float64
                         )
        else:
            raise ValueError("The line_cluster_pos argument must be either 'top' or 'center'")

        tops.shape = (len(tops), 1)
        
        kmeans = KMeans(n_clusters=k)
#         print tops
        kmeans.fit(tops)
        
        
        ################## 
        ######## mark cluster centroids on original image and show them
#        img_arr = shapes.img_arr.copy()
#        for centroid in kmeans.cluster_centers_:
##            print centroid[0]
#            img_arr[centroid[0],:] = 0
#            
#        import Image
#        Image.fromarray(img_arr*255).show()
        #######################3
        
        lines = [[] for i in range(k)]
        
        ind = shapes.get_indices()
        
        ### Assign char pointers (ind) to the appropriate line ###
#        [lines[kmeans.labels_[i]].append(ind[i]) for i in range(len(ind))]
        [lines[kmeans.predict([[shapes.get_boxes()[ind[i]][1]]])[0]].append(ind[i]) for i in range(len(ind))]
        lines = [l for l in lines if l]
        self.k = len(lines)
        boxes = shapes.get_boxes()
        
        
        ### Sort indices so they are in order from top to bottom using y from the first box in each line
        
        sort_inx = list(argsort([boxes[line[0]][1] for line in lines]))
        lines.sort(key=lambda line: boxes[line[0]][1])
        
        ### Get breaklines for splitting up lines
        ### Uses the topmost box in each line cluster to determine breakline
        
        try:
            topmosts = [min([boxes[i][1] for i in line]) for line in lines]
        except ValueError:
            print('failed to get topmosts...')
            raise
        
        vsums = self.page_array.sum(axis=1)
        breaklines = []
        delta = 25
        for c in topmosts:
            if c - delta < 0:
                lower = 0
            else:
                lower = c-delta
            e = argmax(vsums[lower:c+delta])
            c = c - delta + e
            if c < 0:
                c = 0
            breaklines.append(c)
    
        breaklines.append(self.page_array.shape[0])
        self.baselines = []
        
        for i, br in enumerate(breaklines[:-1]):
            
            try:
                baseline_area = vsums[br:breaklines[i+1]]
                if baseline_area.any():
                    self.baselines.append(br + argmin(baseline_area))
                else:
                    print(i)
                    print('No baseline info')
            except ValueError:
                print('ValueError. exiting...HERE')
                import traceback;traceback.print_exc()
                
                raise

        final_ind = dict((i, []) for i in range(len(lines)))
        self.new_contours = {}
        for j, br in enumerate(breaklines[1:-1]):
            topcount = 0
            bottomcount = 0
            for i in lines[j]:
                # if char extends into next line, break it
                # 253 is roughly global line height avg + 1 std
                # The following lines says that a box/char must be extending over 
                # breakline by a non trivial amount eg. 30 px and must itself
                # be a tall-ish box (roughly the height of average line) in order
                # for it to be broken. 
    #            if (bounding[i][1] + bounding[i][3]) - br >= 30 and bounding[i][3] > 205:
                if (boxes[i][1] + boxes[i][3]) - br >= 30 and \
                    (boxes[i][1] + boxes[i][3]) - topmosts[j] > self.shapes.char_mean*2.85:
                    chars = ones((boxes[i][3]+2, boxes[i][2]+2), dtype=uint8)
                    contours = shapes.contours
                    cv.drawContours(chars, [contours[i]], -1,0, \
                        thickness = -1, offset=(-boxes[i][0]+1,-boxes[i][1]+1))
                    cv.dilate(chars, None, chars)
                    y_offset = boxes[i][1]
                    new_br = br - y_offset
                    prd_cut = []

                    ### Iterate through potential cut-points and 
                    ### and cut where top half has the highest probability
                    ### that is not a tsek
#                     print 'bottom bound cut point', int(.75*shapes.tsek_mean)
                    for delta in range(-3, int(.75*shapes.tsek_mean), 1):
#                     for delta in range(-3, 100, 1):
                        cut_point = new_br + delta
#                        chars[cut_point, :] = 0
#                        import Image
#                        Image.fromarray(chars*255).show()
                        tchr = chars[:cut_point,:]
                        tchr = ftrim(tchr)
                        if not tchr.any():
                            continue
                        tchr = normalize_and_extract_features(tchr)
                        probs = cls.predict_proba(tchr)
                        max_prob_ind = argmax(probs)
                        chr = label_chars[max_prob_ind]
                        prd_cut.append((probs[0,max_prob_ind], chr, cut_point))
                    
                    prd_cut = [q for q in prd_cut if q[1] != '་']
                    try:
                        cut_point = max(prd_cut)[-1]
                    except:
                        print('No max prob for vertical char break, using default breakline. Usually this means the top half of the attempted segmentation looks like a tsek blob')
                        cut_point = br-boxes[i][1]

                    #######FOLLWNG NOT WORKING ATTEMPTS TO GET A BETTER BREAK LINE
    #                br2 = br-bounding[i][1]
    #                
    #                csum = chars.sum(axis=1)
    #                bzone = csum[br2-25:br2+40]
    #                if bzone.any():
    #                    br2 = np.argmax(bzone) + (br-25)
    ##                    print br, 'br'
    #                chars = chars*255
    #                nbr = br
    #                cv.line(chars, (0, br2), (chars.shape[1], br2), 0)
    #                Image.fromarray(chars).save('/tmp/outt.tiff')
    #                sys.exit()
                    #############
                    
                    tarr = chars[:cut_point,:]
                    tarr, top_offset = ftrim(tarr, new_offset=True)
                    tarr = fadd_padding(tarr, 3)
                    barr = chars[cut_point:,:]
                    barr = ftrim(barr, sides='brt') 
                    barr = fadd_padding(barr, 3)
                    
                    # Handle different OpenCV versions that return different numbers of values
                    contour_result = cv.findContours(image=tarr, mode=cv.RETR_LIST, method=cv.CHAIN_APPROX_SIMPLE, offset=(boxes[i][0]+top_offset['left'],boxes[i][1]))
                    if len(contour_result) == 2:
                        c1, h = contour_result
                    else:
                        _, c1, h = contour_result

                    c1 = c1[argmax([len(t) for t in c1])] # use the most complex contour

                    bnc1 = cv.boundingRect(c1)

                    # Handle different OpenCV versions that return different numbers of values
                    contour_result2 = cv.findContours(barr, mode=cv.RETR_LIST, 
                                            method=cv.CHAIN_APPROX_SIMPLE,
                                            offset=(boxes[i][0]-3,boxes[i][1]+cut_point-3))
                    if len(contour_result2) == 2:
                        c2, h = contour_result2
                    else:
                        _, c2, h = contour_result2

                    c2 = c2[argmax([len(t) for t in c2])]
                    bnc2 = cv.boundingRect(c2)

                    topbox_name = 't%d_%d' % (j, topcount)
                    final_ind[j].append(topbox_name)
                    self.new_contours[topbox_name] = (bnc1, c1)
                    topcount += 1
                    
                    if bnc2[-1] > 8: #only add bottom contour if not trivially small
                        bottombox_name = 'b%d_%d' % (j, bottomcount)
                        final_ind[j+1].append(bottombox_name)
                        self.new_contours[bottombox_name] = (bnc2, c2)
                        bottomcount += 1
                    
                else:
                    final_ind[j].append(i)
            # Don't forget to include the last line
        list(map(final_ind[len(lines)-1].append, lines[len(lines)-1]))
        
        self.lines_chars = final_ind
        
        cctops = [self.shapes.get_boxes()[i][1] for i in self.shapes.small_contour_indices]
        char_tops = list(zip(cctops, self.shapes.small_contour_indices))
        char_tops.sort(key=lambda x: x[0])
        sorted_indices = [i[1] for i in char_tops]
        _line_insert_indxs = []
        _line_insert_indxs.extend([bisect_right(char_tops, (i - 1,))
                                   for i in breaklines])
        self.small_cc_lines_chars = []
        if not _line_insert_indxs: sys.exit()

        for i, l in enumerate(_line_insert_indxs[:-1]):
            self.small_cc_lines_chars.append(sorted_indices[l:_line_insert_indxs[i+1]])
        
        self.small_cc_lines_chars.append(sorted_indices[_line_insert_indxs[-1]:])
        
        self.small_cc_lines_chars = [self.small_cc_lines_chars[i] for i in range(len(self.lines_chars)) if self.lines_chars[i]]
       
        cctops = [self.shapes.get_boxes()[i][1] for i in self.shapes.emph_symbols]
        char_tops = list(zip(cctops, self.shapes.emph_symbols))
        char_tops.sort(key=lambda x: x[0])

        empred = [kmeans.predict(shapes.get_boxes()[i][1])[0] for i in self.shapes.emph_symbols]
        
        self.emph_lines = [[] for i in range(k)]
        for nn, e in enumerate(empred):
            self.emph_lines[sort_inx.index(e)].append(self.shapes.emph_symbols[nn])
        
    
        if self.shapes.detect_o:
            cctops = [self.shapes.get_boxes()[i][1] for i in self.shapes.naros]
            char_tops = list(zip(cctops, self.shapes.naros))
            char_tops.sort(key=lambda x: x[0])
            sorted_indices = [i[1] for i in char_tops]
            _line_insert_indxs = []
            _line_insert_indxs.extend([bisect_right(char_tops, (i - 1,))
                                       for i in breaklines])
            
            if not _line_insert_indxs: sys.exit()
            
            self.line_naros = []
            for i, l in enumerate(_line_insert_indxs[:-1]):
    #   
                self.line_naros.append(sorted_indices[l:_line_insert_indxs[i+1]])
            
            self.line_naros.append(sorted_indices[_line_insert_indxs[-1]:])
            
            self.line_naros  = [self.line_naros[i] for i in range(len(self.lines_chars)) if self.lines_chars[i]]
            self.line_naro_spans = []
            for ll, mm in enumerate(self.line_naros):
                thisline = []
                for nn, naro in enumerate(mm):
                    box = self.get_box(naro)
                    thisline.append(box)
                thisline.sort(key=lambda x: x[0])
                self.line_naros[ll].sort(key=lambda x: self.get_box(x)[0])
                self.line_naro_spans.append(thisline)
    
        if self.shapes.low_ink:
            
            cctops = [lib[1] for lib in self.shapes.low_ink_boxes]
            char_tops = list(zip(cctops, self.shapes.low_ink_boxes))
            char_tops.sort(key=lambda x: x[0])
            sorted_indices = [i[1] for i in char_tops]
            _line_insert_indxs = []
            _line_insert_indxs.extend([bisect_right(char_tops, (i - 1,))
                                       for i in breaklines])
            
            self.low_ink_boxes = []
            if not _line_insert_indxs: sys.exit()
    
            for i, l in enumerate(_line_insert_indxs[:-1]):
                self.low_ink_boxes.append(sorted_indices[l:_line_insert_indxs[i+1]])
            
            self.low_ink_boxes.append(sorted_indices[_line_insert_indxs[-1]:])
            
            self.low_ink_boxes = [self.low_ink_boxes[i] for i in range(len(self.lines_chars)) if self.lines_chars[i]]

    def check_naro_overlap(self, line_num, box):
        line = self.line_naro_spans[line_num]
        left_edges = [l[0] for l in line]
        insert = bisect(left_edges, box[0])
        for r in range(insert-1, insert + 1):
            if 0 <= r < len(line):
                sp = line[r]
                if check_for_overlap(sp, box):
                    return self.line_naros[line_num][r]
        return False
        
    def get_box(self, ind):
        try:
            return self.shapes.get_boxes()[ind]
        except (TypeError, IndexError):
            return self.new_contours[ind][0]

    def get_contour(self, ind):
        try:
            return self.shapes.contours[ind]
        except (IndexError, TypeError):
            return self.new_contours[ind][1]
     
    def get_baselines(self):
        return self.baselines

