#! /usr/bin/python
# encoding: utf-8
'''Primary routines that manage OCR recognition'''
from PIL import Image
from bisect import bisect, bisect_left
import pickle as pickle
try:
    from .classify import load_cls
    from .config_manager import Config
except ImportError:
    from classify import load_cls
    from config_manager import Config
import codecs
from cv2 import drawContours
import cv2 as cv
import datetime
try:
    from .fast_utils import fadd_padding, ftrim
    from .feature_extraction import normalize_and_extract_features
    from .line_breaker import LineCut, LineCluster
except ImportError:
    from fast_utils import fadd_padding, ftrim
    from feature_extraction import normalize_and_extract_features
    from line_breaker import LineCut, LineCluster
import logging
import numpy as np
import os
try:
    from .page_elements2 import PageElements as PE2
    from .root_based_finder import is_non_std, word_parts
    from .segment import Segmenter, combine_many_boxes
except ImportError:
    from page_elements2 import PageElements as PE2
    from root_based_finder import is_non_std, word_parts
    from segment import Segmenter, combine_many_boxes
from random import choice
import shelve
import signal
import json
import joblib
import sys
try:
    from .termset import syllables
    from .tparser import parse_syllables
    from .utils import local_file
except ImportError:
    from termset import syllables
    from tparser import parse_syllables
    from utils import local_file
# Import Viterbi with fallback for NumPy compatibility issues
try:
    from viterbi_cython import viterbi_cython
    VITERBI_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import viterbi_cython: {e}")
    VITERBI_AVAILABLE = False
    
    def viterbi_cython(n_observations, n_components, log_startprob, log_transmatT, framelogprob):
        """Fallback implementation when Cython Viterbi is not available"""
        # Simple argmax prediction as fallback
        predictions = []
        for i in range(n_observations):
            predictions.append(np.argmax(framelogprob[i]))
        return 0.0, predictions

# from viterbi_search import viterbi_search, word_bigram
import warnings

# print("[DEBUG] Loading logistic classifier...")
cls = load_cls('logistic-cls')
# print(f"[DEBUG] Classifier loaded: {type(cls)}")
# print(f"[DEBUG] Has predict_log_proba: {hasattr(cls, 'predict_log_proba')}")
# if hasattr(cls, 'classes_'):
#     print(f"[DEBUG] Number of classes: {len(cls.classes_)}")
# else:
#     print("[DEBUG] No classes_ attribute")

## Ignore warnings. THis is mostlu in response to incessant sklearn
## warnings about passing in 1d arrays
warnings.filterwarnings("ignore")
print('ignoring all warnings')
###

rbfcls = load_cls('rbf-cls')
predict_log_proba = cls.predict_log_proba
predict_proba = cls.predict_proba

# Trained characters are labeled by number. Open the shelve that contains
# the mappings between the Unicode character and its number label.
try:
    # Try multiple database backend formats for cross-platform compatibility
    allchars_path = local_file('allchars_dict2')
    char_to_dig = {}
    dig_to_char = {}
    
    # Try different shelve backend formats
    backends_to_try = []
    
    # Add platform-specific backends
    import platform
    if platform.system() == "Windows":
        backends_to_try.extend(['dbm.dumb', 'dbm.ndbm'])
    else:
        backends_to_try.extend(['dbm.gnu', 'dbm.ndbm', 'dbm.dumb'])
    
    loaded_successfully = False
    
    for backend in backends_to_try:
        try:
            # Force specific dbm backend
            import dbm
            if hasattr(dbm, backend.split('.')[1]):
                backend_module = getattr(dbm, backend.split('.')[1])
                # Open with specific backend
                allchars = shelve.Shelf(backend_module.open(allchars_path, 'r'))
                char_to_dig = dict(allchars['allchars'])
                dig_to_char = dict(allchars['label_chars'])
                allchars.close()
                
              #  print(f"[DEBUG] Successfully loaded character mappings using {backend}")
             #   print(f"[DEBUG] Character mappings loaded: {len(char_to_dig)} char_to_dig, {len(dig_to_char)} dig_to_char")
                
                # Check if mappings are sufficient
                if len(dig_to_char) >= 2000:
                    loaded_successfully = True
                    break
                else:
                    print(f"[DEBUG] Insufficient character mappings ({len(dig_to_char)} < 2000) with {backend}")
                    
        except Exception as backend_e:
            continue
    
    if not loaded_successfully:
        # Try default shelve open
        try:
            allchars = shelve.open(allchars_path, 'r')
            char_to_dig = dict(allchars['allchars'])
            dig_to_char = dict(allchars['label_chars'])
            allchars.close()
            
            if len(dig_to_char) >= 2000:
              #  print(f"[DEBUG] Successfully loaded character mappings using default shelve")
               # print(f"[DEBUG] Character mappings loaded: {len(char_to_dig)} char_to_dig, {len(dig_to_char)} dig_to_char")
                loaded_successfully = True
            else:
                raise ValueError(f"Insufficient character mappings ({len(dig_to_char)} < 2000)")
                
        except Exception as shelve_e:
            print(f"Warning: Could not load character mappings from shelve: {shelve_e}")
            raise shelve_e
    
    if loaded_successfully and dig_to_char:
        keys = list(dig_to_char.keys())
        #print(f"[DEBUG] dig_to_char key range: {min(keys)} to {max(keys)}, total: {len(keys)}")
        
except Exception as e:
    print(f"Warning: Could not load character dictionary: {e}")
    # Try to load from backup pickle files
    try:
        import pickle
        with open(local_file('allchars.pkl'), 'rb') as f:
            char_to_dig = pickle.load(f)
        with open(local_file('label_chars.pkl'), 'rb') as f:
            dig_to_char = pickle.load(f)
        print(f"Loaded backup character mappings: {len(char_to_dig)} label_chars, {len(dig_to_char)} chars_label")
        print("ignoring all warnings")
        
        if dig_to_char:
            keys = list(dig_to_char.keys())
           # print(f"[DEBUG] dig_to_char key range: {min(keys)} to {max(keys)}, total: {len(keys)}")
    except Exception as backup_e:
        print(f"Warning: Could not load backup character mappings: {backup_e}")
        char_to_dig = {}
        dig_to_char = {}

## Uncomment the line below when enabling viterbi_hidden_tsek
try:
    gram3 = pickle.load(open(local_file('3gram_stack_dict.pkl'),'rb'))
except Exception as e:
    print(f"Warning: Could not load 3gram dictionary: {e}")
    gram3 = {}

word_parts = set(word_parts)

PCA_TRANS = False

trs_prob = np.load(open(local_file('stack_bigram_mat.npz'),'rb'))
trs_prob = trs_prob[trs_prob.files[0]]

cdmap = pickle.load(open(local_file('extended_char_dig.pkl'),'rb'))

# HMM data structures
trans_p = np.load(open(local_file('stack_bigram_logprob32.npz'),'rb'))
trans_p = trans_p[trans_p.files[0]].transpose()
start_p = np.load(open(local_file('stack_start_logprob32.npz'),'rb'))
start_p = start_p[start_p.files[0]]

start_p_nonlog = np.exp(start_p)

## Uncomment below for syllable bigram
syllable_bigram = pickle.load(open(local_file('syllable_bigram.pkl'), 'rb')) #THIS ONE

def get_trans_prob(stack1, stack2):
    try:
        return trs_prob[cdmap[stack1], cdmap[stack2]]
    except KeyError:
        print('Warning: Transition matrix char-dig map has not been updated with new chars')
        return .25

def prd_prob(feature_vect):
    '''Predict character and probability from feature vector
    
    Parameters:
    -----------
    feature_vect: numpy array of features
    
    Returns:
    --------
    tuple: (character_string, probability_float)
    '''
    try:
        # Ensure feature vector is in the right shape for prediction
        if hasattr(feature_vect, 'flatten'):
            feature_vect = feature_vect.flatten()
        
        # Reshape to 2D array if needed for sklearn
        if len(feature_vect.shape) == 1:
            feature_vect = feature_vect.reshape(1, -1)
        
        # DEBUG: Save feature information
        print(f"[FEATURE DEBUG] Feature vector shape: {feature_vect.shape}")
        print(f"[FEATURE DEBUG] Feature stats: min={feature_vect.min():.3f}, max={feature_vect.max():.3f}, mean={feature_vect.mean():.3f}")
        
        # Get prediction probabilities from classifier
        probs = cls.predict_proba(feature_vect)[0]
        predicted_idx = np.argmax(probs)
        prob = probs[predicted_idx]
        
        # DEBUG: Save prediction information
        top3_indices = np.argsort(probs)[-3:][::-1]  # Top 3 predictions
        print(f"[PREDICTION DEBUG] Top 3 predictions:")
        for i, idx in enumerate(top3_indices):
            if idx in dig_to_char:
                pred_char = dig_to_char[idx]
                print(f"[PREDICTION DEBUG]   {i+1}. Index {idx} -> '{pred_char}' (prob: {probs[idx]:.3f})")
            else:
                print(f"[PREDICTION DEBUG]   {i+1}. Index {idx} -> UNKNOWN (prob: {probs[idx]:.3f})")
        
        # Convert prediction index to character using bounds checking
        if predicted_idx in dig_to_char:
            char = dig_to_char[predicted_idx]
        else:
            # Fallback for out-of-bounds predictions
            print(f"[DEBUG] prd_prob: prediction {predicted_idx} out of bounds, using fallback")
            # Use the highest valid key as fallback
            valid_keys = [k for k in dig_to_char.keys() if k <= predicted_idx]
            if valid_keys:
                fallback_key = max(valid_keys)
                char = dig_to_char[fallback_key]
            else:
                char = '�'  # Unicode replacement character
        
        # Debug: show prediction details
        print(f"[DEBUG] prd_prob: pred={predicted_idx}, prob={prob:.4f}, char='{char}'")
        print(f"[FINAL PREDICTION] Character classified as: '{char}' with confidence {prob:.3f}")
        print("=" * 50)
        return char, float(prob)
        
    except Exception as e:
        print(f"[DEBUG] prd_prob error: {e}")
        # Return a safe fallback
        return '�', 0.1


#############################################
### Post-processing functions ###
#############################################

def viterbi(states, start_p, trans_p, emit_prob):
    '''A basic viterbi decoder implementation
    
    states: a vector or list of states 0 to n
    start_p: a matrix or vector of start probabilities
    trans_p: a matrix of transition probabilities
    emit_prob: an nxT matrix of per-class output probabilities
        where n is the number of states and t is the number
        of transitions
    '''
    V = [{}]
    path = {}
    for y in states:
        V[0][y] = start_p[y] * emit_prob[0][y]
        path[y] = [y]
        
    # Run Viterbi for t > 0
    for t in range(1,len(emit_prob)):
        V.append({})
        newpath = {}
        for y in states:
            (prob, state) = max([(V[t-1][y0] * trans_p[y0][y] * emit_prob[t][y], y0) for y0 in states])
            V[t][y] = prob
            newpath[y] = path[state] + [y]
        path = newpath
    (prob, state) = max([(V[len(emit_prob) - 1][y], y) for y in states])
    return ''.join(dig_to_char[s] for s in path[state])

def viterbi_hidden_tsek(states, start_p, trans_p, emit_prob):
    '''Given a series of recognized characters, infer
likely positions of missing punctuation
    
    Parameters
    --------
    states: the possible classes that can be assigned to (integer codes of stacks)
    start_p: pre-computed starting probabilities of Tibetan syllables
    trans_p: pre-computed transition probabilities between Tibetan stacks
    emit_prob: matrix of per-class probability for t steps
    
    Returns:
    List of possible string candidates with tsek inserted
    '''
    V = [{}]
    path = {}
    tsek_dig = char_to_dig['་']
    # Initialize base cases (t == 0)
    # Find intersection of valid states across all arrays
    max_start_state = len(start_p) - 1
    max_emit_state = len(emit_prob[0]) - 1 if emit_prob and len(emit_prob) > 0 else 0
    max_trans_state = len(trans_p) - 1
    max_char_state = max(dig_to_char.keys()) if dig_to_char else 0
    
    # Use the most restrictive bound
    max_valid_state = min(max_start_state, max_emit_state, max_trans_state)
    
    print(f"Array bounds: start_p={max_start_state}, emit_prob={max_emit_state}, trans_p={max_trans_state}, chars={max_char_state}")
    print(f"Using max valid state: {max_valid_state}")
    
    valid_states = []
    for y in states:
        # Ensure y is within ALL array bounds AND has character mapping
        if (y <= max_valid_state and 
            y in dig_to_char and 
            y < len(start_p) and 
            y < len(emit_prob[0]) and 
            y < len(trans_p)):
            V[0][y] = start_p[y] * emit_prob[0][y]
            path[y] = [y]
            valid_states.append(y)
        elif y > max_valid_state:
            # Only warn for out-of-bounds states, not missing character mappings
            pass  # Silent skip to reduce noise
    
    # Update states to only include valid ones
    states = valid_states
    print(f"Filtered to {len(states)} valid states from bounds 0-{max_valid_state}")
    
    if not states:
        print("Error: No valid states after bounds checking")
        return [""]
    num_obs = len(emit_prob)
    # Run Viterbi for t > 0
    for t in range(1,num_obs*2-1):
        V.append({})
        newpath = {}

        if t % 2 == 1:                
            prob_states = []
            for y0 in states:
                im_path = path.get(y0)
                if not im_path:
                    continue
                if len(im_path) > 1:
                    run_without_tsek = 0
                    for i in im_path[::-1]:
                        if i != tsek_dig:
                            run_without_tsek += 1
                        else:
                            break
                    pr3 = gram3.get(path[y0][-2], {}).get(path[y0][-1],{}).get(tsek_dig,.5)*(1+run_without_tsek*2)
                else:
                    pr3 = .75
                
                try:
                    prob_states.append((V[t-1][y0]*trans_p[y0][tsek_dig]*pr3, y0))
                except:
                    print('-'*20)
                    print(trans_p[y0])
                    print(V[t-1])
                    print('-'*20)
                    raise
            prob, state = max(prob_states)
            V[t][tsek_dig] = prob
            newpath[tsek_dig] = path[state] + [tsek_dig]
            path.update(newpath)
        else:
            srted = np.argsort(emit_prob[t//2])

            for y in srted[-50:]:
                #### normal
#                prob, state = max([(V[t-2][y0]*trans_p[y0][y]*emit_prob[t/2][y], y0) for y0 in states])
                ####
                
                #### Experimental
                prob_states = []
                for y0 in states:
                    im_path = path.get(y0,[])[-4:] # immediate n-2 in path
                    t_m2 = V[t-2].get(y0)
                    if not im_path or not t_m2:
                        continue
                    
                    prob_states.append((V[t-2][y0]*trans_p[y0][y]*emit_prob[t//2][y], y0))
                if not prob_states:
                    continue
                prob, state = max(prob_states)
                
                tsek_prob, tsek_dig = (V[t-1][tsek_dig]*trans_p[tsek_dig][y]*emit_prob[t//2][y], tsek_dig)
                
                if tsek_prob > prob:
                    prob = tsek_prob
                    state = tsek_dig
                
                V[t][y] = prob
                newpath[y] = path[state] + [y]
                
            path = newpath
        if not list(V[t].keys()):
            print(f"Warning: No valid paths at step {t}, returning empty result")
            return [""]
        (prob, state) = max([(V[t][y], y) for y in list(V[t].keys())])
    (prob, state) = max([(V[len(V)-1][y], y) for y in list(V[len(V)-1].keys())])
        
    str_perms = _get_tsek_permutations(''.join(dig_to_char[s] for s in path[state]))
    return str_perms

def _get_tsek_permutations(tsr):
    tsek_count = tsr.count('་')
    syls = parse_syllables(tsr, omit_tsek=False)

    all_candidates = []
    if tsek_count > 8:
        print('too many permutations')
        return [tsr]
    elif tsek_count == 0:
        print('no tsek')
        return [tsr]
    else:
        ops = [['0','1'] for i in range(tsek_count)]
        allops = iter(_enumrate_full_paths(ops))
        for op in allops:
            nstr = []
            op = list(op[::-1])
            for i in syls:
                if i == '་' :
                    cur_op = op.pop()
                    if cur_op == '0':
                        continue
                    else:
                        nstr.append(i)
                else:
                    nstr.append(i)
            
            nstr = ''.join(nstr)
            new_parse = parse_syllables(nstr)
            for p in new_parse:
                if is_non_std(p) and p not in syllables:
                    print(nstr, 'rejected')
                    break
            else:
                print(nstr, 'accepted')
                all_candidates.append(nstr)
    if len(all_candidates) == 0:
        all_candidates = [tsr]
    return all_candidates
        
def hmm_recognize(segmentation):
    '''Only used in speical case where doing tsek-insertion post-process
    
    Parameters:
    __________
    segmentioatn: a segmentation object
    
    
    Returns
    _______
    A tuple (prob, string) corresponding the probability of a
    segmented and recognized string, and its probability
    
    '''
    nstates = trs_prob.shape[0]
    states = list(range(start_p.shape[0]))
    
    obs = []
    bxs = []
    for num, line in enumerate(segmentation.vectors):
        line_boxes = segmentation.new_boxes[num]
        for obn, ob in enumerate(line):
            if not isinstance(ob, str):
                obs.append(ob.flatten())
                bxs.append(line_boxes[obn])
            else:
                print(ob, end=' ')
                print('hmm omitting unicode part')
    if bxs:
        outbox = list(combine_many_boxes(bxs))
    else:
        print('RETURNED NONE')
        return (0, '')

    emit_p = cls.predict_proba(obs)
    results = []
    syllable = []
    for em in emit_p:
        char = dig_to_char[int(cls.classes_[np.argmax(em)])]
        if char in ('་', '།'):
            if syllable:
                prob, res = viterbi_hidden_tsek(states, start_p, trs_prob, syllable)
                results.append(res)
                results.append(char)
                syllable = []
        else:
            syllable.append(em)
    if syllable:
        prob, hmm_out = viterbi_hidden_tsek(states, start_p, trs_prob, syllable)
        results.append(hmm_out)
    else:
        prob = 0
        hmm_out = ''
    
    results = ''.join(results)
    print(results, '<---RESULT')
    return (prob, results)

def _enumrate_full_paths(tree):
    if len(tree) == 1:
        return tree[0]
    combs = []
    frow = tree[-1]
    srow = tree[-2]
    
    for s in srow:
        for f in frow:
            combs.append(s+f)
    tree.pop()
    tree.pop()
    tree.append(combs)
    return _enumrate_full_paths(tree)

def bigram_prob(syl_list):
    return np.prod([syllable_bigram.get(syl_list[i], {}).get(syl_list[i+1], 1e-5) \
                    for i in range(len(syl_list) -1 )])

def max_syllable_bigram(choices):
    best_prob = 0.0
    best_s = ''
    for s in choices:
        print(s, 'is a choice')
        if not isinstance(s, list):
            s = parse_syllables(s)
        prob = bigram_prob(s)
        if prob > best_prob:
            best_prob = prob
            best_s = s
    best_s = '་'.join(best_s)
    return best_prob, best_s

def hmm_recognize_bigram(segmentation):
    # Filter states to only those that exist in character mappings and will fit in emit_prob
    max_hmm_states = start_p.shape[0]  # 871 
    valid_char_states = set(dig_to_char.keys())  # States that have character mappings
    # Use intersection of HMM states and character-mapped states, but limit to classifier range
    states = [s for s in range(min(max_hmm_states, 799)) if s in valid_char_states]
    print(f"Using {len(states)} valid states (0-{max(states) if states else 0}) from {max_hmm_states} HMM states")
    
    obs = []
    bxs = []
    for num, line in enumerate(segmentation.vectors):
        line_boxes = segmentation.new_boxes[num]
        print(f"[VECTOR DEBUG] Line {num}: {len(line)} vectors, {len(line_boxes)} boxes")
        for obn, ob in enumerate(line):
            if hasattr(ob, 'flatten'):
                obs.append(ob.flatten())
                bxs.append(line_boxes[obn])
                if num == 0:  # Debug first line
                    box = line_boxes[obn] if obn < len(line_boxes) else "NO BOX"
                    print(f"[VECTOR DEBUG] Line {num}, vector {obn}: box={box}")
            else:
                print(ob, end=' ')
                print('hmm omitting unicode part')

    if not obs:
        return (0, '')
    
    emit_p = cls.predict_proba(obs)

    results = []
    syllable = []
    for em in emit_p:
        char = dig_to_char[int(cls.classes_[np.argmax(em)])]
        if char in ('་', '།'):
            if syllable:

                res = viterbi_hidden_tsek(states, start_p_nonlog, trs_prob, syllable)

                results.append(res)
                results.append(char)
                syllable = []
        else:
            syllable.append(em)
    if syllable:
        hmm_out = viterbi_hidden_tsek(states, start_p_nonlog, trs_prob, syllable)
        
        results.append(hmm_out)
    else:
        prob = 0
        hmm_out = ''

    all_paths = _enumrate_full_paths(results)
    prob, results = max_syllable_bigram(all_paths)
        
    print(results, 'RESULTS')
    return (prob, results)

#############################################
### Recognizers
#############################################

def recognize_chars(segmentation, tsek_insert_method='baseline', ):
    '''Recognize characters using segmented char data
    
    Parameters:
    --------------------
    segmentation: an instance of PechaCharSegmenter or Segmenter
    
    Returns:
    --------------
    results: Unicode string containing recognized text'''
    
    results = []
    debug_chars_saved = []  # Debug: track saved character images

    tsek_mean = segmentation.final_box_info.tsek_mean
    width_dists = {}
    for l, vectors in enumerate(segmentation.vectors):
        
        if not vectors:
            print('no vectors...')
            continue
        
        tmp_result = []
        new_boxes = segmentation.new_boxes[l]
        
        small_chars = segmentation.line_info.small_cc_lines_chars[l]
        
        #FIXME: define emph lines for line cut
        #### Line Cut has no emph_lines object so need to work around for now...
        emph_markers = getattr(segmentation.line_info, 'emph_lines', [])
        if emph_markers:
            emph_markers = emph_markers[l]
        
        img_arr = segmentation.line_info.shapes.img_arr

        left_edges = [b[0] for b in new_boxes]
        tsek_widths = []

        for s in small_chars[::-1]: # consider small char from end of line going backward. backward useful for misplaced tsek often and maybe for TOC though should check
#        for s in small_chars: # consider small char from end of line going backward. backward useful for misplaced tsek often and maybe for TOC though should check
            cnt = segmentation.line_info.shapes.contours[s]
            bx = segmentation.line_info.shapes.get_boxes()[s]
            bx = list(bx)
            x,y,w,h = bx
            char_arr = np.ones((h,w), dtype=np.uint8)
            offset = (-x, -y)
            drawContours(char_arr, [cnt], -1,0, thickness = -1, offset=offset)
            
            # Debug: save sample character images to understand classifier input
            # MODIFIED: Save all characters to compare with page_elements2 debug
            import cv2 as cv
            debug_img = (char_arr * 255).astype(np.uint8)
            cv.imwrite(f'debug_RECOGNIZE_char_{s}_size_{w}x{h}.png', debug_img)
            print(f"[RECOGNIZE DEBUG] Saved character {s} from recognize.py: shape=({h},{w})")
            if l == 0 and len(debug_chars_saved) < 10:  # Keep original logic
                debug_chars_saved.append(s)
            
            feature_vect = normalize_and_extract_features(char_arr)

            prd = prd_prob(feature_vect)

            insertion_pos = bisect(left_edges, x)

            left_items = 6
            right_items = 5
            if insertion_pos >= len(new_boxes):
                # insertion is at or near end of line and needs more left 
                # neighbors to compensate for there being less chars to define the baseline
                left_items = 12
            elif insertion_pos <= len(new_boxes):
                # same as above except at front of line
                right_items = 12

            if tsek_insert_method == 'baseline':
                top = 1000000 # arbitrary high number
                bottom = 0
                
                #### Get min or max index to avoid reaching beyond edges of the line
                lower = max(insertion_pos - left_items, 0)
                upper = min(len(new_boxes)-1, insertion_pos+right_items)
                ####
                
                left = new_boxes[lower][0]
                right = new_boxes[upper][0] + new_boxes[upper][2]
                if insertion_pos < len(new_boxes):
                    mid = new_boxes[insertion_pos][0] + new_boxes[insertion_pos][2]
                else:
                    mid = right
                for j in new_boxes[lower:upper]:
                    if j[1] < top:
                        top = j[1]
                    if j[1] + j[3] > bottom:
                        bottom = j[1] + j[3]
                local_span = bottom - top

                if prd == '་' and local_span > 0:

                    left_sum = img_arr[top:bottom,left:mid].sum(axis=1)
                    right_sum = img_arr[top:bottom,mid:right].sum(axis=1)
                    local_baseline_left = top + left_sum.argmin()
                    if mid != right:
                        local_baseline_right = top + right_sum.argmin()
                    else:
                        local_baseline_right = local_baseline_left
                    
                    if ((local_baseline_left >= bx[1] and local_baseline_left <= bx[1] + bx[3]) or 
                    (local_baseline_right >= bx[1] and local_baseline_right <= bx[1] + bx[3])): #or 
#                    (entire_local_baseline >= bx[1] and entire_local_baseline <= bx[1] + bx[3])):
                        ### Account for fact that the placement of a tsek could be 
                        # before or after its indicated insertion pos
                        ### experimental.. only need with certain fonts e.g. "book 6"
                        ## in samples
                        if insertion_pos <= len(new_boxes):
    #                        cur_box_in_pos = new_boxes[insertion_pos]
                            prev_box = new_boxes[insertion_pos-1]
    #                        left_cur = cur_box_in_pos[0]
                            left_prev = prev_box[0]
                            if 0 <= x - left_prev < w and 2*w < prev_box[2]:
                                insertion_pos -= 1

                        vectors.insert(insertion_pos, prd)
                        new_boxes.insert(insertion_pos, bx)
                        left_edges.insert(insertion_pos, bx[0])
                        tsek_widths.append(bx[2])

                elif bx[1] >= top -.25*local_span and bx[1] + bx[3] <= bottom + local_span*.25:
                    vectors.insert(insertion_pos, prd)
                    new_boxes.insert(insertion_pos, bx)
                    left_edges.insert(insertion_pos, bx[0])
            
            else:
                vectors.insert(insertion_pos, prd)
                new_boxes.insert(insertion_pos, bx)
                left_edges.insert(insertion_pos, bx[0])
        
        tsek_mean = np.mean(tsek_widths)
        
        for em in emph_markers:
            marker = dig_to_char[segmentation.line_info.shapes.cached_pred_prob[em][0]]
            marker_prob = segmentation.line_info.shapes.cached_pred_prob[em][1]
            bx = segmentation.line_info.shapes.get_boxes()[em]
            bx = list(bx)
            x,y,w,h = bx
            insertion_pos = bisect(left_edges, x)
            bx.append(marker_prob)
            bx.append(marker)
            vectors.insert(insertion_pos, marker)
            new_boxes.insert(insertion_pos, bx)
            left_edges.insert(insertion_pos, bx[0])
#        tsek_std = np.std(tsek_widths)
        if len(vectors) == 1: i = -1
        
        for i, v in enumerate(vectors[:-1]):
            gap = new_boxes[i+1][0] - (new_boxes[i][0] + new_boxes[i][2])
            threshold = 8.0*tsek_mean  # Increased from default to be more conservative
            if gap >= threshold:
                if not isinstance(v, str):
                    prd = prd_prob(v)
                else:
                    prd = v

                new_boxes[i].append(prd)
                tmp_result.append(new_boxes[i])
                tmp_result.append([-1,-1,-1,-1, ' '])
            else:
                if not isinstance(v, str):
                    prd = prd_prob(v)

                    ### Assume that a tsek shouldn't show up at this point
                    ### a more reliable way to do this is to better
#                    if prd == u'་':
#                        prbs = cls.predict_proba(v)[0]
#                        ind_probs = zip(range(len(prbs)), prbs)
#                        ind_probs.sort(key=lambda x: x[1])
#                        prd = dig_to_char[ind_probs[-2][0]]
                else:
                    prd = v
                
                if not width_dists.get(prd):
                    width_dists[prd] = [new_boxes[i][2]]
                else:
                    width_dists[prd].append(new_boxes[i][2])
                
                new_boxes[i].append(prd)
                tmp_result.append(new_boxes[i])
            
        if not isinstance(vectors[-1], str):
            prd = prd_prob(vectors[-1])
        else:
            prd = vectors[-1]
        new_boxes[-1].append(prd)
        tmp_result.append(new_boxes[-1])
        results.append(tmp_result)

    return results

def recognize_chars_hmm(segmentation, tsek_insert_method='baseline', ):
    '''Recognize characters using segmented char data
    
    Parameters:
    --------------------
    segmentation: an instance of PechaCharSegmenter or Segmenter
    
    Returns:
    --------------
    results: list of lists containing [x,y,width, height, prob, unicode], specifying the
    coordinates of the bounding box of stack, it probability, and its unicode
    characters -- on each line of the page
    '''
    print(f"[TSHEG DEBUG] recognize_chars_hmm called with tsek_insert_method='{tsek_insert_method}'")
    n_states = trans_p.shape[0]
    
    results = []
    tsek_mean = segmentation.final_box_info.tsek_mean
    cached_features = segmentation.line_info.shapes.cached_features
    cached_pred_prob = segmentation.line_info.shapes.cached_pred_prob
    
    # DEBUG: Check what's in the cache from page_elements2.py
    print(f"[CACHE DEBUG] recognize_chars_hmm: Cached features for {len(cached_features)} characters")
    print(f"[CACHE DEBUG] recognize_chars_hmm: Cached predictions for {len(cached_pred_prob)} characters") 
    if len(cached_pred_prob) > 0:
        # Show a few cached predictions
        for char_id in list(cached_pred_prob.keys())[:5]:
            class_id, prob_array = cached_pred_prob[char_id]
            confidence = prob_array[class_id] if hasattr(prob_array, '__getitem__') else prob_array
            char_label = dig_to_char.get(class_id, '?')
            print(f"[CACHE DEBUG] Character {char_id}: cached class {class_id} = '{char_label}' (conf: {confidence:.3f})")
    
#     width_dists = {}
#     times = []
    for l, vectors in enumerate(segmentation.vectors):
        
        if not vectors:
            print('no vectors...')
            continue
        
        print(f"[VECTOR DEBUG] Line {l}: {len(vectors)} vectors, {len(segmentation.new_boxes[l])} new_boxes")
        for i, v in enumerate(vectors):
            if l == 0 and i < 10:  # Debug first line, first 10 vectors
                box = segmentation.new_boxes[l][i] if i < len(segmentation.new_boxes[l]) else "NO BOX"
                v_type = type(v).__name__
                v_summary = f"shape={v.shape}" if hasattr(v, 'shape') else f"type={v_type}, val={str(v)[:50]}"
                print(f"[VECTOR DEBUG] Line {l}, vector {i}: {v_summary}, box={box}")
        
        tmp_result = []
        new_boxes = segmentation.new_boxes[l]
        small_chars = segmentation.line_info.small_cc_lines_chars[l]
        
        #FIXME: define emph lines for line cut
        #### Line Cut has no emph_lines object so need to work around for now...
        emph_markers = getattr(segmentation.line_info, 'emph_lines', [])
        if emph_markers:
            emph_markers = emph_markers[l]
        
        img_arr = segmentation.line_info.shapes.img_arr
        left_edges = [b[0] for b in new_boxes]
        tsek_widths = []
        
        # Debug: Check small_chars content for all lines (temporarily)
        print(f"[TSHEG DEBUG HMM] Line {l}: small_chars = {small_chars}")
        print(f"[TSHEG DEBUG HMM] Line {l}: left_edges = {left_edges}")
        print(f"[TSHEG DEBUG HMM] Line {l}: new_boxes count = {len(new_boxes)}")
        
        for s in small_chars[::-1]: # consider small char from end of line going backward. backward useful for misplaced tsek often and maybe for TOC though should check
            bx = segmentation.line_info.shapes.get_boxes()[s]
            bx = list(bx)
            x,y,w,h = bx
            try:
                feature_vect = cached_features[s]
                inx, probs = cached_pred_prob[s]
                prob = probs[inx]
                # Bounds check for character mapping
                if inx in dig_to_char:
                    prd = dig_to_char[inx]
                    print(f"[DEBUG] Line {l}, char {s}: prds[{s}] = {inx}, maps to = {prd}")
                else:
                    print(f"[DEBUG] Line {l}, char {s}: prds[{s}] = {inx}, NO MAPPING, using fallback")
                    # Use a fallback character or the nearest valid mapping
                    valid_keys = [k for k in dig_to_char.keys() if k <= inx]
                    if valid_keys:
                        fallback_key = max(valid_keys)
                        prd = dig_to_char[fallback_key]
                    else:
                        prd = '�'  # Unicode replacement character for unrecognized
#             else:
#                 vect = normalize_and_extract_features(letter)
            except:
                cnt = segmentation.line_info.shapes.contours[s]
                char_arr = np.ones((h,w), dtype=np.uint8)
                offset = (-x, -y)
                drawContours(char_arr, [cnt], -1,0, thickness = -1, offset=offset)
                
                # DEBUG: Save character from this path too
                import cv2 as cv
                debug_img = (char_arr * 255).astype(np.uint8)
                cv.imwrite(f'debug_RECOGNIZE2_char_{s}_size_{w}x{h}.png', debug_img)
                print(f"[RECOGNIZE2 DEBUG] Saved character {s} from recognize.py path 2: shape=({h},{w})")
                
                feature_vect = normalize_and_extract_features(char_arr)
#            prd = prd_prob(feature_vect)
                prd, prob = prd_prob(feature_vect)

#            print prd, max(cls.predict_proba(feature_vect)[0])
            # Debug: Track each small character processing in HMM
            if l == 0:
                print(f"[TSHEG DEBUG HMM] Processing small char {s}: box=({x},{y},{w},{h}), pred='{prd}', prob={prob:.3f}")
            
            insertion_pos = bisect(left_edges, x)
            
            # Debug: Track insertion position calculation in HMM
            if l == 0:
                print(f"[TSHEG DEBUG HMM] Character {s}: x={x}, insertion_pos={insertion_pos}, left_edges={left_edges}")
            
            left_items = 6
            right_items = 5
            if insertion_pos >= len(new_boxes):
                left_items = 12
            elif insertion_pos <= len(new_boxes):
                # same as above except at front of line
                right_items = 12
            if tsek_insert_method == 'baseline':
                # Debug: Track tsek_insert_method execution in HMM
                if l == 0:
                    print(f"[TSHEG DEBUG HMM] Character {s}: Using baseline method, tsek_insert_method='{tsek_insert_method}'")
                top = 1000000 # arbitrary high number
                bottom = 0
                
                #### Get min or max index to avoid reaching beyond edges of the line
                lower = max(insertion_pos - left_items, 0)
                upper = min(len(new_boxes)-1, insertion_pos+right_items)
                ####
                
                
                left = new_boxes[lower][0]
                right = new_boxes[upper][0] + new_boxes[upper][2]
                if insertion_pos < len(new_boxes):
                    mid = new_boxes[insertion_pos][0] + new_boxes[insertion_pos][2]
                else:
                    mid = right
                for j in new_boxes[lower:upper]:
                    if j[1] < top:
                        top = j[1]
                    try:
                        if j[1] + j[3] > bottom:
                            bottom = j[1] + j[3]
                    except IndexError:
                        print(new_boxes[lower:upper])
                        print(j)
                        raise
                local_span = bottom - top

                left_sum = img_arr[top:bottom,left:mid].sum(axis=1)
                right_sum = img_arr[top:bottom,mid:right].sum(axis=1)
                try:
                    local_baseline_left = top + left_sum.argmin()
                except:
                    local_baseline_left = top 
                    
                if mid != right:
                    local_baseline_right = top + right_sum.argmin()
                else:
                    local_baseline_right = local_baseline_left
                if prd == '་':  # Remove local_span > 0 condition to allow tsheg even without line context
                    baseline_check = ((local_baseline_left >= bx[1] and local_baseline_left <= bx[1] + bx[3]) or 
                                    (local_baseline_right >= bx[1] and local_baseline_right <= bx[1] + bx[3]))
                    end_check = (insertion_pos == len(vectors))
                    span_check = (local_span <= 0)
                    
                    # Enhanced validation for small tshegs: allow if positioned anywhere within line range
                    # Some tshegs can be positioned at various y-coordinates due to image quality issues
                    if not baseline_check and bx[2] <= 8 and bx[3] <= 8:  # Small tsheg (including 2x1 first tsheg)
                        # Check if tsheg is positioned within reasonable range of other characters on the line
                        line_top = min(box[1] for box in new_boxes)  # Top of line
                        line_bottom = max(box[1] + box[3] for box in new_boxes)  # Bottom of line
                        tsheg_y = bx[1]
                        
                        # Allow tsheg if positioned anywhere within expanded line bounds (more permissive)
                        # Expand range to cover tshegs that might be slightly above or below main text
                        line_height = line_bottom - line_top
                        expanded_top = line_top - line_height * 0.5  # Allow above main text
                        expanded_bottom = line_bottom + line_height * 0.5  # Allow below main text
                        
                        if expanded_top <= tsheg_y <= expanded_bottom:
                            baseline_check = True
                            if l == 0:
                                print(f"[TSHEG DEBUG HMM] Character {s}: Enhanced check PASSED - tsheg_y={tsheg_y} within expanded range [{expanded_top:.1f}, {expanded_bottom:.1f}]")
                    
                    
                    if l == 0:
                        print(f"[TSHEG DEBUG HMM] Character {s}: baseline_check={baseline_check}, end_check={end_check}, span_check={span_check}")
                        print(f"[TSHEG DEBUG HMM] Character {s}: local_span={local_span}, baseline_left={local_baseline_left}, baseline_right={local_baseline_right}")
                        
                    if baseline_check or end_check or span_check: #or 
                        if l == 0:
                            print(f"[TSHEG DEBUG HMM] Character {s}: INSERTING tsheg '{prd}' at position {insertion_pos}")
                            
                        if insertion_pos <= len(new_boxes):
                            prev_box = new_boxes[insertion_pos-1]
                            left_prev = prev_box[0]
                            if 0 <= x - left_prev < w and 2*w < prev_box[2]:
                                insertion_pos -= 1

                        new_boxes.insert(insertion_pos, bx)
                        bx.append(prob)
                        bx.append(prd)
                        vectors.insert(insertion_pos, bx)
                        left_edges.insert(insertion_pos, bx[0])  # Update left_edges for subsequent insertions
                    else:
                        if l == 0:
                            print(f"[TSHEG DEBUG HMM] Character {s}: SKIPPING tsheg '{prd}' - conditions not met")

                        left_edges.insert(insertion_pos, bx[0])
                        tsek_widths.append(bx[2])

                elif ((bx[1] >= top -.25*local_span and bx[1] + bx[3] <= 
                       bottom + local_span*.25) or 
                      (insertion_pos == len(vectors))) and bx[1] - local_baseline_left < 2*tsek_mean:
                    vectors.insert(insertion_pos, prd)
                    new_boxes.insert(insertion_pos, bx)
                    bx.append(prob)
                    bx.append(prd)
                    left_edges.insert(insertion_pos, bx[0])
                    
                else:
                    print('small contour reject at', l, s, 'local height span', local_span, 'box height', bx[3])
            
            else:
                vectors.insert(insertion_pos, prd)
                new_boxes.insert(insertion_pos, bx)
                bx.append(prob)
                bx.append(prd)
                left_edges.insert(insertion_pos, bx[0])
        
        for em in emph_markers:
            mkinx = segmentation.line_info.shapes.cached_pred_prob[em][0]
            # Bounds check for character mapping
            if mkinx in dig_to_char:
                marker = dig_to_char[mkinx]
            else:
                print(f"[DEBUG] Emphasis marker {em}: inx={mkinx} NO MAPPING, using �")
                marker = '�'  # Unicode replacement character for unrecognized
            marker_prob = segmentation.line_info.shapes.cached_pred_prob[em][1][mkinx]
            bx = segmentation.line_info.shapes.get_boxes()[em]
            bx = list(bx)
            x,y,w,h = bx
            insertion_pos = bisect(left_edges, x)
            vectors.insert(insertion_pos, marker)
            bx.append(marker_prob)
            bx.append(marker)
            new_boxes.insert(insertion_pos, bx)
            left_edges.insert(insertion_pos, bx[0])
        if len(vectors) == 1: i = -1
        
        skip_next_n = 0
        
        ###HMM PHASE

        allstrs = []
        curstr = []
        allinx = []
        curinx = []
        
        for j, v in enumerate(vectors):
            
            islist = isinstance(v, list)
            if isinstance(v, str) or islist:
                allstrs.append(curstr)
                allinx.append(curinx)
                curstr = []
                curinx = []
            else:
                curstr.append(v)
                curinx.append(j)
        if curstr:
            allstrs.append(curstr)
            allinx.append(curinx)
        for f, group in enumerate(allstrs):
            if not group: continue
            
            # CRITICAL FIX: Check if we can use cached predictions for this group
            inx = allinx[f]
            use_cached = True
            cached_predictions = []
            cached_probs = []
            
            # Check if all characters in this group have cached predictions
            for i, vector_idx in enumerate(inx):
                # Create robust char_id mapping that handles X-coordinate reordering
                char_id = None
                try:
                    # Get the original character indices that created this vector
                    original_char_indices = segmentation.final_box_info.final_indices[l][vector_idx]
                    if original_char_indices and isinstance(original_char_indices, (list, tuple)):
                        # Multiple characters combined into one vector - use first one
                        char_id = original_char_indices[0]
                        print(f"[CACHE MAPPING] Vector {vector_idx} -> combined chars {original_char_indices} -> using char {char_id}")
                    elif original_char_indices is not None:
                        # Single character mapping
                        char_id = original_char_indices
                        print(f"[CACHE MAPPING] Vector {vector_idx} -> direct char {char_id}")
                    else:
                        # Empty or None mapping - try box-based fallback
                        print(f"[CACHE MAPPING] Vector {vector_idx} -> empty mapping, trying box fallback")
                        char_id = None
                except (IndexError, AttributeError, TypeError) as e:
                    print(f"[CACHE MAPPING] Vector {vector_idx} -> final_indices error ({e}), trying box fallback")
                    char_id = None
                
                # IMPROVED Box-based fallback: find character by matching vector box to cached character box
                if char_id is None:
                    try:
                        vector_box = segmentation.new_boxes[l][vector_idx]
                        vector_x, vector_y = vector_box[0], vector_box[1]
                        
                        # Find cached character with closest position (both X and Y coordinates)
                        best_match = None
                        best_distance = float('inf')
                        
                        # Get original character boxes for better matching
                        page_elements = getattr(segmentation, 'page_elements', None)
                        if page_elements is not None:
                            original_boxes = page_elements.get_boxes()
                            
                            for cached_char_id in cached_pred_prob.keys():
                                if cached_char_id < len(original_boxes):
                                    cached_box = original_boxes[cached_char_id]
                                    cached_x, cached_y = cached_box[0], cached_box[1]
                                    
                                    # Use 2D distance for better matching
                                    distance = ((vector_x - cached_x) ** 2 + (vector_y - cached_y) ** 2) ** 0.5
                                    if distance < best_distance:
                                        best_distance = distance
                                        best_match = cached_char_id
                        
                        if best_match is not None and best_distance < 30:  # Reduced tolerance for better accuracy
                            char_id = best_match
                            print(f"[CACHE MAPPING] Vector {vector_idx} -> improved box fallback char {char_id} (distance: {best_distance:.1f}px)")
                        else:
                            # Final fallback: use vector index directly but with position validation
                            char_id = vector_idx
                            print(f"[CACHE MAPPING] Vector {vector_idx} -> fallback direct mapping")
                    except Exception as e:
                        char_id = vector_idx
                        print(f"[CACHE MAPPING] Vector {vector_idx} -> exception fallback ({e})")
                
                if char_id is not None and char_id in cached_pred_prob:
                    class_id, prob_array = cached_pred_prob[char_id]
                    confidence = prob_array[class_id] if hasattr(prob_array, '__getitem__') else prob_array
                    cached_predictions.append(class_id)
                    cached_probs.append(confidence)
                    if l == 0 and vector_idx < 5:
                        print(f"[CACHE FIX] Using cached prediction for vector {vector_idx} (char {char_id}): class {class_id} = '{dig_to_char.get(class_id, '?')}' (conf: {confidence:.3f})")
                else:
                    use_cached = False
                    print(f"[CACHE MISS] Vector {vector_idx} (char {char_id}) not in cache - will re-classify")
                    break
            
            if use_cached and cached_predictions:
                # Use cached predictions instead of re-classifying
                prds = cached_predictions
                probs = [[0.0] * len(dig_to_char) for _ in range(len(cached_predictions))]
                for i, (pred, prob) in enumerate(zip(cached_predictions, cached_probs)):
                    if pred < len(probs[i]):
                        probs[i][pred] = prob
                probs = np.array(probs)
                print(f"[CACHE FIX] Line {l}: Using cached predictions for {len(prds)} characters")
            else:
                # Fallback to original classification
                try:
                    probs = predict_log_proba(group)
                except:
                    print(v, end=' ')
                
                LPROB = len(probs)
                if LPROB == 1:
                    inx_local = probs[0].argmax()
                    prb = probs[0][inx_local]
                    # Convert classifier index to actual class label
                    actual_class = int(cls.classes_[inx_local])
                    prds = [actual_class]
                else:
                    probs = probs.astype(np.float32)

                    try:
                        # Suppress specific NumPy warnings during Viterbi call
                        import warnings
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", message=".*np.int.*deprecated.*")
                            warnings.filterwarnings("ignore", message=".*numpy.*has no attribute.*int.*")
                            prb, prds = viterbi_cython(LPROB, n_states, start_p, trans_p, probs)
                    except (AttributeError, TypeError, DeprecationWarning) as e:
                        if "np.int" in str(e) or "numpy" in str(e) and "int" in str(e):
                            print(f"Warning: Viterbi algorithm failed due to NumPy compatibility: {e}")
                        else:
                            print(f"Warning: Viterbi algorithm failed: {e}")
                        # Use fallback: simple argmax prediction
                        # Convert classifier indices to actual class labels
                        prds = [int(cls.classes_[np.argmax(prob_row)]) for prob_row in probs]
                        prb = np.sum([np.max(prob_row) for prob_row in probs])
                
                if not use_cached:
                    prb = np.exp(prb)
            
            # Process predictions regardless of source (cached or new)
            for vv, c in enumerate(range(len(prds))):
                ind = inx[c]
                if use_cached:
                    cprob = cached_probs[c]
                else:
                    cprob = probs[c].max()
                
                new_boxes[ind].append(cprob if use_cached else np.exp(cprob))
                
                # Debug: print prediction details
                if l == 0 and ind < 5:
                    source = "CACHED" if use_cached else "NEW"
                    print(f"[DEBUG {source}] Line {l}, char {ind}: prds[{c}] = {prds[c]}, maps to = {dig_to_char.get(prds[c], 'NOT_FOUND')}")
                
                # Bounds check for character mapping - the critical fix!
                prediction_key = int(prds[c])  # Ensure it's an integer
                if prediction_key in dig_to_char:
                    character = dig_to_char[prediction_key]
                    new_boxes[ind].append(character)
                    print(f"[DEBUG] Line {l}, char {ind}: prediction {prediction_key} → '{character}'")
                else:
                    print(f"[DEBUG] Line {l}, char {ind}: prediction {prediction_key} NOT IN MAPPING, using �")
                    new_boxes[ind].append('�')  # Clear error indicator, not misleading character
        # Handle pre-segmented characters (lists) that were skipped in HMM processing
        processed_indices = set()
        for group_indices in allinx:
            processed_indices.update(group_indices)
        
        for ind, b in enumerate(new_boxes):
            # If this box wasn't processed by HMM (pre-segmented), ensure it's properly formatted
            if ind not in processed_indices and isinstance(vectors[ind], list):
                # Pre-segmented character - already has full format [x, y, w, h, prob, char]
                if len(new_boxes[ind]) >= 6:
                    print(f"[DEBUG] Line {l}, char {ind}: pre-segmented → '{new_boxes[ind][-1]}'")
                else:
                    print(f"[DEBUG] Line {l}, char {ind}: pre-segmented incomplete format: {new_boxes[ind]}")
            
            tmp_result.append(new_boxes[ind])
            if not len(new_boxes[ind]) == 6:
                print(l, ind, new_boxes[ind], '<----- INCOMPLETE FORMAT')
            if ind + 1 < len(new_boxes):
                gap = new_boxes[ind+1][0] - (new_boxes[ind][0] + new_boxes[ind][2])
                threshold = 8.0*tsek_mean  # Increased from default to be more conservative
                if gap >= threshold:
                    tmp_result.append([-1,-1,-1,-1, 1.0, ' '])
            
        results.append(tmp_result)
    return results


def recognize_chars_probout(segmentation, tsek_insert_method='baseline', ):
    '''Recognize characters using segmented char data
    
    Parameters:
    --------------------
    segmentation: an instance of PechaCharSegmenter or Segmenter
    
    Returns:
    --------------
    results: list of lists containing [x,y,width, height, prob, unicode], specifying the
    coordinates of the bounding box of stack, it probability, and its unicode
    characters -- on each line of the page'''
    
    print(f"[TSHEG DEBUG] recognize_chars_probout called with tsek_insert_method='{tsek_insert_method}'")
    results = []
    tsek_mean = segmentation.final_box_info.tsek_mean
    cached_features = segmentation.line_info.shapes.cached_features
    cached_pred_prob = segmentation.line_info.shapes.cached_pred_prob

    for l, vectors in enumerate(segmentation.vectors):
        
        if not vectors:
            print('no vectors...')
            continue
        
        tmp_result = []

        new_boxes = segmentation.new_boxes[l]
        scale_w = segmentation.final_box_info.transitions[l]

        small_chars = segmentation.line_info.small_cc_lines_chars[l]
        
        #FIXME: define emph lines for line cut
        #### Line Cut has no emph_lines object so need to work around for now...
        emph_markers = getattr(segmentation.line_info, 'emph_lines', [])
        if emph_markers:
            emph_markers = emph_markers[l]
        
        img_arr = segmentation.line_info.shapes.img_arr

        left_edges = [b[0] for b in new_boxes]
        tsek_widths = []
        
        # Debug: Check small_chars content for Line 0
        if l == 0:
            print(f"[TSHEG DEBUG] Line {l}: small_chars = {small_chars}")
            print(f"[TSHEG DEBUG] Line {l}: left_edges = {left_edges}")
            print(f"[TSHEG DEBUG] Line {l}: new_boxes count = {len(new_boxes)}")
        
        # DISABLED: Tsheg pipeline - testing if this interferes with main character pipeline
        print(f"[TSHEG PIPELINE] DISABLED: Skipping small_chars processing to test main character pipeline")
        print(f"[TSHEG PIPELINE] Would have processed {len(small_chars)} small characters: {small_chars}")
        
        # Entire tsheg processing loop commented out for testing
        # Original code: for s in small_chars[::-1]: ...
        pass  # Skip all tsheg processing
        
        # Commented out tsheg processing code:
        # Original tsheg loop would be here but is disabled for testing
        
        if False:  # All tsheg code disabled
            try:
                feature_vect = cached_features[s]
                inx, probs = cached_pred_prob[s]
                prob = probs[inx]
                prd = dig_to_char[inx]
                
                # DEBUG: Show what's in the cache
                print(f"[CACHE DEBUG] Character {s}: cached class={inx}, char='{prd}', prob={prob:.3f}")
                if s == 49:  # Special debug for character 49
                    print(f"[CACHE DEBUG] Character 49 SPECIAL: This is our mystery character!")
                    print(f"[CACHE DEBUG] Character 49: feature_vect shape={feature_vect.shape}")
                    print(f"[CACHE DEBUG] Character 49: cached from page_elements2.py")

            except:
                cnt = segmentation.line_info.shapes.contours[s]
                char_arr = np.ones((h,w), dtype=np.uint8)
                offset = (-x, -y)
                drawContours(char_arr, [cnt], -1,0, thickness = -1, offset=offset)
                feature_vect = normalize_and_extract_features(char_arr)
                prd, prob = prd_prob(feature_vect)
            
            # Debug: Track each small character processing
            if l == 0:
                print(f"[TSHEG DEBUG] Processing small char {s}: box=({x},{y},{w},{h}), pred='{prd}', prob={prob:.3f}")
            
            insertion_pos = bisect(left_edges, x)
            
            # Debug: Track insertion position calculation
            if l == 0:
                print(f"[TSHEG DEBUG] Character {s}: x={x}, insertion_pos={insertion_pos}, left_edges={left_edges}")

            left_items = 6
            right_items = 5
            if insertion_pos >= len(new_boxes):
                # insertion is at or near end of line and needs more left 
                # neighbors to compensate for there being less chars to define the baseline
                left_items = 12
            elif insertion_pos <= len(new_boxes):
                # same as above except at front of line
                right_items = 12
#            right_items = 5 # bias slightly toward the left. 
            if tsek_insert_method == 'baseline':
                # Debug: Track tsek_insert_method execution
                if l == 0:
                    print(f"[TSHEG DEBUG] Character {s}: Using baseline method, tsek_insert_method='{tsek_insert_method}'")
                    
                top = 1000000 # arbitrary high number
                bottom = 0
                
                #### Get min or max index to avoid reaching beyond edges of the line
                lower = max(insertion_pos - left_items, 0)
                upper = min(len(new_boxes)-1, insertion_pos+right_items)
                
                if l == 0:
                    print(f"[TSHEG DEBUG] Character {s}: lower={lower}, upper={upper}, left_items={left_items}, right_items={right_items}")
                
                left = new_boxes[lower][0]
                right = new_boxes[upper][0] + new_boxes[upper][2]
                if insertion_pos < len(new_boxes):
                    mid = new_boxes[insertion_pos][0] + new_boxes[insertion_pos][2]
                else:
                    mid = right
                for j in new_boxes[lower:upper]:
                    if j[1] < top:
                        top = j[1]
                    if j[1] + j[3] > bottom:
                        bottom = j[1] + j[3]
                local_span = bottom - top

                top, bottom, left, right, mid = [int(np.round(ff)) for ff in [top, bottom, left, right, mid]]
                
                if l == 0:
                    print(f"[TSHEG DEBUG] Character {s}: prd='{prd}', local_span={local_span}, baseline bounds: top={top}, bottom={bottom}")
                
                if prd == '་' and local_span > 0:
                    # More permissive tsek detection - check if it's roughly in the right vertical area
                    try:
                        left_sum = img_arr[top:bottom,left:mid].sum(axis=1)
                        right_sum = img_arr[top:bottom,mid:right].sum(axis=1)
                        local_baseline_left = top + left_sum.argmin()
                        if mid != right:
                            local_baseline_right = top + right_sum.argmin()
                        else:
                            local_baseline_right = local_baseline_left
                    except (IndexError, ValueError):
                        # If baseline detection fails, use a more permissive approach
                        local_baseline_left = top + local_span // 2
                        local_baseline_right = local_baseline_left
                    
                    # More permissive tsek placement - allow if roughly in middle or baseline area
                    char_middle_y = bx[1] + bx[3] // 2
                    line_middle_y = top + local_span // 2
                    baseline_tolerance = local_span * 0.4  # Allow 40% tolerance around baseline/middle
                    
                    tsek_valid = (
                        # Original baseline check (keep for compatibility)
                        ((local_baseline_left >= bx[1] and local_baseline_left <= bx[1] + bx[3]) or 
                         (local_baseline_right >= bx[1] and local_baseline_right <= bx[1] + bx[3])) or
                        # New middle-area check for tsek
                        (abs(char_middle_y - line_middle_y) <= baseline_tolerance) or
                        # End of line check
                        (insertion_pos == len(vectors)) or
                        # Small tsek check - if tsek is small, be more permissive
                        (bx[3] <= local_span * 0.3)
                    )
                    
                    if l == 0:
                        print(f"[TSHEG DEBUG] Character {s}: tsek_valid={tsek_valid}")
                        
                    if tsek_valid:
                        if l == 0:
                            print(f"[TSHEG DEBUG] Character {s}: INSERTING tsheg '{prd}' at position {insertion_pos}")
                            
                        ### Account for fact that the placement of a tsek could be 
                        # before or after its indicated insertion pos
                        if insertion_pos <= len(new_boxes):
                            prev_box = new_boxes[insertion_pos-1]
                            left_prev = prev_box[0]
                            if 0 <= x - left_prev < w and 2*w < prev_box[2]:
                                insertion_pos -= 1
                        
                        vectors.insert(insertion_pos, prd)
                        new_boxes.insert(insertion_pos, bx)
                        new_boxes[insertion_pos].append(prob)
                    else:
                        if l == 0:
                            print(f"[TSHEG DEBUG] Character {s}: SKIPPING tsheg '{prd}' - tsek_valid=False")
                        new_boxes[insertion_pos].append(prd)
                        left_edges.insert(insertion_pos, bx[0])
                        tsek_widths.append(bx[2])
                elif (bx[1] >= top -.25*local_span and bx[1] + bx[3] <= bottom + local_span*.25) or (insertion_pos == len(vectors)):
                    vectors.insert(insertion_pos, prd)
                    new_boxes.insert(insertion_pos, bx)
                    new_boxes[insertion_pos].append(prob)
                    new_boxes[insertion_pos].append(prd)
                    left_edges.insert(insertion_pos, bx[0])
                    
            else:
                vectors.insert(insertion_pos, prd)
                new_boxes.insert(insertion_pos, bx)
                new_boxes[insertion_pos].append(prob)
                new_boxes[insertion_pos].append(prd)
                left_edges.insert(insertion_pos, bx[0])
        
        for em in emph_markers:
            bx = segmentation.line_info.shapes.get_boxes()[em]
            mkinx = segmentation.line_info.shapes.cached_pred_prob[em][0]
            marker = dig_to_char[mkinx]
            marker_prob = segmentation.line_info.shapes.cached_pred_prob[em][1][mkinx]
            
            bx = list(bx)
            x,y,w,h = bx
            bx.append(marker_prob)
            bx.append(marker)
            insertion_pos = bisect(left_edges, x)
            vectors.insert(insertion_pos, marker)
            new_boxes.insert(insertion_pos, bx)
            left_edges.insert(insertion_pos, bx[0])

        if len(vectors) == 1: i = -1
        
        skip_next_n = 0
        for i, v in enumerate(vectors[:-1]):

            if skip_next_n:
                skip_next_n -= 1
                continue

            # Improved spacing detection with multiple criteria
            gap_width = new_boxes[i+1][0] - (new_boxes[i][0] + new_boxes[i][2])
            char_width = new_boxes[i][2]
            next_char_width = new_boxes[i+1][2]
            avg_char_width = (char_width + next_char_width) / 2
            
            # Multiple spacing criteria (OR logic - any one can trigger a space)
            insert_space = False
            
            # Criterion 1: Traditional large gap (keep original)
            if gap_width >= 2*tsek_mean:
                insert_space = True
                
            # Criterion 2: Gap relative to character width (more adaptive)
            elif gap_width >= 0.8 * avg_char_width:
                insert_space = True
                
            # Criterion 3: Detect missing tsek by checking if gap is substantial
            elif gap_width >= 0.5 * tsek_mean and gap_width >= 3:  # Min 3 pixels and half tsek width
                insert_space = True
            
            if insert_space:
                if not len(new_boxes[i]) == 6 and not isinstance(v, str):
                    prd, prob = prd_prob(v)
                else:
                    if len(new_boxes[i]) == 6:
                        prob, prd = new_boxes[i][4:]
                    else:
                        ## v is unicode stack, likely from segmentation step
                        prd = v
                        prob = .95 # NEED ACTUAL PROB

                new_boxes[i].append(prob)
                new_boxes[i].append(prd)
                tmp_result.append(new_boxes[i])
                # Insert space character for word boundary
                tmp_result.append([-1,-1,-1,-1, 1.0, ' '])
            else:
                if hasattr(v, 'dtype'):
                    try:
                        prd, prob = prd_prob(v)
                    except:
                        print(v)
                    
                    new_boxes[i].append(prob)
                    new_boxes[i].append(prd)
                else:
                    if len(new_boxes[i]) == 6:
                        prob, prd = new_boxes[i][4:]
                    else:
                        prd = v
                
                if len(new_boxes[i]) < 6:
                    try:
                        new_boxes[i].append(prob)
                    except:
                        new_boxes[i].append(1)
                    new_boxes[i].append(prd)
                tmp_result.append(new_boxes[i])
                print(f"[DEBUG] Line {l}, added to tmp_result: '{prd}' at pos {i}")
            
            
        if hasattr(vectors[-1], 'dtype'):
            prd, prob = prd_prob(vectors[-1])
            new_boxes[-1].append(prob)
            new_boxes[-1].append(prd)
            print(f"[DEBUG] Line {l}, last vector: '{prd}'")
        else:
            # Handle case where last vector is already a string
            if len(new_boxes[-1]) >= 6:
                prd = new_boxes[-1][-1]  # Get the character from the box
            else:
                prd = vectors[-1] if isinstance(vectors[-1], str) else "unknown"
            print(f"[DEBUG] Line {l}, last vector (string): '{prd}'")
        tmp_result.append(new_boxes[-1])
        print(f"[DEBUG] Line {l}, added last to tmp_result: total tmp_result = {len(tmp_result)}")
        results.append(tmp_result)
    return results

def viterbi_post_process(img_arr, results):
    '''Go through all results and attempts to correct invalid syllables'''
    final = [[] for i in range(len(results))]
    for i, line in enumerate(results):
        syllable = []
        for j, char in enumerate(line):
            if char[-1] in '་། ' or not word_parts.intersection(char[-1]) or j == len(line)-1:
                if syllable:
                    syl_str = ''.join(s[-1] for s in syllable)
                    
                    if is_non_std(syl_str) and syl_str not in syllables:
                        print(syl_str, 'HAS PROBLEMS. TRYING TO FIX')
                        bx = combine_many_boxes([ch[0:4] for ch in syllable])
                        bx = list(bx)
                        arr = img_arr[bx[1]:bx[1]+bx[3], bx[0]:bx[0]+bx[2]]
                        arr = fadd_padding(arr, 3)
                        try:
                            
                            prob, hmm_res = main(arr, Config(line_break_method='line_cut', page_type='book', postprocess=False, viterbi_postprocess=True, clear_hr=False), page_info={'flname':''})
                        except TypeError:
                            print('HMM run exited with an error.')
                            prob = 0
                            hmm_res = ''
                        
#                         corrections[syl_str].append(hmm_res) 
                        logging.info('VPP Correction: %s\t%s' % (syl_str, hmm_res))
                        if prob == 0 and hmm_res == '':
                            print('hit problem. using unmodified output')
                            for s in syllable:
                                final[i].append(s)
                        else:
                            bx.append(prob)
                            bx.append(hmm_res)
                            final[i].append(bx)
                    else:
                        for s in syllable:
                            final[i].append(s)
                final[i].append(char)
                syllable = []
            else:
                syllable.append(char)
        if syllable:
            for s in syllable:
                final[i].append(s)

    return final

def main(page_array, conf=Config(viterbi_postprocess=False, line_break_method = None, page_type = None), retries=0,
         text=False, page_info={}):
    '''Main procedure for processing a page from start to finish
    
    Parameters:
    --------------------
    page_array: a 2 dimensional numpy array containing binary pixel data of 
        the image
    
    page_info: dictionary, optional
        A dictionary containing metadata about the page to be recognized.
        Define strings for the keywords "flname" and "volume" if saving
        a serialized copy of the OCR results. 

    retries: Used internally when system attempts to reboot a failed attempt
    
    text: boolean flag. If true, return text rather than char-position data
    
    Returns:
    --------------
    text: str
        Recognized text for entire page
        
    if text=False, return character position and label data as a python dictionary
    '''
    
    print(page_info.get('flname',''))
    
    confpath = conf.path
    conf = conf.conf
    
    line_break_method = conf['line_break_method']
    page_type = conf['page_type']

    ### Set the line_break method automatically if it hasn't been
    ### specified beforehand
    if not line_break_method and not page_type:
        if page_array.shape[1] > 2*page_array.shape[0]:
            print('setting page type as pecha')
            line_break_method = 'line_cluster'
            page_type = 'pecha'
        else: 
            print('setting page type as book')
            line_break_method = 'line_cut'
            page_type = 'book' 
            
    conf['page_type'] = page_type
    conf['line_break_method'] = line_break_method
    detect_o = conf.get('detect_o', False)
    print('clear hr', conf.get('clear_hr', False))

    results = []
    out = ''
    try:
        ### Get information about the pages
        shapes = PE2(page_array, cls, page_type=page_type, 
                     low_ink=conf['low_ink'], 
                     flpath=page_info.get('flname',''),
                     detect_o=detect_o, 
                     clear_hr =  conf.get('clear_hr', False))
        shapes.conf = conf
        
        ### Separate the lines on a page
        if page_type == 'pecha':
            k_groups = shapes.num_lines
        shapes.viterbi_post = conf['viterbi_postprocess']
        
        if line_break_method == 'line_cut':
            line_info = LineCut(shapes)
            if not line_info: # immediately skip to re-run with LineCluster
                sys.exit()
        elif line_break_method == 'line_cluster':
            line_info = LineCluster(shapes, k=k_groups)
        
        
        ### Perform segmentation of characters
        segmentation = Segmenter(line_info)

        ###Perform recognition
        if not conf['viterbi_postprocess']:
            # Force use of probout recognizer to fix Line 0 tsheg insertion issues
            # The hmm recognizer has problematic tsheg insertion logic
            if conf['recognizer'] == 'probout':
                results = recognize_chars_probout(segmentation)
            elif conf['recognizer'] == 'hmm':
                results = recognize_chars_hmm(segmentation)
            elif conf['recognizer'] == 'kama':
                results = recognize_chars_probout(segmentation)
                results = recognize_chars_kama(results, segmentation)
            if conf['postprocess']:
                results = viterbi_post_process(segmentation.line_info.shapes.img_arr, results)
        else: # Should only be call from *within* a non viterbi run...

            prob, results = hmm_recognize_bigram(segmentation)
            return prob, results
        
        
        ### Construct an output string
        output  = []
        for n, line in enumerate(results):
            for m,k in enumerate(line):
#                 if isinstance(k[-1], int):
#                     print n,m,k
#                     page_array[k[1]:k[1]+k[3], k[0]:k[0]+k[2]] = 0
#                     Image.fromarray(page_array*255).show()
                    
                output.append(k[-1])
            output.append('\n')

        out =  ''.join(output)
        print(out)
    
        if text:
            results = out
        
        return results
    except:
        ### Retry and assume the error was cause by use of the
        ### wrong line_break_method...
        import traceback;traceback.print_exc()
        if not results and not conf['viterbi_postprocess']:
            print('WARNING', '*'*40)
            print(page_info['flname'], 'failed to return a result.')
            print('WARNING', '*'*40)
            print()
            if line_break_method == 'line_cut' and retries < 1:
                print('retrying with line_cluster instead of line_cut')
                try:
                    return main(page_array, conf=Config(path=confpath, line_break_method='line_cluster', page_type='pecha'), page_info=page_info, retries = 1, text=text)
                except:
                    logging.info('Exited after failure of second run.')
                    return []
        if not conf['viterbi_postprocess']: 
            if not results:
                logging.info('***** No OCR output for %s *****' % page_info['flname'])
            return results

def run_main(fl, conf=None, text=False):
    '''Helper function to do recognition'''
    if not conf:
#         conf = Config(low_ink=False, segmenter='stochastic', recognizer='hmm', 
#               break_width=2.0, page_type='pecha', line_break_method='line_cluster', 
#               line_cluster_pos='center', postprocess=False, detect_o=False,
#               clear_hr = False)
# 
        conf = Config(segmenter='stochastic', recognizer='hmm', break_width=2.5,  
                      line_break_method='line_cut', postprocess=False,
                      low_ink=False, stop_line_cut=False, clear_hr=True, 
                      detect_o=False)

    return main(np.asarray(Image.open(fl).convert('L'))/255, conf=conf, 
                page_info={'flname':os.path.basename(fl), 'volume': VOL}, 
                text=text)


if __name__ == '__main__':
    fls = ['/Users/zach/random-tibetan-tiff.tif']

    lbmethod = 'line_cluster'
    page_type = 'pecha'
    VOL = 'single_volumes'
    
    def run_main(fl):
        try:
            return main(np.asarray(Image.open(fl).convert('L'))/255, 
                        conf=Config(break_width=2.5, recognizer='hmm', 
                                    segmenter='stochastic', page_type='pecha', 
                                    line_break_method='line_cluster'), 
                        page_info={'flname':fl, 'volume': VOL})
        except:
            return []
    import datetime
    start = datetime.datetime.now()
    print('starting')
    outfile = codecs.open('/home/zr/latest-ocr-outfile.txt', 'w', 'utf-8')
    
    for fl in fls:
        
        #### line cut
#         ret = main((np.asarray(Image.open(fl).convert('L'))/255), 
#            conf=Config(break_width=2., recognizer='probout', 
#            segmenter='stochastic', line_break_method='line_cut', 
#            postprocess=False, stop_line_cut=False, low_ink=False, clear_hr=True), 
#                    page_info={'flname':fl, 'volume': VOL}, text=True)

        #### line cluster
        ret = main((np.asarray(Image.open(fl).convert('L'))/255), 
                   conf=Config(segmenter='stochastic', recognizer='hmm', 
                               break_width=2.0, page_type='pecha', 
                               line_break_method='line_cluster',
                               line_cluster_pos='center', postprocess=False,
                                detect_o=False, low_ink=False, clear_hr=True), 
                    page_info={'flname':fl, 'volume': VOL}, text=True)
        outfile.write(ret)
        outfile.write('\n\n')

    print(datetime.datetime.now() - start, 'time taken')
 
