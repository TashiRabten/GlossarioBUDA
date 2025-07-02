from __future__ import division
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt, atan2, fmod,floor, log, abs
import cv2 as cv
GaussianBlur = cv.GaussianBlur
Sobel = cv.Sobel
resize = cv.resize
INTER_CUBIC = cv.INTER_CUBIC
from sklearn.externals import joblib
from utils import local_file
scaler = joblib.load(local_file('zernike_scaler'))
transform = scaler.transform
empty = np.empty
ctypedef np.uint8_t DTYPEUINT_t
ctypedef np.double_t DTYPEDOUBLE_t
DTYPEUINT = np.uint8 
DTYPEDOUBLE = np.double 

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.nonecheck(False)
cdef fnormalize(np.ndarray[DTYPEUINT_t, ndim=2] a):
    cdef double h = a.shape[0]
    cdef double w = a.shape[1] 
    cdef double L = 32.0
    cdef int LL = 32
    cdef double o_2 = 1.0/2.0
    cdef double R1, R2, H2, W2, offset, start, end, alpha, beta, sm, bg, smn, df
    cdef np.ndarray[DTYPEDOUBLE_t, ndim=2] c = empty((32,32), dtype=DTYPEDOUBLE)
    cdef np.ndarray[DTYPEUINT_t, ndim=2] b
    cdef int smi, bgi, i, j, starti, endi
    if h >= w:
        bg = h
        sm = w
        smi = 1
        bgi = 0
    else:
        bg = w
        sm = h
        smi = 0
        bgi = 1
        
    R1 = sm/bg

    R2 = sqrt(R1)

        
    if sm == h:
        H2 = L*R2
        W2 = L
    else:
        H2 = L
        W2 = L*R2
    
    alpha = W2 / w
    beta = H2 / h
#    print alpha,beta

    b = resize(a, (0,0), fy=beta, fx=alpha, interpolation=INTER_CUBIC)
    smn = b.shape[smi]
    df = L - smn
    offset = floor(df * o_2)
    
    if fmod(df, 2) == 1.0:
        start = offset+1.0
        end = offset
    else:
        start = end = offset
    
    starti = int(start)
    endi = int(end)
    
    if sm == h:

        for i in range(LL):
            for j in range(LL):
                if i < starti or i >= LL-endi:
                    c[i,j] = 1.0
                else:
                    c[i,j] = b[i-starti,j]
                     
    else:
        
        for i in range(32):
            for j in range(32):
                if j < starti or j >= LL-endi:
                    c[i,j] = 1.0
                else:
                    c[i,j] = b[i,j-starti]

    return c


# @cython.boundscheck(False)
# @cython.nonecheck(False)
# @cython.cdivision(True)
# cdef transition_features(DTYPEUINT_t [:,:] a):
#     cdef int imgh = a.shape[0]
#     cdef int imgw = a.shape[1]
#     cdef int row2, col1, trs_v, trs_h, prev_h, prev_v, j, i, totale
#     totale = imgh+imgw
#     cdef np.ndarray[DTYPEUINT_t, ndim=1] allv = empty(totale, dtype=DTYPEUINT)
#     cdef DTYPEUINT_t [:,:] b = a.T
#     
# 
#     for i in range(imgh):
#         trs_h = 0
#         trs_v = 0
#         prev_h = 1
#         prev_v = 1
#         for j in range(imgw):
#             col1 = a[i,j]
#             row2 = b[i,j]
#             if col1 == 1 and prev_h == 0:
#                 trs_h = trs_h + 1
#             prev_h = col1
#             if row2 == 1 and prev_v == 0:
#                 trs_v = trs_v + 1
#             prev_v = row2
#                     
#         if col1 == 0: # If the last pixel in the row is black, count it as a run
#             trs_h = trs_h + 1
#         if row2 == 0:
#             trs_v = trs_v + 1
#         allv[i] = trs_h
#         allv[i+imgw] = trs_v
#     
#     return allv



@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef zernike_features(DTYPEDOUBLE_t [:,:] A,DTYPEDOUBLE_t [:,:] D, DTYPEDOUBLE_t [:,:,:] Bpqk, DTYPEDOUBLE_t [:,:] Ipi):
    cdef int p, q, k, j, S, diff, m, i, pp1
    cdef int N = 32 # A.shape[0]
    cdef double o_PI = 1.0/np.pi
    cdef double yiq, mpq, val, zrl, zim
    cdef int deg = 17
    cdef double complex w = -1.0j
    cdef double complex zpk, rpq, dw
    cdef np.ndarray[DTYPEDOUBLE_t, ndim=2] Mpqs = np.empty((deg+1,deg+1))
    cdef np.ndarray[complex, ndim=2] Rpq = empty((deg+1,deg+1), complex)
    cdef np.ndarray[DTYPEDOUBLE_t] Zpq = empty((90), DTYPEDOUBLE) # deg =17

    cdef np.ndarray[complex] ws = np.array([1,-1j,-1,1j], complex)
    

    cdef np.ndarray[DTYPEDOUBLE_t, ndim=2] Yiq = np.zeros((deg+1,N), DTYPEDOUBLE)
    
    
    # Only iterate through black pixels (this isn't really much faster...)
    # presumably np.where is the overhead of this approach
    
    for i in range(0,N):
        for j in range(0,N):
            val = A[i,j]
#            if not val:
            if not val:
                for q in range(0, deg+1):
                    Yiq[q,i] += Ipi[q,j]#*val
            
    
    for p in range(0,deg+1):
        
        for q in range(0,deg+1):
            
            mpq = 0.0
#            
            for i in range(0,N):
                mpq += Yiq[q,i]*Ipi[p,i]
            Mpqs[p,q] = mpq
            
    for p in range(0,deg+1):
        for q in range(0,deg+1):
            
            rpq = 0.0
            diff = p-q
            if diff%2 == 0:
                S = diff//2
            else:
                continue
            for j in range(S+1):
                for m in range(q+1):
                    rpq = rpq + ws[m%4]*D[S,j]*D[q,m]*Mpqs[p-2*j-m,2*j+m]
    
            Rpq[p,q] = rpq
    i = 0
    for p in range(deg+1):
        pp1 = p + 1
        for q in range(p%2,pp1, 2):
            zpk = 0.0
            
            for k in range(q,pp1,2):
    
                zpk = zpk + Bpqk[p,q,k]*Rpq[k,q]
    

            ### The absolute value gives us the magniture of the zernike moment.
            ### The zernike moment itself is pp1*o_PI*zpk and is complex
            Zpq[i] = abs(pp1*o_PI*zpk)
            i = i+1
    return Zpq


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef sobel_features(np.ndarray[DTYPEDOUBLE_t, ndim=2] a):
    cdef int imgh = a.shape[0]
    cdef int imgw = a.shape[1]
    cdef double o_tsize = 1.0/(imgh*imgw)
    cdef int ix0, ix1, i, ftmat_len, j, sec, cls, cls1, cls2, l
    cdef double rbar, msum, curdeg, dx, dy, mg
    cdef double d_30 = 1.0 / 30.0
    cdef double degc = 180.0 / np.pi
    cdef np.ndarray[DTYPEDOUBLE_t, ndim=2] magnitude = empty((imgh, imgw), DTYPEDOUBLE)
    cdef np.ndarray[DTYPEDOUBLE_t, ndim=2] direction = empty((imgh, imgw), DTYPEDOUBLE)
    cdef np.ndarray[DTYPEDOUBLE_t, ndim=2] sx = empty((imgh, imgw), DTYPEDOUBLE)
    cdef np.ndarray[DTYPEDOUBLE_t, ndim=2] sy = empty((imgh, imgw), DTYPEDOUBLE)
    
    
    ###########
#     cdef int imgh = a.shape[0]
#     cdef int imgw = a.shape[1]
    cdef double row2, col1,prev_h, prev_v, trs_v, trs_h
    cdef int totale
    totale = imgh+imgw
#     cdef np.ndarray[DTYPEUINT_t, ndim=1] allv = empty(totale, dtype=DTYPEUINT)
    cdef DTYPEDOUBLE_t [:,:] b = a.T
    ###########
    
    cdef np.ndarray[DTYPEDOUBLE_t] vector = np.zeros((192+totale), DTYPEDOUBLE)
    

    Sobel(a, dst=sx, ddepth=-1, dx=1, dy=0, ksize=3)
    Sobel(a, dst=sy, ddepth=-1, dx=0, dy=1, ksize=3)

    
    for i in range(imgh):
        ############
        trs_h = 0.0
        trs_v = 0.0
        prev_h = 1.0
        prev_v = 1.0
        ###########
        for j in range(imgw):
            ##################
            col1 = a[i,j]
            row2 = b[i,j]
            if col1 == 1 and prev_h == 0:
                trs_h = trs_h + 1.0
            prev_h = col1
            if row2 == 1 and prev_v == 0:
                trs_v = trs_v + 1.0
            prev_v = row2
            #################
            
            dx = sx[i,j]
            dy = sy[i,j]
            mg = sqrt(dx*dx + dy*dy)
            magnitude[i,j] = mg
            msum += mg
            direction[i,j] = atan2(dy,dx)
            
        if col1 == 0: # If the last pixel in the row is black, count it as a run
            trs_h = trs_h + 1.0
        if row2 == 0:
            trs_v = trs_v + 1.0
        vector[192+i] = trs_h
        vector[i+imgw+192] = trs_v


                    

    
    rbar = msum * o_tsize
    for k in range(imgh):
        for l in range(imgh):
            if magnitude[k,l] >= rbar:
                curdeg =  direction[k,l] * degc
                ix0 = k
                ix1 = l
                if ix1 < 8:
                    j = 0
                elif 8 <= ix1 < 16:
                    j = 1
                elif 16 <= ix1 < 24:
                    j = 2
                else:
                    j = 3
                
                if ix0 < 8:
                    sec = 0 + j
                elif 8 <= ix0 < 16:
                    sec = 4 + j
                elif 16 <= ix0 < 24:
                    sec = 8 + j
                else:
                    sec = 12 +j

#                 if curdeg % 30.0 != 0.0:
                if fmod(curdeg, 30.0) != 0.0:
                    if 0.0 < curdeg < 30.0:
                        cls1 = 0
                        cls2 = 1
                    elif 30.0 < curdeg < 60.0:
                        cls1 = 1
                        cls2 = 2
                    elif 60.0 < curdeg < 90.0:
                        cls1 = 2
                        cls2 = 3
                    elif 90.0 < curdeg < 120.0:
                        cls1 = 3
                        cls2 = 4
                    elif 120.0 < curdeg < 150.0:
                        cls1 = 4
                        cls2 = 5
                    elif 150.0 < curdeg < 180.0:
                        cls1 = 5
                        cls2 = 6
        
                    elif -30.0 < curdeg < 0.0:
                        cls1 = 0
                        cls2 = 11
                    elif -60.0 < curdeg < -30.0:
                        cls1 = 11
                        cls2 = 10
                    elif -90.0 < curdeg < -60.0:
                        cls1 = 10
                        cls2 = 9
                    elif -120.0 < curdeg < -90.0:
                        cls1 = 9
                        cls2 = 8
                    elif -150.0 < curdeg < -120.0:
                        cls1 = 8
                        cls2 = 7
                    elif -180.0 < curdeg < -150.0:
                        cls1 = 7
                        cls2 = 6
                    
                    sec = sec * 12
                    vector[sec+cls1] += 1.0
                    vector[sec+cls2] += 1.0
                else:
                    cls = int(curdeg * d_30)
                    sec = sec*12+cls
                    vector[sec] += 1.0


    return vector



# @cython.wraparound(False)

@cython.boundscheck(False)
@cython.nonecheck(False)
cpdef fnormalize_and_extract_features(
        np.ndarray[DTYPEUINT_t, ndim=2] arr,
        DTYPEDOUBLE_t [:,:] D,
        DTYPEDOUBLE_t [:,:,:] Bpqk,
        DTYPEDOUBLE_t [:,:] Ipi
                                    ):
    
    cdef int i                           
    cdef np.ndarray[DTYPEDOUBLE_t, ndim=2] newarr = fnormalize(arr)
#     cdef np.ndarray[DTYPEUINT_t, ndim=1] x3 = transition_features(newarr)
#     print arr.dtype
#     raise
    cdef np.ndarray[DTYPEDOUBLE_t, ndim=1] x1 = zernike_features(newarr,D,Bpqk,Ipi)

#     cdef np.ndarray[DTYPEDOUBLE_t, ndim=2] arr2 = newarr.astype(np.double)
#     arr = -1.0*arr+1.0
    GaussianBlur(newarr, ksize=(5,5), sigmaX=1, dst=newarr)
#     arr = arr.astype(np.float32)
    cdef np.ndarray[DTYPEDOUBLE_t, ndim=1] x2 = sobel_features(newarr)
#     if not custom_scaler:
#     x3 = np.hstack((x1,x2))
#     print x1
    cdef np.ndarray[DTYPEDOUBLE_t] x3 = empty((346,), dtype=DTYPEDOUBLE)
    

    for i in range(346):
        if i < 90:
            x3[i] = x1[i]
        else:
            x3[i] = x2[90-i]
#     print x3.shape
#     return pca.transform(transform(x1))
#     print x3
    return transform(x3, copy=False)