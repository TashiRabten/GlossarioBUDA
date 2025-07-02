'''Implemented from Hosny, K. M. 2007. “Fast computation of accurate Zernike moments.” Journal of Real-Time Image
Processing 3.1-2: 97–107.

Author: Zach Rowinski, 2016
'''

from __future__ import division
import numpy as np
cimport numpy as np
cimport cython
 
 
 
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cpdef zernike_features(np.ndarray[double, ndim=2] A, 
        np.ndarray[double, ndim=2] D,
        np.ndarray[double, ndim=3] Bpqk,
        double[::1, :] Ipi,
        double[::1, :] Mpqs,
        np.ndarray[complex, ndim=2] Rpq,
        double[::1, :] Yiq,
        np.ndarray[complex] ws,
        np.ndarray[double] Zpq
                     ):
    cdef int p, q, k, j, S, diff, m, i, pp1
    cdef int N = 32 # A.shape[0]
    cdef double o_PI = 1.0/np.pi
    cdef double yiq, mpq, val, zrl, zim
    cdef int deg = 17
    cdef double complex w = -1.0j
    cdef double complex zpk, rpq, dw

    
    # Only iterate through black pixels    
    for i in range(0,N):
        for j in range(0,N):
            val = A[i,j]
            if not val:
                for q in range(0, deg+1):
                    Yiq[q,i] += Ipi[q,j]#*val
    
    for p in range(0,deg+1): # Could replace this loop with DGEMM
            
        for q in range(0,deg+1):
                
            mpq = 0.0
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
    
            ### The absolute value gives us the magnitude of the zernike moment.
            ### The zernike moment itself is pp1*o_PI*zpk and is complex
            Zpq[i] = abs(pp1*o_PI*zpk)
            i = i+1
