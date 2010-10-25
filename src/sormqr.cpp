/*
    -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010
*/

#include "cuda_runtime_api.h"
#include "cublas.h"
#include "magma.h"
#include <stdio.h>

extern "C" int sorm2r_(char *, char *, int *, int *, int *, float *, int *, 
		       float *, float *, int *, float *, int *);

extern "C" magma_int_t
magma_sormqr(char side_, char trans_, magma_int_t m_, magma_int_t n_, 
	     magma_int_t k_, float *a, magma_int_t lda_, float *tau, float *c__, magma_int_t ldc_,
	     float *work, magma_int_t *lwork, magma_int_t *info)
{
/*  -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010

    Purpose   
    =======   
    SORMQR overwrites the general real M-by-N matrix C with   

                    SIDE = 'L'     SIDE = 'R'   
    TRANS = 'N':      Q * C          C * Q   
    TRANS = 'T':      Q**T * C       C * Q**T   

    where Q is a real orthogonal matrix defined as the product of k   
    elementary reflectors   

          Q = H(1) H(2) . . . H(k)   

    as returned by SGEQRF. Q is of order M if SIDE = 'L' and of order N   
    if SIDE = 'R'.   

    Arguments   
    =========   

    SIDE    (input) CHARACTER*1   
            = 'L': apply Q or Q**T from the Left;   
            = 'R': apply Q or Q**T from the Right.   

    TRANS   (input) CHARACTER*1   
            = 'N':  No transpose, apply Q;   
            = 'T':  Transpose, apply Q**T.   

    M       (input) INTEGER   
            The number of rows of the matrix C. M >= 0.   

    N       (input) INTEGER   
            The number of columns of the matrix C. N >= 0.   

    K       (input) INTEGER   
            The number of elementary reflectors whose product defines   
            the matrix Q.   
            If SIDE = 'L', M >= K >= 0;   
            if SIDE = 'R', N >= K >= 0.   

    A       (input) REAL array, dimension (LDA,K)   
            The i-th column must contain the vector which defines the   
            elementary reflector H(i), for i = 1,2,...,k, as returned by   
            SGEQRF in the first k columns of its array argument A.   
            A is modified by the routine but restored on exit.   

    LDA     (input) INTEGER   
            The leading dimension of the array A.   
            If SIDE = 'L', LDA >= max(1,M);   
            if SIDE = 'R', LDA >= max(1,N).   

    TAU     (input) REAL array, dimension (K)   
            TAU(i) must contain the scalar factor of the elementary   
            reflector H(i), as returned by SGEQRF.   

    C       (input/output) REAL array, dimension (LDC,N)   
            On entry, the M-by-N matrix C.   
            On exit, C is overwritten by Q*C or Q**T*C or C*Q**T or C*Q.   

    LDC     (input) INTEGER   
            The leading dimension of the array C. LDC >= max(1,M).   

    WORK    (workspace/output) REAL array, dimension (MAX(1,LWORK))   
            On exit, if INFO = 0, WORK(1) returns the optimal LWORK.   

    LWORK   (input) INTEGER   
            The dimension of the array WORK.   
            If SIDE = 'L', LWORK >= max(1,N);   
            if SIDE = 'R', LWORK >= max(1,M).   
            For optimum performance LWORK >= N*NB if SIDE = 'L', and   
            LWORK >= M*NB if SIDE = 'R', where NB is the optimal   
            blocksize.   

            If LWORK = -1, then a workspace query is assumed; the routine   
            only calculates the optimal size of the WORK array, returns   
            this value as the first entry of the WORK array, and no error   
            message related to LWORK is issued by XERBLA.   

    INFO    (output) INTEGER   
            = 0:  successful exit   
            < 0:  if INFO = -i, the i-th argument had an illegal value   

    =====================================================================   */

    #define min(a,b)  (((a)<(b))?(a):(b))
    #define max(a,b)  (((a)>(b))?(a):(b))
    
    char side[2] = {side_, 0};
    char trans[2] = {trans_, 0};
    int *m = &m_;
    int *n = &n_;
    int *k = &k_;
    int *lda = &lda_;
    int *ldc = &ldc_;

    static int c__1 = 1;
    static int c_n1 = -1;
    static int c__2 = 2;
    static int c__65 = 65;
    
    // TTT --------------------------------------------------------------------
    float *dwork, *dc;
    cublasAlloc((*m)*(*n), sizeof(float), (void**)&dc);
    cublasAlloc(2*(*m+64)*64, sizeof(float), (void**)&dwork);
    
    cublasSetMatrix( *m, *n, sizeof(float), c__, *ldc, dc, *ldc);
    dc -= (1 + *m);
    //-------------------------------------------------------------------------

    int a_dim1, a_offset, c_dim1, c_offset, i__4, i__5;
    /* Local variables */
    static int i__;
    static float t[2*4160]	/* was [65][64] */;
    static int i1, i2, i3, ib, ic, jc, nb, mi, ni, nq, nw, iws;
    long int left, notran, lquery;
    static int nbmin, iinfo;
    static int ldwork, lwkopt;

    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --tau;
    c_dim1 = *ldc;
    c_offset = 1 + c_dim1;
    c__ -= c_offset;
    --work;

    /* Function Body */
    *info = 0;
    left = lsame_(side, "L");
    notran = lsame_(trans, "N");
    lquery = *lwork == -1;

    /* NQ is the order of Q and NW is the minimum dimension of WORK */

    if (left) {
	nq = *m;
	nw = *n;
    } else {
	nq = *n;
	nw = *m;
    }
    if (! left && ! lsame_(side, "R")) {
	*info = -1;
    } else if (! notran && ! lsame_(trans, "T")) {
	*info = -2;
    } else if (*m < 0) {
	*info = -3;
    } else if (*n < 0) {
	*info = -4;
    } else if (*k < 0 || *k > nq) {
	*info = -5;
    } else if (*lda < max(1,nq)) {
	*info = -7;
    } else if (*ldc < max(1,*m)) {
	*info = -10;
    } else if (*lwork < max(1,nw) && ! lquery) {
	*info = -12;
    }

    if (*info == 0) 
      {
	/* Determine the block size.  NB may be at most NBMAX, where NBMAX   
	   is used to define the local array T.    */
	nb = 64;
	lwkopt = max(1,nw) * nb;
	work[1] = (float) lwkopt;
    }

    if (*info != 0) {
	return 0;
    } else if (lquery) {
	return 0;
    }

    /* Quick return if possible */

    if (*m == 0 || *n == 0 || *k == 0) {
	work[1] = 1.f;
	return 0;
    }

    nbmin = 2;
    ldwork = nw;
    if (nb > 1 && nb < *k) {
	iws = nw * nb;
	if (*lwork < iws) {
	    nb = *lwork / ldwork;
	    nbmin = 64;
	}
    } else {
	iws = nw;
    }

    if (nb < nbmin || nb >= *k) 
      {
	/* Use unblocked code */
	sorm2r_(side, trans, m, n, k, &a[a_offset], lda, &tau[1], 
		&c__[c_offset], ldc, &work[1], &iinfo);
      } 
    else 
      {
	/* Use blocked code */
	if (left && ! notran || ! left && notran) {
	    i1 = 1;
	    i2 = *k;
	    i3 = nb;
	} else {
	    i1 = (*k - 1) / nb * nb + 1;
	    i2 = 1;
	    i3 = -nb;
	}

	if (left) {
	    ni = *n;
	    jc = 1;
	} else {
	    mi = *m;
	    ic = 1;
	}
	
	for (i__ = i1; i3 < 0 ? i__ >= i2 : i__ <= i2; i__ += i3) 
	  {
	    /* Computing MIN */
	    i__4 = nb, i__5 = *k - i__ + 1;
	    ib = min(i__4,i__5);

	    /* Form the triangular factor of the block reflector   
	       H = H(i) H(i+1) . . . H(i+ib-1) */
	    i__4 = nq - i__ + 1;
	    slarft_("F", "C", &i__4, &ib, &a[i__ + i__ * a_dim1], lda, 
		    &tau[i__], t, &ib);

	    // TTT ------------------------------------------------------------
	    spanel_to_q('U', ib, &a[i__ + i__ * a_dim1], *lda, t+ib*ib);
	    cublasSetMatrix(i__4, ib, sizeof(float),
			    &a[i__ + i__ * a_dim1], *lda, 
			    dwork, i__4);
	    sq_to_panel('U', ib, &a[i__ + i__ * a_dim1], *lda, t+ib*ib);
	    //-----------------------------------------------------------------

	    if (left) 
	      {
		/* H or H' is applied to C(i:m,1:n) */
		mi = *m - i__ + 1;
		ic = i__;
	      } 
	    else 
	      {
		/* H or H' is applied to C(1:m,i:n) */
		ni = *n - i__ + 1;
		jc = i__;
	      }
	    
	    /* Apply H or H' */
	    // TTT ------------------------------------------------------------
	    //printf("%5d %5d %5d\n", mi, ni, ic + 1 + *m);
	    cublasSetMatrix(ib, ib, sizeof(float), t, ib, dwork+i__4*ib, ib);
	    magma_slarfb('F','C', mi, ni, ib,
			 dwork, i__4, dwork+i__4*ib, ib,
			 &dc[ic + jc * c_dim1], *ldc, 
			 dwork+i__4*ib + ib*ib, ni);
	    //-----------------------------------------------------------------
	    /*
	    slarfb_(side, trans, "Forward", "Columnwise", &mi, &ni, &ib, 
		    &a[i__ + i__ * a_dim1], lda, t, &c__65, 
		    &c__[ic + jc * c_dim1], ldc, &work[1], &ldwork);
	    */
	  }
      }
    work[1] = (float) lwkopt;

    dc += (1 + *m);
    cublasFree(dc);
    cublasFree(dwork);

    return 0;
    
/*     End of SORMQR */

} /* sormqr_ */

#undef min
#undef max

