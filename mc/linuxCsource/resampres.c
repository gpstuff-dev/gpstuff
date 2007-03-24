/*RESAMPRES Residual resampling
 *
 *   Description:
 *   S = RESAMPRES(P) returns a new set of indices according to 
 *   the probabilities P. P is array of probabilities, which are
 *   not necessarily normalized, though they must be non-negative,
 *   and not all zero. The size of S is the size of P.
 *
 *   Note that stratified and deterministic resampling have smaller
 *   variance.
 *
 *   Residual resampling retains k_j=floor(n*p_j) copies of index j,
 *   where n is length of P and p is normalized probabilities.
 *   Additionally n_r=n-k_1-...-k_n indices are sampled with
 *   probabilities proportional to n*p_j-k_j. See, e.g., Liu, J. S.,
 *   Monte Carlo Strategies in Scientific Computing, Springer, 2001,
 *   p. 72.
 *
 * Last modified: 2003-03-20 12:54:03 EET
 *
 */

/* Copyright (C) 1999-2003 Aki Vehtari
 * 
 *This software is distributed under the GNU General Public 
 *License (version 2 or later); please refer to the file 
 *License.txt, included with the software, for details.
 *
 */
 
#include "mex.h"
#include "binsgeq.h"

void mexFunction(int nlhs, mxArray *plhs[],
		 int nrhs, const mxArray *prhs[])
{

  if (nlhs > 1 ) 
    mexErrMsgTxt( "Too many output arguments." );
  
  if (nrhs != 1)
    mexErrMsgTxt( "Wrong number of input arguments." );

  {
    mxArray *MN[2], *R;
    double *p, *q, *r, *s, sum=0; 
    const int *dims;
    int i, len, rlen, a, b;

    dims = mxGetDimensions(prhs[0]);
    len = dims[0]*dims[1];
    rlen = len;
    p = mxGetPr(prhs[0]);
    q = mxMalloc(len*sizeof(double));

    plhs[0]=mxCreateDoubleMatrix(dims[0],dims[1],mxREAL);
    s = mxGetPr(plhs[0]);

    /* integer part */
    for (i = 0; i < len; i++)
      sum+=p[i];
    for (i = 0; i < len; i++) {
      q[i]=p[i]/sum*len;
      if (q[i]>=1.0) {
	a=(int)(q[i]);
	q[i]-=a;
	rlen-=a;
	for (b=i+1.0;a>0;a--)
	  *s++=b;
      }
    }
    /* normalize residual */
    for (i = 0; i < len; i++) {
      q[i]=q[i]/rlen;
    }

    /* rand */
    MN[0]=mxCreateDoubleScalar((double)rlen);
    MN[1]=mxCreateDoubleScalar(1.0);
    mexCallMATLAB(1,&R,2,MN,"rand");
    mxDestroyArray(MN[0]);
    mxDestroyArray(MN[1]);
    r = mxGetPr(R);

    /* cumsum residual */
    for (i = 0; i < len-1; i++)
      q[i+1] = q[i]+q[i+1];
    q[len-1]=1;
    
    /* generate values from residual */
    for (i = 0; i < rlen; i++)
      *s++=binsgeq(q,len,r[i]);

    mxFree(q);
    mxDestroyArray(R); 
  }

  return;
}
