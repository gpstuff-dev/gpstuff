/* RESAMPSIM Simple random resampling
 *
 *   Description:
 *   S = RESAMPSIM(P) returns a new set of indices according to 
 *   the probabilities P. P is array of probabilities, which are
 *   not necessarily normalized, though they must be non-negative,
 *   and not all zero. The size of S is the size of P.
 *
 *   Note that residual, stratified and deterministic resampling all
 *   have smaller variance.
 *
 *   Simple random resampling samples indices randomly according
 *   to the probabilities P. See, e.g., Liu, J. S., Monte Carlo
 *   Strategies in Scientific Computing, Springer, 2001, p. 72.
 *
 *   See also RESAMPRES, RESAMPSTR, RESAMPDET
 *
 * Last modified: 2003-03-20 12:54:13 EET
 *
 */

/* Copyright (C) 2003 Aki Vehtari
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
    mxArray *MN[2];
    double *p, *q, *r; 
    const int *dims;
    int i, len;

    dims = mxGetDimensions(prhs[0]);
    len = dims[0]*dims[1];
    p = mxGetPr(prhs[0]);
    q = mxMalloc(len*sizeof(double));

    MN[0]=mxCreateDoubleScalar(dims[0]);
    MN[1]=mxCreateDoubleScalar(dims[1]);
    mexCallMATLAB(1,&plhs[0],2,MN,"rand");
    mxDestroyArray(MN[0]);
    mxDestroyArray(MN[1]);
    r = mxGetPr(plhs[0]);
    
    /* cumsum */
    q[0]=p[0];
    for (i = 0; i < len-1; i++) {
      q[i+1] = q[i]+p[i+1];
    }
    /* normalize */
    for (i = 0; i < len-1; i++)
      q[i]=q[i]/q[len-1];
    q[len-1]=1;
    
    /* generate values */
    for (i = 0; i < len; i++)
      r[i]=binsgeq(q,len,r[i]);

    mxFree(q);
  }

  return;
}
