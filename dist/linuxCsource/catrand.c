/* CATRAND Random matrices from categorical distribution.
 *
 *  R = CATRAND(P) returns a matrix of random numbers chosen   
 *  from the categorical distribution with parameter P.
 *  P is array of probabilities, which are not necessarily
 *  normalized, though they must be non-negative, and not all zero
 *  The size of R is the size of P. Alternatively,
 *  R = CATRAND(P,M,N) returns an M by N matrix.
 *
 * Copyright (C) 1999-2003 Aki Vehtari
 *
 * Last modified: 2010-06-16 14:41:31 EEST
 *
 */

/* 
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
  
  if ((nrhs < 1) | (nrhs > 3))
    mexErrMsgTxt( "Wrong number of input arguments." );

  {
    mxArray *MN[2];
    double *p, *q, *r; 
    const int *dims;
    int i, len, rlen;

    /* rand */
    if (nrhs < 2) {
      rlen=1;
      mexCallMATLAB(1,&plhs[0],0,NULL,"rand");
    }
    else if (nrhs < 3) {
      rlen=(int)mxGetScalar(prhs[1]);
      MN[0]=(mxArray *) prhs[1];
      MN[1]=mxCreateDoubleScalar(1.0);
      mexCallMATLAB(1,&plhs[0],2,MN,"rand");
      mxDestroyArray(MN[1]);
    }
    else {
      rlen=(int)mxGetScalar(prhs[1]) * (int)mxGetScalar(prhs[2]);
      MN[0]=(mxArray *) prhs[1];
      MN[1]=(mxArray *) prhs[2];
      mexCallMATLAB(1,&plhs[0],2,MN,"rand");
    }

    r = mxGetPr(plhs[0]);
    
    dims = mxGetDimensions(prhs[0]);
    len = dims[0]*dims[1];
    p = mxGetPr(prhs[0]);
    q = mxMalloc(len*sizeof(double));

    /* cumsum */
    q[0]=p[0];
    for (i = 0; i < len-1; i++) {
      if (p[i]<0) 
 	mexErrMsgTxt( "Negative probability." ); 
      q[i+1] = q[i]+p[i+1];
    }

    /* normalize */
    for (i = 0; i < len-1; i++)
      q[i]=q[i]/q[len-1];
    q[len-1]=1;
    
    /* generate values */
     for (i = 0; i < rlen; i++) 
       r[i]=binsgeq(q,len,r[i]); 

    mxFree(q);
  }

  return;
}
