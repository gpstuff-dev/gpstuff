/* GPEXPEDATA     Evaluate error function for gp.
 *
 *     Description
 *	E = GPEXPEDATA(GP, P, T) takes a gp data structure GP together
 *      with a matrix P of input vectors and a matrix T of target vectors,
 *	and evaluates the error function E. The choice of error function
 *	corresponds to the output unit activation function. Each row of P
 *	corresponds to one input vector and each row of T corresponds to one
 *	target vector.
 *
 *      [E, EDATA, EPRIOR] = GP2R_E(GP, P, T) also returns the data and
 * 	prior components of the total error.
 *
 *       See also
 *	GP2, GP2PAK, GP2UNPAK, GP2FWD, GP2R_G
 *
 * Last modified: 2003-10-24 15:46:09 EEST
 *
 */

/* Copyright (C) 1998-2001 Aki Vehtari
 * 
 *This software is distributed under the GNU General Public 
 *License (version 2 or later); please refer to the file 
 *License.txt, included with the software, for details.
 *
 */

#include <math.h>
#include "mex.h"

void mexFunction(const int nlhs, mxArray *plhs[],
		 const int nrhs, const mxArray *prhs[])
{

  if (nlhs>1)
    mexErrMsgTxt( "Wrong number of output arguments.");
  
  if (nrhs!=2)
    mexErrMsgTxt( "Wrong number of input arguments." );
  
  {
    const double *x, *A, dzero=0.0, done=1.0;
    double *y;
    const int *dims, one=1;
    int n;
    char *uplo="L";

    /* y=A*x, where A symmetric */
    dims = mxGetDimensions(prhs[0]);
    A = mxGetPr(prhs[0]);
    n = dims[0];
    x = mxGetPr(prhs[1]);
    plhs[0]=mxCreateDoubleMatrix(n, 1, mxREAL);
    y = mxGetPr(plhs[0]);
    dsymv_(uplo, &n, &done, A, &n, x, &one, &dzero, y, &one);

  }
  
  return;
}     
