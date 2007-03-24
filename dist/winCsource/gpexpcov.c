/* GPEXPTRCOV     Evaluate covariance matrix.
 *
 *  Description
 *  C = GPEXPTRCOV(GP, TX) takes in Gaussian process GP and 
 *  matrix TX that contains input vectors to GP. Returns 
 *  covariance matrix C. Every element ij of C contains covariance 
 *  between inputs i and j in TX.
 *
 *  Last modified: 2004-09-07 13:55:55 EEST
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
#define max(a,b) (((a) > (b)) ? (a) : (b))

void mexFunction(const int nlhs, mxArray *plhs[],
		 const int nrhs, const mxArray *prhs[])
{

  if (nlhs > 1) 
    mexErrMsgTxt( "Too many output arguments.");
  
  if (nrhs<2 || nrhs>4)
    mexErrMsgTxt( "Wrong number of input arguments." );
  
  {
    double *tx, *x=NULL, s, es, *r, rr, *K, d, eps;
    const int *dims;
    int i, j, k, m, n, m2, lr;
    mxArray *field;
    
    dims = mxGetDimensions(prhs[1]);
    tx = mxGetPr(prhs[1]);
    m = dims[0];
    n = dims[1];

    if (nrhs>2) {
      dims = mxGetDimensions(prhs[2]);
      if (dims[0]==0 && dims[1]==0) {
	x=NULL;
	m2=m;
      }
      else if (dims[1]!=n)
	mexErrMsgTxt( "TX and X must have same number of columns." );
      else {
	x=mxGetPr(prhs[2]);
	m2 = dims[0];
      }
    } else {
      x=NULL;
      m2=m;
    }
    
    if((field=mxGetField(*prhs, 0, "expScale"))==NULL)
      mexErrMsgTxt("Could not get gp.expScale");
    dims = mxGetDimensions(field);
    if (dims[0]!=1 || dims[1]!=1)
      mexErrMsgTxt( "S must be a scalar." );
    s = 2*log(mxGetScalar(field));
    es=exp(s);
    
    if((field=mxGetField(*prhs, 0, "expSigmas"))==NULL)
      mexErrMsgTxt("Could not get gp.expSigmas");
    dims = mxGetDimensions(field);
    if (dims[0]!=1 && dims[1]!=1)
      mexErrMsgTxt( "R must be a scalar or a vector." );
    lr=max(dims[0],dims[1]);
    if (lr!=1 && lr!=n)
      mexErrMsgTxt( "R must be scalar or length of R must same as number of columns in X." );
    r = mxGetPr(field);
    
    plhs[0]=mxCreateDoubleMatrix(m, m2, mxREAL);
    K = mxGetPr(plhs[0]);

    eps=mxGetEps();

    if (x==NULL) {
      for (i=0;i<n;i++,tx+=m) {
	rr=(lr>1)?(r[i]*r[i]):(r[0]*r[0]);
	for (j=0;j<m;j++) {
	  for (k=0;k<j;k++) {
	    d=tx[j]-tx[k];
	    K[j+k*m]+=d*d*rr;
	  }
	}
      }
      for (j=0;j<m;j++) {
	for (k=0;k<j;k++) {
	  d=exp(s-K[j+k*m]);
	  K[k+j*m]=K[j+k*m]=(d>eps) ? d : 0;
	}
	K[j*(m+1)]=es;
      }
    } else {
      for (i=0;i<n;i++,tx+=m,x+=m2) {
	rr=(lr>1)?(r[i]*r[i]):(r[0]*r[0]);
	for (j=0;j<m;j++) {
	  for (k=0;k<m2;k++) {
	    d=tx[j]-x[k];
	    K[j+k*m]+=d*d*rr;
	  }
	}
      }
      for (j=0;j<m;j++) {
	for (k=0;k<m2;k++) {
	  d=exp(s-K[j+k*m]);
	  K[j+k*m]=(d>eps) ? d : 0;
	}
      }
    }
  }
  return;
}     
