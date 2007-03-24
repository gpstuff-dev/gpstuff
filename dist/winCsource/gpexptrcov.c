/* GPEXPTRCOV     Evaluate covariance matrix.
 *
 *         Description
 *         C = GPEXPTRCOV(GP, TX) takes in Gaussian process GP and 
 *         matrix TX that contains input vectors to GP. Returns 
 *         covariance matrix C. Every element ij of C contains covariance 
 *         between inputs i and j in TX.
 *
 *
 * Last modified: 2004-09-08 14:18:10 EEST
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

  if (nlhs!=1 && nlhs!=3) 
    mexErrMsgTxt( "Wrong number of output arguments.");
  
  if (nrhs!=2)
    mexErrMsgTxt( "Wrong number of input arguments." );
  
  {
    double *x, s, es, *r, rr, js, ns, *nv, *C, *c, *cc, d, eps;
    const int *dims;
    int i, j, k, m, n, lr;
    mxArray *field;
    
    dims = mxGetDimensions(prhs[1]);
    x = mxGetPr(prhs[1]);
    m = dims[0];
    n = dims[1];
    
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

    if((field=mxGetField(*prhs, 0, "jitterSigmas"))==NULL)
      mexErrMsgTxt("Could not get gp.jitterSigmas");
    dims = mxGetDimensions(field);
    if (dims[0]!=1 || dims[1]!=1)
      mexErrMsgTxt( "JS must be a scalar." );
    js = mxGetScalar(field);
    js*=js;
      
    if((field=mxGetField(*prhs, 0, "noiseVariances"))==NULL)
      mexErrMsgTxt("Could not get gp.noiseVariances");
    dims = mxGetDimensions(field);
    if (dims[0]==0 && dims[1]==0) {
      nv=NULL;
      if((field=mxGetField(*prhs, 0, "noiseSigmas"))==NULL)
	mexErrMsgTxt("Could not get gp.noiseSigmas");
      dims = mxGetDimensions(field);
      if (dims[0]==0 || dims[1]==0)
	ns=0;
      else {
	if (dims[0]!=1 || dims[1]!=1)
	  mexErrMsgTxt( "NS must be a scalar." );
	ns = mxGetScalar(field);
	ns*=ns;
      }
    } else {
      if (dims[0]!=1 && dims[1]!=1)
	mexErrMsgTxt( "NV must be a vector." );
      if (dims[1]!=m && dims[0]!=m)
	mexErrMsgTxt( "Length of NV must same as number of rows in X." );
      nv = mxGetPr(field);
    }

    plhs[0]=mxCreateDoubleMatrix(m, m, mxREAL);
    C = mxGetPr(plhs[0]);
    if (nlhs>1) {
      plhs[1]=mxCreateDoubleMatrix(m, m, mxREAL);
      c = mxGetPr(plhs[1]);
      if (lr>1) {
	int dim3[]={m,m,n};
	plhs[2]=mxCreateNumericArray(3, dim3, mxDOUBLE_CLASS, mxREAL);
      } else {
	plhs[2]=mxCreateDoubleMatrix(m, m, mxREAL);
      }
      cc = mxGetPr(plhs[2]);
    }

    eps=mxGetEps();

    if (nlhs==1) {
      for (i=0;i<n;i++,x+=m) {
	rr=(lr>1)?(r[i]*r[i]):(r[0]*r[0]);
	for (j=0;j<m;j++) {
	  for (k=0;k<j;k++) {
	    d=x[j]-x[k];
	    C[j*m+k]+=d*d*rr;
	  }
	}
      }
      if (nv==NULL) {
	for (j=0;j<m;j++) {
	  for (k=0;k<j;k++) {
	    d=exp(s-C[j*m+k]);
	    d=(d>eps) ? d : 0;
	    C[j*m+k]=d;
	    C[j+k*m]=d;
	  }
	  C[j*(m+1)]=es+js+ns;
	}
      } else {
	for (j=0;j<m;j++) {
	  for (k=0;k<j;k++) {
	    d=exp(s-C[j*m+k]);
	    d=(d>eps) ? d : 0;
	    C[j*m+k]=d;
	    C[j+k*m]=d;
	  }
	  C[j*(m+1)]=es+js+nv[j];
	}
      }      
    } else {
      for (i=0;i<n;i++,x+=m,cc+=(lr>1)?(m*m):(0)) {
	rr=(lr>1)?(r[i]*r[i]):(r[0]*r[0]);
	for (j=0;j<m;j++) {
	  for (k=0;k<j;k++) {
	    d=x[j]-x[k];
	    d*=d*rr;
	    C[j*m+k]+=d;
	    cc[j+k*m]=d;
	    cc[j*m+k]=d;
	  }
	}
      }
      if (nv==NULL) {
	for (j=0;j<m;j++) {
	  for (k=0;k<j;k++) {
	    d=exp(s-C[j*m+k]);
	    d=(d>eps) ? d : 0;
	    C[j*m+k]=d;
	    C[j+k*m]=d;
	    c[j*m+k]=d;
	    c[j+k*m]=d;
	  }
	  c[j*(m+1)]=es;
	  C[j*(m+1)]=es+js+ns;
	}
      } else {
	for (j=0;j<m;j++) {
	  for (k=0;k<j;k++) {
	    d=exp(s-C[j*m+k]);
	    d=(d>eps) ? d : 0;
	    C[j*m+k]=d;
	    C[j+k*m]=d;
	    c[j*m+k]=d;
	    c[j+k*m]=d;
	  }
	  c[j*(m+1)]=es;
	  C[j*(m+1)]+=es+js+nv[j];
	}
      }      
    }
  }
  return;
}     
