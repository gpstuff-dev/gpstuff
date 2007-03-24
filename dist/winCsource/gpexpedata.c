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
#define max(a,b) (((a) > (b)) ? (a) : (b))
#ifndef M_PI
  #define M_PI       3.1415926535897932385E0  /*Hex  2^ 1 * 1.921FB54442D18 */
#endif

void mexFunction(const int nlhs, mxArray *plhs[],
		 const int nrhs, const mxArray *prhs[])
{

  if (nlhs>1)
    mexErrMsgTxt( "Wrong number of output arguments.");
  
  if (nrhs!=3)
    mexErrMsgTxt( "Wrong number of input arguments." );
  
  if(!mxIsStruct(prhs[0]))
    mexErrMsgTxt("First input must be a structure.");
  
  {
    double *x, *t, s, es, *r, rr, js, ns, *nv, *C, *c, d, *b, eps, *ed;
    const int *dims, one=1;
    int i, j, k, m, n, info, lr;
    char *uplo="L", *trans="N", *diag="N";
    mxArray *field;

    dims = mxGetDimensions(prhs[1]);
    x = mxGetPr(prhs[1]);
    m = dims[0];
    n = dims[1];
    
    dims = mxGetDimensions(prhs[2]);
    if (dims[0]!=m || dims[1]!=1)
      mexErrMsgTxt( "T must be M x 1 vector." );
    t = mxGetPr(prhs[2]);

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
    
    C=mxCalloc((m*(m+1))/2,sizeof(double));
    
    plhs[0]=mxCreateDoubleMatrix(1, 1, mxREAL);
    ed = mxGetPr(plhs[0]);
    
    b = mxCalloc(m,sizeof(double));
    for (i=0;i<m;i++)
      b[i]=t[i];
    
    eps=mxGetEps();

    for (i=0;i<n;i++,x+=m) {
      for (j=0,c=C,rr=(lr>1)?(r[i]*r[i]):(r[0]*r[0]);j<m;j++) {
	c++;
	for (k=j+1;k<m;k++) {
	  d=x[j]-x[k];
	  *c+=d*d*rr;
	  c++;
	}
      }
    }
    if (nv==NULL) {
      for (j=0,c=C;j<m;j++) {
	*c=es+js+ns;
	c++;
	for (k=j+1;k<m;k++) {
	  d=exp(s-*c);
	  *c=(d>eps) ? d : 0;
	  c++;
	}
      }
    } else {
      for (j=0,c=C;j<m;j++) {
	*c=es+js+nv[j];
	c++;
	for (k=j+1;k<m;k++) {
	  d=exp(s-*c);
	  *c=(d>eps) ? d : 0;
	  c++;
	}
      }
    }

    /* www.netlib.org/cgi-bin/netlibget.pl/lapack/double/dpotrf.f */
    dpptrf(uplo, &m, C, &info);
    /* www.netlib.org/blas/dtrsv.f */
    dtpsv(uplo, trans, diag, &m, C, b, &one);

    for (i=0,c=C,d=0;i<m;c+=m-i,i++) {
      d+=b[i]*b[i];
      *ed+=log(*c);
    }
    *ed+=d/2+m*log(2*M_PI)/2;
    
    mxFree(C);
    mxFree(b);

  }
  
  return;
}     
