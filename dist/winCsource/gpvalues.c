/* GPVALUES Sample latent values
 *	Y = GPVALUES(GP, TX, TY) takes a gp data structure GP
 *	together with a matrix TX of input vectors, and an output
 *      vector TY and returns samples from latent value distribution.
 *
 *	[Y, invC] = GPVALUES(GP, TX, TY) also returns the inverse of
 *      covariance matrix.
 *
 *	Y = GPVALUES(GP, TX, TY, invC) uses ready computed inverse
 *	of covariance matrix.
 *
 *      See gpvalues.m for same in m-code.
 *
 *
 * Last modified: 2006-01-05 15:48:53 EET
 *
 */

/* Copyright (C) 1998-2004 Aki Vehtari
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

  /* Check arguments */
  if (nlhs>2)
    mexErrMsgTxt( "Wrong number of output arguments.");
  
  if (nrhs<3 || nrhs>4)
    mexErrMsgTxt( "Wrong number of input arguments." );
  
  if(!mxIsStruct(prhs[0]))
    mexErrMsgTxt("First input must be a structure.");
  
  {
    const double *x, *t, *r, *nv, dzero=0.0, done=1.0, dmone=-1.0;
    double s, es, js, ns=0.0, rr, *C, *Cl, d, *b, *R, *lcov, *y, *ly, eps;
    const int *dims, one=1;
    int i, j, k, m, n, info, lr;
    char *uplo="L", *trans="N", *diag="N", *sidel="L", *sider="R";
    mxArray *field, *MN[2];

    /* TX */
    dims = mxGetDimensions(prhs[1]);
    x = mxGetPr(prhs[1]);
    m = dims[0];
    n = dims[1];

    /* TY */
    dims = mxGetDimensions(prhs[2]);
    if (dims[0]!=m || dims[1]!=1)
      mexErrMsgTxt( "TY must be M x 1 vector." );
    t = mxGetPr(prhs[2]);

    /* Optional invC, computation of this takes most of the time,
       so sometimes we can save time by reusing it */
    if (nrhs>3) {
      dims = mxGetDimensions(prhs[3]);
      if (dims[0]!=m && dims[1]!=m)
	mexErrMsgTxt( "invC must be MxM matrix." );
      C=mxGetPr(prhs[3]);
    } else {
      C=NULL;
    }

    /* Parameters of the covariance function */
    /* Note that only simple quadratic covariance function is supported */
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

    /* We need 'm' random numbers, MN is used only temporarily */
    /* Store these to sampled latent value vector */ 
    MN[0]=mxCreateDoubleScalar((double)m);
    MN[1]=mxCreateDoubleScalar(1.0);
    mexCallMATLAB(1,&plhs[0],2,MN,"randn");
    mxDestroyArray(MN[0]);
    mxDestroyArray(MN[1]);
    /* Pointer to sampled latent value vector, note that currently
       ranodm numbers are stored here */
    ly = mxGetPr(plhs[0]);
    
    if (nlhs>1) {
      /* Need to return invC too */
      if (C!=NULL)
	/* Oh well, just passing through */
	plhs[1]=mxDuplicateArray(prhs[3]);
      else
	plhs[1]=mxCreateDoubleMatrix(m, m, mxREAL);
    }

    /* very small number */
    eps=mxGetEps();
    
    /* First compute invC and Cl */
    if (nrhs<4) {
      /* No invC given */
      if (nlhs>1)
	/* if returning invC, we have already allocated memory for it */
	C=mxGetPr(plhs[1]);
      else
	/* otherwise allocate memory temorarily */
	C=mxCalloc(m*m,sizeof(double));
      /* covariance matrix for latent values */
      Cl=mxCalloc(m*m,sizeof(double));
      /* quadratic part r=expSigmas, lr is length of expSigmas */
      for (i=0;i<n;i++,x+=m) {
	for (j=0,rr=(lr==1)?(r[0]*r[0]):(r[i]*r[i]);j<m;j++) {
	  for (k=j+1;k<m;k++) {
	    d=x[j]-x[k];
	    d*=d*rr;
	    C[j*m+k]+=d;
	    Cl[j*m+k]+=d;
	  }
	}
      }
      /* quadratic part s=expScale, jitter and noise */
      for (j=0;j<m;j++) {
	for (k=j+1;k<m;k++) {
	  d=exp(s-C[j*m+k]);
	  d=(d>eps) ? d : 0;
	  C[j*m+k]=d;
	  Cl[j*m+k]=d;
	}
	/* jitter is added to both, but noise only to C */
	/* ns is for homoskedastic and nv is for heteroskedastic noise */
	C[j*(m+1)]=es+js+ (nv==NULL ? ns : nv[j]);
	Cl[j*(m+1)]=es+js;
      }
      /* compute lower triangular half of the inverse of C */
      dpotrf(uplo, &m, C, &m, &info);
      dpotri(uplo, &m, C, &m, &info);
      /* fill the upper triangular part */
      for (j=0;j<m;j++) {
	for (k=j+1;k<m;k++) {
	  C[j+k*m]=C[j*m+k];
	}
      }
    } else {
      /* invC given */
      /* covariance matrix for latent values */
      Cl=mxCalloc(m*m,sizeof(double));
      /* quadratic part r=expSigmas */
      for (i=0;i<n;i++,x+=m) {
	for (j=0,rr=(lr==1)?(r[0]*r[0]):(r[i]*r[i]);j<m;j++) {
	  for (k=j+1;k<m;k++) {
	    d=x[j]-x[k];
	    Cl[j*m+k]+=d*d*rr;;
	  }
	}
      }
      /* quadratic part s=expScale and jitter */
      for (j=0;j<m;j++) {
	for (k=j+1;k<m;k++) {
	  d=exp(s-Cl[j*m+k]);
	  Cl[j*m+k]=(d>eps) ? d : 0; 
	}
	Cl[j*(m+1)]=es+js;
      }
    }
    /* prepare to compute following */
    /* R=Cl'*invC; */
    /* y=R*ty; */
    /* lcov=Cl-R*Cl; */
    /* ly=y+chol(lcov)'*randn(size(y)); */
    /* note that actual computing has been arranged in slightly different way */
    R=mxCalloc(m*m,sizeof(double));
    lcov=mxCalloc(m*m,sizeof(double));
    /* lcov = Cl */
    for (i=0;i<m*m;i++) 
      lcov[i]=Cl[i];
    /* b = ty */
    b=mxCalloc(m,sizeof(double));
    for (i=0;i<m;i++) 
      b[i]=t[i];
    y=mxCalloc(m,sizeof(double));
    /* R = Cl'*invC */
    dsymm(sidel, uplo, &m, &m, &done, Cl, &m, C, &m, &dzero, R, &m);
    /* y=R*b */
    dgemv(trans, &m, &m, &done, R, &m, b, &one, &dzero, y, &one);
    /* lcov=lcov-R*Cl */
    dsymm(sider, uplo, &m, &m, &dmone, Cl, &m, R, &m, &done, lcov, &m);
    /* lcov=chol(lcov) */
    dpotrf(uplo, &m, lcov, &m, &info);
    /* ly=lcov'*ly (remember that ly contains the random numbers) */
    dtrmv(uplo, trans, diag, &m, lcov, &m, ly, &one);
    /* ly = ly + y */
    for (i=0;i<m;i++) 
       ly[i]+=y[i]; 

    /* let the bits be free */
    mxFree(Cl);
    mxFree(R);
    mxFree(lcov);
    mxFree(b);
    if (nlhs<2 && nrhs<4)
      mxFree(C);
    
  }
  
  return;
}     
