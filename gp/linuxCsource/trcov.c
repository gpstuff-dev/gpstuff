/* TRCOV     Evaluate covariance matrix.
 *
 *         Description
 *         C = GPEXPTRCOV(GPCF, TX) takes in Gaussian process GP and 
 *         matrix TX that contains input vectors to GP. Returns 
 *         covariance matrix C. Every element ij of C contains covariance 
 *         between inputs i and j in TX.
 *
 *
 * Last modified: 2009-07-03 09:22:31 EEST
 *
 */

/* Copyright (C) 1998-2001 Aki Vehtari
 * Copyright (C) 2008      Jarno Vanhatalo
 * 
 *This software is distributed under the GNU General Public 
 *License (version 2 or later); please refer to the file 
 *License.txt, included with the software, for details.
 *
 */

#include <stdlib.h>
#include <math.h>
#include "mex.h"
#define max(a,b) (((a) > (b)) ? (a) : (b))
void cumsum2 (mwIndex *p, mwIndex *c, mwIndex n);

void mexFunction(const int nlhs, mxArray *plhs[],
		 const int nrhs, const mxArray *prhs[])
{

  if (nlhs!=1 && nlhs!=3)
    mexErrMsgTxt( "Wrong number of output arguments.");
  
  if (nrhs!=2)
    mexErrMsgTxt( "Wrong number of input arguments." );
  
  {
    double *x, lms, ms, *l, rr, *rr2, *C, d, eps, c, *Ct, c1, c2;
    const int *dims;
    char *type;
    int i, j, k, m, n, lr, nnz, ind, D, D2, *p, *w, *w2;
    mwIndex *I, *J, *It, *Jt, *Jc;
    mxArray *field;
    
    dims = mxGetDimensions(prhs[1]);
    x = mxGetPr(prhs[1]);
    m = dims[0];
    n = dims[1];
    
    if((field=mxGetField(*prhs, 0, "magnSigma2"))==NULL)
      mexErrMsgTxt("Could not get gpcf.magnSigma2");
    dims = mxGetDimensions(field);
    if (dims[0]!=1 || dims[1]!=1)
      mexErrMsgTxt( "gpcf.magnSigma2 must be a scalar." );
    lms = log(mxGetScalar(field));
    ms = exp(lms);
    
    if((field=mxGetField(*prhs, 0, "lengthScale"))==NULL)
      mexErrMsgTxt("Could not get gpcf.lengthScale");
    dims = mxGetDimensions(field);
    if (dims[0]!=1 && dims[1]!=1)
      mexErrMsgTxt( "gpcf.lengthScale must be a scalar or a vector." );
    lr=max(dims[0],dims[1]);
    if (lr!=1 && lr!=n)
      mexErrMsgTxt( "gpcf.lengthScale must be scalar or its lenght must same as number of columns in X." );
    l = mxGetPr(field);

    if((field=mxGetField(*prhs, 0, "type"))==NULL)
      mexErrMsgTxt("Could not get gpcf.type");
    if (mxIsChar(field) !=1)
      mexErrMsgTxt( "gpcf.type must be a string." );
    type = mxArrayToString(field);

    
    /*
     * squared exponential covariance
     */
    if( strcmp( type, "gpcf_sexp" ) == 0 ) {
      plhs[0]=mxCreateDoubleMatrix(m, m, mxREAL);
      C = mxGetPr(plhs[0]);
      eps=mxGetEps();

      for (i=0;i<n;i++,x+=m) {
	rr=(lr>1)?(l[i]*l[i]):(l[0]*l[0]);
	for (j=0;j<m;j++) {
	  for (k=0;k<j;k++) {
	    d=x[j]-x[k];
	    C[j*m+k]+=d*d/rr;
	  }
	}
      }
      for (j=0;j<m;j++) {
	for (k=0;k<j;k++) {
	  d=exp(lms-C[j*m+k]);
	  d=(d>eps) ? d : 0;
	  C[j*m+k]=d;
	  C[j+k*m]=d;
	}
	C[j*(m+1)]=ms;
      }
    }
    /*
     * exponential covariance
     */
    else if(strcmp( type, "gpcf_exp" ) == 0 ){
      plhs[0]=mxCreateDoubleMatrix(m, m, mxREAL);
      C = mxGetPr(plhs[0]);
      eps=mxGetEps();

      for (i=0;i<n;i++,x+=m) {
	rr=(lr>1)?(l[i]*l[i]):(l[0]*l[0]);
	for (j=0;j<m;j++) {
	  for (k=0;k<j;k++) {
	    d=x[j]-x[k];
	    C[j*m+k]+=d*d/rr;
	  }
	}
      }
      for (j=0;j<m;j++) {
	for (k=0;k<j;k++) {
	  d=exp(lms-sqrt(C[j*m+k]));
	  d=(d>eps) ? d : 0;
	  C[j*m+k]=d;
	  C[j+k*m]=d;
	}
	C[j*(m+1)]=ms;
      }
    }
    /*
     * matern nu = 3/2 covariance
     */
    else if(strcmp( type, "gpcf_matern32" ) == 0 ){
      plhs[0]=mxCreateDoubleMatrix(m, m, mxREAL);
      C = mxGetPr(plhs[0]);
      eps=mxGetEps();

      for (i=0;i<n;i++,x+=m) {
	rr=(lr>1)?(l[i]*l[i]):(l[0]*l[0]);
	for (j=0;j<m;j++) {
	  for (k=0;k<j;k++) {
	    d=x[j]-x[k];
	    C[j*m+k]+=d*d/rr;
	  }
	}
      }
      for (j=0;j<m;j++) {
	for (k=0;k<j;k++) {
	  c = sqrt(3*C[j*m+k]);
	  d=(1+c)*exp(lms-c);
	  d=(d>eps) ? d : 0;
	  C[j*m+k]=d;
	  C[j+k*m]=d;
	}
	C[j*(m+1)]=ms;
      }
    }
    /*
     * matern nu = 5/2 covariance
     */
    else if(strcmp( type, "gpcf_matern52" ) == 0 ){
      plhs[0]=mxCreateDoubleMatrix(m, m, mxREAL);
      C = mxGetPr(plhs[0]);
      eps=mxGetEps();

      for (i=0;i<n;i++,x+=m) {
	rr=(lr>1)?(l[i]*l[i]):(l[0]*l[0]);
	for (j=0;j<m;j++) {
	  for (k=0;k<j;k++) {
	    d=x[j]-x[k];
	    C[j*m+k]+=d*d/rr;
	  }
	}
      }
      for (j=0;j<m;j++) {
	for (k=0;k<j;k++) {
	  c = sqrt(5*C[j*m+k]);
	  d=(1+c+5*C[j*m+k]/3)*exp(lms-c);
	  d=(d>eps) ? d : 0;
	  C[j*m+k]=d;
	  C[j+k*m]=d;
	}
	C[j*(m+1)]=ms;
      }
    }
    /*
     * piece wise polynomial 2 covariance
     */
    else if(strcmp( type, "gpcf_ppcs2" ) == 0 ){
      if((field=mxGetField(*prhs, 0, "l"))==NULL)
	mexErrMsgTxt("Could not get gpcf.l");
      dims = mxGetDimensions(field);
      if (dims[0]!=1 || dims[1]!=1)
	mexErrMsgTxt( "gpcf.l must be a scalar." );
      D = mxGetScalar(field);

      nnz = (int) max(1,floor(0.05*m*m));
      It = mxCalloc(nnz,sizeof(int));
      Jt = mxCalloc(nnz,sizeof(int));
      Ct = mxCalloc(nnz,sizeof(double));
      eps=mxGetEps();
      ind = 0;

      /* Set the length-scales in vector of length of number of inputs */
      rr2 = mxCalloc(n,sizeof(double));
      for (i=0;i<n;i++) {
	rr2[i]=(lr>1)?(l[i]*l[i]):(l[0]*l[0]);
      }	
      
      /* Evaluate the distances that are less than onem, */
      /* and evaluate the covariance at them. /*
      /* This is strictly upper triangular matrix */
      c1=pow(D,2)+4*D+3;
      c2=3*D+6;
      D2=D+2;
      for (j=0;j<m;j++) {
	for (k=0;k<j;k++) {
	  c = 0;
	  for (i=0;i<n;i++) {
	    d=x[j+i*m]-x[k+i*m];
	    c+=d*d/rr2[i];
	  }
	  if (c<1 && nnz>ind){   /* store the covariance */
	    d = c1*c + c2*sqrt(c) + 3;
	    d = ms*pow(1-sqrt(c),D2)*d/3;
	    d=(d>eps) ? d : 0;
	    It[ind] = k;
	    Jt[ind] = j;
	    Ct[ind] = d;
	    ind+=1;
	  } else if (c<1){       /* allocate more memory and store the covariance  */
	    nnz=2*nnz;
	    It = mxRealloc(It, (size_t)(nnz*sizeof(int)));
	    Jt = mxRealloc(Jt, (size_t)(nnz*sizeof(int)));
	    Ct = mxRealloc(Ct, (size_t)(nnz*sizeof(double)));
	    d = c1*c + c2*sqrt(c) + 3;
	    d = ms*pow(1-sqrt(c),D2)*d/3;
	    d=(d>eps) ? d : 0;
	    It[ind] = k;
	    Jt[ind] = j;
	    Ct[ind] = d;
	    ind+=1;
	  }
	}
      }
      
      /* resize the vectors */
      It = mxRealloc(It, (size_t)(ind*sizeof(int)));      
      Jt = mxRealloc(Jt, (size_t)(ind*sizeof(int)));
      Ct = mxRealloc(Ct, (size_t)(ind*sizeof(double)));
      
      /* evaluate the row and column counts */
      w = mxCalloc(m,sizeof(int));           /* workspace */
      w2 = mxCalloc(m,sizeof(int));          /* workspace */
      for (k=0;k<ind;k++) w[It[k]]++;        /* row counts of the upper triangular */
      for (k=0;k<ind;k++) w2[Jt[k]]++;       /* column counts of the upper triangular */
      for (k=0;k<m;k++) w[k] += w2[k] + 1;   /* column counts of the sparse inverse */
      Jc = mxCalloc((m+1),sizeof(int));
      cumsum2(Jc, w2, m);                    /* column starting points of the upper triangle */
      
      /* Create sparse matrix. Note! The matrix can contain only real numbers  */
      nnz = 2*ind+m;
      plhs[0] = mxCreateSparse(m,m,(mwSize)nnz,mxREAL);
      I = mxGetIr(plhs[0]);
      J = mxGetJc(plhs[0]);
      C = mxGetPr(plhs[0]);

      /* Set the elements in the sparse matrix */
      cumsum2(J, w, m);                   /* column starting points */
      for (j = 0 ; j < m ; j++){             /* fill the upper triangular */
	for (k = Jc[j] ; k < Jc[j+1] ; k++){
	  I[i = w[j]++] = It[k] ;
	  if (C) C[i] = Ct[k] ;
	}
      }
      for (j = 0 ; j < m ; j++){             /* fill the diagonal */
	I[i = w[j]++] = j ;
	if (C) C[i] = ms;
      } 
      for (j = 0 ; j < m ; j++){             /* fill the lower triangular */
	for (k = Jc[j] ; k < Jc[j+1] ; k++){
	  I[i = w[ It[k]]++] = j ;	    
	  if (C) C[i] = Ct[k] ;
	}
      }

      mxFree(It);
      mxFree(Jt);
      mxFree(Jc);
      mxFree(Ct);
      mxFree(rr2);
      mxFree(w);
      mxFree(w2);
      
    } else{
      mexErrMsgTxt( "Undefined type of covariance function." );
    }
  }
  return;
}     

void cumsum2 (mwIndex *p, mwIndex *c, mwIndex n)
{
  mwIndex i;
  mwIndex nz = 0;
  if(!p || !c) return;
  for (i=0;i<n;i++){
    p[i]=nz;
    nz+=c[i];
    c[i]=p[i];
  }
  p[n]=nz;

}
