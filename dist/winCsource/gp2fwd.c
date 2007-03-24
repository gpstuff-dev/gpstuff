
/* GP2FWD	Forward propagation through Gaussian Process
 *
 *	Description
 *	Y = GP2FWD(GP, P, T, X) takes a gp data structure GP together with a
 *	matrix X of input vectors, Matrix P of training inputs and vector T of 
 *      targets, and forward propagates the inputs through the gp to generate 
 *      a matrix Y of output vectors. Each row of X corresponds to one input 
 *      vector and each row of Y corresponds to one output vector.
 *
 *      BUGS: - only exp2 covariance function is supported
 *            - only 1 output allowed
 *            - only mean values
 *
 *	See also
 *	GP2, GP2PAK, GP2UNPAK, 
 *
 * Last modified: 2006-01-19 10:01:50 EET
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

  if (nlhs>1)
    mexErrMsgTxt( "Wrong number of output arguments.");
  
  if (nrhs<3 || nrhs>5)
    mexErrMsgTxt( "Wrong number of input arguments." );
  
  if(!mxIsStruct(prhs[0]))
    mexErrMsgTxt("First input must be a structure.");
  
  {
    /* Initialize the parameters
     */
    const double *tx, *tx0, *ty, *x=NULL, *r, *nv, dzero=0.0, done=1.0;
    double s, es, js, ns=0, rr, *K, *kp, *C, *cp, d, *b, *y, eps;
    const int *dims, one=1;
    int i, j, k, m, n, m2, info, lr;
    char *uplo="L", *trans="N";
    mxArray *field;

    dims = mxGetDimensions(prhs[1]); /* Dimensions of training inputs matrix */
    tx = mxGetPr(prhs[1]);           /* Training inputs matrix */
    m = dims[0];                     /* The number of training input vectors */
    n = dims[1];                     /* The lenght of training input vector */
    tx0=tx;
    
    dims = mxGetDimensions(prhs[2]);
    if (dims[0]!=m || dims[1]!=1)
      mexErrMsgTxt( "TY must be M x 1 vector." );
    ty = mxGetPr(prhs[2]);           /* Training outputs matrix */

    /* Check that the new input and training inputs have same dimensionality */
    if (nrhs>3) {
      dims = mxGetDimensions(prhs[3]);
      if (dims[0]==0 && dims[1]==0) {
	x=NULL;
	m2=m;
      }
      else if (dims[1]!=n)
	mexErrMsgTxt( "TX and X must have same number of columns." );
      else {
	x=mxGetPr(prhs[3]);
	m2 = dims[0];               /* number of new inputs */
      }
    } else {
      x=NULL;
      m2=m;
    }

    /* If inverse of correlation matrix is given set it to C */ 
    if (nrhs>4) {
      dims = mxGetDimensions(prhs[4]);
      if (dims[0]!=m && dims[1]!=m)
	mexErrMsgTxt( "invC must be MxM matrix." );
      C=mxGetPr(prhs[4]);
    } else {
      C=NULL;
    }

    /* Check that field expScale is in the gp structure and set it to es */
    if((field=mxGetField(*prhs, 0, "expScale"))==NULL)
      mexErrMsgTxt("Could not get gp.expScale");
    dims = mxGetDimensions(field);
    if (dims[0]!=1 || dims[1]!=1)
      mexErrMsgTxt( "S must be a scalar." );
    s = 2*log(mxGetScalar(field));    /* logarithm of expScale */
    es=exp(s);

    /* Check that field expSigmas is in the gp structure and set it to r */
    if((field=mxGetField(*prhs, 0, "expSigmas"))==NULL)
      mexErrMsgTxt("Could not get gp.expSigmas");
    dims = mxGetDimensions(field);
    if (dims[0]!=1 && dims[1]!=1)
      mexErrMsgTxt( "R must be a scalar or a vector." );
    lr=max(dims[0],dims[1]);          /* lr>1 is ARD is used */
    if (lr!=1 && lr!=n)
      mexErrMsgTxt("R must be scalar or length of R must same as number of columns in X.");
    r = mxGetPr(field);               /* espSigmas */

    /* Check that field jitterSigmas is in the gp structure and set it to js */
    if((field=mxGetField(*prhs, 0, "jitterSigmas"))==NULL)
      mexErrMsgTxt("Could not get gp.jitterSigmas");
    dims = mxGetDimensions(field);
    if (dims[0]!=1 || dims[1]!=1)
      mexErrMsgTxt( "JS must be a scalar." );
    js = mxGetScalar(field);
    js*=js;                           /* jitterSigmas */

    /* Check that field noiseVariances is in the gp structure and set it to nv */
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
	ns*=ns;                      /* noiseSigmas */
      }
    } else {
      if (dims[0]!=1 && dims[1]!=1)
	mexErrMsgTxt( "NV must be a vector." );
      if (dims[1]!=m && dims[0]!=m)
	mexErrMsgTxt( "Length of NV must same as number of rows in X." );
      nv = mxGetPr(field);          /* noiseVariances */
    }

    /* Create an output vector for GP */
    plhs[0]=mxCreateDoubleMatrix(m2, 1, mxREAL);
    y = mxGetPr(plhs[0]);          /* Output vector of GP */

    /* The value of Matlab eps variable */
    eps=mxGetEps();

    if (x!=NULL) {
      /* Cases function is called either
	 g=gp2fwd(gp, p, t, pp);   or
	 g=gp2fwd(gp, p, t, pp, invC);*/
      
      /* Evaluate the covariance between new input and training inputs == K*/
      K=mxCalloc(m*m2,sizeof(double));
      /* First evaluate the exponential part of the covariance function */
      if (lr>1) {          /* Do this if ARD is used */
	for (i=0;i<n;i++,tx+=m,x+=m2) {
	  for (j=0,rr=r[i]*r[i];j<m;j++) {
	    for (k=0;k<m2;k++) {
	      d=tx[j]-x[k];
	      K[j*m2+k]+=d*d*rr;
	    }
	  }
	}
      }else {              /* Do this if ARD is not used */
	rr=r[0]*r[0];
	for (i=0;i<n;i++,tx+=m,x+=m2) {
	  for (j=0;j<m;j++) {
	    for (k=0;k<m2;k++) {
	      d=tx[j]-x[k];
	      K[j*m2+k]+=d*d*rr;
	    }
	  }
	}
      }
      /* Evaluate the rest of the covariance between new input and training inputs
       (s == log of expScale) */
      for (j=0;j<m;j++) {
	for (k=0;k<m2;k++) {
	  d=exp(s-K[j*m2+k]);
	  K[j*m2+k]=(d>eps) ? d : 0;
	}
      }
      /* Evaluation of the expectation of output */ 

      /* Case: function is called as g=gp2fwd(gp, p, t, pp, invC);
	 Here Inverse of the covariance matrix between training inputs is given
	 in C, evaluate the expectation of new ouput, y=K'*(C*ty). */ 
      if (C!=NULL) {
	b=mxCalloc(m,sizeof(double));    /* PITÄISIKÖ OLLA m2 */
	/* Evaluates b=C*ty */
	dsymv(uplo, &m, &done, C, &m, ty, &one, &dzero, b, &one);
	/* evaluates y=K*b */
 	dgemv(trans, &m2, &m, &done, K, &m, b, &one, &dzero, y, &one);   
	mxFree(b);
      }
      
      /* Case: function is called as g=gp2fwd(gp, p, t, pp);
	 Here the inverse of the covariance matrix between training inputs C
	 is not given, first evaluate C.  */
      else {
	/* Evaluate the covariance matrix of training inputs */
	x=tx0;
	C=mxCalloc((m*(m+1))/2,sizeof(double));         /* Miksi tällainen muistin varaus ? */
	/* First evaluate the exponential part */
	if (lr>1) {             /* Do this if ARD is used */
	  for (i=0;i<n;i++,x+=m) {
	    for (j=0,cp=C,rr=r[i]*r[i];j<m;j++) {
	      cp++;
	      for (k=j+1;k<m;k++) {
		d=x[j]-x[k];
		*cp+=d*d*rr;
		cp++;
	      }
	    }
	  }
	} else {               /* Do this if ARD is not used */
	  rr=r[0]*r[0];
	  for (i=0;i<n;i++,x+=m) {
	    for (j=0,cp=C;j<m;j++) {
	      cp++;
	      for (k=j+1;k<m;k++) {
		d=x[j]-x[k];
		*cp+=d*d*rr;
		cp++;
	      }
	    }
	  }
	}
	/* Evaluate the rest of the covariance  of training inputs (s == log of expScale) */
	if (nv==NULL) {             /* Do this if noise variances are not given */
	  for (j=0,cp=C;j<m;j++) {
	    *cp=es+js+ns;
	    cp++;
	    for (k=j+1;k<m;k++) {
	      d=exp(s-*cp);
	      *cp=(d>eps) ? d : 0;
	      cp++;
	    }
	  }
	} else {                    /* Do this if noise variances are given */
	  for (j=0,cp=C;j<m;j++) {
	    *cp=es+js+nv[j];
	    cp++;
	    for (k=j+1;k<m;k++) {
	      d=exp(s-*cp);
	      *cp=(d>eps) ? d : 0;
	      cp++;
	    }
	  }
	}
	/* evaluate the expectation of new output, y=K'*(C*ty). */
	b=mxCalloc(m,sizeof(double));
	for (i=0;i<m;i++)
	  b[i]=ty[i];
	/* Evaluates C <- Chol(C) (lower triangular) */
	dpptrf(uplo, &m, C, &info);
	/* Evaluates b <- inv(C)*b  (solves C*x= b to b) */
	dpptrs(uplo, &m, &one, C, b, &m, &info);
	/* Evaluates y=K*b
	   NOTE: Above in evaluation of K the we get K' */
 	dgemv(trans, &m2, &m, &done, K, &m2, b, &one, &dzero, y, &one);
	mxFree(C);
	mxFree(b);
      }
      mxFree(K);
    }else { /* x==NULL */
      /* Cases function is called either
	 g=gp2fwd(gp, p, t, []);   or
	 g=gp2fwd(gp, p, t, [], invC);*/
      
      
      /* Case: function is called as g=gp2fwd(gp, p, t, [], InvC);
	 Here the inverse of the covariance matrix of training inputs C is given. */ 
      if (C!=NULL) {
	/* Evaluate the covariance function between "new inputs" and training inputs.
	 Notice: Same inputs are used as new and training set. Although the covariance
	 function between new-training and training-training is different. */
	x=tx0;
	K=mxCalloc((m*(m+1))/2,sizeof(double));
	/* Evaluate the exponential part */
	if (lr>1) {                 /* Do this if ARD is used */
	  for (i=0;i<n;i++,x+=m) {
	    for (j=0,kp=K,rr=r[i]*r[i];j<m;j++) {
	      kp++;
	      for (k=j+1;k<m;k++) {
		d=x[j]-x[k];
		*kp+=d*d*rr;
		kp++;
	      }
	    }
	  }
	} else {                   /* Do this if ARD is not used */
	  rr=r[0]*r[0];
	  for (i=0;i<n;i++,x+=m) {
	    for (j=0,kp=K;j<m;j++) {
	      kp++;
	      for (k=j+1;k<m;k++) {
		d=x[j]-x[k];
		*kp+=d*d*rr;
		kp++;
	      }
	    }
	  }
	}
	/* Evaluate rest of the covariance */
	for (j=0,kp=K;j<m;j++) {
	  *kp=es;
	  kp++;
	  for (k=j+1;k<m;k++) {
	    d=exp(s-*kp);
	    *kp=(d>eps) ? d : 0;
	    kp++;
	  }
	}
	/* evaluate the expectation of new ouput, y=K'*(C*ty). */ 
	b=mxCalloc(m,sizeof(double));
	/* Evaluates b=C*ty */
	dsymv(uplo, &m, &done, C, &m, ty, &one, &dzero, b, &one);
	/* Evaluates y=K*b
	   NOTE: Here K is (exceptionally) symmentric mxm matrix */
 	dspmv(uplo, &m, &done, K, b, &one, &dzero, y, &one);
	mxFree(K);
	mxFree(b);
      }
      
      /* Case: function is called as g=gp2fwd(gp, p, t, []);
	 Here the inverse of the covariance matrix of training inputs C
	 is not given. */ 
      else {
	/* First evaluate C and K */
	x=tx0;
	C=mxCalloc((m*(m+1))/2,sizeof(double));
	K=mxCalloc((m*(m+1))/2,sizeof(double));
	/* Evaluate the exponential part of covariance between
	   new inputs and training inputs K */
	if (lr>1) {                /* Do this if ARD is used */  
	  for (i=0;i<n;i++,x+=m) {
	    for (j=0,cp=C,kp=K,rr=r[i]*r[i];j<m;j++) {
	      cp++;
	      kp++;
	      for (k=j+1;k<m;k++) {
		d=x[j]-x[k];
		d*=d*rr;
		*cp+=d;
		cp++;
		*kp+=d;
		kp++;
	      }
	    }
	  }
	} else {                   /* Do this if ARD is not used */
	  rr=r[0]*r[0];
	  for (i=0;i<n;i++,x+=m) {
	    for (j=0,cp=C,kp=K;j<m;j++) {
	      cp++;
	      kp++;
	      for (k=j+1;k<m;k++) {
		d=x[j]-x[k];
		d*=d*rr;
		*cp+=d;
		cp++;
		*kp+=d;
		kp++;
	      }
	    }
	  }
	}
	/* Evaluate the exponential part of covariance between
	   training inputs C */
	if (nv==NULL) {                /* Do this if noise variances are given */
	  for (j=0,cp=C,kp=K;j<m;j++) {
	    *cp=es+js+ns;
	    cp++;
	    *kp=es;
	    kp++;
	    for (k=j+1;k<m;k++) {
	      d=exp(s-*cp);
	      d=(d>eps) ? d : 0;
	      *cp=d;
	      cp++;
	      *kp=d;
	      kp++;
	    }
	  }
	} else {                      /* Do this if noise variances are not given */
	  for (j=0,cp=C,kp=K;j<m;j++) {
	    *cp=es+js+nv[j];
	    cp++;
	    *kp=es;
	    kp++;
	    for (k=j+1;k<m;k++) {
	      d=exp(s-*cp);
	      d=(d>eps) ? d : 0;
	      *cp=d;
	      cp++;
	      *kp=d;
	      kp++;
	    }
	  }
	}
	/* evaluate the expectation of new output, y=K'*(C*ty). */ 
	b=mxCalloc(m,sizeof(double));
	for (i=0;i<m;i++)
	  b[i]=ty[i];
	/* Evaluates C <- Chol(C) (lower triangular) */
	dpptrf(uplo, &m, C, &info);
	/* Evaluates b <- inv(C)*b  (solves C*x= b to b) */
	dpptrs(uplo, &m, &one, C, b, &m, &info);
	/* Evaluates y=K*b
	   NOTE: Here K is (exceptionally) symmentric mxm matrix */
 	dspmv(uplo, &m, &done, K, b, &one, &dzero, y, &one);
	mxFree(C);
	mxFree(K);
	mxFree(b);
      }
    }
  }
  
  return;
}
