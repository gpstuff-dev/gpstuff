/* GP2FWDS	Forward propagation through Gaussian Processes
 *
 *	Description
 *	Y = GP2FWDS(GP, TX, TY, X) takes a gp data structures GP together with a
 *	matrix X of input vectors, matrix TX of training inputs and vector TY of 
 *      training targets, and forward propagates the inputs through the GPs to  
 *      generate a vector Y of outputs. Each row of X corresponds to one input 
 *      vector and each row of Y corresponds to one output.
 *
 *      LATENT VALUES
 *      TY can be also matrix of latent values in which case it is of size MxNMC. 
 *      In this case TY is handled as all the other sampled parameters.
 *
 *      BUGS: - only exp2 covariance function is supported
 *            - only 1 output allowed
 *            - only mean values
 *
 *	See also
 *	GP2, GP2PAK, GP2UNPAK
 *
 * Last modified: 2003-10-24 15:45:16 EEST
 *
 */

/* 
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
  
  if (nrhs<3 || nrhs>4)
    mexErrMsgTxt( "Wrong number of input arguments." );
  
  if(!mxIsStruct(prhs[0]))
    mexErrMsgTxt("First input must be a structure.");
  
  {
    /* Initialize the parameters
     */
    const double *tx, *tx0, *ty, *x=NULL, *x0, *pr, *nv=NULL, *ps, *pjs, *pns=NULL;
    const double dzero=0.0, done=1.0;
    double s, es, js, ns, r, *K, *kp, *C, *cp, d, *b, *y, eps;
    const int *dims, one=1;
    int h, i, j, k, m, n, N, m2=0, nmc, info, lr, dims3[]={0,1,0};
    char *uplo="L", *trans="N";
    mxArray *field;

    dims = mxGetDimensions(prhs[1]);  /* Dimensions of training inputs matrix */
    tx = mxGetPr(prhs[1]);            /* Training inputs matrix */
    m = dims[0];                      /* The number of training input vectors */
    n = dims[1];                      /* The lenght of training input vector */
    tx0=tx;                           
    
    dims = mxGetDimensions(prhs[2]);
    /* Set N=1 if latent values are present */
    N = ((dims[1]) > (1)) ? (1) : (0);
    if (dims[0]!=m)
      mexErrMsgTxt( "TY must be M x 1 or MxN matrix." );
    ty = mxGetPr(prhs[2]);            /* Training outputs matrix */

    /* Check that the new input and training inputs have same dimensionality */
    if (nrhs>3) {
      dims = mxGetDimensions(prhs[3]);
      if (dims[0]==0 && dims[1]==0) {
	m2=m;
      }
      else if (dims[1]!=n)
	mexErrMsgTxt( "TX and X must have same number of columns." );
      else {
	x=mxGetPr(prhs[3]);           /* new inputs matrix */
	m2 = dims[0];                 /* number of new inputs */
      }
    } else {
      m2=m;
    }
    x0=x;

    /* Check that field expScale is in the gp structure and set it to es */
    if((field=mxGetField(*prhs, 0, "expScale"))==NULL)
      mexErrMsgTxt("Could not get gp.expScale");
    dims = mxGetDimensions(field);
    if (dims[1]!=1)
      mexErrMsgTxt( "S must be a nmc x 1." );
    nmc=dims[0];                      /* number of MCMC samples */
    ps=mxGetPr(field);                /* expScale */

    /* Check that field expSigmas is in the gp structure and set it to r */
    if((field=mxGetField(*prhs, 0, "expSigmas"))==NULL)
      mexErrMsgTxt("Could not get gp.expSigmas");
    dims = mxGetDimensions(field);
    if (dims[0]!=nmc || (dims[1]!=n && dims[1]!=1))
      mexErrMsgTxt( "R must be a nmc x n or nmc x 1." );
    lr=dims[1];                       /* if lr>1 ARD is used */
    pr=mxGetPr(field);                /* espSigmas */

    /* Check that field jitterSigmas is in the gp structure and set it to js */
    if((field=mxGetField(*prhs, 0, "jitterSigmas"))==NULL)
      mexErrMsgTxt("Could not get gp.jitterSigmas");
    dims = mxGetDimensions(field);
    if (dims[0]!=nmc || dims[1]!=1)
      mexErrMsgTxt( "JS must be a nmc x 1." );
    pjs=mxGetPr(field);               /* jitterSigmas */

    /* Check that field noiseVariances is in the gp structure and set it to nv */
    if((field=mxGetField(*prhs, 0, "noiseVariances"))==NULL)
      mexErrMsgTxt("Could not get gp.noiseVariances");
    dims = mxGetDimensions(field);
    if (dims[0]==0 && dims[1]==0) {
      nv=NULL;
      if((field=mxGetField(*prhs, 0, "noiseSigmas"))==NULL)
	mexErrMsgTxt("Could not get gp.noiseSigmas");
      dims = mxGetDimensions(field);
      if (!(dims[0]==0 || dims[1]==0)) {
	if (dims[0]!=nmc || dims[1]!=1)
	  mexErrMsgTxt( "NS must be a nmc x 1." );
	pns = mxGetPr(field);         /* noiseSigmas */
      }
    } else {
      if (dims[0]!=nmc || dims[1]!=m)
	mexErrMsgTxt( "NV must be a nmc x m." );
      nv = mxGetPr(field);            /* noiseVariances */
    }

    /* Create an output matrix for GP */
    dims3[0]=m2;dims3[2]=nmc;
    plhs[0]=mxCreateNumericArray(3,dims3,mxDOUBLE_CLASS,mxREAL);
    y = mxGetPr(plhs[0]);             /* Output matrix */

    /* The value of Matlab eps variable */
    eps=mxGetEps();

    if (x!=NULL) {
      /* Case function is called as:
	 g=gp2fwd(gp, p, t, pp)
      
      /* Allocate memory for covariance matrices and output vector */
      K=mxCalloc(m*m2,sizeof(double));
      C=mxCalloc((m*(m+1))/2,sizeof(double));
      b=mxCalloc(m,sizeof(double));
      /* Loop over all the MCMC samples */ 
      for (h=0;h<nmc;h++,y=y+m2) {
	s=ps[h];
	s*=s;
	es=s;                    /* expScale^2 */
	s=log(s);                /* log of expScale^2 */ 
	for (i=0,d=m*m2;i<d;i++)
	  K[i]=0;
	for (i=0,d=(m*(m+1))/2;i<d;i++)
	  C[i]=0;
	for (i=0;i<m;i++)
	  b[i]=ty[i+N*h*m];     /* Here N==1 if latent values are present, otherways N==0 */
	tx=tx0;
	x=x0;
	/* Evaluate the covariance between new input and training
	   inputs K and covariance between training inputs C. */
	   
	/* Evaluate the exponential part of K */
	for (i=0;i<n;i++,tx+=m,x+=m2) {
 	  r=(lr>1) ? (pr[h+i*nmc]) : (pr[0]);       /* Check if ARD used */
 	  r*=r; 
 	  for (j=0;j<m;j++) { 
 	    for (k=0;k<m2;k++) { 
	      d=tx[j]-x[k];
	      K[j*m2+k]+=d*d*r;
	    }
	  }
	}
	/* Evaluate the rest of the K */
	for (j=0;j<m;j++) {
	  for (k=0;k<m2;k++) {
	    d=exp(s-K[j*m2+k]);
	    K[j*m2+k]=(d>eps) ? d : 0;
	  }
	}
	/* Evaluate the exponential part of C */
	tx=tx0;
	for (i=0;i<n;i++,tx+=m) {
 	  r=(lr>1) ? (pr[h+i*nmc]) : (pr[0]); 
	  r*=r;
	  for (j=0,cp=C;j<m;j++) {    /* cp = C */ 
	    cp++;
	    for (k=j+1;k<m;k++) {
	      d=tx[j]-tx[k];
	      *cp+=d*d*r;
	      cp++;
	    }
	  }
	}
	/* Evaluate rest of C */
	js=pjs[h];
	js*=js;
	if (nv==NULL) {
	  if (pns==NULL) {    /* If there is not noiseVariances or noiseSigmas  */
	    for (j=0,cp=C;j<m;j++) {
	      *cp=es+js;
	      cp++;
	      for (k=j+1;k<m;k++) {
		d=exp(s-*cp);
		*cp=(d>eps) ? d : 0;
		cp++;
	      }
	    }
	  } else {            /* If there is not noiseVariances but there is noiseSigmas */
	    ns=pns[h];
	    ns*=ns;
	    for (j=0,cp=C;j<m;j++) {
	      *cp=es+js+ns;
	      cp++;
	      for (k=j+1;k<m;k++) {
		d=exp(s-*cp);
		*cp=(d>eps) ? d : 0;
		cp++;
	      }
	    }
	  }
	} else {               /* If there is noiseVariances */
	  for (j=0,cp=C;j<m;j++) {
	    *cp=es+js+nv[h+j*nmc];
	    cp++;
	    for (k=j+1;k<m;k++) {
	      d=exp(s-*cp);
	      *cp=(d>eps) ? d : 0;
	      cp++;
	    }
	  }
 	}
	/* Evaluates C <- Chol(C) (lower triangular) */
 	dpptrf(uplo, &m, C, &info);
	/* Evaluates b <- inv(C)*b  (solves C*x= b to b) */
 	dpptrs(uplo, &m, &one, C, b, &m, &info);
	/* Evaluates y=K*b
	   NOTE: Above in evaluation of K the we get K' */
  	dgemv(trans, &m2, &m, &done, K, &m2, b, &one, &dzero, y, &one); 
      }
      mxFree(C);
      mxFree(K);
      mxFree(b);
    }
    /* Case function is called as:
       g=gp2fwd(gp, p, t)*/
    
    else {
      C=mxCalloc((m*(m+1))/2,sizeof(double));
      K=mxCalloc((m*(m+1))/2,sizeof(double));
      b=mxCalloc(m,sizeof(double));

      /* Loop over MCMC samples */
      for (h=0;h<nmc;h++,y=y+m2) {
	s=ps[h];
	s*=s;
	es=s;
	s=log(s);
	for (i=(m*(m+1))/2-1;i>=0;i--) {
	  C[i]=0;
	  K[i]=0;
	}
	for (i=0;i<m;i++)
	  b[i]=ty[i+N*h*m];           /* Here N==1 if latent values are present, otherways N==0 */
	for (i=0,x=tx0;i<n;i++,x+=m) { 	 

	  for (j=0,cp=C,kp=K,r=(lr>1)?(pr[h+i*nmc]*pr[h+i*nmc]):(pr[0]*pr[0]);j<m;j++) {
	    cp++;
	    kp++;
	    for (k=j+1;k<m;k++) {
	      d=x[j]-x[k];
	      d*=d*r;
	      *cp+=d;
	      cp++;
	      *kp+=d;
	      kp++;
	    }
	  }
	}
	js=pjs[h];
	js*=js;
	if (nv==NULL) {
	  if (pns==NULL) {
	    for (j=0,cp=C,kp=K;j<m;j++) {
	      *cp=es+js;
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
	  } else {
	    ns=pns[h];
	    ns*=ns;
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
	  }
	} else {
	  for (j=0,cp=C,kp=K;j<m;j++) {
	    *cp=es+js+nv[h+j*nmc];
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
	dpptrf(uplo, &m, C, &info);
	dpptrs(uplo, &m, &one, C, b, &m, &info);
	dspmv(uplo, &m, &done, K, b, &one, &dzero, y, &one);
      }
      mxFree(C);
      mxFree(K);
      mxFree(b);
    }

  }
  return;
}     
