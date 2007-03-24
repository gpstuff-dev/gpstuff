/* Stratified Resampling No-Sort
 *
 * (C) 2003 Aki Vehtari
 *
 * Last modified: 2004-10-12 15:23:58 EEST
 *
 */

#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[],
		 int nrhs, const mxArray *prhs[])
{

  if (nlhs > 1 ) 
    mexErrMsgTxt( "Too many output arguments." );
  
  if (nrhs != 1)
    mexErrMsgTxt( "Wrong number of input arguments." );

  {
    mxArray *MN[2], *R;
    double *p, *q, *r, *s, sum=0, b, c=0; 
    const int *dims;
    int i, n, a, k=0;

    dims = mxGetDimensions(prhs[0]);
    n = dims[0]*dims[1];            /* number of weights */
    p = mxGetPr(prhs[0]);           /* pointer to unormalized weights */
    q = mxMalloc(n*sizeof(double)); /* pointer to normalized and scaled weights */

    /* 'n' uniform random numbers, 'MN' is needed only temporarily */
    MN[0]=mxCreateDoubleScalar(dims[0]);
    MN[1]=mxCreateDoubleScalar(dims[1]);
    mexCallMATLAB(1,&R,2,MN,"rand");
    mxDestroyArray(MN[0]);
    mxDestroyArray(MN[1]);
    r = mxGetPr(R);                 /* pointer to uniform random numbers */

    /* allocate matrix for return value */
    plhs[0]=mxCreateDoubleMatrix(dims[0],dims[1],mxREAL);
    s = mxGetPr(plhs[0]);     /* pointer to samples */

    for (i = 0; i < n; i++)
      sum+=p[i];              /* compute sum for normalization */
    for (i = 0; i < n; i++) {
      q[i]=p[i]/sum*n;        /* normalize and scale weight */
      c+=q[i];                /* cumulate weights */
      if (c>=1.0) {           /* if cumulative weight over 1 */
	a=(int)(c);           /* integer part of the cumulative weight */
	c-=a;                 /* subract integer part from the cum weight */
	k+=a;                 
	for (b=i+1.0;a>0;a--)	  
	  *s++=b;             /* fill vector 'a' times with sample 'b' */
      }
      if (k<n && c>=r[k]) {   /* if cumulative larger than a random number...*/
	*s++=i+1.0;           /* .. add sample... */
	c-=1.0;               /* ...and subract one from the cumulative */
	k+=1;                 
      }
    }

    mxDestroyArray(R);        /* no need for this anymore */

  }

  return;
}
