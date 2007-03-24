% PROBABILITY DISTRIBUTION FUNCTIONS (contents of the dist-folder):
%
% probability density functions
%    BETA_LPDF     - Beta log-probability density function (lpdf).
%    BETA_PDF      - Beta probability density function (pdf).
%    DIR_LPDF      - Log probability density function of uniform Dirichlet
%                    distribution
%    DIR_PDF       - Probability density function of uniform Dirichlet
%                    distribution
%    GAM_CDF       - Cumulative of Gamma probability density function (cdf).
%    GAM_LPDF      - Log of Gamma probability density function (lpdf).
%    GAM_PDF       - Gamma probability density function (pdf).
%    INVGAM_LPDF   - Inverse-Gamma log probability density function.
%    INVGAM_PDF    - Inverse-Gamma probability density function.
%    LAPLACE_LPDF  - Laplace log-probability density function (lpdf).
%    LAPLACE_PDF   - Laplace probability density function (pdf).
%    LOGN_LPDF     - Log normal log-probability density function (lpdf)
%    LOGT_LPDF     - Log probability density function (lpdf) for log Student's T
%    MNORM_LPDF    - Multivariate-Normal log-probability density function (lpdf).
%    MNORM_PDF     - Multivariate-Normal log-probability density function (lpdf).
%    KERNELP       - Kernel density estimator for one dimensional distribution.
%    NORM_LPDF     - Normal log-probability density function (lpdf).
%    NORM_PDF      - Normal probability density function (pdf).
%    POISS_LPDF    - Poisson log-probability density function.
%    POISS_PDF     - Poisson probability density function.
%    SINVCHI2_LPDF - Scaled inverse-chi log-probability density function.
%    SINVCHI2_PDF  - Scaled inverse-chi probability density function.
%    T_LPDF        - Student's T log-probability density function (lpdf)
%    T_PDF         - Student's T probability density function (pdf)
%    NORM_P      - create Gaussian (multivariate) (hierarchical) prior
%    INVGAM_P            - Create inverse-Gamma prior
%
%  Error and gradient functions:
%    INVGAM_E    - compute an error term for a parameter with inverse
%                  gamma distribution (single parameter).
%    INVGAM_G    - compute a gradient term for a parameter with inverse
%                  gamma distribution (single parameter).
%    LAPLACE_E   - compute an error term for a parameter with Laplace
%                  distribution (single parameter). 
%    LAPLACE_G   - compute a gradient for a parameter with Laplace 
%                  distribution (single parameter).
%    MNORM_E     - compute an error term for parameters with normal
%                  distribution (multiple parameters).
%    MNORM_G     - compute a gradient for parameters with normal 
%                  distribution (multible parameters)
%    MNORM_S     - Maximum log likelihood second derivatives
%    NORM_E      - compute an error term for a parameter with normal
%                  distribution (single parameter). 
%    NORM_G      - compute a gradient for a parameter with normal 
%                  distribution (single parameter).
%    NORM_S      - Maximum log likelihood second derivatives (single variable)
%    T_E         - compute an error term for a parameter with Student's 
%                  t-distribution (single parameter). 
%    T_G         - compute a gradient for a parameter with Student's 
%                  t-distribution (single parameter).
%    DIR_E       - compute an error term for a parameter with Dirichlet
%                  distribution (single parameter). 
%    GINVGAM_E   - Compute an error term for a parameter with inverse
%                  gamma distribution (single parameter).
%    GINVGAM_G   - Compute a gradient term for a parameter with inverse
%                  gamma distribution (single parameter).
%    GP2R_E	 - Evaluate error function for Gaussian Process.
%    GP2R_G      - Evaluate gradient of error for Gaussian Process.
%    GNORM_E     - Compute an error term for a parameter with normal
%                  distribution (single parameter). 
%    GNORM_G     - Compute a gradient for a parameter with normal 
%                  distribution (single parameter).
%    GNORM_S     - Maximum log likelihood second derivatives.
%    GT_E        - Compute an error term for a parameter with Student's
%                  t-distribution (single parameter). 
%    GT_G        - Compute a gradient for a parameter with Student's 
%                  t-distribution (single parameter).
%    GT_S        - Maximum log likelihood second derivatives for 
%                  t-distribution.
%    T_S         - Maximum log likelihood second derivatives for t-distribution
%    T_P         - Create student t prior
%
%  Functions to sample from full conditional distribution
%    COND_GINVGAM_CAT    - Sample conditional distribution from 
%                          inverse gamma likelihood for a group and
%                          categorical prior. 
%    COND_GNORM_INVGAM   - Sample conditional distribution from
%                          normal likelihood for group and
%                          inverse gamma prior.
%    COND_GNORM_NORM     - Sample conditional distribution from normal
%                          likelihood for a group and normal prior.
%    COND_GT_CAT         - Sample conditional distribution from t 
%                          likelihood for a group and categorical prior.
%    COND_GT_INVGAM      - Sample conditional distribution from t 
%                          likelihood for a group and inverse gamma prior.
%    COND_INVGAM_CAT     - Sample conditional distribution from
%                          inverse gamma likelihood and categorical prior. 
%    COND_INVGAM_INVGAM  - Sample conditional distribution from
%                          inverse gamma likelihood and prior
%    COND_LAPLACE_INVGAM - Sample conditional distribution from Laplace
%                          likelihood and inverse gamma prior.
%    COND_MNORM_INVWISH  - Sample conditional distribution from normal
%                          likelihood for multiparameter group and
%                          inverse wishard prior.
%    COND_NORM_GINVGAM   - Sample conditional distribution from
%                          normal likelihood and inverse gamma prior
%                          for a group
%    COND_NORM_INVGAM    - Sample conditional distribution from
%                          normal likelihood and inverse gamma prior
%    COND_T_CAT          - Sample conditional distribution from t
%                          likelihood and categorical prior.
%    COND_T_INVGAM       - Sample conditional distribution from t
%                          likelihood and inverse gamma prior.
%
% Random number generators
%    CATRAND       - Random matrices from categorical distribution.
%    DIRRAND       - Uniform dirichlet random vectors
%    EXPRAND       - Random matrices from exponential distribution.
%    GAMRAND       - Random matrices from gamma distribution.
%    INTRAND       - Random matrices from uniform integer distribution.
%    INVGAMRAND    - Random matrices from inverse gamma distribution
%    INVGAMRAND1   - Random matrices from inverse gamma distribution
%    INVWISHRND    - Random matrices from inverse Wishart distribution.
%    NORMLTRAND    - Random draws from a left-truncated normal
%                    distribution, with mean = mu, variance = sigma2
%    NORMRTRAND    - Random draws from a right-truncated normal
%                    distribution, with mean = mu, variance = sigma2
%    NORMTRAND     - Random draws from a normal truncated to interval
%    NORMTZRAND    - Random draws from a normal distribution truncated by zero
%    WISHRND       - Random matrices from Wishart distribution.
%    SINVCHI2RAND  - Random matrices from scaled inverse-chi distribution
%    TRAND         - Random numbers from Student's t-distribution
%    UNIFRAND      - Generate unifrom random numberm from interval [A,B]
