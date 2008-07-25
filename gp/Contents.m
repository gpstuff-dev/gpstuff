% THE GP TOOLS (/in the GP/ folder):
% 
%  Gaussian process utilities:
%  GP_INIT            - Create a Gaussian Process
%  GP_COV             - Evaluate covariance matrix between two input vectors. 
%  GP_COVVEC          - Evaluate covariance vector between two input vectors. 
%  GP_TRCOV           - Evaluate training covariance matrix
%  GP_TRVAR           - Evaluate training variance vector
%
%
%  Covariance functions:
%  GPCF_EXP           - Create an exponential covariance function for Gaussian Process
%  GPCF_MATERN32      - Create a Matern nu=3/2 covariance function for Gaussian Process
%  GPCF_MATERN52      - Create a Matern nu=5/2 covariance function for Gaussian Process
%  GPCF_NOISE         - Create a noise covariance function for Gaussian Process
%  GPCF_PPCS2         - Create a piece wise polynomial covariance function for Gaussian Process
%                       (this is compactly supported covariance function)
%  GPCF_SEXP          - Create a squared exponential covariance function for Gaussian Process
%
%
%  Likelihood functions:
%  LIKELIH_LOGIT      - Create a Logit likelihood structure for Gaussian Proces
%  LIKELIHOOD_PROBIT  - Create a Probit likelihood structure for Gaussian Process
%  LIKELIH_POISSON    - Create a Poisson likelihood structure for Gaussian Process
%
%
% Inference utilities:
%  EP_PRED            - Predictions with Gaussian Process EP
%  GP_E               - Evaluate energy function for Gaussian Process 
%  GP_G               - Evaluate gradient of energy for Gaussian Process
%  GPEP_E             - Conduct Expectation propagation and return marginal 
%                       log posterior estimate
%  GPEP_G             - Evaluate gradient of EP's marginal log posterior estimate 
%  GPLA_E             - Conduct LAplace approximation and return marginal log 
%                       posterior estimate
%  GPLA_G             - Evaluate gradient of Laplace approximation's marginal 
%                       log posterior estimate 
%  GP_MC              - Monte Carlo sampling for Gaussian process models
%  GP_MCOPT           - Default options for GP_MC and GP_MC
%  GP_PAK             - Combine GP hyper-parameters into one vector
%  GP_PRED            - Make predictions for Gaussian process
%  GP_PREDS           - (Multible) Predictions of Gaussian Processes
%  GP_UNPAK           - Set GP hyper-parameters from vector to structure
%  LA_PRED            - Predictions with Gaussian Process Laplace approximation
%  SINV               - Evaluate the sparse inverse matrix
%
%
%  Demonstration programs:
%  DEMO_CLAASIFIC1    - Classification problem demonstration for 2 classes via MCMC
%  DEMO_CLAASIFIC1    - Classification problem demonstration for 2 classes via Laplace 
%                       approximation and EP
%  DEMO_REGRESSION1   - Regression problem demonstration for 2-input 
%                       function with Gaussian process
%  DEMO_REGRESSION2   - Regression problem demonstration for modeling 
%                       multible phenomenon
%  DEMO_SPATIAL1      - Demonstration for a disease mapping problem
%                       with Gaussian process prior