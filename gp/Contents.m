% THE GP TOOLS (/in the GP/ folder):
% 
%  Gaussian process utilities:
%   GP_SET    Create and modify a Gaussian Process structure. 
%   GP_PAK    Combine GP parameters into one vector.
%   GP_UNPAK  Set GP parameters from vector to structure
%   GP_COV    Evaluate covariance matrix between two input vectors. 
%   GP_TRCOV  Evaluate training covariance matrix (gp_cov + noise covariance). 
%   GP_TRVAR  Evaluate training variance vector. 
%   GP_RND    Random draws from the postrior Gaussian process
%
%  Covariance functions:
%   GPCF_CAT           Create a categorigal covariance function
%   GPCF_CONSTANT      Create a constant covariance function 
%   GPCF_EXP           Create a squared exponential covariance function
%   GPCF_LINEAR        Create a linear covariance function
%   GPCF_MATERN32      Create a squared exponential covariance function
%   GPCF_MATERN52      Create a squared exponential covariance function
%   GPCF_NEURALNETWORK Create a squared exponential covariance function
%   GPCF_PERIODIC      Create a periodic covariance function
%   GPCF_PPCS0         Create a piece wise polynomial (q=0) covariance function 
%   GPCF_PPCS1         Create a piece wise polynomial (q=1) covariance function 
%   GPCF_PPCS2         Create a piece wise polynomial (q=2) covariance function 
%   GPCF_PPCS3         Create a piece wise polynomial (q=3) covariance function 
%   GPCF_PROD          Create a product form covariance function 
%   GPCF_RQ            Create an rational quadratic covariance function 
%   GPCF_SEXP          Create a squared exponential covariance function
%
%  Likelihood functions:
%   LIK_GAUSSIAN    Create a Gaussian likelihood structure
%   LIK_GAUSSIANSMT Create a Gaussian scale mixture approximating t
%   LIK_BINOMIAL    Create a binomial likelihood structure 
%   LIK_LOGIT       Create a Logit likelihood structure 
%   LIK_NEGBIN      Create a Negbin likelihood structure 
%   LIK_POISSON     Create a Poisson likelihood structure 
%   LIK_PROBIT      Create a Probit likelihood structure 
%   LIK_T           Create a Student-t likelihood structure 
%
% Inference utilities:
%   GP_E          Evaluate energy function (un-normalized negative marginal 
%                 log posterior) 
%   GP_G          Evaluate gradient of energy (GP_E) for Gaussian Process
%   GP_EG         Evaluate both GP_E and GP_G. Useful in optimisation.
%   GP_PRED       Make predictions with Gaussian process 
%   GPEP_E        Conduct Expectation propagation and return negative marginal 
%                 log posterior estimate
%   GPEP_G        Evaluate gradient of EP's negative marginal log posterior 
%                 estimate
%   EP_PRED       Predictions with Gaussian Process EP approximation
%   GPLA_E        Construct Laplace approximation and return negative marginal 
%                 log posterior estimate
%   GPLA_G        Evaluate gradient of Laplace approximation's marginal 
%                 log posterior estimate
%   LA_PRED       Predictions with Gaussian Process Laplace approximation
%   GP_MC         Markov chain sampling for Gaussian process models
%   MC_PRED       Predictions with Gaussian Process MCMC approximation.
%   MC_PREDS      Conditional predictions with Gaussian Process MCMC
%                 approximation.
%   GP_IA         Integration approximation with grid, Monte Carlo or
%                 CCD integration
%   IA_PRED       Prediction with Gaussian Process GP_IA solution.
%   LGCP          Log Gaussian Cox Process intensity estimate for 1D and 
%                 2D data
%
%  Model assesment and comparison:
%   GP_DIC        The DIC statistics and efective number of parameters in a GP model
%   GP_KFCV       K-fold cross validation for a GP model
%   GP_LOOE       Evaluate the leave-one-out predictive density in case of
%                 Gaussian observation model
%   GP_LOOG       Evaluate the gradient of the leave-one-out predictive 
%                 density (GP_LOOE) in case of Gaussian observation model 
%   GP_EPLOOE     Evaluate the leave-one-out predictive density in case of
%                 non-Gaussian observation model and EP
%   EP_LOOPRED    Leave-one-out-predictions with Gaussian Process EP approximation
%   GP_PEFF       The efective number of parameters in GP model with focus 
%                 on latent variables.
%
%  Metrics:
%   METRIC_EUCLIDEAN   An Euclidean distance for Gaussian process models.
%  
%  Misc:
%    LDLROWMODIFY  Function to modify the sparse cholesky factorization 
%                  L*D*L' = C, when a row and column k of C have changed 
%    LDLROWUPDATE  Multiple-rank update or downdate of a sparse LDL' factorization.
%    SPINV         Evaluate the sparsified inverse matrix
%    SCALED_HMC    A scaled hybric Monte Carlo samping for latent values
%    SCALED_MH     A scaled Metropolis Hastings samping for latent values
%    GP_INSTALL    Matlab function to compile all the c-files to mex in the 
%                  GPstuff/gp folder.
%
%  Demonstration programs:
%   DEMO_BINOMIAL1          Demonstration of Gaussian process model with binomial
%                           likelihood
%   DEMO_BINOMIAL_APC       Demonstration for modeling age-period-cohort data
%                           by a binomial model combined with GP prior.
%   DEMO_CLASSIFIC          Classification problem demonstration for 2 classes 
%   DEMO_LGCP               Demonstration for a log Gaussian Cox process
%                           with inference via EP or Laplace approximation
%   DEMO_MODELASSESMENT1    Demonstration for model assesment with DIC, number 
%                           of effective parameters and ten-fold cross validation
%   DEMO_MODELASSESMENT2    Demonstration for model assesment when the observation 
%                           model is non-Gaussian
%   DEMO_NEURALNETWORKCOV   Demonstration of Gaussian process with a neural
%                           network covariance function
%   DEMO_PERIODIC           Regression problem demonstration for periodic data
%   DEMO_REGRESSION1        Regression problem demonstration for 2-input 
%                           function with Gaussian process
%   DEMO_REGRESSION_PPCS    Regression problem demonstration for 2-input 
%                           function with Gaussian process using CS covariance
%   DEMO_REGRESSION_ADDITIVE1 Regression problem demonstration with additive model
%   DEMO_REGRESSION_ADDITIVE2 Regression demonstration with additive Gaussian
%                           process using linear, squared exponential and
%                           neural network covariance fucntions 
%   DEMO_REGRESSION_HIER    Hierarchical regression demonstration
%   DEMO_REGRESSION_ROBUST  A regression demo with Student-t distribution as a 
%                           residual model.
%   DEMO_REGRESSION_SPARSE1 Regression problem demonstration for 2-input 
%                           function with sparse Gaussian processes
%   DEMO_REGRESSION_SPARSE2 Regression demo comparing different sparse
%                           approximations
%   DEMO_SPATIAL1           Demonstration for a disease mapping problem
%                           with Gaussian process prior and Poisson likelihood
%   DEMO_SPATIAL2           Demonstration for a disease mapping problem with 
%                           Gaussian process prior and negative binomial 
%                           observation model
