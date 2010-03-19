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



% To be included into the package

demo_classific1.m
demo_regression1.m
demo_regression2.m
demo_sparseRegression.m


ep_pred.m

gpcf_constant.m
gpcf_dotproduct.m
gpcf_exp.m
gpcf_linear.m
gpcf_matern32.m
gpcf_matern52.m
gpcf_neuralnetwork.m
gpcf_noise.m
gpcf_noiset.m
gpcf_ppcs0.m
gpcf_ppcs1.m
gpcf_ppcs2.m
gpcf_ppcs3.m
gpcf_prod.m
gpcf_rq.m
gpcf_sexp.m

gp_cov.m
gp_e.m
gpep_e.m
gpep_g.m
gp_g.m
gp_ia.m
gp_iaopt.m
gp_init.m
gp_install.m

gpla_e.m
gpla_g.m
gp_mc.m
gp_mcopt.m
gp_pak.m
gp_pred.m
gp_rnd.m
gp_trcov.m
gp_trvar.m
gp_unpak.m
ia_pred.m

la_pred_alk.m
la_pred.m
ldlrowmodify.m

likelih_binomial.m
likelih_logit.m
likelih_negbin.m
likelih_poisson.m
likelih_probit.m
likelih_t.m
matlab_install.m
mc_pred.m
metric_euclidean.m

scaled_hmc.m
scaled_mh.m

spinv.m
test_package.m

trcov.m





% To be decided if is included in the package
demo_binomial.m
demo_censored_t.m
demo_classific2.m
demo_classific3.m
demo_ep_clFull_mcmc.m
demo_infneuralnetwork.m
demo_lgcpdens.m
demo_modelassesment1.m
demo_modelassesment2.m
demo_nb_FIC.m
demo_nb_Full.m
demo_regression3.m
demo_regression4.m
demo_regression_metric.m
demo_robustRegression.m
demo_spatial1.m
demo_spatial2.m
demo_spatial3.m
demo_st_cancer_FIC.m
demo_st_FIC.m
demo_st_Full.m
demo_st_PIC.m

ep_post.m

gp_ais2.m
gp_ais3.m
gp_ais.m
gpcf_noise_cent.m
gpcf_Ssexp.m
gpcf_ssgp.m
gpcf_SSsexp.m

gp_cve.m
gp_cvg.m
gp_dic.m
gp_kfcv.m
gp_peff.m

la_post.m
lgcpdens.m

likelih_cen_t.m

pred_e.m
pred_g.m

quadgk2.m
quadgk.m

temp.m
test_nested.m
test_rowmod.m