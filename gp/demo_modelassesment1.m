%DEMO_MODELASSESMENT1   Demonstration for model assesment with DIC, number 
%                       of effective parameters and ten-fold cross validation
%                       
%
%    Description
%    We will consider the regression problem in demo_regression1. The
%    analysis is conducted with full Gaussian process, and FIC and PIC
%    sparse approximations. The performance of these models are
%    compared by evaluating the DIC statistics, number of efficient
%    parameters and ten-fold cross validation. The inference will be
%    conducted using maximum a postrior (MAP) estimate for the
%    hyperparameters, via full Markov chain Monte Carlo (MCMC) and
%    with an integration approximation (IA) for the hyperparameters.
%
%    This demo is organised in three parts:
%     1) data analysis with full GP model
%     2) data analysis with FIC approximation
%     3) data analysis with PIC approximation
%
%   See also  DEMO_REGRESSION1, DEMO_SPARSEREGRESSION

% Copyright (c) 2009-2010 Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.


%========================================================
% PART 1 data analysis with full GP model
%========================================================

% Load the data
S = which('demo_regression1');
L = strrep(S,'demo_regression1.m','demos/dat.1');
data=load(L);
x = [data(:,1) data(:,2)];
y = data(:,3);
[n, nin] = size(x);

% ---------------------------
% --- Construct the model ---
gpcf1 = gpcf_sexp('init', 'lengthScale', [1 1], 'magnSigma2', 0.2^2);
gpcf2 = gpcf_noise('init', 'noiseSigma2', 0.2^2);
gp = gp_init('init', 'FULL', 'gaussian', {gpcf1}, {gpcf2}, 'jitterSigma2', 0.0001)

% -----------------------------
% --- Conduct the inference ---
%
% We will make the inference first by finding a maximum a posterior estimate 
% for the hyperparameters via gradient based optimization. After this we will
% perform an extensive Markov chain Monte Carlo sampling for the hyperparameters.
% 

% --- MAP estimate using scaled conjugate gradient algorithm ---
%     (see scg for more details)

w=gp_pak(gp);  % pack the hyperparameters into one vector
fe=str2fun('gp_e');     % create a function handle to negative log posterior
fg=str2fun('gp_g');     % create a function handle to gradient of negative log posterior

% set the options for scg2
opt = scg2_opt;
opt.tolfun = 1e-3;
opt.tolx = 1e-3;
opt.display = 1;

% do the optimization
w=scg2(fe, w, opt, fg, gp, x, y);

% Set the optimized hyperparameter values back to the gp structure
gp=gp_unpak(gp,w);

% Evaluate the effective number of parameters and DIC with focus on
% latent variables.
models{1} = 'full_MAP';
p_eff_latent = gp_peff(gp, x, y);
[DIC_latent, p_eff_latent2] = gp_dic(gp, x, y, 'focus', 'latent');

% Evaluate the 10-fold cross-validation results.
cvres =  gp_kfcv(gp, x, y);
mlpd_cv(1) = cvres.mlpd_cv;
mrmse_cv(1) = cvres.mrmse_cv;

% --- MCMC approach ---

% The sampling options are set to 'opt' structure, which is given to
% 'gp_mc' sampler
opt = [];
opt.nsamples= 100;
opt.repeat=4;
opt.hmc_opt = hmc2_opt;
opt.hmc_opt.steps=4;
opt.hmc_opt.stepadj=0.05;
opt.hmc_opt.persistence=0;
opt.hmc_opt.decay=0.6;
opt.hmc_opt.nsamples=1;
hmc2('state', sum(100*clock));

% Do the sampling (this takes few minutes)
rfull = gp_mc(gp, x, y, opt);

% After sampling we delete the burn-in and thin the sample chain
rfull = thin(rfull, 10, 2);

% Evaluate the effective number of parameters and DIC. 
models{2} = 'full_MCMC';
[DIC(2), p_eff(2)] =  gp_dic(rfull, x, y, 'focus', 'hyper');
[DIC2(2), p_eff2(2)] =  gp_dic(rfull, x, y, 'focus', 'all');

% Evaluate the 10-fold cross validation results. 
%
% We reduce the number of samples so that the sampling takes less time. 
% 50 is too small sample size, though, and for reliable results the 10-CV 
% should be run with larger sample size. We also set the save option to 0.
opt.nsamples= 50; 
cvres =  gp_kfcv(gp, x, y, 'inf_method', 'MCMC', 'opt', opt);
mlpd_cv(2) = cvres.mlpd_cv;
mrmse_cv(2) = cvres.mrmse_cv;


% --- Integration approximation approach ---
clear('opt')
opt.opt_scg = scg2_opt;
opt.int_method = 'grid';
opt.step_size = 2;

gp_array = gp_ia(gp, x, y, opt);

models{3} = 'full_IA'; 
[DIC(3), p_eff(3)] =  gp_dic(gp_array, x, y, 'focus', 'hyper');
[DIC2(3), p_eff2(3)] =  gp_dic(gp_array, x, y, 'focus', 'all');

% Then the 10 fold cross-validation.
cvres = gp_kfcv(gp, x, y, 'inf_method', 'IA', 'opt', opt);
mlpd_cv(3) = cvres.mlpd_cv;
mrmse_cv(3) = cvres.mrmse_cv;


%========================================================
% PART 2 data analysis with FIC GP
%========================================================

% ---------------------------
% --- Construct the model ---

% Here we conduct the same analysis as in part 1, but this time we 
% use FIC approximation

% Initialize the inducing inputs in a regular grid over the input space
[u1,u2]=meshgrid(linspace(-1.8,1.8,6),linspace(-1.8,1.8,6));
X_u = [u1(:) u2(:)];

% Create the FIC GP data structure
gp_fic = gp_init('init', 'FIC', 'gaussian', {gpcf1}, {gpcf2}, 'jitterSigma2', 0.0001, 'X_u', X_u)

% -----------------------------
% --- Conduct the inference ---

% --- MAP estimate using scaled conjugate gradient algorithm ---

gp_fic = gp_init('set', gp_fic, 'infer_params', 'covariance');           % optimize only hyperparameters

% set the options
fe=str2fun('gp_e');     % create a function handle to negative log posterior
fg=str2fun('gp_g');     % create a function handle to gradient of negative log posterior
opt = scg2_opt;
opt.tolfun = 1e-3;
opt.tolx = 1e-3;
opt.display = 1;
opt.maxiter = 20;

w = gp_pak(gp_fic);          % pack the hyperparameters into one vector
w=scg2(fe, w, opt, fg, gp_fic, x, y);       % do the optimization
gp_fic = gp_unpak(gp_fic,w);     % Set the optimized hyperparameter values back to the gp structure

% Evaluate the effective number of parameters and DIC with focus on latent variables. 
models{4} = 'FIC_MAP';
p_eff_latent(4) = gp_peff(gp_fic, x, y);
[DIC_latent(4), p_eff_latent2(4)] = gp_dic(gp_fic, x, y, 'focus', 'latent');

% Evaluate the 10-fold cross validation results. 
cvres = gp_kfcv(gp_fic, x, y);
mlpd_cv(4) = cvres.mlpd_cv;
mrmse_cv(4) = cvres.mrmse_cv;

% --- MCMC approach ---
% (the inducing inputs are fixed)
clear('opt')
opt.nsamples= 100;
opt.repeat=5;
opt.hmc_opt.steps=3;
opt.hmc_opt.stepadj=0.02;
opt.hmc_opt.persistence=0;
opt.hmc_opt.decay=0.6;
opt.hmc_opt.nsamples=1;
hmc2('state', sum(100*clock));

% Do the sampling (this takes approximately 3-5 minutes)
rfic = gp_mc(gp_fic, x, y, opt);

% After sampling we delete the burn-in and thin the sample chain
rfic = thin(rfic, 10, 2);

% Evaluate the effective number of parameters and DIC. Note that 
% the efective number of parameters as a second output, but here 
% we use explicitly the gp_peff function
models{5} = 'FIC_MCMC'; 
[DIC(5), p_eff(5)] =  gp_dic(rfic, x, y, 'focus', 'hyper');
[DIC2(5), p_eff2(5)] =  gp_dic(rfic, x, y, 'focus', 'all');

% We reduce the number of samples so that the sampling takes less time. 
% 50 is too small sample size, though, and for reliable results the 10-CV 
% should be run with larger sample size. We also set the save option to 0.
opt.nsamples= 50; 

cvres = gp_kfcv(gp_fic, x, y, 'inf_method', 'MCMC', 'opt', opt);
mlpd_cv(5) = cvres.mlpd_cv;
mrmse_cv(5) = cvres.mrmse_cv;


% --- Integration approximation approach ---
clear('opt')
opt.opt_scg = scg2_opt;
opt.int_method = 'grid';
opt.step_size = 2;

gpfic_array = gp_ia(gp_fic, x, y, opt);

models{6} = 'FIC_IA'; 
[DIC(6), p_eff(6)] =  gp_dic(gpfic_array, x, y, 'hyper');
[DIC2(6), p_eff2(6)] =  gp_dic(gpfic_array, x, y, 'all');

% Then the 10 fold cross-validation.
cvres = gp_kfcv(gp_fic, x, y, 'inf_method', 'IA', 'opt', opt);
mlpd_cv(6) = cvres.mlpd_cv;
mrmse_cv(6) = cvres.mrmse_cv;

%========================================================
% PART 3 data analysis with PIC approximation
%========================================================

[u1,u2]=meshgrid(linspace(-1.8,1.8,6),linspace(-1.8,1.8,6));
X_u = [u1(:) u2(:)];

% Initialize test points
[p1,p2]=meshgrid(-1.8:0.1:1.8,-1.8:0.1:1.8);
p=[p1(:) p2(:)];

% set the data points into clusters. Here we construct two cell arrays. 
%  trindex  contains the block index vectors for training data. That is 
%           x(trindex{i},:) and y(trindex{i},:) belong to the i'th block.
b1 = [-1.7 -0.8 0.1 1 1.9];
mask = zeros(size(x,1),size(x,1));
trindex={}; 
for i1=1:4
    for i2=1:4
        ind = 1:size(x,1);
        ind = ind(: , b1(i1)<=x(ind',1) & x(ind',1) < b1(i1+1));
        ind = ind(: , b1(i2)<=x(ind',2) & x(ind',2) < b1(i2+1));
        trindex{4*(i1-1)+i2} = ind';
    end
end

% Create the PIC GP data structure and set the inducing inputs and block indeces
gpcf1 = gpcf_sexp('init', 'lengthScale', [1 1], 'magnSigma2', 0.2^2);
gpcf2 = gpcf_noise('init', 'noiseSigma2', 0.2^2);

gp_pic = gp_init('init', 'PIC', 'gaussian', {gpcf1}, {gpcf2}, 'jitterSigma2', 0.001, 'X_u', X_u);
gp_pic = gp_init('set', gp_pic, 'tr_index', trindex)

% -----------------------------
% --- Conduct the inference ---

% --- MAP estimate using scaled conjugate gradient algorithm ---

gp_pic = gp_init('set', gp_pic, 'infer_params', 'covariance');           % optimize only hyperparameters

% set the options
opt = scg2_opt;
opt.tolfun = 1e-3;
opt.tolx = 1e-3;
opt.display = 1;
opt.maxiter = 20;

w = gp_pak(gp_pic);          % pack the hyperparameters into one vector
w=scg2(fe, w, opt, fg, gp_pic, x, y);       % do the optimization
gp_pic = gp_unpak(gp_pic,w);     % Set the optimized hyperparameter values back to the gp structure

models{7} = 'PIC_MAP';
p_eff_latent(7) = gp_peff(gp_pic, x, y);
[DIC_latent(7), p_eff_latent2(7)] = gp_dic(gp_pic, x, y, 'latent');

% Evaluate the 10-fold cross validation results. 
cvres = gp_kfcv(gp_pic, x, y);
mlpd_cv(7) = cvres.mlpd_cv;
mrmse_cv(7) = cvres.mrmse_cv;

% --- MCMC approach ---

clear('opt')
opt.nsamples= 100;
opt.repeat=5;
opt.hmc_opt.steps=3;
opt.hmc_opt.stepadj=0.02;
opt.hmc_opt.persistence=0;
opt.hmc_opt.decay=0.6;
opt.hmc_opt.nsamples=1;
hmc2('state', sum(100*clock));

% Do the sampling (this takes approximately 3-5 minutes)
rpic = gp_mc(gp_pic, x, y, opt);

% After sampling we delete the burn-in and thin the sample chain
rpic = rmfield(rpic, 'tr_index');
rpic = thin(rpic, 10, 2);
rpic.tr_index = trindex;

% Evaluate the effective number of parameters and DIC. Note that 
% the efective number of parameters as a second output, but here 
% we use explicitly the gp_peff function
models{8} = 'PIC_MCMC'; 
[DIC(8), p_eff(8)] =  gp_dic(rpic, x, y, 'hyper');
[DIC2(8), p_eff2(8)] =  gp_dic(rpic, x, y, 'all');

% We reduce the number of samples so that the sampling takes less time. 
% 50 is too small sample size, though, and for reliable results the 10-CV 
% should be run with larger sample size. We also set the save option to 0.
opt.nsamples= 50; 
cvres = gp_kfcv(gp_pic, x, y, 'inf_method', 'MCMC', 'opt', opt);
mlpd_cv(8) = cvres.mlpd_cv;
mrmse_cv(8) = cvres.mrmse_cv;

% --- Integration approximation approach ---
clear('opt')
opt.opt_scg = scg2_opt;
opt.int_method = 'grid';
opt.step_size = 2;

gppic_array = gp_ia(gp_pic, x, y, [], opt);

models{9} = 'PIC_IA'; 
[DIC(9), p_eff(9)] =  gp_dic(gppic_array, x, y, 'hyper');
[DIC2(9), p_eff2(9)] =  gp_dic(gppic_array, x, y, 'all');

% Then the 10 fold cross-validation.
cvres = gp_kfcv(gp_pic, x, y, 'inf_method', 'IA', 'opt', opt);
mlpd_cv(9) = cvres.mlpd_cv;
mrmse_cv(9) = cvres.mrmse_cv;



%========================================================
% PART 4 Print the results
%========================================================

S = '       ';
for i = 1:length(models)
    S = [S '  ' models{i}];
end

S = sprintf([S '\n DIC_h   %.2f      %.2f      %.2f    %.2f     %.2f     %.2f   %.2f     %.2f     %.2f'], DIC);
S = sprintf([S '\n DIC_a   %.2f     %.2f     %.2f    %.2f     %.2f   %.2f   %.2f    %.2f     %.2f'], DIC2);
S = sprintf([S '\n DIC_l  %.2f      %.2f      %.2f    %.2f    %.2f      %.2f   %.2f     %.2f     %.2f'], DIC_latent);
S = sprintf([S '\n peff_h  %.2f       %.2f      %.2f     %.2f     %.2f      %.2f    %.2f      %.2f     %.2f'], p_eff);
S = sprintf([S '\n peff_a  %.2f      %.2f     %.2f     %.2f     %.2f     %.2f   %.2f     %.2f     %.2f'], p_eff2);
S = sprintf([S '\n peff_l  %.2f      %.2f      %.2f     %.2f    %.2f      %.2f    %.2f     %.2f     %.2f'], p_eff_latent);
S = sprintf([S '\n peff_l2 %.2f      %.2f      %.2f     %.2f    %.2f      %.2f    %.2f     %.2f     %.2f'], p_eff_latent2);
S = sprintf([S '\n ']);
S = sprintf([S '\n mlpd    %.2f       %.2f      %.2f     %.2f     %.2f      %.2f    %.2f    %.2f    %.2f'], mlpd_cv);
S = sprintf([S '\n rmse    %.2f       %.2f      %.2f     %.2f     %.2f      %.2f     %.2f     %.2f     %.2f'], mrmse_cv);
S = sprintf([S '\n ']);
S = sprintf([S '\n ']);
S = sprintf([S '\n The notation is as follows:']);
S = sprintf([S '\n DIC_h   = DIC with focus on hyperparameters. ']);
S = sprintf([S '\n DIC_a   = DIC with focus on hyperparameters and laten variables (all). ']);
S = sprintf([S '\n DIC_l   = DIC with focus on latent variables. ']);
S = sprintf([S '\n peff_h  = effective number of hyperparameters (latent variables marginalized). ']);
S = sprintf([S '\n peff_a  = effective number of hyperparameters and latent variables. ']);
S = sprintf([S '\n peff_l  = effective number of latent variables evaluated with gp_peff. ']);
S = sprintf([S '\n peff_l2 = effective number of latent variables evaluated with gp_dic. ']);
S = sprintf([S '\n mlpd    = mean log predictive density from the 10-fold CV. ']);
S = sprintf([S '\n mrmse   = mean root mean squared error from the 10-fold CV. ']);
S = sprintf([S '\n '])
