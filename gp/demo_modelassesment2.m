%DEMO_MODELASSESMENT2    Demonstration for model assesment when the observation 
%                        model is non-Gaussian
%
%    Description
%    We will consider the classification problem in
%    demo_classific. The analysis is conducted with full Gaussian
%    process usign both probit and logit likelihood. The performance
%    of these two models are compared by evaluating the DIC
%    statistics, number of efficient parameters and ten-fold cross
%    validation. The inference will be conducted using maximum a
%    posetrior (MAP) estimate for the hyperparameters using EP and
%    Laplace approximation, via full Markov chain Monte Carlo (MCMC)
%    and with an integration approximation (IA) for the
%    hyperparameters.
%
%    This demo is organised in two parts:
%     1) data analysis with with probit likelihood
%     2) data analysis with with logit likelihood
%
%   See also  DEMO_CLASSIFIC1, DEMO_MODELASSESMENT1

% Copyright (c) 2009-2010 Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.


% =====================================
% 1) data analysis with probit likelihood
% =====================================

S = which('demo_classific');
L = strrep(S,'demo_classific.m','demos/synth.tr');
x=load(L);
y=x(:,end);
y = 2.*y-1;
x(:,end)=[];
[n, nin] = size(x);

% Create covariance functions
gpcf1 = gpcf_sexp('init', 'lengthScale', [0.9 0.9], 'magnSigma2', 2);

% Set the prior for the parameters of covariance functions 
pl = prior_logunif('init');
gpcf1 = gpcf_sexp('set', gpcf1, 'lengthScale_prior', pl,'magnSigma2_prior', pl); %

% Create the likelihood structure
likelih = likelih_probit('init');

% Create the GP data structure
gp = gp_init('init', 'FULL', likelih, {gpcf1}, [],'jitterSigma2', 0.01.^2);


% ------- Laplace approximation --------

% Set the approximate inference method
gp = gp_init('set', gp, 'latent_method', {'Laplace', x, y});

fe=str2fun('gpla_e');
fg=str2fun('gpla_g');
n=length(y);
opt = scg2_opt;
opt.tolfun = 1e-3;
opt.tolx = 1e-3;
opt.display = 1;
opt.maxiter = 20;

% do scaled conjugate gradient optimization 
w=gp_pak(gp);
w=scg2(fe, w, opt, fg, gp, x, y);
gp=gp_unpak(gp,w);

% Evaluate the effective number of parameters and DIC with focus on latent variables.
models{1} = 'pr_Laplace';
p_eff_latent = gp_peff(gp, x, y);
[DIC_latent, p_eff_latent2] = gp_dic(gp, x, y, 'latent');

% Evaluate the 10-fold cross validation results. 
cvres = gp_kfcv(gp, x, y);
mlpd_cv(1) = cvres.mlpd_cv;
mrmse_cv(1) = cvres.mrmse_cv;

% ------- Expectation propagation --------

% Set the approximate inference method
gp = gp_init('set', gp, 'latent_method', {'EP', x, y});

w = gp_pak(gp);
fe=str2fun('gpep_e');
fg=str2fun('gpep_g');
n=length(y);
opt = scg2_opt;
opt.tolfun = 1e-3;
opt.tolx = 1e-3;
opt.display = 1;

% do scaled conjugate gradient optimization 
w=gp_pak(gp);
w=scg2(fe, w, opt, fg, gp, x, y);
gp=gp_unpak(gp,w);

% Evaluate the effective number of parameters and DIC with focus on latent variables.
models{2} = 'pr_EP';
p_eff_latent(2) = gp_peff(gp, x, y) ;
[DIC_latent(2), p_eff_latent2(2)] = gp_dic(gp, x, y, 'latent');

% Evaluate the 10-fold cross validation results. 
cvres = gp_kfcv(gp, x, y);
mlpd_cv(2) = cvres.mlpd_cv;
mrmse_cv(2) = cvres.mrmse_cv;

% ------- MCMC ---------------
% Set the approximate inference method
gp = gp_init('set', gp, 'latent_method', {'MCMC', zeros(size(y))', @scaled_mh});

% Set the parameters for MCMC...
clear('opt')
opt.hmc_opt.steps=10;
opt.hmc_opt.stepadj=0.1;
opt.hmc_opt.nsamples=1;
opt.latent_opt.display=0;
opt.latent_opt.repeat = 20;
opt.latent_opt.sample_latent_scale = 0.5;
hmc2('state', sum(100*clock))
[r, gp, opt] = gp_mc(gp, x, y, opt);

% Set the sampling options
opt.nsamples=200;
opt.repeat=1;
opt.hmc_opt.steps=4;
opt.hmc_opt.stepadj=0.02;
opt.latent_opt.repeat = 5;
hmc2('state', sum(100*clock));

% Sample 
rgp=gp_mc(gp, x, y, 'record', r, opt);

% Evaluate the effective number of parameters and DIC with focus on latent variables.
models{3} = 'pr_MCMC';
[DIC(3), p_eff(3)] =  gp_dic(rgp, x, y, 'hyper');
[DIC2(3), p_eff2(3)] =  gp_dic(rgp, x, y, 'all');

% Evaluate the 10-fold cross validation results. 
opt.nsamples=50;
cvres = gp_kfcv(gp, x, y, 'inf_method', 'MCMC', 'opt', opt);
mlpd_cv(3) = cvres.mlpd_cv;
mrmse_cv(3) = cvres.mrmse_cv;

% --- Integration approximation approach ---

% Use EP
gp = gp_init('set', gp, 'latent_method', {'EP', x, y});

% Find first the mode
w = gp_pak(gp);
fe=str2fun('gpep_e');
fg=str2fun('gpep_g');
n=length(y);
opt = scg2_opt;
opt.tolfun = 1e-3;
opt.tolx = 1e-3;
opt.display = 1;
w=gp_pak(gp);
w=scg2(fe, w, opt, fg, gp, x, y);
gp=gp_unpak(gp,w);

% now perform the integration
clear('opt')
opt.opt_scg = scg2_opt;
opt.int_method = 'grid';
opt.step_size = 2;

gp_array = gp_ia(gp, x, y, [], opt);

models{4} = 'pr_IA'; 
[DIC(4), p_eff(4)] =  gp_dic(gp_array, x, y, 'hyper');
[DIC2(4), p_eff2(4)] =  gp_dic(gp_array, x, y, 'all');

% Then the 10 fold cross-validation.
cvres = gp_kfcv(gp, x, y, 'inf_method', 'IA', 'opt', opt);
mlpd_cv(4) = cvres.mlpd_cv;
mrmse_cv(4) = cvres.mrmse_cv;

% =====================================
% 2) data analysis with logit likelihood
% =====================================

S = which('demo_classific');
L = strrep(S,'demo_classific.m','demos/synth.tr');
x=load(L);
y=x(:,end);
y = 2.*y-1;
x(:,end)=[];
[n, nin] = size(x);

% Create covariance functions
gpcf1 = gpcf_sexp('init', 'lengthScale', [0.9 0.9], 'magnSigma2', 2);

% Set the prior for the parameters of covariance functions 
pl = prior_logunif('init');
gpcf1 = gpcf_sexp('set', gpcf1, 'lengthScale_prior', pl,'magnSigma2_prior', pl); %

% Create the likelihood structure
likelih = likelih_logit('init');

% Create the GP data structure
gp = gp_init('init', 'FULL', likelih, {gpcf1}, [],'jitterSigma2', 0.01.^2);


% ------- Laplace approximation --------

% Set the approximate inference method
gp = gp_init('set', gp, 'latent_method', {'Laplace', x, y});

fe=str2fun('gpla_e');
fg=str2fun('gpla_g');
n=length(y);
opt = scg2_opt;
opt.tolfun = 1e-3;
opt.tolx = 1e-3;
opt.display = 1;
opt.maxiter = 20;

% do scaled conjugate gradient optimization 
w=gp_pak(gp);
w=scg2(fe, w, opt, fg, gp, x, y);
gp=gp_unpak(gp,w);

% Evaluate the effective number of parameters and DIC with focus on latent variables.
models{5} = 'lo_Laplace';
p_eff_latent(5) = gp_peff(gp, x, y);
[DIC_latent(5), p_eff_latent2(5)] = gp_dic(gp, x, y, 'latent');

% Evaluate the 10-fold cross validation results. NOTE! This saves the results in a 
% folder cv_resultsX (where X is a number) in your current workin directory. We save the 
% results only for this case so that you can study them. The other models are not saved.
cvres = gp_kfcv(gp, x, y);
mlpd_cv(5) = cvres.mlpd_cv;
mrmse_cv(5) = cvres.mrmse_cv;

% ------- Expectation propagation --------

% Set the approximate inference method
gp = gp_init('set', gp, 'latent_method', {'EP', x, y});

w = gp_pak(gp);
fe=str2fun('gpep_e');
fg=str2fun('gpep_g');
n=length(y);
opt = scg2_opt;
opt.tolfun = 1e-3;
opt.tolx = 1e-3;
opt.display = 1;

% do scaled conjugate gradient optimization 
w=gp_pak(gp);
w=scg2(fe, w, opt, fg, gp, x, y);
gp=gp_unpak(gp,w);

% Evaluate the effective number of parameters and DIC with focus on latent variables.
models{6} = 'lo_EP';
p_eff_latent(6) = gp_peff(gp, x, y) ;
[DIC_latent(6), p_eff_latent2(6)] = gp_dic(gp, x, y, 'latent');

% Evaluate the 10-fold cross validation results. 
cvres = gp_kfcv(gp, x, y);
mlpd_cv(6) = cvres.mlpd_cv;
mrmse_cv(6) = cvres.mrmse_cv;


% ------- MCMC ---------------
% Set the approximate inference method
gp = gp_init('set', gp, 'latent_method', {'MCMC', zeros(size(y))', @scaled_mh});

% Set the parameters for MCMC...
clear('opt')
opt.hmc_opt.steps=10;
opt.hmc_opt.stepadj=0.1;
opt.hmc_opt.nsamples=1;
opt.latent_opt.display=0;
opt.latent_opt.repeat = 20;
opt.latent_opt.sample_latent_scale = 0.5;
hmc2('state', sum(100*clock))
[r, gp, opt] = gp_mc(gp, x, y, opt);

% Set the sampling options
opt.nsamples=200;
opt.repeat=1;
opt.hmc_opt.steps=4;
opt.hmc_opt.stepadj=0.02;
opt.latent_opt.repeat = 5;
hmc2('state', sum(100*clock));

% Sample 
rgp = gp_mc(gp, x, y, 'record', r, opt);

% Evaluate the effective number of parameters and DIC with focus on latent variables.
models{7} = 'lo_MCMC';
[DIC(7), p_eff(7)] =  gp_dic(rgp, x, y, 'hyper');
[DIC2(7), p_eff2(7)] =  gp_dic(rgp, x, y, 'all');

% Evaluate the 10-fold cross validation results. 
opt.nsamples=50;
cvres = gp_kfcv(gp, x, y, 'inf_method', 'MCMC', 'opt', opt);
mlpd_cv(7) = cvres.mlpd_cv;
mrmse_cv(7) = cvres.mrmse_cv;


% --- Integration approximation approach ---

% Use EP
gp = gp_init('set', gp, 'latent_method', {'EP', x, y});

% Find first the mode
w = gp_pak(gp);
fe=str2fun('gpep_e');
fg=str2fun('gpep_g');
n=length(y);
opt = scg2_opt;
opt.tolfun = 1e-3;
opt.tolx = 1e-3;
opt.display = 1;
w=gp_pak(gp);
w=scg2(fe, w, opt, fg, gp, x, y);
gp=gp_unpak(gp,w);

% now perform the integration
clear('opt')
opt.opt_scg = scg2_opt;
opt.int_method = 'grid';
opt.step_size = 2;

gp_array = gp_ia(gp, x, y, [], opt);

models{8} = 'lo_IA'; 
[DIC(8), p_eff(8)] =  gp_dic(gp_array, x, y, 'hyper');
[DIC2(8), p_eff2(8)] =  gp_dic(gp_array, x, y, 'all');

% Then the 10 fold cross-validation.
cvres = gp_kfcv(gp, x, y, 'inf_method', 'IA', 'opt', opt);
mlpd_cv(8) = cvres.mlpd_cv;
mrmse_cv(8) = cvres.mrmse_cv;


%========================================================
% PART 4 Print the results
%========================================================

S = '       ';
for i = 1:length(models)
    S = [S '  ' models{i}];
end

S = sprintf([S '\n DIC_h     %.2f      %.2f    %.2f  %.2f     %.2f     %.2f   %.2f   %.2f'], DIC);
S = sprintf([S '\n DIC_a     %.2f      %.2f    %.2f  %.2f     %.2f     %.2f   %.2f   %.2f'], DIC2);
S = sprintf([S '\n DIC_l   %.2f    %.2f      %.2f    %.2f   %.2f   %.2f    %.2f     %.2f'], DIC_latent);
S = sprintf([S '\n peff_h    %.2f      %.2f      %.2f    %.2f     %.2f     %.2f     %.2f     %.2f'], p_eff);
S = sprintf([S '\n peff_a    %.2f      %.2f      %.2f    %.2f     %.2f     %.2f     %.2f     %.2f'], p_eff2);
S = sprintf([S '\n peff_l    %.2f      %.2f      %.2f    %.2f     %.2f     %.2f    %.2f     %.2f'], p_eff_latent);
S = sprintf([S '\n peff_l2  %.2f      %.2f      %.2f    %.2f    %.2f     %.2f    %.2f     %.2f'], p_eff_latent2);
S = sprintf([S '\n ']);
S = sprintf([S '\n mlpd    %.2f      %.2f     %.2f   %.2f    %.2f    %.2f    %.2f    %.2f'], mlpd_cv);
S = sprintf([S '\n ']);
S = sprintf([S '\n ']);
S = sprintf([S '\n The notation is as follows:']);
S = sprintf([S '\n pr_*    = probit likelihood and inference method']);
S = sprintf([S '\n lo_*    = logit likelihood and inference method']);
S = sprintf([S '\n DIC_h   = DIC with focus on hyperparameters. ']);
S = sprintf([S '\n DIC_a   = DIC with focus on hyperparameters and laten variables (all). ']);
S = sprintf([S '\n DIC_l   = DIC with focus on latent variables. ']);
S = sprintf([S '\n peff_h  = effective number of hyperparameters (latent variables marginalized). ']);
S = sprintf([S '\n peff_a  = effective number of hyperparameters and latent variables. ']);
S = sprintf([S '\n peff_l  = effective number of latent variables evaluated with gp_peff. ']);
S = sprintf([S '\n peff_l2 = effective number of latent variables evaluated with gp_dic. ']);
S = sprintf([S '\n mlpd    = mean log predictive density from the 10-fold CV. ']);
S = sprintf([S '\n '])














































% 
% 
% 
% % =============================================
% % =============================================
% % FIC model
% % =============================================
% % =============================================
% 
% 
% S = which('demo_classific');
% L = strrep(S,'demo_classific.m','demos/synth.tr');
% x=load(L);
% y=x(:,end);
% y = 2.*y-1;
% x(:,end)=[];
% [n, nin] = size(x);
% 
% % Create covariance functions
% gpcf1 = gpcf_sexp('init', 'lengthScale', [0.9 0.9], 'magnSigma2', 2);
% 
% % Set the prior for the parameters of covariance functions 
% pl = prior_logunif('init');
% gpcf1 = gpcf_sexp('set', gpcf1, 'lengthScale_prior', pl,'magnSigma2_prior', pl); %
% 
% % Create the likelihood structure
% likelih = likelih_probit('init');
% %likelih = likelih_logit('init');
% 
% % Set the inducing inputs
% [u1,u2]=meshgrid(linspace(-1.25, 0.9,6),linspace(-0.2, 1.1,6));
% Xu=[u1(:) u2(:)];
% Xu = Xu([3 4 7:18 20:24 26:30 33:36],:);
% 
% % Create the GP data structure
% gp = gp_init('init', 'FIC', likelih, {gpcf1}, [],'jitterSigma2', 0.01, 'X_u', Xu, 'infer_params', 'covariance');
% 
% gp = gp_init('set', gp, 'infer_params', 'covariance');           % optimize only hyperparameters
% 
% % ------- Laplace approximation --------
% 
% % Set the approximate inference method
% gp = gp_init('set', gp, 'latent_method', {'Laplace', x, y});
% 
% fe=str2fun('gpla_e');
% fg=str2fun('gpla_g');
% n=length(y);
% opt = scg2_opt;
% opt.tolfun = 1e-3;
% opt.tolx = 1e-3;
% opt.display = 1;
% opt.maxiter = 20;
% 
% % do scaled conjugate gradient optimization 
% w=gp_pak(gp);
% w=scg2(fe, w, opt, fg, gp, x, y);
% gp=gp_unpak(gp,w);
% 
% % Evaluate the effective number of parameters and DIC with focus on latent variables.
% models{5} = 'full_Laplace';
% p_eff_latent(5) = gp_peff(gp, x, y);
% [DIC_latent(5), p_eff_latent2(5)] = gp_dic(gp, x, y);
% 
% % Evaluate the 10-fold cross validation results. NOTE! This saves the results in a 
% % folder cv_resultsX (where X is a number) in your current workin directory. We save the 
% % results only for this case so that you can study them. The other models are not saved.
% cvres = gp_kfcv(gp, x, y)
% 
% 
% % ------- Expectation propagation --------
% 
% % Set the approximate inference method
% gp = gp_init('set', gp, 'latent_method', {'EP', x, y});
% 
% w = gp_pak(gp);
% fe=str2fun('gpep_e');
% fg=str2fun('gpep_g');
% n=length(y);
% opt = scg2_opt;
% opt.tolfun = 1e-3;
% opt.tolx = 1e-3;
% opt.display = 1;
% 
% % do scaled conjugate gradient optimization 
% w=gp_pak(gp);
% w=scg2(fe, w, opt, fg, gp, x, y);
% gp=gp_unpak(gp,w);
% 
% % Evaluate the effective number of parameters and DIC with focus on latent variables.
% models{6} = 'fic_EP';
% p_eff_latent(6) = gp_peff(gp, x, y) ;
% [DIC_latent(6), p_eff_latent2(6)] = gp_dic(gp, x, y);
% 
% % Evaluate the 10-fold cross validation results. 
% cvres = gp_kfcv(gp, x, y);
% 
% 
% % ------- MCMC ---------------
% % Set the approximate inference method
% gp = gp_init('set', gp, 'latent_method', {'MCMC', zeros(size(y))', @scaled_mh});
% 
% % Set the parameters for MCMC...
% clear('opt')
% opt.repeat=1;
% opt.nsamples=200;
% opt.hmc_opt.steps=4;
% opt.hmc_opt.stepadj=0.02;
% opt.hmc_opt.nsamples=1;
% opt.latent_opt.display=0;
% opt.latent_opt.repeat = 5;
% opt.latent_opt.sample_latent_scale = 0.5;
% hmc2('state', sum(100*clock));
% 
% % Sample 
% rgp=gp_mc(opt, gp, x, y);
% 
% % Evaluate the effective number of parameters and DIC with focus on latent variables.
% models{7} = 'FIC_MCMC';
% [DIC(7), p_eff(7)] =  gp_dic(rgp, x, y, 'hyper');
% [DIC2(7), p_eff2(7)] =  gp_dic(rgp, x, y, 'all');
% 
% % Evaluate the 10-fold cross validation results. 
% opt.nsamples=50;
% [mlpd_cv(7), Var_lpd_cv(7)] = gp_kfcv(gp, x, y, 'inf_method', 'MCMC', 'opt', opt, 'SAVE', 0);
% 
% 
% % --- Integration approximation approach ---
% 
% % Use EP
% gp = gp_init('set', gp, 'latent_method', {'EP', x, y});
% 
% % Find first the mode
% w = gp_pak(gp);
% fe=str2fun('gpep_e');
% fg=str2fun('gpep_g');
% n=length(y);
% opt = scg2_opt;
% opt.tolfun = 1e-3;
% opt.tolx = 1e-3;
% opt.display = 1;
% w=gp_pak(gp);
% w=scg2(fe, w, opt, fg, gp, x, y);
% gp=gp_unpak(gp,w);
% 
% % now perform the integration
% clear('opt')
% opt.opt_scg = scg2_opt;
% opt.int_method = 'grid';
% opt.step_size = 2.5;
% 
% gp_array = gp_ia(gp, x, y, [], opt);
% 
% models{8} = 'full_IA'; 
% [DIC(8), p_eff(8)] =  gp_dic(gp_array, x, y);
% [DIC2(8), p_eff2(8)] =  gp_dic(gp_array, x, y, 'all');
% 
% % Then the 10 fold cross-validation.
% [mlpd_cv(8), Var_lpd_cv(8), mrmse_cv(8), Var_rmse_cv(8)] = gp_kfcv(gp, x, y, 'inf_method', 'IA', 'opt', opt, 'SAVE', 0);
% 
% 
% 
% 
% % =============================================
% % =============================================
% % PIC model
% % =============================================
% % =============================================
% 
% 
% S = which('demo_classific1');
% L = strrep(S,'demo_classific1.m','demos/synth.tr');
% x=load(L);
% y=x(:,end);
% y = 2.*y-1;
% x(:,end)=[];
% [n, nin] = size(x);
% 
% % Create covariance functions
% gpcf1 = gpcf_sexp('init', 'lengthScale', [0.9 0.9], 'magnSigma2', 2);
% 
% % Set the prior for the parameters of covariance functions 
% pl = prior_logunif('init');
% gpcf1 = gpcf_sexp('set', gpcf1, 'lengthScale_prior', pl,'magnSigma2_prior', pl); %
% 
% % Create the likelihood structure
% likelih = likelih_probit('init');
% %likelih = likelih_logit('init', y);
% 
% % Set the inducing inputs
% [u1,u2]=meshgrid(linspace(-1.25, 0.9,6),linspace(-0.2, 1.1,6));
% Xu=[u1(:) u2(:)];
% Xu = Xu([3 4 7:18 20:24 26:30 33:36],:);
% 
% % Set the blocks
% xtmp = x - repmat(min(x),n,1) +1;
% [trindex] = set_PIC(xtmp, [1 max(xtmp(:,2))+.01 1 max(xtmp(:,1))+.01], 0.3, 'corners', 0);
% trindex{2} = [trindex{2} ; trindex{1}]; trindex = {trindex{2:19}};
% 
% % Create the GP data structure
% gp = gp_init('init', 'PIC', likelih, {gpcf1}, [],'jitterSigma2', 0.01);
% gp = gp_init('set', gp, 'X_u', Xu, 'blocks', trindex, 'infer_params', 'covariance')
% 
% % ------- Laplace approximation --------
% 
% % Set the approximate inference method
% gp = gp_init('set', gp, 'latent_method', {'Laplace', x, y});
% 
% fe=str2fun('gpla_e');
% fg=str2fun('gpla_g');
% n=length(y);
% opt = scg2_opt;
% opt.tolfun = 1e-3;
% opt.tolx = 1e-3;
% opt.display = 1;
% opt.maxiter = 20;
% 
% % do scaled conjugate gradient optimization 
% w=gp_pak(gp);
% w=scg2(fe, w, opt, fg, gp, x, y);
% gp=gp_unpak(gp,w);
% 
% % Evaluate the effective number of parameters and DIC with focus on latent variables.
% models{9} = 'PIC_Laplace';
% p_eff_latent(9) = gp_peff(gp, x, y);
% [DIC_latent(9), p_eff_latent2(9)] = gp_dic(gp, x, y, 'latent');
% 
% % Evaluate the 10-fold cross validation results. NOTE! This saves the results in a 
% % folder cv_resultsX (where X is a number) in your current workin directory. We save the 
% % results only for this case so that you can study them. The other models are not saved.
% [mlpd_cv(9), Var_lpd_cv(9)] = gp_kfcv(gp, x, y, 'SAVE', 0);
% 
% 
% % ------- Expectation propagation --------
% 
% % Set the approximate inference method
% gp = gp_init('set', gp, 'latent_method', {'EP', x, y});
% 
% w = gp_pak(gp);
% fe=str2fun('gpep_e');
% fg=str2fun('gpep_g');
% n=length(y);
% opt = scg2_opt;
% opt.tolfun = 1e-3;
% opt.tolx = 1e-3;
% opt.display = 1;
% 
% % do scaled conjugate gradient optimization 
% w=gp_pak(gp);
% w=scg2(fe, w, opt, fg, gp, x, y);
% gp=gp_unpak(gp,w);
% 
% % Evaluate the effective number of parameters and DIC with focus on latent variables.
% models{10} = 'PIC_EP';
% p_eff_latent(10) = gp_peff(gp, x, y);
% [DIC_latent(10), p_eff_latent2(10)] = gp_dic(gp, x, y, 'latent');
% 
% % Evaluate the 10-fold cross validation results. 
% [mlpd_cv(10), Var_lpd_cv(10)] = gp_kfcv(gp, x, y, 'MAP_scg2', opt, 'covariance', 0);
% 
% 
% % ------- MCMC ---------------
% % Set the approximate inference method
% gp = gp_init('set', gp, 'latent_method', {'MCMC', zeros(size(y))', @scaled_mh});
% 
% % Set the parameters for MCMC...
% clear('opt')
% opt.repeat=1;
% opt.nsamples=200;
% opt.hmc_opt.steps=4;
% opt.hmc_opt.stepadj=0.02;
% opt.hmc_opt.nsamples=1;
% opt.latent_opt.display=0;
% opt.latent_opt.repeat = 5;
% opt.latent_opt.sample_latent_scale = 0.5;
% hmc2('state', sum(100*clock));
% 
% % Sample 
% [rgp,g,rstate2]=gp_mc(opt, gp, x, y);
% 
% % Evaluate the effective number of parameters and DIC with focus on latent variables.
% models{7} = 'PIC_MCMC';
% [DIC(11), p_eff(11)] =  gp_dic(rgp, x, y, 'covariance');
% [DIC2(11), p_eff2(11)] =  gp_dic(rgp, x, y, 'covariance', 'all');
% 
% % Evaluate the 10-fold cross validation results. 
% opt.nsamples=50;
% [mlpd_cv(11), Var_lpd_cv(11)] = gp_kfcv(gp, x, y, 'MCMC', opt, 'covariance', 0);
% 
% 
% % --- Integration approximation approach ---
% 
% % Use EP
% gp = gp_init('set', gp, 'latent_method', {'EP', x, y});
% 
% % Find first the mode
% w = gp_pak(gp);
% fe=str2fun('gpep_e');
% fg=str2fun('gpep_g');
% n=length(y);
% opt = scg2_opt;
% opt.tolfun = 1e-3;
% opt.tolx = 1e-3;
% opt.display = 1;
% w=gp_pak(gp);
% w=scg2(fe, w, opt, fg, gp, x, y);
% gp=gp_unpak(gp,w);
% 
% % now perform the integration
% clear('opt')
% opt.opt_scg = scg2_opt;
% opt.int_method = 'grid';
% opt.step_size = 2.5;
% 
% gp_array = gp_ia(gp, x, y, [], opt);
% 
% 
% models{12} = 'full_IA'; 
% [DIC(12), p_eff(12)] =  gp_dic(gp_array, x, y);
% [DIC2(12), p_eff2(12)] =  gp_dic(gp_array, x, y, [], 'all');
% 
% % Then the 10 fold cross-validation.
% [mlpd_cv(12), Var_lpd_cv(12), mrmse_cv(12), Var_rmse_cv(12)] = gp_kfcv(gp, x, y, 'IA', opt, 'covariance', 0);
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% % =============================================
% % =============================================
% % CS+FIC model
% % =============================================
% % =============================================
% 
% S = which('demo_classific1');
% L = strrep(S,'demo_classific1.m','demos/synth.tr');
% x=load(L);
% y=x(:,end);
% y = 2.*y-1;
% x(:,end)=[];
% [n, nin] = size(x);
% 
% % Create covariance functions
% gpcf1 = gpcf_sexp('init', 'lengthScale', [0.9 0.9], 'magnSigma2', 2);
% 
% % Set the prior for the parameters of covariance functions 
% pl = prior_logunif('init');
% gpcf1 = gpcf_sexp('set', gpcf1, 'lengthScale_prior', pl,'magnSigma2_prior', pl); %
% 
% % Create the likelihood structure
% likelih = likelih_probit('init', y);
% %likelih = likelih_logit('init', y);
% 
% % Set the inducing inputs
% [u1,u2]=meshgrid(linspace(-1.25, 0.9,6),linspace(-0.2, 1.1,6));
% Xu=[u1(:) u2(:)];
% Xu = Xu([3 4 7:18 20:24 26:30 33:36],:);
% 
% pl = prior_t('init', 's2', 3, 'nu', 4);
% pm = prior_t('init', 's2', 0.3, 'nu', 4);
% gpcf1 = gpcf_sexp('init', 'lengthScale', 5, 'magnSigma2', 3, 'lengthScale_prior', pl, 'magnSigma2_prior', pm);
% gpcf3 = gpcf_ppcs2('init', 'nin', nin, 'lengthScale', 2, 'magnSigma2', 3, 'lengthScale_prior', pm, 'magnSigma2_prior', pm);
% 
% % Create the GP data structure
% gp = gp_init('init', 'CS+FIC', likelih, {gpcf1, gpcf3}, [],'jitterSigma2', 0.01, 'X_u', Xu, 'infer_params', 'covariance');
% 
% 
% % ------- Laplace approximation --------
% 
% % Set the approximate inference method
% gp = gp_init('set', gp, 'latent_method', {'Laplace', x, y});
% 
% fe=str2fun('gpla_e');
% fg=str2fun('gpla_g');
% n=length(y);
% opt = scg2_opt;
% opt.tolfun = 1e-3;
% opt.tolx = 1e-3;
% opt.display = 1;
% opt.maxiter = 20;
% 
% % do scaled conjugate gradient optimization 
% w=gp_pak(gp);
% w=scg2(fe, w, opt, fg, gp, x, y);
% gp=gp_unpak(gp,w);
% 
% % Evaluate the effective number of parameters and DIC with focus on latent variables.
% models{13} = 'CS+FIC_Laplace';
% p_eff_latent(13) = gp_peff(gp, x, y);
% [DIC_latent(13), p_eff_latent2(13)] = gp_dic(gp, x, y);
% 
% % Evaluate the 10-fold cross validation results. NOTE! This saves the results in a 
% % folder cv_resultsX (where X is a number) in your current workin directory. We save the 
% % results only for this case so that you can study them. The other models are not saved.
% [mlpd_cv(13), Var_lpd_cv(13)] = gp_kfcv(gp, x, y, 'MAP_scg2', opt, 'covariance', 0);
% 
% 
% % ------- Expectation propagation --------
% 
% % Set the approximate inference method
% gp = gp_init('set', gp, 'latent_method', {'EP', x, y, 'covariance'});
% 
% w = gp_pak(gp, 'covariance');
% fe=str2fun('gpep_e');
% fg=str2fun('gpep_g');
% n=length(y);
% opt = scg2_opt;
% opt.tolfun = 1e-3;
% opt.tolx = 1e-3;
% opt.display = 1;
% 
% % do scaled conjugate gradient optimization 
% w=gp_pak(gp, 'covariance');
% w=scg2(fe, w, opt, fg, gp, x, y, 'covariance');
% gp=gp_unpak(gp,w, 'covariance');
% 
% % Evaluate the effective number of parameters and DIC with focus on latent variables.
% models{14} = 'CS+FIC_EP';
% p_eff_latent(14) = gp_peff(gp, x, y, 'covariance') ;
% [DIC_latent(14), p_eff_latent2(14)] = gp_dic(gp, x, y, 'covariance');
% 
% % Evaluate the 10-fold cross validation results. 
% [mlpd_cv(14), Var_lpd_cv(14)] = gp_kfcv(gp, x, y, 'MAP_scg2', opt, 'covariance', 0);
% 
% 
% % ------- MCMC ---------------
% % Set the approximate inference method
% gp = gp_init('set', gp, 'latent_method', {'MCMC', zeros(size(y))', @scaled_mh});
% 
% % Set the parameters for MCMC...
% opt=gp_mcopt;
% opt.repeat=1;
% opt.nsamples=200;
% opt.hmc_opt.steps=4;
% opt.hmc_opt.stepadj=0.02;
% opt.hmc_opt.nsamples=1;
% opt.latent_opt.display=0;
% opt.latent_opt.repeat = 5;
% opt.latent_opt.sample_latent_scale = 0.5;
% hmc2('state', sum(100*clock));
% 
% % Sample 
% [rgp,g,rstate2]=gp_mc(opt, gp, x, y);
% 
% % Evaluate the effective number of parameters and DIC with focus on latent variables.
% models{15} = 'CS+FIC_MCMC';
% [DIC(15), p_eff(15)] =  gp_dic(rgp, x, y, 'covariance');
% [DIC2(15), p_eff2(15)] =  gp_dic(rgp, x, y, 'covariance', 'all');
% 
% % Evaluate the 10-fold cross validation results. 
% opt.nsamples=50;
% [mlpd_cv(15), Var_lpd_cv(15)] = gp_kfcv(gp, x, y, 'MCMC', opt, 'covariance', 0);
% 
% 
% % --- Integration approximation approach ---
% 
% % Use EP
% gp = gp_init('set', gp, 'latent_method', {'EP', x, y, 'covariance'});
% 
% % Find first the mode
% w = gp_pak(gp, 'covariance');
% fe=str2fun('gpep_e');
% fg=str2fun('gpep_g');
% n=length(y);
% opt = scg2_opt;
% opt.tolfun = 1e-3;
% opt.tolx = 1e-3;
% opt.display = 1;
% w=gp_pak(gp, 'covariance');
% w=scg2(fe, w, opt, fg, gp, x, y, 'covariance');
% gp=gp_unpak(gp,w, 'covariance');
% 
% % now perform the integration
% opt = gp_iaopt([], 'grid');
% opt.step_size = 2;
% gp_array = gp_ia(opt, gp, x, y, [], 'covariance');
% 
% models{16} = 'full_IA'; 
% [DIC(16), p_eff(16)] =  gp_dic(gp_array, x, y);
% [DIC2(16), p_eff2(16)] =  gp_dic(gp_array, x, y, 'covariance', 'all');
% 
% % Then the 10 fold cross-validation.
% [mlpd_cv(16), Var_lpd_cv(16), mrmse_cv(16), Var_rmse_cv(16)] = gp_kfcv(gp, x, y, 'IA', opt, 'covariance', 0);
% 
