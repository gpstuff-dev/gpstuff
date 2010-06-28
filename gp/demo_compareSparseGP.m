%DEMO_COMPARESPARSEGP     Regression demo comparing different sparse
%                         approximations
%
%   Description
%   A regression problem with one input variable and one output
%   variable with Gaussian noise. The output is assumed to be
%   realization of additive functions and Gaussian noise.
% 
%   For standard full GP demonstration, see for example
%   DEMO_REGRESSION1, DEMO_REGRESSION2, and for detailed discussion,
%   Rasmussen and Williams (2006)
% 
%   In this demo, sparse approximations for the full GP model are
%   compared. We use
%     - FIC, fully independent conditional
%     - DTC, deterministic training conditional
%     - VAR, variational approach
%     
%   For technical details, see Quinonero-Candela and Rasmussen (2005)
%   for the FIC and DTC models and Titsias (2009) for the VAR model.
% 
%   We use a simple one dimensional data set to present the three
%   methods.
% 
%   See also DEMO_REGRESSION1, DEMO_REGRESSION2
%
%
%   References:
% 
%    Quinonero-Candela, J. and Rasmussen, C. E. (2005). A Unifying
%    View of Sparse Approximate Gaussian Process Regression. Journal
%    of Machine Learning Research.
% 
%    Rasmussen, C. E. and Williams, C. K. I. (2006). Gaussian
%    Processes for Machine Learning. The MIT Press.
% 
%    Titsias, M. K. (2009). Variational Model Selection for Sparse
%    Gaussian Process Regression. Technical Report, University of
%    Manchester.

% Copyright (c) 2010 Heikki Peura

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.



% Start by creating 1D data
x=linspace(1,10,901);

% Choose a subset of data so that the data are less dense in the right end.
% xt are the inputs and yt are the outputs, xstar are the values we want to
% predict.
x1=logspace(0,1,100);
x1=round(x1*100)-99;
xt=x(x1)';
yt=2*sin(4*xt)+0.2*randn(size(xt));
xstar=[1:0.01:14]';

% Initialize full GP with a squared exponential component and set
% priors for their hyperparameters.
[n,nin] = size(xt);
gpcfse = gpcf_sexp('init','lengthScale',1.3,'magnSigma2',5);
gpcfn = gpcf_noise('init', 'noiseSigma2', 0.3);

ppl = prior_t('init', 's2', 10, 'nu', 3);
ppm = prior_t('init', 's2', 10, 'nu', 3);
pn = prior_t('init', 's2', 10, 'nu', 4);

gpcfn = gpcf_noise('set', gpcfn, 'noiseSigma2_prior', pn);
gpcfse = gpcf_periodic('set', gpcfse, 'lengthScale_prior', ppl, 'magnSigma2_prior', ppm);

gp = gp_init('init', 'FULL', 'gaussian', {gpcfse}, {gpcfn}, 'jitterSigma2', 0.001,'infer_params','covariance') 

opt=optimset('GradObj','on');
opt=optimset(opt,'TolX', 1e-5);
opt=optimset(opt,'TolFun', 1e-5);
opt=optimset(opt,'LargeScale', 'off');
opt=optimset(opt,'Display', 'testing');
% Learn the hyperparameters
w0 = gp_pak(gp); % pack hyperparameters into a (log) vector for the optimization
mydeal = @(varargin)varargin{1:nargout};
[w,fval,exitflag] = fminunc(@(ww) mydeal(gp_e(ww, gp, xt, yt), gp_g(ww, gp, xt, yt)), w0, opt);
gp = gp_unpak(gp,w);

[Ef_full, Varf_full] = gp_pred(gp, xt, yt, xstar);
Varf_full = Varf_full + gp.noise{1}.noiseSigma2;

figure;hold on
% Blue crosses are the initial inducing input locations, red ones are
% the optimised ones. Black circles represent the distance to the next
% optimized location, with a dashed trendline.');

subplot(2,2,1);hold on;
plot(xstar,Ef_full,'k', 'LineWidth', 2)
plot(xstar,Ef_full-2.*sqrt(Varf_full),'g--')
plot(xstar,Ef_full+2.*sqrt(Varf_full),'g--')
plot(xt,yt,'.', 'MarkerSize',7)
title('FULL GP')
w_full=w; % optimized hyperparameters

% Run FIC approximation fot the same data: choose the inducing inputs Xu,
% then proceed with the inference with the optimized hyperparameters from
% the full GP: here, we optimize only the locations of the inducing inputs
% for the FIC model.
Xu=round(10+90*rand(18,1))/10; % Random placement

gp_fic = gp_init('init', 'FIC', 'gaussian', {gpcfse}, {gpcfn}, 'jitterSigma2', 0.001,'infer_params','inducing','X_u',Xu);
gp_fic.cf{1}.lengthScale=exp(w_full(2));
gp_fic.cf{1}.magnSigma2=exp(w_full(1));
gp_fic.noise{1}.noiseSigma2=exp(w_full(end));

opt=optimset('GradObj','on');
opt=optimset(opt,'TolX', 1e-5);
opt=optimset(opt,'TolFun', 1e-5);
opt=optimset(opt,'LargeScale', 'off');
opt=optimset(opt,'Display', 'testing');
% Learn the hyperparameters
w0 = gp_pak(gp_fic);
mydeal = @(varargin)varargin{1:nargout};
[w,fval,exitflag] = fminunc(@(ww) mydeal(gp_e(ww, gp_fic, xt, yt), gp_g(ww, gp_fic, xt, yt)), w0, opt);
gp_fic = gp_unpak(gp_fic,w);

[Ef_full, Varf_full] = gp_pred(gp_fic, xt, yt, xstar);
Varf_full = Varf_full + gp.noise{1}.noiseSigma2;

XuSorted=sort(gp_fic.X_u);
dXuSorted=diff(XuSorted);
bb=regress(dXuSorted,[ones(size(dXuSorted)) XuSorted(1:end-1)]);
plotbb=bb(1)+(min(XuSorted):0.1:max(XuSorted))*bb(2);

%figure;hold on
subplot(2,2,2);hold on;
plot(xstar,Ef_full,'k', 'LineWidth', 2)
plot(xstar,Ef_full-2.*sqrt(Varf_full),'g--')
plot(xstar,Ef_full+2.*sqrt(Varf_full),'g--')
plot(xt,yt,'.', 'MarkerSize',7)
plot(XuSorted(1:end-1),dXuSorted,'ko');
plot(min(XuSorted):0.1:max(XuSorted),plotbb,'k--')

plot(gp_fic.X_u, -3, 'rx', 'MarkerSize', 5, 'LineWidth', 2)
plot(Xu, -2.8, 'bx', 'MarkerSize', 5, 'LineWidth', 2)
title('FIC')


% Run the VAR model similarly to the FIC model with the same starting
% inducing inputs. The difference in the optimized results is notable.
% The VAR model places the inducing inputs quite evenly (slightly increasing as
% the data becomes more sparse), with predictions closely matching the full 
% GP model. The other two sparse approximations yield less reliable
% results.
gp_var = gp_init('init', 'VAR', 'gaussian', {gpcfse}, {gpcfn}, 'jitterSigma2', 0.001,'infer_params','inducing','X_u',Xu);
gp_var.cf{1}.lengthScale=exp(w_full(2));
gp_var.cf{1}.magnSigma2=exp(w_full(1));
gp_var.noise{1}.noiseSigma2=exp(w_full(end));

opt=optimset('GradObj','on');
opt=optimset(opt,'TolX', 1e-5);
opt=optimset(opt,'TolFun', 1e-5);
opt=optimset(opt,'LargeScale', 'off');
opt=optimset(opt,'Display', 'testing');
% Learn the hyperparameters
w0 = gp_pak(gp_var);
mydeal = @(varargin)varargin{1:nargout};
[w,fval,exitflag] = fminunc(@(ww) mydeal(gp_e(ww, gp_var, xt, yt), gp_g(ww, gp_var, xt, yt)), w0, opt);
gp_var = gp_unpak(gp_var,w);

[Ef_full, Varf_full] = gp_pred(gp_var, xt, yt, xstar);
Varf_full = Varf_full + gp.noise{1}.noiseSigma2;

XuSorted=sort(gp_var.X_u);
dXuSorted=diff(XuSorted);
bb=regress(dXuSorted,[ones(size(dXuSorted)) XuSorted(1:end-1)]);
plotbb=bb(1)+(min(XuSorted):0.1:max(XuSorted))*bb(2);


%figure;hold on
subplot(2,2,4);hold on
plot(xstar,Ef_full,'k', 'LineWidth', 2)
plot(xstar,Ef_full-2.*sqrt(Varf_full),'g--')
plot(xstar,Ef_full+2.*sqrt(Varf_full),'g--')
plot(xt,yt,'.', 'MarkerSize',7)
plot(XuSorted(1:end-1),dXuSorted,'ko');
plot(min(XuSorted):0.1:max(XuSorted),plotbb,'k--')

plot(gp_var.X_u, -3, 'rx', 'MarkerSize', 5, 'LineWidth', 2)
plot(Xu, -2.8, 'bx', 'MarkerSize', 5, 'LineWidth', 2)
title('VAR')


% Run the DTC model similarly to the FIC model with the same starting
% inducing inputs. The difference in the optimized results is notable.
gp_dtc = gp_init('init', 'DTC', 'gaussian', {gpcfse}, {gpcfn}, 'jitterSigma2', 0.001,'infer_params','inducing','X_u',Xu);
gp_dtc.cf{1}.lengthScale=exp(w_full(2));
gp_dtc.cf{1}.magnSigma2=exp(w_full(1));
gp_dtc.noise{1}.noiseSigma2=exp(w_full(end));

opt=optimset('GradObj','on');
opt=optimset(opt,'TolX', 1e-5);
opt=optimset(opt,'TolFun', 1e-5);
opt=optimset(opt,'LargeScale', 'off');
opt=optimset(opt,'Display', 'testing');
% Learn the hyperparameters
w0 = gp_pak(gp_dtc);
mydeal = @(varargin)varargin{1:nargout};
[w,fval,exitflag] = fminunc(@(ww) mydeal(gp_e(ww, gp_dtc, xt, yt), gp_g(ww, gp_dtc, xt, yt)), w0, opt);
gp_dtc = gp_unpak(gp_dtc,w);

[Ef_full, Varf_full] = gp_pred(gp_dtc, xt, yt, xstar);
Varf_full = Varf_full + gp.noise{1}.noiseSigma2;

XuSorted=sort(gp_dtc.X_u);
dXuSorted=diff(XuSorted);
bb=regress(dXuSorted,[ones(size(dXuSorted)) XuSorted(1:end-1)]);
plotbb=bb(1)+(min(XuSorted):0.1:max(XuSorted))*bb(2);


%figure;hold on
subplot(2,2,3);hold on
plot(xstar,Ef_full,'k', 'LineWidth', 2)
plot(xstar,Ef_full-2.*sqrt(Varf_full),'g--')
plot(xstar,Ef_full+2.*sqrt(Varf_full),'g--')
plot(xt,yt,'.', 'MarkerSize',7)
plot(XuSorted(1:end-1),dXuSorted,'ko');
plot(min(XuSorted):0.1:max(XuSorted),plotbb,'k--')

plot(gp_dtc.X_u, -3, 'rx', 'MarkerSize', 5, 'LineWidth', 2)
plot(Xu, -2.8, 'bx', 'MarkerSize', 5, 'LineWidth', 2)
title('DTC')
