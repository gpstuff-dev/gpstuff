%DEMO_COMPARESPARSEGP Regression demo comparing different sparse
%                     approximations
%
%  Description
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
%  See also DEMO_REGRESSION1, DEMO_REGRESSION2
%
%
%  References:
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
[n,nin] = size(xt);

% Initialize full GP with a squared exponential component and set
% priors for their hyperparameters.
ppl = prior_t('s2', 10, 'nu', 3);
ppm = prior_sqrtt('s2', 10, 'nu', 3);
pn = prior_t('s2', 10, 'nu', 4);

gpcfse = gpcf_sexp('lengthScale',1.3,'magnSigma2',5, 'lengthScale_prior', ppl, 'magnSigma2_prior', ppm);
gpcfn = gpcf_noise('noiseSigma2', 0.3, 'noiseSigma2_prior', pn);

gp = gp_set('cf', {gpcfse}, 'noisef' ,{gpcfn}, 'jitterSigma2', 0.001,'infer_params','covariance') 

opt=optimset('GradObj','on');
opt=optimset(opt,'TolX', 1e-5);
opt=optimset(opt,'TolFun', 1e-5);
opt=optimset(opt,'LargeScale', 'off');
opt=optimset(opt,'Display', 'testing');
% Use fminunc to optimize the hyperparameters
w0 = gp_pak(gp); 
[w,fval,exitflag] = fminunc(@(ww) gp_eg(ww, gp, xt, yt), w0, opt);
gp = gp_unpak(gp,w);

[Ef_full, Varf_full] = gp_pred(gp, xt, yt, xstar);
Varf_full = Varf_full + gp.noisef{1}.noiseSigma2;

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

% Run FIC approximation for the same data: choose the inducing inputs Xu,
% then proceed with the inference with the optimized hyperparameters from
% the full GP: here, we optimize only the locations of the inducing inputs
% for the FIC model.
Xu=round(10+90*rand(18,1))/10; % Random placement

% Change type to FIC, add inducing inputs, and optimize only inducing inputs
gp_fic = gp_set(gp, 'type','FIC','X_u',Xu,'infer_params','inducing');

opt=optimset('GradObj','on');
opt=optimset(opt,'TolX', 1e-5);
opt=optimset(opt,'TolFun', 1e-5);
opt=optimset(opt,'LargeScale', 'off');
opt=optimset(opt,'Display', 'testing');
% Learn the hyperparameters
w0 = gp_pak(gp_fic);
[w,fval,exitflag] = fminunc(@(ww) gp_eg(ww, gp_fic, xt, yt), w0, opt);
gp_fic = gp_unpak(gp_fic,w);

[Ef_fic, Varf_fic] = gp_pred(gp_fic, xt, yt, xstar);
Varf_fic = Varf_fic + gp.noise{1}.noiseSigma2;

XuSorted=sort(gp_fic.X_u);
dXuSorted=diff(XuSorted);
bb=regress(dXuSorted,[ones(size(dXuSorted)) XuSorted(1:end-1)]);
plotbb=bb(1)+(min(XuSorted):0.1:max(XuSorted))*bb(2);

%figure;hold on
subplot(2,2,2);hold on;
plot(xstar,Ef_fic,'k', 'LineWidth', 2)
plot(xstar,Ef_fic-2.*sqrt(Varf_fic),'g--')
plot(xstar,Ef_fic+2.*sqrt(Varf_fic),'g--')
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
gp_var = gp_set(gp,'type','VAR','X_u',Xu,'infer_params','inducing');

opt=optimset('GradObj','on');
opt=optimset(opt,'TolX', 1e-5);
opt=optimset(opt,'TolFun', 1e-5);
opt=optimset(opt,'LargeScale', 'off');
opt=optimset(opt,'Display', 'testing');
% Learn the hyperparameters
w0 = gp_pak(gp_var);
[w,fval,exitflag] = fminunc(@(ww) gp_eg(ww, gp_var, xt, yt), w0, opt);
gp_var = gp_unpak(gp_var,w);

[Ef_var, Varf_var] = gp_pred(gp_var, xt, yt, xstar);
Varf_var = Varf_var + gp.noise{1}.noiseSigma2;

XuSorted=sort(gp_var.X_u);
dXuSorted=diff(XuSorted);
bb=regress(dXuSorted,[ones(size(dXuSorted)) XuSorted(1:end-1)]);
plotbb=bb(1)+(min(XuSorted):0.1:max(XuSorted))*bb(2);


%figure;hold on
subplot(2,2,4);hold on
plot(xstar,Ef_var,'k', 'LineWidth', 2)
plot(xstar,Ef_var-2.*sqrt(Varf_var),'g--')
plot(xstar,Ef_var+2.*sqrt(Varf_var),'g--')
plot(xt,yt,'.', 'MarkerSize',7)
plot(XuSorted(1:end-1),dXuSorted,'ko');
plot(min(XuSorted):0.1:max(XuSorted),plotbb,'k--')

plot(gp_var.X_u, -3, 'rx', 'MarkerSize', 5, 'LineWidth', 2)
plot(Xu, -2.8, 'bx', 'MarkerSize', 5, 'LineWidth', 2)
title('VAR')

% Run the DTC model similarly to the FIC model with the same starting
% inducing inputs. The difference in the optimized results is notable.
gp_dtc = gp_set(gp,'type','DTC','X_u',Xu,'infer_params','inducing');

opt=optimset('GradObj','on');
opt=optimset(opt,'TolX', 1e-5);
opt=optimset(opt,'TolFun', 1e-5);
opt=optimset(opt,'LargeScale', 'off');
opt=optimset(opt,'Display', 'testing');
% Learn the hyperparameters
w0 = gp_pak(gp_dtc);
[w,fval,exitflag] = fminunc(@(ww) gp_eg(ww, gp_dtc, xt, yt), w0, opt);
gp_dtc = gp_unpak(gp_dtc,w);

[Ef_dtc, Varf_dtc] = gp_pred(gp_dtc, xt, yt, xstar);
Varf_dtc = Varf_dtc + gp.noise{1}.noiseSigma2;

XuSorted=sort(gp_dtc.X_u);
dXuSorted=diff(XuSorted);
bb=regress(dXuSorted,[ones(size(dXuSorted)) XuSorted(1:end-1)]);
plotbb=bb(1)+(min(XuSorted):0.1:max(XuSorted))*bb(2);


%figure;hold on
subplot(2,2,3);hold on
plot(xstar,Ef_dtc,'k', 'LineWidth', 2)
plot(xstar,Ef_dtc-2.*sqrt(Varf_dtc),'g--')
plot(xstar,Ef_dtc+2.*sqrt(Varf_dtc),'g--')
plot(xt,yt,'.', 'MarkerSize',7)
plot(XuSorted(1:end-1),dXuSorted,'ko');
plot(min(XuSorted):0.1:max(XuSorted),plotbb,'k--')

plot(gp_dtc.X_u, -3, 'rx', 'MarkerSize', 5, 'LineWidth', 2)
plot(Xu, -2.8, 'bx', 'MarkerSize', 5, 'LineWidth', 2)
title('DTC')
