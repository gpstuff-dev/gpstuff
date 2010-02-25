%DEMO_REGRESSION2    Regression problem demonstration for modeling multible phenomenon
%                    
%
%    Description
%    A regression problem with one input variable and one output variable with 
%    Gaussian noise. The output is assumed to be realization of two additive 
%    functions and Gaussian noise.
%
%    The model constructed is following:
%
%    The observations y are assumed to satisfy
%
%         y = f + g + e,    where e ~ N(0, s^2).
%
%    f and g are underlying latent functions, which we are interested in. 
%    We place a zero mean Gaussian process prior them, which implies that
%    at the observed input locations latent values have prior
%
%         f ~ N(0, Kf) and g ~ N(0,Kg)
%
%    where K is the covariance matrix, whose elements are given as 
%    K_ij = k(x_i, x_j | th). The function k(x_i, x_j | th) is covariance 
%    function and th its parameters, hyperparameters. 
%
%    Since both likelihoods and prior are Gaussian, we obtain a Gaussian 
%    marginal likelihood
%
%        p(y|th) = N(0, Kf + Kg + I*s^2).
%    
%   By placing a hyperprior for hyperparameters, p(th), we can find the 
%   maximum a posterior (MAP) estimate for them by maximizing
%
%       argmax   log p(y|th) + log p(th).
%         th
%   
%   If we want to find an approximation for the posterior of the hyperparameters, 
%   we can sample them using Markov chain Monte Carlo (MCMC) methods.
%
%   After finding MAP estimate or posterior samples of hyperparameters, we can 
%   use them to make predictions for f:
%
%       p(f | y, th) = N(m, S),
%       m = 
%       S =
%   
%   For more detailed discussion of Gaussian process regression see
%   Vanhatalo and Vehtari (2008).
%
%
%   See also  DEMO_REGRESSION1

% Copyright (c) 2008 Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.


% This file is organised in three parts:
%  1) data analysis with full GP model
%  2) data analysis with FIC approximation
%  3) data analysis with PIC approximation
%  4) data analysis with CS+FIC model

%========================================================
% PART 1 data analysis with full GP model
%========================================================

% Load the data
S = which('demo_regression2');
L = strrep(S,'demo_regression2.m','demos/maunaloa_data.txt');
data=load(L);
y = data(:, 2:13);
y=y';
y=y(:);
x = [1:1:length(y)]';
x = x(y>0);
y = y(y>0);
avgy = mean(y);
y = y-avgy;

[n,nin] = size(x);
% Now 'x' consist of the inputs and 'y' of the output. 
% 'n' and 'nin' are the number of data points and the 
% dimensionality of 'x' (the number of inputs).

% ---------------------------
% --- Construct the model ---
% 
% First create squared exponential and piecewise polynomial 2 covariance functions and 
% Gaussian noise data structures and set priors for their hyperparameters
pl = prior_t('init', 's2', 3, 'nu', 4);
pm = prior_t('init', 's2', 0.3, 'nu', 4);
gpcf1 = gpcf_sexp('init', 'lengthScale', 5, 'magnSigma2', 3, 'lengthScale_prior', pl, 'magnSigma2_prior', pm);
gpcf2 = gpcf_ppcs2('init', nin, 'lengthScale', 2, 'magnSigma2', 3, 'lengthScale_prior', pm, 'magnSigma2_prior', pm);
gpcfn = gpcf_noise('init', 'noiseSigma2', 1, 'noiseSigma2_prior', pm);

% Finally create the GP data structure
gp = gp_init('init', 'FULL', 'regr', {gpcf1, gpcf2}, {gpcfn}, 'jitterSigma2', 0.001.^2)    

% -----------------------------
% --- Conduct the inference ---
%
% --- MAP estimate -----------
w=gp_pak(gp, 'covariance');  % pack the hyperparameters into one vector
fe=str2fun('gp_e');     % create a function handle to negative log posterior
fg=str2fun('gp_g');     % create a function handle to gradient of negative log posterior

% set the options for scg2
opt = scg2_opt;
opt.tolfun = 1e-3;
opt.tolx = 1e-3;
opt.display = 1;

% do the optimization
w=scg2(fe, w, opt, fg, gp, x, y, 'covariance');

% Set the optimized hyperparameter values back to the gp structure
gp=gp_unpak(gp,w, 'covariance');

% NOTICE here that when the hyperparameters are packed into vector with 'gp_pak'
% they are also transformed through logarithm. The reason for this is that they 
% are easier to sample with MCMC after log transformation.

% Make predictions. Below Ef_full is the predictive mean and Varf_full the 
% predictive variance.
[Ef_full, Varf_full, Ey_full, Vary_full] = gp_pred(gp, x, y, x);
[Ef_full1, Varf_full1] = gp_pred(gp, x, y, x, 1);
[Ef_full2, Varf_full2] = gp_pred(gp, x, y, x, 2);

% Plot the prediction and data
figure(1)
subplot(2,1,1)
hold on
plot(x,y,'.', 'MarkerSize',7)
plot(x,Ey_full,'k', 'LineWidth', 2)
plot(x,Ey_full-2.*sqrt(Vary_full),'g--')
plot(x,Ey_full+2.*sqrt(Vary_full),'g--')
axis tight
caption1 = sprintf('Full GP:  l_1= %.2f, s^2_1 = %.2f, \n l_2= %.2f, s^2_2 = %.2f \n s^2_{noise} = %.2f', gp.cf{1}.lengthScale, gp.cf{1}.magnSigma2, gp.cf{2}.lengthScale, gp.cf{2}.magnSigma2, gp.noise{1}.noiseSigma2);
title(caption1)
legend('Data point', 'predicted mean', '2\sigma error')

subplot(2,1,2)
[AX, H1, H2] = plotyy(x, Ef_full2, x, Ef_full1);
set(H2,'LineStyle','--')
set(H2, 'LineWidth', 2)
%set(H1, 'Color', 'k')
set(H1,'LineStyle','-')
set(H1, 'LineWidth', 0.8)
title('The long and short term trend')

%========================================================
% PART 2 data analysis with FIC approximation
%========================================================

% Here we conduct the same analysis as in part 1, but this time we 
% use FIC approximation

% Place inducing inputs evenly
Xu = [min(x):24:max(x)+10]';

% Create the FIC GP data structure
gp_fic = gp_init('init', 'FIC', 'regr', {gpcf1,gpcf2}, {gpcfn}, 'jitterSigma2', 0.001.^2, 'X_u', Xu)

% -----------------------------
% --- Conduct the inference ---

% --- MAP estimate using modified Newton algorithm ---

% Now you can choose, if you want to optimize only hyperparameters or 
% optimize simultaneously hyperparameters and inducing inputs. Note that 
% the inducing inputs are not transformed through logarithm when packed

% param = 'hyper+inducing'; % optimize hyperparameters and inducing inputs
param = 'hyper';          % optimize only hyperparameters

opt=optimset('GradObj','on');
opt=optimset(opt,'TolX', 1e-4);
opt=optimset(opt,'TolFun', 1e-4);
opt=optimset(opt,'LargeScale', 'off');
opt=optimset(opt,'Display', 'iter');
param = 'hyper'

% Learn the hyperparameters
w0 = gp_pak(gp_fic, param);
mydeal = @(varargin)varargin{1:nargout};
[w,fval,exitflag] = fminunc(@(ww) mydeal(gp_e(ww, gp_fic, x, y, param), gp_g(ww, gp_fic, x, y, param)), w0, opt);
gp_fic = gp_unpak(gp_fic,w,param);

% Make the prediction
[Ef_fic, Varf_fic, Ey_fic, Vary_fic] = gp_pred(gp_fic, x, y, x);

% Plot the solution of FIC
figure(2)
%subplot(4,1,1)
hold on
plot(x,y,'.', 'MarkerSize',7)
plot(x,Ey_fic,'k', 'LineWidth', 2)
plot(x,Ey_fic-2.*sqrt(Varf_fic),'g--', 'LineWidth', 2)
plot(Xu, -30, 'rx', 'MarkerSize', 5, 'LineWidth', 2)
plot(x,Ey_fic+2.*sqrt(Varf_fic),'g--', 'LineWidth', 2)
axis tight
caption2 = sprintf('FIC:  l_1= %.2f, s^2_1 = %.2f, \n l_2= %.2f, s^2_2 = %.2f \n s^2_{noise} = %.2f', gp_fic.cf{1}.lengthScale, gp_fic.cf{1}.magnSigma2, gp_fic.cf{2}.lengthScale, gp_fic.cf{2}.magnSigma2, gp_fic.noise{1}.noiseSigma2);
title(caption2)
legend('Data point', 'predicted mean', '2\sigma error', 'inducing input')


%========================================================
% PART 3 data analysis with PIC approximation
%========================================================

% set the data points into clusters
edges = linspace(-1,max(x)+1,20);
tot=0; 
for i=1:length(edges)-1
    trindex{i} = find(x>edges(i) & x<edges(i+1));
end
% Create the FIC GP data structure
gp_pic = gp_init('init', 'PIC', 'regr', {gpcf1, gpcf2}, {gpcfn}, 'jitterSigma2', 0.001.^2, 'X_u', Xu)
gp_pic = gp_init('set', gp_pic, 'blocks', {'manual', x, trindex});

% -----------------------------
% --- Conduct the inference ---

% --- MAP estimate using modified Newton algorithm ---

% Now you can choose, if you want to optimize only hyperparameters or 
% optimize simultaneously hyperparameters and inducing inputs. Note that 
% the inducing inputs are not transformed through logarithm when packed

% param = 'hyper+inducing'; % optimize hyperparameters and inducing inputs
param = 'hyper';          % optimize only hyperparameters

opt=optimset('GradObj','on');
opt=optimset(opt,'TolX', 1e-4);
opt=optimset(opt,'TolFun', 1e-4);
opt=optimset(opt,'LargeScale', 'off');
opt=optimset(opt,'Display', 'iter');
param = 'hyper'

% Learn the hyperparameters
w0 = gp_pak(gp_pic, param);
mydeal = @(varargin)varargin{1:nargout};
[w,fval,exitflag] = fminunc(@(ww) mydeal(gp_e(ww, gp_pic, x, y, param), gp_g(ww, gp_pic, x, y, param)), w0, opt);
gp_pic = gp_unpak(gp_pic,w,param);

% Make the prediction
[Ef_pic, Varf_pic] = gp_pred(gp_pic, x, y, x, trindex);
Varf_pic = Varf_pic + gp_fic.noise{1}.noiseSigmas2;

% Plot the solution of FIC
figure(3)
%subplot(4,1,1)
hold on
plot(x,y,'.', 'MarkerSize',7)
plot(x,Ef_pic,'k', 'LineWidth', 2)
plot(x,Ef_pic-2.*sqrt(Varf_pic),'g--', 'LineWidth', 2)
plot(Xu, -30, 'rx', 'MarkerSize', 5, 'LineWidth', 2)
plot(x,Ef_pic+2.*sqrt(Varf_pic),'g--', 'LineWidth', 2)
axis tight
caption2 = sprintf('PIC:  l_1= %.2f, s^2_1 = %.2f, \n l_2= %.2f, s^2_2 = %.2f \n s^2_{noise} = %.2f', gp_pic.cf{1}.lengthScale, gp_pic.cf{1}.magnSigma2, gp_pic.cf{2}.lengthScale, gp_pic.cf{2}.magnSigma2, gp_pic.noise{1}.noiseSigmas2);
title(caption2)
legend('Data point', 'predicted mean', '2\sigma error', 'inducing input')

%========================================================
% PART 4 data analysis with CS+FIC model
%========================================================

% Here we conduct the same analysis as in part 1, but this time we 
% use FIC approximation

% Create the CS+FIC GP data structure
gp_csfic = gp_init('init', 'CS+FIC', 'regr', {gpcf1, gpcf2}, {gpcfn}, 'jitterSigma2', 0.001.^2, 'X_u', Xu)


w = gp_pak(gp_csfic);
gradcheck(w, @gp_e, @gp_g, gp_csfic, x, y);

gp_csfic = gp_init('set', gp_csfic, 'Xu_prior', prior_unif('init'));
w = gp_pak(gp_csfic);
gradcheck(w, @gp_e, @gp_g, gp_csfic, x, y);

gpcf1 = gpcf_sexp('init', 'lengthScale', [1.1], 'magnSigma2', 0.4^2, 'lengthScale_prior', []);
gp_csfic = gp_init('init', 'CS+FIC', 'regr', {gpcf1,gpcf2}, {gpcfn}, 'jitterSigma2', 0, 'X_u', Xu);
w = gp_pak(gp_csfic);
gradcheck(w, @gp_e, @gp_g, gp_csfic, x, y);

gp_csfic = gp_init('set', gp_csfic, 'Xu_prior', prior_unif('init'));
w = gp_pak(gp_csfic);
gradcheck(w, @gp_e, @gp_g, gp_csfic, x, y);


% -----------------------------
% --- Conduct the inference ---

% --- MAP estimate using modified Newton algorithm ---

% Now you can choose, if you want to optimize only hyperparameters or 
% optimize simultaneously hyperparameters and inducing inputs. Note that 
% the inducing inputs are not transformed through logarithm when packed

% param = 'hyper+inducing'; % optimize hyperparameters and inducing inputs
param = 'hyper';          % optimize only hyperparameters

opt=optimset('GradObj','on');
opt=optimset(opt,'TolX', 1e-4);
opt=optimset(opt,'TolFun', 1e-4);
opt=optimset(opt,'LargeScale', 'off');
opt=optimset(opt,'Display', 'iter');
param = 'hyper'

% Learn the hyperparameters
w0 = gp_pak(gp_csfic, param);
mydeal = @(varargin)varargin{1:nargout};
[w,fval,exitflag] = fminunc(@(ww) mydeal(gp_e(ww, gp_csfic, x, y, param), gp_g(ww, gp_csfic, x, y, param)), w0, opt);
gp_csfic = gp_unpak(gp_csfic,w,param);

% Make the prediction
[Ef_csfic, Varf_csfic] = gp_pred(gp_csfic, x, y, x);
Varf_csfic = Varf_csfic + gp_csfic.noise{1}.noiseSigmas2;

% Plot the solution of FIC
figure(4)
%subplot(4,1,1)
hold on
plot(x,y,'.', 'MarkerSize',7)
plot(x,Ef_csfic,'k', 'LineWidth', 2)
plot(x,Ef_csfic-2.*sqrt(Varf_csfic),'g--', 'LineWidth', 1)
plot(Xu, -30, 'rx', 'MarkerSize', 5, 'LineWidth', 2)
plot(x,Ef_csfic+2.*sqrt(Varf_csfic),'g--', 'LineWidth', 1)
axis tight
caption2 = sprintf('CS+FIC:  l_1= %.2f, s^2_1 = %.2f, \n l_2= %.2f, s^2_2 = %.2f \n s^2_{noise} = %.2f', gp_csfic.cf{1}.lengthScale, gp_csfic.cf{1}.magnSigma2, gp_csfic.cf{2}.lengthScale, gp_csfic.cf{2}.magnSigma2, gp_csfic.noise{1}.noiseSigmas2);
title(caption2)
legend('Data point', 'predicted mean', '2\sigma error', 'inducing input')


% Make predictions of the two components separately
gp = gp_csfic;

tx=x;
ty=y;
tn=n;
u = gp.X_u;
ncf = length(gp.cf);
cf_orig = gp.cf;

cf1 = {};
cf2 = {};
j = 1;
k = 1;
for i = 1:ncf
    if ~isfield(gp.cf{i},'cs')
        cf1{j} = gp.cf{i};
        j = j + 1;
    else
        cf2{k} = gp.cf{i};
        k = k + 1;
    end
end
gp.cf = cf1;

% First evaluate needed covariance matrices
% v defines that parameter is a vector
[Kv_ff, Cv_ff] = gp_trvar(gp, tx);  % f x 1  vector
K_fu = gp_cov(gp, tx, u);         % f x u
K_uu = gp_trcov(gp, u);    % u x u, noiseles covariance K_uu
K_uu = (K_uu+K_uu')./2;     % ensure the symmetry of K_uu
Luu = chol(K_uu)';
K_nu = gp_cov(gp, x, u);         % n x u

% Evaluate the Lambda (La)
% Q_ff = K_fu*inv(K_uu)*K_fu'
B=Luu\(K_fu');       % u x f
Qv_ff=sum(B.^2)';
Lav = Cv_ff-Qv_ff;   % f x 1, Vector of diagonal elements

gp.cf = cf2;
K_cs = gp_trcov(gp,tx);
Kcs_nf = gp_cov(gp, x, tx);
La = sparse(1:tn,1:tn,Lav,tn,tn) + K_cs;
gp.cf = cf_orig;

iLaKfu = La\K_fu;
A = K_uu+K_fu'*iLaKfu;
A = (A+A')./2;     % Ensure symmetry
L = iLaKfu/chol(A);

p = La\ty - L*(L'*ty);

%p2 = ty./Lav - iLaKfu*(A\(iLaKfu'*ty));
%    Knf = K_nu*(K_uu\K_fu');
ylong = K_nu*(K_uu\(K_fu'*p))+avgy;
yshort = Kcs_nf*p;

B2=Luu\(K_nu');
VarYlong = Kv_ff - sum(B2'.*(B*(La\B')*B2)',2)  + sum((K_nu*(K_uu\(K_fu'*L))).^2, 2);
VarYshort = diag(K_cs) - sum((Kcs_nf/chol(La)).^2,2) + sum((Kcs_nf*L).^2, 2);



figure(2)
set(gcf,'units','centimeters');
set(gcf,'DefaultAxesPosition',[0.08  0.13   0.84   0.85]);
set(gcf,'DefaultAxesFontSize',16)   %6 8
set(gcf,'DefaultTextFontSize',16)   %6 8
hold on
[AX, H1, H2] = plotyy(x, yshort, x, ylong);
set(H2,'LineStyle','--')
set(H2, 'LineWidth', 3)
%set(H1, 'Color', 'k')
set(H1,'LineStyle','-')
set(H1, 'LineWidth', 1)
%set(H2, 'Color', 'g')

set(AX(2), 'XLim', [-1 559])
set(AX(1), 'XLim', [-1 559])
set(AX(2), 'YLim', [310 380])
set(AX(1), 'YLim', [-5 5])
set(AX(2), 'XTick' ,[0 276 557])
set(AX(2), 'XTicklabel' ,[1958 1981 2004])
set(AX(1), 'XTick' ,[0 276 557])
set(AX(1), 'XTicklabel' ,[1958 1981 2004])
set(AX(2),'YTick',[310 350 380])
set(AX(2),'YTicklabel',[310 350 380])
set(AX(1),'YTick',[-5 0 5])
set(AX(1),'YTicklabel',[-5 0 5])
%set(get(AX(2),'Ylabel'),'String','ppmv')
%set(get(AX(1),'Ylabel'),'String','ppmv') 
set(get(AX(2),'Xlabel'),'String','year')
set(get(AX(1),'Xlabel'),'String','year') 

set(gcf,'pos',[10    8   18  10.7])
set(gcf,'paperunits',get(gcf,'units'))
set(gcf,'paperpos',get(gcf,'pos'))