function demo_classific3
%DEMO_CLAASIFIC3    Classification problem demonstration for 2 classes via Laplace 
%                   approximation and EP
%
%      Description
%      The demonstration program is based on synthetic two 
%      class data used by B.D. Ripley (Pattern Regocnition and
%      Neural Networks, 1996}. The data consists of 2-dimensional
%      vectors that are divided into to classes, labeled -1 or 1.
%      Each class has a bimodal distribution generated from equal
%      mixtures of Gaussian distributions with identical covariance
%      matrices. A Bayesian aproach is used to find the decision
%      line and predict the classes of new data points.
%
%      The probability of y being one is assumed to be 
%
%            p(y=1|f) = int_{-inf}^{yf} N(x|0,1) dx
%
%      (Compare this to logistic likelihood in demo_classific1 and see 
%      Rasmussen and Williams (2006) for details). 
%
%      The latent values f are given a zero mean Gaussian process prior. 
%      This implies that at the observed input locations latent values
%      have prior 
%
%         f ~ N(0, K),
%
%      where K is the covariance matrix, whose elements are given as 
%      K_ij = k(x_i, x_j | th). The function k(x_i, x_j | th) is covariance 
%      function and th its parameters, hyperparameters. 
% 
%      Here we Laplace approximation and expectation propagation to find normal 
%      approximation for the posterior of the latent values and optimize the
%      hyperparameters in their MAP point. See Rasmussen and Williams (2006) for 
%      details.
%
%      NOTE! The class labels have to be {-1,1} for probit likelihood 
%      (different from the logit likelihood).

% Copyright (c) 2008 Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

% This demonstration is based on the dataset used in the book Pattern Recognition and
% Neural Networks by B.D. Ripley (1996) Cambridge University Press ISBN 0 521
% 46986 7

%==================================================================
% PART 1 data analysis with full GP model and EP
%==================================================================

S = which('demo_classific1');
L = strrep(S,'demo_classific1.m','demos/synth.tr');
x=load(L);
y=x(:,end);
y=y*2-1;
x(:,end)=[];
[n, nin] = size(x);

% Create covariance functions
gpcf1 = gpcf_sexp('init', 'lengthScale', [1 1], 'magnSigma2', 1);

% Set the prior for the parameters of covariance functions 
gpcf1.p.lengthScale = gamma_p({3 7});
gpcf1.p.magnSigma2 = sinvchi2_p({0.05^2 0.5});

% Create the likelihood structure
likelih = likelih_probit('init', y);

% Create the GP data structure
gp = gp_init('init', 'FULL', likelih, {gpcf1}, [], 'jitterSigma2', 0.0001.^2);

% Set the approximate inference method
gp.ep_opt.display = 1;
gp = gp_init('set', gp, 'latent_method', {'EP', x, y, 'hyper'});

fe=str2fun('gpep_e');
fg=str2fun('gpep_g');
n=length(y);
opt(1) = 1;
opt(2) = 1e-2;
opt(3) = 3e-1;
opt(9) = 0;
opt(10) = 0;
opt(11) = 0;
opt(14) = 0;

% do scaled conjugate gradient optimization 
w=gp_pak(gp, 'hyper');
[w, opt, flog]=scg(fe, w, opt, fg, gp, x, y, 'hyper');
gp=gp_unpak(gp,w, 'hyper');

% Print some figures that show results
% First create data for predictions
xt1=repmat(linspace(min(x(:,1)),max(x(:,1)),20)',1,20);
xt2=repmat(linspace(min(x(:,2)),max(x(:,2)),20)',1,20)';
xstar=[xt1(:) xt2(:)];

% make the prediction
[Ef, Varf, p1] = ep_pred(gp, x, y, xstar, 'hyper');

figure, hold on;
n_pred=size(xstar,1);
h1=pcolor(reshape(xstar(:,1),20,20),reshape(xstar(:,2),20,20),reshape(p1,20,20))
set(h1, 'edgealpha', 0), set(h1, 'facecolor', 'interp')
colormap(repmat(linspace(1,0,64)', 1, 3).*repmat(ones(1,3), 64,1))
axis([-inf inf -inf inf]), axis off
plot(x(y==-1,1),x(y==-1,2),'o', 'markersize', 8, 'linewidth', 2);
plot(x(y==1,1),x(y==1,2),'rx', 'markersize', 8, 'linewidth', 2);
set(gcf, 'color', 'w'), title('predictive probability and training cases, full GP with Laplace', 'fontsize', 14)

% visualise predictive probability  p(ystar = 1) with contours
figure, hold on
[cs,h]=contour(reshape(xstar(:,1),20,20),reshape(xstar(:,2),20,20),reshape(p1,20,20),[0.025 0.25 0.5 0.75 0.975], 'linewidth', 3);
text_handle = clabel(cs,h);
set(text_handle,'BackgroundColor',[1 1 .6],'Edgecolor',[.7 .7 .7],'linewidth', 2, 'fontsize',14)
c1=[linspace(0,1,64)' 0*ones(64,1) linspace(1,0,64)'];
colormap(c1)
plot(x(y==1,1), x(y==1,2), 'rx', 'markersize', 8, 'linewidth', 2),
plot(x(y==-1,1), x(y==-1,2), 'bo', 'markersize', 8, 'linewidth', 2)
plot(xstar(:,1), xstar(:,2), 'k.'), axis([-inf inf -inf inf]), axis off
set(gcf, 'color', 'w'), title('predictive probability contours, full GP with Laplace', 'fontsize', 14)


%========================================================
% PART 2 data analysis with SSGP and EP
%========================================================

% Create the squared exponential sparse spectral covariance function
gpcf3 = gpcf_SSsexp('init', 'lengthScale', [1 1], 'magnSigma2', 1);
gpcf3 = gpcf_SSsexp('set', gpcf3, 'nfreq', 200);

% ... Then set the prior for the parameters of covariance functions...
gpcf3.p.lengthScale = gamma_p({3 7});  
gpcf3.p.magnSigma2 = sinvchi2_p({0.05^2 0.5});

% Create the likelihood structure
likelih = likelih_probit('init', y);

% ... Finally create the GP data structure
gp2 = gp_init('init', 'SSGP', likelih, {gpcf3}, [])

% Set the approximate inference method
gp2.ep_opt.display = 1;
gp2 = gp_init('set', gp2, 'latent_method', {'EP', x, y, 'hyper'});

% do scaled conjugate gradient optimization 
w2 = gp_pak(gp2, 'hyper');
[w2, opt, flog]=scg(fe, w2, opt, fg, gp2, x, y, 'hyper');
gp2=gp_unpak(gp2, w2 , 'hyper');


% Print some figures that show results
% First create data for predictions
xt1=repmat(linspace(min(x(:,1)),max(x(:,1)),20)',1,20);
xt2=repmat(linspace(min(x(:,2)),max(x(:,2)),20)',1,20)';
xstar=[xt1(:) xt2(:)];

% make the prediction
[Ef_2, Varf_2, p1_2] = ep_pred(gp2, x, y, xstar, 'hyper');

figure, hold on;
n_pred=size(xstar,1);
h1=pcolor(reshape(xstar(:,1),20,20),reshape(xstar(:,2),20,20),reshape(p1_2,20,20))
set(h1, 'edgealpha', 0), set(h1, 'facecolor', 'interp')
colormap(repmat(linspace(1,0,64)', 1, 3).*repmat(ones(1,3), 64,1))
axis([-inf inf -inf inf]), axis off
plot(x(y==-1,1),x(y==-1,2),'o', 'markersize', 8, 'linewidth', 2);
plot(x(y==1,1),x(y==1,2),'rx', 'markersize', 8, 'linewidth', 2);
set(gcf, 'color', 'w'), title('predictive probability and training cases, full GP with Laplace', 'fontsize', 14)

% visualise predictive probability  p(ystar = 1) with contours
figure, hold on
[cs,h]=contour(reshape(xstar(:,1),20,20),reshape(xstar(:,2),20,20),reshape(p1_2,20,20),[0.025 0.25 0.5 0.75 0.975], 'linewidth', 3);
text_handle = clabel(cs,h);
set(text_handle,'BackgroundColor',[1 1 .6],'Edgecolor',[.7 .7 .7],'linewidth', 2, 'fontsize',14)
c1=[linspace(0,1,64)' 0*ones(64,1) linspace(1,0,64)'];
colormap(c1)
plot(x(y==1,1), x(y==1,2), 'rx', 'markersize', 8, 'linewidth', 2),
plot(x(y==-1,1), x(y==-1,2), 'bo', 'markersize', 8, 'linewidth', 2)
plot(xstar(:,1), xstar(:,2), 'k.'), axis([-inf inf -inf inf]), axis off
set(gcf, 'color', 'w'), title('predictive probability contours, full GP with Laplace', 'fontsize', 14)




































%==================================================================
% PART 3 data analysis with full GP model and expectation propagation
%==================================================================

S = which('demo_classific1');
L = strrep(S,'demo_classific1.m','demos/synth.tr');
x=load(L);
y=x(:,end);
y=y*2-1;
x(:,end)=[];
[n, nin] = size(x);

% Create covariance functions
gpcf1 = gpcf_sexp('init', 'lengthScale', [0.9 0.9], 'magnSigma2', 1);

% Set the prior for the parameters of covariance functions 
gpcf1.p.lengthScale = gamma_p({3 7 3 7});
gpcf1.p.magnSigma2 = sinvchi2_p({0.05^2 0.5});

% Create the likelihood structure
likelih = likelih_probit('init', y);

% Create the GP data structure
gp = gp_init('init', 'FULL', likelih, {gpcf1}, [], 'jitterSigma2', 0.01.^2);

% Set the approximate inference method
gp = gp_init('set', gp, 'latent_method', {'EP', x, y, 'hyper'});

fe=str2fun('gpep_e');
fg=str2fun('gpep_g');
n=length(y);
opt(1) = 1;
opt(2) = 1e-2;
opt(3) = 3e-1;
opt(9) = 0;
opt(10) = 0;
opt(11) = 0;
opt(14) = 0;

% do scaled conjugate gradient optimization 
gp.ep_opt.display = 1;
w=gp_pak(gp, 'hyper');
[w, opt, flog]=scg(fe, w, opt, fg, gp, x, y, 'hyper');
gp=gp_unpak(gp,w, 'hyper');

% Print some figures that show results
% First create data for predictions
xt1=repmat(linspace(min(x(:,1)),max(x(:,1)),20)',1,20);
xt2=repmat(linspace(min(x(:,2)),max(x(:,2)),20)',1,20)';
xstar=[xt1(:) xt2(:)];

% make the prediction
[Ef, Varf, p1] = ep_pred(gp, x, y, xstar, 'hyper');

figure, hold on;
n_pred=size(xstar,1);
h1=pcolor(reshape(xstar(:,1),20,20),reshape(xstar(:,2),20,20),reshape(p1,20,20))
set(h1, 'edgealpha', 0), set(h1, 'facecolor', 'interp')
colormap(repmat(linspace(1,0,64)', 1, 3).*repmat(ones(1,3), 64,1))
axis([-inf inf -inf inf]), axis off
plot(x(y==-1,1),x(y==-1,2),'o', 'markersize', 8, 'linewidth', 2);
plot(x(y==1,1),x(y==1,2),'rx', 'markersize', 8, 'linewidth', 2);
set(gcf, 'color', 'w'), title('predictive probability and training cases, full GP with EP', 'fontsize', 14)

% visualise predictive probability  p(ystar = 1) with contours
figure, hold on
[cs,h]=contour(reshape(xstar(:,1),20,20),reshape(xstar(:,2),20,20),reshape(p1,20,20),[0.025 0.25 0.5 0.75 0.975], 'linewidth', 3);
text_handle = clabel(cs,h);
set(text_handle,'BackgroundColor',[1 1 .6],'Edgecolor',[.7 .7 .7],'linewidth', 2, 'fontsize',14)
c1=[linspace(0,1,64)' 0*ones(64,1) linspace(1,0,64)'];
colormap(c1)
plot(x(y==1,1), x(y==1,2), 'rx', 'markersize', 8, 'linewidth', 2),
plot(x(y==-1,1), x(y==-1,2), 'bo', 'markersize', 8, 'linewidth', 2)
plot(xstar(:,1), xstar(:,2), 'k.'), axis([-inf inf -inf inf]), axis off
set(gcf, 'color', 'w'), title('predictive probability contours, full GP with EP', 'fontsize', 14)


%========================================================
% PART 4 data analysis with FIC GP model and expectation propagation
%========================================================

% Set the inducing inputs
[u1,u2]=meshgrid(linspace(-1.25, 0.9,6),linspace(-0.2, 1.1,6));
Xu=[u1(:) u2(:)];
Xu = Xu([3 4 7:18 20:24 26:30 33:36],:);

% Create the GP data structure
gp_fic = gp_init('init', 'FIC', likelih, {gpcf1}, [], 'jitterSigma2', 0.01.^2, 'X_u', Xu);

% Set the approximate inference method
gp_fic = gp_init('set', gp_fic, 'latent_method', {'EP', x, y, 'hyper'});

fe=str2fun('gpep_e');
fg=str2fun('gpep_g');
n=length(y);
opt(1) = 1;
opt(2) = 1e-2;
opt(3) = 3e-1;
opt(9) = 0;
opt(10) = 0;
opt(11) = 0;
opt(14) = 0;

% do scaled conjugate gradient optimization 
gp_fic.ep_opt.display = 1;
w=gp_pak(gp_fic, 'hyper');
[w, opt, flog]=scg(fe, w, opt, fg, gp_fic, x, y, 'hyper');
gp_fic=gp_unpak(gp_fic,w, 'hyper');

% Print some figures that show results
% First create data for predictions
xt1=repmat(linspace(min(x(:,1)),max(x(:,1)),20)',1,20);
xt2=repmat(linspace(min(x(:,2)),max(x(:,2)),20)',1,20)';
xstar=[xt1(:) xt2(:)];

% make the prediction
[Ef_fic, Varf_fic, p1_fic] = ep_pred(gp_fic, x, y, xstar, 'hyper');

figure, hold on;
n_pred=size(xstar,1);
h1=pcolor(reshape(xstar(:,1),20,20),reshape(xstar(:,2),20,20),reshape(p1_fic,20,20))
set(h1, 'edgealpha', 0), set(h1, 'facecolor', 'interp')
colormap(repmat(linspace(1,0,64)', 1, 3).*repmat(ones(1,3), 64,1))
axis([-inf inf -inf inf]), axis off
plot(x(y==-1,1),x(y==-1,2),'o', 'markersize', 8, 'linewidth', 2);
plot(x(y==1,1),x(y==1,2),'rx', 'markersize', 8, 'linewidth', 2);
set(gcf, 'color', 'w'), title('predictive probability and training cases, FIC with EP', 'fontsize', 14)

% visualise predictive probability  p(ystar = 1) with contours
figure, hold on
[cs,h]=contour(reshape(xstar(:,1),20,20),reshape(xstar(:,2),20,20),reshape(p1_fic,20,20),[0.025 0.25 0.5 0.75 0.975], 'linewidth', 3);
text_handle = clabel(cs,h);
set(text_handle,'BackgroundColor',[1 1 .6],'Edgecolor',[.7 .7 .7],'linewidth', 2, 'fontsize',14)
c1=[linspace(0,1,64)' 0*ones(64,1) linspace(1,0,64)'];
colormap(c1)
plot(x(y==1,1), x(y==1,2), 'rx', 'markersize', 8, 'linewidth', 2),
plot(x(y==-1,1), x(y==-1,2), 'bo', 'markersize', 8, 'linewidth', 2)
plot(xstar(:,1), xstar(:,2), 'k.'), axis([-inf inf -inf inf]), axis off
set(gcf, 'color', 'w'), title('predictive probability contours, FIC with EP', 'fontsize', 14)


