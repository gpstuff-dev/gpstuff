%DEMO_SPATIAL2    Demonstration for a disease mapping problem with Gaussian
%                 process prior and negative binomial observation
%                 model
%
%    Description
%    The disease mapping problem consist of a data with number of
%    death cases, Y, and background population, N, appointed to
%    co-ordinates, X.  The goal is to find a relative risk surface,
%    which describes if the number of death cases in certain areas is
%    lower or higher than expected.  The data is simulated.
%
%    The model is constructed as follows:
%
%    The number of death cases Y_i in area i is assumed to satisfy
%
%         Y_i ~ Neg-Bin(Y_i| d, E_i * r_i)
%
%    where E_i is the expected number of deaths (see Vanhatalo and
%    Vehtari (2007), how E_i is evaluated) at area i, r_i is the
%    relative risk and d is the dispersion parameter coverning the
%    variance.
%
%    We place a zero mean Gaussian process prior for log(R), R = [r_1,
%    r_2,...,r_n], which implies that at the observed input locations
%    latent values, f, have prior
%
%         f = log(R) ~ N(0, K),
%
%    where K is the covariance matrix, whose elements are given as
%    K_ij = k(x_i, x_j | th). The function k(x_i, x_j | th) is
%    covariance function and th its parameters, hyperparameters.  We
%    place a hyperprior for hyperparameters, p(th).
%
%    The inference is conducted first with Laplace approximation and
%    then with EP. We use compactly supported covariance function
%    which leads to sparse covariance matrix.
%
%    See also  DEMO_REGRESSION1, DEMO_CLASSIFIC1, DEMO_SPATIAL1

% Copyright (c) 2008-2010 Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.


% =====================================
% Laplace approximation
% =====================================

% load the data
S = which('demo_spatial2');
data = load(strrep(S,'demo_spatial2.m','demos/spatial2.txt'));


xx = data(:,1:2);
ye = data(:,3);
yy = data(:,4);

% Now we have loaded the following parameters
% xx = co-ordinates 
% yy = number of deaths
% ye = the expexted number of deaths

% Create the covariance functions
gpcf1 = gpcf_ppcs2('init', 'nin', 2, 'lengthScale', 5, 'magnSigma2', 0.05);
pl = prior_t('init');
pm = prior_t('init', 's2', 0.3);
gpcf1 = gpcf_ppcs2('set', gpcf1, 'lengthScale_prior', pl, 'magnSigma2_prior', pm);

% Create the likelihood structure
likelih = likelih_negbin('init');

% Create the GP data structure
gp = gp_init('init', 'FULL', likelih, {gpcf1}, [], 'jitterSigma2', 0.001); 

% Set the approximate inference method to EP
%gp = gp_init('set', gp, 'latent_method', {'EP', xx, yy, 'z', ye});
gp = gp_init('set', gp, 'latent_method', {'Laplace', xx, yy, 'z', ye});

w=gp_pak(gp);      % pack the hyperparameters into one vector
fe=str2fun('gpla_e');     % create a function handle to negative log posterior
fg=str2fun('gpla_g');     % create a function handle to gradient of negative log posterior

% set the options for scg2
opt = scg2_opt;
opt.tolfun = 1e-2;
opt.tolx = 1e-2;
opt.display = 1;

% do the optimization and set the optimized hyperparameter values back to the gp structure
w=scg2(fe, w, opt, fg, gp, xx, yy, 'z', ye);
gp = gp_unpak(gp,w);

C = gp_trcov(gp,xx);
nnz(C) / prod(size(C))
p = amd(C);
figure
spy(C(p,p))

% make prediction to the data points
[Ef, Varf] = la_pred(gp, xx, yy, xx, 'z', ye);

% Define help parameters for plotting
xxii=sub2ind([120 70],xx(:,2),xx(:,1));
[X1,X2]=meshgrid(1:70,1:120);

% Plot the figures
figure
G=repmat(NaN,size(X1));
G(xxii)=exp(Ef);
pcolor(X1,X2,G),shading flat
colormap(mapcolor(G)),colorbar
set(gca, 'Clim', [0.2    1.5])
axis equal
axis([0 70 0 120])
title('Posterior median of the relative risk (Laplace)')

figure
G=repmat(NaN,size(X1));
G(xxii)=(exp(Varf) - 1).*exp(2*Ef+Varf);
pcolor(X1,X2,G),shading flat
colormap(mapcolor(G)),colorbar
%set(gca, 'Clim', [0.005    0.03])
axis equal
axis([0 70 0 120])
title('Posterior variance of the relative risk (Laplace)')


% =====================================
% EP approximation
% =====================================

% load the data
S = which('demo_spatial2');
data = load(strrep(S,'demo_spatial2.m','demos/spatial2.txt'));

xx = data(:,1:2);
ye = data(:,3);
yy = data(:,4);

% Now we have loaded the following parameters
% xx = co-ordinates 
% yy = number of deaths
% ye = the expexted number of deaths

% Create the covariance functions
gpcf1 = gpcf_ppcs2('init', 'nin', 2, 'lengthScale', 5, 'magnSigma2', 0.05);
pl = prior_t('init');
pm = prior_t('init', 's2', 0.3);
gpcf1 = gpcf_ppcs2('set', gpcf1, 'lengthScale_prior', pl, 'magnSigma2_prior', pm);

% Create the likelihood structure
likelih = likelih_negbin('init');

% Create the GP data structure
gp = gp_init('init', 'FULL', likelih, {gpcf1}, [], 'jitterSigma2', 0.001, 'infer_params', 'covariance'); 

% Set the approximate inference method to EP
gp = gp_init('set', gp, 'latent_method', {'EP', xx, yy, 'z', ye});

w=gp_pak(gp);      % pack the hyperparameters into one vector
fe=str2fun('gpep_e');     % create a function handle to negative log posterior
fg=str2fun('gpep_g');     % create a function handle to gradient of negative log posterior

% set the options for scg2
opt = scg2_opt;
opt.tolfun = 1e-2;
opt.tolx = 1e-2;
opt.display = 1;

% do the optimization and set the optimized hyperparameter values back to the gp structure
w=scg2(fe, w, opt, fg, gp, xx, yy, 'z', ye);
gp = gp_unpak(gp,w);

C = gp_trcov(gp,xx);
nnz(C) / prod(size(C))
p = amd(C);
figure
spy(C(p,p))

% make prediction to the data points
[Ef, Varf] = ep_pred(gp, xx, yy, xx, 'z', ye);

% Define help parameters for plotting
xxii=sub2ind([120 70],xx(:,2),xx(:,1));
[X1,X2]=meshgrid(1:70,1:120);

% Plot the figures
figure
G=repmat(NaN,size(X1));
G(xxii)=exp(Ef);
pcolor(X1,X2,G),shading flat
colormap(mapcolor(G)),colorbar
set(gca, 'Clim', [0.2    1.5])
axis equal
axis([0 70 0 120])
title('Posterior median of the relative risk (Laplace)')

figure
G=repmat(NaN,size(X1));
G(xxii)=(exp(Varf) - 1).*exp(2*Ef+Varf);
pcolor(X1,X2,G),shading flat
colormap(mapcolor(G)),colorbar
%set(gca, 'Clim', [0.005    0.03])
axis equal
axis([0 70 0 120])
title('Posterior variance of the relative risk (Laplace)')
