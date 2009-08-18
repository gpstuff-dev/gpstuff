%DEMO_SPATIAL2    Demonstration for a disease mapping problem
%                 with Gaussian process prior and inference via Laplace and EP
%
%    Description
%    The disease mapping problem consist of a data with number of death 
%    cases, Y, and background population, N, appointed to co-ordinates, X.
%    The goal is to find a relative risk surface, which describes if the 
%    number of death cases in certain areas is lower or higher than expected.
%    The data consists of the heart attacks in Finland from 1996-2000 aggregated 
%    into 20kmx20km lattice cells.
%
%    The model constructed is as follows:
%
%    The number of death cases Y_i in area i is assumed to satisfy
%
%         Y_i ~ Poisson(Y_i| E_i * r_i)
%
%    where E_i is the expected number of deaths (see Vanhatalo and Vehtari (2007), 
%    how E_i is evaluated) at area i and r_i is the relative risk.
%
%    We place a zero mean Gaussian process prior for log(R), R = [r_1, r_2,...,r_n], 
%    which implies that at the observed input locations latent values, f, have prior
%
%         f = log(R) ~ N(0, K),
%
%    where K is the covariance matrix, whose elements are given as 
%    K_ij = k(x_i, x_j | th). The function k(x_i, x_j | th) is covariance 
%    function and th its parameters, hyperparameters.  We place a hyperprior for
%    hyperparameters, p(th).
%
%    The inference is conducted via EP, where we find Gaussian approximation for
%    p(f| th, data), where th is the maximum a posterior (MAP) estimate for the 
%    hyper-parameters. 
%
%    See Vanhatalo and Vehtari (2008) (to be published) for more detailed 
%    discussion.
%
%    See also  DEMO_REGRESSION1, DEMO_CLASSIFIC1, DEMO_SPATIAL1

% Copyright (c) 2008 Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.


% This file is organised in three parts:
%  1) data analysis with full GP model
%  2) data analysis with FIC approximation
%  3) data analysis with PIC approximation


% ===================================================
% ===================================================
% Laplace APPROACH
% ===================================================
% ===================================================



% =====================================
% 1) FULL model
% =====================================

S = which('demo_spatial3');
L = strrep(S,'demo_spatial3.m','demos/spatial.mat');
load(L)

% Now we have loaded the following parameters
% xx = co-ordinates 
% yy = number of deaths
% ye = the expexted number of deaths

% reduce the data in order to make the demo faster
ind = find(xx(:,2)<25);
xx = xx(ind,:);
yy = yy(ind,:);
ye = ye(ind,:);


[n,nin] = size(xx);

% Create the covariance functions
gpcf1 = gpcf_matern32('init', nin, 'lengthScale', 5, 'magnSigma2', 0.05);
gpcf1.p.lengthScale = t_p({1 4});
gpcf1.p.magnSigma2 = t_p({0.3 4});

% Create the likelihood structure
likelih = likelih_poisson('init', yy, ye);

% Create the GP data structure
gp = gp_init('init', 'FULL', nin, likelih, {gpcf1}, []);   %{gpcf2}

% Set the approximate inference method
gp = gp_init('set', gp, 'latent_method', {'Laplace', xx, yy, 'hyper'});

param = 'hyper';
opt=optimset('GradObj','on');
opt=optimset(opt,'TolX', 1e-3);
opt=optimset(opt,'LargeScale', 'off');
opt=optimset(opt,'Display', 'iter');

w0 = gp_pak(gp, param);
mydeal = @(varargin)varargin{1:nargout};
w = fminunc(@(ww) mydeal(gpla_e(ww, gp, xx, yy, param), gpla_g(ww, gp, xx, yy, param)), w0, opt);
gp = gp_unpak(gp,w,param);

% make prediction to the data points
[Ef, Varf] = la_pred(gp, xx, yy, xx, param);

% Define help parameters for plotting
xxii=sub2ind([60 35],xx(:,2),xx(:,1));
[X1,X2]=meshgrid(1:35,1:60);

% plot figures
figure(1)
G=repmat(NaN,size(X1));
G(xxii)=exp(Ef + Varf/2);
pcolor(X1,X2,G),shading flat
colormap(mapcolor(G)),colorbar
set(gca, 'Clim', [0.6    1.5])
axis equal
axis([0 35 0 60])
title('Posterior mean of the relative risk, full GP')

figure(2)
G=repmat(NaN,size(X1));
G(xxii)=(exp(Varf) - 1).*exp(2*Ef+Varf);
pcolor(X1,X2,G),shading flat
colormap(mapcolor(G)),colorbar
set(gca, 'Clim', [0.005    0.03])
axis equal
axis([0 35 0 60])
title('Posterior variance of the relative risk, full GP')

% the MAP estimate of the hyperparameters in kilometers. Notice that the 
% co-ordinates in the data are not in kilometers. x=1 corresponds to 20km 
% in real life
S1 = sprintf('lengt-scale: %.3f, magnSigma2: %.3f \n', gp.cf{1}.lengthScale, gp.cf{1}.magnSigma2)

% =====================================
% 2) FIC model
% =====================================

% load the data
S = which('demo_spatial2');
L = strrep(S,'demo_spatial2.m','demos/spatial.mat');
load(L)

% Now we have loaded the following parameters
% xx = co-ordinates 
% yy = number of deaths
% ye = the expexted number of deaths

% Set the inducing inputs in a regular grid.
% Set_PIC returns the induving inputs and blockindeces for PIC. It also plots the 
% data points, inducing inputs and blocks.
dims = [1    60     1    35];
[trindex, Xu] = set_PIC(xx, dims, 5, 'corners+1xside', 1);

[n,nin] = size(xx);

% Create the covariance functions
gpcf1 = gpcf_matern32('init', nin, 'lengthScale', 5, 'magnSigma2', 0.05);
gpcf1.p.lengthScale = unif_p; %t_p({1 4});
gpcf1.p.magnSigma2 = unif_p; %t_p({0.3 4});

% Create the likelihood structure
likelih = likelih_poisson('init', yy, ye);

% Create the FIC GP data structure
gp = gp_init('init', 'FIC', nin, likelih, {gpcf1}, [], 'X_u', Xu); %, 'jitterSigmas', 0.01

% Set the approximate inference method to EP
gp = gp_init('set', gp, 'latent_method', {'Laplace', xx, yy, 'hyper'});

% Set the optimization parameters
param = 'hyper';
opt=optimset('GradObj','on');
opt=optimset(opt,'TolX', 1e-3);
opt=optimset(opt,'LargeScale', 'off');
opt=optimset(opt,'Display', 'iter');

% conduct the hyper-parameter optimization
w0 = gp_pak(gp, param);
mydeal = @(varargin)varargin{1:nargout};
w = fminunc(@(ww) mydeal(gpla_e(ww, gp, xx, yy, param), gpla_g(ww, gp, xx, yy, param)), w0, opt);
gp = gp_unpak(gp,w,param);

% make prediction to the data points
[Ef, Varf] = la_pred(gp, xx, yy, xx, param);

% Define help parameters for plotting
xxii=sub2ind([60 35],xx(:,2),xx(:,1));
[X1,X2]=meshgrid(1:35,1:60);

% Plot the figures
% In the results it should be noticed that:
% - there is much more people living in the south than in the north. 
%   This results in rather high variance in the north
% - The eastern Finland is known to be worse than western Finland in 
%   heart diseases also from other studies.
% - The inducing inputs are set slightly too sparsely for this data, 
%   which results in oversmoothness in the maps
figure(3)
G=repmat(NaN,size(X1));
G(xxii)=exp(Ef);
pcolor(X1,X2,G),shading flat
colormap(mapcolor(G)),colorbar
set(gca, 'Clim', [0.6    1.5])
axis equal
axis([0 35 0 60])
title('Posterior median of the relative risk, FIC')

figure(4)
G=repmat(NaN,size(X1));
G(xxii)=(exp(Varf) - 1).*exp(2*Ef+Varf);
pcolor(X1,X2,G),shading flat
colormap(mapcolor(G)),colorbar
set(gca, 'Clim', [0.005    0.03])
axis equal
axis([0 35 0 60])
title('Posterior variance of the relative risk, FIC')

% the MAP estimate of the hyperparameters in kilometers. Notice that the 
% co-ordinates in the data are not in kilometers. x=1 corresponds to 20km 
% in real life
S2 = sprintf('lengt-scale: %.3f, magnSigma2: %.3f \n', gp.cf{1}.lengthScale, gp.cf{1}.magnSigma2)

% =====================================
% 3) PIC model
% =====================================

% load the data
S = which('demo_spatial3');
L = strrep(S,'demo_spatial3.m','demos/spatial.mat');
load(L)

% Now we have loaded the following parameters
% xx = co-ordinates 
% yy = number of deaths
% ye = the expexted number of deaths

% Set the inducing inputs in a regular grid.
% Set_PIC returns the induving inputs and blockindeces for PIC. It also plots the 
% data points, inducing inputs and blocks.
dims = [1    60     1    35];
[trindex, Xu] = set_PIC(xx, dims, 5, 'corners+1xside', 1);

[n,nin] = size(xx);

% Create the covariance functions
gpcf1 = gpcf_matern32('init', nin, 'lengthScale', 5, 'magnSigma2', 0.05);
gpcf1.p.lengthScale = t_p({1 4});
gpcf1.p.magnSigma2 = t_p({0.3 4});

% Create the likelihood structure
likelih = likelih_poisson('init', yy, ye);

% Create the PIC GP data structure
gp = gp_init('init', 'PIC', nin, likelih, {gpcf1}, [], 'jitterSigmas', 0.01, 'X_u', Xu);
gp = gp_init('set', gp, 'blocks', {'manual', xx, trindex});

% Set the approximate inference method to EP
gp = gp_init('set', gp, 'latent_method', {'Laplace', xx, yy, 'hyper'});

% Set the optimization parameters
param = 'hyper';
opt=optimset('GradObj','on');
opt=optimset(opt,'TolX', 1e-3);
opt=optimset(opt,'LargeScale', 'off');
opt=optimset(opt,'Display', 'iter');

% conduct the hyper-parameter optimization
w0 = gp_pak(gp, param);
mydeal = @(varargin)varargin{1:nargout};
w = fminunc(@(ww) mydeal(gpla_e(ww, gp, xx, yy, param), gpla_g(ww, gp, xx, yy, param)), w0, opt);
gp = gp_unpak(gp,w,param);

% make prediction to the data points
[Ef, Varf] = la_pred(gp, xx, yy, xx, param, trindex);

% Define help parameters for plotting
xxii=sub2ind([60 35],xx(:,2),xx(:,1));
[X1,X2]=meshgrid(1:35,1:60);

% Plot the figures
% In the results it should be noticed that:
% - there is much more people living in the south than in the north. 
%   This results in rather high variance in the north
% - The eastern Finland is known to be worse than western Finland in 
%   heart diseases also from other studies.
% - PIC has fixed the oversmoothness present in FIC
figure(6)
G=repmat(NaN,size(X1));
G(xxii)=exp(Ef);
pcolor(X1,X2,G),shading flat
colormap(mapcolor(G)),colorbar
set(gca, 'Clim', [0.6    1.5])
axis equal
axis([0 35 0 60])
title('Posterior median of the relative risk, PIC')

figure(7)
G=repmat(NaN,size(X1));
G(xxii)=(exp(Varf) - 1).*exp(2*Ef+Varf);
pcolor(X1,X2,G),shading flat
colormap(mapcolor(G)),colorbar
set(gca, 'Clim', [0.005    0.03])
axis equal
axis([0 35 0 60])
title('Posterior variance of the relative risk, PIC')

% the MAP estimate of the hyperparameters in kilometers. Notice that the 
% co-ordinates in the data are not in kilometers. x=1 corresponds to 20km 
% in real life
S3 = sprintf('lengt-scale: %.3f, magnSigma2: %.3f \n', gp.cf{1}.lengthScale, gp.cf{1}.magnSigma2)

% =====================================
% 4) CS+FIC model
% =====================================

% load the data
S = which('demo_spatial3');
L = strrep(S,'demo_spatial3.m','demos/spatial.mat');
load(L)

% Now we have loaded the following parameters
% xx = co-ordinates 
% yy = number of deaths
% ye = the expexted number of deaths

% Set the inducing inputs in a regular grid.
% Set_PIC returns the induving inputs and blockindeces for PIC. It also plots the 
% data points, inducing inputs and blocks.
dims = [1    60     1    35];
[trindex, Xu] = set_PIC(xx, dims, 5, 'corners', 0);

[n,nin] = size(xx);

% Create the covariance functions
gpcf1 = gpcf_matern32('init', nin, 'lengthScale', 4, 'magnSigma2', 0.05);
gpcf1.p.lengthScale = t_p({1 4});
gpcf1.p.magnSigma2 = t_p({0.3 4});

gpcf2 = gpcf_ppcs2('init', nin, 'lengthScale', 3, 'magnSigma2', 0.03);
gpcf2.p.lengthScale = t_p({1 4});
gpcf2.p.magnSigma2 = t_p({0.3 4});

% Create the likelihood structure
likelih = likelih_poisson('init', yy, ye);

% Create the FIC GP data structure
gp = gp_init('init', 'CS+FIC', nin, likelih, {gpcf1, gpcf2}, [], 'X_u', Xu); %, 'jitterSigmas', 0.01 

% Set the approximate inference method to EP
gp = gp_init('set', gp, 'latent_method', {'Laplace', xx, yy, 'hyper'});

% Set the optimization parameters
param = 'hyper';
opt=optimset('GradObj','on');
opt=optimset(opt,'TolX', 1e-3);
opt=optimset(opt,'LargeScale', 'off');
opt=optimset(opt,'Display', 'iter');

gradcheck(gp_pak(gp, param), @gpla_e, @gpla_g, gp, xx, yy, param)

% conduct the hyper-parameter optimization
w0 = gp_pak(gp, param);
mydeal = @(varargin)varargin{1:nargout};
w = fminunc(@(ww) mydeal(gpla_e(ww, gp, xx, yy, param), gpla_g(ww, gp, xx, yy, param)), w0, opt);
gp = gp_unpak(gp,w,param);

% make prediction to the data points
[Ef, Varf] = la_pred(gp, xx, yy, xx, param);

% Define help parameters for plotting
xxii=sub2ind([60 35],xx(:,2),xx(:,1));
[X1,X2]=meshgrid(1:35,1:60);

% Plot the figures
% In the results it should be noticed that:
% - there is much more people living in the south than in the north. 
%   This results in rather high variance in the north
% - The eastern Finland is known to be worse than western Finland in 
%   heart diseases also from other studies.
figure(8)
G=repmat(NaN,size(X1));
G(xxii)=exp(Ef);
pcolor(X1,X2,G),shading flat
colormap(mapcolor(G)),colorbar
set(gca, 'Clim', [0.6    1.5])
axis equal
axis([0 35 0 60])
title('Posterior median of the relative risk, CS+FIC')

figure(9)
G=repmat(NaN,size(X1));
G(xxii)=(exp(Varf) - 1).*exp(2*Ef+Varf);
pcolor(X1,X2,G),shading flat
colormap(mapcolor(G)),colorbar
set(gca, 'Clim', [0.005    0.03])
axis equal
axis([0 35 0 60])
title('Posterior variance of the relative risk, CS+FIC')

% the MAP estimate of the hyperparameters in kilometers. Notice that the 
% co-ordinates in the data are not in kilometers. x=1 corresponds to 20km 
% in real life
S4 = sprintf('lengt-scale (matern32): %.3f, magnSigma2 (matern32): %.3f \n lengt-scale (ppcs2): %.3f, magnSigma2 (ppcs2): %.3f \n', gp.cf{1}.lengthScale, gp.cf{1}.magnSigma2, gp.cf{2}.lengthScale, gp.cf{2}.magnSigma2)




% ===================================================
% ===================================================
% EP APPROACH
% ===================================================
% ===================================================





% =====================================
% 1) FULL model
% =====================================

S = which('demo_spatial2');
L = strrep(S,'demo_spatial2.m','demos/spatial.mat');
load(L)

% Now we have loaded the following parameters
% xx = co-ordinates 
% yy = number of deaths
% ye = the expexted number of deaths

% reduce the data in order to make the demo faster
ind = find(xx(:,2)<25);
xx = xx(ind,:);
yy = yy(ind,:);
ye = ye(ind,:);


[n,nin] = size(xx);

% Create the covariance functions
gpcf1 = gpcf_matern32('init', nin, 'lengthScale', 5, 'magnSigma2', 0.05);
gpcf1.p.lengthScale = t_p({1 4});
gpcf1.p.magnSigma2 = t_p({0.3 4});

% Create the likelihood structure
likelih = likelih_poisson('init', yy, ye);

% Create the GP data structure
gp = gp_init('init', 'FULL', nin, likelih, {gpcf1}, []);   %{gpcf2}

% Set the approximate inference method
gp = gp_init('set', gp, 'latent_method', {'EP', xx, yy, 'hyper'});

param = 'hyper';
opt=optimset('GradObj','on');
opt=optimset(opt,'TolX', 1e-3);
opt=optimset(opt,'LargeScale', 'off');
opt=optimset(opt,'Display', 'iter');

w0 = gp_pak(gp, param);
mydeal = @(varargin)varargin{1:nargout};
w = fminunc(@(ww) mydeal(gpep_e(ww, gp, xx, yy, param), gpep_g(ww, gp, xx, yy, param)), w0, opt);
gp = gp_unpak(gp,w,param);

% make prediction to the data points
[Ef, Varf] = ep_pred(gp, xx, yy, xx, param);

% Define help parameters for plotting
xxii=sub2ind([60 35],xx(:,2),xx(:,1));
[X1,X2]=meshgrid(1:35,1:60);

% plot figures
figure(1)
G=repmat(NaN,size(X1));
G(xxii)=exp(Ef + Varf/2);
pcolor(X1,X2,G),shading flat
colormap(mapcolor(G)),colorbar
set(gca, 'Clim', [0.6    1.5])
axis equal
axis([0 35 0 60])
title('Posterior mean of the relative risk, full GP')

figure(2)
G=repmat(NaN,size(X1));
G(xxii)=(exp(Varf) - 1).*exp(2*Ef+Varf);
pcolor(X1,X2,G),shading flat
colormap(mapcolor(G)),colorbar
set(gca, 'Clim', [0.005    0.03])
axis equal
axis([0 35 0 60])
title('Posterior variance of the relative risk, full GP')

% the MAP estimate of the hyperparameters in kilometers. Notice that the 
% co-ordinates in the data are not in kilometers. x=1 corresponds to 20km 
% in real life
S1 = sprintf('lengt-scale: %.3f, magnSigma2: %.3f \n', gp.cf{1}.lengthScale, gp.cf{1}.magnSigma2)

% =====================================
% 2) FIC model
% =====================================

% load the data
S = which('demo_spatial2');
L = strrep(S,'demo_spatial2.m','demos/spatial.mat');
load(L)

% Now we have loaded the following parameters
% xx = co-ordinates 
% yy = number of deaths
% ye = the expexted number of deaths

% Set the inducing inputs in a regular grid.
% Set_PIC returns the induving inputs and blockindeces for PIC. It also plots the 
% data points, inducing inputs and blocks.
dims = [1    60     1    35];
[trindex, Xu] = set_PIC(xx, dims, 2, 'corners', 0);

[n,nin] = size(xx);

% Create the covariance functions
gpcf1 = gpcf_matern32('init', nin, 'lengthScale', 5, 'magnSigma2', 0.05);
gpcf1.p.lengthScale = t_p({1 4});
gpcf1.p.magnSigma2 = t_p({0.3 4});

% Create the likelihood structure
likelih = likelih_poisson('init', yy, ye);

% Create the FIC GP data structure
gp = gp_init('init', 'FIC', nin, likelih, {gpcf1}, [], 'jitterSigmas', 0.01, 'X_u', Xu);

% Set the approximate inference method to EP
gp = gp_init('set', gp, 'latent_method', {'EP', xx, yy, 'hyper'});

% Set the optimization parameters
param = 'hyper';
opt=optimset('GradObj','on');
opt=optimset(opt,'TolX', 1e-3);
opt=optimset(opt,'LargeScale', 'off');
opt=optimset(opt,'Display', 'iter');

% conduct the hyper-parameter optimization
w0 = gp_pak(gp, param);
mydeal = @(varargin)varargin{1:nargout};
w = fminunc(@(ww) mydeal(gpep_e(ww, gp, xx, yy, param), gpep_g(ww, gp, xx, yy, param)), w0, opt);
gp = gp_unpak(gp,w,param);

% make prediction to the data points
[Ef, Varf] = ep_pred(gp, xx, yy, xx, param);

% Define help parameters for plotting
xxii=sub2ind([60 35],xx(:,2),xx(:,1));
[X1,X2]=meshgrid(1:35,1:60);

% Plot the figures
% In the results it should be noticed that:
% - there is much more people living in the south than in the north. 
%   This results in rather high variance in the north
% - The eastern Finland is known to be worse than western Finland in 
%   heart diseases also from other studies.
% - The inducing inputs are set slightly too sparsely for this data, 
%   which results in oversmoothness in the maps
figure(3)
G=repmat(NaN,size(X1));
G(xxii)=exp(Ef);
pcolor(X1,X2,G),shading flat
colormap(mapcolor(G)),colorbar
set(gca, 'Clim', [0.6    1.5])
axis equal
axis([0 35 0 60])
title('Posterior median of the relative risk, FIC')

figure(4)
G=repmat(NaN,size(X1));
G(xxii)=(exp(Varf) - 1).*exp(2*Ef+Varf);
pcolor(X1,X2,G),shading flat
colormap(mapcolor(G)),colorbar
set(gca, 'Clim', [0.005    0.03])
axis equal
axis([0 35 0 60])
title('Posterior variance of the relative risk, FIC')

% the MAP estimate of the hyperparameters in kilometers. Notice that the 
% co-ordinates in the data are not in kilometers. x=1 corresponds to 20km 
% in real life
S2 = sprintf('lengt-scale: %.3f, magnSigma2: %.3f \n', gp.cf{1}.lengthScale, gp.cf{1}.magnSigma2)

% =====================================
% 3) PIC model
% =====================================

% load the data
S = which('demo_spatial2');
L = strrep(S,'demo_spatial2.m','demos/spatial.mat');
load(L)

% Now we have loaded the following parameters
% xx = co-ordinates 
% yy = number of deaths
% ye = the expexted number of deaths

% Set the inducing inputs in a regular grid.
% Set_PIC returns the induving inputs and blockindeces for PIC. It also plots the 
% data points, inducing inputs and blocks.
dims = [1    60     1    35];
[trindex, Xu] = set_PIC(xx, dims, 5, 'corners+1xside', 1);

[n,nin] = size(xx);

% Create the covariance functions
gpcf1 = gpcf_matern32('init', nin, 'lengthScale', 15, 'magnSigma2', 0.05);
gpcf1.p.lengthScale = t_p({1 4});
gpcf1.p.magnSigma2 = t_p({0.3 4});

% Create the likelihood structure
likelih = likelih_poisson('init', yy, ye);

% Create the PIC GP data structure
gp = gp_init('init', 'PIC', nin, likelih, {gpcf1}, [], 'jitterSigmas', 0.01, 'X_u', Xu);
gp = gp_init('set', gp, 'blocks', {'manual', xx, trindex});

% Set the approximate inference method to EP
gp = gp_init('set', gp, 'latent_method', {'EP', xx, yy, 'hyper'});

% Set the optimization parameters
param = 'hyper';
opt=optimset('GradObj','on');
opt=optimset(opt,'TolX', 1e-3);
opt=optimset(opt,'LargeScale', 'off');
opt=optimset(opt,'Display', 'iter');

% conduct the hyper-parameter optimization
w0 = gp_pak(gp, param);
mydeal = @(varargin)varargin{1:nargout};
w = fminunc(@(ww) mydeal(gpep_e(ww, gp, xx, yy, param), gpep_g(ww, gp, xx, yy, param)), w0, opt);
gp = gp_unpak(gp,w,param);

% make prediction to the data points
[Ef, Varf] = ep_pred(gp, xx, yy, xx, param, trindex);

% Define help parameters for plotting
xxii=sub2ind([60 35],xx(:,2),xx(:,1));
[X1,X2]=meshgrid(1:35,1:60);

% Plot the figures
% In the results it should be noticed that:
% - there is much more people living in the south than in the north. 
%   This results in rather high variance in the north
% - The eastern Finland is known to be worse than western Finland in 
%   heart diseases also from other studies.
% - PIC has fixed the oversmoothness present in FIC
figure(6)
G=repmat(NaN,size(X1));
G(xxii)=exp(Ef);
pcolor(X1,X2,G),shading flat
colormap(mapcolor(G)),colorbar
set(gca, 'Clim', [0.6    1.5])
axis equal
axis([0 35 0 60])
title('Posterior median of the relative risk, PIC')

figure(7)
G=repmat(NaN,size(X1));
G(xxii)=(exp(Varf) - 1).*exp(2*Ef+Varf);
pcolor(X1,X2,G),shading flat
colormap(mapcolor(G)),colorbar
set(gca, 'Clim', [0.005    0.03])
axis equal
axis([0 35 0 60])
title('Posterior variance of the relative risk, PIC')

% the MAP estimate of the hyperparameters in kilometers. Notice that the 
% co-ordinates in the data are not in kilometers. x=1 corresponds to 20km 
% in real life
S3 = sprintf('lengt-scale: %.3f, magnSigma2: %.3f \n', gp.cf{1}.lengthScale, gp.cf{1}.magnSigma2)

% =====================================
% 4) CS+FIC model
% =====================================

% load the data
S = which('demo_spatial2');
L = strrep(S,'demo_spatial2.m','demos/spatial.mat');
load(L)

% Now we have loaded the following parameters
% xx = co-ordinates 
% yy = number of deaths
% ye = the expexted number of deaths

% Set the inducing inputs in a regular grid.
% Set_PIC returns the induving inputs and blockindeces for PIC. It also plots the 
% data points, inducing inputs and blocks.
dims = [1    60     1    35];
[trindex, Xu] = set_PIC(xx, dims, 5, 'corners', 0);

[n,nin] = size(xx);

% Create the covariance functions
gpcf1 = gpcf_matern32('init', nin, 'lengthScale', 4, 'magnSigma2', 0.05);
gpcf1.p.lengthScale = t_p({1 4});
gpcf1.p.magnSigma2 = t_p({0.3 4});

gpcf2 = gpcf_ppcs2('init', nin, 'lengthScale', 3, 'magnSigma2', 0.03);
gpcf2.p.lengthScale = t_p({1 4});
gpcf2.p.magnSigma2 = t_p({0.3 4});

% Create the likelihood structure
likelih = likelih_poisson('init', yy, ye);

% Create the FIC GP data structure
ti = cputime;
gp = gp_init('init', 'CS+FIC', nin, likelih, {gpcf1, gpcf2}, [], 'jitterSigmas', 0.01, 'X_u', Xu);

% Set the approximate inference method to EP
tic
gp = gp_init('set', gp, 'latent_method', {'EP', xx, yy, 'hyper'});
toc

% Set the optimization parameters
param = 'hyper';
opt=optimset('GradObj','on');
opt=optimset(opt,'TolX', 1e-3);
opt=optimset(opt,'LargeScale', 'off');
opt=optimset(opt,'Display', 'iter');

% conduct the hyper-parameter optimization
w0 = gp_pak(gp, param);
mydeal = @(varargin)varargin{1:nargout};
w = fminunc(@(ww) mydeal(gpep_e(ww, gp, xx, yy, param), gpep_g(ww, gp, xx, yy, param)), w0, opt);
gp = gp_unpak(gp,w,param);
ti = cputime- ti;

% make prediction to the data points
[Ef, Varf] = ep_pred(gp, xx, yy, xx, param);

% Define help parameters for plotting
xxii=sub2ind([60 35],xx(:,2),xx(:,1));
[X1,X2]=meshgrid(1:35,1:60);

% Plot the figures
% In the results it should be noticed that:
% - there is much more people living in the south than in the north. 
%   This results in rather high variance in the north
% - The eastern Finland is known to be worse than western Finland in 
%   heart diseases also from other studies.
figure(8)
G=repmat(NaN,size(X1));
G(xxii)=exp(Ef);
pcolor(X1,X2,G),shading flat
colormap(mapcolor(G)),colorbar
set(gca, 'Clim', [0.6    1.5])
axis equal
axis([0 35 0 60])
title('Posterior median of the relative risk, CS+FIC')

figure(9)
G=repmat(NaN,size(X1));
G(xxii)=(exp(Varf) - 1).*exp(2*Ef+Varf);
pcolor(X1,X2,G),shading flat
colormap(mapcolor(G)),colorbar
set(gca, 'Clim', [0.005    0.03])
axis equal
axis([0 35 0 60])
title('Posterior variance of the relative risk, CS+FIC')

% the MAP estimate of the hyperparameters in kilometers. Notice that the 
% co-ordinates in the data are not in kilometers. x=1 corresponds to 20km 
% in real life
S4 = sprintf('lengt-scale (matern32): %.3f, magnSigma2 (matern32): %.3f \n lengt-scale (ppcs2): %.3f, magnSigma2 (ppcs2): %.3f \n', gp.cf{1}.lengthScale, gp.cf{1}.magnSigma2, gp.cf{2}.lengthScale, gp.cf{2}.magnSigma2)
