%DEMO_SPATIAL1    Demonstration for a disease mapping problem
%                 with Gaussian process prior
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
%    The inference is conducted via MCMC. We sample from the full posterior 
%    p(f, th| data) by alternating the sampling from the conditional posteriors
%    p(f | th, data) and p(th | f, data). The sampling from the conditional 
%    posteriors is done by hybrid Monte Carlo (see, for example, Neal, 1996).
%
%    See Vanhatalo and Vehtari (2007) for more detailed discussion.
%
%    See also  DEMO_REGRESSION1, DEMO_CLASSIFIC1

% Copyright (c) 2008 Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.


% This file is organised in three parts:
%  1) data analysis with full GP model
%  2) data analysis with FIC approximation
%  3) data analysis with PIC approximation


% =====================================
% 1) FULL model
% =====================================

S = which('demo_spatial1');
L = strrep(S,'demo_spatial1.m','demos/spatial.mat');
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

% Create the covariance function
% The hyper-parameters are initialized very close to posterior mode in order to 
% speed up convergence
gpcf1 = gpcf_matern32('init', 'lengthScale', 2, 'magnSigma2', 0.03);
pl = prior_t('init');
pm = prior_t('init', 'scale', 0.3);
gpcf1 = gpcf_matern32('set', gpcf1, 'lengthScale_prior', pl, 'magnSigma2_prior', pm);

% Create the likelihood structure
likelih = likelih_poisson('init', yy, ye);

% Create the GP data structure
gp = gp_init('init', 'FULL', likelih, {gpcf1}, []);   %{gpcf2}

% Set the approximate inference method
gp = gp_init('set', gp, 'latent_method', {'MCMC', zeros(size(yy))', @scaled_hmc});

% Set the sampling options
opt=gp_mcopt;
opt.nsamples=1;
opt.repeat=1;

% HMC-hyper
opt.hmc_opt.steps=3;
opt.hmc_opt.stepadj=0.01;
opt.hmc_opt.nsamples=1;
opt.hmc_opt.persistence=0;
opt.hmc_opt.decay=0.8;
    
% HMC-latent
opt.latent_opt.nsamples=1;
opt.latent_opt.nomit=0;
opt.latent_opt.persistence=0;
opt.latent_opt.repeat=20;
opt.latent_opt.steps=20;
opt.latent_opt.stepadj=0.15;
opt.latent_opt.window=5;

% Here we make an initialization with 
% slow sampling parameters
opt.display = 1;
[rgp,gp,opt]=gp_mc(opt, gp, xx, yy);

% Now we reset the sampling parameters to 
% achieve faster sampling
opt.latent_opt.repeat=1;
opt.latent_opt.steps=7;
opt.latent_opt.window=1;
opt.latent_opt.stepadj=0.15;
opt.hmc_opt.persistence=0;
opt.hmc_opt.stepadj=0.005;
opt.hmc_opt.steps=2;

opt.display = 1;
opt.hmc_opt.display = 0;
opt.latent_opt.display=0;

% Define help parameters for plotting
xxii=sub2ind([60 35],xx(:,2),xx(:,1));
[X1,X2]=meshgrid(1:35,1:60);

% Conduct the actual sampling.
% Inside the loop we sample one sample from the latent values and 
% hyper-parameters at each iteration. After that we plot the samples 
% so that we can visually inspect the progress of sampling
while length(rgp.edata)<1000 %   1000
    [rgp,gp,opt]=gp_mc(opt, gp, xx, yy, rgp);
    fprintf('        mean hmcrej: %.2f latrej: %.2f\n', mean(rgp.hmcrejects), mean(rgp.lrejects))
    figure(1)
    clf
    subplot(1,2,1)
    plot(rgp.cf{1}.lengthScale, rgp.cf{1}.magnSigma2)
    hold on
    plot(rgp.cf{1}.lengthScale(end), rgp.cf{1}.magnSigma2(end),'r*')
    xlabel('lenght-scale')
    ylabel('magnitude')
    drawnow
    %    subplot(2,2,[2 4])
    subplot(1,2,2)
    G=repmat(NaN,size(X1));
    G(xxii)=exp(gp.latentValues);
    pcolor(X1,X2,G),shading flat
    colormap(mapcolor(G)),colorbar
    axis equal
    axis([0 35 0 60])
    title('relative risk')
    drawnow
end

figure(1)
clf
G=repmat(NaN,size(X1));
G(xxii)=median(exp(rgp.latentValues));
pcolor(X1,X2,G),shading flat
colormap(mapcolor(G)),colorbar
set(gca, 'Clim', [0.6    1.5])
axis equal
axis([0 35 0 60])
title('Posterior median of relative risk, full GP')

figure(2)
G=repmat(NaN,size(X1));
G(xxii)=std(exp(rgp.latentValues), [], 1).^2;
pcolor(X1,X2,G),shading flat
colormap(mapcolor(G)),colorbar
set(gca, 'Clim', [0.005    0.03])
axis equal
axis([0 35 0 60])
title('Posterior variance of relative risk, full GP')

% =====================================
% 2) FIC model
% =====================================

% load the data
S = which('demo_spatial1');
L = strrep(S,'demo_spatial1.m','demos/spatial.mat');
load(L)

% Now we have loaded the following parameters
% xx = co-ordinates 
% yy = number of deaths
% ye = the expexted number of deaths

% Set the inducing inputs in a regular grid.
% Set_PIC returns the induving inputs and blockindeces for PIC. It also plots the 
% data points, inducing inputs and blocks.
dims = [1    60     1    35];
[trindex, Xu] = set_PIC(xx, dims, 3, 'corners', 0);

[n,nin] = size(xx);

% Create the covariance functions
% The hyper-parameters are initialized very close to posterior mode in order to 
% speed up convergence
gpcf1 = gpcf_matern32('init', 'lengthScale', 2, 'magnSigma2', 0.03);
pl = prior_t('init');
pm = prior_t('init', 'scale', 0.3);
gpcf1 = gpcf_matern32('set', gpcf1, 'lengthScale_prior', pl, 'magnSigma2_prior', pm);


% Create the likelihood structure
likelih = likelih_poisson('init', yy, ye);

% Create the FIC GP data structure
gp = gp_init('init', 'FIC', likelih, {gpcf1}, [], 'jitterSigma2', 0.01.^2, 'X_u', Xu);

% Set the approximate inference method to MCMC
gp = gp_init('set', gp, 'latent_method', {'MCMC', zeros(size(yy))', @scaled_hmc});

% Set the sampling options
opt=gp_mcopt;
opt.nsamples=1;
opt.repeat=1;

% HMC-hyper
opt.hmc_opt.steps=3;
opt.hmc_opt.stepadj=0.01;
opt.hmc_opt.nsamples=1;
opt.hmc_opt.persistence=0;
opt.hmc_opt.decay=0.8;
    
% HMC-latent
opt.latent_opt.nsamples=1;
opt.latent_opt.nomit=0;
opt.latent_opt.persistence=0;
opt.latent_opt.repeat=20;
opt.latent_opt.steps=20;
opt.latent_opt.stepadj=0.15;
opt.latent_opt.window=5;

% Here we make an initialization with 
% slow sampling parameters
opt.display = 0;
[rgp,gp,opt]=gp_mc(opt, gp, xx, yy);

% Now we reset the sampling parameters to 
% achieve faster sampling
opt.latent_opt.repeat=1;
opt.latent_opt.steps=7;
opt.latent_opt.window=1;
opt.latent_opt.stepadj=0.15;
opt.hmc_opt.persistence=0;
opt.hmc_opt.stepadj=0.01;
opt.hmc_opt.steps=2;

opt.display = 1;
opt.hmc_opt.display = 0;
opt.latent_opt.display=0;

% Define help parameters for plotting
xxii=sub2ind([60 35],xx(:,2),xx(:,1));
[X1,X2]=meshgrid(1:35,1:60);

% Conduct the actual sampling.
% Inside the loop we sample one sample from the latent values and 
% hyper-parameters at each iteration. After that we plot the samples 
% so that we can visually inspect the progress of sampling
while length(rgp.edata)<1000 %   1000
    [rgp,gp,opt]=gp_mc(opt, gp, xx, yy, rgp);
    fprintf('        mean hmcrej: %.2f latrej: %.2f\n', mean(rgp.hmcrejects), mean(rgp.lrejects))
    figure(3)
    clf
    subplot(1,2,1)
    plot(rgp.cf{1}.lengthScale, rgp.cf{1}.magnSigma2)
    xlabel('lenght-scale')
    ylabel('magnitude')
    hold on
    plot(rgp.cf{1}.lengthScale(end), rgp.cf{1}.magnSigma2(end),'r*')
    drawnow
    %    subplot(2,2,[2 4])
    subplot(1,2,2)
    G=repmat(NaN,size(X1));
    G(xxii)=exp(gp.latentValues);
    pcolor(X1,X2,G),shading flat
    colormap(mapcolor(G)),colorbar
    axis equal
    axis([0 35 0 60])
    title('relative risk')
    drawnow
end

figure(3)
clf
G=repmat(NaN,size(X1));
G(xxii)=median(exp(rgp.latentValues));
pcolor(X1,X2,G),shading flat
colormap(mapcolor(G)),colorbar
set(gca, 'Clim', [0.6    1.5])
axis equal
axis([0 35 0 60])
title('Posterior median of relative risk, FIC GP')

figure(4)
G=repmat(NaN,size(X1));
G(xxii)=std(exp(rgp.latentValues), [], 1).^2;
pcolor(X1,X2,G),shading flat
colormap(mapcolor(G)),colorbar
set(gca, 'Clim', [0.005    0.03])
axis equal
axis([0 35 0 60])
title('Posterior variance of relative risk, FIC GP')


% =====================================
% 3) PIC model
% =====================================

% load the data
S = which('demo_spatial1');
L = strrep(S,'demo_spatial1.m','demos/spatial.mat');
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
gpcf1 = gpcf_matern32('init', 'lengthScale', 2, 'magnSigma2', 0.03);
pl = prior_t('init');
pm = prior_t('init', 'scale', 0.3);
gpcf1 = gpcf_matern32('set', gpcf1, 'lengthScale_prior', pl, 'magnSigma2_prior', pm);

% Create the likelihood structure
likelih = likelih_poisson('init', yy, ye);

% Create the PIC GP data structure
gp = gp_init('init', 'PIC', likelih, {gpcf1}, [], 'jitterSigma2', 0.01.^2, 'X_u', Xu);
gp = gp_init('set', gp, 'blocks', {'manual', xx, trindex});

% Set the approximate inference method to MCMC
gp = gp_init('set', gp, 'latent_method', {'MCMC', zeros(size(yy))', @scaled_hmc});

% Set the sampling options
opt=gp_mcopt;
opt.nsamples=1;
opt.repeat=1;

% HMC-hyper
opt.hmc_opt.steps=3;
opt.hmc_opt.stepadj=0.01;
opt.hmc_opt.nsamples=1;
opt.hmc_opt.persistence=0;
opt.hmc_opt.decay=0.8;
    
% HMC-latent
opt.latent_opt.nsamples=1;
opt.latent_opt.nomit=0;
opt.latent_opt.persistence=0;
opt.latent_opt.repeat=20;
opt.latent_opt.steps=20;
opt.latent_opt.stepadj=0.15;
opt.latent_opt.window=5;

% Here we make an initialization with 
% slow sampling parameters
opt.display = 0;
[rgp,gp,opt]=gp_mc(opt, gp, xx, yy);

% Now we reset the sampling parameters to 
% achieve faster sampling
opt.latent_opt.repeat=1;
opt.latent_opt.steps=7;
opt.latent_opt.window=1;
opt.latent_opt.stepadj=0.15;
opt.hmc_opt.persistence=0;
opt.hmc_opt.stepadj=0.01;
opt.hmc_opt.steps=2;

opt.display = 1;
opt.hmc_opt.display = 0;
opt.latent_opt.display=0;

% Define help parameters for plotting
xxii=sub2ind([60 35],xx(:,2),xx(:,1));
[X1,X2]=meshgrid(1:35,1:60);

% Conduct the actual sampling.
% Inside the loop we sample one sample from the latent values and 
% hyper-parameters at each iteration. After that we plot the samples 
% so that we can visually inspect the progress of sampling
while length(rgp.edata)<1000 %   1000
    [rgp,gp,opt]=gp_mc(opt, gp, xx, yy, rgp);
    fprintf('        mean hmcrej: %.2f latrej: %.2f\n', mean(rgp.hmcrejects), mean(rgp.lrejects))
    figure(6)
    clf
    subplot(1,2,1)
    plot(rgp.cf{1}.lengthScale, rgp.cf{1}.magnSigma2)
    xlabel('lenght-scale')
    ylabel('magnitude')
    hold on
    plot(rgp.cf{1}.lengthScale(end), rgp.cf{1}.magnSigma2(end),'r*')
    drawnow
    %    subplot(2,2,[2 4])
    subplot(1,2,2)
    G=repmat(NaN,size(X1));
    G(xxii)=exp(gp.latentValues);
    pcolor(X1,X2,G),shading flat
    colormap(mapcolor(G)),colorbar
    axis equal
    axis([0 35 0 60])
    title('relative risk')
    drawnow
end

figure(6)
clf
G=repmat(NaN,size(X1));
G(xxii)=median(exp(rgp.latentValues));
pcolor(X1,X2,G),shading flat
colormap(mapcolor(G)),colorbar
set(gca, 'Clim', [0.6    1.5])
axis equal
axis([0 35 0 60])
title('Posterior median of relative risk, PIC GP')

figure(7)
G=repmat(NaN,size(X1));
G(xxii)=std(exp(rgp.latentValues), [], 1).^2;
pcolor(X1,X2,G),shading flat
colormap(mapcolor(G)),colorbar
set(gca, 'Clim', [0.005    0.03])
axis equal
axis([0 35 0 60])
title('Posterior variance of relative risk, PIC GP')


% =====================================
% 4) CS+FIC model
% =====================================

% NOTE! The CS+FIC model forms a full nxn matrix. The latent 
% value transformation is not yet implemented efficiently.

% load the data
S = which('demo_spatial1');
L = strrep(S,'demo_spatial1.m','demos/spatial.mat');
load(L)

% Now we have loaded the following parameters
% xx = co-ordinates 
% yy = number of deaths
% ye = the expexted number of deaths

% Set the inducing inputs in a regular grid.
% Set_PIC returns the induving inputs and blockindeces for PIC. It also plots the 
% data points, inducing inputs and blocks.
dims = [1    60     1    35];
[trindex, Xu] = set_PIC(xx, dims, 3, 'corners', 1);

[n,nin] = size(xx);

% Create the covariance functions
gpcf1 = gpcf_matern32('init', 'lengthScale', 4, 'magnSigma2', 0.05);
pl = prior_t('init');
pm = prior_t('init', 'scale', 0.3);
gpcf1 = gpcf_matern32('set', gpcf1, 'lengthScale_prior', pl, 'magnSigma2_prior', pm);

gpcf2 = gpcf_ppcs2('init', nin, 'lengthScale', 3, 'magnSigma2', 0.03);
gpcf2 = gpcf_ppcs2('set', gpcf2, 'lengthScale_prior', pl, 'magnSigma2_prior', pm);

% Create the likelihood structure
likelih = likelih_poisson('init', yy, ye);

% Create the FIC GP data structure
gp = gp_init('init', 'CS+FIC', likelih, {gpcf1, gpcf2}, [], 'jitterSigma2', 0.01.^2, 'X_u', Xu);

% Set the approximate inference method to MCMC
gp = gp_init('set', gp, 'latent_method', {'MCMC', zeros(size(yy))', @scaled_hmc});

% Set the sampling options
opt=gp_mcopt;
opt.nsamples=1;
opt.repeat=1;

% HMC-hyper
opt.hmc_opt.steps=3;
opt.hmc_opt.stepadj=0.01;
opt.hmc_opt.nsamples=1;
opt.hmc_opt.persistence=0;
opt.hmc_opt.decay=0.8;
    
% HMC-latent
opt.latent_opt.nsamples=1;
opt.latent_opt.nomit=0;
opt.latent_opt.persistence=0;
opt.latent_opt.repeat=20;
opt.latent_opt.steps=20;
opt.latent_opt.stepadj=0.15;
opt.latent_opt.window=5;

% Here we make an initialization with 
% slow sampling parameters
opt.display = 0;
[rgp,gp,opt]=gp_mc(opt, gp, xx, yy);

% Now we reset the sampling parameters to 
% achieve faster sampling
opt.latent_opt.repeat=1;
opt.latent_opt.steps=7;
opt.latent_opt.window=1;
opt.latent_opt.stepadj=0.15;
opt.hmc_opt.persistence=0;
opt.hmc_opt.stepadj=0.01;
opt.hmc_opt.steps=2;

opt.display = 1;
opt.hmc_opt.display = 0;
opt.latent_opt.display=0;

% Define help parameters for plotting
xxii=sub2ind([60 35],xx(:,2),xx(:,1));
[X1,X2]=meshgrid(1:35,1:60);

% Conduct the actual sampling.
% Inside the loop we sample one sample from the latent values and 
% hyper-parameters at each iteration. After that we plot the samples 
% so that we can visually inspect the progress of sampling
while length(rgp.edata)<200 %   1000
    [rgp,gp,opt]=gp_mc(opt, gp, xx, yy, rgp);
    fprintf('        mean hmcrej: %.2f latrej: %.2f\n', mean(rgp.hmcrejects), mean(rgp.lrejects))
    figure(7)
    clf
    subplot(2,2,1)
    plot(rgp.cf{1}.lengthScale, rgp.cf{1}.magnSigma2)
    xlabel('lenght-scale')
    ylabel('magnitude')
    hold on
    plot(rgp.cf{1}.lengthScale(end), rgp.cf{1}.magnSigma2(end),'r*')
    title('sexp')
    drawnow
    subplot(2,2,3)
    plot(rgp.cf{1}.lengthScale, rgp.cf{2}.magnSigma2)
    xlabel('lenght-scale')
    ylabel('magnitude')
    hold on
    plot(rgp.cf{1}.lengthScale(end), rgp.cf{2}.magnSigma2(end),'r*')
    title('ppcs2')
    drawnow
    subplot(2,2,[2 4])
    G=repmat(NaN,size(X1));
    G(xxii)=exp(gp.latentValues);
    pcolor(X1,X2,G),shading flat
    colormap(mapcolor(G)),colorbar
    axis equal
    axis([0 35 0 60])
    title('relative risk')
    drawnow
end

figure(8)
G=repmat(NaN,size(X1));
G(xxii)=median(exp(rgp.latentValues));
pcolor(X1,X2,G),shading flat
colormap(mapcolor(G)),colorbar
axis equal
axis([0 35 0 60])
title('Posterior median of relative risk, FIC')