function demo_noiset
%    DEMO_TGP      A regression problem demo for Gaussian process. Uses Students 
%                  t-distribution  for residual model.
%
%
%       Description
%       The synthetic data used here  is the same used by Radford M. Neal 
%       in his regression problem with outliers example in Software for
%       Flexible Bayesian Modeling (http://www.cs.toronto.edu/~radford/fbm.software.html).
%       The problem consist of one dimensional input and target variables. The
%       input data, x, is sampled from standard Gaussian distribution and
%       the corresponding target values come from a distribution with mean
%       given by 
%
%       y = 0.3 + 0.4x + 0.5sin(2.7x) + 1.1/(1+x^2).
%
%       For most of the cases the distribution about this mean is Gaussian
%       with standard deviation of 0.1, but with probability 0.05 a case is an
%       outlier for wchich the standard deviation is 1.0. There are total 200
%       cases from which the first 100 are used for training and the last 100
%       for testing. 


% Copyright (c) 2005 Jarno Vanhatalo, Aki Vehtari 

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

disp(' ')
disp(' The synthetic data used here  is the same used by Radford M. Neal ')
disp(' in his regression problem with outliers example in Software for ')
disp(' Flexible Bayesian Modeling (http://www.cs.toronto.edu/~radford/fbm.software.html).')
disp(' The problem consist of one dimensional input and target variables. The ')
disp(' input data, x, is sampled from standard Gaussian distribution and ')
disp(' the corresponding target values come from a distribution with mean ')
disp(' given by ')
disp(' ')
disp(' y = 0.3 + 0.4x + 0.5sin(2.7x) + 1.1/(1+x^2).')
disp(' ')
disp(' For most of the cases the distribution about this mean is Gaussian ')
disp(' with standard deviation of 0.1, but with probability 0.05 a case is an ')
disp(' outlier for wchich the standard deviation is 1.0. There are total 200 ')
disp(' cases from which the first 100 are used for training and the last 100 ')
disp(' for testing. ')
disp(' ')

% load the data. First 100 variables are for training
% and last 100 for test
S = which('demo_noiset');
L = strrep(S,'demo_noiset.m','demos/odata');
x = load(L);
xt = x(101:end,1);
yt = x(101:end,2);
y = x(1:100,2);
x = x(1:100,1);

% plot the training data with dots and the underlying 
% mean of it as a line
xx = [-2.7:0.1:2.7];
yy = 0.3+0.4*xx+0.5*sin(2.7*xx)+1.1./(1+xx.^2);
figure
plot(x,y,'.')
hold on
plot(xx,yy)
title('training data')

disp(' ')
disp(' We create a Gaussian process and priors for GP parameters. Prior for GP')
disp(' parameters is Gaussian multivariate hierarchical. The residual is given at ')
disp(' first Gaussian prior to find good starting value for noiseSigmas..')
disp(' ')

% create the Gaussian process
[n, nin] = size(x);
gpcf1 = gpcf_sexp('init', nin, 'lengthScale', repmat(1,1,nin), 'magnSigma2', 0.2^2);
gpcf2 = gpcf_noiset('init', nin, n, 'noiseSigmas2', repmat(1^2,n,1));   % Here set own Sigma2 for every data point

% Set the prior for the parameters of covariance functions 
gpcf1.p.lengthScale = gamma_p({3 7 3 7});  
gpcf1.p.magnSigma2 = sinvchi2_p({0.05^2 0.5});

gp = gp_init('init', 'FULL', nin, 'regr', {gpcf1}, {gpcf2}) %, 'jitterSigmas', 1e-4
w = gp_pak(gp, 'hyper')
gp2 = gp_unpak(gp,w, 'hyper')

opt=gp_mcopt;
opt.repeat=10;
opt.nsamples=10;
opt.hmc_opt.steps=10;
opt.hmc_opt.stepadj=0.1;
opt.hmc_opt.nsamples=1;
hmc2('state', sum(100*clock));

opt.noise_opt = sls1mm_opt;
opt.noise_opt.mmlimits = [2.1 40];

% Sample 
[r,g,rstate1]=gp_mc(opt, gp, x, y);

opt.hmc_opt.stepadj=0.08;
opt.nsamples=300;
opt.hmc_opt.steps=10;
opt.hmc_opt.persistence=1;
opt.hmc_opt.decay=0.6;

[r,g,rstate2]=gp_mc(opt, gp, x, y, [], [], r);




% set the sampling options for first two rounds of sampling
opt=gp2_mcopt;
opt.repeat=20;
opt.nsamples=2;
opt.hmc_opt.steps=20;
opt.hmc_opt.stepadj=0.4;
opt.hmc_opt.nsamples=1;
opt.hmc_opt.persistence=1;
opt.sample_variances=0;
hmc2('state', sum(100*clock));

[r,gp,rstate1]=gp2r_mc(opt, gp, x, y);

disp(' ')
disp(' After sampling two samples with normal residuel model we construct ')
disp(' hierarchical Students t-distribution for residual. The prior is ')
disp(' constructed by giving each data point different noise variance. ')
disp(' By integrating ower the priors hyperparameters the likelihood ') 
disp(' approuches Students t-distribution as the number of data points ')
disp(' increeses. At first The number of degrees of freedom ')
disp(' for residual model is fixed. ')
disp(' ')

% Create Student's t prior for residual and sample with fixed number of
% degrees of freedom.
gp.p.noiseSigmas=invgam_p({gp.noiseSigmas 4 0.05 1});
gp.noiseSigmas=repmat(gp.noiseSigmas,1,n);
gp.noiseVariances=gp.noiseSigmas.^2;
opt.sample_variances=1;

[r,gp,rstate]=gp2r_mc(opt, gp, x, y);

disp(' ')
disp(' After sampling for two rounds with fixed number of degrees ')
disp(' of freedom we give an inverse Gamma prior for the number of ')
disp(' degrees of freedom. ')
disp(' ')
disp(' After this we would do the main sampling. The main sampling is ')
disp(' skipped and an old network record structure is loaded in order ')
disp(' to save time. To do the main sampling, uncomment mlp2r_mc line  ')
disp(' from program and comment out the load lines.')
disp(' ')

% Add a hyperprior for degrees of freedom in t-distribution
gp.p.noiseSigmas.p.nu=invgam_p({6 1});
opt.hmc_opt.stepadj=0.4;
opt.nsamples= 300;

% Here we would do the main sampling. In order to save time we have
% saved one GP record structure in the software. The record (and though 
% the samples) are loaded and used in the demo. In order to do your own 
% sampling uncomment the line below.
%
%[r,gp,rstate]=gp2r_mc(opt, gp, x, y, [], [], r, rstate);

S = which('demo_tgp');
L = strrep(S,'demo_tgp.m','demos/tgprecord.mat');
load(L)

disp(' ')
disp(' Finally we are ready to forward propagate new input data and plot')
disp(' the outcome. The test data and network predictions are plotted ')
disp(' together with the underlying mean. ')
disp(' ')

% thin the record
rr = thin(r,10,2);

% make predictions for test set
tga = mean(squeeze(gp2fwds(rr,x,y,xt)),2);

% Plot the network outputs as '.', and underlying mean with '--'
figure
plot(xt,tga,'.')
hold on
plot(xx,yy,'--')
legend('prediction', 'mean')
title('prediction from MLP')

% plot the test data set as '.', and underlying mean with '--'
figure
plot(xt,yt,'.')
hold on
plot(xx,yy,'--')
legend('data', 'mean')
title('test data')

% evaluate the RMSE error with respect to mean
yyt = 0.3+0.4*xt+0.5*sin(2.7*xt)+1.1./(1+xt.^2);
er = (yyt-tga)'*(yyt-tga)/length(yyt);
