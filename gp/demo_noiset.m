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

% This file is organized as follows:
%
% 1) Optimization approach with Normal noise
% 2) MCMC approach with scale mixture noise model (~=Student-t)
%    All parameters sampled
% 3) Laplace approximation Student-t likelihood
%    All parameters optimized
% 4) MCMC approach with scale mixture noise model (~=Student-t)
%    nu kept fixed to 4
% 5) Laplace approximation Student-t likelihood
%    nu kept fixed to 4
% 6) Comparing the conditional posterior distributions of latent 
%    variables from MCMC and Laplace approach


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

% ========================================
% Optimization approach with Normal noise
% ========================================

% load the data. First 100 variables are for training
% and last 100 for test
S = which('demo_noiset');
L = strrep(S,'demo_noiset.m','demos/odata');
x = load(L);
xt = x(101:end,1);
yt = x(101:end,2);
y = x(1:100,2);
x = x(1:100,1);
[n, nin] = size(x); 

% Test data
xx = [-2.7:0.01:2.7];
yy = 0.3+0.4*xx+0.5*sin(2.7*xx)+1.1./(1+xx.^2);


disp(' ')
disp(' We create a Gaussian process and priors for GP parameters. Prior for GP')
disp(' parameters is Gaussian multivariate hierarchical. The residual is given at ')
disp(' first Gaussian prior to find good starting value for noiseSigmas..')
disp(' ')

% create the Gaussian process
gpcf1 = gpcf_sexp('init', nin, 'lengthScale', 1, 'magnSigma2', 0.2^2);
gpcf2 = gpcf_noise('init', nin, 'noiseSigmas2', 0.2^2);

% ... Then set the prior for the parameters of covariance functions...
gpcf2.p.noiseSigmas2 = sinvchi2_p({0.05^2 0.5});
gpcf1.p.lengthScale = gamma_p({3 7});  
gpcf1.p.magnSigma2 = sinvchi2_p({0.05^2 0.5});

% ... Finally create the GP data structure
gp = gp_init('init', 'FULL', nin, 'regr', {gpcf1}, {gpcf2}, 'jitterSigmas', 0.001)    

w=gp_pak(gp, 'hyper');  % pack the hyperparameters into one vector
fe=str2fun('gp_e');     % create a function handle to negative log posterior
fg=str2fun('gp_g');     % create a function handle to gradient of negative log posterior

% set the options
opt(1) = 1;
opt(2) = 1e-3;
opt(3) = 3e-3;
opt(9) = 0;
opt(10) = 0;
opt(11) = 0;
opt(14) = 0;

% do the optimization
[w, opt, flog]=scg(fe, w, opt, fg, gp, x, y, 'hyper');

% Set the optimized hyperparameter values back to the gp structure
gp=gp_unpak(gp,w, 'hyper');

% Prediction
[Ef, Varf] = gp_pred(gp, x, y, xx');
std_f = sqrt(Varf + gp.noise{1}.noiseSigmas2);

% Plot the prediction and data
% plot the training data with dots and the underlying 
% mean of it as a line
figure
hold on
plot(xx,yy, 'k')
plot(xx, Ef)
plot(xx, Ef-2*std_f, 'r--')
plot(x,y,'b.')
%plot(xt,yt,'r.')
legend('real f', 'Ef', 'Ef+std(f)','y')
plot(xx, Ef+2*std_f, 'r--')
axis on;
title('The predictions and the data points (MAP solution and normal noise)');
S1 = sprintf('lengt-scale: %.3f, magnSigma2: %.3f,  noiseSigma2: %.3f  \n', gp.cf{1}.lengthScale, gp.cf{1}.magnSigma2, gp.noise{1}.noiseSigmas2)


% ========================================
% MCMC approach with scale mixture noise model (~=Student-t)
% Here we sample all the variables 
%     (lenghtScale, magnSigma, sigma(noise-t) and nu)
% ========================================
[n, nin] = size(x);
gpcf1 = gpcf_sexp('init', nin, 'lengthScale', repmat(1,1,nin), 'magnSigma2', 0.2^2);
gpcf2 = gpcf_noiset('init', nin, n, 'noiseSigmas2', repmat(1^2,n,1));   % Here set own Sigma2 for every data point

% Un-freeze nu
gpcf2 = gpcf_noiset('set', gpcf2, 'freeze_nu', 0);

% Set the prior for the parameters of covariance functions 
gpcf1.p.lengthScale = gamma_p({3 7 3 7});  
gpcf1.p.magnSigma2 = sinvchi2_p({0.05^2 0.5});

gp = gp_init('init', 'FULL', nin, 'regr', {gpcf1}, {gpcf2}) %
w = gp_pak(gp, 'hyper')
gp2 = gp_unpak(gp,w, 'hyper')

opt=gp_mcopt;
opt.repeat=10;
opt.nsamples=10;
opt.hmc_opt.steps=10;
opt.hmc_opt.stepadj=0.1;
opt.hmc_opt.nsamples=1;
hmc2('state', sum(100*clock));

opt.gibbs_opt = sls1mm_opt;
opt.gibbs_opt.maxiter = 50;
opt.gibbs_opt.mmlimits = [0 40];
opt.gibbs_opt.method = 'minmax';

% Sample 
[r,g,rstate1]=gp_mc(opt, gp, x, y);

opt.hmc_opt.stepadj=0.08;
opt.nsamples=500;
opt.hmc_opt.steps=10;
opt.hmc_opt.persistence=1;
opt.hmc_opt.decay=0.6;

[r,g,rstate2]=gp_mc(opt, g, x, y, [], [], r);
rr = r;

% thin the record
rr = thin(r,100,2);

figure 
hist(rr.noise{1}.nu,20)
title('Mixture model, \nu')
figure 
hist(sqrt(rr.noise{1}.tau2).*rr.noise{1}.alpha,20)
title('Mixture model, \sigma')
figure 
hist(rr.cf{1}.lengthScale,20)
title('Mixture model, length-scale')
figure 
hist(rr.cf{1}.magnSigma2,20)
title('Mixture model, magnSigma2')


% $$$ >> mean(rr.noise{1}.nu)
% $$$ ans =
% $$$     1.5096
% $$$ >> mean(sqrt(rr.noise{1}.tau2).*rr.noise{1}.alpha)
% $$$ ans =
% $$$     0.0683
% $$$ >> mean(rr.cf{1}.lengthScale)
% $$$ ans =
% $$$     1.0197
% $$$ >> mean(rr.cf{1}.magnSigma2)
% $$$ ans =
% $$$     1.2903

% make predictions for test set
[Ef, Varf] = gp_preds(rr,x,y,xx');
Ef = mean(squeeze(Ef),2);
std_f = sqrt(mean(squeeze(Varf),2) );

% Plot the network outputs as '.', and underlying mean with '--'
figure
plot(xx,yy,'k')
hold on
plot(xx,Ef)
plot(xx, Ef-2*std_f, 'r--')
plot(x,y,'.')
legend('real f', 'Ef', 'Ef+std(f)','y')
plot(xx, Ef+2*std_f, 'r--')
title('The predictions and the data points (MAP solution and hierarchical noise)')
S2 = sprintf('lengt-scale: %.3f, magnSigma2: %.3f \n', mean(rr.cf{1}.lengthScale), mean(rr.cf{1}.magnSigma2))

% ========================================
% Laplace approximation Student-t likelihood
%  Here we optimize all the variables 
%  (lenghtScale, magnSigma, sigma(noise-t) and nu)
% ========================================

% load the data. First 100 variables are for training
% and last 100 for test
S = which('demo_noiset');
L = strrep(S,'demo_noiset.m','demos/odata');
x = load(L);
xt = x(101:end,1);
yt = x(101:end,2);
y = x(1:100,2);
x = x(1:100,1);
[n, nin] = size(x); 

% Test data
xx = [-2.7:0.01:2.7];
yy = 0.3+0.4*xx+0.5*sin(2.7*xx)+1.1./(1+xx.^2);

gpcf1 = gpcf_sexp('init', nin, 'lengthScale', 0.5, 'magnSigma2', 2^2);

% ... Then set the prior for the parameters of covariance functions...
gpcf1.p.lengthScale = gamma_p({3 7});  
gpcf1.p.magnSigma2 = sinvchi2_p({0.5^2 0.5});

% Create the likelihood structure
likelih = likelih_t('init', 4, 0.2);
likelih.p.nu = loglogunif_p;
likelih.p.sigma = logunif_p;
% Set freeze_nu = 0 so that nu is also optimized
likelih = likelih_t('set', likelih, 'freeze_nu', 0)

% ... Finally create the GP data structure
param = 'hyper+likelih'
gp = gp_init('init', 'FULL', nin, likelih, {gpcf1}, {}, 'jitterSigmas', 0.01);
gp = gp_init('set', gp, 'latent_method', {'Laplace', x, y, param});

w = randn(size(gp_pak(gp,param)));
%w = gp_pak(gp,param);
gradcheck(w, @gpla_e, @gpla_g, gp, x, y, param)
exp(w) 

opt=optimset('GradObj','on');
opt=optimset(opt,'TolX', 1e-3);
opt=optimset(opt,'LargeScale', 'off');
opt=optimset(opt,'Display', 'iter');
w0 = gp_pak(gp, param);
mydeal = @(varargin)varargin{1:nargout};
w = fminunc(@(ww) mydeal(gpla_e(ww, gp, x, y, param), gpla_g(ww, gp, x, y, param)), w0, opt);
gp = gp_unpak(gp,w,param);

% Predictions to test points
[Ef, Varf] = la_pred(gp, x, y, xx', param);
std_f = sqrt(Varf);

% Plot the prediction and data
figure
plot(xx,yy,'k')
hold on
plot(xx,Ef)
plot(xx, Ef-2*std_f, 'r--')
plot(x,y,'.')
legend('real f', 'Ef', 'Ef+std(f)','y')
plot(xx, Ef+2*std_f, 'r--')
title(sprintf('The predictions and the data points (MAP solution, Student-t (nu=%.2f,sigma=%.3f) noise)',gp.likelih.nu, gp.likelih.sigma));
S4 = sprintf('lengt-scale: %.3f, magnSigma2: %.3f \n', gp.cf{1}.lengthScale, gp.cf{1}.magnSigma2)

% ========================================
% MCMC approach with scale mixture noise model (~=Student-t)
%  Here we analyse the model with fixed degrees of freedom
%   n = 4 
%   Notice that the default value for freeze_nu = 1, 
%   which means that degrees of freedom is not sampled/optimized
% ========================================
[n, nin] = size(x);
gpcf1 = gpcf_sexp('init', nin, 'lengthScale', repmat(1,1,nin), 'magnSigma2', 0.2^2);
gpcf2 = gpcf_noiset('init', nin, n, 'noiseSigmas2', repmat(1^2,n,1), 'nu', 4);   % Here set own Sigma2 for every data point

% Set the prior for the parameters of covariance functions 
gpcf1.p.lengthScale = gamma_p({3 7 3 7});  
gpcf1.p.magnSigma2 = sinvchi2_p({0.05^2 0.5});

gp = gp_init('init', 'FULL', nin, 'regr', {gpcf1}, {gpcf2}) %
w = gp_pak(gp, 'hyper')
gp2 = gp_unpak(gp,w, 'hyper')

opt=gp_mcopt;
opt.repeat=10;
opt.nsamples=10;
opt.hmc_opt.steps=10;
opt.hmc_opt.stepadj=0.1;
opt.hmc_opt.nsamples=1;
hmc2('state', sum(100*clock));

opt.gibbs_opt = sls1mm_opt;
opt.gibbs_opt.maxiter = 50;
opt.gibbs_opt.mmlimits = [0 40];
opt.gibbs_opt.method = 'minmax';

% Sample 
[r,g,rstate1]=gp_mc(opt, gp, x, y);

opt.hmc_opt.stepadj=0.08;
opt.nsamples=100;
opt.hmc_opt.steps=10;
opt.hmc_opt.persistence=1;
opt.hmc_opt.decay=0.6;

[r,g,rstate2]=gp_mc(opt, g, x, y, [], [], r);
rr = r;

% thin the record
rr = thin(r,100,2);

figure 
hist(rr.noise{1}.nu,20)
title('Mixture model, \nu')
figure 
hist(sqrt(rr.noise{1}.tau2).*rr.noise{1}.alpha,20)
title('Mixture model, \sigma')
figure 
hist(rr.cf{1}.lengthScale,20)
title('Mixture model, length-scale')
figure 
hist(rr.cf{1}.magnSigma2,20)
title('Mixture model, magnSigma2')


% $$$ >> mean(sqrt(rr.noise{1}.tau2).*rr.noise{1}.alpha)
% $$$ ans =
% $$$     
% $$$ >> mean(rr.cf{1}.lengthScale)
% $$$ ans =
% $$$     
% $$$ >> mean(rr.cf{1}.magnSigma2)
% $$$ ans =
% $$$     

% make predictions for test set
[Ef, Varf] = gp_preds(rr,x,y,xx');
Ef = mean(squeeze(Ef),2);
std_f = sqrt(mean(squeeze(Varf),2) );

% Plot the network outputs as '.', and underlying mean with '--'
figure
plot(xx,yy,'k')
hold on
plot(xx,Ef)
plot(xx, Ef-2*std_f, 'r--')
plot(x,y,'.')
legend('real f', 'Ef', 'Ef+std(f)','y')
plot(xx, Ef+2*std_f, 'r--')
title('The predictions and the data points (MAP solution and hierarchical noise)')
S2 = sprintf('lengt-scale: %.3f, magnSigma2: %.3f \n', mean(rr.cf{1}.lengthScale), mean(rr.cf{1}.magnSigma2))

% ========================================
% Laplace approximation Student-t likelihood
%  Here we analyse the model with fixed degrees of freedom
%   n = 4 
%   Notice that the default value for freeze_nu = 1, 
%   which means that degrees of freedom is not sampled/optimized
% ========================================

% load the data. First 100 variables are for training
% and last 100 for test
S = which('demo_noiset');
L = strrep(S,'demo_noiset.m','demos/odata');
x = load(L);
xt = x(101:end,1);
yt = x(101:end,2);
y = x(1:100,2);
x = x(1:100,1);
[n, nin] = size(x); 

% Test data
xx = [-2.7:0.01:2.7];
yy = 0.3+0.4*xx+0.5*sin(2.7*xx)+1.1./(1+xx.^2);

gpcf1 = gpcf_sexp('init', nin, 'lengthScale', 0.5, 'magnSigma2', 2^2);

% ... Then set the prior for the parameters of covariance functions...
gpcf1.p.lengthScale = gamma_p({3 7});  
gpcf1.p.magnSigma2 = sinvchi2_p({0.5^2 0.5});

% Create the likelihood structure
likelih = likelih_t('init', 4, 0.2);
likelih.p.nu = loglogunif_p;
likelih.p.sigma = logunif_p;

% ... Finally create the GP data structure
param = 'hyper+likelih'
gp = gp_init('init', 'FULL', nin, likelih, {gpcf1}, {}, 'jitterSigmas', 0.01);
gp = gp_init('set', gp, 'latent_method', {'Laplace', x, y, param});

% gradient checking
w = randn(size(gp_pak(gp,param)));
gradcheck(w, @gpla_e, @gpla_g, gp, x, y, param)
exp(w) 

opt=optimset('GradObj','on');
opt=optimset(opt,'TolX', 1e-3);
opt=optimset(opt,'LargeScale', 'off');
opt=optimset(opt,'Display', 'iter');
w0 = gp_pak(gp, param);
mydeal = @(varargin)varargin{1:nargout};
w = fminunc(@(ww) mydeal(gpla_e(ww, gp, x, y, param), gpla_g(ww, gp, x, y, param)), w0, opt);
gp = gp_unpak(gp,w,param);

% Predictions to test points
[Ef, Varf] = la_pred(gp, x, y, xx', param);
std_f = sqrt(Varf);

% Plot the prediction and data
figure
plot(xx,yy,'k')
hold on
plot(xx,Ef)
plot(xx, Ef-2*std_f, 'r--')
plot(x,y,'.')
legend('real f', 'Ef', 'Ef+std(f)','y')
plot(xx, Ef+2*std_f, 'r--')
title(sprintf('The predictions and the data points (MAP solution, Student-t (nu=%.2f,sigma=%.3f) noise)',gp.likelih.nu, gp.likelih.sigma));
S4 = sprintf('lengt-scale: %.3f, magnSigma2: %.3f \n', gp.cf{1}.lengthScale, gp.cf{1}.magnSigma2)


% ========================================
% Comparing theconditional posterior distributions of the latent 
% variables from MCMC and Laplace approach
%
% Here, we compare the Laplace approximation of p(f|theta,y)
% to the MCMC samples from p(f|theta,y). That is all the 
% hyperparameters are fixed
%  The posterior means of the parameters are:
%     nu    = 1.5
%     sigma = 0.07
%     lengthScale = 1.02
%     magnSigma2  = 1.29
%
% ===================================================

% ========================================
% The MCMC for latent values
% ========================================

% load the data. First 100 variables are for training
% and last 100 for test
S = which('demo_noiset');
L = strrep(S,'demo_noiset.m','demos/odata');
x = load(L);
xt = x(101:end,1);
yt = x(101:end,2);
y = x(1:100,2);
x = x(1:100,1);
[n, nin] = size(x); 

% Test data
xx = [-2.7:0.01:2.7];
yy = 0.3+0.4*xx+0.5*sin(2.7*xx)+1.1./(1+xx.^2);

gpcf1 = gpcf_sexp('init', nin, 'lengthScale', 1.02, 'magnSigma2', 1.29);

% ... Then set the prior for the parameters of covariance functions...
gpcf1.p.lengthScale = gamma_p({3 7});  
gpcf1.p.magnSigma2 = sinvchi2_p({0.05^2 0.5});

% Create the likelihood structure
likelih = likelih_t('init', 1.5, 0.07);
likelih.p.nu = loglogunif_p;
likelih.p.sigma = logunif_p;

% the GP data structure
gp = gp_init('init', 'FULL', nin, likelih, {gpcf1}, {}, 'jitterSigmas', 0.0001);
gp = gp_init('set', gp, 'latent_method', {'MCMC', zeros(size(y))'});

opt=gp_mcopt;
    
% HMC-latent
opt.latent_opt.nsamples=1;
opt.latent_opt.nomit=0;
opt.latent_opt.persistence=0;
opt.latent_opt.repeat=20;
opt.latent_opt.steps=20;
opt.latent_opt.stepadj=0.15;
opt.latent_opt.window=5;

% Now we reset the sampling parameters to 
% achieve faster sampling
opt.latent_opt.repeat=20;
% $$$ opt.latent_opt.steps=7;
% $$$ opt.latent_opt.window=1;
% $$$ opt.latent_opt.stepadj=0.3;
opt.display = 1;
opt.latent_opt.display=0;
opt.nsamples=5000;

% Conduct the actual sampling.
[rgp,gp,opt]=gp_mc(opt, gp, x, y);
%[rgp,gp,opt]=gp_mc(opt, gp, x, y, [], [], rgp);

% thin the record
rr = thin(rgp,10,2);

[Ef_samp, Varf_samp] = gp_preds(rr, x, rr.latentValues', xx');
Ef = mean(squeeze(Ef_samp),2);
std_samp = std(Ef_samp,[],2);

figure
plot(xx,yy,'k')
hold on
plot(xx,Ef)
plot(xx, Ef-2*std_samp, 'r--')
plot(x,y,'.')
legend('real f', 'mean', '2xstd(f)', 'data')
plot(xx, Ef+2*std_samp, 'r--')
title('The predictions and the data points (MAP solution and Student-t noise)')
S1 = sprintf('lengt-scale: %.3f, magnSigma2: %.3f \n', mean(rr.cf{1}.lengthScale), mean(rr.cf{1}.magnSigma2))

% ========================================
% Laplace approximation Student-t likelihood
% ========================================

% ... the GP data structure
gp_la = gp_init('init', 'FULL', nin, likelih, {gpcf1}, {}, 'jitterSigmas', 0.01);
gp_la = gp_init('set', gp_la, 'latent_method', {'Laplace', x, y, 'hyper'});

[Ef_la, Varf_la] = la_pred(gp_la, x, y, xx', 'hyper');
std_la = sqrt(Varf_la);

% Plot the prediction and data
figure
plot(xx,yy,'k')
hold on
plot(xx,Ef_la)
plot(xx, Ef_la-2*std_la, 'r--')
plot(x,y,'.')
legend('real f', 'Ef', 'Ef+std(f)','y')
plot(xx, Ef_la+2*std_la, 'r--')
title(sprintf('The predictions and the data points (MAP solution, Student-t (nu=%.2f,sigma=%.3f) noise)',gp.likelih.nu, gp.likelih.sigma));
S2 = sprintf('lengt-scale: %.3f, magnSigma2: %.3f \n', gp.cf{1}.lengthScale, gp.cf{1}.magnSigma2)

ff_samp = rr.latentValues';
[Ef_dat, Varf_dat] = la_pred(gp_la, x, y, x, 'hyper');
for j=0:4
    figure
    k=1;
    for i = j*20+1:j*20+20
        ff = [Ef_dat(i)-3*sqrt(Varf_dat(i)):0.01:Ef_dat(i)+3*sqrt(Varf_dat(i))];
        pdff = normpdf(ff, Ef_dat(i), sqrt(Varf_dat(i)));
        subplot(5,4,k)
        hist(ff_samp(i,:),30);
        h = hist(ff_samp(i,:),30);    
        hold on 
        plot(ff, max(h).*pdff./max(pdff), 'r', 'linewidth', 2)
        plot(y(i),1,'rx', 'Markersize',10, 'Linewidth',3)
        k=k+1;
        title(sprintf('# %d', i))
    end
end
