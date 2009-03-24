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
% ========================================
[n, nin] = size(x);
gpcf1 = gpcf_sexp('init', nin, 'lengthScale', repmat(1,1,nin), 'magnSigma2', 0.2^2);
gpcf2 = gpcf_noiset('init', nin, n, 'noiseSigmas2', repmat(1^2,n,1));   % Here set own Sigma2 for every data point

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
opt.nsamples=1500;
opt.hmc_opt.steps=10;
opt.hmc_opt.persistence=1;
opt.hmc_opt.decay=0.6;

[r,g,rstate2]=gp_mc(opt, g, x, y, [], [], r);
rr = r;

% thin the record
rr = thin(r,10,2);

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
std = sqrt(mean(squeeze(Varf),2) );

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

save('noiset_samples', 'rr', 'r')

% ========================================
% Laplace approximation Student-t likelihood
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

% $$$ w = randn(size(y'));
% $$$ gradcheck(w, likelih.fh_e, likelih.fh_g, likelih, y', 'latent')
% $$$ 
% $$$ w = randn(size(y'));
% $$$ gradcheck(w, likelih.fh_g, likelih.fh_hessian, likelih, y', 'latent')
% $$$ 
% $$$ w = randn(size(y'));
% $$$ gradcheck(w, likelih.fh_hessian, likelih.fh_g3, likelih, y', 'latent')

% $$$ w = randn(size(gp_pak(gp,'likelih')));
% $$$ gradcheck(w, likelih.fh_e, likelih.fh_g, likelih, y, Ef, 'hyper')
% $$$ 
% $$$ w = randn(size(gp_pak(gp,'likelih')));
% $$$ gradcheck(w, likelih.fh_g, likelih.fh_hessian, likelih, y, Ef, 'latent+hyper')
% $$$ 
% $$$ w = randn(size(gp_pak(gp,'likelih')));
% $$$ gradcheck(w, likelih.fh_hessian, likelih.fh_g3, likelih, y, Ef, 'latent2+hyper')

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


% Set the optimized hyperparameter values back to the gp structure
gp=gp_unpak(gp,w, param);

[Ef, Varf] = la_pred(gp, x, y, xx', 'hyper');
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


[Ef_samp, Varf_samp] = gp_preds(rr,x,y,x);
[Ef, Varf] = la_pred(gp, x, y, x, 'hyper');
for j=0:4
    figure
    k=1;
    for i = j*20+1:j*20+20
        ff = [Ef(i)-3*sqrt(Varf(i)):0.01:Ef(i)+3*sqrt(Varf(i))];
        pdff = normpdf(ff, Ef(i), sqrt(Varf(i)));
        subplot(5,4,k)
        hist(Ef_samp(i,:),30);
        h = hist(Ef_samp(i,:),30);    
        hold on 
        plot(ff, max(h).*pdff./max(pdff))
        plot(y(i),1,'rx', 'Markersize',10, 'Linewidth',3)
        k=k+1;
    end
end


ff = la_pred(gp, x, y, x, 'hyper');
K = gp_trcov(gp, x);

f = [-1.8:0.01:2.5];
dat = 4;
for i=1:length(f)
    ff(dat) = f(i);
    
    eprior(i) = -0.5*log(2*pi) - sum(log(diag(chol(K)))) - 0.5*ff'*(K\ff);
    temp = K\ff;
    gprior(i) = - temp(dat);
    temp = inv(K);
    hprior(i) = - temp(dat,dat);
    
    ee(i) = feval(likelih.fh_e, likelih, y(dat), f(i), 'latent');
    gg(i) = feval(likelih.fh_g, likelih, y(dat), f(i), 'latent');
    hh(i) = feval(likelih.fh_hessian, likelih, y(dat), f(i), 'latent');
end

figure
subplot(3,1,1)
plot(f,ee)
subplot(3,1,2)
plot(f,gg)
subplot(3,1,3)
plot(f,hh)

figure
subplot(3,1,1)
plot(f,ee+eprior)
subplot(3,1,2)
plot(f,gg+gprior)
subplot(3,1,3)
plot(f,hh+hprior)



w = randn(size(y'));
gradcheck(w, likelih.fh_e, likelih.fh_g, likelih, y', 'latent')

w = randn(size(y'));
gradcheck(w, likelih.fh_g, likelih.fh_hessian, likelih, y', 'latent')

w = randn(size(y'));
gradcheck(w, likelih.fh_hessian, likelih.fh_g3, likelih, y', 'latent')


% ========================================
% MCMC approach Student-t likelihood
% ========================================
gpcf1 = gpcf_sexp('init', nin, 'lengthScale', 1, 'magnSigma2', 0.2^2);

% ... Then set the prior for the parameters of covariance functions...
gpcf1.p.lengthScale = gamma_p({3 7});  
gpcf1.p.magnSigma2 = sinvchi2_p({0.05^2 0.5});

% Create the likelihood structure
likelih = likelih_t('init', 4, 2);
likelih.p.nu = loglogunif_p;
likelih.p.sigma = logunif_p;


% ... Finally create the GP data structure
gp = gp_init('init', 'FULL', nin, likelih, {gpcf1}, {}, 'jitterSigmas', 0.0001);
gp = gp_init('set', gp, 'latent_method', {'MCMC', zeros(size(y))'});

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

% SLS-likelihood
opt.nb_sls_opt = sls_opt;
opt.nb_sls_opt.maxiter = 200;
opt.nb_sls_opt.mmlimits = [-5;5];

% Here we make an initialization with 
% slow sampling parameters
opt.display = 0;
[rgp,gp,opt]=gp_mc(opt, gp, x, y);

% Now we reset the sampling parameters to 
% achieve faster sampling
opt.latent_opt.repeat=20;
% $$$ opt.latent_opt.steps=7;
% $$$ opt.latent_opt.window=1;
% $$$ opt.latent_opt.stepadj=0.3;
opt.hmc_opt.persistence=0;
opt.hmc_opt.stepadj=0.05;
opt.hmc_opt.steps=4;

opt.display = 1;
opt.hmc_opt.display = 0;
opt.latent_opt.display=0;
opt.nsamples=200;

% Conduct the actual sampling.
[rgp,gp,opt]=gp_mc(opt, gp, x, y, [], [], rgp);

% thin the record
rr = thin(rgp,10,2);

figure 
hist(rr.likelih.nu,20)
title('Student-t, \nu')
figure 
hist(rr.likelih.sigma,20)
title('Student-t, \sigma')
figure 
hist(rr.cf{1}.lengthScale,20)
title('Student-t, length-scale')
figure 
hist(rr.cf{1}.magnSigma2,20)
title('Student-t, magnSigma2')

[Ef1, Varf1] = gp_preds(rr, x, rr.latentValues', xx');
Ef = mean(squeeze(Ef1),2);
std = sqrt(mean(squeeze(Varf1),2) );

figure
plot(xx,yy,'k')
hold on
plot(x,y,'.')
plot(xx,Ef)
plot(xx, Ef-2*std_f, 'r--')
plot(xx, Ef+2*std_f, 'r--')
legend('real f', 'mean', '2xstd(f)')
title('The predictions and the data points (MAP solution and Student-t noise)')

S3 = sprintf('lengt-scale: %.3f, magnSigma2: %.3f \n', mean(rr.cf{1}.lengthScale), mean(rr.cf{1}.magnSigma2))

S
S2 
S3

figure
hold on
[p, I] = sort(x);
for i = 1:length(rr.hmcrejects)
    plot(x(I),rr.latentValues(i,I)')
end
plot(x(I),mean(rr.latentValues(:,I))', 'r', 'LineWidth', 2)
plot(x,y,'r.')
title('The latent values and the data points (MAP solution and Student-t noise)')




% =====================================================
% EP approach Student-t (nu=4, sigma = 1) likelihood
% =====================================================

% =====================================================
% DOES NOT WORK (YET)!
% =====================================================
gpcf1 = gpcf_sexp('init', nin, 'lengthScale', 1, 'magnSigma2', 0.2^2);

% ... Then set the prior for the parameters of covariance functions...
gpcf1.p.lengthScale = gamma_p({3 7});  
gpcf1.p.magnSigma2 = sinvchi2_p({0.05^2 0.5});

% Create the likelihood structure
likelih = likelih_t('init', 4, 1);


% ... Finally create the GP data structure
gp = gp_init('init', 'FULL', nin, likelih, {gpcf1}, {}, 'jitterSigmas', 0.0001);
gp = gp_init('set', gp, 'latent_method', {'EP', x, y, 'hyper'});

gradcheck(gp_pak(gp,'hyper'), @gpep_e, @gpep_g, gp, x, y, 'hyper')

w=gp_pak(gp, 'hyper');  % pack the hyperparameters into one vector
fe=str2fun('gpep_e');     % create a function handle to negative log posterior
fg=str2fun('gpep_g');     % create a function handle to gradient of negative log posterior

% set the options
opt = [];
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

[Ef, Varf] = ep_pred(gp, x, y, xx', 'hyper');
std_f = sqrt(Varf);

% Plot the prediction and data
figure
plot(xx,yy,'k')
hold on
plot(xx, Ef)
plot(xx, Ef-2*std_f, 'r--')
plot(xx, Ef+2*std_f, 'r--')
plot(xt,yt,'r.')
plot(x,y,'b.')
axis on;
title('The predictions and the data points (MAP solution, Student-t (nu=4,sigma=1) noise)');
S4 = sprintf('lengt-scale: %.3f, magnSigma2: %.3f \n', gp.cf{1}.lengthScale, gp.cf{1}.magnSigma2)


% ================================================================
% EP approach Student-t likelihood (all parameters are optimized)
% ================================================================
gpcf1 = gpcf_sexp('init', nin, 'lengthScale', 1, 'magnSigma2', 0.2^2);

% ... Then set the prior for the parameters of covariance functions...
gpcf1.p.lengthScale = gamma_p({3 7});  
gpcf1.p.magnSigma2 = sinvchi2_p({0.05^2 0.5});

% Create the likelihood structure and set priors for the parameters
likelih = likelih_t('init', 3, 3);
likelih.p.nu = loglogunif_p;
likelih.p.sigma = logunif_p;

% ... Finally create the GP data structure
gp = gp_init('init', 'FULL', nin, likelih, {gpcf1}, {}, 'jitterSigmas', 0.0001);
gp = gp_init('set', gp, 'latent_method', {'EP', x, y, 'hyper+likelih'});

gradcheck(gp_pak(gp,'hyper+likelih'), @gpep_e, @gpep_g, gp, x, y, 'hyper+likelih')

w=gp_pak(gp, 'hyper+likelih');  % pack the hyperparameters into one vector
fe=str2fun('gpep_e');     % create a function handle to negative log posterior
fg=str2fun('gpep_g');     % create a function handle to gradient of negative log posterior

% set the options
opt = [];
opt(1) = 1;
opt(2) = 1e-3;
opt(3) = 3e-3;
opt(9) = 0;
opt(10) = 0;
opt(11) = 0;
opt(14) = 0;

% do the optimization
[w, opt, flog]=scg(fe, w, opt, fg, gp, x, y, 'hyper+likelih');
 
% Set the optimized hyperparameter values back to the gp structure
gp=gp_unpak(gp,w, 'hyper+likelih');

[Ef, Varf] = ep_pred(gp, x, y, xx', 'hyper+likelih');
std_f = sqrt(Varf);

% Plot the prediction and data
figure(1)
plot(xx, Ef)
hold on
plot(xx, Ef-2*std_f, 'r--')
plot(xx, Ef+2*std_f, 'r--')
plot(xt,yt,'r.')
plot(x,y,'b.')
axis on;
title('The predictions and the data points (MAP solution, Student-t noise)');
S5 = sprintf('lengt-scale: %.3f, magnSigma2: %.3f \n', gp.cf{1}.lengthScale, gp.cf{1}.magnSigma2)




















r = 0.5;
nu = 4;
t2 = 0.2;
alpha = 1;



U=sinvchi2rand(nu+1, (nu.*t2+(r./alpha).^2)./(nu+1),10000);
U2=invgamrand((nu.*t2+(r./alpha).^2)./(nu+1),nu+1,1,10000);
plot(sort(U),sort(U2'),'.'); line([0 4], [0 4])

r = r.noise{1}.r(2,:);
nu = rsinv.noise{1}.nu(1);
t2 = rsinv.noise{1}.tau2(1);
alpha = rsinv.noise{1}.alpha(1);



rsinv = rr;
rinvgam = r;





mydeal = @(varargin) varargin{1:nargout};

mydeal = @(e1, e2) e1 + e2
    
mydeal = @(varargout) sum(varargout);




e = @(x) exp(exp(x))
g = @(x) exp(exp(x)+x)