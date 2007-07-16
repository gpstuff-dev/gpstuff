function demo_clFull
%DEMO_GP2CLASS    Classification problem demonstration for 2
%                 classes. 
%
%      Description
%      The demonstration program is based on synthetic two 
%      class data used by B.D. Ripley (Pattern Regocnition and
%      Neural Networks, 1996}. The data consists of 2-dimensional
%      vectors that are divided into to classes, labeled 0 or 1.
%      Each class has a bimodal distribution generated from equal
%      mixtures of Gaussian distributions with identical covariance
%      matrices. A Bayesian aprouch is used to find the decision
%      line and predict the classes of new data points.
%
%      The demonstration program does not sample for real, because
%      it would require so much time. The main sampling state is
%      commented out from the program and instead a saved network
%      structure is loaded and used to make predictions (see lines
%      143-146).
%

% Copyright (c) 2005 Jarno Vanhatalo, Aki Vehtari 

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

% This demonstration is based on the dataset used in the book Pattern Recognition and
% Neural Networks by B.D. Ripley (1996) Cambridge University Press ISBN 0 521
% 46986 7

disp(' ')
disp(' The demonstration program is based on synthetic two ')
disp(' class data used by B.D. Ripley (Pattern Regocnition and')
disp(' Neural Networks, 1996}. The data consists of 2-dimensional')
disp(' vectors that are divided into to classes, labeled 0 or 1.')
disp(' Each class has a bimodal distribution generated from equal')
disp(' mixtures of Gaussian distributions with identical covariance')
disp(' matrices. A Gaussian process is used to find the decision')
disp(' line and predict the classes of new data points.')
disp(' ')
disp(' ')

  
% Load the data
S = which('demo_clFull');
L = strrep(S,'demo_clFull.m','demos/synth.tr');
x=load(L);
y=x(:,end);
x(:,end)=[];
[n, nin] = size(x);

disp(' ')
disp(' First we create a Gaussian process for classification problem. ')
disp(' A Gaussian multivariate hierarchical prior with ARD is created')
disp(' for GP. ')
disp(' ')

% Create covariance functions
% Create covariance functions
gpcf1 = gpcf_sexp('init', nin, 'lengthScale', 1, 'magnSigma2', 0.2^2);
%gpcf1 = gpcf_sexp('init', nin, 'lengthScale', [1 1], 'magnSigma2', 0.2^2);

%gpcf1 = gpcf_exp('init', nin, 'lengthScale', 1, 'magnSigma2', 0.2^2);
%gpcf1 = gpcf_exp('init', nin, 'lengthScale', [1 1], 'magnSigma2', 0.2^2);

%gpcf1 = gpcf_matern32('init', nin, 'lengthScale', 1, 'magnSigma2', 0.2^2);
%gpcf1 = gpcf_matern32('init', nin, 'lengthScale', [1 1], 'magnSigma2', 0.2^2);

%gpcf1 = gpcf_matern52('init', nin, 'lengthScale', 1, 'magnSigma2', 0.2^2);
%gpcf1 = gpcf_matern52('init', nin, 'lengthScale', [1 1], 'magnSigma2', 0.2^2);

% Set the prior for the parameters of covariance functions 
gpcf1.p.lengthScale = gamma_p({3 7 3 7});
gpcf1.p.magnSigma2 = sinvchi2_p({0.05^2 0.5});

%gp = gp_init('init', nin, 'lh_2class', {gpcf1}, [], 'jitterSigmas', 1)   %{gpcf2}
gp = gp_init('init', 'FULL', nin, 'logistic', {gpcf1}, [], 'jitterSigmas', 0.01)   %{gpcf2}
gp = gp_init('set', gp, 'latent_method', {'MCMC', @latent_mh, randn(size(y))'});


disp(' ')
disp(' The starting values for sampling the parameters are found with early ')
disp(' stop method. This is a quick way to get better starting point ')
disp(' for the Markov chain.') 
disp(' ')
% See Vehtari et al (2000). On MCMC sampling in Bayesian MLP neural networks.
% In Proc. IJCNN'2000.
%
% <http://www.lce.hut.fi/publications/pdf/VehtariEtAl_ijcnn2000.pdf>

% Intialize weights to zero and set the optimization parameters...

opt=gp_mcopt;
opt.repeat=15;
opt.nsamples=1;
opt.hmc_opt.steps=10;
opt.hmc_opt.stepadj=0.1;
opt.hmc_opt.nsamples=1;


opt.latent_opt.display=0;
opt.latent_opt.repeat = 20;
opt.latent_opt.sample_latent_scale = 0.5;
hmc2('state', sum(100*clock))

[r,g,rstate1]=gp_mc(opt, gp, x, y);

% Set the sampling options
opt.nsamples=50;
opt.repeat=1;
opt.hmc_opt.steps=2;
opt.hmc_opt.stepadj=0.001;
opt.latent_opt.repeat = 5;
opt.hmc_opt.stepadj=0.1;
opt.hmc_opt.nsamples=1;
hmc2('state', sum(100*clock));

% Here we would do the main sampling. In order to save time we have
% saved one GP record structure in the software. The record (and though 
% the samples) are loaded and used in the demo. In order to do your own 
% sampling uncomment the line below.

% Sample 
t0 = cputime;
[r,g,rstate2]=gp_mc(opt, gp, x, y, [], [], r);
rr=r;
p1 = mean((logsig(gp_fwds(rr, x, rr.latentValues', xstar))),3);
cputime - t0

% $$$ % Load a saved record structure
% $$$ L = strrep(S,'demo_2classgp.m','demos/2classgprecord');
% $$$ load(L)

%rr=thin(r,50,8);
rr=thin(r,20,2);

% Print the hyperparameter values
fprintf(' The mean of the length-scale is: %.3f \n The magnitude mean of the sigma is: %.3f \n', ...
        mean(rr.cf{1}.lengthScale), mean(rr.cf{1}.magnSigma2))


% [mean(rr.cf{1}.lengthScale), mean(rr.cf{1}.magnSigma2)]

disp(' ')
disp(' For last the decision line and the training points are ')
disp(' drawn in the same plot. ')
disp(' ')

% Draw the decision line and training points in the same plot
xt1=repmat(linspace(min(x(:,1)),max(x(:,1)),20)',1,20);
xt2=repmat(linspace(min(x(:,2)),max(x(:,2)),20)',1,20)';
xstar=[xt1(:) xt2(:)];

p1 = mean((logsig(gp_fwds(rr, x, rr.latentValues', xstar))),3);

% visualise predictive probability  p(ystar = 1)
figure, hold on;
n_pred=size(xstar,1);
h1=pcolor(reshape(xstar(:,1),20,20),reshape(xstar(:,2),20,20),reshape(p1,20,20))
set(h1, 'edgealpha', 0), set(h1, 'facecolor', 'interp')
colormap(repmat(linspace(1,0,64)', 1, 3).*repmat(ones(1,3), 64,1))
axis([-inf inf -inf inf]), axis off
plot(x(y==0,1),x(y==0,2),'o', 'markersize', 8, 'linewidth', 2);
plot(x(y==1,1),x(y==1,2),'rx', 'markersize', 8, 'linewidth', 2);
set(gcf, 'color', 'w'), title('predictive probability and training cases', 'fontsize', 14)

% $$$ % Visualize the predictive variance
% $$$ figure, hold on;
% $$$ n_pred=size(xstar,1);
% $$$ h1=pcolor(reshape(xstar(:,1),20,20),reshape(xstar(:,2),20,20),reshape(Varf,20,20))
% $$$ set(h1, 'edgealpha', 0), set(h1, 'facecolor', 'interp')
% $$$ colormap(repmat(linspace(1,0,64)', 1, 3).*repmat(ones(1,3), 64,1))
% $$$ axis([-inf inf -inf inf]), axis off
% $$$ plot(x(y==-1,1),x(y==-1,2),'o', 'markersize', 8, 'linewidth', 2);
% $$$ plot(x(y==1,1),x(y==1,2),'rx', 'markersize', 8, 'linewidth', 2);
% $$$ set(gcf, 'color', 'w'), title('marginal predictive latent variance', 'fontsize', 14)


% visualise predictive probability  p(ystar = 1) with contours
figure, hold on
[cs,h]=contour(reshape(xstar(:,1),20,20),reshape(xstar(:,2),20,20),reshape(p1,20,20),[0.025 0.25 0.5 0.75 0.975], 'linewidth', 3);
text_handle = clabel(cs,h);
set(text_handle,'BackgroundColor',[1 1 .6],'Edgecolor',[.7 .7 .7],'linewidth', 2, 'fontsize',14)
c1=[linspace(0,1,64)' 0*ones(64,1) linspace(1,0,64)'];
colormap(c1)
plot(x(y==1,1), x(y==1,2), 'rx', 'markersize', 8, 'linewidth', 2),
plot(x(y==0,1), x(y==0,2), 'bo', 'markersize', 8, 'linewidth', 2)
plot(xstar(:,1), xstar(:,2), 'k.'), axis([-inf inf -inf inf]), axis off
set(gcf, 'color', 'w'), title('predictive probability contours', 'fontsize', 14)


% test how well the network works for the test data. 
L = strrep(S,'demo_clFull.m','demos/synth.ts');
tx=load(L);
ty=tx(:,end);
tx(:,end)=[];

tga=mean(logsig(gp_fwds(r, x, r.latentValues', tx)),3);

% calculate the percentage of misclassified points
missed = sum(abs(round(tga)-ty))/size(ty,1)*100

% Plot the training and test cases in the same figure
figure, hold on;
set(text_handle,'BackgroundColor',[1 1 .6],'Edgecolor',[.7 .7 .7],'linewidth', 2, 'fontsize',14)
c1=[linspace(0,1,64)' 0*ones(64,1) linspace(1,0,64)'];
colormap(c1)
plot(x(y==-1,1),x(y==-1,2),'o', 'markersize', 8, 'linewidth', 2);
plot(x(y==1,1),x(y==1,2),'rx', 'markersize', 8, 'linewidth', 2);
plot(tx(ty==0,1),tx(ty==0,2),'go', 'markersize', 8, 'linewidth', 2);
plot(tx(ty==1,1),tx(ty==1,2),'cx', 'markersize', 8, 'linewidth', 2);
plot(xstar(:,1), xstar(:,2), 'k.'), axis([-inf inf -inf inf]), axis off
set(gcf, 'color', 'w'), title('training and test cases', 'fontsize', 14)

% visualise predictive probability  p(ystar = 1) with the test cases
figure, hold on;
n_pred=size(xstar,1);
h1=pcolor(reshape(xstar(:,1),20,20),reshape(xstar(:,2),20,20),reshape(p1,20,20))
set(h1, 'edgealpha', 0), set(h1, 'facecolor', 'interp')
colormap(repmat(linspace(1,0,64)', 1, 3).*repmat(ones(1,3), 64,1))
axis([-inf inf -inf inf]), axis off
plot(tx(ty==0,1),tx(ty==0,2),'o', 'markersize', 8, 'linewidth', 2);
plot(tx(ty==1,1),tx(ty==1,2),'rx', 'markersize', 8, 'linewidth', 2);
set(gcf, 'color', 'w'), title('predictive probability and test cases', 'fontsize', 14)
