function demo_clFIC
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
S = which('demo_clFIC');
L = strrep(S,'demo_clFIC.m','demos/synth.tr');
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
gp = gp_init('init', 'FIC', nin, 'logistic', {gpcf1}, [], 'jitterSigmas', 0.01)   %{gpcf2}
gp = gp_init('set', gp, 'latent_method', {'MCMC', @latent_mh, randn(size(y))'});

% Set the inducing inputs
[u1,u2]=meshgrid(linspace(-1.25, 0.9,6),linspace(-0.2, 1.1,6));
U=[u1(:) u2(:)];
U = U([3 4 7:18 20:24 26:30 33:36],:);
plot(x(:,1), x(:,2),'*'), hold on
plot(U(:,1), U(:,2), 'kX', 'MarkerSize', 12, 'LineWidth', 2)

gp = gp_init('set', gp, 'X_u', U);

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
% $$$ w=randn(size(gp_pak(gp,'hyper')))*0.01;
% $$$ 
% $$$ fe=str2fun('gp_e');
% $$$ fg=str2fun('gp_g');
% $$$ n=length(y);
% $$$ itr=1:floor(0.5*n);     % training set of data for early stop
% $$$ its=floor(0.5*n)+1:n;   % test set of data for early stop
% $$$ optes=scges_opt;
% $$$ optes.display=1;
% $$$ optes.tolfun=1e-1;
% $$$ optes.tolx=1e-1;
% $$$ 
% $$$ % do scaled conjugate gradient optimization with early stopping.
% $$$ [w,fs,vs]=scges(fe, w, optes, fg, gp, x(itr,:),y(itr,:),'hyper', gp,x(its,:),y(its,:),'hyper');
% $$$ gp=gp_unpak(gp,w,'hyper');

disp(' ')
disp(' Now that the starting values are found we set the main sampling ')
disp(' options and define the latent values. ')
disp(' ')


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
opt.nsamples=100;
opt.repeat=3;
opt.hmc_opt.steps=1;
opt.hmc_opt.stepadj=0.001;
opt.latent_opt.repeat = 3;
opt.hmc_opt.stepadj=0.1;
opt.hmc_opt.nsamples=1;
hmc2('state', sum(100*clock));

% Here we would do the main sampling. In order to save time we have
% saved one GP record structure in the software. The record (and though 
% the samples) are loaded and used in the demo. In order to do your own 
% sampling uncomment the line below.

% Sample 
[r,g,rstate2]=gp_mc(opt, gp, x, y, [], [], r);

% $$$ % Load a saved record structure
% $$$ L = strrep(S,'demo_2classgp.m','demos/2classgprecord');
% $$$ load(L)

%rr=thin(r,50,8);
rr=thin(r,20,2);

disp(' ')
disp(' For last the decision line and the training points are ')
disp(' drawn in the same plot. ')
disp(' ')

% Draw the decision line and training points in the same plot
[p1,p2]=meshgrid(-1.3:0.05:1.1,-0.5:0.05:1.1);
p=[p1(:) p2(:)];

tms2=mean((logsig(gp_fwds(rr, x, rr.latentValues', p))),3);

%Plot the decision line
gp=zeros(size(p1));
gp(:)=tms2;
contour(p1,p2,gp,[0.5 0.5],'k');

hold on;
% plot the train data o=0, x=1
plot(x(y==0,1),x(y==0,2),'o');
plot(x(y==1,1),x(y==1,2),'x');
hold off;

% test how well the network works for the test data. 
L = strrep(S,'demo_clFIC.m','demos/synth.ts');
tx=load(L);
ty=tx(:,end);
tx(:,end)=[];

tga=mean(logsig(gp_fwds(r, x, r.latentValues', tx)),3);

% calculate the percentage of misclassified points
missed = sum(abs(round(tga)-ty))/size(ty,1)*100;