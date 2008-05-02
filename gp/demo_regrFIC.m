function demo_regrFIC
%DEMO_GPREGR    Regression problem demonstration for 2-input 
%              function with Gaussian process
%
%    Description
%    The problem consist of a data with two input variables
%    and one output variable with Gaussian noise. 
%

% Copyright (c) 2005-2006 Jarno Vanhatalo, Aki Vehtari 

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

% Load the data
S = which('demo_regrFIC');
L = strrep(S,'demo_regrFIC.m','demos/dat.1');
data=load(L);
x = [data(:,1) data(:,2)];
y = data(:,3);
[n, nin] = size(x);

% Create covariance functions
gpcf1 = gpcf_sexp('init', nin, 'lengthScale', 1, 'magnSigma2', 0.2^2);
%gpcf1 = gpcf_sexp('init', nin, 'lengthScale', [1 1], 'magnSigma2', 0.2^2);

%gpcf1 = gpcf_exp('init', nin, 'lengthScale', 1, 'magnSigma2', 0.2^2);
%gpcf1 = gpcf_exp('init', nin, 'lengthScale', [1 1], 'magnSigma2', 0.2^2);

%gpcf1 = gpcf_matern32('init', nin, 'lengthScale', 1, 'magnSigma2', 0.2^2);
%gpcf1 = gpcf_matern32('init', nin, 'lengthScale', [1 1], 'magnSigma2', 0.2^2);

%gpcf1 = gpcf_matern52('init', nin, 'lengthScale', 1, 'magnSigma2', 0.2^2);
%gpcf1 = gpcf_matern52('init', nin, 'lengthScale', [1 1], 'magnSigma2', 0.2^2);

gpcf2 = gpcf_noise('init', nin, 'noiseSigmas2', 0.2^2);

% Set the prior for the parameters of covariance functions 
gpcf2.p.noiseSigmas2 = sinvchi2_p({0.05^2 0.5});    % MUUTA tässä invgam_p saman näköiseksi kuin gpcf_sexp('set'...)
gpcf1.p.lengthScale = gamma_p({3 7});  
gpcf1.p.magnSigma2 = sinvchi2_p({0.05^2 0.5});

% sparse model. Set the inducing points to the GP
gp = gp_init('init', 'FIC', nin, 'regr', {gpcf1}, {gpcf2}, 'jitterSigmas', 0.001);
[u1,u2]=meshgrid(linspace(-1.8,1.8,6),linspace(-1.8,1.8,6));
U=[u1(:) u2(:)];
plot(u1(:), u2(:), 'kX', 'MarkerSize', 12, 'LineWidth', 2)

gp = gp_init('set', gp, 'X_u', U);

gradcheck(gp_pak(gp,'hyper'), @gp_e, @gp_g, gp, x, y, 'hyper')
gradcheck(gp_pak(gp,'inducing'), @gp_e, @gp_g, gp, x, y, 'inducing')
gradcheck(gp_pak(gp,'all'), @gp_e, @gp_g, gp, x, y, 'all')

% $$$ w=gp_pak(gp, 'hyper');
% $$$ [e, edata, eprior] = gp_e(w, gp, x, y, 'hyper')     % answer  488.9708 
% $$$ %w=randn(size(gp_pak(gp, 'all')))*0.01;
% $$$ 
% $$$ e =
% $$$   488.8794
% $$$ edata =
% $$$   484.1012
% $$$ eprior =
% $$$     4.7781

% find starting point using scaled conjucate gradient algorithm
% Intialize weights to zero and set the optimization parameters
w=gp_pak(gp, 'hyper');
fe=str2fun('gp_e');
fg=str2fun('gp_g');
n=length(y);
itr=1:floor(0.5*n);     % training set of data for early stop
its=floor(0.5*n)+1:n;   % test set of data for early stop
optes=scges_opt;
optes.display=1;
optes.tolfun=1e-1;
optes.tolx=1e-1;

% do scaled conjugate gradient optimization with early stopping.
% First for hyperparameters
[w,fs,vs,lambda]=scges(fe, w, optes, fg, gp, x(itr,:),y(itr,:), 'hyper', gp,x(its,:), y(its,:), 'hyper');
gp=gp_unpak(gp,w, 'hyper');

w=gp_pak(gp, 'inducing');
[w,fs,vs,lambda]=scges(fe, w, optes, fg, gp, x(itr,:),y(itr,:), 'inducing', gp,x(its,:), y(its,:), 'inducing');
gp=gp_unpak(gp,w, 'inducing');

figure
hold on
plot(gp.X_u(:,1),gp.X_u(:,2),'*')
axis([-2.5 2.5 -2.5 2.5])

opt=gp_mcopt;
opt.repeat=10;
opt.nsamples=300;

opt.hmc_opt.steps=5;
opt.hmc_opt.stepadj=0.01;
opt.hmc_opt.nsamples=1;
opt.hmc_opt.window=1;

% $$$ opt.inducing_opt.steps=4;
% $$$ opt.inducing_opt.stepadj=0.001;
% $$$ opt.inducing_opt.nsamples=1;
% $$$ opt.inducing_opt.window=1;
% $$$ opt.inducing_opt.persistence =0;

% Sample sparse model
t0 = cputime;
[r,g,opt]=gp_mc(opt, gp, x, y);
tsparse = cputime - t0;

% Evaluate the MSE for the predictions
out=gp_fwds(r, x, y, x);
mout = mean(squeeze(out)');
pred = zeros(size(x,1),1);
pred(:)=mout;

figure
title('The prediction');
[xi,yi,zi]=griddata(data(:,1),data(:,2),pred,-1.8:0.01:1.8,[-1.8:0.01:1.8]');
mesh(xi,yi,zi)

(pred-y)'*(pred-y)/length(y)

% New input
[p1,p2]=meshgrid(-1.8:0.05:1.8,-1.8:0.05:1.8);
p=[p1(:) p2(:)];

% The predictions for the new inputs of sparse model
yn = gp_fwds(r, x, y, p);

pred = zeros(size(p1));
pred(:)=mean(squeeze(yn)');
figure
mesh(p1,p2,pred);
qc=caxis;
title('sparse')

gp.cf{1}.lengthScale = [1.0640 1.0525];
gp.cf{1}.magnSigma2 = 2.0215;
gp.noise{1}.noiseSigmas2 = 0.027;
yn = gp_fwd(gp, x, y, p);
pred(:)=yn;
figure
mesh(p1,p2,pred);
qc=caxis;
title('sparse')
