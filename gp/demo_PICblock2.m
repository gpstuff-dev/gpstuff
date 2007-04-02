function demo_PICblock2
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
S = which('demo_gpregr');
L = strrep(S,'demo_gpregr.m','demos/dat.1');
data=load(L);
x = [data(:,1) data(:,2)];
y = data(:,3);
[n, nin] = size(x);

% Create covariance functions
gpcf1 = gpcf_matern52('init', nin, 'lengthScale', 1, 'magnSigma2', 0.2^2);
gpcf2 = gpcf_noise('init', nin, 'noiseSigmas2', 0.2^2);

% Set the prior for the parameters of covariance functions 
gpcf2.p.noiseSigmas2 = sinvchi2_p({0.05^2 0.5});    % MUUTA tässä invgam_p saman näköiseksi kuin gpcf_sexp('set'...)
gpcf1.p.lengthScale = gamma_p({3 7});  
gpcf1.p.magnSigma2 = sinvchi2_p({0.05^2 0.5});

% sparse model. Set the inducing points to the GP
gp = gp_init('init', nin, 'regr', {gpcf1}, {gpcf2}, 'sparse', 'FIC', 'jitterSigmas', 1);
U = x(1:4:end,:);

% find starting point using scaled conjucate gradient algorithm
% Intialize weights to zero and set the optimization parameters
w=randn(size(gp_pak(gp)))*0.01;

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
% sparse model
[w,fs,vs]=scges(fe, w, optes, fg, gp, x(itr,:),y(itr,:), U, gp,x(its,:),y(its,:), U);
gp=gp_unpak(gp,w);
% full model
w=randn(size(gp_pak(gp2)))*0.01;
[w,fs,vs]=scges(fe, w, optes, fg, gp2, x(itr,:),y(itr,:), gp2,x(its,:),y(its,:));
gp2=gp_unpak(gp2,w);

opt=gp_mcopt;
opt.repeat=1;
opt.nsamples=100;
opt.hmc_opt.steps=10;
opt.hmc_opt.stepadj=0.1;
opt.hmc_opt.nsamples=1;
opt.hmc_opt.window=1;
hmc2('state', sum(100*clock));

% Sample sparse model
t = cputime;
[r,g,rstate2]=gp_mc(opt, gp, x, y, [], [], [], U);
tsparse = cputime - t;

% New input
[p1,p2]=meshgrid(-1.8:0.05:1.8,-1.8:0.05:1.8);
p=[p1(:) p2(:)];

% The predictions for the new inputs of sparse model
rr=thin(r,10,2);
yn = gp_fwds(rr, x, y, p, U);
Ey = mean(squeeze(yn)');

pred = zeros(size(p1));
pred(:)=Ey;
figure
mesh(p1,p2,pred);
qc=caxis;
title('sparse')
