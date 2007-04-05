function demo_FIC2
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
gpcf1 = gpcf_sexp('init', nin, 'lengthScale', [1, 1], 'magnSigma2', 0.2^2);
%gpcf1 = gpcf_exp('init', nin, 'lengthScale', [1, 1], 'magnSigma2', 0.2^2);
%gpcf1 = gpcf_matern32('init', nin, 'lengthScale', 1, 'magnSigma2', 0.2^2);
%gpcf1 = gpcf_matern52('init', nin, 'lengthScale', 1, 'magnSigma2', 0.2^2);
gpcf2 = gpcf_noise('init', nin, 'noiseSigmas2', 0.2^2);

% Set the prior for the parameters of covariance functions 
gpcf2.p.noiseSigmas2 = sinvchi2_p({0.05^2 0.5});    % MUUTA tässä invgam_p saman näköiseksi kuin gpcf_sexp('set'...)
gpcf1.p.lengthScale = gamma_p({3 7});  
gpcf1.p.magnSigma2 = sinvchi2_p({0.05^2 0.5});

% sparse model. Set the inducing points to the GP
gp = gp_init('init', 'FIC', nin, 'regr', {gpcf1}, {gpcf2}, 'jitterSigmas', 1);
% $$$ [u1,u2]=meshgrid(-1.8:1:1.8,-1.8:1:1.8);
% $$$ U=[u1(:) u2(:)];
U = randn(7,2);
gp = gp_init('set', gp, 'X_u', U);

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
[r,g,rstate2]=gp_mc(opt, gp, x, y);
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


















% full model
gp2 = gp_init('init', nin, 'regr', {gpcf1}, {gpcf2}, 'jitterSigmas', 1)

% Sample full model
t = cputime;
[r2,g2,rstate2]=gp_mc(opt, gp2, x, y);
tfull = cputime - t;

% The predictions for the new inputs of full model
rr2=thin(r2,1,1);
% $$$ yn2 = gp_fwds(rr2, x, y, p);
% $$$ Ey2 = mean(squeeze(yn2)');
Ey2 = gp_fwd(rr2, x, y, p);
pred(:)=Ey2;
figure
mesh(p1,p2,pred);
caxis(qc)
title('full')

% ----------------------
% Lets test how well "full" GP works if only as many training 
% points are used as there are inducing points
x = x(1:4:end,:);
y = y(1:4:end,:);

gp3 = gp_init('init', nin, 'regr', {gpcf1}, {gpcf2}, 'jitterSigmas', 1)

% find starting point using scaled conjucate gradient algorithm
% Intialize weights to zero and set the optimization parameters
w=randn(size(gp_pak(gp)))*0.01;

% do scaled conjugate gradient optimization with early stopping.
% "full" model
w=randn(size(gp_pak(gp3)))*0.01;
n=length(y);
itr=1:floor(0.5*n);     % training set of data for early stop
its=floor(0.5*n)+1:n;   % test set of data for early stop
[w,fs,vs]=scges(fe, w, optes, fg, gp3, x(itr,:),y(itr,:), gp3,x(its,:),y(its,:));
gp3=gp_unpak(gp3,w);

% Sample "full" model
opt.hmc_opt.stepadj=0.07;
t = cputime;
[r3,g3,rstate3]=gp_mc(opt, gp3, x, y);
tfull2 = cputime - t;

% The predictions for the new inputs of "full" model
rr3=thin(r3,1,1);
% $$$ yn3 = gp_fwds(rr3, x, y, p);
% $$$ Ey3 = mean(squeeze(yn3)');
Ey3 = gp_fwd(rr3, x, y, p);
pred(:)=Ey3;
figure
mesh(p1,p2,pred);
caxis(qc)
title('full with reduced training points')