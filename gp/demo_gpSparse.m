function demo_gpSparse
%DEMO_GPSparse    Regression problem demonstration for 2-input 
%              function with Gaussian process
%
%    Description
%    The problem consist of a data with two input variables
%    and one output variable with Gaussian noise. 
%

% Copyright (c) 2005 Jarno Vanhatalo, Aki Vehtari 

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.


% Create the data
x = rand(1,500)'*3.5-1.75;
y = 3*sin(3*x)+x.^3+cos(9.*x)+0.5*randn(size(x));

% Create covariance functions and Gaussian process
[n, nin] = size(x);
gpcf1 = gpcf_sexp('init', nin, 'lengthScale', repmat(0.7,1,nin), 'magnSigma2', 0.2^2);
gpcf2 = gpcf_noise('init', nin, 'noiseSigmas2', 0.2^2);

% Set the prior for the parameters of covariance functions 
gpcf2.p.noiseSigmas2 = sinvchi2_p({0.05^2 0.5});         % MUUTA tässä invgam_p saman näköiseksi kuin gpcf_sexp('set'...)
gpcf1.p.lengthScale = gamma_p({3 7 3 7});  
gpcf1.p.magnSigma2 = sinvchi2_p({0.05^2 0.5});

% sparse model. Set the inducing points to the GP
gp = gp_init('init', nin, 'regr', {gpcf1}, {gpcf2}, 'sparse', 'FIC', 'jitterSigmas', 1);
gp.inducing = x(1:3:end)
% full model
gp2 = gp_init('init', nin, 'regr', {gpcf1}, {gpcf2}, 'jitterSigmas', 1)

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
[w,fs,vs]=scges(fe, w, optes, fg, gp, x(itr,:),y(itr,:), gp,x(its,:),y(its,:));
gp=gp_unpak(gp,w);
% full model
w=randn(size(gp_pak(gp2)))*0.01;
[w,fs,vs]=scges(fe, w, optes, fg, gp2, x(itr,:),y(itr,:), gp2,x(its,:),y(its,:));
gp2=gp_unpak(gp2,w);


% the sampling options
opt=gp_mcopt;
opt.repeat=10;
opt.nsamples=10;
opt.hmc_opt.steps=20;
opt.hmc_opt.stepadj=0.08;
opt.hmc_opt.nsamples=1;
hmc2('state', sum(100*clock));

% Sample sparse model
t = cputime;
[r,g,rstate2]=gp_mc(opt, gp, x, y);
tsparse = cputime - t;
% Sample full model
t = cputime;
[r2,g2,rstate2]=gp_mc(opt, gp2, x, y);
tfull = cputime - t;

opt.hmc_opt.stepadj=0.08;
opt.nsamples=300;
opt.hmc_opt.steps=10;
opt.hmc_opt.persistence=1;
opt.hmc_opt.decay=0.6;

% load saved record
load 'demos/demo_gpSparse'

% Sample sparse model
t = cputime;
[r,g,rstate2]=gp_mc(opt, gp, x, y, [], [], r);
tsparse = cputime - t;
% Sample full model
t = cputime;
[r2,g2,rstate2]=gp_mc(opt, gp2, x, y, [], [], r2);
tfull = cputime - t;

% Visualization of results

% Plot the data and the underlying mean
xx = [-2:0.01:2]';
yy = 3*sin(3*xx)+xx.^3+cos(9.*xx);
plot(xx,yy,'k') 
hold on
plot(x,y,'.')
axis([-2 2 -6 6])
% Plot the inducing points
plot(gp.inducing,3*sin(3*gp.inducing)+gp.inducing.^3+cos(9.*gp.inducing),'r*')

% The predictions for the new inputs of sparse model
rr=thin(r,50,3);
[yn, covYn] = gp_fwds(rr, x, y, xx);
Ey = mean(squeeze(yn)');
plot(xx,Ey)

% The predictions for the new inputs of full model
rr2=thin(r2,50,3);
[yn2, covYn2] = gp_fwds(rr2, x, y, xx);
Ey2 = mean(squeeze(yn2)');
plot(xx,Ey2, 'g--')

legend('E[y]','y','u','sparse', 'full')