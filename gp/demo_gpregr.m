function demo_gpregr
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
gpcf1 = gpcf_sexp('init', nin, 'lengthScale', [1 1], 'magnSigma2', 0.2^2);
gpcf2 = gpcf_noise('init', nin, 'noiseSigmas2', 0.2^2);

% Set the prior for the parameters of covariance functions 
gpcf2.p.noiseSigmas2 = sinvchi2_p({0.05^2 0.5});
gpcf1.p.lengthScale = gamma_p({3 7});  
gpcf1.p.magnSigma2 = sinvchi2_p({0.05^2 0.5});

gp = gp_init('init', 'FULL', nin, 'regr', {gpcf1}, {gpcf2}, 'jitterSigmas', 0.1)    

w=gp_pak(gp, 'hyper');
gp_e(w, gp, x, y, 'hyper')    % answer 370.9230



% Set the sampling options
opt=gp_mcopt;
opt.repeat=10;
opt.nsamples=10;
opt.hmc_opt.steps=10;
opt.hmc_opt.stepadj=0.1;
opt.hmc_opt.nsamples=1;
hmc2('state', sum(100*clock));

% Sample 
[r,g,rstate1]=gp_mc(opt, gp, x, y);

opt.hmc_opt.stepadj=0.08;
opt.nsamples=200;
opt.repeat=5;
opt.hmc_opt.steps=6;
opt.hmc_opt.stepadj=0.05;
opt.hmc_opt.persistence=0;
opt.hmc_opt.decay=0.6;

[r,g,rstate2]=gp_mc(opt, gp, x, y, [], [], r);

% The predictions for the new inputs
[p1,p2]=meshgrid(-1.8:0.05:1.8,-1.8:0.05:1.8);
p=[p1(:) p2(:)];
rr=thin(r,50,2);
out=gp_fwds(rr, x, y, p);
mout = mean(squeeze(out)');
pred = zeros(size(p1));
pred(:)=mout;

% Plot the old data and the new data
figure
title({'The noisy teaching data'});
[xi,yi,zi]=griddata(data(:,1),data(:,2),data(:,3),-1.8:0.01:1.8,[-1.8:0.01:1.8]');
mesh(xi,yi,zi)
figure
title({'The underlying function'});
mesh(p1,p2,pred);
axis on;

