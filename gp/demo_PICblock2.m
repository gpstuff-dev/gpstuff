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
%gpcf1 = gpcf_sexp('init', nin, 'lengthScale', 1, 'magnSigma2', 0.2^2);
%gpcf1 = gpcf_sexp('init', nin, 'lengthScale', [1, 1], 'magnSigma2', 0.2^2);
gpcf1 = gpcf_exp('init', nin, 'lengthScale', [1, 1], 'magnSigma2', 0.2^2);
%gpcf1 = gpcf_matern32('init', nin, 'lengthScale', 1, 'magnSigma2', 0.2^2);
%gpcf1 = gpcf_matern52('init', nin, 'lengthScale', 1, 'magnSigma2', 0.2^2);
gpcf2 = gpcf_noise('init', nin, 'noiseSigmas2', 0.2^2);

% Set the prior for the parameters of covariance functions 
gpcf2.p.noiseSigmas2 = sinvchi2_p({0.05^2 0.5});    % MUUTA tässä invgam_p saman näköiseksi kuin gpcf_sexp('set'...)
gpcf1.p.lengthScale = gamma_p({3 7});  
gpcf1.p.magnSigma2 = sinvchi2_p({0.05^2 0.5});

% sparse model. Set the inducing points to the GP
gp = gp_init('init', 'PIC_BLOCK', nin, 'regr', {gpcf1}, {gpcf2}, 'jitterSigmas', 0.1);

% Set the inducing inputs
[u1,u2]=meshgrid(linspace(-1.8,1.8,5),linspace(-1.8,1.8,4));
U=[u1(:) u2(:)];
%U = 3.6.*rand(14,2)-1.8;

% define the blocks by cubes
b1 = [-1.9 -0.95 0 0.95 1.9];
mask = zeros(size(x,1),size(x,1));
for i1=1:4
    for i2=1:4
        ind = 1:size(x,1);
        ind = ind(: , b1(i1)<=x(ind',1) & x(ind',1) < b1(i1+1));
        ind = ind(: , b1(i2)<=x(ind',2) & x(ind',2) < b1(i2+1));        
        index{4*(i1-1)+i2} = ind';
        mask(ind,ind) = 1;
    end
end
% plot the data points in each block with different colors and marks
col = {'b*','g*','r*','c*','m*','y*','k*','b*','b.','g.','r.','c.','m.','y.','k.','b.'};
hold on
for i=1:16
    plot(x(index{i},1),x(index{i},2),col{i})
end
for i=1:size(x,1);
    index{i} = i;
end
index = {[1:122]' [123:225]'}
index = {[1:225]}'
% For testing
mask = eye(size(x,1),size(x,1));


gp = gp_init('set', gp, 'X_u', U, 'blocks', {'manual', x, index});
gp.mask = mask;


% find starting point using scaled conjucate gradient algorithm
% Intialize weights to zero and set the optimization parameters
w=gp_pak(gp, 'hyper');
gp_e(w, gp, x, y, 'hyper')          % with all 370.9320
%w=randn(size(gp_pak(gp, 'all')))*0.01;



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
