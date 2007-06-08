function demo_PICband
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

% gp_e(gp_pak(gp,'hyper'), gp,x,y,'hyper')
% gradcheck(gp_pak(gp,'hyper'), @gp_e, @gp_g, gp, x, y, 'hyper')

% Load the data
S = which('demo_gpregr');
L = strrep(S,'demo_gpregr.m','demos/dat.1');
data=load(L);
x = [data(:,1) data(:,2)];
y = data(:,3);
[n, nin] = size(x);

% Create covariance functions
gpcf1 = gpcf_sexp('init', nin, 'lengthScale', 1, 'magnSigma2', 0.1);
%gpcf1 = gpcf_sexp('init', nin, 'lengthScale', [1, 1], 'magnSigma2', 0.2^2);
%gpcf1 = gpcf_exp('init', nin, 'lengthScale', 1, 'magnSigma2', 0.2^2);
%gpcf1 = gpcf_matern32('init', nin, 'lengthScale', 1, 'magnSigma2', 0.2^2);
%gpcf1 = gpcf_matern52('init', nin, 'lengthScale', 1, 'magnSigma2', 0.2^2);

gpcf2 = gpcf_noise('init', nin, 'noiseSigmas2', 0.2);

% Set the prior for the parameters of covariance functions 
gpcf2.p.noiseSigmas2 = sinvchi2_p({0.05^2 0.5});    % MUUTA tässä invgam_p saman näköiseksi kuin gpcf_sexp('set'...)
gpcf1.p.lengthScale = gamma_p({3 7});  
gpcf1.p.magnSigma2 = sinvchi2_p({0.05^2 0.5});

% sparse model. Set the inducing points to the GP
gp = gp_init('init', 'PIC_BAND', nin, 'regr', {gpcf1}, {gpcf2}, 'jitterSigmas', 0.1);

% Set the inducing inputs
[u1,u2]=meshgrid(linspace(-1.8,1.8,5),linspace(-1.8,1.8,5));
U=[u1(:) u2(:)];
%U = 3.6.*rand(14,2)-1.8;

% Set the indexes for the truncated model
%R = 0.33;
R = 0.6;

C = sparse([],[],[],n,n,0);
for i1=2:n
    i1n=(i1-1)*n;
    for i2=1:i1-1
        ii=i1+(i2-1)*n;
        D = 0;
        for i3=1:nin
            D =D+(x(i1,i3)-x(i2,i3)).^2;       % the covariance function
        end
        if sqrt(D) < R
            C(ii)=1;
            C(i1n+i2)=C(ii); 
        end
    end
end
C= C+ speye(n,n);
spy(C)
nnz(C)/prod(size(C))
[I,J,s] = find(C);

%gp.tr_index = [I(:) J(:)];

gp = gp_init('set', gp, 'X_u', U, 'truncated', {x, R, 1});

% find starting point using scaled conjucate gradient algorithm
% Intialize weights to zero and set the optimization parameters
% $$$ w=gp_pak(gp, 'hyper');
% $$$ gp_e(w, gp, x, y, 'hyper')          % with all 370.9320
% $$$ %w=randn(size(gp_pak(gp, 'all')))*0.01;
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
% $$$ % sparse model
% $$$ [w,fs,vs]=scges(fe, w, optes, fg, gp, x(itr,:),y(itr,:), U, gp,x(its,:),y(its,:), U);
% $$$ gp=gp_unpak(gp,w);
% $$$ % full model
% $$$ w=randn(size(gp_pak(gp2)))*0.01;
% $$$ [w,fs,vs]=scges(fe, w, optes, fg, gp2, x(itr,:),y(itr,:), gp2,x(its,:),y(its,:));
% $$$ gp2=gp_unpak(gp2,w);

opt=gp_mcopt;
opt.repeat=1;
opt.nsamples=300;
opt.hmc_opt.steps=3;
opt.hmc_opt.stepadj=0.01;
opt.hmc_opt.nsamples=1;
opt.hmc_opt.window=1;
hmc2('state', sum(100*clock));

% Sample sparse model
t = cputime;
[r,g,rstate2]=gp_mc(opt, gp, x, y, [], []);
tsparse = cputime - t;

% New input
[p1,p2]=meshgrid(-1.8:0.05:1.8,-1.8:0.05:1.8);
p=[p1(:) p2(:)];

% Evaluate the MSE for the predictions
out=gp_fwds(rr, x, y, x);
mout = mean(squeeze(out)');
pred = zeros(size(p1));
pred(:)=mout;










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


















gradcheck(gp_pak(gp,'hyper'), @gp_e, @gp_g, gp, x, y, 'hyper')

w=gp_pak(gp, 'hyper');
[e, edata, eprior] = gp_e(w, gp, x, y, 'hyper')     % answer  488.9708 
[g, gdata, gprior] = gp_g(w, gp, x, y, 'hyper')


u = gp.X_u;
ind = gp.tr_index;
K_fu = gp_cov(gp, x, u);         % f x u
K_uu = gp_trcov(gp, u);          % u x u, noiseles covariance K_uu
K_uu = (K_uu+K_uu')./2;          % ensure the symmetry of K_uu

t = cputime;
[K_ff_f, C_ff_f] = gp_trcov(gp,x);
Q_ff_f = (K_fu/K_uu)*K_fu';
La_f = C.*(C_ff_f - Q_ff_f);
cputime-t

K_ff_2 = gp_cov(gp,x,x);
[Kv_f, Cv_f] = gp_trvar(gp,x);

min(min(C_ff_f - (K_ff_2-diag(Kv_f-Cv_f)) ))

min(min(C.*C_ff_f-C_ff))

Ksp_ff=C.*K_ff_f;
p = symamd(C);
figure; spy(chol(Ksp_ff(p,p)))
figure; spy(chol(Ksp_ff))

iLaKfu = La_f\K_fu;    % Check if works by changing inv(Labl{i})!!!
A = K_uu+K_fu'*iLaKfu;
A = (A+A')./2;            % Ensure symmetry

L = iLaKfu/chol(A);
b2 = t'/La_f - (t'*L)*L';
