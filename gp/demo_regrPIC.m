function demo_regrPIC
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
S = which('demo_regrPIC');
L = strrep(S,'demo_regrPIC.m','demos/dat.1');
data=load(L);
x = [data(:,1) data(:,2)];
y = data(:,3);
[n, nin] = size(x);

% Create covariance functions
%gpcf1 = gpcf_sexp('init', nin, 'lengthScale', 1, 'magnSigma2', 0.2^2);
gpcf1 = gpcf_sexp('init', nin, 'lengthScale', [1 1], 'magnSigma2', 0.2^2);

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
gp = gp_init('init', 'PIC_BLOCK', nin, 'regr', {gpcf1}, {gpcf2}, 'jitterSigmas', 0.001);

% Set the blocks and the inducing inputs
b1 = [-1.7 -0.8 0.1 1 1.9];
mask = zeros(size(x,1),size(x,1));
tot = 0;
for i1=1:4
    for i2=1:4
        ind = 1:size(x,1);
        ind = ind(: , b1(i1)<=x(ind',1) & x(ind',1) < b1(i1+1));
        ind = ind(: , b1(i2)<=x(ind',2) & x(ind',2) < b1(i2+1));        
        index{4*(i1-1)+i2} = ind';
        mask(ind,ind) = 1;
        tot = tot  + length(ind);
    end
end

figure
d=symamd(mask);
spy(mask(d,d))
nnz(mask)/prod(size(mask))
pcolor(mask(d,d)), shading flat
title('The correlation matrix')

[u1,u2]=meshgrid(linspace(-1.7,1.9,5),linspace(-1.7,1.9,5));
U=[u1(:) u2(:)];

% plot the data points in each block with different colors and marks
figure
col = {'b*','g*','r*','c*','m*','y*','k*','b*','b.','g.','r.','c.','m.','y.','k.','b.'};
hold on
for i=1:16
    plot(x(index{i},1),x(index{i},2),col{i})
end
% plot the inducing inputs
plot(u1(:), u2(:), 'kX', 'MarkerSize', 12, 'LineWidth', 2)
title('Blocks and inducing inputs')

% Set the inducing inputs and blocks into the gp structure
gp = gp_init('set', gp, 'X_u', U, 'blocks', {'manual', x, index});

% $$$ % Check the gradients
% $$$ gradcheck(gp_pak(gp,'hyper'), @gp_e, @gp_g, gp, x, y, 'hyper')


% $$$ w=gp_pak(gp, 'hyper');
% $$$ [e, edata, eprior] = gp_e(w, gp, x, y, 'hyper')     % answer  488.9708 
% $$$ [g, gdata, gprior] = gp_g(w, gp, x, y, 'hyper')
% $$$ 
% $$$ gradcheck(gp_pak(gp,'hyper'), @gp_e, @gp_g, gp, x, y, 'hyper')
% $$$ 
% $$$ e =
% $$$   406.7074
% $$$ edata =
% $$$   401.9293
% $$$ eprior =
% $$$     4.7781
% $$$ 
% $$$ g =
% $$$  -274.8715  407.3560 -105.4114
% $$$ gdata =
% $$$  -275.1214  410.3560 -105.6614
% $$$ gprior =
% $$$     0.2500   -3.0000    0.2500


% plot the data points in each block with different colors and marks
% $$$ col = {'b*','g*','r*','c*','m*','y*','k*','b*','b.','g.','r.','c.','m.','y.','k.','b.'};
% $$$ hold on
% $$$ for i=1:16
% $$$     plot(x(index{i},1),x(index{i},2),col{i})
% $$$ end
% $$$ for i=1:size(x,1);
% $$$     index{i} = i;
% $$$ end
% $$$ index = {[1:122]' [123:225]'}
% $$$ index = {[1:225]}'
% $$$ % For testing
% $$$ mask = eye(size(x,1),size(x,1));
% $$$ 
% $$$ 
% $$$ gp = gp_init('set', gp, 'X_u', U, 'blocks', {'manual', x, index});
% $$$ gp.mask = mask;
% $$$ 
% $$$ 
% $$$ % find starting point using scaled conjucate gradient algorithm
% $$$ % Intialize weights to zero and set the optimization parameters
% $$$ w=gp_pak(gp, 'hyper');
% $$$ gp_e(w, gp, x, y, 'hyper')          % with all 370.9320
% $$$ %w=randn(size(gp_pak(gp, 'all')))*0.01;

% $$$ w=gp_pak(gp, 'hyper');
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
% $$$ [w,fs,vs]=scges(fe, w, optes, fg, gp, x(itr,:),y(itr,:), 'hyper', gp,x(its,:),y(its,:), 'hyper');
% $$$ gp=gp_unpak(gp,w);
% $$$ % full model
% $$$ w=randn(size(gp_pak(gp2)))*0.01;
% $$$ [w,fs,vs]=scges(fe, w, optes, fg, gp2, x(itr,:),y(itr,:), gp2,x(its,:),y(its,:));
% $$$ gp2=gp_unpak(gp2,w);

opt=gp_mcopt;
opt.repeat=1;
opt.nsamples=300;
opt.hmc_opt.steps=4;
opt.hmc_opt.stepadj=0.04;
opt.hmc_opt.window=1;
hmc2('state', sum(100*clock));

% Sample sparse model
t = cputime;
[r,g,rstate2]=gp_mc(opt, gp, x, y, [], []);
tsparse = cputime - t;

% Evaluate the MSE for the predictions
% Evaluate the MSE for the predictions
r = rmfield(r, 'tr_index');
r = thin(r,50);
out=gp_fwds(r, x, y, x, gp.tr_index, gp.tr_index);
mout = mean(squeeze(out)');
pred = zeros(size(x,1),1);
pred(:)=mout;

figure
title('The prediction');
[xi,yi,zi]=griddata(data(:,1),data(:,2),pred,-1.8:0.01:1.8,[-1.8:0.01:1.8]');
mesh(xi,yi,zi)

(pred-y)'*(pred-y)/length(y)


% $$$ >> mean(r.cf{1}.lengthScale)
% $$$ ans =
% $$$     1.1690    1.0770
% $$$ >> 


% New input
[p1,p2]=meshgrid(-1.7:0.05:1.8,-1.7:0.05:1.8);
p=[p1(:) p2(:)];
tot = 0;
for i1=1:4
    for i2=1:4
        ind = 1:size(p,1);
        ind = ind(: , b1(i1)<=p(ind',1) & p(ind',1) < b1(i1+1));
        ind = ind(: , b1(i2)<=p(ind',2) & p(ind',2) < b1(i2+1));        
        tst_index{4*(i1-1)+i2} = ind';
        tot = tot  + length(ind);
    end
end

figure
col = {'b*','g*','r*','c*','m*','y*','k*','b*','b.','g.','r.','c.','m.','y.','k.','b.'};
hold on
for i=1:16
    plot(p(tst_index{i},1),p(tst_index{i},2),col{i})
end
% plot the inducing inputs
plot(u1(:), u2(:), 'kX', 'MarkerSize', 12, 'LineWidth', 2)
title('Blocks and inducing inputs')


% The predictions for the new inputs of sparse model
%rr=thin(r,10,2);
yn = gp_fwds(r, x, y, p, gp.tr_index, tst_index);

pred = zeros(size(p1));
pred(:)=mean(squeeze(yn)');
figure
mesh(p1,p2,pred);
qc=caxis;
title('The prediction into dence grid')



gp.cf{1}.lengthScale = [1.0640 1.0525];
gp.cf{1}.magnSigma2 = 2.0215;
gp.noise{1}.noiseSigmas2 = 0.027;
yn = gp_fwd(gp, x, y, p, gp.tr_index, tst_index);
pred(:)=yn;
figure
mesh(p1,p2,pred);
qc=caxis;
title('sparse')
