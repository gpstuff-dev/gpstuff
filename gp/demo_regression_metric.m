% Demonstration for the metric structures

% Load the data
S = which('demo_regression1');
L = strrep(S,'demo_regression1.m','demos/dat.1');
data=load(L);
x = [data(:,1) data(:,2)];
y = data(:,3);
[n, nin] = size(x);

% Construct additive covariance functions for two dimensional
% regression (Gaussian noise) data.

% This part is done as usual
gpcf1 = gpcf_sexp('init', nin, 'magnSigma2', 0.2^2);
gpcf1.p.magnSigma2 = sinvchi2_p({0.05^2 0.5});

% Lets now initialize an euclidean metric structure, which uses only the first
% component of the input vector for calculating distances used by the
% covariance function. 
% 
% This is done by specifying a cell array whose entries are vectors
% representing subsets of input components, which are used in calculating
% the distances such that each subset is given it's own lenght scale. In
% this case, we only need to have vector containing the number one representing
% the first component of the input vector. 
% 
% Note that isotropic covariance is constructed with cell array {[1:d]}
% (d being the number of inputs), and ARD metric with {[1] ... [d]}.
%
% Notice also that lengthscale hyperparameters are now stored
% inside the metric structures as they are essentially parameters
% of the euclidean metric.

metric1 = metric_euclidean('init', nin, {[1]},'params',[0.8]);

% We also need to specify a prior for the length scales.
metric1.p.params = gamma_p({3 7});  

% Lastly, plug the metric to the covariance function structure.
gpcf1 = gpcf_sexp('set', gpcf1, 'metric', metric1);

% Do the same for the second input
gpcf2 = gpcf_sexp('init', nin, 'magnSigma2', 0.2^2);
gpcf2.p.magnSigma2 = sinvchi2_p({0.05^2 0.5});
metric2 = metric_euclidean('init', nin, {[2]},'params',[0.8]);
metric2.p.params = gamma_p({3 7});  
gpcf2 = gpcf_sexp('set', gpcf2, 'metric', metric2);

% We also need the noise component
gpcfn = gpcf_noise('init', nin, 'noiseSigmas2', 0.2^2);
gpcfn.p.noiseSigmas2 = sinvchi2_p({0.05^2 0.5});

% ... Finally create the GP data structure
gp = gp_init('init', 'FULL', nin, 'regr', {gpcf1,gpcf2}, {gpcfn}, 'jitterSigmas', 0.001)    

% Uncomment these if you want to use a sparse model instead
% $$$ gp = gp_init('init', 'PIC_BLOCK', nin, 'regr', {gpcf1}, {gpcf2}, 'jitterSigmas', 0.001)    
% $$$ [U1 U2] = meshgrid(-1.0:1:1.0,-1.0:1:1.0);
% $$$ [U1 U2] = meshgrid(-0.5:1:0.5,0);
% $$$ U = [U1(:) U2(:)];
% $$$ gp = gp_init('set', gp, 'X_u', U);
% $$$ 
% $$$ 
% $$$ [p1,p2]=meshgrid(-1.8:0.1:1.8,-1.8:0.1:1.8);
% $$$ p=[p1(:) p2(:)];
% $$$ 
% $$$ % set the data points into clusters
% $$$ b1 = [-1.7 -0.8 0.1 1 1.9];
% $$$ mask = zeros(size(x,1),size(x,1));
% $$$ trindex={}; tstindex={};
% $$$ for i1=1:4
% $$$     for i2=1:4
% $$$         ind = 1:size(x,1);
% $$$         ind = ind(: , b1(i1)<=x(ind',1) & x(ind',1) < b1(i1+1));
% $$$         ind = ind(: , b1(i2)<=x(ind',2) & x(ind',2) < b1(i2+1));
% $$$         trindex{4*(i1-1)+i2} = ind';
% $$$         ind2 = 1:size(p,1);
% $$$         ind2 = ind2(: , b1(i1)<=p(ind2',1) & p(ind2',1) < b1(i1+1));
% $$$         ind2 = ind2(: , b1(i2)<=p(ind2',2) & p(ind2',2) < b1(i2+1));
% $$$         tstindex{4*(i1-1)+i2} = ind2';
% $$$     end
% $$$ end
% $$$ 
% $$$ gp = gp_init('set', gp, 'blocks', {'manual', x, trindex});

param = 'hyper';
%gradcheck(gp_pak(gp,param), @gp_e, @gp_g, gp, x, y, param)

% Conduct the inference
w=gp_pak(gp, param);  % pack the hyperparameters into one vector
fe=str2fun('gp_e');     % create a function handle to negative log posterior
fg=str2fun('gp_g');     % create a function handle to gradient of negative log posterior

% set the options
opt = scg2_opt;
opt.tolfun = 1e-3;
opt.tolx = 1e-3;
opt.display = 1;

% do the optimization
[w, opt, flog]=scg2(fe, w, opt, fg, gp, x, y, param);

% Set the optimized hyperparameter values back to the gp structure
gp=gp_unpak(gp,w, param);

% Make predictions
[p1,p2]=meshgrid(-1.8:0.1:1.8,-1.8:0.1:1.8);
p=[p1(:) p2(:)];
[Ef_full, Varf_full] = gp_pred(gp, x, y, p, []);
[Ef_full1, Varf_full1] = gp_pred(gp, x, y, p, [1]);
[Ef_full2, Varf_full2] = gp_pred(gp, x, y, p, [2]);

% Plot the prediction and data
figure(1)
subplot(1,3,1)
mesh(p1, p2, reshape(Ef_full,37,37)); hold on;
plot3(x(:,1), x(:,2), y, '*'); hold off;
axis on;
title('Prediction with both covariances.');

subplot(1,3,2)
mesh(p1, p2, reshape(Ef_full1,37,37)); hold on;
plot3(x(:,1), x(:,2), y, '*'); hold off;
axis on;
title('Prediction with only the first input.');

subplot(1,3,3)
mesh(p1, p2, reshape(Ef_full2,37,37)); hold on;
plot3(x(:,1), x(:,2), y, '*'); hold off;
axis on;
title('Prediction with only the second input.');

