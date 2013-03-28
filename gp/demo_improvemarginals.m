%DEMO_IMPROVEMARGINALS  Demonstration of marginal likelihood improvements 
%                       in Laplace and EP algorithms.
%
%  Description
%    Demonstrates improvements to marginal likelihoods with
%    gpep_fact/gpla_fact/gpla_cm2. gpla_cm2 improvement is the fastest
%    while providing better or at least as good corrections (in our
%    test cases) as the slower gpla_fact.
%

% This software is distributed under the GNU General Public 
% License (version 3 or later); please refer to the file 
% License.txt, included with the software, for details.

% ---------------------------
% Probit likelihood with EP
% ---------------------------
tic
S = which('demo_improvemarginals');
L = strrep(S,'demo_improvemarginals.m','demodata/synth.tr');
x=load(L);
y=x(:,end);
y = 2.*y-1;
x(:,end)=[];
[n, nin] = size(x);

% Load sampled results so the demo won't take hours or days to finish...
L = strrep(S,'demo_improvemarginals.m','demodata/samples_marginal.mat');
load(L);

% Test data
xt1=repmat(linspace(min(x(:,1)),max(x(:,1)),20)',1,20);
xt2=repmat(linspace(min(x(:,2)),max(x(:,2)),20)',1,20)';
xt=[xt1(:) xt2(:)];

% Create a likelihood function
lik = lik_probit();
%lik = lik_logit();

% Create a covariance function
gpcf = gpcf_sexp('lengthScale', [0.9 0.9], 'magnSigma2', 10);

% Set the prior for the parameters of the covariance function
pl = prior_unif();
pm = prior_sqrtunif();
gpcf = gpcf_sexp(gpcf, 'lengthScale_prior', pl,'magnSigma2_prior', pm); %

% Create the GP structure (type is by default FULL)
gp_probit = gp_set('lik', lik, 'cf', gpcf, 'jitterSigma2', 1e-9);

% Set the approximate inference method 
gp_probit = gp_set(gp_probit, 'latent_method', 'EP');

ind = 22;

% % Get MCMC samples

% gp_probit2 = gp_set(gp_probit, 'latent_method', 'MCMC', 'jitterSigma2', 1e-4);
% 
% % set MC options
% latent_opt.display=0;
% latent_opt.repeat = 5;
% latent_opt.sample_latent_scale = 0.5;
% hmc2('state', sum(100*clock))


% % obtain MC samples
% [rgp_probit]=gp_mc(gp_probit2, x, y, 'latent_opt', latent_opt, 'nsamples', 5000, 'repeat', 2, 'display', 0);
% rgp_probit=thin(rgp_probit,200);

% f_mc_probit = rgp_probit.latentValues(:,ind);

subplot(2,1,1);
% figure;

for i=1:length(ind)
%   figure;
  [testi, testi2] = hist(f_mc_probit(:,i),50);
  width = testi2(2)-testi2(1);
  area = sum(testi.*width);
  testi = testi./area;
  b = bar(testi2,testi);
  h = findobj(gca,'Type','patch');
  set(h,'FaceColor','w')
  hold on;
  [Eft_probit, Varft_probit] = gp_pred(gp_probit,x,y,x);
  fvec_probit(:,i) = linspace(Eft_probit(ind(i))-6*sqrt(Varft_probit(ind(i))), Eft_probit(ind(i))+6*sqrt(Varft_probit(ind(i))), 50)';
  [p_probit(:,i), pc_probit(:,i)] = gpep_fact(gp_probit,x,y,fvec_probit(:,i), 'ind', ind(i));
  s = plot(fvec_probit(:,i), p_probit(:,i), '-k', fvec_probit(:,i), norm_pdf(fvec_probit(:,i), Eft_probit(ind(i)), sqrt(Varft_probit(ind(i)))),'-m', fvec_probit(:,i), pc_probit(:,i), '-r');
  set(s,'LineWidth',2)
  set(get(get(b,'Annotation'),'LegendInformation'),...
    'IconDisplayStyle','off');
  legend('EP-L', 'EP-G', 'EP-FACT');
  title('Marginal corrections for probit likelihood (EP)');
end

subplot(2,1,2)

% Predictive corrections
for i=1:length(ind)
%   [Efs_probit, Varfs_probit] = gpmc_preds(rgp_probit, x, y, xt(ind(i),:));
  [Eft_probit_pred, Varft_probit_pred] = gp_pred(gp_probit,x,y,xt);
  fvec_probit_pred(:,i) = linspace(Eft_probit_pred(ind(i))-6*sqrt(Varft_probit_pred(ind(i))), Eft_probit_pred(ind(i))+6*sqrt(Varft_probit_pred(ind(i))), 30)';
%   ptx_prob = [];
%   for i3=1:size(Efs_probit,2)
%     ptx_prob = [ptx_prob; norm_pdf(fvec_probit_pred', Efs_probit(i3), sqrt(Varfs_probit(i3)))];
%   end
  [p_probit_pred(:,i), pc_probit_pred(:,i)] = gpep_fact(gp_probit,x,y,fvec_probit_pred(:,i),xt, 'ind', ind(i));
  plot(fvec_probit_pred(:,i), p_probit_pred(:,i), '-k', fvec_probit_pred(:,i), norm_pdf(fvec_probit_pred(:,i), Eft_probit_pred(ind(i)), sqrt(Varft_probit_pred(ind(i)))), '-m', fvec_probit_pred(:,i), pc_probit_pred(:,i), '-r', fvec_probit_pred(:,i), mean(ptx_prob,1), '-c');
  set(s,'LineWidth',2)
  legend('EP-L', 'EP-G', 'EP-FACT', 'MCMC');
  title('Predictive marginal corrections for probit likelihood (EP)');
end
% clear('fvec_probit', 'p_probit', 'pc_probit')

% ---------------------------
% Probit likelihood with Laplace
% ---------------------------

% Create the GP structure (type is by default FULL)
gp_probit_laplace = gp_set('lik', lik, 'cf', gpcf, 'jitterSigma2', 1e-9);

% Set the approximate inference method 
% (Laplace is the default, so this could be skipped)
gp_probit_laplace = gp_set(gp_probit_laplace, 'latent_method', 'Laplace');

ind = 22;
% Get MCMC samples

% gp_probit_laplace2 = gp_set(gp_probit_laplace, 'latent_method', 'MCMC', 'jitterSigma2', 1e-4);
% 
% % set MC options
% latent_opt.display=0;
% latent_opt.repeat = 5;
% latent_opt.sample_latent_scale = 0.5;
% hmc2('state', sum(100*clock))

% % obtain MC samples
% [rgp_probit_laplace]=gp_mc(gp_probit_laplace2, x, y, 'latent_opt', latent_opt, 'nsamples', 5000, 'repeat', 2, 'display', 0);
% rgp_probit_laplace=thin(rgp_probit_laplace,200);

% f_mc_probit_laplace = rgp_probit_laplace.latentValues(:,ind);

figure; subplot(2,1,1);


for i=1:length(ind)
  [testi, testi2] = hist(f_mc_probit_laplace(:,i),50);
  width = testi2(2)-testi2(1);
  area = sum(testi.*width);
  testi = testi./area;
  b = bar(testi2,testi);
  h = findobj(gca,'Type','patch');
  set(h,'FaceColor','w')
  hold on;
  [Eft_probit, Varft_probit] = gp_pred(gp_probit_laplace,x,y,x);
  fvec_probit_laplace(:,i) = linspace(Eft_probit(ind(i))-6*sqrt(Varft_probit(ind(i))), Eft_probit(ind(i))+6*sqrt(Varft_probit(ind(i))), 50)';
  [p_probit_laplace(:,i), pc_probit_laplace(:,i), c_probit_laplace(:,i)] = gpla_cm2(gp_probit_laplace,x,y,fvec_probit_laplace(:,i), 'ind', ind(i)); 
  [p_probit_laplace2(:,i), pc_probit_laplace2(:,i), c_probit_laplace2(:,i)] = gpla_fact(gp_probit_laplace,x,y,fvec_probit_laplace(:,i), 'ind', ind(i));
  s = plot(fvec_probit_laplace(:,i), p_probit_laplace(:,i), '-k', fvec_probit_laplace(:,i), norm_pdf(fvec_probit_laplace(:,i), Eft_probit(ind(i)), sqrt(Varft_probit(ind(i)))), '-m', fvec_probit_laplace(:,i), pc_probit_laplace2(:,i), '-r', fvec_probit_laplace(:,i), pc_probit_laplace(:,i), '-b');
  set(s,'LineWidth',2)
  set(get(get(b,'Annotation'),'LegendInformation'),...
    'IconDisplayStyle','off');
  legend('LA-L', 'LA-G', 'LA-FACT', 'LA-CM2');
  title('Marginal corrections for probit likelihood (Laplace)');
end
% clear('fvec_probit_laplace', 'p_probit_laplace', 'pc_probit_laplace', 'p_probit_laplace2', 'pc_probit_laplace2', 'c_probit_laplace','c_probit_laplace2')

% Predictive corrections
subplot(2,1,2);

for i=1:length(ind)
%   [Efs, Varfs] = gpmc_preds(rgp_probit_laplace, x, y, xt(ind(i),:));
  [Eft_probit_pred, Varft_probit_pred] = gp_pred(gp_probit_laplace,x,y,xt);
  fvec_probit_laplace_pred(:,i) = linspace(Eft_probit_pred(ind(i))-6*sqrt(Varft_probit_pred(ind(i))), Eft_probit_pred(ind(i))+6*sqrt(Varft_probit_pred(ind(i))), 30)';
%   ptx_lap = [];
%   for i3=1:size(Efs,2)
%     ptx_lap = [ptx_lap; norm_pdf(fvec_probit_laplace_pred', Efs(i3), sqrt(Varfs(i3)))];
%   end
  [p_pred(:,i), pc_pred(:,i), c_pred(:,i)] = gpla_cm2(gp_probit_laplace,x,y,fvec_probit_laplace_pred(:,i),xt, 'ind', ind(i)); 
  [p_pred2(:,i), pc_pred2(:,i), c_pred2(:,i)] = gpla_fact(gp_probit_laplace,x,y,fvec_probit_laplace_pred(:,i),xt, 'ind', ind(i));
  s = plot(fvec_probit_laplace_pred(:,i), p_pred(:,i), '-k', fvec_probit_laplace_pred(:,i), norm_pdf(fvec_probit_laplace_pred(:,i), Eft_probit_pred(ind(i)), sqrt(Varft_probit_pred(ind(i)))), '-m', fvec_probit_laplace_pred(:,i), pc_pred(:,i), '-r', fvec_probit_laplace_pred(:,i), pc_pred2(:,i), '-b', fvec_probit_laplace_pred(:,i), mean(ptx_lap,1), '-c');
  legend('LA-L', 'LA-G', 'LA-CM2', 'LA-FACT', 'MCMC');
  title('Predictive marginal corrections for probit likelihood (Laplace)');
end
% clear('p_pred', 'p_pred2', 'pc_pred', 'pc_pred2', 'fvec_probit_laplace_pred', 'c_pred', 'c_pred2')
toc
