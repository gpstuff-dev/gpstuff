%DEMO_IMPROVEMARGINALS  Demonstration of marginal posterior improvements 
%                       in Laplace and EP algorithms.
%
%  Description
%    Demonstration of marginal posterior corrections of latent
%    variables in classification task. Demonstrated corrections are
%    'fact' (EP & Laplace) and 'cm2' (Laplace). The corrected
%    posterior distributions are compared to histograms of MCMC
%    samples to assess the quality of corrections.
%
%  See also
%    GP_PREDCM

% This software is distributed under the GNU General Public 
% License (version 3 or later); please refer to the file 
% License.txt, included with the software, for details.

% ---------------------------
% Probit likelihood with EP
% ---------------------------
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
pl = prior_t();
pm = prior_sqrtunif();
gpcf = gpcf_sexp(gpcf, 'lengthScale_prior', pl,'magnSigma2_prior', pm); %

% Create the GP structure (type is by default FULL)
gp_probit = gp_set('lik', lik, 'cf', gpcf, 'jitterSigma2', 1e-9);

% Set the approximate inference method 
gp_probit = gp_set(gp_probit, 'latent_method', 'EP');

ind = 22;

% If we didnt load previously computed samples we would run the following
% to get the MCMC samples for latents

% gp_probit2 = gp_set(gp_probit, 'latent_method', 'MCMC');
% 
% % set MC options
% latent_opt.repeat=10;
% 
% % obtain MC samples
% [rgp_probit]=gp_mc(gp_probit2, x, y, 'latent_opt', latent_opt, 'nsamples', 2000, 'repeat', 2, 'display', 100);
% rgp_probit=thin(rgp_probit,100);
% 
% f_mc_probit = rgp_probit.latentValues(:,ind);

for i=1:length(ind)
  figure;
  subplot(2,1,1);
  
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
  start=tic;
  [pc_probit(:,i) p_probit(:,i)] = gp_predcm(gp_probit,x,y,fvec_probit(:,i), 'ind', ind(i), 'correction', 'fact'); tt_epfact=toc(start);
  s = plot(fvec_probit(:,i), p_probit(:,i), '-k', fvec_probit(:,i), norm_pdf(fvec_probit(:,i), Eft_probit(ind(i)), sqrt(Varft_probit(ind(i)))),'-m', fvec_probit(:,i), pc_probit(:,i), '-r');
  set(s,'LineWidth',2)
  set(get(get(b,'Annotation'),'LegendInformation'),...
    'IconDisplayStyle','off');
  legend('EP-L', 'EP-G', 'EP-FACT');
  title('Marginal corrections for probit likelihood (EP)');

  subplot(2,1,2)

  % Predictive corrections
  [Eft_probit_pred, Varft_probit_pred] = gp_pred(gp_probit,x,y,xt);
  fvec_probit_pred(:,i) = linspace(Eft_probit_pred(ind(i))-6*sqrt(Varft_probit_pred(ind(i))), Eft_probit_pred(ind(i))+6*sqrt(Varft_probit_pred(ind(i))), 30)';
  start=tic;
  [pc_probit_pred(:,i), p_probit_pred(:,i)] = gp_predcm(gp_probit,x,y,fvec_probit_pred(:,i),xt, 'ind', ind(i), 'correction', 'fact');tt_epfact2=toc(start);
  plot(fvec_probit_pred(:,i), p_probit_pred(:,i), '-k', fvec_probit_pred(:,i), pc_probit_pred(:,i), '-r', fvec_probit_pred(:,i), mean(ptx_prob,1), '-c');
  set(s,'LineWidth',2)
  legend('EP-G', 'EP-FACT', 'MCMC');
  title('Predictive marginal corrections for probit likelihood (EP)');
end

% ---------------------------
% Probit likelihood with Laplace
% ---------------------------

% Create the GP structure (type is by default FULL)
gp_probit_laplace = gp_set(gp_probit, 'latent_method', 'Laplace');

% Index for comparison values
ind = 22;

for i=1:length(ind)
  figure; subplot(2,1,1);
  [testi, testi2] = hist(f_mc_probit(:,i),50);
  width = testi2(2)-testi2(1);
  area = sum(testi.*width);
  testi = testi./area;
  b = bar(testi2,testi);
  h = findobj(gca,'Type','patch');
  set(h,'FaceColor','w')
  hold on;
  [Eft_probit, Varft_probit] = gp_pred(gp_probit_laplace,x,y,x);
  fvec_probit_laplace(:,i) = linspace(Eft_probit(ind(i))-6*sqrt(Varft_probit(ind(i))), Eft_probit(ind(i))+6*sqrt(Varft_probit(ind(i))), 50)';
  start=tic;[pc_probit_laplace(:,i), p_probit_laplace(:,i), c_probit_laplace(:,i)] = gp_predcm(gp_probit_laplace,x,y,fvec_probit_laplace(:,i), 'ind', ind(i), 'correction', 'cm2'); tt_lacm2=toc(start); 
  start=tic;[pc_probit_laplace2(:,i), p_probit_laplace2(:,i), c_probit_laplace2(:,i)] = gp_predcm(gp_probit_laplace,x,y,fvec_probit_laplace(:,i), 'ind', ind(i), 'correction', 'fact'); tt_lafact=toc(start);
  s = plot(fvec_probit_laplace(:,i), p_probit_laplace(:,i), '-k', fvec_probit_laplace(:,i), norm_pdf(fvec_probit_laplace(:,i), Eft_probit(ind(i)), sqrt(Varft_probit(ind(i)))), '-m', fvec_probit_laplace(:,i), pc_probit_laplace2(:,i), '-r', fvec_probit_laplace(:,i), pc_probit_laplace(:,i), '-b');
  set(s,'LineWidth',2)
  set(get(get(b,'Annotation'),'LegendInformation'),...
    'IconDisplayStyle','off');
  legend('LA-L', 'LA-G', 'LA-FACT', 'LA-CM2');
  title('Marginal corrections for probit likelihood (Laplace)');

  % Predictive corrections
  subplot(2,1,2);
  [Eft_probit_pred, Varft_probit_pred] = gp_pred(gp_probit_laplace,x,y,xt);
  fvec_probit_laplace_pred(:,i) = linspace(Eft_probit_pred(ind(i))-6*sqrt(Varft_probit_pred(ind(i))), Eft_probit_pred(ind(i))+6*sqrt(Varft_probit_pred(ind(i))), 30)';
  start=tic;[pc_pred(:,i), p_pred(:,i), c_pred(:,i)] = gp_predcm(gp_probit_laplace,x,y,fvec_probit_laplace_pred(:,i),xt, 'ind', ind(i), 'correction', 'cm2'); tt_lacm22=toc(start);
  start=tic;[pc_pred2(:,i), p_pred2(:,i), c_pred2(:,i)] = gp_predcm(gp_probit_laplace,x,y,fvec_probit_laplace_pred(:,i),xt, 'ind', ind(i), 'correction', 'fact'); tt_lafact2=toc(start);
  s = plot(fvec_probit_laplace_pred(:,i), p_pred2(:,i), '-k', fvec_probit_laplace_pred(:,i), pc_pred(:,i), '-r', fvec_probit_laplace_pred(:,i), pc_pred2(:,i), '-b', fvec_probit_laplace_pred(:,i), mean(ptx_lap,1), '-c');
  legend('LA-G', 'LA-CM2', 'LA-FACT', 'MCMC');
  title('Predictive marginal corrections for probit likelihood (Laplace)');
end

fprintf('Time elapsed for marginal corrections with EP-FACT: %.1f s and for predictions %.1f s\n', tt_epfact, tt_epfact2);
fprintf('Time elapsed for marginal corrections with LA-CM2: %.1f s and for predictions %.1f s\n', tt_lacm2, tt_lacm22);
fprintf('Time elapsed for marginal corrections with LA-FACT: %.1f s and for predictions %.1f s\n', tt_lafact, tt_lafact2);
