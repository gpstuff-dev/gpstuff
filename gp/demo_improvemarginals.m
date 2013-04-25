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
%   Reference
%     Cseke & Heskes (2011). Approximate Marginals in Latent Gaussian
%     Models. Journal of Machine Learning Research 12 (2011), 417-454
%
%  See also
%    GP_PREDCM, DEMO_IMPROVEMARGINALS2
%

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
gp = gp_set('lik', lik, 'cf', gpcf, 'jitterSigma2', 1e-9);

% Set the approximate inference method 
gp = gp_set(gp, 'latent_method', 'EP');

ind = 22;
ng=50;
ngt=30;
pc_ep=zeros(ng, 1); p_ep=zeros(ng,1); c_ep=zeros(ng,1); fvec_ep=zeros(ng,1);
pc_ep_pred=zeros(ngt, 1); p_ep_pred=zeros(ngt,1); c_ep_pred=zeros(ngt,1); fvec_ep_pred=zeros(ngt,1);

% If we didnt load previously computed samples we would run the following
% to get the MCMC samples for latents

% gp2 = gp_set(gp, 'latent_method', 'MCMC');
% 
% % set MC options
% latent_opt.repeat=10;
% 
% % obtain MC samples
% rgp=gp_mc(gp2, x, y, 'latent_opt', latent_opt, 'nsamples', 4000, 'repeat', 2, 'display', 100);
% rgp=thin(rgp,100);
% 
% f_mc = rgp.latentValues(:,ind);
f_mc = f_mc_probit;
for i=1:length(ind)
  figure;
  subplot(2,1,1);
  
  [testi, testi2] = hist(f_mc(:,i),50);
  width = testi2(2)-testi2(1);
  area = sum(testi.*width);
  testi = testi./area;
  b = bar(testi2,testi);
  h = findobj(gca,'Type','patch');
  set(h,'FaceColor','w')
  hold on;
  [Eft_ep, Varft_ep] = gp_pred(gp,x,y,x);
  start=tic;
  [pc_ep(:,i), fvec_ep(:,i), p_ep(:,i)] = gp_predcm(gp,x,y,'ind', ind(i), 'fcorr', 'fact'); tt_epfact=toc(start);
  s = plot(fvec_ep(:,i), p_ep(:,i), '-k', fvec_ep(:,i), norm_pdf(fvec_ep(:,i), Eft_ep(ind(i)), sqrt(Varft_ep(ind(i)))),'-m', fvec_ep(:,i), pc_ep(:,i), '-r');
  set(s,'LineWidth',2)
  set(get(get(b,'Annotation'),'LegendInformation'),...
    'IconDisplayStyle','off');
  legend('EP-L', 'EP-G', 'EP-FACT');
  title('Marginal corrections for probit likelihood (EP)');

  subplot(2,1,2)

  % Predictive corrections
  [Eft_ep_pred, Varft_ep_pred] = gp_pred(gp,x,y,xt);
  start=tic;
  [pc_ep_pred(:,i), fvec_ep_pred(:,i), p_ep_pred(:,i)] = gp_predcm(gp,x,y,xt, 'ind', ind(i), 'fcorr', 'fact', 'ng', ngt);tt_epfact2=toc(start);
  
%   if ~exist(p_mc, 'var')
%     % If sampled before
%     [Ef_mc, Varf_mc]=gpmc_preds(rgp, x, y, xt(ind,:));
%     p_mc=[];
%     for i2=1:size(rgp.etr,1)
%       p_mc=[p_mc norm_pdf(fvec_ep_pred(:,i), Ef_mc(:,i2), sqrt(Varf_mc(:,i2)))]; 
%     end
%     p_mc=mean(p_mc,2);
%   end
  
  plot(fvec_ep_pred(:,i), p_ep_pred(:,i), '-k', fvec_ep_pred(:,i), pc_ep_pred(:,i), '-r', fvec_ep_pred(:,i), ptx_prob, '-c');
  set(s,'LineWidth',2)
  legend('EP-G', 'EP-FACT', 'MCMC');
  title('Predictive marginal corrections for probit likelihood (EP)');
end

% ---------------------------
% Probit likelihood with Laplace
% ---------------------------

% Create the GP structure (type is by default FULL)
gp = gp_set(gp, 'latent_method', 'Laplace');

% Index for comparison values
ind = 22;
pc_la=zeros(ng, 1); p_la=zeros(ng,1); c_la=zeros(ng,1); fvec_la=zeros(ng,1);
pc_la_pred=zeros(ngt, 1); p_la_pred=zeros(ngt,1); c_la_pred=zeros(ngt,1); fvec_la_pred=zeros(ngt,1);
pc_la2=zeros(ng, 1); p_la2=zeros(ng,1); c_la2=zeros(ng,1); fvec_la2=zeros(ng,1);
pc_la_pred2=zeros(ngt, 1); p_la_pred2=zeros(ngt,1); c_la_pred2=zeros(ngt,1); fvec_la_pred2=zeros(ngt,1);

for i=1:length(ind)
  figure; subplot(2,1,1);
  [testi, testi2] = hist(f_mc(:,i),50);
  width = testi2(2)-testi2(1);
  area = sum(testi.*width);
  testi = testi./area;
  b = bar(testi2,testi);
  h = findobj(gca,'Type','patch');
  set(h,'FaceColor','w')
  hold on;
  [Eft_la, Varft_la] = gp_pred(gp,x,y,x);
  start=tic;[pc_la(:,i), fvec_la(:,i), p_la(:,i), c_la(:,i)] = gp_predcm(gp,x,y,'ind', ind(i), 'fcorr', 'cm2'); tt_lacm2=toc(start); 
  start=tic;[pc_la2(:,i), fvec_la(:,i), p_la2(:,i), c_la2(:,i)] = gp_predcm(gp,x,y,'ind', ind(i), 'fcorr', 'fact'); tt_lafact=toc(start);
  s = plot(fvec_la(:,i), p_la(:,i), '-k', fvec_la(:,i), norm_pdf(fvec_la(:,i), Eft_la(ind(i)), sqrt(Varft_la(ind(i)))), '-m', ...
          fvec_la(:,i), pc_la2(:,i), '-r', fvec_la(:,i), pc_la(:,i), '-b');
  set(s,'LineWidth',2)
  set(get(get(b,'Annotation'),'LegendInformation'),...
    'IconDisplayStyle','off');
  legend('LA-L', 'LA-G', 'LA-FACT', 'LA-CM2');
  title('Marginal corrections for probit likelihood (Laplace)');

  % Predictive corrections
  subplot(2,1,2);
  [Eft_la_pred, Varft_la_pred] = gp_pred(gp,x,y,xt);
  start=tic;[pc_la_pred(:,i), fvec_la_pred(:,i), p_la_pred(:,i), c_la_pred(:,i)] = gp_predcm(gp,x,y,xt, 'ind', ind(i), 'fcorr', 'cm2', 'ng', 30); tt_lacm22=toc(start);
  start=tic;[pc_la_pred2(:,i), fvec_la_pred(:,i), p_la_pred2(:,i), c_la_pred2(:,i)] = gp_predcm(gp,x,y,xt, 'ind', ind(i), 'fcorr', 'fact','ng', 30); tt_lafact2=toc(start);
  s = plot(fvec_la_pred(:,i), p_la_pred2(:,i), '-k', fvec_la_pred(:,i), pc_la_pred(:,i), '-r', fvec_la_pred(:,i), pc_la_pred2(:,i), '-b', fvec_la_pred(:,i), ptx_lap, '-c');
  legend('LA-G', 'LA-CM2', 'LA-FACT', 'MCMC');
  title('Predictive marginal corrections for probit likelihood (Laplace)');
end

fprintf('Time elapsed for marginal corrections with EP-FACT: %.1f s and for predictions %.1f s\n', tt_epfact, tt_epfact2);
fprintf('Time elapsed for marginal corrections with LA-CM2: %.1f s and for predictions %.1f s\n', tt_lacm2, tt_lacm22);
fprintf('Time elapsed for marginal corrections with LA-FACT: %.1f s and for predictions %.1f s\n', tt_lafact, tt_lafact2);
