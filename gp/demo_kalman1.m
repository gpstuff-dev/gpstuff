%% Kalman Demo 1 (Sinc Regression)
%
% [... add description here ...]
%
% The implementation that is used below is primarily based on the
% methods presented in the following publication. If you find this
% useful as a part of your own research, please cite the paper.
%
% Simo Sarkka, Arno Solin, Jouni Hartikainen (2013).
%   Spatiotemporal Learning via Infinite-Dimensional Bayesian
%   Filtering and Smoothing. IEEE Signal Processing Magazine,
%   30(4):51-61.
%
% Copyright (c) 2014 Arno Solin and Jukka Koskenranta
%
% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.
%
%% Generate data
  
  % Discretize x (measurement points)
  x = pi*randn(128,1)/2;
  
  % Test points (evaluation points)
  xt = linspace(-5,5,128)';
  
  % Measurement noise variance
  sigma2 = .01;
  
  % Simulate data
  y = sinc(x) + sqrt(sigma2)*randn(size(x));
  
  
%% Do the inference

  % Noise variance prior
  ps2 = prior_logunif(); 

  % The likelihood model
  lik = lik_gaussian('sigma2', 1, 'sigma2_prior', ps2);
  
  % Covariance function hyperparameter priors
  pl = prior_logunif(); 
  pm = prior_logunif();
  
  % The GP covariance function
  gpcf = gpcf_matern52('lengthScale', 1, ...
                       'magnSigma2', 1, ...
                       'lengthScale_prior', pl, ...
                       'magnSigma2_prior', pm);
  
  % Define Gaussian process model using type 'KALMAN'
  gp = gp_set('lik', lik, 'cf', gpcf, 'type', 'KALMAN');

  % Hyperparameter optimization (full model)
  gp_full = gp_optim(gp_set(gp,'type','FULL'), x, y);
  
  % Hyperparameter optimization (state space model)
  gp = gp_optim(gp, x, y);
  
  % Predict values at test inputs xt
  [meanf,Varf] = gp_pred(gp, x, y, 'xt', xt);
  
  
%% Compare against full GP solution (table)

  % Show table with comparison to full GP results
  fprintf('\n%12s | %8s | %8s \n', ...
      'Parameter','FULL', 'KALMAN')
  
  fprintf('-----------------------------------\n')
  
  fprintf('%12s | %8.4f | %8.4f \n', ...
      'magnSigma2',gp_full.cf{1}.magnSigma2,gp.cf{1}.magnSigma2)
  fprintf('%12s | %8.4f | %8.4f \n', ...
      'lengthScale',gp_full.cf{1}.lengthScale,gp.cf{1}.lengthScale)
  fprintf('%12s | %8.4f | %8.4f \n', ...
      'sigma2',gp_full.lik.sigma2,gp.lik.sigma2)
  
  
%% Show result

  % Plot the state space prediction
  figure(1); clf; hold on
  
    % Draw 95% uncertainty interval
    color=0.85*[1,1,1];
    p=patch([xt; flipud(xt)], ...
            [meanf+1.96*sqrt(Varf); flipud(meanf-1.96*sqrt(Varf))],color);
    set(p,'EdgeColor','none')
  
    % Show data and predicted mean
    h=plot(xt,sinc(xt),'-r', ...
           x,y,'+k', ...
           xt,meanf,'--','LineWidth', 1, 'MarkerSize',5);
       
    % Legend and labels   
    xlabel('Input, x');
    ylabel('Output, y');
    title('Kalman Demo 1 (Sinc Regression)');
    legend([h;p], 'Sinc function', ...
                  'Noisy measurements', ...
                  'Estimated mean', ...
                  '95% confidence region');
  
    % Bring axes to front
    box on; set(gca,'Layer','top')
  
    
%% Compare the full posteriors

  % Return the posterior mean EFT and covariance COVFT of latent 
  % variables:
  %     Eft =  E[f | xt,x,y,th]  = K_fy*(Kyy+s^2I)^(-1)*y
  %   Covft = Var[f | xt,x,y,th] = K_fy - K_fy*(Kyy+s^2I)^(-1)*K_yf.

  figure(2); clf
  
  % The full GP
  subplot(121)
  
    % Calculate
    [Eft,Covft] = gp_jpred(gp_full,x,y,xt);  
  
    % Plot
    imagesc(Covft)
    axis equal tight

    % Title
    title('The full GP posterior covariance')
    
  % The state space model ('KALMAN')
  subplot(122)
  
    % Calculate
    [Eft,Covft] = gp_jpred(gp,x,y,xt);  
  
    % Plot
    imagesc(Covft)
    axis equal tight
    
    % Title
    title('The state space GP posterior covariance')
    