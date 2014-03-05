%% Kalman demo 1, sinc regression

%% Generate data

  % Discretize x (measurement points)
  x = sort(pi*randn(50,1)/2);
  
  % Test points (evaluation points)
  xt = (-5:0.01:5)';
  
  % Measurement noise variance
  sigma2 = .01;
    
  % Simulate data
  yr = sinc(x);
  y = yr + sqrt(sigma2)*randn(size(x));
    
%% Do the inference

  % Choose likelihood model
  % Prior for noice variance
  ps2 = prior_logunif(); 
  lik = lik_gaussian('sigma2',Sigma2,'sigma2_prior', ps2);
  
  % Choose covariance function and its parameters and their priors
  lengthScale = 1;
  pl = prior_logunif(); 
  magnSigma2 = 1;
  pm = prior_logunif();
  gpcf = gpcf_matern52('lengthScale',lengthScale,'magnSigma2',magnSigma2,'lengthScale_prior',pl, 'magnSigma2_prior', pm);
  
  % Set Gaussian process model using type KALMAN
  gp = gp_set('lik',lik,'cf',gpcf,'type','KALMAN');

  % Find hyperparameters
  gp = gp_optim(gp,x,y);

  % Predict values
  [meanf,Varf] = gp_pred(gp,x,y,'xt',xt);
 
%% Show result

  % Plot
  figure(1); clf; hold on
  color=0.85*[1,1,1];
  p=patch([xt' fliplr(xt')], ...
      [meanf' + 1.96*sqrt(Varf') fliplr(meanf' - 1.96*sqrt(Varf'))],color);
  set(p,'EdgeColor','none')
  
  % Show data and predicted mean
  h=plot(xt', sinc(xt'),'-r', ...
       x', y','xk', ...
       xt',meanf','--', ...
       'LineWidth', 1, 'MarkerSize',5);
  legend([h;p],'Sinc function', 'Noisy measurements from sinc function', 'Estimated mean','95% condidence region');
  title('Kalman demo 1, sinc regression');
  ylabel('Output y');
  xlabel('Input x');
  
  % Bring axes to front
  set(gca,'Layer','top')
  
  
