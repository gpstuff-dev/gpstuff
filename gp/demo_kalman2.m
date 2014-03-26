%% Kalman Demo 2  (Mauna Loa Periodic CO2 Readings)
%
%
%
%
%% Load the data

  % Load the data
  S = which('demo_kalman2.m');
  L = strrep(S,'demo_kalman2.m','demodata/maunaloa_data.txt');

  % Set training data (x = time in years, y = CO2 observations)
  data=load(L);
  y = data(:, 2:13)';  y=y(:);
  x = (data(1,1):1/12:data(end,1)+11/12)';
  
  % Remove data points with missing information
  x = x(y>0); y = y(y>0);
  
  % Make data zero mean
  ymean = mean(y); y = y-ymean;
  
  % Show original data
  figure(1); clf
    plot(x,y+ymean,'-k')
    xlabel('Time (year)');
    ylabel('CO_2 concentration (PPM)')
    title('Kalman Demo 2  (Mauna Loa Periodic CO2 Readings) - Data')
  
  
%% Construct the model

  % We use a squared exponential function to deal with long term change.
  gpcf1 = gpcf_sexp('lengthScale', 100, 'magnSigma2', 5000);
  
  % Matern52 deals with short term effects
  gpcf3 = gpcf_matern52('lengthScale', 10, 'magnSigma2', 10);
  
  % Periodic covariance function multiplied with matern52 deals with
  % cyclic nature of the data. We use decay 0, because we want to use
  % matern52 as a decay function instead of default squared exponential. 
  gpcf21 = gpcf_periodic('decay',0,'magnSigma2',1,'period',1,'lengthScale',.1);
  gpcf22 = gpcf_matern52('lengthScale', 10, 'magnSigma2', 10);
    
  % We do not optimize period, because we strongly believe it is one
  % year. We use same t-distribution as a priori for all of the other 
  % covariance function hyperparameters
  pl = prior_t('s2', 10,'nu',3);
  gpcf1 = gpcf_sexp(gpcf1, 'lengthScale_prior', pl, 'magnSigma2_prior', pl);
  gpcf3 = gpcf_matern52(gpcf3, 'lengthScale_prior', pl, 'magnSigma2_prior', pl);
  gpcf21 = gpcf_periodic(gpcf21, 'lengthScale_prior', pl,...
      'magnSigma2_prior', [], 'period_prior',[]);
  gpcf22 = gpcf_matern52(gpcf22, 'lengthScale_prior', pl, 'magnSigma2_prior', pl);
  
  % Then we multiply the periodic covariance function and matern52
  % covariance functions
  gpcf2 = gpcf_prod('cf',{gpcf21,gpcf22});
  
  % We use Gaussian likelihood with t-distribution as a priori
  pn = prior_t('s2', 10,'nu',4);
  lik = lik_gaussian('sigma2_prior',pn);
  
  % Finally create the GP structure
  gp = gp_set('lik', lik, 'cf', {gpcf1,gpcf2,gpcf3});
  
  % Set used type to KALMAN
  gp = gp_set(gp,'type','KALMAN');
  
  
%% Set up GP model

  % Noise variance prior
  ps2 = prior_logunif(); 

  % The likelihood model
  lik = lik_gaussian('sigma2', 1, 'sigma2_prior', ps2);
  
  % Covariance function hyperparameter priors
  pl = prior_logunif(); 
  pm = prior_logunif();
  
  % A squared exponential covariance function 
  % to deal with the smooth long term effects
  gpcf1 = gpcf_sexp('lengthScale', 100, 'magnSigma2', 5000, ...
                    'lengthScale_prior',pl,'magnSigma2_prior',pl);

  % A quasi-periodic covariance function deals with peridic 
  % variation in the data. The quasi-periodic covariance function 
  % is a product of a periodic covariance function and a squared
  % exponential. 
  gpcf2 = gpcf_periodic('magnSigma2',1,'lengthScale',1,'period',1, ...
                        'decay',1,'lengthScale_sexp',100, ...
                        'lengthScale_prior',pl,'magnSigma2_prior',pl, ...
                        'lengthScale_sexp_prior',pl);

                
  % A Matern52 covariance function deals with short term
  % non-periodic effects that remain otherwise unexplained
  gpcf3 = gpcf_matern52('lengthScale', 10, 'magnSigma2', 10, ...
                        'lengthScale_prior',pl,'magnSigma2_prior',pl);

  % Finally create the GP structure
  gp = gp_set('lik', lik, 'cf', {gpcf1,gpcf2,gpcf3});
  
  % Set used type to KALMAN
  gp = gp_set(gp,'type','KALMAN');

  
%% Optimize hyperparameters and predict

  % Find hyperparameters
  opt=optimset('TolFun',1e-4,'TolX',1e-4,'Display','iter','GradObj','off');
  gp=gp_optim(gp,x,y,'opt',opt,'optimf',@fminlbfgs);
  
  % Set the predicted area
  xt = (x(end):1/12:x(end)+ 10)';
  
  % Predict values
  [meanf,Varf] =  gp_pred(gp, x, y,'xt',xt);
  
  % Predict also the latent components separately
  [Eft_full1, Varft_full1] = gp_pred(gp, x, y, x, 'predcf', 1);
  [Eft_full2, Varft_full2] = gp_pred(gp, x, y, x, 'predcf', [2 3]);
  
 
%% Show result

  % Plot
  figure(2); clf; hold on
  p=patch([xt' fliplr(xt')], ...
      [meanf' + 1.96*sqrt(Varf') fliplr(meanf' - 1.96*sqrt(Varf'))],[0.9,0.9,0.9]);
  set(p,'EdgeColor','none')
  plot(x,y,'.:k','markersize', 6)
  xlabel('Time (years)');
  ylabel('CO_2 concentration (PPM)')
  box on
  legend('95% confidence region', 'Monthly average measurements',...
      'Location', 'NorthWest');

  % Title rows 1-5
  caption1 = sprintf('Kalman demo 2, Periodic Mauna Loa - Prediction \n');
  caption2 = sprintf('sexp+periodic*matern52+matern52+noise: \n');
  caption3 = sprintf('l_1= %.2f, s^2_1 = %.2f, \n ',... 
      gp.cf{1}.lengthScale, gp.cf{1}.magnSigma2);
  caption4 = sprintf(['l_{2,per}= %.2f, s^2_2 = %.2f,',...
                      ' p=%.2f, l_{2,mat52} = %.2f, \n '],... 
      gp.cf{2}.cf{1}.lengthScale, gp.cf{2}.cf{2}.magnSigma2,...
      gp.cf{2}.cf{1}.period, gp.cf{2}.cf{2}.lengthScale);
  caption5 = sprintf('l_3= %.2f, s^2_3 = %.2f, \n s^2_{noise} = %.2f',... 
      gp.cf{3}.lengthScale, gp.cf{3}.magnSigma2, gp.lik.sigma2);
  
  % Set title
  title([caption1,caption2,caption3,caption4,caption5])
  
  figure(3); clf;
  [AX, H1, H2] = plotyy(x, Eft_full2, x, Eft_full1);
  set(H2,'LineStyle','--')
  set(H2, 'LineWidth', 2)
  set(H1,'LineStyle','-')
  set(H1, 'LineWidth', 0.8)
  title('The long and short term latent component')
  
