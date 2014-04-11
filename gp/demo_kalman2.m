%Kalman Demo 2 (Mauna Loa Periodic CO2 Readings)
%
%  In this demo we apply the state space inference methods to the 
%  well-known time series data consisting of atmospheric CO2 
%  concentration readings in parts per million (ppm) by volume from 
%  air samples collected at the Mauna Loa observatory, Hawaii (see
%  [1] for details and further references).
%
%  The benefit from the state space formulation is that the 
%  computational complexity is linear with respect to the number
%  of data points. Due to efficient matrix solvers in Matlab 
%  (favoring the traditional GP solution over sequential looping),
%  the advantages in speed start to show in datasets with thousands
%  of data points.
%
%  The take-home message from this demo is that the GP can be
%  set up exactly as any other GP regression model in GPstuff, 
%  and then solved by Kalman filtering methods by specifying the
%  'type' in the GP structure to be 'KALMAN'.
%
%  The methods in this demo are based on the paper:
%
%  [1] Arno Solin and Simo Sarkka (2014). Explicit link between periodic 
%      covariance functions and state space models. Accepted for 
%      publication in Proceedings of the Seventeenth International 
%      Conference on Artifcial Intelligence and Statistics (AISTATS 2014).
%
% Copyright (c) 2014 Arno Solin and Jukka Koskenranta

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

%% Load the data

  % Load the data
  S = which('demo_kalman2.m');
  L = strrep(S,'demo_kalman2.m','demodata/maunaloa_data.txt');

  % Set training data (x = time in years, y = CO2 observations)
  data = load(L);
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
  
  % Set type to KALMAN
  gp = gp_set(gp,'type','KALMAN');
  
  
%% Optimize hyperparameters and predict

  % Optimization parameters
  opt=optimset('TolFun',1e-4,'TolX',1e-4,'Display','iter');

  % Find hyperparameters by optimization (BFGS)
  gp=gp_optim(gp,x,y,'opt',opt,'optimf',@fminlbfgs);
  
  % Set the test points
  xt = (x(end):1/12:x(end)+10)';
  
  % Predict values
  [Eft,Varft] =  gp_pred(gp, x, y,'xt',xt);
  
  % Also predict the latent components separately
  [Eft1, Varft1] = gp_pred(gp, x, y, x, 'predcf', [1 3]);
  [Eft2, Varft2] = gp_pred(gp, x, y, x, 'predcf', [2]);

  
%% Visualize results

  % Plot
  figure(2); clf; hold on
  
    % Plot the 95% confidence interval of the predictions
    p=patch([xt; flipud(xt)], ...
       [ymean + Eft + 1.96*sqrt(Varft); ...
        flipud(ymean + Eft - 1.96*sqrt(Varft))],[0.9,0.9,0.9]);
    set(p,'EdgeColor','none')
   
    % Plot observations
    plot(x,ymean+y,'.k','MarkerSize',5)

    % Labels and legends
    title('Kalman Demo 2 (Mauna Loa Periodic CO2 Readings)')
    xlabel('Time (years)');
    ylabel('CO_2 concentration (PPM)')
    legend('95% confidence region', ...
           'Monthly average measurements',...
           'Location', 'NorthWest');
    
    % Axis options
    box on; axis tight; set(gca,'Layer','top')

    
%% Show components separately
    
  figure(3); clf;
  subplot(211)
  
    plot(x,Eft1,'-k')
    
    title('Long-term trend and short-scale variation')
    xlabel('Time (years)');
    ylabel('Effect on CO_2 concentration (PPM)')
    
    axis tight
    
  subplot(212)
  
    plot(x,Eft2,'-k')
    
    title('Quasi-periodic effect')
    xlabel('Time (years)');
    ylabel('Effect on CO_2 concentration (PPM)')
    
    axis tight

