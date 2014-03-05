%% Kalman demo 2, Periodic Mauna Loa


%% Load the data

  % Load the data
  S = which('demo_periodic');
  L = strrep(S,'demo_periodic.m','demodata/maunaloa_data.txt');
  
  data=load(L);
  y = data(:, 2:13);
  y=y';
  y=y(:);
  x = (data(1):1/12:data(end,1)+11/12)';
  x = x(y>0);
  y = y(y>0);
  
  % Show original data
  figure(1); clf
    plot(x,y,'-k')
    xlabel('Time (year)');
    ylabel('CO_2 concentration (PPM)')
  
  
%% Construct the model

  % First create squared exponential covariance function with ARD and 
  % Gaussian noise structures...
  gpcf1 = gpcf_sexp('lengthScale', 100, 'magnSigma2', 100);
  gpcf2 = gpcf_sexp('lengthScale', 1, 'magnSigma2', 1);
  gpcf3 = gpcf_periodic('decay',0,'magnSigma2',10,'period',1,'lengthScale',1);
  lik = lik_gaussian();
  
  % ... Then set the prior for the parameters of covariance functions...
  pl = prior_t('s2', 3);
  pm = prior_sqrtunif();
  gpcf1 = gpcf_sexp(gpcf1, 'lengthScale_prior', pl, 'magnSigma2_prior', pm);
  gpcf2 = gpcf_sexp(gpcf2, 'lengthScale_prior', pl, 'magnSigma2_prior', pm);
  gpcf3 = gpcf_sexp(gpcf3, 'lengthScale_prior', pl, 'magnSigma2_prior', pm);
  
  % ... Finally create the GP structure
  gp = gp_set('lik', lik, 'cf', {gpcf1,gpcf2,gpcf3});
  
  % Set used type to KALMAN
%   gp = gp_set(gp,'type','KALMAN');
  
%% Optimize hyperparameters and predict

  % Find hyperparameters
  opt=optimset('TolFun',1e-4,'TolX',1e-4,'Display','iter','GradObj','off');
  gp=gp_optim(gp,x,y,'optimf',@fminunc);
  
  % Set the predicted area
  xt = (x(end):1/12:x(end)+ 20)';
  
  % Predict values
  [meanf,Varf] =  gp_pred(gp, x, y,'xt',xt);
 
%% Show result

  % Plot
  figure(2); clf; hold on
  p=patch([xt' fliplr(xt')], ...
      [meanf' + 1.96*sqrt(Varf') fliplr(meanf' - 1.96*sqrt(Varf'))],[0.9,0.9,0.9]);
  set(p,'EdgeColor','none')
  plot(x,y,'.k')
  xlabel('Time (years)');
  ylabel('CO_2 concentration (PPM)')
  legend('95% confidence region', 'Monthly average measurements');
  
    
  
