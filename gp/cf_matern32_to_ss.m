function [F,L,Qc,H,Pinf,dF,dQc,dPinf,params] = cf_matern32_to_ss(magnSigma2, lengthScale)
% CF_MATERN32_TO_SS - Matern covariance functions, nu = 3/2, to state space
%
% Syntax:
%   [F,L,Qc,H,Pinf,dF,dQc,dPinf,params] = cf_matern32_to_ss(magnSigma2, lengthScale)
%
% In:
%   magnSigma2  - Matern magnitude scale parameter (default: 1)
%   lengthScale - Matern distance scale parameter (default: 1)
%
% Out:
%   F           - Feedback matrix
%   L           - Noise effect matrix
%   Qc          - Spectral density of white noise process w(t)
%   H           - Observation model matrix
%   Pinf        - Covariance of the stationary process
%   dF          - Derivatives of F w.r.t. parameters
%   dQc         - Derivatives of Qc w.r.t. parameters
%   dPinf       - Derivatives of Pinf w.r.t. parameters
%   params      - Input and output parameter information
%
% Description:
%   This function converts one-dimensional covariance functions of
%   the Matern class to state space models. The covariance function
%   parametrization is as follows
%
%     k(tau) = magnSigma2 (1+sqrt(3) |tau|/lengthScale) exp(-sqrt(3) |tau|/lengthScale),
%
%   where magnSigma2 is the magnitude scale parameter, lengthScale the  
%   distance scale parameter, nu the smoothness scale parameter and tau 
%   time difference between states, tau = t-t'.
%     This function takes the covariance function parameters as inputs and
%   outputs the corresponding state space model matrices. The state space
%   model is given as follows in terms of a stochastic differential
%   equation
%
%      df(t)/dt = F f(t) + L w(t),
%
%   where w(t) is a white noise process with spectral denisty Qc.
%   Observation model for discrete observation y_k of f(t_k) at step k, 
%   is as follows 
%
%      y_k = H f(t_k) + r_k, r_k ~ N(0, R) 
%
%   where r_k is the Gaussian measurement noise wit covariance R.
%     Pinf is the stationary covariance between states. Value of Pinf at 
%   the column j and row i, Pinf(i,j), is defined as follow,
%   
%      Pinf(i,j) = E[(f_i(t)-E[f_i(t)])(f_j(t)-E[f_j(t)])],
%
%   where f_i(t) is component i of state vector f(t).
%     Derivatives: All have same form, for example, dF has the following
%   form:
%
%       dF(:,:,1) = dF/d(magnSigma2 = input parameter_1),
%       dF(:,:,i) = dF/d(input parameter_i),
%
%   where d is partial derivative. All derivatives (d*) has 3 dimensions.
%
% See also:
%   COV_MATERN32, SPEC_MATERN32
%
% Copyright:
%   2012-2013 Arno Solin
%   2013      Jukka Koskenranta
%
%% Apply defaults

  % Check if magnSigm2 is given
  if nargin < 1 || isempty(magnSigma2), magnSigma2 = 1; end

  % Check if lengthScale is given
  if nargin < 2 || isempty(lengthScale), lengthScale = 1; end
  
%% Form state space model
  
  % Derived constants
  lambda = sqrt(3)/lengthScale;
  
  % Feedback matrix
  F = [0,          1;
       -lambda^2,  -2*lambda];

  % Noise effect matrix
  L = [0;   1];

  % Spectral density
  Qc = 12*sqrt(3)/lengthScale^3*magnSigma2;

  % Observation model
  H = [1,   0];
  

%% Stationary covariance
  
  % Calculate Pinf only if requested
  if nargout > 4,
      
    Pinf = [magnSigma2, 0;
            0,          3*magnSigma2/lengthScale^2];
        
  end
  
  
%% Calculate derivatives

  % Calculate derivatives only if requested
  if nargout > 5
    
    % Derivative of F w.r.t. parameter magnSigma2
    dFmagnSigma2    =  [0,  0;
                        0,  0];
    
    % Derivative of F w.r.t parameter lengthScale
    dFlengthScale   =  [0,                  0;
                        6/lengthScale^3,    2*sqrt(3)/lengthScale^2];
    % Derivative of Qc w.r.t. parameter magnSigma2
    dQcmagnSigma2   =   12*sqrt(3)/lengthScale^3;
    
    % Derivative of Qc w.r.t. parameter lengthScale
    dQclengthScale  =   -3*12*sqrt(3)/lengthScale^4*magnSigma2;
    
    % Derivative of Pinf w.r.t. parameter magnSigma2
    dPinfmagnSigma2 = [1,0;
                       0,3/lengthScale^2];
    
    % Derivative of Pinf w.r.t. parameter lengthScale
    dPinflengthScale = [0,0;
                        0,-6*magnSigma2/lengthScale^3];
  
    % Stack all derivatives
    dF = zeros(2,2,2);  
    dQc = zeros(1,1,2); 
    dPinf = zeros(2,2,2);
    
    dF(:,:,1) = dFmagnSigma2;
    dF(:,:,2) = dFlengthScale;
    dQc(:,:,1) = dQcmagnSigma2;
    dQc(:,:,2) = dQclengthScale;
    dPinf(:,:,1) = dPinfmagnSigma2;
    dPinf(:,:,2) = dPinflengthScale;
  
  end
  
  
%% Return parameter names

  % Only return if requested
  if nargout > 8
        
    % Input parameter information
    p.in{1}.name = 'magnSigma2';  p.in{1}.default = 1; p.in{1}.opt = true;
    p.in{2}.name = 'lengthScale'; p.in{2}.default = 1; p.in{2}.opt = true;
    
    % Return parameter setup
    params = p;
    
  end

  
