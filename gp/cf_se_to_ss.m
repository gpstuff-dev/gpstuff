function [F,L,Qc,H,Pinf,dF,dQc,dPinf,params] = cf_se_to_ss (magnSigma2, lengthScale, N)
% CF_SE_TO_SS - Squared exponential covariance functions to state space
%
% Syntax:
%   [F,L,Qc,H,Pinf,dF,dQc,dPinf,params] = cf_se_to_ss(magnSigma2, lengthScale, N)
%
% In:
%   magnSigma2  - Magnitude scale parameter (default: 1)
%   lengthScale - Length scale parameter (default: 1)
%   N           - Degree of approximation (default: 6)
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
%   the square exponential to state space models. The covariance function
%   parametrization is as follows
%
%     k(t) = magnSigma2 exp(-|t|^2/(2 lengthScale^2))
%
%   where magnSigma2 is the magnitude scale parameter, lengthScale the  
%   distance scale parameter.
%     This function takes the covariance function parameters as inputs and
%   outputs the corresponding state space model matrices. The state space
%   model is given as follows in terms of a stochastic differential
%   equation
%
%      df(t)/dt = F f(t) + L w(t),
%
%   where w(t) is a white noise process with spectral denisty Qc. The
%   observation model for discrete observation y_k of f(t_k) at step k, 
%   is as follows 
%
%      y_k = H f(t_k) + r_k, r_k ~ N(0, R),
%
%   where r_k is the Gaussian measurement noise with covariance R.
%     Pinf is the stationary covariance, where the value of Pinf(i,j), 
%   is defined as follows
%   
%      Pinf(i,j) = E[(f_i(t)-E[f_i(t)])(f_j(t)-E[f_j(t)])],
%
%   where f_i(t) is component i of state vector f(t).
%     Derivatives: All have same form. For example, dF has the following
%   form:
%
%       dF(:,:,1) = dF/d(magnSigma2 = input parameter_1),
%       dF(:,:,i) = dF/d(input parameter_i).
%
%     NOTE: In case of the squared exponential covariance function, the
%   state space model is just an approximation. In this implementation
%   the approximation is constructed by a Taylor expansion. The degree
%   of the approximation may be controlled through the parameter 'N'.
%   The resulting state space model might become numerically unstable.
%   Numerical robustness may be enhanced by calling 'ss_balance'.
%
% References:
%
%   [1] Simo Sarkka, Arno Solin, Jouni Hartikainen (2013).
%       Spatiotemporal learning via infinite-dimensional Bayesian
%       filtering and smoothing. IEEE Signal Processing Magazine,
%       30(4):51-61.
%
%   [2] Jouni Hartikainen and Simo Sarkka (2010). Kalman filtering and 
%       smoothing solutions to temporal Gaussian process regression 
%       models. Proceedings of IEEE International Workshop on Machine 
%       Learning for Signal Processing (MLSP).
%
% See also:
%   COV_SE, SPEC_SE, SS_BALANCE
%
% Copyright:
%   2012-2014   Arno Solin
%   2013-2014   Jukka Koskenranta
%
%  This software is distributed under the GNU General Public
%  License (version 3 or later); please refer to the file
%  License.txt, included with the software, for details.


%% Apply defaults

  % Check if magnSigm2 is given
  if nargin < 1 || isempty(magnSigma2), magnSigma2 = 1; end

  % Check if lengthScale is given
  if nargin < 2 || isempty(lengthScale), lengthScale = 1; end 

  % Check if N is given
  if nargin < 3 || isempty(N), N = 6; end

  
%% Form state space model
  
  % Derived constants
  kappa = 1/2/lengthScale^2;
  
  % Precalculate factorial
  fn = factorial(N);
  
  % Process noise spectral density
  Qc = magnSigma2*sqrt(pi/kappa)*fn*(4*kappa)^N;
  
  % Make polynomial
  p = zeros(1,2*N+1);
  for n=0:N
      p(end - 2*n) = fn*(4*kappa)^(N-n)/factorial(n)/(-1)^(n);
  end
  
  % All the coefficients of polynomial p are real, and thus the
  % Complex Conjugate Theorem states that the roots are of form
  % a+/-ib, which means they are symmetrically distributed in the
  % complex plane.
  r = roots(p);
  a = poly(r(real(r) < 0));
  
  % Form the white noise driven state space model
  % Feedback matrix
  F = diag(ones(N-1,1),1);
  F(end,:) = -a(end:-1:2); % Controllable canonical form

  % Noise effect matrix
  L = zeros(N,1); L(end) = 1;

  % Observation model
  H = zeros(1,N); H(1) = 1;
  

%% Approximative stationary covariance (Squared exponential)
  
  % Compute Pinf only if requested
  if nargout > 4,
    
    % Solve the corresponding Lyapunov problem
    %   F*Pinf + Pinf*F' + L*Qc*L' = 0
    Pinf = lyap(F,L*Qc*L');

    % The same thing can be solved as a solution to the
    % algebraic Riccati equation (less stable) 
    %Pinf = are(F',zeros(size(F)),L*Qc*L');

  end
  
  
%% Calculate derivatives

  % Calculate derivatives only if requested
  if nargout > 5

    % Derivative of F w.r.t. parameter magnSigma2
    dFmagnSigma2 = zeros(size(F));
    
    % Derivative of F w.r.t parameter lengthScale
    dFlengthScale = zeros(size(F));
    dFlengthScale(end,:) = -a(end:-1:2)/lengthScale.*(-N:1:-1);
    
    % Derivative of Qc w.r.t. parameter magnSigma2
    dQcmagnSigma2 = sqrt(pi/kappa)*fn*(4*kappa)^N;
    
    % Derivative of Qc w.r.t. parameter lengthScale
    dQclengthScale = magnSigma2*sqrt(2*pi)*fn*2^N*lengthScale^(-2*N)*(1-2*N);
    
    % Derivative of Pinf w.r.t. parameter magnSigma2
    dPinfmagnSigma2 = Pinf/magnSigma2;
    
    % Derivative of Pinf w.r.t. parameter lengthScale
    lp=size(Pinf,1);
    coef = bsxfun(@plus,1:lp,(1:lp)')-2;
    coef(mod(coef,2)~=0)=0;
    dPinflengthScale = -1/lengthScale*Pinf.*coef;
    
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
    
    % Stationarity
    pa.stationary = true;
    
    % Input parameter information
    pa.in{1}.name = 'magnSigma2';   pa.in{1}.default = 1; pa.in{1}.opt = true;
    pa.in{2}.name = 'lengthScale';  pa.in{2}.default = 1; pa.in{2}.opt = true;
    pa.in{3}.name = 'N';            pa.in{3}.default = 6; pa.in{3}.opt = false;
    
    % Return parameter setup
    params = pa;
    
  end
  
  
