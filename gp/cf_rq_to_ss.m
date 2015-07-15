function [F,L,Qc,H,Pinf,dF,dQc,dPinf,params] = cf_rq_to_ss(magnSigma2, lengthScale, alpha, N, NSE)
% CF_RQ_TO_SS - Rational quadratic covariance functions to state space
%
% Syntax:
%   [F,L,Qc,H,Pinf,dF,dQc,dPinf,params] = cf_rq_to_ss(magnSigma2, lengthScale, alpha, N, NSE)
%
% In:
%   magnSigma2  - Magnitude scale parameter (default: 1)
%   lengthScale - Length scale parameter (default: 1)
%   alpha       - Shape parameter (default: 1)
%   N           - Degree of approximation (default: 6)
%   NSE         - Degree of SE approximation (default: 6)
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
%   the rational quadratic (RQ) form to state space models. The RQ
%   covariance function parametrization is as follows
%
%     k(t) = magnSigma2 exp(1 + t^2/(2 alpha lengthScale^2))^-alpha,
%
%   where magnSigma2 is the magnitude scale parameter, lengthScale the  
%   distance scale parameter, and alpha the decay parameter.
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
%   [1] Arno Solin and Simo Sarkka (2014). Gaussian quadratures for state 
%       space approximation of scale mixtures of squared exponential 
%       covariance functions. Proceedings of IEEE International Workshop 
%       on Machine Learning for Signal Processing (MLSP). Reims, France.
%
% See also:
%   COV_RQ, SPEC_RQ, SS_BALANCE
%
% Copyright:
%   2014   Arno Solin
%
%  This software is distributed under the GNU General Public
%  License (version 3 or later); please refer to the file
%  License.txt, included with the software, for details.

%% Apply defaults

  % Check if magnSigm2 is given
  if nargin < 1 || isempty(magnSigma2), magnSigma2 = 1; end

  % Check if lengthScale is given
  if nargin < 2 || isempty(lengthScale), lengthScale = 1; end 

  % Check if alpha is given
  if nargin < 3 || isempty(alpha), alpha = 1; end
    
  % Check if N is given
  if nargin < 4 || isempty(N), N = 6; end

  % Check if NSE is given
  if nargin < 5 || isempty(NSE), NSE = 6; end


%% Set up the Gauss-Laguerre quadrature rule

  % Set up the Gauss-Laguerre rule
  [x,w] = GaussLaguerre(N,alpha-1);

  % For derivatives w.r.t. alpha
  [dmagnSigma2,dlengthScale] = dalpha(N,alpha);

  % Take the current scale into account
  dmagnSigma2  = dmagnSigma2*magnSigma2;
  dlengthScale = dlengthScale*lengthScale;
  

%% Form model

  % Allocate space for matrices
  F     = [];
  L     = [];
  Qc    = [];
  H     = [];
  Pinf  = [];
  dF    = zeros(0,0,3);
  dQc   = zeros(0,0,3);
  dPinf = zeros(0,0,3);
  
  % Stack models
  for j=1:N
      
    % Parameters
    SE_magnSigma2 = magnSigma2*w(j)/gamma(alpha);
    SE_lengthScale = lengthScale*sqrt(alpha/x(j));
    
    % Make model
    [Fj,Lj,Qcj,Hj,Pinfj,dFj,dQcj,dPinfj] = ...
        cf_se_to_ss(SE_magnSigma2,SE_lengthScale, NSE);
    
    % Balance model
    [Fj,Lj,Qcj,Hj,Pinfj,dFj,dQcj,dPinfj] = ...
        ss_balance(Fj,Lj,Qcj,Hj,Pinfj,dFj,dQcj,dPinfj);
    
    % Stack
    F = blkdiag(F,Fj);
    L = blkdiag(L,Lj);
    Qc = blkdiag(Qc,Qcj);
    H = [H Hj];
    Pinf = blkdiag(Pinf,Pinfj);
    
    % Start derivative stacking
    dFs    = zeros(size(dF,1)+size(dFj,1), ...
                   size(dF,2)+size(dFj,2),3);
    dQcs   = zeros(size(dQc,1)+size(dQcj,1), ...
                   size(dQc,2)+size(dQcj,2),3);
    dPinfs = zeros(size(dPinf,1)+size(dPinfj,1), ...
                   size(dPinf,2)+size(dPinfj,2),3);
  
    % The derivatives w.r.t. magnSigma2
    scale1 = w(j)/gamma(alpha);
    dFs(:,:,1)    = blkdiag(dF(:,:,1),scale1*dFj(:,:,1));
    dQcs(:,:,1)   = blkdiag(dQc(:,:,1),scale1*dQcj(:,:,1));
    dPinfs(:,:,1) = blkdiag(dPinf(:,:,1),scale1*dPinfj(:,:,1));
    
    % The derivatives w.r.t. lengthScale
    scale2 = sqrt(alpha/x(j));
    dFs(:,:,2)    = blkdiag(dF(:,:,2),scale2*dFj(:,:,2));
    dQcs(:,:,2)   = blkdiag(dQc(:,:,2),scale2*dQcj(:,:,2));
    dPinfs(:,:,2) = blkdiag(dPinf(:,:,2),scale2*dPinfj(:,:,2));

    % The derivatives w.r.t. alpha
    dFs(:,:,3)    = blkdiag(dF(:,:,3), ...
      dFj(:,:,1)*dmagnSigma2(j) + dFj(:,:,2)*dlengthScale(j));
    dQcs(:,:,3)   = blkdiag(dQc(:,:,3), ...
      dQcj(:,:,1)*dmagnSigma2(j) + dQcj(:,:,2)*dlengthScale(j));
    dPinfs(:,:,3) = blkdiag(dPinf(:,:,3), ...
      dPinfj(:,:,1)*dmagnSigma2(j) + dPinfj(:,:,2)*dlengthScale(j));
    
    % Set derivatives
    dF = dFs; dQc = dQcs; dPinf = dPinfs; 
    
  end

  
%% Return parameter names

  % Only return if requested
  if nargout > 8
    
    % Stationarity
    pa.stationary = true;
    
    % Input parameter information
    pa.in{1}.name = 'magnSigma2';   pa.in{1}.default = 1; pa.in{1}.opt = true;
    pa.in{2}.name = 'lengthScale';  pa.in{2}.default = 1; pa.in{2}.opt = true;
    pa.in{3}.name = 'alpha';        pa.in{3}.default = 1; pa.in{3}.opt = true;
    pa.in{4}.name = 'N';            pa.in{4}.default = 6; pa.in{4}.opt = false;
    pa.in{5}.name = 'NSE';          pa.in{5}.default = 6; pa.in{5}.opt = false;
    
    % Return parameter setup
    params = pa;
    
  end
  
end


function [c] = LaguerreL(n, beta)
% Evalaute the generalized Laguerre polynomial coefficients

  % Determine the coefficients of the associated Laguerre polynomial of 
  % order n, by determining the coefficients of the characteristic 
  % polynomial of its companion matrix:
  
  % Diagonals
  i = 1:n;
  a = (2*i-1) + beta;
  b = sqrt(i(1:n-1) .* ((1:n-1) + beta));
  
  % Companion matrix
  CM = diag(a) + diag(b,1) + diag(b,-1);

  % Solve characteristic polynomial
  c = (-1)^n/factorial(n) * poly(CM);
  
end
  
function [x,w] = GaussLaguerre(n,beta) 
% Points and weights for the generalized Gauss-Laguerre quadrature 

  % Evaluate the Laguerre polynomial coefficients for degree n
  p = LaguerreL(n,beta);  

  % The abscissae are given as the roots of the Laguerre polynomial
  x = roots(p);

  % Evaluate the Laguerre polynomial coefficients for degree n+1
  p = LaguerreL(n+1,beta);  
  
  % The corresponding weights are
  w = gamma(n + beta + 1)*x ./ ...
      (factorial(n)*(n+1)^2*polyval(p,x).^2);

end

function [dmagnSigma2,dlengthScale] = dalpha(n,alpha)
% Helper function for the derivatives w.r.t. to alpha

 % Choose (Binomial coefficient)
  choose = @(n,k) gamma(n+1)./gamma(k+1)./gamma(n-k+1);

  % Generalized LaguerreL (Ln)
  Ln = @(n,alpha) (-1).^(n:-1:0).* ...
                          choose(n+alpha,0:n)./factorial(n:-1:0);

  % Derivative coefficients dLn/dalpha
  dLn = @(n,alpha) (-1).^(n:-1:0).* ...
                          choose(n+alpha,0:n)./factorial(n:-1:0).* ...
                          (digamma1(alpha+n+1) - digamma1(alpha+1+(n:-1:0)));

  % Polynomial coefficients
  c  = Ln(n, alpha-1);
  
  % Evaluate x
  x = roots(c);

  % The derivatives of x w.r.t. alpha: (dLn/dalpha)/(dLn/dx) = dx/dalpha
  dx = polyval(dLn(n, alpha-1),x)./ ...
       polyval(Ln(n-1, alpha),x);

  % Additional polynomials
  c2 = Ln(n+1, alpha-1);
  dc2 = dLn(n+1, alpha-1);

  % Evaluate dmagnSigma2/dalpha
  dmagnSigma2 = (gamma(n+alpha)*(polyval(c2,x).*dx ...
       + x.*(polyval(c2,x).*(digamma1(alpha+n) - digamma1(alpha)) ...
       + 2*polyval(Ln(n,alpha),x).*dx - 2*polyval(dc2,x))))./ ...
       ((1 + n)^2*factorial(n)*gamma(alpha).*polyval(c2,x).^3);

  % Evaluate dlengtScale/dalpha
  dlengthScale = (alpha./x).^(3/2)/(2*alpha^2).*(x-alpha*dx);

end

