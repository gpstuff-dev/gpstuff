function [F,L,Qc,H,Pinf,dF,dQc,dPinf,params] = cf_periodic_to_ss(magnSigma2, lengthScale, period, N, valid)
%% CF_PERIODIC_TO_SS - Convert periodic covariance functions to continuous state space
%
% Syntax:
%   [F,L,Qc,H,Pinf,dF,dQc,dPinf,params] = 
%           cf_periodic_to_ss (magnSigma2, lengthScale, period, N, valid)
%
% In:
%   magnSigma2  - Magnitude scale parameter (default: 1)
%   lengthScale - Distance scale parameter (default: 1)
%   period      - Repetition period (default: 1)
%   N           - Degree of approximation (default: 6)
%   valid       - If false, uses Bessel function (default: false)
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
%   This function converts the so-called canonical periodic covariance
%   function to a state space model. The covariance function is
%   parameterized as follows:
%
%     k(tau) = magnSigma2 exp(-2 [sin(pi*tau/period)]^2/lengthScale^2)
%
%   where magnSigma2 is the magnitude scale parameter, lengthScale the  
%   distance scale parameter and period the repetition period length.
%   The parameter N is the degree of the approximation (see the reference
%   for details).
%     This function takes the covariance function parameters as inputs and
%   outputs the corresponding state space model matrices. The state space
%   model is given as follows in terms of a stochastic differential
%   equation
%
%      df(t)/dt = F f(t) + L w(t),
%
%   where w(t) is a white noise process with spectral denisty Qc.
%   The observation model for discrete observations y_k of f(t_k) at 
%   step k is as follows 
%
%      y_k = H f(t_k) + r_k, r_k ~ N(0, R),
%
%   where r_k is the Gaussian measurement noise wit covariance R.
%
% References:
%   [1] Arno Solin and Simo Särkkä (2014). Explicit link between periodic 
%       covariance functions and state space models. In Proceedings of the 
%       Seventeenth International Conference on Artifcial Intelligence and 
%       Statistics (AISTATS 2014). JMLR: W&CP, volume 33.
%
% See also:
%   COV_PERIODIC, SPEC_PERIODIC
%
% Copyright:
%   2012-2014 Arno Solin
%
%  This software is distributed under the GNU General Public
%  License (version 3 or later); please refer to the file
%  License.txt, included with the software, for details.

%% Apply defaults

  % Check if magnSigm2 is given
  if nargin < 1 || isempty(magnSigma2), magnSigma2 = 1; end

  % Check if lengthScale is given
  if nargin < 2 || isempty(lengthScale), lengthScale = 1; end 

  % Check if period (~frequency) is given
  if nargin < 3 || isempty(period), period = 1; end
  
  % Check if N is given
  if nargin < 4 || isempty(N), N = 6; end  
  
  % Check if 'valid' is given
  if nargin < 5 || isempty(valid), valid = false; end  
  
  
%% Form state space model

  % The series coefficients
  [q2,dq2l] = seriescoeff(N,lengthScale,magnSigma2,valid);
  
  % The angular frequency
  w0   = 2*pi/period;
  
  % The model
  F    = kron(diag(0:N),[0 -w0; w0 0]);
  L    = eye(2*(N+1));
  Qc   = zeros(2*(N+1));
  Pinf = kron(diag(q2),eye(2));
  H    = kron(ones(1,N+1),[1 0]);
  
  
%% Calculate derivatives

  % Calculate derivatives only if requested
  if nargout > 5
    
    % Derivative of F w.r.t. parameter magnSigma2
    dFmagnSigma2 = zeros(size(F));
    
    % Derivative of F w.r.t parameter lengthScale
    dFlengthScale = zeros(size(F));

    % Derivative of F w.r.t parameter period
    dFperiod = -kron(diag((0:N)/period^2),[0 -2*pi; 2*pi 0]);
    
    
    % Derivative of Qc w.r.t. parameter magnSigma2
    dQcmagnSigma2 = zeros(size(Qc));
    
    % Derivative of Qc w.r.t. parameter lengthScale
    dQclengthScale = zeros(size(Qc));
    
    % Derivative of Qc w.r.t. parameter period
    dQcperiod = zeros(size(Qc));
    
    
    % Derivative of Pinf w.r.t. parameter magnSigma2
    dPinfmagnSigma2 = Pinf/magnSigma2;
    
    % Derivative of Pinf w.r.t. parameter lengthScale
    dPinflengthScale = kron(diag(dq2l),eye(2));
    
    % Derivative of Pinf w.r.t. parameter period  
    dPinfperiod = zeros(size(Pinf));
    
    % Assign
    dF(:,:,1) = dFmagnSigma2;
    dF(:,:,2) = dFlengthScale;
    dF(:,:,3) = dFperiod;    
    
    dQc(:,:,1) = dQcmagnSigma2;
    dQc(:,:,2) = dQclengthScale;
    dQc(:,:,3) = dQcperiod;
    
    dPinf(:,:,1) = dPinfmagnSigma2;
    dPinf(:,:,2) = dPinflengthScale; 
    dPinf(:,:,3) = dPinfperiod; 
    
  end
  
  
%% Return parameter names

  % Only return if requested
  if nargout > 8
    
    % Stationarity
    pa.stationary = true;
    
    % Input parameter information
    pa.in{1}.name = 'magnSigma2';   pa.in{1}.default = 1;     pa.in{1}.opt = true;
    pa.in{2}.name = 'lengthScale';  pa.in{2}.default = 1;     pa.in{2}.opt = true;
    pa.in{3}.name = 'period';       pa.in{3}.default = 1;     pa.in{3}.opt = true;
    pa.in{4}.name = 'N';            pa.in{4}.default = 6;     pa.in{4}.opt = false;
    pa.in{5}.name = 'valid';        pa.in{5}.default = false; pa.in{5}.opt = false;
    
    % Sizes of output matrices
    pa.out.size = [size(F);
                   size(L);
                   size(Qc);
                   size(H);
                   size(Pinf);
                   0,   0;
                   0,   0;
                   0,   0];
    
    % Return parameter setup
    params = pa;
    
  end
  
  
end


function [a,dal] = seriescoeff(m,lengthScale,magnSigma2,valid_covariance)
%% Calculate the coefficients q_j^2 for the approximation
  
  % Check defaults
  if nargin<2 || isempty(lengthScale),      lengthScale=1;          end
  if nargin<3 || isempty(magnSigma2),       magnSigma2=1;           end
  if nargin<4 || isempty(valid_covariance), valid_covariance=false; end
  
  if (valid_covariance==true)
  
    % Set up the coefficients for cos(t)^k in terms of \sum_j b_j cos(j*t)
    b = @(k,j) 2.*choose(k,floor((k-j)/2).*(j<=k)) ./ ...
             (1+(j==0)*1) .* (j<=k) .* (mod(k-j,2)==0);
  
    % Set up mesh of indices
    [J,K] = meshgrid(0:m,0:m);
  
    % Calculate the coefficients
    a = b(K,J)                .* ...
        lengthScale.^(-2*K)   .* ...
        1./factorial(K)       .* ...
        exp(-1/lengthScale^2) .* ...
        2.^-K                 .* ...
        magnSigma2;
  
    % Derivatives of a w.r.t. lengthScale
    dal = sum(a.*lengthScale^-3.*(4-2*K*lengthScale^2),1);
  
    % Sum over the Js
    a = sum(a,1);
  
  else
    
    % Alternate approach by Bessel functions (NOTE!!!)
    a = 2*magnSigma2*exp(-lengthScale^-2)*besseli(0:m,1/lengthScale^2);
    a(1) = .5*a(1);

    % The derivatives
    dal = zeros(1,m+1);
    dal(2:end) = magnSigma2*lengthScale^-3/exp(lengthScale^-2)* ...
       (-4*besseli(0:m-1,lengthScale^-2) + ...
         4*(1+(1:m)*lengthScale^2).*besseli(1:m,lengthScale^-2));    
        
    % The first element
    dal(1) = magnSigma2*lengthScale^-3/exp(lengthScale^-2)* ...
        (2*besseli(0,lengthScale^-2) - ...
         2*besseli(1,lengthScale^-2));     
         
  end
    
end

function m = choose(n,k)

  if all(size(n)==size(k))
    N = n;
    K = k;
  else
    K = repmat(k,[size(n,1) size(n,2)]);  
    N = repmat(n,[size(k,1) size(k,2)]);  
  end

  m = factorial(N)./factorial(K)./factorial(N-K);

end