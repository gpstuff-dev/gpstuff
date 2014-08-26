function [F,L,Qc,H,Pinf,dF,dQc,dPinf,params] = cf_quasiperiodic_to_ss(magnSigma2, lengthScale, period, mlengthScale, N, nu, mN, valid)
%% CF_QUASIPERIODIC_TO_SS - Convert quasi-periodic covariance functions to continuous state space
%
% Syntax:
%   [F,L,Qc,H,Pinf,dF,dQc,dPinf,params] = 
%     cf_quasiperiodic_to_ss(magnSigma2, lengthScale, period, mlengthScale, 
%       N, nu, mN, valid)
%
% In:
%   magnSigma2   - Magnitude scale parameter (default: 1)
%   lengthScale  - Distance scale parameter (default: 1)
%   period       - Length of repetition period (default: 1)
%   mlengthScale - Matern lengthScale (default: 1)
%   N            - Degree of approximation (default: 6)
%   nu           - Matern smoothness parameter (default: 3/2)
%   mN           - Degree of approximation if squared exonential (default: 6)
%   valid        - If false, uses Bessel functions (default: false)
%
% Out:
%   F           - Feedback matrix
%   L           - Noise effect matrix
%   Qc          - Spectral density of white noise process w(t)
%   H           - Observation model matrix
%   Pinf        - Covariance of the stationary process
%   dF          - Derivatives of F w.r.t. parameters
%   dQc         - Derivatives of Qc w.r.t. parameters
%   params      - Input and output parameter information
%
% Description:
%   This function converts one-dimensional covariance functions to state 
%   space models. The covariance function parametrization is as follows
%
%     k(tau) = k_p(tau) k_m(tau),
%
%   where k_p(tau) is the so called canonical periodic covariance function
%   (see cf_periodic_to_ss for details) and k_m(tau) is a covariance
%   function of the Matern class (see Solin and Särkkä 2014 for details).
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
%   CF_PERIODIC_TO_SS, CF_MATERN_TO_SS, SPEC_QUASIPERIODIC, COV_QUASIPERIODIC
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

  % Check if Matern lengthScale is given
  if nargin < 4 || isempty(mlengthScale), mlengthScale = 1; end  
  
  % Check if N is given
  if nargin < 5 || isempty(N), N = 6; end  
  
  % Check if Matern nu is given
  if nargin < 6 || isempty(nu), nu = 3/2; end  

  % Check if Matern mN is given
  if nargin < 7 || isempty(mN), mN = 6; end  

  % Check if 'valid' is given
  if nargin < 8 || isempty(valid), valid = false; end  

  
%% Form state space models

  % Periodic covariance function
  [F1,L1,Qc1,H1,Pinf1,dF1,dQc1,dPinf1] = ...
    cf_periodic_to_ss(magnSigma2,lengthScale,period,N,valid);

  % The damping part
  if isinf(nu), % nu -> infinity

    % Form the model
    [F2,L2,Qc2,H2,Pinf2,dF2,dQc2,dPinf2] = ...
        cf_se_to_ss(1,mlengthScale,mN);
 
  else
    
    % Form the model
    [F2,L2,Qc2,H2,Pinf2,dF2,dQc2,dPinf2] = ...
        cf_matern_to_ss(1,mlengthScale,nu);  
      
  end

  
%% Combine models to one

  F    = kron(F1,eye(size(F2))) + kron(eye(size(F1)),F2);
  L    = kron(L1,L2);
  Qc   = kron(Pinf1,Qc2);   % This is ok. See paper for details
  Pinf = kron(Pinf1,Pinf2);
  H    = kron(H1,H2);

  
%% Calculate derivatives %TODO

  % Calculate derivatives only if requested
  if nargout > 5
    
    % Derivative of F w.r.t. parameter magnSigma2
    dFmagnSigma2 = kron(dF1(:,:,1),eye(size(F2)));
    
    % Derivative of F w.r.t parameter lengthScale
    dFlengthScale = kron(dF1(:,:,2),eye(size(F2)));

    % Derivative of F w.r.t parameter period
    dFperiod = kron(dF1(:,:,3),eye(size(F2)));
    
    % Derivative of F w.r.t parameter mlengthScale
    dFmlengthScale = kron(eye(size(F1)),dF2(:,:,2));
    
    
    % Derivative of Qc w.r.t. parameter magnSigma2
    dQcmagnSigma2 = kron(dPinf1(:,:,1),Qc2);
    
    % Derivative of Qc w.r.t. parameter lengthScale
    dQclengthScale = kron(dPinf1(:,:,2),Qc2);
    
    % Derivative of Qc w.r.t. parameter period
    dQcperiod = zeros(size(Qc));
    
    % Derivative of Qc w.r.t. parameter mlengthScale
    dQcmlengthScale = kron(Pinf1,dQc2(:,:,2));
    
    
    % Derivative of Pinf w.r.t. parameter magnSigma2
    dPinfmagnSigma2 = kron(dPinf1(:,:,1),Pinf2);
    
    % Derivative of Pinf w.r.t. parameter lengthScale
    dPinflengthScale = kron(dPinf1(:,:,2),Pinf2);
    
    % Derivative of Pinf w.r.t. parameter period  
    dPinfperiod = zeros(size(Pinf));

    % Derivative of Pinf w.r.t. parameter mlengthScale
    dPinfmlengthScale = kron(Pinf1,dPinf2(:,:,2));
    
    % Assign
    dF(:,:,1) = dFmagnSigma2;
    dF(:,:,2) = dFlengthScale;
    dF(:,:,3) = dFperiod;
    dF(:,:,4) = dFmlengthScale;
    
    dQc(:,:,1) = dQcmagnSigma2;
    dQc(:,:,2) = dQclengthScale;
    dQc(:,:,3) = dQcperiod;
    dQc(:,:,4) = dQcmlengthScale;
    
    dPinf(:,:,1) = dPinfmagnSigma2;
    dPinf(:,:,2) = dPinflengthScale; 
    dPinf(:,:,3) = dPinfperiod; 
    dPinf(:,:,4) = dPinfmlengthScale;    
    
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
    pa.in{4}.name = 'mlengthScale'; pa.in{4}.default = 1;     pa.in{4}.opt = true;
    pa.in{5}.name = 'N';            pa.in{5}.default = 6;     pa.in{5}.opt = false;
    pa.in{6}.name = 'nu';           pa.in{6}.default = 1/2;   pa.in{6}.opt = false;
    pa.in{7}.name = 'mN';           pa.in{7}.default = 6;     pa.in{7}.opt = false;
    pa.in{8}.name = 'valid';        pa.in{8}.default = false; pa.in{8}.opt = false;

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

