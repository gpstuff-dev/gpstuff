function [F,L,Qc,H,Pinf,dF,dQc,dPinf,params] = cf_prod_to_ss(cf2ss)
%% CF_PROD_TO_SS - Product of several state space models
%
% Syntax:
%   [F,L,Qc,H,Pinf,dF,dQc,dPinf,params] = cf_prod_to_ss(cf2ss)
%
% In:
%   cf2ss       - Cell vector of function handles (see below)
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
%   Calculate the state space formulation corresponding the the following
%   covariance function:
%
%     k(t) = k_1(t)*k_2(t)* ... *k_n(t),
%    
%   that is, the state space model of a product of several state space
%   models. The state space model corresponding to covaraince function
%   k_j is given by the function handle in cf2ss{j} such that:
%
%     [F,L,Qc,H,Pinf,dF,dQc,dPinf] = cf2ss{k}();
%
%   This is based on the formulation explained in [1].
%
% References:
%
%   [1] Arno Solin and Simo Särkkä (2014). Explicit link between periodic 
%       covariance functions and state space models. In Proceedings of the 
%       Seventeenth International Conference on Artifcial Intelligence and 
%       Statistics (AISTATS 2014). JMLR: W&CP, volume 33.
%
% See also:
%   CF_QUASIPERIODIC_TO_SS
%
% Copyright:
%   2012-2014   Arno Solin
%   2013-2014   Jukka Koskenranta
%
%   This software is distributed under the GNU General Public
%   License (version 3 or later); please refer to the file
%   License.txt, included with the software, for details.

%% Product of two state space models by kronecker sums and products

  % Initialize model matrices
  F      = 0;
  L      = 1;
  Qc     = 1;
  H      = 1;
  Pinf   = 1;
  dF     = [];
  dQc    = [];
  dPinf  = [];
  
  % Initialize
  params.stationary = true;

  % Loop through all models
  for j = 1:length(cf2ss)
      
    % Set up the jth state space model
    [Fj,Lj,Qcj,Hj,Pinfj,dFj,dQcj,dPinfj,paramsj] = cf2ss{j}();

    % Make derivative model
    dF    = dskron(F,dF,Fj,dFj);
    dQc   = dkron(Qc,dQc,Qcj,dQcj);
    dPinf = dkron(Pinf,dPinf,Pinfj,dPinfj);
    
    % Make state space model
    F    = kron(F,eye(size(Fj))) + kron(eye(size(F)),Fj);
    L    = kron(L,Lj);
    Qc   = kron(Qc,Qcj);
    Pinf = kron(Pinf,Pinfj);
    H    = kron(H,Hj);
    
    % Parameters
    params.stationary = params.stationary & paramsj.stationary;
    
  end

end

function dC = dkron(A,dA,B,dB)
% Derivative version of the kron function

  % Get sizes
  sA=size(dA); 
  sB=size(dB);

  % Check if A or B is empty
  Ae = ~any(sA==0); Be = ~any(sB==0);
  
  % Make sure no empty krons..
  sA(1:2) = max([sA(1:2);1,1],[],1);
  sB(1:2) = max([sB(1:2);1,1],[],1);
  
  % Numel of sizes to 3
  sA = [sA Ae]; sA = sA(1:3);
  sB = [sB Be]; sB = sB(1:3);
  
  % Assign space for dC
  dC = zeros([sA(1:2).*sB(1:2),sA(3)+sB(3)]);
  
  % Set values of dA if there is any
  for k = 1:sA(3)
      dC(:,:,k) = kron(dA(:,:,k),B);
  end
  
  % Set values of dB if there is any
  for k = 1:sB(3)
      dC(:,:,sA(3)+k) = kron(A,dB(:,:,k));
  end
  
end

function dC = dskron(A,dA,B,dB)
% Derivative version of the sum of two krons

  % Get sizes
  sA=size(dA); 
  sB=size(dB);

  % Check if A or B is empty
  Ae = ~any(sA==0); Be = ~any(sB==0);
  
  % Make sure no empty krons..
  sA(1:2) = max([sA(1:2);1,1],[],1);
  sB(1:2) = max([sB(1:2);1,1],[],1);
  
  % Numel of sizes to 3
  sA = [sA Ae]; sA = sA(1:3);
  sB = [sB Be]; sB = sB(1:3);
  
  % Assign space for dC
  dC = zeros([sA(1:2).*sB(1:2),sA(3)+sB(3)]);
  
  % Set values of dA if there is any
  for k = 1:sA(3)
      dC(:,:,k) = kron(dA(:,:,k),eye(length(B)));
  end
  
  % Set values of dB if there is any
  for k = 1:sB(3)
      dC(:,:,sA(3)+k) = kron(eye(length(A)),dB(:,:,k));
  end
  
end