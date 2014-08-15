function [F,L,Qc,H,Pinf,dF,dQc,dPinf,params] = cf_sum_to_ss(cf2ss)  
%% CF_SUM_TO_SS - Sum of several state space models
%
% Syntax:
%   [F,L,Qc,H,Pinf,dF,dQc,dPinf,params] = cf_sum_to_ss(cf2ss)
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
%     k(t) = k_1(t) + k_2(t) + ... + k_n(t),
%    
%   that is, the state space model of a sum of several state space
%   models. The state space model corresponding to covaraince function
%   k_j is given by the function handle in cf2ss{j} such that:
%
%     [F,L,Qc,H,Pinf,dF,dQc,dPinf] = cf2ss{k}();
%
% Copyright:
%   2012-2014   Arno Solin
%   2013-2014   Jukka Koskenranta
%
%   This software is distributed under the GNU General Public
%   License (version 3 or later); please refer to the file
%   License.txt, included with the software, for details.
%
% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

%% Sums of two state space models by stacking the models

  % Initialize model matrices
  F      = [];
  L      = [];
  Qc     = [];
  H      = [];
  Pinf   = [];
  dF     = [];
  dQc    = [];
  dPinf  = [];
  params = {};
  
  % Initialize
  params.stationary = true;
  
  % Stack state space models
  for k=1:length(cf2ss)    
    
    % Form ss model j
    if nargout > 5
        [jF,jL,jQc,jH,jPinf,jdF,jdQc,jdPinf,jparams] = cf2ss{k}();
    else
        [jF,jL,jQc,jH,jPinf] = cf2ss{k}();
    end
    
    % Stack matrices
    F  = blkdiag(F,jF);
    L  = blkdiag(L,jL);
    Qc = blkdiag(Qc,jQc);
    H  = [H jH];    
    Pinf = blkdiag(Pinf,jPinf);
    
    % Stack derivative matrices
    if nargout > 5
        
      dF  = mblk(dF,jdF);
      dQc = mblk(dQc,jdQc);
      dPinf = mblk(dPinf,jdPinf);
      
      % Parameters
      params.stationary = params.stationary & jparams.stationary;

    end    
  end

end

function C = mblk(A,B)
% Three-dimensional version of the blk function

  % Get sizes
  sA=size(A); sB=size(B);

  % Check if A or B is empty
  Ae = ~any(sA==0); Be = ~any(sB==0);
  
  % Numel of sizes to 3
  sA = [sA Ae]; sA = sA(1:3);
  sB = [sB Be]; sB = sB(1:3);
  
  % Assign space for C
  C = zeros(sA+sB);
  
  % Set values of A if there is any
  if Ae
      C(1:sA(1), 1:sA(2), 1:sA(3)) = A;
  end
  
  % Set values of B if there is any
  if Be
      C(sA(1)+(1:sB(1)), ...
        sA(2)+(1:sB(2)), ...
        sA(3)+(1:sB(3))) = B;
  end

end


