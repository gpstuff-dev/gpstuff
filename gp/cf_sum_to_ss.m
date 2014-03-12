function [F,L,Qc,H,Pinf,dF,dQc,dPinf,params] = cf_sum_to_ss(cf2ss)  
% CF_SUM_TO_SS - Multiple temporal SS forms into a one model
%
% Syntax:
%   [F,L,Qc,H,P0,dF,dQc,dPinf,dR,Hs] = cf_sum_to_ss(cf2ss)  
%
% In:
%   cf2ss       - Cell vector of function handles,
%                 [Fj,Lj,Qcj,Hj,Pinfj,dFj,dQcj,dPinfj] = cf2ss{k}();  
%
% Out:
%   F           - Feedback matrix for superposition model
%   L           - Noise effect matrix for superposition model
%   Qc          - Spectral density of white noise process w(t) for superposition model
%   H           - Observation model matrix for superposition model
%   Pinf        - Covariance of the stationary process for superposition model
%   dF          - Derivatives of F w.r.t. parameters for superposition model
%   dQc         - Derivatives of Qc w.r.t. parameters for superposition model
%   dPinf       - Derivatives of Pinf w.r.t. parameters for superposition model
%   Hs          - Componentwise observation model matrix 
%
% Copyright:
%   2014      Jukka Koskenranta

  F      = [];
  L      = [];
  Qc     = [];
  H      = [];
  Pinf   = [];
  dF     = [];
  dQc    = [];
  dPinf  = [];
  params = {};
  
  % Stack state space models
  for k=1:length(cf2ss)    
    
    % Form ss model j
    if nargout > 5
        [jF,jL,jQc,jH,jPinf,jdF,jdQc,jdPinf] = cf2ss{k}();
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
      
    end    
  end
end


function C = mblk(A,B)
% 3 dimensional version of blk function

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


