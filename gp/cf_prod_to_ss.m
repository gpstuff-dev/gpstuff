function [F,L,Qc,H,Pinf,dF,dQc,dPinf,params] = cf_prod_to_ss(cf2ss)
% CF_PROD_TO_SS - 
% Syntax:
%   [F,L,Qc,H,Pinf,dF,dQc,dPinf,params] = cf_prod_to_ss(cf2ss)
%
% In:
%   cf2ss       - Cell vector of function handles,
%                 [Fj,Lj,Qcj,Hj,Pinfj,dFj,dQcj,dPinfj] = cf2ss{k}();
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

% Initialize model matrices
F      = 0;
L      = 1;
Qc     = 1;
H      = 1;
Pinf   = 1;
dF     = [];
dQc    = [];
dPinf  = [];
params = {};

for k = 1:length(cf2ss)
    [Fj,Lj,Qcj,Hj,Pinfj,dFj,dQcj,dPinfj] = cf2ss{k}();
    
    dF    = dskron(F,dF,Fj,dFj);
    dQc   = dkron(Qc,dQc,Qcj,dQcj);
%     dQc   = dkron(Pinf,dPinf,Qcj,dQcj); % case if other is periodic?
    dPinf = dkron(Pinf,dPinf,Pinfj,dPinfj);
    
    F    = kron(F,eye(size(Fj))) + kron(eye(size(F)),Fj);
    L    = kron(L,Lj);
    Qc   = kron(Qc,Qcj);
%     Qc   = kron(Pinf,Qcj); % case if other is periodic?
    Pinf = kron(Pinf,Pinfj);
    H    = kron(H,Hj);
end
end

function dC = dkron(A,dA,B,dB)
% Derivative version of kron function

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
% Derivative version of sum of two krons

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