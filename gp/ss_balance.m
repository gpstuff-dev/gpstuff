function [NF,NL,NQc,NH,NPinf,NdF,NdQc,NdPinf,T] = ss_balance(F,L,Qc,H,Pinf,dF,dQc,dPinf)
% SS_BALANCE - Balance state space model for improved numerical stability
%
% Syntax:
%   [F,L,Qc,H,Pinf,dF,dQc,dPinf,T] = ss_balance(F,L,Qc,H,Pinf,dF,dQc,dPinf)
%
% In:
%   F           - Feedback matrix
%   L           - Noise effect matrix
%   Qc          - Spectral density of white noise process w(t)
%   H           - Observation model matrix
%   Pinf        - Covariance of the stationary process
%   dF          - Derivatives of F w.r.t. parameters
%   dQc         - Derivatives of Qc w.r.t. parameters
%   dPinf       - Derivatives of Pinf w.r.t. parameters
%
% Out:
%   (...)       - As above, but balanced
%   T           - T is the balancing matrix from 'balance'
%
% Description:
%   This function takes the SDE state space model matrices as inputs and
%   outputs the same matrices, but in a numerically more stable form. 
%   This balancing is based on the Matlab 'balance' function (see the
%   reference).
%
%   The state space model is given as follows in terms of a stochastic 
%   differential equation
%
%      df(t)/dt = F f(t) + L w(t),
%
%   where w(t) is a white noise process with spectral denisty Qc. The 
%   observation model matrix is denoted by matrix H.
%
% References:
%   [1] Beresford N. Parlett and Christian Reinsch (1969). Balancing 
%       a matrix for calculation of eigenvalues and eigenvectors. 
%       Numerische Mathematik, 13(4): 293-304.
%
% See also:
%   BALANCE
%
% Copyright:
%   2014 Arno Solin and Simo Sarkka
%
%  This software is distributed under the GNU General Public
%  License (version 3 or later); please refer to the file
%  License.txt, included with the software, for details.

%% Balance the state space model

  % This is based on the following:
  %
  %  dx/dt = F x + L w
  %      y = H x
  %
  % Let T z = x, which gives
  %
  %  dz/dt = inv(T) F T z + inv(T) L w
  %      y = H T z
  %
  % See also: [A,B,C,E,S,P] = aebalance(A,B,C,E)
  
  % Balance the dynamic model matrix
  [T,NF] = balance(F);
  
  % Balance noise effect matrix
  NL = T\L;
  
  % Balance the measurement model
  NH = H*T;
  
  % Balance spectral density
  NQc = Qc;
  
  % Balance stationary state covariance matrix
  if nargin < 5
    NPinf = [];
  else
    L = chol(Pinf,'lower');
    LL = T\L;
    NPinf = LL*LL';
    %NPinf = T\Pinf/T;
  end
  
  % Balance partial derivatives in F
  if nargin < 6
    NdF = [];
  else
    NdF = dF;
    for j=1:size(dF,3)
      NdF(:,:,j) = T\dF(:,:,j)*T;
    end
  end
  
  % Balane partial derivatives in dQc;
  if nargin < 7
    NdQc = [];
  else
    NdQc = dQc;
  end
  
  % Balance partial derivatives in dF
  if nargin < 8
    NdPinf = [];
  else
    NdPinf = dPinf;
    for j=1:size(dF,3)
      NdPinf(:,:,j) = T\dPinf(:,:,j)/T;
    end
  end
