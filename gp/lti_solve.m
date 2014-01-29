function [A,Q] = lti_disc(F,L,Q,dt)
%LTI_DISC - Discretize LTI ODE with Gaussian Noise
%
% Syntax:
%   [A,Q] = lti_disc(F,L,Qc,dt)
%
% In:
%   F  - NxN Feedback matrix
%   L  - NxL Noise effect matrix        (optional, default identity)
%   Qc - LxL Diagonal Spectral Density  (optional, default zeros)
%   dt - Time Step                      (optional, default 1)
%
% Out:
%   A - Transition matrix
%   Q - Discrete Process Covariance
%
% Description:
%   Discretize LTI ODE with Gaussian Noise. The original
%   ODE model is in form
%
%     dx/dt = F x + L w,  w ~ N(0,Qc)
%
%   Result of discretization is the model
%
%     x[k] = A x[k-1] + q, q ~ N(0,Q)
%
%   Which can be used for integrating the model
%   exactly over time steps, which are multiples
%   of dt.

% History:
%   11.01.2003  Covariance propagation by matrix fractions
%   20.11.2002  The first official version.
%
% Copyright (C) 2002, 2003 Simo S�rkk�
%
% $Id: lti_disc.m 111 2007-09-04 12:09:23Z ssarkka $
%
% This software is distributed under the GNU General Public 
% Licence (version 2 or later); please refer to the file 
% Licence.txt, included with the software, for details.

  %
  % Check number of arguments
  %
  if nargin < 1
    error('Too few arguments');
  end
  if nargin < 2
    L = [];
  end
  if nargin < 3
    Q = [];
  end
  if nargin < 4
    dt = [];
  end

  if isempty(L)
    L = eye(size(F,1));
  end
  if isempty(Q)
    Q = zeros(size(F,1),size(F,1));
  end
  if isempty(dt)
    dt = 1;
  end

  %
  % Closed form integration of transition matrix
  %
  A = expm(F*dt);

  %
  % Closed form integration of covariance
  % by matrix fraction decomposition
  %
  n   = size(F,1);
  Phi = [F L*Q*L'; zeros(n,n) -F'];
  AB  = expm(Phi*dt)*[zeros(n,n);eye(n)];
  Q   = AB(1:n,:)/AB((n+1):(2*n),:);
