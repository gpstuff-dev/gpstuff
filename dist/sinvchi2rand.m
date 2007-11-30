function r = sinvchi2rand(nu, s2, M, N)
% SINVCHI2RAND  Random matrices from scaled inverse-chi distribution
%
%  R = SINVCHI2RAND(NU, S2)
%  R = SINVCHI2RAND(NU, S2, M, N)
%
%  Returns a randon number/matrix R from scaled inverse-chi square 
%  distribution. Nu is the degrees of freedom and S2 is the scale 
%  squared. Parametrisation is according to Gelman et. al. (2004).

% Copyright (c) 1998-2004 Aki Vehtari

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

if nargin < 2
  error('Too few arguments');
end
if nargin < 3
  M=1;
end
if nargin < 4
  N=1;
end
r=nu*s2./chi2rnd(nu,M,N);