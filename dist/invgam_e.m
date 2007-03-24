function e = invgam_e(x, a)
% INVGAM_E  compute an error term for a parameter with inverse
%           gamma distribution (single parameter).
%
%        Description
%        E = INVGAM_E(X,A) takes parameter  matrix X and 
%        hyper-parameter structure A and returns a vector
%        containing minus log from X^2 ~ INVGAM(A.s^2,A.nu) 
%        distribution  for given parameter X(:,1).

% Copyright (c) 2000 Aki Vehtari

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

x2=x.^2;nu=a.nu;s2=a.s.^2;
e = sum((nu./2-1) .* log(x2) + (s2.*nu./2./x2) + (nu/2) .* log(2./(s2.*nu))+ gammaln(nu/2)) ;
