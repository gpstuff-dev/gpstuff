function e = gamma_e(x, a)
% GAMMA_E  compute an error term for a parameter with
%           gamma distribution (single parameter).
%
%        Description
%        E = GAMMA_E(X,A) takes parameter  matrix X and 
%        hyper-parameter structure A and returns a vector
%        containing minus log from X^2 ~ Gamma(A.s^2, A.nu) distribution 
%        for given parameter X(:,1).

% Copyright (c) 2000 Aki Vehtari
% Copyright (c) 2006 Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

x2=x.^2; nu=a.nu; s2=a.s.^2;
e = sum(nu.*x2 - (s2-1).*log(x2) -s2.*log(nu)  + gammaln(s2)) ;
