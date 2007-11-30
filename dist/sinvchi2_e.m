function e = sinvchi2_e(x, a)
% SINVCHI2_E  compute an error term for a parameter with scaled 
%             inverse-chi-squared distribution (single parameter).
%
%        Description
%        E = SINVCHI2_E(X,A) takes parameter matrix X and 
%        hyper-parameter structure A and returns a vector
%        containing minus log from X ~ S-Inv-Chi2(A.nu, A.s) 
%        distribution  for given parameter X(:,1).
%
%        Parameterisation is done by Bayesian Data Analysis,  
%        second edition, Gelman et.al 2004.

% Copyright (c) 2000 Aki Vehtari
% Copyright (c) 2006 Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

nu = a.nu; s2=a.s;
e = sum((nu./2+1) .* log(x) + (s2.*nu./2./x) + (nu/2) .* log(2./(s2.*nu))+ gammaln(nu/2)) ;