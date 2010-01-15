function g = sinvchi2_g(x, a1, a)
% SINVCHI2_G  compute a gradient term with respect to a parameter of scaled 
%             Inverse-Chi-Squared distribution (single parameter).
%
%        Description
%        E = SINVCHI2_G(X,A1,A) takes a position vector X, 
%        hyper-parameter structure A1 and string A, which defines the
%        parameter of Inv-gamma with respect to the gradient is evaluated.
%        Function returns a vector containing minus log from  
%        X ~ S-Inv-Chi2(A.nu, A.s) distribution for given
%        parameter X(:,1).
%
%          Parameterisation is done by Bayesian Data Analysis,  
%          second edition, Gelman et.al 2004.

% Copyright (c) 2000 Aki Vehtari
% Copyright (c) 2006 Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

if nargin < 3
  a='x';
end
nu=a1.nu;s2=a1.s2;
switch a
 case 'x'
  g = (nu/2+1)./x-nu.*s2./(2*x.^2);
 case 's2'
  g=sum(nu/2.*(1./x-1./s2));
 case 'nu'
  g=sum(0.5*(log(x) + s2./x + log(2./s2./nu) - 1 + digamma1(nu/2)));
end
