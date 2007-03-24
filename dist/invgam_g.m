function g = invgam_g(x, a1, a)
% INVGAM_G  compute a gradient term with respect to a parameter of inverse
%           gamma distribution (single parameter).
%
%        Description
%        E = INVGAM_G(X,A1,A) takes a position vector X, 
%        hyper-parameter structure A1 and string A, which defines the
%        parameter of Inv-gamma with respect to the gradient is evaluated.
%        Function eturns a vector containing minus log from  
%        X^2 ~ INVGAM(A.s^2, A.nu) distribution for given
%         parameter X(:,1).

% Copyright (c) 2000 Aki Vehtari

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

if nargin < 3
  a='x';
end
nu=a1.nu;s=a1.s;
switch a
 case 'x'
  g=((nu-2) - s.^2.*nu./x.^2)./x;
 case 's'
  g=sum(nu.*(s./x.^2 -1./s));
 case 'nu'
  x2=x.^2;s2=s.^2;
  g=sum(0.5*(log(x2) + s2./x2 + log(2./s2./nu) - 1 + digamma1(nu/2)));
end
