function g = gamma_g(x, a1, a)
% GAMMA_G  Compute a gradient term with respect to a parameter of
%          gamma distribution (single parameter).
%
%        Description
%        E = GAMMA_G(X,A1,A) takes a position vector X, 
%        hyper-parameter structure A1 and string A, which defines the
%        parameter of Gamma with respect to the gradient is evaluated.
%        Function returns a vector containing derivative with respect 
%        to A of minus log from  X^2 ~ GAMMA(A.s^2, A.nu) distribution 
%        for given parameter X(:,1).

% Copyright (c) 2000 Aki Vehtari
% Copyright (c) 2006 Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

if nargin < 3
  a='x';
end
nu=a1.nu;s=a1.s;
switch a
 case 'x'
  g= 2*((1-s^2)./x + nu.*x);
 case 's'
  g=sum(2*s.*digamma1(s.^2) - 2*s.*log(nu)-2*s.*log(x.^2) );
 case 'nu'
  g=sum(-s^2./nu+x.^2);
end
