function y = gam_pdf(x,a,b)
%GAM_PDF Gamma probability density function (pdf).
%
%   Y = GAM_PDF(X,A,B) Returns the gamma pdf with
%   shape A and inverse scale B, at the values in X.
%
%   The size of X is the common size of the input arguments. A
%   scalar input functions as a constant matrix of the same size as
%   the other inputs.

% Copyright (c) 1998-2004 Aki Vehtari

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

if nargin < 3, 
  error('Requires three input arguments.');
end

%y = b.^a/gamma(a)*x^(a-1)*exp(-b*x);
y = exp(a.*log(b)-gammaln(a)+(a-1).*log(x)-b.*x);
