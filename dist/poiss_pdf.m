function y = poiss_pdf(x,l)
%POISS_PDF Poisson probability density function.
%
%   Description
%   Y = POISS_PDF(X,L) returns the Poisson probability density 
%   function with parameter L at the values in X.
%
%   The size of Y is the common size of X and L. A scalar input   
%   functions as a constant matrix of the same size as the other input.    
%
%   Note that the density function is zero unless X is an integer.
%
%   See also POISS_LPDF

% Copyright (c) 1998-2004 Aki Vehtari

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

y = zeros(size(x));
y(l < 0) = NaN;

k = (x >= 0 & x == round(x) & l >= 0);
if (any(k))
  y(k) = exp(-l(k) +x(k).*log(l(k) + realmin.*(l(k)==0)) -gammaln(x(k)+1));
end
