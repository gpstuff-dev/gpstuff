function y = geo_lpdf(x,p)
%GEOPDF Geometric log probability density function (lpdf).
%   Y = GEO_LPDF(X,P) returns the log of geometric pdf with probability, P, 
%   at the values in X.
%
%   The size of Y is the common size of X and P. A scalar input   
%   functions as a constant matrix of the same size as the other input.    

%   References:
%      [1]  M. Abramowitz and I. A. Stegun, "Handbook of Mathematical
%      Functions", Government Printing Office, 1964, 26.1.24.

%   Copyright 1993-2000 The MathWorks, Inc. 
%   $Revision: 2.8 $  $Date: 2000/05/26 18:52:54 $

%if nargin < 2, 
%   error('Requires two input arguments.'); 
%end

%[errorcode x p] = distchck(2,x,p);

%if errorcode > 0
%   error('Requires non-scalar arguments to match in size.');
%end

y = log(p) + log(1-p) * x;
