function s = norm_s(a);
%NORM_S      Maximum log likelihood second derivatives (single variable)
%
%            Description
%            S = NORM_S(A) takes a structure A, which contains
%            information about sigma of a parameter, and returns
%            the maximum log likelihood second derivative estimate.
%            Estimate is equal to:
%            s = 1./a.s.^2
%

% Copyright (c) 1998-2004 Aki Vehtari

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

s = 1./a.s.^2;
