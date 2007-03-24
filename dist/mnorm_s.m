function s = mnorm_s(a);
%MNORM_S       Maximum log likelihood second derivatives (multiple variables)
%
%              Description
%              S = MNORM_S takes a structure A, which contains
%              information about sigmas of parameters, and returns
%              maximum log likelihood second derivative estimation.
%              Estimate is equal to:
%              s = diag(inv(a.s^2))'
%

% Copyright (c) 1998-2004 Aki Vehtari

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

s = diag(inv(a.s^2))';
