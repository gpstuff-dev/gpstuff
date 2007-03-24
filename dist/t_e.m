function e = t_e(x, a, a2)
%T_E     compute an error term for a parameter with Student's
%        t-distribution (single parameter). 
%
%        Description
%        E = T_E(X,A) takes parameter  matrix X and 
%        hyper-parameter structure A and returns a vector
%        containing error term E.

% Copyright (c) 1998-2004 Aki Vehtari

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

e = sum(log(1 + (x./a.s).^2 ./ a.nu)).* (a.nu+1)/2;
