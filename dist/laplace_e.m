function e = laplace_e(x, a)
%LAPLACE_E  compute an error term for a parameter with Laplace
%           distribution (single parameter). 
%
%        Description
%        E = LAPLACE_E(X,A) takes parameter  matrix X and 
%        hyper-parameter structure A and returns an error
%        vector E.

% Copyright (c) 1998-2004 Aki Vehtari

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.


e = sum(sum(abs(x),1)./a.s);
