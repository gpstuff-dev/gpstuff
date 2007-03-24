function e = norm_e(x, a)
% NORM_E compute an error term for a parameter with normal
%        distribution (single parameter). 
%
%        Description
%        E = NORM_E(X,A) takes parameter  matrix X and 
%        hyper-parameter structure A and returns a vector
%        containing minus log from scaled N(0,A) distribution 
%        for given parameter X(:,1).

% Copyright (c) 1998-2004 Aki Vehtari

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

e = 0.5*sum(1./a.s.^2 .* sum(x.^2,1));
