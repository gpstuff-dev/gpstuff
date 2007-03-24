function g = norm_g(x, a)
% NORM_G  compute a gradient for a parameter with normal 
%         distribution (single parameter).
%
%        Description
%        E = NORM_G(X,A) takes parameter vector X and 
%        hyper-parameter structure A and returns scalar 
%        containing gradient from minus log from scaled 
%        N(0,A) distribution for given parameter X(:,1).

% Copyright (c) 1998-2004 Aki Vehtari

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

g = x*diag(1./a.s.^2);
