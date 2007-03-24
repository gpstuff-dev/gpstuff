function g = t_g(x, a, a2)
%T_G     compute a gradient for a parameter with Student's 
%        t-distribution (single parameter).
%
%        Description
%        E = T_G(X,A) takes parameter vector X and 
%        hyper-parameter structure A and returns scalar 
%        containing gradient G.

% Copyright (c) 1998-2004 Aki Vehtari

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.n

d=x./a.s;
g=(a.nu+1)./a.nu .* (d./a.s) ./ (1 + (d.^2)./a.nu);
