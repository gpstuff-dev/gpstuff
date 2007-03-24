function g = laplace_g(x, a)
%LAPLACE_G   compute a gradient for a parameter with Laplace 
%            distribution (single parameter).
%
%        Description
%        E = LAPLACE_G(X,A) takes parameter vector X and 
%        hyper-parameter structure A and returns a gradient 
%        scalar G

% Copyright (c) 1998-2004 Aki Vehtari

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.


g=sign(x)./repmat(a.s,size(x,1),1);
