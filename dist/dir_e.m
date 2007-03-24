function lp=dir_e(a,x)      
%DIR_E   compute an error term for a parameter with Dirichlet
%        distribution (single parameter). 
%
%        Description
%        LP = DIR_E(A,X) takes parameter  matrix X and 
%        hyper-parameter structure A and returns an error
%        vector LP 

% Copyright (c) 2000 Aki Vehtari

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

if any(a<0)
  lp=500;
  return
end
lp=-(gammaln(sum(a))-sum(gammaln(a))+sum(log(x).*(a-1)));
