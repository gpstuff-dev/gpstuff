function e = gt_e(x, a)
%GT_E     compute an error term for a parameter with Student's
%        t-distribution (single parameter). 
%
%        Description
%        E = GT_E(X,A) takes parameter  matrix X and 
%        hyper-parameter structure A and returns a vector
%        containing error term E.

% Copyright (c) 1998-2004 Aki Vehtari

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

e=0;
ii=a.ii;
nu=a.nu;
s=a.s;
for i=1:length(ii)
  e = e+sum(log(1 + (x(ii{i})./s(i)).^2 ./ nu(i))).* (nu(i)+1)/2;
end
