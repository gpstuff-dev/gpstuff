function e = gnorm_e(x, a)
%GNORM_E compute an error term for a parameter with normal
%        distribution (single parameter). 
%
%        Description
%        E = GNORM_E(X,A) takes parameter  matrix X and 
%        hyper-parameter structure A and returns a vector
%        containing minus log from scaled N(0,A) distribution 
%        for given parameter X(:,1).

% Copyright (c) 1998-2004 Aki Vehtari

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

e=0;
s2=a.s.^2;
ii=a.ii;
for i=1:length(ii)
  e=e+0.5./s2(i) .* sum(x(ii{i}).^2,1);
end
