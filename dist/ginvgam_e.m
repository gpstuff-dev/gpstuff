function e = ginvgam_e(x, a)
%GINVGAM_E  compute an error term for a parameter with inverse
%           gamma distribution (single parameter).
%
%        Description
%        E = GINVGAM_E(X,A) takes parameter  matrix X and 
%        hyper-parameter structure A and returns a vector
%        containing minus log from INVGAM(A.s,A.nu) distribution 
%        for given parameter X(:,1).

% Copyright (c) 1998-2004 Aki Vehtari

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.


nu=a.nu;s=a.s;
ii=a.ii;
e=0;
for i=1:length(ii)
  xi=x(ii{i});
  ai.nu=nu(i);
  ai.s=s(i);
  e = e + invgam_e(xi, ai);
end
