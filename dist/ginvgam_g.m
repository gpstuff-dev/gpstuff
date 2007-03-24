function g = ginvgam_g(x, a1, a)
%GINVGAM_G  compute a gradient term for a parameter with inverse
%           gamma distribution (single parameter).
%
%        Description
%        E = GINVGAM_G(X,A) takes parameter  matrix X and 
%        hyper-parameter structure A and returns a vector
%        containing minus log from INVGAM(A.s,A.nu) distribution 
%        for given parameter X(:,1).

% Copyright (c) 1998-2004 Aki Vehtari

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.


nu=a1.nu;s=a1.s;
ii=a1.ii;
for i=1:length(ii)
  ai.nu=nu(i);
  ai.s=s(i);
  xi=x(ii{i});
  switch a
   case x
    g(ii{i})=invgam_g(xi, ai, a);
   otherwise
    g(i)=invgam_g(xi, ai, a);
  end
end
