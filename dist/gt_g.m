function g = gt_g(x, a)
%GT_G    compute a gradient for a parameter with Student's 
%        t-distribution (single parameter).
%
%        Description
%        G = GT_G(X,A) takes parameter vector X and 
%        hyper-parameter structure A and returns scalar 
%        containing gradient G.

% Copyright (c) 1998-2004 Aki Vehtari

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

g = zeros(size(x));
ii=a.ii;
nu=a.nu;
s=a.s;
for i=1:length(ii)
  d=x(ii{i})./s(i);
  g(ii{i})=(nu(i)+1)./nu(i) .* (d./s(i)) ./ (1 + (d.^2)./nu(i));
end
