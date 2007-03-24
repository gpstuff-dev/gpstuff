function g = gnorm_g(x, a)
% GNORM_G  compute a gradient for a parameter with normal 
%         distribution (single parameter).
%
%        Description
%        E = GNORM_G(X,A) takes parameter vector X and 
%        hyper-parameter structure A and returns scalar 
%        containing gradient from minus log from scaled 
%        N(0,A) distribution for given parameter X(:,1).

% Copyright (c) Aki Vehtari

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

g = zeros(size(x));
s2=a.s.^2;
ii=a.ii;
for i=1:length(ii)
  g(ii{i})=x(ii{i})./s2(i);
end
