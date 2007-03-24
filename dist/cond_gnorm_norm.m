function r = cond_gnorm_norm(a, a1, a2, x)
% COND_GNORM_NORM      Sample conditional distribution from normal
%                      likelihood for a group and normal prior.
%
%
%           Description
%           R = COND_GNORM_NORM(A, A1, A2, X) generates one sample
%           from the conditional distribution of A given
%           parameter structure X of lower level, structure A1 of
%           same level hyper-parameters and A2 of higher level, i.e 
%           is r~P(A|A1,A2,X). Returns one new sample R from the 
%           distribution above.

% Copyright (c) 1999-2000 Aki Vehtari


% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

ii=a1.ii;
m=length(ii);
if size(a2.s,2)<m
  a2.s=a2.s(ones(1,m));
end
r=zeros(1,m);
for i=1:m
  n=length(ii{i});
  p1=n./a1.s(i).^2;p2=1./a2.s(i).^2;
  r(i)=(p2*a2.mu+p1*mean(x(ii{i})))./(p1+p2)+randn(1)*sqrt(1./(p1+p2));
end
