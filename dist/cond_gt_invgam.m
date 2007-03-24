function r = cond_gt_invgam(a, a1, a2, x)
% COND_GT_INVGAM  Sample conditional distribution from t likelihood
%                 for a group and inverse gamma prior.
%
%       Description
%       R = COND_GT_INVGAM(A, A1, A2, X) generates one sample
%       from the conditional distribution of A given
%       parameter structure X of lower level, structure A1 of
%       same level hyper-parameters and A2 of higher level, i.e 
%       is r~P(A|A1,A2,X). Returns one new sample R from the 
%       distribution above.

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
nu=a1.nu;
s=a1.s;
for i=1:m
  xii=x(ii{i});
  n=length(xii);
  ss=zeros(1,n);
  for j=1:n
    ss(j)=invgamrand1((nu(i)*s(i).^2 + xii(j).^2)/(nu(i)+1),nu(i)+1);
  end
  r(i)=sqrt(cond_invgam_invgam1(a2.s.^2,a2.nu,nu(i),hmean(ss),n));
end
