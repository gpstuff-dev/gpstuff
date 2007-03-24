function r = cond_gt_cat(a, a1, a2, x)
% COND_GT_CAT  Sample conditional distribution from T likelihood
%              for a group and categorical prior.
%
%       Description
%       R = COND_GT_CAT(A, A1, A2, X) generates one sample
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
nus=a2.nus;
p=nus;
s=a1.s;
r=zeros(1,m);
for i=1:m
  xii=x(ii{i});
  for j=1:length(nus)
    p(j)=sum(t_lpdf(xii,nus(j),0,s(i)));
  end
  p=exp(p-max(p));
  r(i)=nus(catrand(p));
end
