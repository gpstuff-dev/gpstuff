function r = cond_ginvgam_cat(a, a1, a2, x)
% COND_GINVGAM_CAT     Sample conditional distribution from inverse
%                      gamma likelihood for a group and categorical prior.
%
%       Description
%       R = COND_GINVGAM_CAT(A, A1, A2, X) generates one sample
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
s=a1.s;
nus=a2.nus;
p=nus;
r=zeros(1,m);
for i=1:m
  xi2=x(ii{i}).^2;
  s2=s(i).^2;
  for j=1:length(nus)
    p(j)=sum(invgam_lpdf(xi2,s2,nus(j)));
  end
  p=exp(p-max(p));
  r(i)=nus(catrand(p));
end
