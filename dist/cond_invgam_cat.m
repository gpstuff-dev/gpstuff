function r = cond_invgam_cat(a, a1, a2, x)
% COND_INVGAM_CAT      Sample conditional distribution from
%                      inverse gamma likelihood and categorical
%                      prior. 
%
%       Description
%       R = COND_INVGAM_CAT(A, A1, A2, X) generates one sample
%       from the conditional distribution of A given
%       parameter structure X of lower level, structure A1 of
%       same level hyper-parameters and A2 of higher level, i.e 
%       is r~P(A|A1,A2,X). Returns one new sample R from the 
%       distribution above

% Copyright (c) 1999-2000 Aki Vehtari

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

x2=x.^2;s2=a1.s.^2;
nus=a2.nus;
p=nus;
for i=1:length(nus)
  p(i)=sum(invgam_lpdf(x2,s2,nus(i)));
end
p=exp(p-max(p));
r=nus(catrand(p));
