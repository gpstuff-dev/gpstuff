function r = cond_t_invgam(a, a1, a2, x)
% COND_T_INVGAM    Sample conditional distribution from T
%                  likelihood and inverse gamma prior.
%
%           Description
%           R = COND_T_GINVGAM(A, A1, A2, X) generates one sample
%           from the conditional distribution of A given
%           parameter structure X of lower level, structure A1 of
%           same level hyper-parameters and A2 of higher level, i.e 
%           is r~P(A|A1,A2,X). Returns one new sample R from the 
%           distribution above.

% Copyright (c) 1999-2000 Aki Vehtari

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

n=size(x,1);
m=size(a1.s,2);
if size(a2.s,2)<m
  a2.s=a2.s(ones(1,m));
end
r=zeros(1,m);
s=zeros(1,n);
for i=1:m
  for j=1:n
    s(j)=invgamrand1((a1.nu*a1.s.^2 + x(j).^2)/(a1.nu+1),a1.nu+1);
  end
  r(i)=sqrt(cond_invgam_invgam1(a2.s.^2,a2.nu,a1.nu,hmean(s),n));
end
