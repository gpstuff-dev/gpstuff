function r = cond_laplace_invgam(a, a1, a2, x)
% COND_LAPLACE_INVGAM   Sample conditional distribution from Laplace
%                       likelihood and inverse gamma prior.
%
%       Description
%       R = COND_LAPLACE_INVGAM(A, A1, A2, X) generates one sample
%       from the conditional distribution of A given
%       parameter structure X of lower level, structure A1 of
%       same level hyper-parameters and A2 of higher level, i.e 
%       is r~P(A|A1,A2,X). Returns one new sample R from the 
%       distribution above.

% Copyright (c) 1999-2000 Aki Vehtari


% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

[n,m]=size(x);
r=zeros(1,m);
if size(a2.s,2)>1
  for i=1:m
    r(i)=invgamrand1((a2.nu*a2.s(i)+sum(abs(x(:,i))))/(a2.nu+n), a2.nu+n);
  end
else
  a2nun=a2.nu+n;
  a2nua2s2=a2.nu*a2.s;
  for i=1:m
    r(i)=invgamrand1((a2nua2s2+sum(abs(x(:,i))))./a2nun, a2nun);
  end
end
