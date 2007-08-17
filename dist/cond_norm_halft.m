function r = cond_norm_halft(a, a1, a2, x)
% COND_NORM_HALFT       Sample conditional distribution from
%                       normal likelihood and half-t prior
%
%           Description
%           R = COND_NORM_HALFT(A, A1, A2, X) generates one sample
%           from the conditional distribution of A given
%           parameter structure X of lower level, structure A1 of
%           same level hyper-parameters and A2 of higher level, i.e 
%           is r~P(A|A1,A2,X). Returns one new sample R from the 
%           distribution above
%

% Copyright (c) 1999-2000,2007 Aki Vehtari

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

[n,m]=size(x);
r=zeros(1,m);
opt.maxiter=50;
if size(a2.s,2)>1
  for i1=1:m
    opt.mmlimits=[eps 20*a2.s(i1)];
    %r(i1)=sqrt(halftrand1((a2.nu*a2.s(i1).^2+sumsqr(x(:,i1)))/(a2.nu+n), a2.nu+n));
    r(i1)=median(a1.s);
    for i2=1:5
      r(i1)=sls1mm(@condnormhalft,r(i1),opt,[],x(:,i1),a2.s(i1),a2.nu);
    end
  end
else
  a2nun=a2.nu+n;
  a2nua2s2=a2.nu*a2.s.^2;
  opt.mmlimits=[eps 20*a2.s];
  r=r+median(a1.s);
  for i1=1:m
    for i2=1:5
      r(i1)=sls1mm(@condnormhalft,r(i1),opt,[],x(:,i1),a2.s,a2.nu);
    end
%    r(i1)=sqrt(halftrand1((a2nua2s2+sum(x(:,i1).^2))./a2nun, a2nun));
  end
end

function e = condnormhalft(r,x,s,nu)
% CONDNORMHALFT - 
e=-norm_lpdf(x,0,r)-t_lpdf(r,nu,0,s);

