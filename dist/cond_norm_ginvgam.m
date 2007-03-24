function r = cond_norm_ginvgam(a, a1, a2, x)
% COND_NORM_GINVGAM       Sample conditional distribution from
%                         normal likelihood and inverse gamma prior
%                         for a group
%
%           Description
%           R = COND_NORM_GINVGAM(A, A1, A2, X) generates one sample
%           from the conditional distribution of A given
%           parameter structure X of lower level, structure A1 of
%           same level hyper-parameters and A2 of higher level, i.e 
%           is r~P(A|A1,A2,X). Returns one new sample R from the 
%           distribution above
%
% Copyright (c) 1999-2000 Aki Vehtari

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

ii=a2.ii;
m=length(ii);
r=zeros(size(x));
nu=a2.nu;s=a2.s;
if size(s,2)>1
  for i=1:m
    xii=x(:,ii{i});
    n=size(x,1);
    nus2=nu(i)*s(i).^2;
    nun=nu(i)+n;
    for j=1:length(xii)
      r(ii{i}(j))=sqrt(invgamrand1((nus2+xii(:,j).^2)/nun,nun));
    end
  end
else
  for i=1:m
    n=length(ii{i});
    r(i)=sqrt(invgamrand1((nu*s.^2+sum(x(ii{i}).^2))/(nu+n),nu+n));
  end
end
