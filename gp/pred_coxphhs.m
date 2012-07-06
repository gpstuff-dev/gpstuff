function [mh,mS] = pred_coxphhs(gp, x, y, xt, varargin)
%PRED_COXPHHS  Return hazard and survival functions
%
%  Description
%    [H,S] = PRED_COXPHHS(GP,X,Y,XT)
%

% Copyright (c) 2012 Ville Tolvanen, Aki Vehtari

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

[Ef1, Ef2, Covf] = pred_coxph(gp,x,y,xt, varargin{:});
nsamps = 10000;
ntime=size(gp.lik.stime,2)-1;
sd=gp.lik.stime(2)-gp.lik.stime(1);
Sigm_tmp=Covf;
Sigm_tmp=(Sigm_tmp+Sigm_tmp')./2;
% f_star=mvnrnd(Ef1, Sigm_tmp(1:ntime,1:ntime), nsamps);
f_star=mvnrnd([Ef1;Ef2], Sigm_tmp, nsamps);

f1=f_star(:,1:ntime);
f2=f_star(:,(ntime+1):end);

la1=exp(f1);
eta2=exp(f2);

mST=zeros(size(eta2,2),1);
hb=(la1'*sd);
cumsumtmp=cumsum(hb)';
for i1=1:size(eta2,2)
  mh(i1,:)=mean(bsxfun(@times,hb',eta2(:,i1)));
  Stime=exp(-bsxfun(@times,cumsumtmp,eta2(:,i1)));
  mS(i1,:)=mean(Stime);
end


end
