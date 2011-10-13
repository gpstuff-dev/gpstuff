function p = pred_coxphp(gp, x, y, xt, pp, varargin)
%PRED_COXPH     Integrates the model from zero to point pp (when time is 
%   scaled to interval 0-1)
%
%  Description
%    P = PRED_COXPHP(GP,X,Y,XT,PP)
%    Integrates the model from zero to point pp with respect to time.
%    Return P, the probability that event has happened before time pp. PP
%    is vector of size 1xM(or Mx1) indicating times in scaled interval ~ Unif(0,1).
%    Returns matrix P of size NxM where columns correspond to points in
%    PP and rows correspond to rows in X (e.g. people).
%

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

if size(pp,1) == 1
  pp=pp';
end
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
cumsumtmp=cumsum(la1'*sd)';
%   t=binsgeq(gp.lik.xtime,pp(i));
for i1=1:size(eta2,2)
  Stime=exp(-bsxfun(@times,cumsumtmp,eta2(:,i1)));
  mStime=mean(Stime);
  for i=1:size(pp,1)
    mST(i1,i)=1-mStime(binsgeq(gp.lik.xtime,pp(i)));
  end
end
p = mST;


end
