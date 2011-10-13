function p = pred_coxphp(gp, x, y, xt, pp, varargin)

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
