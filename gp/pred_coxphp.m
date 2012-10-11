function p = pred_coxphp(gp, x, y, xt, yt, varargin)
%PRED_COXPHP  Integrates the model from zero to point yt (when time is 
%  scaled to interval 0-1)
%
%  Description
%    P = PRED_COXPHP(GP,X,Y,XT,YT)
%    If given 1D vector Y, integrates the model from zero to point yt with 
%    respect to time. Return P, the probability that event has happened 
%    before time yt. YT is vector of size 1xM(or Mx1) indicating times in 
%    scaled interval ~ Unif(0,1). Returns matrix P of size NxM where columns 
%    correspond to points in YT and rows correspond to rows in X (e.g. people).
%    In case of 2D Y, Integrate model from starting time YT(:,1) to end time
%    YT(:,2). YT is matrix of size Mx2, indicating starting time and end
%    time for every test point.
%

% Copyright (c) 2012 Ville Tolvanen

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.


[Ef1, Ef2, Covf] = pred_coxph(gp,x,y,xt, varargin{:});
nsamps = 10000;
ntime=size(gp.lik.stime,2)-1;
sd=gp.lik.stime(2)-gp.lik.stime(1);
Sigm_tmp=Covf;
Sigm_tmp=(Sigm_tmp+Sigm_tmp')./2;
f_star=mvnrnd([Ef1;Ef2], Sigm_tmp, nsamps);

f1=f_star(:,1:ntime);
f2=f_star(:,(ntime+1):end);

la1=exp(f1);
eta2=exp(f2);

mST=zeros(size(eta2,2),1);
if size(y,2) == 1
  % Integrate from zero to yt
  cumsumtmp=cumsum(la1'*sd)';
  for i1=1:size(eta2,2)
    Stime=exp(-bsxfun(@times,cumsumtmp,eta2(:,i1)));
    mStime=mean(Stime);
    if size(yt,1)==size(y,1) && size(yt,2)==size(y,2)
      % regular case y and yt are equal size
      mST(i1,1)=1-mStime(binsgeq(gp.lik.xtime,yt(i1)));
    elseif size(yt,1)==1 && size(yt,2)>=1
      % evaluate each individual using multiple time points
      for i=1:size(yt,2)
        mST(i1,i)=1-mStime(binsgeq(gp.lik.xtime,yt(1,i)));
      end
    else
      error('Size of yt is not equal to size of y or 1xT')
    end
  end
  p = mST;

else
  if size(y,2) ~= size(yt,2)
    error('size(y,2) ~= size(yt,2)');
  end
  
  % Integrate from yt(:,1) to yt(:,2)
  sb=sum(bsxfun(@gt,yt(:,1),gp.lik.stime),2);
  se=sum(bsxfun(@gt,yt(:,2),gp.lik.stime),2);
  
  for i1 =1:size(eta2,2)
    if sb(i1) ~= se(i1)
      hb=(la1(:,sb(i1)+1:se(i1)-1)'*sd);
      hb=[((gp.lik.stime(sb(i1)+1)-yt(i1,1)).*la1(:,sb(i1)))'; hb; ((yt(i1,2)-gp.lik.stime(se(i1))).*la1(:,se(i1)))'];
    else
      hb = la1(:, se(i1))'*(yt(i1,2) - yt(i1,1));
    end
    cumsumtmp=[zeros(nsamps, sb(i1)) cumsum(hb)'];
    Stime=exp(-bsxfun(@times,cumsumtmp,eta2(:,i1)));
    mStime=mean(Stime);
    mST(i1,1)=1-mStime(end);
  end
  p = mST;
  
end


end
