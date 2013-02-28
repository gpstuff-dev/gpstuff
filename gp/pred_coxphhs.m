function [mh,mS] = pred_coxphhs(gp, x, y, xt, varargin)
%PRED_COXPHHS  Return hazard and survival functions
%
%  Description
%    [H,S] = PRED_COXPHHS(GP,X,Y,XT)
%

% Copyright (c) 2012-2013 Ville Tolvanen, Aki Vehtari

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

[Ef1, Ef2, Covf] = pred_coxph(gp,x,y,xt, varargin{:});
nsamps = 10000;  
ntime=size(gp.lik.xtime,1);
if isfield(gp.lik, 'stratificationVariables')
  ind_str=gp.lik.stratificationVariables;
  ux=unique([x(:,ind_str); xt(:,ind_str)],'rows');
  nu=size(ux,1);
  for i1=1:size(ux,1)
    uind{i1}=find(xt(:,ind_str)==ux(i1,:));
  end
  nf1=ntime*nu;
else
  nf1=ntime;
end
sd=gp.lik.stime(2)-gp.lik.stime(1);
if ~isfield(gp, 'etr')
  % Single gp solution
  Sigm_tmp=Covf;
  Sigm_tmp=(Sigm_tmp+Sigm_tmp')./2;
  % f_star=mvnrnd(Ef1, Sigm_tmp(1:ntime,1:ntime), nsamps);
  f_star=mvnrnd([Ef1;Ef2], Sigm_tmp, nsamps);
  
  f1=f_star(:,1:nf1);
  f2=f_star(:,(nf1+1):end);
  
  la1=exp(f1);
  eta2=exp(f2);
  
  if ~isfield(gp.lik, 'stratificationVariables')
    hb=(la1'*sd);
    cumsumtmp=cumsum(hb)';
    for i1=1:size(eta2,2)
      mh(i1,:)=mean(bsxfun(@times,hb',eta2(:,i1)));
      Stime=exp(-bsxfun(@times,cumsumtmp,eta2(:,i1)));
      mS(i1,:)=mean(Stime);
    end
  else
    for i2=1:length(uind)
      hb=(la1(:,(i2-1)*ntime+1:i2*ntime)'*sd);
      cumsumtmp=cumsum(hb)';
      for i1=1:size(uind{i2},1)
        ind=uind{i2}(i1);
        mh(ind,:)=mean(bsxfun(@times,hb',eta2(:,ind)));
        Stime=exp(-bsxfun(@times,cumsumtmp,eta2(:,ind)));
        mS(ind,:)=mean(Stime);
      end
    end
  end
else
  % MCMC solution
  nmc=size(gp.etr,1);
  for i2=1:nmc
    i2
    if size(Covf,3)==1
      Sigm_tmp=Covf(:,i2);      
      f_star=bsxfun(@plus,[Ef1(:,i2);Ef2(:,i2)], bsxfun(@times, Sigm_tmp, randn(nf1+size(xt,1),nsamps)))';
    else
      Sigm_tmp=Covf(:,:,i2);
      Sigm_tmp=(Sigm_tmp+Sigm_tmp')./2;
      f_star=mvnrnd([Ef1(:,i2);Ef2(:,i2)], Sigm_tmp, nsamps);
    end
    
    f1=f_star(:,1:nf1);
    f2=f_star(:,(nf1+1):end);
    
    la1=exp(f1);
    eta2=exp(f2);
    
    if ~isfield(gp.lik, 'stratificationVariables')
      hb=(la1'*sd);
      cumsumtmp=cumsum(hb)';
      for i1=1:size(eta2,2)
        mh(i1,:,i2)=mean(bsxfun(@times,hb',eta2(:,i1)));
        Stime=exp(-bsxfun(@times,cumsumtmp,eta2(:,i1)));
        mS(i1,:,i2)=mean(Stime);
      end
    else
      for i3=1:length(uind)
        hb=(la1(:,(i3-1)*ntime+1:i3*ntime)'*sd);
        cumsumtmp=cumsum(hb)';
        for i1=1:size(uind{i3},1)
          ind=uind{i3}(i1);
          mh(ind,:,i2)=mean(bsxfun(@times,hb',eta2(:,ind)));
          Stime=exp(-bsxfun(@times,cumsumtmp,eta2(:,ind)));
          mS(ind,:,i2)=mean(Stime);
        end
      end
    end
  end
  mh=mean(mh,3);
  mS=mean(mS,3);
  
end


end
