function [g, gdata, gprior] = gpsvi_g(w, gp, x, y, varargin)
%GPSVI_G  Evaluate approximate natural gradient for SVIGP
%
%  Description
%    G = GPLA_G(W, GP, X, Y, OPTIONS) takes a full GP parameter vector W,
%    structure GP a matrix X of input vectors and a matrix Y of target
%    vectors, and evaluates the approximate natural gradient G of the
%    marginal log likelihood estimate. Each row of X corresponds to one
%    input vector and each row of Y corresponds to one target vector.
%
%    [G, GDATA, GPRIOR] = GPLA_G(W, GP, X, Y, OPTIONS) also returns
%    the data and prior contributions to the gradient.
%
%    OPTIONS is optional parameter-value pair
%      z -             optional observed quantity in triplet (x_i,y_i,z_i)
%                      Some likelihoods may use this. For example, in case
%                      of Poisson likelihood we have z_i=E_i, that is,
%                      expected value for ith case.
%      gpsvi_e_param - precomputed output variable 'param' returned by
%                      calling gpsvi_e with the same input parameters, i.e.
%                      [~,~,~,param] = gpsvi_e(w,gp,x,y,'z',z);.
%
%  See also
%    GP_SET, GP_G, GPSVI_E, GPSVI_PRED

% Copyright (c) 2014 Ville Tolvanen

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

ip=inputParser;
ip.FunctionName = 'GPSVI_G';
ip.addRequired('w', @(x) isvector(x) && isreal(x) && all(isfinite(x)));
ip.addRequired('gp',@isstruct);
ip.addRequired('x', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addRequired('y', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addParamValue('z', [], @(x) isreal(x) && all(isfinite(x(:))))
ip.addParamValue('gpsvi_e_param', [], @(p) isstruct(p) || isnan(p))
ip.parse(w, gp, x, y, varargin{:});
z = ip.Results.z;
param = ip.Results.gpsvi_e_param;
if ~isstruct(param)
  g = nan;
  gdata = nan;
  gprior = nan;
  return
end

%   gp = gp_unpak(gp, w);       % unpak the parameters
ncf = length(gp.cf);
n=size(x,1);

g = [];
gdata = [];
gprior = [];

if isfield(gp, 'savememory') && gp.savememory
  savememory=1;
else
  savememory=0;
end

% First Evaluate the data contribution to the error
if isempty(param)
  [~,~,~,param]=gpsvi_e(w,gp,x,y, 'z', z);
end
u=gp.X_u;
nu=size(u,1);
n=size(x,1);
gp=gp_unpak(gp,w);

% Variational parameters
theta1=gp.t1;
theta2=gp.t2;
m=gp.m;
S=gp.S;

if isfield(gp, 'lik_mono') && isfield(gp, 'derivobs') && gp.derivobs==1
  % Monotonicty with SVI-GP
  yv=round(gp.nvd./abs(gp.nvd));
  yv=bsxfun(@times, yv, ones(size(gp.xv,1),length(gp.nvd)));
  yv=yv(:);
  Kv_ff = gp_trvar(rmfield(gp, 'derivobs'), x);
  % New function maybe?
  xv=gp.xv;
  kd = diag(gp_dtrcov(gp,xv,xv));
  kd(1:size(xv,1))=[];      
  Kv_ff = [Kv_ff; kd];
  % Compute covariance matrix between f(u) and [f(x) df(xv)/dx_1,
  % df(xv)/dx_2, ...]
  K_fu=gp_dcov2(gp,u,[],x,xv)';
  K_uu=gp_trcov(rmfield(gp, 'derivobs'),u);
  beta=gp.lik_mono.sigma2;
else
  Kv_ff = gp_trvar(gp, x);
  K_fu=gp_cov(gp,x,u);
  K_uu=gp_trcov(gp,u);
end
K_uu = (K_uu+K_uu')./2;          % ensure the symmetry of K_uu
Luu=param.Luu;
s2=gp.lik.sigma2;
B=Luu\(K_fu');       % m x n
Qv_ff=sum(B.^2)';

%lambda=param.lambda;
iKuuKuf = Luu'\(Luu\K_fu');
KfuiKuum=iKuuKuf'*m;

LL=param.L;
LS=param.LS;
if isfield(gp, 'lik_mono') && isfield(gp, 'derivobs') && gp.derivobs==1
  nv=size(yv,1);
  Qv_ff_mono=Qv_ff(size(x,1)+1:end);
  Qv_ff(size(x,1)+1:end)=[];
  Kv_ff_mono=Kv_ff(size(x,1)+1:end);
  Kv_ff(size(x,1)+1:end)=[];
%       KfuiKuum_mono=KfuiKuum(size(x,1)+1:end);
%       KfuiKuum(size(x,1)+1:end)=[];
%       iKuuKfu_mono=iKuuKfu(:,size(x,1)+1:end);
%       iKuuKfu(:,size(x,1)+1:end)=[];
%       term1=sum(gp.lik_mono.fh.tiltedMoments(gp.lik, yv, 1:length(yv), beta, KfuiKuum_mono));
%       term2=-sum(1./(2*beta).*(Kv_ff_mono-Qv_ff_mono));
%       term3=-0.5/beta.*sum(sum(S.*(iKuuKfu_mono*iKuuKfu_mono')));
end
%     gdata(1:nu)=(1./s2.*(K_uu\(K_fu'*y)) - theta1);
%gdata(1:nu)=(1./s2.*(K_uu\(K_fu'*y)) - 1./s2.*LL*(LL'*m))./gp.data_prop - K_uu\m;
if isequal(gp.lik.type, 'Gaussian')
  if isfield(gp, 'lik_mono') && isfield(gp, 'derivobs') && gp.derivobs==1
    g_gaussian = 1./s2.*(K_uu\(K_fu(1:size(x,1),:)'*y))./gp.data_prop + 2*theta2*m;
    ss=[s2*ones(size(x,1),1); beta.*ones(size(K_fu,1)-size(x,1),1)];
    if isfield(gp.lik_mono, 'nu')
      nuu=gp.lik_mono.nu;
    else
      nuu=1;
    end
    zi=yv.*(iKuuKuf(:,size(x,1)+1:end)'*m)./sqrt(nuu.^2+beta);
    lnormzi=norm_lpdf(zi);
    lncdfzi=log(norm_cdf(zi));
    gdata(1:nu)=g_gaussian + (K_uu\sum(bsxfun(@times, yv'.*exp(lnormzi - lncdfzi - 0.5.*log(nuu.^2+beta))', ...
      K_fu(size(x,1)+1:end,:)'),2))./gp.data_prop;
  else
    gdata(1:nu) = 1./s2.*(K_uu\(K_fu'*y))./gp.data_prop + 2*theta2*m;
  end
else % elseif isequal(gp.lik.type, 'Probit')
  if isfield(gp, 'lik_mono') && isfield(gp, 'derivobs') && gp.derivobs==1
    ss=[s2*ones(size(x,1),1); beta.*ones(size(K_fu,1)-size(x,1),1)];
    if isfield(gp.lik_mono, 'nu')
      nuu=[ones(size(x,1),1); gp.lik_mono.nu.*ones(size(K_fu,1)-size(x,1),1)];
    else
      nuu=1;
    end
    zi=[y;yv].*(iKuuKuf'*m)./sqrt(nuu.^2+ss);
    lnormzi=norm_lpdf(zi);
    lncdfzi=log(norm_cdf(zi));
    lncdfzi(isinf(lncdfzi)) = log(realmin);
%         lncdfzi=gp.lik.fh.tiltedMoments(gp.lik_mono, y, 1:n, s2, iKuuKuf'*m);
    gdata(1:nu)=(K_uu\sum(bsxfun(@times, [y;yv]'.*exp(lnormzi - lncdfzi - 0.5.*log(nuu.^2+ss))', ...
      K_fu'),2))./gp.data_prop - K_uu\m;
  else
    zi=y.*(iKuuKuf'*m)./sqrt(1+s2);
    lnormzi=norm_lpdf(zi);
    lncdfzi=log(norm_cdf(zi));
    %lncdfzi=gp.lik.fh.tiltedMoments(gp.lik, y, 1:n, s2, iKuuKuf'*m);
    gdata(1:nu)=(K_uu\sum(bsxfun(@times, y'.*exp(lnormzi - lncdfzi - 0.5.*log(1+s2))', ...
      K_fu'),2))./gp.data_prop - K_uu\m;
  end
end
%gt=-0.5.*1./s2*((LL*LL')./gp.data_prop + K_uu\eye(size(K_uu))) + 0.5.*(S\eye(size(S)));
gt=-0.5.*(1./s2*(LL*LL')./gp.data_prop + K_uu\eye(size(K_uu))) - theta2;
if isfield(gp, 'lik_mono') && isfield(gp, 'derivobs') && gp.derivobs==1
  gt=gt - 0.5.*1./beta*((param.La2*param.La2')./gp.data_prop);
end
gdata(nu+1:nu^2+nu)=gt(:);

gprior(1:nu^2+nu)=0;

if 0%isfield(gp, 'lik_mono') && isfield(gp, 'derivobs') && gp.derivobs==1

else
  if ~isempty(strfind(gp.infer_params, 'covariance'))
    i1=nu^2+nu;
    for i=1:ncf
      gpcf = gp.cf{i};
      DKuuc = gpcf.fh.cfg(gpcf, u);
      if isfield(gp, 'lik_mono')
        DKffa = gpcf.fh.cfg(gpcf, x, [], 1);
        if ~isempty(DKffa)
          DKdd = gpcf.fh.cfdg2(gpcf, xv);
          DKuuf = gpcf.fh.cfg(gpcf, u, x);
          DKuudf = gpcf.fh.cfdg(gpcf, xv, u);
          % Select monotonic dimensions
          inds=[];
          nvd=abs(gp.nvd);
          for idd=1:length(gp.nvd)
            inds=[inds size(xv,1)*(nvd(idd)-1)+1:size(xv,1)*nvd(idd)];
          end
          for ijj=1:length(DKffa)
            DKdd{ijj}=DKdd{ijj}(inds,inds);
            DKuudf{ijj}=DKuudf{ijj}(inds,:);
          end

          DKffc{1}=[DKffa{1}; diag(DKdd{1})];
          DKufc{1}=[DKuuf{1} DKuudf{1}'];
          for i2=2:length(DKffa)
            if length(DKffa{i2})==1 && size(x,1)>1
              DKffc{i2}=[repmat(DKffa{i2},n,1); diag(DKdd{i2})];
            else
              DKffc{i2}=[DKffa{i2}; diag(DKdd{i2})];
            end
            DKufc{i2}=[DKuuf{i2} DKuudf{i2}'];
          end
        end
        np=length(DKffa);
      else
        DKffc = gpcf.fh.cfg(gpcf, x, [], 1);
        DKufc = gpcf.fh.cfg(gpcf, u, x);
        np=length(DKffc);
      end
      gprior_cf = gpcf.fh.lpg(gpcf);

      % Are there specified mean functions
      if  ~isfield(gp,'meanf')
        % Evaluate the gradient with respect to covariance function
        % parameters
        for i2 = 1:np

          DKff=DKffc{i2};
          DKuu=DKuuc{i2};
          DKuf=DKufc{i2};
          i1 = i1+1;

          DKfuiKuum=DKuf'*(K_uu\m) - K_fu*(K_uu\(DKuu*(K_uu\m)));
          %iKuuKfu=K_uu\K_fu';
          %lambda=1/s2*(iKuuKfu*iKuuKfu');
          %term3=sum(sum(S.*lambda));
          %lambda=lambda+K_uu\eye(nu);
          KfuiKuuKuu = iKuuKuf'*DKuu;


          if isequal(gp.lik.type, 'Gaussian')
            %term1=-sum(1./(2*s2)*(y-KfuiKuum).^2);
            if isfield(gp, 'lik_mono') && isfield(gp, 'derivobs') && gp.derivobs==1
              gterm1=sum(1./s2.*(y-KfuiKuum(1:n)).*DKfuiKuum(1:n)) ...
                +sum(exp(lnormzi - lncdfzi).*yv./sqrt(nuu.^2+beta).*DKfuiKuum(n+1:end));
            else
              gterm1=sum(1./s2.*(y-KfuiKuum).*DKfuiKuum);
            end
          else % elseif isequal(gp.lik.type, 'Probit')
            %term1=sum(log(norm_cdf(zi)));
            if isfield(gp, 'lik_mono')
              gterm1=sum(exp(lnormzi - lncdfzi).*[y;yv]./sqrt(nuu.^2+ss).*DKfuiKuum);
            else
              gterm1=sum(exp(lnormzi - lncdfzi).*y./sqrt(1+s2).*DKfuiKuum);
            end
          end
          %term2=-sum(1/(2*s2)*(Kv_ff-Qv_ff))
          if isfield(gp, 'lik_mono') && isfield(gp, 'derivobs') && gp.derivobs==1
            gterm2=-sum(1./(2*ss).*(DKff - 2.*sum(DKuf'.*iKuuKuf',2) ...
              + sum(KfuiKuuKuu.*iKuuKuf',2)));
            iKuuKuf_mono=iKuuKuf(:,size(x,1)+1:end);
            iKuuKuf(:,size(x,1)+1:end)=[];
            DKuf_mono=DKuf(:,size(x,1)+1:end);
            DKuf(:,size(x,1)+1:end)=[];
            %term3=-0.5/s2.*sum(sum(S.*(iKuuKfu*iKuuKfu')));
            gterm3=-1/(2*s2).*sum(sum(S.*(-K_uu\(DKuu*(iKuuKuf*iKuuKuf')) ...
              + K_uu\(DKuf*iKuuKuf') + iKuuKuf*(DKuf'/K_uu) ...
              - ((iKuuKuf*iKuuKuf')*DKuu)/K_uu)));
            % Monotonicity part
            gterm3=gterm3-1/(2*beta).*sum(sum(S.*(-K_uu\(DKuu*(iKuuKuf_mono*iKuuKuf_mono')) ...
              + K_uu\(DKuf_mono*iKuuKuf_mono') + iKuuKuf_mono*(DKuf_mono'/K_uu) ...
              - ((iKuuKuf_mono*iKuuKuf_mono')*DKuu)/K_uu)));
            iKuuKuf=[iKuuKuf iKuuKuf_mono];
          else
            gterm2=-sum(1./(2*s2).*(DKff - 2.*sum(DKuf'.*iKuuKuf',2) ...
              + sum(KfuiKuuKuu.*iKuuKuf',2)));
            %term3=-0.5/s2.*sum(sum(S.*(iKuuKfu*iKuuKfu')));
            gterm3=-1/(2*s2).*sum(sum(S.*(-K_uu\(DKuu*(iKuuKuf*iKuuKuf')) ...
              + K_uu\(DKuf*iKuuKuf') + iKuuKuf*(DKuf'/K_uu) ...
              - ((iKuuKuf*iKuuKuf')*DKuu)/K_uu)));
          end

          %term4=0.5.*(trace(K_uu\S)+m'*(K_uu\m) - nu ...
          % - 2.*sum(log(diag(LS))) + log(det(K_uu))
          gterm4=0.5.*(trace(-K_uu\(DKuu*(K_uu\S))) - m'*(K_uu\(DKuu*(K_uu\m))) ...
            +trace(K_uu\DKuu));
          gterm5=0;

          gdata(i1) = (gterm5+gterm1+gterm2+gterm3)./gp.data_prop - gterm4;
        end
      else
        % Not support at the moment
      end
      gprior=[gprior gprior_cf];
    end
  end
end

if isfield(gp, 'lik_mono')
  ss(nu+nu^2+1:end)=0;
  KfuiKuum_mono=KfuiKuum(size(x,1)+1:end);
  KfuiKuum(size(x,1)+1:end)=[];
  iKuuKuf_mono=iKuuKuf(:,size(x,1)+1:end);
  iKuuKuf(:,size(x,1)+1:end)=[];
  if isequal(gp.lik.type, 'Gaussian')
    lnormzi_mono=lnormzi;
    lncdfzi_mono=lncdfzi;
    nuu=repmat(nuu,nv+n,1);
  else
    lnormzi_mono=lnormzi(size(x,1)+1:end);
    lnormzi(size(x,1)+1:end)=[];
    lncdfzi_mono=lncdfzi(size(x,1)+1:end);
    lncdfzi(size(x,1)+1:end)=[];
  end
  %       gterm1=sum(-0.5.*exp(lnormzi - lncdfzi).*y.*ss./(nuu+ss).^(3/2).*KfuiKuum);
end
% =================================================================
% Gradient with respect to Gaussian likelihood function parameters
if ~isempty(strfind(gp.infer_params, 'likelihood')) ...
    && ~isempty(gp.lik.p.sigma2) 
  % Prior contribution
  gprior_lik = gp.lik.fh.lpg(gp.lik);
  % Evaluate the gradient from Gaussian likelihood
  if isequal(gp.lik.type, 'Gaussian')
    gterm1=-n/2 + sum(1./(2*s2)*(y-KfuiKuum).^2);
  else % elseif isequal(gp.lik.type, 'Probit')
    gterm1=sum(-0.5.*exp(lnormzi - lncdfzi).*y.*s2./(1+s2).^(3/2).*KfuiKuum);
  end
  gterm2=sum(1/(2*s2)*(Kv_ff-Qv_ff));
  gterm3=0.5/s2.*sum(sum(S.*(iKuuKuf*iKuuKuf')));
  gterm4=0;
  %gterm5=-n/2;
  
  gdata_lik = (gterm1+gterm2+gterm3)./gp.data_prop - gterm4;
  
  gdata=[gdata gdata_lik];
  gprior=[gprior gprior_lik];
end
% Gradient w.r.t monotonicity likelihood sigma2
if ~isempty(strfind(gp.infer_params, 'likelihood')) ...
    && isfield(gp, 'lik_mono') && ~isempty(gp.lik_mono.p.sigma2) 
  % Prior contribution
  gprior_lik = gp.lik_mono.fh.lpg(gp.lik_mono);
  gterm1=sum(-0.5.*exp(lnormzi_mono - lncdfzi_mono).*yv.*beta ...
    ./(nuu(size(x,1)+1:end)+beta).^(3/2).*KfuiKuum_mono);
  gterm2=sum(1/(2*beta)*(Kv_ff_mono-Qv_ff_mono));
  gterm3=0.5/beta.*sum(sum(S.*(iKuuKuf_mono*iKuuKuf_mono')));
  gterm4=0;
  %gterm5=-n/2;
  
  gdata_lik = (gterm1+gterm2+gterm3)./gp.data_prop - gterm4;
  
  gdata=[gdata gdata_lik];
  gprior=[gprior gprior_lik];
end

assert(isreal(gdata))
assert(isreal(gprior))
% gprior(1:nu^2+nu)=0;

g=gdata + gprior;
end
