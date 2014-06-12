function [e, edata, eprior, param] = gpsvi_e(w, gp, varargin)
%GPSVI_E  Return log marginal likelihood energy for SVI model
%
%  Description
%    E = GPSVI_E(W, GP, X, Y, OPTIONS) takes a GP structure GP
%    together with a matrix X of input vectors and a matrix Y of
%    target vectors, and returns the energy of the log marginal
%    likelihood lower bound (L3) given the existing approximation.
%    Each row of X corresponds to one input vector and each row of Y
%    corresponds to one target vector.
%
%    [E, EDATA, EPRIOR] = GPSVI_E(W, GP, X, Y, OPTIONS) returns also 
%    the data and prior components of the total energy.
%
%    The energy is minus log posterior cost function for th:
%      E = EDATA + EPRIOR 
%        = - log p(Y|X, th) - log p(th),
%      where th represents the parameters (lengthScale,
%      magnSigma2...), X is inputs and Y is observations.
%
%    OPTIONS is optional parameter-value pair
%      z - optional observed quantity in triplet (x_i,y_i,z_i)
%          Some likelihoods may use this. For example, in case of
%          Poisson likelihood we have z_i=E_i, that is, expected
%          value for ith case.
%
%  See also
%    GP_SET, GP_E, GPSVI_G, GPSVI_PRED
%
%  Description 2
%    Additional properties meant only for internal use.
%  
%    GP = GPSVI_E('init', GP) takes a GP structure GP and
%    initializes required fields for the SVI approximation.
% 
%    GP = GPSVI_E('clearcache', GP) takes a GP structure GP and clears the
%    internal cache stored in the nested function workspace.
%
%    [e, edata, eprior, f, L, a, La2, p] = GPSVI_E(w, gp, x, y, varargin)
%    returns many useful quantities produced by EP algorithm.
  
% Copyright (c) 2014 Ville Tolvanen

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

  % parse inputs
  ip=inputParser;
  ip.FunctionName = 'GPSVI_E';
  ip.addRequired('w', @(x) ...
                 isempty(x) || ...
                 (ischar(x) && strcmp(w, 'init')) || ...
                 isvector(x) && isreal(x) && all(isfinite(x)) ...
                 || all(isnan(x)));
  ip.addRequired('gp',@isstruct);
  ip.addOptional('x', @(x) isnumeric(x) && isreal(x) && all(isfinite(x(:))))
  ip.addOptional('y', @(x) isnumeric(x) && isreal(x) && all(isfinite(x(:))))
  ip.addParamValue('z', [], @(x) isnumeric(x) && isreal(x) && all(isfinite(x(:))))
  ip.parse(w, gp, varargin{:});
  x=ip.Results.x;
  y=ip.Results.y;
  z=ip.Results.z;
  
  if strcmp(w, 'init')
    % Initialize cache
    ch = [];

    % set function handle to the nested function svi_algorithm
    % this way each gp has its own peristent memory for SVI
    gp.fh.ne = @svi_algorithm;
    % set other function handles
    gp.fh.e=@gpsvi_e;
    gp.fh.g=@gpsvi_g;
    gp.fh.pred=@gpsvi_pred;
    % gp.fh.jpred=@gp_jpred;
    if isfield(gp.fh, 'jpred')
      gp.fh = rmfield(gp.fh, 'jpred');
    end
    gp.fh.looe=@gp_looe;
    gp.fh.loog=@gp_loog;
    gp.fh.loopred=@gpsvi_loopred;
    e = gp;
    % remove clutter from the nested workspace
    clear w gp varargin ip x y z
  elseif strcmp(w, 'clearcache')
    % clear the cache
    gp.fh.ne('clearcache');
  else
    % call svi_algorithm using the function handle to the nested function
    % this way each gp has its own peristent memory for SVI
    %[e, edata, eprior, f, L, a, La2, p] = gp.fh.ne(w, gp, x, y, z);
    [e, edata, eprior, param] = gp.fh.ne(w, gp, x, y, z);
  end

  function [e, edata, eprior, param] = svi_algorithm(w, gp, x, y, z)
      
  if strcmp(w, 'clearcache')
      ch=[];
      return
  end
  % code for the SVI algorithm

  % check whether saved values can be used
    if isempty(z)
      datahash=hash_sha512([x y]);
    else
      datahash=hash_sha512([x y z]);
    end
    if ~isempty(ch) && all(size(w)==size(ch.w)) && all(abs(w-ch.w)<1e-8) && ...
          isequal(datahash,ch.datahash)
      % The covariance function parameters or data haven't changed so we
      % can return the energy and the site parameters that are
      % saved in the cache
      e = ch.e;
      edata = ch.edata;
      eprior = ch.eprior;
      param=ch;      
    else
      % The parameters or data have changed since
      % the last call for gpsvi_e. In this case we need to
      % re-evaluate the svi approximation
      
      gp=gp_unpak(gp,w);
      u=gp.X_u;
      nu=size(u,1);
      n=size(x,1);
      % Variational approximation parameters
      
      theta1=gp.t1;      
      theta2=gp.t2;
      S=gp.S;
      m=gp.m;
      
      if isfield(gp, 'lik_mono') && isfield(gp, 'derivobs') && gp.derivobs==1
        % Monotonicty with SVI-GP        
%         inds=gp.vind;        
%         gp.xv=x(inds==2);
%         x=x(inds==1);
        yv=round(gp.nvd./abs(gp.nvd));
        yv=bsxfun(@times, yv, ones(size(gp.xv,1),length(gp.nvd)));
        yv=yv(:);
        Kv_ff = gp_trvar(rmfield(gp, 'derivobs'), x);
        % New function maybe?
        kd = diag(gp_dtrcov(gp,gp.xv,gp.xv));
        kd(1:size(gp.xv,1))=[];
        Kv_ff = [Kv_ff; kd];
        % Compute covariance matrix between f(u) and [f(x) df(xv)/dx_1,
        % ...]
        K_fu=gp_dcov2(gp,u,[],x,gp.xv)';
        K_uu=gp_trcov(rmfield(gp, 'derivobs'),u);
        beta=gp.lik_mono.sigma2;
      else
        Kv_ff = gp_trvar(gp, x);
        K_fu=gp_cov(gp,x,u);
        K_uu=gp_trcov(gp,u);
      end
      K_uu = (K_uu+K_uu')./2;          % ensure the symmetry of K_uu
      [Luu, notpositivedefinite] = chol(K_uu, 'lower');
      if notpositivedefinite
        [edata,e,eprior,~,ch] = set_output_for_notpositivedefinite;
        param=NaN;
        return
      end
      % Evaluate the Lambda (La)
      % Q_ff = K_fu*inv(K_uu)*K_fu';
      % Here we need only the diag(Q_ff), which is evaluated below
      B=Luu\(K_fu');       % m x n
      Qv_ff=sum(B.^2)';
      
      s2=gp.lik.sigma2;
      
      KfuiKuum=K_fu*(K_uu\m);      
     
%       iKuuKfu=Luu'\B;
%       iKuuKfu=K_uu\K_fu';
      iKuuKfu = Luu'\(Luu\K_fu');
      lambda=1/s2*(iKuuKfu*iKuuKfu');
%       lambda=1./s2.*(K_uu\K_fu')*(K_fu/K_uu);
      term3=sum(sum(S.*lambda));
      lambda=lambda+K_uu\eye(nu);
%       lambda=lambda/gp.data_prop+K_uu\eye(nu);
      
      term1=0;
      term2=0;
      term3=0;
      
      if isfield(gp, 'lik_mono') && isfield(gp, 'derivobs') && gp.derivobs==1
        Qv_ff_mono=Qv_ff(size(x,1)+1:end);
        Qv_ff(size(x,1)+1:end)=[];
        Kv_ff_mono=Kv_ff(size(x,1)+1:end);
        Kv_ff(size(x,1)+1:end)=[];
        KfuiKuum_mono=KfuiKuum(size(x,1)+1:end);
        KfuiKuum(size(x,1)+1:end)=[];
        iKuuKfu_mono=iKuuKfu(:,size(x,1)+1:end);
        iKuuKfu(:,size(x,1)+1:end)=[];
        term1=sum(gp.lik_mono.fh.tiltedMoments(gp.lik_mono, yv, 1:length(yv), beta, KfuiKuum_mono));
        term2=-sum(1./(2*beta).*(Kv_ff_mono-Qv_ff_mono));
        term3=-0.5/beta.*sum(sum(S.*(iKuuKfu_mono*iKuuKfu_mono')));
      end

      if isequal(gp.lik.type, 'Gaussian')
        term1=term1-n/2*log(2*pi*s2)-sum(1./(2*s2)*(y-KfuiKuum).^2);
      else
        term1=term1+sum(gp.lik.fh.tiltedMoments(gp.lik, y, 1:n, s2, KfuiKuum, z));
      end
      term2=term2-sum(1/(2*s2)*(Kv_ff-Qv_ff));
      term3=term3-0.5/s2.*sum(sum(S.*(iKuuKfu*iKuuKfu')));
      % KL divergence from q(u), N(m,S), to p(u), N(0,Kuu).
      [LS, notpositivedefinite]=chol(S,'lower');
      if notpositivedefinite
        [edata,e,eprior,~,ch] = set_output_for_notpositivedefinite;
        param=NaN;
        return
      end
      kl=0.5.*(trace(K_uu\S)+m'*(K_uu\m) - nu ...
        - 2.*sum(log(diag(LS))) + 2.*sum(log(diag(Luu))));
      edata=(term1+term2+term3)./gp.data_prop - kl;
      param.term1=term1;
      param.term2=term2;
      param.term3=term3;
      param.term4=kl;
      % ======================================================================
      % Evaluate the prior contribution to the error from covariance functions
      % ======================================================================
      eprior = 0;
      ncf=length(gp.cf);
      for i1=1:ncf
        gpcf = gp.cf{i1};
        eprior = eprior + gpcf.fh.lp(gpcf);
      end

      % ======================================================================
      % Evaluate the prior contribution to the error from likelihood function
      % ======================================================================
      if isfield(gp, 'lik') && isfield(gp.lik, 'p')
        lik = gp.lik;
        eprior = eprior + lik.fh.lp(lik);
      end
      if isfield(gp, 'lik_mono') && isfield(gp.lik, 'p')
        lik = gp.lik_mono;
        eprior = eprior + lik.fh.lp(lik);
      end

      e = edata + eprior;
      
      % store values to struct param
%       param=[];
      param.m=m;
      param.LS=LS;
      param.S=S;
      param.Luu = Luu;
      param.lambda = lambda;
      param.B = B;
      param.L=iKuuKfu;
      if isfield(gp, 'lik_mono') && isfield(gp, 'derivobs') && gp.derivobs==1
        param.La2 = iKuuKfu_mono;
      end
%       param.a = a;
%       param.p=p;
    
      % store values to the cache
      ch = param;
      ch.w = w;
      ch.e = e;
      ch.edata = edata;
      ch.eprior = eprior;
      ch.n = size(x,1);
      ch.datahash=datahash;
    end
    
end
function [edata,e,eprior,param,ch] = set_output_for_notpositivedefinite()
  % Instead of stopping to chol error, return NaN
  edata=NaN;
  e=NaN;
  eprior=NaN;
  param.f=NaN;
  param.L=NaN;
  param.a=NaN;
  param.La2=NaN;
  param.p=NaN;
  param.LS=NaN;
  param.w=NaN;
  datahash = NaN;
  ch=param;
  ch.e = e;
  ch.edata = edata;
  ch.eprior = eprior;
  ch.datahash=datahash;
  ch.w = NaN;
end

end
