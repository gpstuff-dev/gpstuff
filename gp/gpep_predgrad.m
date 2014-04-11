function [Eft, Varft, lpyt, Eyt, Varyt] = gpep_predgrad(gp, x, y, varargin)
%GPEP_PRED  Predictions with Gaussian Process EP approximation
%
%  Description
%    [EFT, VARFT] = GPEP_PREDGRAD(GP, X, Y, XT, OPTIONS)
%    takes a GP structure together with matrix X of training
%    inputs and vector Y of training targets, and evaluates the
%    predictive distribution of the derivative of f at test inputs XT. 
%    Returns a posterior mean EFT and variance VARFT of latent variables.
%    Results are stacked so that first NT elements correspond to the
%    derivative w.r.t first dimension, elements from NT+1 to 2*NT
%    correspond to derivative w.r.t second dimensions and so on. Dimensions
%    which to predict are determined by field NVD in the GP structure.
%
%    [EFT, VARFT, LPYT] = GPEP_PRED(GP, X, Y, XT, 'yt', YT, OPTIONS)
%    returns also logarithm of the predictive density LPYT of the
%    observations YT at test input locations XT. This can be used
%    for example in the cross-validation. Here Y has to be a vector.
%
%    [EFT, VARFT, LPYT, EYT, VARYT] = GPEP_PRED(GP, X, Y, XT, OPTIONS)
%    returns also the posterior predictive mean EYT and variance VARYT.
%
%    [EF, VARF, LPY, EY, VARY] = GPEP_PRED(GP, X, Y, OPTIONS)
%    evaluates the predictive distribution at training inputs X
%    and logarithm of the predictive density LPY of the training
%    observations Y.
%
%    OPTIONS is optional parameter-value pair
%      predcf - an index vector telling which covariance functions are
%               used for prediction. Default is all (1:gpcfn).
%               See additional information below.
%      tstind - a vector/cell array defining, which rows of X belong
%               to which training block in *IC type sparse models.
%               Default is []. In case of PIC, a cell array
%               containing index vectors specifying the blocking
%               structure for test data. IN FIC and CS+FIC a
%               vector of length n that points out the test inputs
%               that are also in the training set (if none, set
%               TSTIND = [])
%      yt     - optional observed yt in test points (see below)
%      z      - optional observed quantity in triplet (x_i,y_i,z_i)
%               Some likelihoods may use this. For example, in case of
%               Poisson likelihood we have z_i=E_i, that is, expected value
%               for ith case.
%      zt     - optional observed quantity in triplet (xt_i,yt_i,zt_i)
%               Some likelihoods may use this. For example, in case of
%               Poisson likelihood we have z_i=E_i, that is, the expected
%               value for the ith case.
%      fcorr  - Method used for latent marginal posterior corrections. 
%               Default is 'off'. For EP possible method is 'fact'.
%               If method is 'on', 'fact' is used for EP.
%
%    NOTE! Sparse approximations are not currently implemented for
%    monotonic GP's.
%
%  See also
%    GPEP_E, GPEP_G, GP_PRED, DEMO_SPATIAL, DEMO_CLASSIFIC
%
% Copyright (c) 2007-2010 Jarno Vanhatalo
% Copyright (c) 2010      Heikki Peura
% Copyright (c) 2011      Pasi Jylänki
% Copyright (c) 2012 Aki Vehtari
% Copyright (c) 2014 Ville Tolvanen

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

  ip=inputParser;
  ip.FunctionName = 'GPEP_PRED';
  ip.addRequired('gp', @isstruct);
  ip.addRequired('x', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
  ip.addRequired('y', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
  ip.addOptional('xt', [], @(x) isempty(x) || (isreal(x) && all(isfinite(x(:)))))
  ip.addParamValue('yt', [], @(x) isreal(x) && all(isfinite(x(:))))
  ip.addParamValue('z', [], @(x) isreal(x) && all(isfinite(x(:))))
  ip.addParamValue('zt', [], @(x) isreal(x) && all(isfinite(x(:))))
  ip.addParamValue('predcf', [], @(x) isempty(x) || ...
                   isvector(x) && isreal(x) && all(isfinite(x)&x>0))
  ip.addParamValue('tstind', [], @(x) isempty(x) || iscell(x) ||...
                   (isvector(x) && isreal(x) && all(isfinite(x)&x>0)))
  ip.addParamValue('fcorr', 'off', @(x) ismember(x, {'off', 'fact', 'cm2', 'on'}))
  if numel(varargin)==0 || isnumeric(varargin{1})
    % inputParser should handle this, but it doesn't
    ip.parse(gp, x, y, varargin{:});
  else
    ip.parse(gp, x, y, [], varargin{:});
  end
  xt=ip.Results.xt;
  yt=ip.Results.yt;
  z=ip.Results.z;
  zt=ip.Results.zt;
  predcf=ip.Results.predcf;
  tstind=ip.Results.tstind;
  fcorr=ip.Results.fcorr;
  if isempty(xt)
    xt=x;
    if isempty(tstind)
      if iscell(gp)
        gptype=gp{1}.type;
      else
        gptype=gp.type;
      end
      switch gptype
        case {'FULL' 'VAR' 'DTC' 'SOR'}
          tstind = [];
        case {'FIC' 'CS+FIC'}
          tstind = 1:size(x,1);
        case 'PIC'
          if iscell(gp)
            tstind = gp{1}.tr_index;
          else
            tstind = gp.tr_index;
          end
      end
    end
    if isempty(yt)
      yt=y;
    end
    if isempty(zt)
      zt=z;
    end
  end

  [tn, tnin] = size(x);
  [n, nout] = size(y);

  if isfield(gp.lik, 'nondiagW')
   
  else
    switch gp.type
      % ============================================================
      % FULL
      % ============================================================
      case 'FULL'        % Predictions with FULL GP model
                         %[e, edata, eprior, tautilde, nutilde, L] = gpep_e(gp_pak(gp), gp, x, y, 'z', z);
        [e, edata, eprior, p] = gpep_e(gp_pak(gp), gp, x, y, 'z', z);
        [tautilde, nutilde, L] = deal(p.tautilde, p.nutilde, p.L);
        
        x2=x;
        y2=y;
        x=gp.xv;
        yv=round(gp.nvd./abs(gp.nvd));
        y=bsxfun(@times,yv,ones(size(gp.xv,1),length(gp.nvd)));
        [K,C]=gp_dtrcov(gp,x2,x);
        kstarstar = diag(gp_dtrcov(gp,xt,xt, predcf));
        kstarstar(1:size(xt,1))=[];
        %           kstarstar=kstarstar(1:size(xt,1));
        ntest=size(xt,1);
        K_nf=gp_dcov(gp,x2,xt,predcf)';
        K_nf(1:size(xt,1),:)=[];
        [n,nin] = size(x);
        
        [Eft,V]=pred_var(tautilde,K,K_nf,nutilde);
        Varft=kstarstar-V;
          
        % ============================================================
        % FIC
        % ============================================================
      case 'FIC'       
        
        % ============================================================
        % PIC
        % ============================================================
      case {'PIC' 'PIC_BLOCK'}       
        
        % ============================================================
        % CS+FIC
        % ============================================================
      case 'CS+FIC'       
        
        % ============================================================
        % DTC/(VAR)
        % ============================================================
      case {'DTC' 'VAR' 'SOR'}      
    end
  end
  
  % ============================================================
  % Evaluate also the predictive mean and variance of new observation(s)
  % ============================================================    
  if nargout == 3
    if isempty(yt)
      lpyt=[];
    else
      lpyt = gp.lik_mono.fh.predy(gp.lik_mono, Eft, Varft, yt, zt);
    end
    lpyt=reshape(lpyt, size(xt,1),  length(gp.nvd));
  elseif nargout > 3
    [lpyt, Eyt, Varyt] = gp.lik_mono.fh.predy(gp.lik_mono, Eft, Varft, yt, zt);
    lpyt=reshape(lpyt, size(xt,1),  length(gp.nvd));
    Eyt=reshape(Eyt, size(xt,1),  length(gp.nvd));
    Varyt=reshape(Varyt, size(xt,1),  length(gp.nvd));
  end
  Eft=reshape(Eft, size(xt,1), length(gp.nvd));
  Varft=reshape(Varft, size(xt,1), length(gp.nvd));
end

function [m,S]=pred_var(tau_q,K,A,b)

% helper function for determining
%
% m = A * inv( K+ inv(diag(tau_q)) ) * inv(diag(tau_q)) *b
% S = diag( A * inv( K+ inv(diag(tau_q)) ) * A)
%
% when the site variances tau_q may be negative
%

  ii1=find(tau_q>0); n1=length(ii1); W1=sqrt(tau_q(ii1));
  ii2=find(tau_q<0); n2=length(ii2); W2=sqrt(abs(tau_q(ii2)));

  m=A*b;
  b=K*b;
  S=zeros(size(A,1),1);
  u=0;
  U=0;
  if ~isempty(ii1)
    % Cholesky decomposition for the positive sites
    L1=(W1*W1').*K(ii1,ii1);
    L1(1:n1+1:end)=L1(1:n1+1:end)+1;
    L1=chol(L1);
    
    U = bsxfun(@times,A(:,ii1),W1')/L1;
    u = L1'\(W1.*b(ii1));
    
    m = m-U*u;
    S = S+sum(U.^2,2);
  end

  if ~isempty(ii2)
    % Cholesky decomposition for the negative sites
    V=bsxfun(@times,K(ii2,ii1),W1')/L1;
    L2=(W2*W2').*(V*V'-K(ii2,ii2));
    L2(1:n2+1:end)=L2(1:n2+1:end)+1;
    
    [L2,pd]=chol(L2);
    if pd==0
      U = bsxfun(@times,A(:,ii2),W2')/L2 -U*(bsxfun(@times,V,W2)'/L2);
      u = L2'\(W2.*b(ii2)) -L2'\(bsxfun(@times,V,W2)*u);
      
      m = m+U*u;
      S = S-sum(U.^2,2);
    else
      fprintf('Posterior covariance is negative definite.\n')
    end
  end

end

function [m_q,S_q]=pred_var2(tautilde,nutilde,L,K_uu,K_fu,D,K_nu)

% function for determining the parameters of the q-distribution
% when site variances tau_q may be negative
%
% q(f) = N(f|0,K)*exp( -0.5*f'*diag(tau_q)*f + nu_q'*f )/Z_q = N(f|m_q,S_q)
%
% S_q = inv(inv(K)+diag(tau_q)) where K is sparse approximation for prior
%       covariance
% m_q = S_q*nu_q;
%
% det(eye(n)+K*diag(tau_q))) = det(L1)^2 * det(L2)^2
% where L1 and L2 are upper triangular
%
% see Expectation consistent approximate inference (Opper & Winther, 2005)

  n=length(nutilde);

  U = K_fu;
  S = 1+tautilde.*D;
  B = tautilde./S;
  BUiL = bsxfun(@times, B, U)/L';
  % iKS = diag(B) - BUiL*BUiL';

  Ktnu = D.*nutilde + U*(K_uu\(U'*nutilde));
  m_q = nutilde - B.*Ktnu + BUiL*(BUiL'*Ktnu);
  kstar = K_nu*(K_uu\K_fu');
  m_q = kstar*m_q;

  S_q = sum(bsxfun(@times,B',kstar.^2),2) - sum((kstar*BUiL).^2,2);
  % S_q = kstar*iKS*kstar';


end
