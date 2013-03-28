function [p, pc c] = gpla_cm2(gp, x, y, fvec,varargin) 
%GPLA_CM2  CM2 correction for marginal likelihood using Laplace
%          approximation
% 
%  Description
%    [P, PC, C] = GPLA_CM2(GP, X, Y, FVEC) Evaluates the marginal likelihood
%    at given grind points FVEC for given indices. Returns tilted
%    distribution without any correction P, with CM2 correction terms
%    PC and the correction terms C.
%
%   OPTIONS is optional parameter-value pair
%      z      - optional observed quantity in triplet (x_i,y_i,z_i)
%               Some likelihoods may use this. For example, in case of 
%               Poisson likelihood we have z_i=E_i, that is, expected value 
%               for ith case. 
%      ind    - Index defining which point in data grid latent grid fvec
%               corresponds to. If ind is vector of size m x 1, fvec must
%               be matrix of size n x m, where n is the number of grid
%               points for each index. Default = 1.
%
%   Reference
%     Cseke & Heskes (2011). Approximate Marginals in Latent Gaussian
%     Models. Journal of Machine Learning Research 12 (2011), 417-454
%         
%   See also
%     GPLA_FACT, GPEP_FACT, DEMO_IMPROVEDMARGINALS

% Copyright (c) 2011 Ville Tolvanen

% This software is distributed under the GNU General Public 
% License (version 3 or later); please refer to the file 
% License.txt, included with the software, for details.

ip=inputParser;
ip.FunctionName = 'GP_PRED';
ip.addRequired('gp',@isstruct);
ip.addRequired('x', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addRequired('y', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addRequired('fvec',  @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addOptional('xt', [], @(x) isreal(x) && all(isfinite(x(:))))
ip.addParamValue('z', [], @(x) isreal(x) && all(isfinite(x(:))))
ip.addParamValue('ind', 1, @(x) isreal(x) && all(isfinite(x(:))))
if rem(size(varargin,2), 2) == 0
  ip.parse(gp, x, y, fvec,[],varargin{:});
else
  ip.parse(gp, x, y, fvec,varargin{:});
end
% ip.parse(gp, x, y, fvec,varargin{:});
z = ip.Results.z;
ind = ip.Results.ind;
xt = ip.Results.xt;
predictive = false;
[n, tmp] = size(x);
[nin, n_ind] = size(fvec);
[Ef, Covf] = gp_jpred(gp,x,y,x, 'z', z);
if size(ind,1)==1
  ind=ind';
end
if size(ind,1) ~= n_ind
  error('given latent grid matrix fvec must be size n x m, where m is the length of ind vector');
end
if ~isempty(xt) && ~isequal(xt, x)
  % Predictive equations if given xt, mind that if xt(ind) is in training
  % input x, predictive equations might not work correctly.
  predictive = true;
  [Ef2, Covf2] = gp_jpred(gp,x,y,xt,'z',z);
end
[e, edata, eprior, f_mode, Lf, af, Wf] = gpla_e(gp_pak(gp), gp, x,y,'z',z);
if iscell(gp)
  gplik = gp{1}.lik;
else
  gplik = gp.lik;
end
pc = zeros(nin,size(ind,1)); p = zeros(nin,size(ind,1)); c = zeros(nin,size(ind,1));

% Loop through grid indices
for i1=1:size(ind,1)
  if ~predictive
    n = n-1;
    cii = Covf(ind(i1),ind(i1));
    if isempty(z)
      z_ind = [];
    else
      z_ind = z(ind(i1));
    end
        
    % Function handle to marginal distribution without any correction parameters
%     K_ff = gp_cov(gp, x(ind(i1),:), x(ind(i1),:));
    t_tilde = @(b) arrayfun(@(f) exp(gplik.fh.ll(gplik, y(ind(i1)), f_mode(ind(i1)), z_ind) + (f-f_mode(ind(i1)))*gplik.fh.llg(gplik, y(ind(1)), f_mode(ind(i1)), 'latent', z_ind) + 0.5*(f-f_mode(ind(i1)))^2*gplik.fh.llg2(gplik, y(ind(i1)), f_mode(ind(i1)), 'latent', z_ind)), b);
%     t_tilde = @(f) norm_pdf(f, f_mode(ind(i1))./(-gplik.fh.llg2(gplik,y(ind(i1)), f_mode(ind(i1)), 'latent', z_ind).*K_ff) + f_mode(ind(i1)), 1./(-gplik.fh.llg2(gplik, y(ind(i1)), f_mode(ind(i1)), 'latent', z_ind)));
    fh_p = @(f) exp(arrayfun(@(a) gplik.fh.ll(gplik, y(ind(i1)), a, z_ind), f))./t_tilde(f).*norm_pdf(f,Ef(ind(i1)),sqrt(cii));
  else
    cii = Covf2(ind(i1), ind(i1));
    fh_p = @(f) norm_pdf(f,Ef2(ind(i1)),sqrt(cii));
  end
 
  % Loop through grid points
  for i=1:nin
    
    % Compute conditional covariance matrices and mean vector
    if ~predictive
      if isempty(z)
        z_tmp = [];
      else
        z_tmp = z; z_tmp(ind(i1)) = [];
      end
      y_tmp = y; y_tmp(ind(i1)) = [];
      x_tmp = x; x_tmp(ind(i1),:) = [];
      Covf_tmp = Covf;
      cji = Covf_tmp(ind(i1),:);
      cji(ind(i1)) = [];
      Covf_tmp(ind(i1),:) = []; Covf_tmp(:,ind(i1)) = [];
      cjj = Covf_tmp;
      ci = cjj - cji'*(1/cii)*cji;
      mf = Ef; mf(ind(i1)) = [];
      mu = mf+cji'./cii.*(fvec(i,i1)-Ef(ind(i1)));
      f_mode_tmp = f_mode; f_mode_tmp(ind(i1)) = [];
    else
      if isempty(z)
        z_tmp = [];
      else
        z_tmp = z;
      end
      y_tmp = y; 
      x_tmp = x;
      K_fstar = gp_cov(gp,  x, xt(ind(i1),:));
      K_ff = gp_trcov(gp, x);
      cjj = Covf;
      cji = (K_fstar'/K_ff)*cjj;
      ci = cjj - cji'*(1/cii)*cji;
      mu = Ef+cji'./cii.*(fvec(i,i1)-Ef2(ind(i1)));
      f_mode_tmp = f_mode;
    end
%     mu = f_mode_tmp + cji' * (1/cii) * (fvec(ind(i1)) - f_mode(ind(i1)));
%     Kff = gp_cov(gp, x_tmp, x_tmp);
    W = -diag(gplik.fh.llg2(gplik, y_tmp, mu, 'latent', z_tmp));
    llg2_mode = diag(gplik.fh.llg2(gplik, y_tmp, f_mode_tmp, 'latent', z_tmp));
    llg = gplik.fh.llg(gplik, y_tmp, mu, 'latent', z_tmp);
    llg_mode = gplik.fh.llg(gplik, y_tmp, f_mode_tmp, 'latent', z_tmp);
    ll = gplik.fh.ll(gplik,y_tmp, mu, z_tmp);
    ll_mode = gplik.fh.ll(gplik,y_tmp, f_mode_tmp, z_tmp);
%     iKffW = eye(n-1)/(Kff*W);
%     lt_tilde_g = -(mu-iKffW*f_mode_tmp)'*W;
    
%     lnZ = mnorm_lpdf(mu', mu', ci) + gplik.fh.ll(gplik, y_tmp, mu, z_tmp) - mnorm_lpdf(mu', (iKffW*f_mode_tmp + f_mode_tmp)', eye(n-1)/W);
%     lnZ = lnZ - (0.5*(llg-lt_tilde_g')'/(-eye(n-1)/ci - 2.*W))*(llg-lt_tilde_g');
%     lnZ = lnZ - 0.5*sum(log(diag(chol(eye(n-1)/ci + 2.*W))));

    % Computation of correction term by integrating the second order taylor
    % expansion of product of global gaussian approximation conditioned on latent
    % value x_i, q(x_-i|x_i), and t_-i(x_-i)/ttilde_-i(x_-i)
    lnZ = -(sum(log(diag(chol(cjj)))) + 1/cii + sum(log(diag(chol(cii + cji*cjj*cji'))))) + ll - ll_mode - (mu-f_mode_tmp)'*llg_mode - 0.5*(mu-f_mode_tmp)'*llg2_mode*(mu-f_mode_tmp);
    lnZ = lnZ - (0.5*(llg-llg_mode-((mu-f_mode_tmp)'*llg2_mode)')'/(-eye(n)/ci - W - llg2_mode))*(llg-llg_mode-((mu-f_mode_tmp)'*llg2_mode)');
    lnZ = lnZ  + sum(log(diag(chol(cjj)))) + 0.5*log(cii) + 0.5*log(cii+cji*cjj*cji') - evaluate_q(diag(W+llg2_mode), ci);
    
    c(i,i1) = exp(lnZ);
    p(i,i1) = fh_p(fvec(i,i1));

  end
  p(:,i1) = p(:,i1)./trapz(fvec,p(:,i1));

  % Take product of correction terms and tilted distribution terms to get 
  % the final, corrected, distribution.
  pc(:,i1) = p(:,i1).*c(:,i1);
  pc(:,i1) = pc(:,i1)./trapz(fvec, pc(:,i1));
end
end

function [lnZ_q,L1,L2]=evaluate_q(tau_q,K)

  %%%%%%%%%%%%%%%%
  % q-distribution

%   n=length(nu_q);
  ii1=find(tau_q>0); n1=length(ii1); W1=sqrt(tau_q(ii1));
  ii2=find(tau_q<0); n2=length(ii2); W2=sqrt(abs(tau_q(ii2)));
  
  if ~isempty(ii1)
    % Cholesky decomposition for positive sites
    L1=(W1*W1').*K(ii1,ii1);
    L1(1:n1+1:end)=L1(1:n1+1:end)+1;
    L1=chol(L1);
    
  else
    L1=1;
  end
  
  if ~isempty(ii2)
    % Cholesky decomposition for negative sites
    V=bsxfun(@times,K(ii2,ii1),W1')/L1;
    L2=(W2*W2').*(V*V'-K(ii2,ii2));
    L2(1:n2+1:end)=L2(1:n2+1:end)+1;
    
    L2=chol(L2);
    
  else
    L2=1;
  end
  
  % log normalization
  lnZ_q = sum(log(diag(L1))) + sum(log(diag(L2)));
end