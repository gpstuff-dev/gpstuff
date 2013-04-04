function [p, pc, c] = gpla_cm2(gp, x, y, fvec,varargin) 
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
[e, edata, eprior, param] = gpla_e(gp_pak(gp), gp, x,y,'z',z);
f_mode = param.f;
if iscell(gp)
  gplik = gp{1}.lik;
else
  gplik = gp.lik;
end
pc = zeros(nin,size(ind,1)); p = zeros(nin,size(ind,1)); c = zeros(nin,size(ind,1));
ll = arrayfun(@(f,yy) gplik.fh.ll(gplik, yy, f, z), f_mode, y);
llg = gplik.fh.llg(gplik, y, f_mode, 'latent', z);
llg2 = gplik.fh.llg2(gplik, y, f_mode, 'latent', z);
% Loop through grid indices
for i1=1:size(ind,1)
  if ~predictive
    cii = Covf(ind(i1),ind(i1));
    if isempty(z)
      z_ind = [];
    else
      z_ind = z(ind(i1));
    end
        
    % Function handle to marginal distribution without any correction parameters
    t_tilde = @(f) exp(ll(ind(i1)) + (f-f_mode(ind(i1)))*llg(ind(i1)) + 0.5*(f-f_mode(ind(i1))).^2*llg2(ind(i1)));
    fh_p = @(f) exp(arrayfun(@(a) gplik.fh.ll(gplik, y(ind(i1)), a, z_ind), f))./t_tilde(f).*norm_pdf(f,Ef(ind(i1)),sqrt(cii));
  else
    cii = Covf2(ind(i1), ind(i1));
    fh_p = @(f) norm_pdf(f,Ef2(ind(i1)),sqrt(cii));
  end
 
  if ~predictive
    cji = Covf(ind(i1),:);
    cji(ind(i1)) = [];
    inds=[1:(ind(i1)-1) (ind(i1)+1):n];
  else
    K_fstar = gp_cov(gp,  x, xt(ind(i1),:));
    K_ff = gp_trcov(gp, x);
    cji = (K_fstar'/K_ff)*Covf;
    inds=1:n;
  end
  mf = Ef(inds);
  cjj = Covf(inds,inds);
  y_tmp = y(inds);
  ci = cjj - cji'*(1/cii)*cji;
  f_mode_tmp=f_mode(inds);
  llg2_mode=diag(llg2(inds));
  llg_mode=llg(inds);
  ll_mode=sum(ll(inds));
  
  % Loop through grid points
  for i=1:nin
    
    if isempty(z)
      z_tmp = [];
    else
      z_tmp = z;
    end
      % Compute conditional covariance matrices and mean vector
    if ~predictive   
      mu = mf+cji'./cii.*(fvec(i,i1)-Ef(ind(i1)));
    else
      mu = mf+cji'./cii.*(fvec(i,i1)-Ef2(ind(i1)));
    end
    W = -diag(gplik.fh.llg2(gplik, y_tmp, mu, 'latent', z_tmp));
    deriv = gplik.fh.llg(gplik, y_tmp, mu, 'latent', z_tmp);
    logll = gplik.fh.ll(gplik,y_tmp, mu, z_tmp);

    % Computation of correction term by integrating the second order taylor
    % expansion of product of global gaussian approximation conditioned on latent
    % value x_i, q(x_-i|x_i), and t_-i(x_-i)/ttilde_-i(x_-i) 
    mu1=mu-f_mode_tmp;
    lnZ = -1/cii + logll - ll_mode - mu1'*llg_mode - 0.5*mu1'*llg2_mode*mu1;
    mu2=deriv-llg_mode-(mu1'*llg2_mode)';
    lnZ = lnZ - (0.5*mu2'/(-eye(size(ci))/ci - W - llg2_mode))*mu2;
    lnZ = lnZ + 0.5*log(cii) - evaluate_q(diag(W+llg2_mode), ci);
    
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