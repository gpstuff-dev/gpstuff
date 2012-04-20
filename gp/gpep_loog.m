function gloo = gpep_loog(w, gp, x, y, varargin)
%GP_LOOG  Evaluate the gradient of the mean negative log
%         leave-one-out predictive density
%
%   Description
%     LOOG = GP_LOOG(W, GP, X, Y) takes a parameter vector W,
%     Gaussian process structure GP, a matrix X of input vectors
%     and a matrix Y of targets, and evaluates the gradient of the
%     mean negative log leave-one-out predictive density (see
%     GP_LOOE).
%
%   References:
%     S. Sundararajan and S. S. Keerthi (2008). Predictive
%     Approaches for Gaussian Process Classifier Model Selection.
%     Tech report.
%
%  See also
%    GP_LOOE, GP_SET, GP_PAK, GP_UNPAK
%

% Copyright (c) 2012 Ville Tolvanen

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

ip=inputParser;
ip.FunctionName = 'GPEP_LOOE';
ip.addRequired('w', @(x) isempty(x) || ...
  isvector(x) && isreal(x) && all(isfinite(x)));
ip.addRequired('gp', @(x) isstruct(x) || iscell(x));
ip.addRequired('x', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addRequired('y', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addParamValue('z', [], @(x) isreal(x) && all(isfinite(x(:))))
ip.parse(w, gp, x, y, varargin{:});
z=ip.Results.z;

if isfield(gp,'mean') & ~isempty(gp.mean.meanFuncs)
  error('GPEP_LOOG: Mean functions not yet supported');
end

gp=gp_unpak(gp, w);
ncf = length(gp.cf);
n = size(x,1);

gloo = [];

[tmp, tmp, tmp, tautilde, nutilde, tmp, tmp, tmp, muvec_i, sigma2vec_i, lnZ_i] = gpep_e(w, gp, x, y, 'z', z);
% [m, C] = gpep_jpred(gp, x, y, x);
zi = muvec_i./sqrt(1+sigma2vec_i);
Nzi = 1./sqrt(2*pi).*exp(-zi.^2/2);

sigma2site = 1./tautilde;
musite = diag(tautilde)\nutilde;

K = gp_trcov(gp,x);
if issparse(K)
  error('Sparse covariance matrices not yet supported in GPEP_LOOG');
else
  C = (K - K/(K + diag(1./tautilde))*K);
  m = C*(diag(sigma2site)\musite);
end

switch gp.type
  case 'FULL'
    i0 = 0;
    for i=1:ncf
      gpcf = gp.cf{i};
      DKff = gpcf.fh.cfg(gpcf, x);
      tmp_C = (eye(n) - (K+diag(1./tautilde))\K);
      for i2=1:size(DKff,2)
        i0 = i0 + 1;
        dC = tmp_C'*DKff{i2}*tmp_C;
        dm = dC*(diag(sigma2site)\musite);
        dsigma2vec_i = sigma2vec_i.^2./diag(C).^2.*diag(dC);
        dmuvec_i = muvec_i./sigma2vec_i.*dsigma2vec_i + sigma2vec_i./diag(C).^2.*(diag(C).*dm - m.*diag(dC));
        
        dpyt{i2} = Nzi.*y./sqrt(1+sigma2vec_i).*(dmuvec_i - 0.5*zi./sqrt(1+sigma2vec_i).*dsigma2vec_i);
        
        gloo(i0) = -1/n*sum(dpyt{i2}./norm_cdf(y.*zi));
      end
      
    end
  case 'FIC'
    
  case {'PIC' 'PIC_BLOCK'}
    
  case 'CS+FIC'
    
  case 'SSGP'
end

