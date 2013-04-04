function [p, pc, c] = gpep_fact(gp, x, y,fvec,varargin) 
%GPEP_FACT  Factorized correction for marginal likelihood using expectation
%           propagation. 
% 
%  Description
%    [P, PC, C] = GPEP_FACT(GP, X, Y, FVEC) Evaluates marginal likelihood
%    at given grind points FVEC for given indices. Returns tilted
%    distribution without any correction P, with factorized correction
%    terms PC and the corrections terms C.
%
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
%
%   References
%     Cseke & Heskes (2011). Approximate Marginals in Latent Gaussian
%     Models. Journal of Machine Learning Research 12 (2011), 417-454
%         
%   See also
%     GPLA_FACT, DEMO_IMPROVEDMARGINALS

% Copyright (c) 2011 Ville Tolvanen

% This software is distributed under the GNU General Public 
% License (version 3 or later); please refer to the file 
% License.txt, included with the software, for details.

ip=inputParser;
ip.FunctionName = 'GPEP_FACT';
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
predictive = false;
z = ip.Results.z;
ind = ip.Results.ind;
xt = ip.Results.xt;
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
[tmp, tmp, tmp, p] = gpep_e(gp_pak(gp), gp, x,y,'z',z);
[tautilde, nutilde, muvec_i, sigm2vec_i] = deal(p.tautilde, p.nutilde, p.muvec_i, p.sigm2vec_i);
if iscell(gp)
  gplik = gp{1}.lik;
else
  gplik = gp.lik;
end
pc = zeros(nin,size(ind,1)); p = zeros(nin,size(ind,1)); c = zeros(nin,size(ind,1));

% Compute tilted moments
logM02 = gp.lik.fh.tiltedMoments(gp.lik, y, 1:n, sigm2vec_i, muvec_i, z);

% Loop through grid indices
for i1=1:size(ind,1)
  if ~predictive
    inds=[1:(ind(i1)-1) (ind(i1)+1):n];
    cii = Covf(ind(i1),ind(i1));
    if isempty(z)
      z_ind = [];
    else
      z_ind = z(ind(i1));
    end
    
    % Here we keep track of normalizing constants so we dont have to
    % normalize distributions at any point. 
    %     Z_q = sqrt(2*pi*cii);
    logM0 = gp.lik.fh.tiltedMoments(gp.lik, y, ind(i1), sigm2vec_i(ind(i1)), muvec_i(ind(i1)), z);
    Z_p = exp(logM0)*sqrt(2*pi)*sqrt(sigm2vec_i(ind(i1))+1./tautilde(ind(i1)))*exp(0.5*(muvec_i(ind(i1))-nutilde(ind(i1))./tautilde(ind(i1))).^2/(sigm2vec_i(ind(i1))+1./tautilde(ind(i1))));
    
    % Function handle to marginal distribution without any correction parameters
    fh_p = @(f) 1/Z_p*exp(arrayfun(@(a) gplik.fh.ll(gplik, y(ind(i1)), a, z_ind), f))./norm_pdf(f, nutilde(ind(i1))/tautilde(ind(i1)), 1/sqrt(tautilde(ind(i1)))).*norm_pdf(f,Ef(ind(i1)),sqrt(cii));
  else
    inds=1:n;
    cii = Covf2(ind(i1), ind(i1));
    fh_p = @(f) norm_pdf(f,Ef2(ind(i1)),sqrt(cii));
  end
  minf = Ef-9.*sqrt(diag(Covf));
  maxf = Ef+9.*sqrt(diag(Covf));
  
  % Loop through grid points
  for i=1:nin
    c_ii = ones(n,1);
    
    % Variance and mean for global gaussian approximation conditioned on
    % other data grid poins, q(x_j|x_i) or in predictive case, q(x_j,
    % x_*)
    if ~predictive
      cji = Covf(ind(i1),:);% cji(ind(i1)) = [];
      cjj = Covf;% cjj(ind(i1),:) = []; cjj(:,ind(i1)) = [];
      ci = diag(cjj)-(cji'*(1/cii)).*cji';
      mf = Ef; %mf(ind(i1)) = [];
      mu = mf+cji'./cii.*(fvec(i,i1)-Ef(ind(i1)));
    else
      K_fstar = gp_cov(gp,  x, xt(ind(i1),:));
      K_ff = gp_trcov(gp, x);
      cjj = Covf;
      cji = (K_fstar'/K_ff)*cjj;
      ci = diag(cjj)-cji'.*(1/cii).*cji';
      mu = Ef+cji'./cii.*(fvec(i,i1)-Ef2(ind(i1)));
    end
    % Loop through other points in x, exclude point to which current latent grid
    % corresponds to (if not predictive).
    Ztilde=exp(logM02).*sqrt(2*pi).*sqrt(sigm2vec_i+1./tautilde).*exp(0.5*(muvec_i-nutilde./tautilde).^2./(sigm2vec_i+1./tautilde));
    m1 = nutilde./tautilde;
    s1 = 1./sqrt(tautilde);
    m2 = mu;
    s2 = sqrt(ci);
    
    s = sqrt(1./(1./s2.^2 - 1./s1.^2));
    m = (m2./s2.^2 - m1./s1.^2).*s.^2;
    Z = s1./s2.*exp(-1./(2*(-s1.^2+s2.^2)).*(m1-m2).^2).*sqrt(2*pi*s.^2);
%     Z = s1./s2.*exp(-m2.^2./(2*s2.^2) + m1.^2./(2*s1.^2) + 0.5.*m.^2.*s.^2).*sqrt(2*pi*s.^2);
    c_ii = Z(inds)./Ztilde(inds).*exp(gp.lik.fh.tiltedMoments(gplik, y(inds), 1:length(inds), s(inds).^2, m(inds), z));
%     for j=1:n
%       if j==ind(i1) && ~predictive
%         continue;
%       end
%       % Correction("unnormalizing") constant for gaussian likelihood approximation.
%       Ztilde=exp(logM02(j))*sqrt(2*pi)*sqrt(sigm2vec_i(j)+1./tautilde(j))*exp(0.5*(muvec_i(j)-nutilde(j)./tautilde(j)).^2/(sigm2vec_i(j)+1./tautilde(j)));
%       
%       % Function handle for gaussian approximation of the likelihood
%       t_i = @(f) Ztilde.*norm_pdf(f, nutilde(j)/tautilde(j), 1/sqrt(tautilde(j)));
%       
%       if isempty(z)
%         z1 = [];
%       else
%         z1 = z(j);
%       end
%       
%       % Finally calculate the correction term by integrating over latent
%       % value x_j
% %       f = linspace(minf(j), maxf(j),100);
% %       plot(f, norm_pdf(f,m(j),s(j)).*arrayfun(@(b) exp(gplik.fh.ll(gplik, y(j), b, z1)),f));
% %       pause
% %       c_ii(j) = Z(j)./Ztilde.*exp(gplik.fh.tiltedMoments(gplik, y(j), 1, s(j)^2, m(j), z));
% %       c_ii(j) = quadgk(@(f) norm_pdf(f,m(j),s(j)).*arrayfun(@(b) exp(gplik.fh.ll(gplik, y(j), b, z1)),f),minf(j),maxf(j));
% %       c_ii(j) = quadgk(@(f) Z(j)./Ztilde.*norm_pdf(f,m(j),s(j)).*arrayfun(@(b) exp(gplik.fh.ll(gplik, y(j), b, z1)),f),minf(j),maxf(j));
%       c_ii(j) = quadgk(@(f) exp(norm_lpdf(f,mu(j),sqrt(ci(j)))).*arrayfun(@(b) exp(gplik.fh.ll(gplik, y(j), b, z1)),f)./t_i(f),minf(j),maxf(j));
%     end
    c_i = prod(c_ii);
    c(i,i1) = c_i;
    p(i,i1) = fh_p(fvec(i,i1));

  end
%   p(:,i1) = p(:,i1)./trapz(fvec,p(:,i1));

  % Take product of correction terms and tilted distribution terms to get 
  % the final, corrected, distribution.
  pc(:,i1) = p(:,i1).*c(:,i1);
%   pc(:,i1) = pc(:,i1) ./ trapz(fvec, pc(:,i1));
end
end
