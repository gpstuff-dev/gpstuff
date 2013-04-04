function [p, pc, c] = gpla_fact(gp, x, y, fvec,varargin) 
%GPLA_FACT  Factorized correction for marginal posterior using Laplace
%           approximation
% 
%  Description
%    [P, PC, C] = GPLA_FACT(GP, X, Y, FVEC) Evaluates the marginal posterior
%    at given grid points FVEC for given indices. Returns tilted
%    distribution without any correction P, with factorized correction
%    terms PC and the correction terms C.
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
%   Reference
%     Cseke & Heskes (2011). Approximate Marginals in Latent Gaussian
%     Models. Journal of Machine Learning Research 12 (2011), 417-454
%         
%   See also
%     GPEP_FACT, GPLA_CM2, DEMO_IMPROVEDMARGINALS

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
predictive = false;
z = ip.Results.z;
ind = ip.Results.ind;
xt = ip.Results.xt;
[n,tmp] = size(x);
[nin, n_ind] = size(fvec);
[Ef, Covf] = gp_jpred(gp,x,y,x, 'z', z);
if size(ind,1)==1
  ind=ind';
end
if size(ind,1) ~= n_ind
  error('Given latent grid matrix fvec must be size n x m, where m is the length of ind vector');
end
if ~isempty(xt) && ~isequal(xt, x)
  % Predictive equations if given xt, mind that if xt(ind) is in training
  % input x, predictive equations might not work correctly.
  predictive = true;
  [Ef2, Covf2] = gp_jpred(gp,x,y,xt,'z',z);
end
[tmp, tmp, tmp, param] = gpla_e(gp_pak(gp), gp, x,y,'z',z);
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
K_ff = gp_trcov(gp, x);
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
%     ll = gplik.fh.ll(gplik, y(ind(i1)), f_mode(ind(i1)), z_ind);
%     llg = gplik.fh.llg(gplik, y(ind(1)), f_mode(ind(i1)), 'latent', z_ind);
%     llg2 = gplik.fh.llg2(gplik, y(ind(i1)), f_mode(ind(i1)), 'latent', z_ind);
    t_tilde = @(f)  exp(ll(ind(i1)) + (f-f_mode(ind(i1)))*llg(ind(i1)) + 0.5*(f-f_mode(ind(i1))).^2*llg2(ind(i1)));
    fh_p = @(f) exp(arrayfun(@(a) gplik.fh.ll(gplik, y(ind(i1)), a, z_ind), f))./t_tilde(f).*norm_pdf(f,Ef(ind(i1)),sqrt(cii));
  else
    cii = Covf2(ind(i1), ind(i1));
    fh_p = @(f) norm_pdf(f,Ef2(ind(i1)),sqrt(cii));
  end
  minf = Ef-9.*sqrt(diag(Covf));
  maxf = Ef+9.*sqrt(diag(Covf));
  
  
  if ~predictive
    cji = Covf(ind(i1),:); %cji(ind(i1)) = [];
    cjj = Covf; %cjj(ind(i1),:) = []; cjj(:,ind(i1)) = [];
    ci = diag(cjj)-(cji'*(1/cii)).*cji';
    mf = Ef; %mf(ind(i1)) = [];
    inds=[1:(ind(i1)-1) (ind(i1)+1):n];
  else
    K_fstar = gp_cov(gp,  x, xt(ind(i1),:));
    cjj = Covf;
    cji = (K_fstar'/K_ff)*cjj;
    ci = diag(cjj)-cji'.*(1/cii).*cji';
    inds=1:n;
  end
  % Loop through grid points
  for i=1:nin
%     sprintf('Grid point %d', i)
    c_ii = ones(n,1);
    c_ii2 = ones(n,1);
    
    % Variance and mean for global gaussian approximation conditioned on
    % other data grid poins, q(x_j|x_i) or in predictive case, q(x_j,
    % x_*)
    if ~predictive
      mu = mf+cji'./cii.*(fvec(i,i1)-Ef(ind(i1)));
    else
      mu = Ef+cji'./cii.*(fvec(i,i1)-Ef2(ind(i1)));
    end
    m1 = (f_mode-llg./llg2);
    s1 = sqrt(-1./llg2);
    C1 = exp(ll+llg2.*f_mode.^2-llg2.*m1.^2 - llg.*f_mode);
    m2 = mu;
    s2 = sqrt(ci);
    C2 = 1./sqrt(2*pi*s2.^2);
    
    s = sqrt(1./(1./s2.^2 - 1./s1.^2));
    m = (m2./s2.^2 - m1./s1.^2).*s.^2;
    
    Z = C1./C2.*exp(-1./(2*(-s1.^2+s2.^2)).*(m1-m2).^2).*sqrt(2*pi*s.^2);
    c_ii = Z(inds).*exp(gp.lik.fh.tiltedMoments(gplik, y(inds), 1:length(inds), s(inds).^2, m(inds), z));

%     % Loop through other points in x, exclude point to which current latent grid
%     % corresponds to (if not predictive).
%     for j=1:n
%       if j==ind(i1) && ~predictive
%         continue;
%       end
%       
%       if isempty(z)
%         z1 = [];
%       else
%         z1 = z(j);
%       end
%       
%       % Function handle for the term approximations of the likelihood
%       t_i = @(f) exp(ll(j) + (f-f_mode(j))*llg(j) + 0.5*(f-f_mode(j)).^2*llg2(j));
%       
%       % Finally calculate the correction term by integrating over latent
%       % value x_j   
%       c_ii(j) = quadgk(@(f) norm_pdf(f,mu(j),sqrt(ci(j))).*arrayfun(@(b) exp(gplik.fh.ll(gplik, y(j), b, z1)),f)./t_i(f),minf(j),maxf(j), 'absTol', 1e-1, 'relTol', 1e-1);
%     end
    c(i,i1) = prod(c_ii);
    p(i,i1) = fh_p(fvec(i,i1));

  end
  p(:,i1) = p(:,i1)./trapz(fvec,p(:,i1));

  % Take product of correction terms and tilted distribution terms to get 
  % the final, corrected, distribution.
  pc(:,i1) = p(:,i1).*c(:,i1);
  pc(:,i1) = pc(:,i1)./trapz(fvec, pc(:,i1));
end
end
