function [pc, p, c] = gp_predcm(gp,x,y,fvec,varargin)
%GP_PREDCM  Corrections for marginal posterior 
%
%  Description
%    [P, PC, C] = GP_PREDCM(GP, X, Y, FVEC, XT, OPTIONS) Evaluates the 
%    marginal posterior at given grid points FVEC for given indices of XT 
%    or X (XT not given). Returns tilted distribution P if XT is empty or 
%    equal to X, otherwise predictive distribution, corrected predictive/tilted 
%    distribution PC and correction terms C, where PC_i = P_i*C_i for every
%    grid point i in fvec.
%
%
%   OPTIONS is optional parameter-value pair
%      z      - optional observed quantity in triplet (x_i,y_i,z_i)
%               Some likelihoods may use this. For example, in case of
%               Poisson likelihood we have z_i=E_i, that is, expected value
%               for ith case.
%      ind    - Index defining which point in data (X or XT) latent grid fvec
%               corresponds to. If ind is vector of size m x 1, fvec must
%               be matrix of size n x m, where n is the number of grid
%               points for each index. Default = 1.
%      correction - Method used for evaluating correction terms C. Possible
%                   methods are 'fact' (default) for EP and either 'fact'
%                   or 'cm2' (default) for Laplace.
%
%   Reference
%     Cseke & Heskes (2011). Approximate Marginals in Latent Gaussian
%     Models. Journal of Machine Learning Research 12 (2011), 417-454
%
%   See also
%     DEMO_IMPROVEDMARGINALS

% Copyright (c) 2011,2013 Ville Tolvanen

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

ip=inputParser;
ip.addRequired('gp',@isstruct);
ip.addRequired('x', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addRequired('y', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addRequired('fvec',  @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addOptional('xt', [], @(x) isreal(x) && all(isfinite(x(:))))
ip.addParamValue('z', [], @(x) isreal(x) && all(isfinite(x(:))))
ip.addParamValue('ind', 1, @(x) isreal(x) && all(isfinite(x(:))))
ip.addParamValue('correction', 'fact', @(x) ismember(x, {'fact', 'cm2'}))
if rem(size(varargin,2), 2) == 0
  ip.parse(gp, x, y, fvec,[],varargin{:});
else
  ip.parse(gp, x, y, fvec,varargin{:});
end
z = ip.Results.z;
ind = ip.Results.ind;
xt = ip.Results.xt;
predictive=false;
gplik=gp.lik;
[nin, n_ind] = size(fvec);
n=size(x,1);
[Ef, Covf] = gp_jpred(gp,x,y,x, 'z', z);
if isfield(ip.UsingDefaults, 'correction') && isequal(gp.latent_method, 'Laplace')
  correction='cm2';
else
  correction = ip.Results.correction;
end
if size(ind,1)==1
  ind=ind';
end
if (size(ind,1) ~= n_ind) || ~(length(ind) == isvector(fvec))
  error('Given latent grid matrix fvec must be size n x m, where m is the length of ind vector');
end
if ~isempty(xt) && ~isequal(xt, x)
  % Predictive equations if given xt, mind that if xt(ind) is in training
  % input x, predictive equations might not work correctly.
  predictive = true;
  [Ef2, Covf2] = gp_jpred(gp,x,y,xt,'z',z);
end
switch gp.latent_method
  case 'EP'
    switch correction
      case 'fact'
        [tmp, tmp, tmp, param] = gpep_e(gp_pak(gp), gp, x,y,'z',z);
        [tautilde, nutilde, muvec_i, sigm2vec_i] = ...
              deal(param.tautilde, param.nutilde, param.muvec_i, param.sigm2vec_i);
        
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
            K_ff = gp_trcov(gp, x);
          end
          
          % Loop through grid points
          for i=1:nin
            
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
            c_ii = Z(inds)./Ztilde(inds).*exp(gp.lik.fh.tiltedMoments(gplik, y(inds), 1:length(inds), s(inds).^2, m(inds), z));
            c_i = prod(c_ii);
            c(i,i1) = c_i;
            p(i,i1) = fh_p(fvec(i,i1));
            
          end
          p(:,i1) = p(:,i1)./trapz(fvec,p(:,i1));
          if any(isnan(c(:,i1)))
            warning('NaNs in moment computations')
            c(isnan(c(:,i1)),i1)=0;
          end
          
          % Take product of correction terms and tilted distribution terms to get
          % the final, corrected, distribution.
          pc(:,i1) = p(:,i1).*c(:,i1);
          pc(:,i1) = pc(:,i1) ./ trapz(fvec, pc(:,i1));
        end
      case 'cm2'
        error('Cant use cm2 correction with EP, use fact');
    end
  case 'Laplace'
    
    [tmp, tmp, tmp, param] = gpla_e(gp_pak(gp), gp, x,y,'z',z);
    f_mode = param.f;
    pc = zeros(nin,size(ind,1)); p = zeros(nin,size(ind,1)); c = zeros(nin,size(ind,1));
    ll = arrayfun(@(f,yy) gplik.fh.ll(gplik, yy, f, z), f_mode, y);
    llg = gplik.fh.llg(gplik, y, f_mode, 'latent', z);
    llg2 = gplik.fh.llg2(gplik, y, f_mode, 'latent', z);
    K_ff = gp_trcov(gp, x);
    
    switch correction
      case 'fact'
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
            t_tilde = @(f)  exp(ll(ind(i1)) + (f-f_mode(ind(i1)))*llg(ind(i1)) + 0.5*(f-f_mode(ind(i1))).^2*llg2(ind(i1)));
            fh_p = @(f) exp(arrayfun(@(a) gplik.fh.ll(gplik, y(ind(i1)), a, z_ind), f))./t_tilde(f).*norm_pdf(f,Ef(ind(i1)),sqrt(cii));
          else
            cii = Covf2(ind(i1), ind(i1));
            fh_p = @(f) norm_pdf(f,Ef2(ind(i1)),sqrt(cii));
          end
          
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
            
            c(i,i1) = prod(c_ii);
            p(i,i1) = fh_p(fvec(i,i1));
            
          end
          p(:,i1) = p(:,i1)./trapz(fvec,p(:,i1));
          if any(isnan(c(:,i1)))
            warning('NaNs in moment computations')
            c(isnan(c(:,i1)),i1)=0;
          end
          
          % Take product of correction terms and tilted distribution terms to get
          % the final, corrected, distribution.
          pc(:,i1) = p(:,i1).*c(:,i1);
          pc(:,i1) = pc(:,i1)./trapz(fvec, pc(:,i1));
        end
        
      case 'cm2'
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
            
            lc(i,i1) = lnZ;
            p(i,i1) = fh_p(fvec(i,i1));
            
          end
          p(:,i1) = p(:,i1)./trapz(fvec,p(:,i1));
          c(:,i1) = exp(lc(:,i1)-mean(lc(:,i1)));
          
          % Take product of correction terms and tilted distribution terms to get
          % the final, corrected, distribution.
          pc(:,i1) = p(:,i1).*c(:,i1);
          pc(:,i1) = pc(:,i1)./trapz(fvec, pc(:,i1));
        end
    end
    
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
