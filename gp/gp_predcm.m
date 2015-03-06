function [pc, fvecm2, p, c] = gp_predcm(gp,x,y,varargin)
%GP_PREDCM  Corrections for latent marginal posterior
%
%  Description
%    [PC, FVEC, P, C] = GP_PREDCM(GP, X, Y, XT, OPTIONS) Evaluates the
%    corrected marginal posterior of latent variable at given indices
%    of XT or X if XT is empty or not given. Marginal posterior
%    corrections are evaluated in 9 Gauss-Hermite points, after which
%    piecewise cubic Hermite interpolation is used interpolate
%    logarithm of the correction terms to a finer grid. Returns tilted
%    distribution P if XT is empty or equal to X, otherwise predictive
%    distribution, corrected predictive/tilted distribution PC and
%    correction terms C, where PC_i = P_i*C_i for every grid point i
%    in grid FVEC. FVEC is linearly spaced grid from predictive
%    distribution between mean minus/plus 4 standard deviations
%
%
%   OPTIONS is optional parameter-value pair
%      z      - optional observed quantity in triplet (x_i,y_i,z_i)
%               Some likelihoods may use this. For example, in case of
%               Poisson likelihood we have z_i=E_i, that is, expected value
%               for ith case.
%      ind    - Index vector or scalar defining the indices of data
%               points at which the marginal posterior corrections are
%               done. Default = 1.
%      fcorr  - Method used for evaluating correction terms C. Possible
%               methods are 'fact' (default) for EP and either 'fact'
%               or 'cm2' (default) for Laplace. If method is 'on',
%               the default methods are used.
%      ng     - Number of grid points evaluated from the spline. Default is 50.
%
%   Reference
%     Cseke & Heskes (2011). Approximate Marginals in Latent Gaussian
%     Models. Journal of Machine Learning Research 12 (2011), 417-454
%
%   See also
%     DEMO_IMPROVEDMARGINALS
%
% Copyright (c) 2011,2013 Ville Tolvanen

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

ip=inputParser;
ip.addRequired('gp',@isstruct);
ip.addRequired('x', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addRequired('y', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
%ip.addRequired('fvec',  @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addOptional('xt', [], @(x) isreal(x) && all(isfinite(x(:))))
ip.addParamValue('z', [], @(x) isreal(x) && all(isfinite(x(:))))
ip.addParamValue('ind', 1, @(x) isreal(x) && all(isfinite(x(:))))
ip.addParamValue('ng', 50, @(x) isreal(x) && all(isfinite(x(:))) && x > 1)
ip.addParamValue('fcorr', 'on', @(x) ismember(x, {'fact', 'cm2', 'on','lr'}))
ip.addParamValue('tstind', [], @(x) isempty(x) || iscell(x) ||...
                   (isvector(x) && isreal(x) && all(isfinite(x)&x>0)))
if rem(size(varargin,2), 2) == 0
  ip.parse(gp, x, y, [],varargin{:});
else
  ip.parse(gp, x, y, varargin{:});
end
tstind = ip.Results.tstind;
z = ip.Results.z;
ind = ip.Results.ind;
xt = ip.Results.xt;
ng = ip.Results.ng;
predictive=false;
gplik=gp.lik;
n=size(x,1);
[Ef, Covf] = gp_jpred(gp,x,y,x, 'z', z, 'tstind', 1:n);
Covf=full(Covf);
if ismember('fcorr', ip.UsingDefaults)
  if isequal(gp.latent_method, 'Laplace')
    % Default for Laplace
    fcorr='cm2';
  else
    % Default for EP
    fcorr='fact';
  end
else
  fcorr=ip.Results.fcorr;
end
ind=ind(:);
if ~isempty(xt) && ~isequal(xt, x)
  % Predictive equations if given xt, mind that if xt(ind) is in training
  % input x, predictive equations might not work correctly.
  predictive = true;
  [Ef2, Varf2]=gp_pred(gp,x,y,xt,'z',z,'tstind',tstind);
%   [Ef2, Covf2] = gp_jpred(gp,x,y,xt,'z',z, 'tstind', tstind);
%   Covf2=full(Covf2);
end
nin = 11;
fvecm=zeros(nin,length(ind));
fvecm2=zeros(ng,length(ind));
for i1=1:length(ind)
  i2=ind(i1);
  % Form evaluation points for spline formation & grid points to be
  % evaluated from the spline
  minf = 6;
  maxf = 6;
  if ~predictive
    fvecm(:,i1)=Ef(i2)+[-3.668 -2.783 -2.026 -1.326 -0.657 0 0.657 1.326 2.026 2.783 3.668].*sqrt(Covf(i2,i2));
%     fvecm(:,i1)=Ef(i2)+[-3.191 -2.267 -1.469 -0.724 0 0.724 1.469 2.267 3.191].*sqrt(Covf(i2,i2));
    fvecm2(:,i1)=linspace(Ef(i2)-minf.*sqrt(Covf(i2,i2)), Ef(i2)+maxf.*sqrt(Covf(i2,i2)),ng)';
  else
    fvecm(:,i1)=Ef2(i2)+[-3.668 -2.783 -2.026 -1.326 -0.657 0 0.657 1.326 2.026 2.783 3.668].*sqrt(Varf2(i2));
%     fvecm(:,i1)=Ef2(i2)+[-3.191 -2.267 -1.469 -0.724 0 0.724 1.469 2.267 3.191].*sqrt(Covf2(i2,i2));
    fvecm2(:,i1)=linspace(Ef2(i2)-minf.*sqrt(Varf2(i2)), Ef2(i2)+maxf.*sqrt(Varf2(i2)),ng)';
  end
end
lc = zeros(nin, length(ind));
pc = zeros(ng, size(ind,1)); lp = zeros(ng,size(ind,1)); c = zeros(ng,size(ind,1));
lp2 = zeros(nin,size(ind,1)); p = zeros(ng,size(ind,1));

switch gp.latent_method
  case 'EP'
    
    if isequal(fcorr, 'lr')
        [Efloo,Varfloo]=gpep_loopred(gp,x,y,'z',z);
    end
    
    switch fcorr
      case {'fact', 'lr'}
        [tmp, tmp, tmp, param] = gpep_e(gp_pak(gp), gp, x,y,'z',z);
        [tautilde, nutilde, muvec_i, sigm2vec_i] = ...
              deal(param.tautilde, param.nutilde, param.muvec_i, param.sigm2vec_i);
        
        
        % Compute tilted moments
        logM02 = gp.lik.fh.tiltedMoments(gp.lik, y, 1:n, sigm2vec_i, muvec_i, z);
        
        if predictive
          K_ff = gp_trcov(gp, x);
        end
        
        % Loop through grid indices
        for i1=1:size(ind,1)
          fvec = fvecm(:,i1);
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
%             logM0 = gp.lik.fh.tiltedMoments(gp.lik, y, ind(i1), sigm2vec_i(ind(i1)), muvec_i(ind(i1)), z);
%             Z_p = exp(logM0)*sqrt(2*pi)*sqrt(sigm2vec_i(ind(i1))+1./tautilde(ind(i1)))*exp(0.5*(muvec_i(ind(i1))-nutilde(ind(i1))./tautilde(ind(i1))).^2/(sigm2vec_i(ind(i1))+1./tautilde(ind(i1))));
            
            % Function handle to marginal distribution without any fcorr parameters
            if isequal(fcorr, 'fact')
                cav = @(f) norm_lpdf(f,Ef(ind(i1)),sqrt(cii)) - norm_lpdf(f, nutilde(ind(i1))/tautilde(ind(i1)), 1/sqrt(tautilde(ind(i1))));
            else
                cav = @(f) norm_lpdf(f, Efloo(ind(i1)), sqrt(Varfloo(ind(i1))));
            end
            fh_p = @(f) (arrayfun(@(a) gplik.fh.ll(gplik, y(ind(i1)), a, z_ind), f)) + cav(f);
          else
            inds=1:n;
            cii = Varf2(ind(i1));
            fh_p = @(f) norm_lpdf(f,Ef2(ind(i1)),sqrt(cii));
          end
          
          % Loop through grid points
          for i=1:nin
            
            % Variance and mean for global Gaussian approximation conditioned on
            % other data grid points, q(x_j|x_i) or in predictive case, q(x_j,
            % x_*)
            if ~predictive
              cji = Covf(ind(i1),:);% cji(ind(i1)) = [];
              cjj = Covf;% cjj(ind(i1),:) = []; cjj(:,ind(i1)) = [];
              ci = diag(cjj)-(cji'*(1/cii)).*cji';
              mf = Ef; %mf(ind(i1)) = [];
              mu = mf+cji'./cii.*(fvec(i)-Ef(ind(i1)));
            else
              K_fstar = gp_cov(gp,  x, xt(ind(i1),:));
              cjj = Covf;
              cji = (K_fstar'/K_ff)*cjj;
              ci = diag(cjj)-cji'.*(1/cii).*cji';
              mu = Ef+cji'./cii.*(fvec(i)-Ef2(ind(i1)));
            end
            % Loop through other points in x, exclude point to which current latent grid
            % corresponds to (if not predictive).
            lZtilde=logM02 + log(sqrt(2*pi)) + log(sqrt(sigm2vec_i+1./tautilde)) + 0.5*(muvec_i-nutilde./tautilde).^2./(sigm2vec_i+1./tautilde);
            m1 = nutilde./tautilde;
            s1 = 1./sqrt(tautilde);
            m2 = mu;
            s2 = sqrt(ci);
            
            s = sqrt(1./(1./s2.^2 - 1./s1.^2));
            m = (m2./s2.^2 - m1./s1.^2).*s.^2;
            lZ = log(s1) - log(s2) - 1./(2*(-s1.^2+s2.^2)).*(m1-m2).^2 +log(sqrt(2*pi*s.^2));
            lc_ii = lZ(inds) - lZtilde(inds) + gp.lik.fh.tiltedMoments(gplik, y(inds), 1:length(inds), s(inds).^2, m(inds), z);
            lc(i,i1) = sum(lc_ii);
            %p(i,i1) = fh_p(fvec(i,i1));
            
          end
          lp(:,i1) = fh_p(fvecm2(:,i1));
          lp2(:,i1) = fh_p(fvecm(:,i1));
          lp(:,i1) = lp(:,i1)-max(lp(:,i1));
          lp2(:,i1) = lp2(:,i1)-max(lp2(:,i1));
        end
      otherwise
        error('Invalid method for EP, use fact');
    end
  case 'Laplace'
    
    [tmp, tmp, tmp, param] = gpla_e(gp_pak(gp), gp, x,y,'z',z);
    f_mode = param.f;
    if ~isempty(z)
      ll = arrayfun(@(f,yy, zz) gplik.fh.ll(gplik, yy, f, zz), f_mode, y, z);
    else
      ll = arrayfun(@(f,yy) gplik.fh.ll(gplik, yy, f, z), f_mode, y);
    end
    llg = gplik.fh.llg(gplik, y, f_mode, 'latent', z);
    llg2 = gplik.fh.llg2(gplik, y, f_mode, 'latent', z);
    K_ff = gp_trcov(gp, x);
    if isequal(fcorr, 'lr')
        [Efloo,Varfloo]=gpla_loopred(gp,x,y,'z',z,'method','lrs');
    end
    
    switch fcorr
      case 'fact'
        % Loop through grid indices
        for i1=1:size(ind,1)
          fvec = fvecm(:,i1);
          if ~predictive
            cii = Covf(ind(i1),ind(i1));
            if isempty(z)
              z_ind = [];
            else
              z_ind = z(ind(i1));
            end
            
            % Function handle to marginal distribution without any fcorr parameters
            t_tilde = @(f)  (ll(ind(i1)) + (f-f_mode(ind(i1)))*llg(ind(i1)) + 0.5*(f-f_mode(ind(i1))).^2*llg2(ind(i1)));
            fh_p = @(f) (arrayfun(@(a) gplik.fh.ll(gplik, y(ind(i1)), a, z_ind), f)) - t_tilde(f) + norm_lpdf(f,Ef(ind(i1)),sqrt(cii));
          else
            cii = Varf2(ind(i1));
            fh_p = @(f) norm_lpdf(f,Ef2(ind(i1)),sqrt(cii));
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
              mu = mf+cji'./cii.*(fvec(i)-Ef(ind(i1)));
            else
              mu = Ef+cji'./cii.*(fvec(i)-Ef2(ind(i1)));
            end
            m1 = (f_mode-llg./llg2);
            s1 = sqrt(-1./llg2);
            lC1 = ll+llg2.*f_mode.^2-llg2.*m1.^2 - llg.*f_mode;
            m2 = mu;
            s2 = sqrt(ci);
            lC2 = log(1./sqrt(2*pi*s2.^2));
            
            s = sqrt(1./(1./s2.^2 - 1./s1.^2));
            m = (m2./s2.^2 - m1./s1.^2).*s.^2;
            
            lZ = lC1 - lC2 - 1./(2*(-s1.^2+s2.^2)).*(m1-m2).^2 + log(sqrt(2*pi*s.^2));
            lc_ii = lZ(inds) + gp.lik.fh.tiltedMoments(gplik, y(inds), 1:length(inds), s(inds).^2, m(inds), z);
            
            %c(i,i1) = prod(c_ii);
            lc(i,i1) = sum(lc_ii);
            %p(i,i1) = fh_p(fvec(i,i1));
            
          end
          lp(:,i1) = fh_p(fvecm2(:,i1));
          lp2(:,i1) = fh_p(fvecm(:,i1));
          lp(:,i1) = lp(:,i1)-max(lp(:,i1));
          lp2(:,i1) = lp2(:,i1)-max(lp2(:,i1));
        end
        
      case {'cm2', 'lr'}
        % Loop through grid indices
        for i1=1:size(ind,1)
          fvec = fvecm(:,i1);
          if ~predictive
            cii = Covf(ind(i1),ind(i1));
            if isempty(z)
              z_ind = [];
            else
              z_ind = z(ind(i1));
            end
            
            % Function handle to marginal distribution without any fcorr parameters
            if isequal(fcorr, 'cm2')
%                t_tilde(f) = @(f) (ll(ind(i1)) + (f-f_mode(ind(i1)))*llg(ind(i1)) + 0.5*(f-f_mode(ind(i1))).^2*llg2(ind(i1)));
                cav = @(f) norm_lpdf(f,Ef(ind(i1)),sqrt(cii)) - (ll(ind(i1)) + (f-f_mode(ind(i1)))*llg(ind(i1)) + 0.5*(f-f_mode(ind(i1))).^2*llg2(ind(i1)));
            else
                cav = @(f) norm_lpdf(f, Efloo(ind(i1)), sqrt(Varfloo(ind(i1))));
            end
            fh_p = @(f) arrayfun(@(a) gplik.fh.ll(gplik, y(ind(i1)), a, z_ind), f) + cav(f);
%             fh_p = @(f) (arrayfun(@(a) gplik.fh.ll(gplik, y(ind(i1)), a, z_ind), f)) - t_tilde(f) + norm_lpdf(f,Ef(ind(i1)),sqrt(cii));
          else
            cii = Varf2(ind(i1));
            fh_p = @(f) norm_lpdf(f,Ef2(ind(i1)),sqrt(cii));
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
          
          lnZ0 = -1/cii - ll_mode + 0.5*log(cii);
          icW = -eye(size(ci))/ci  - llg2_mode;
          % Loop through grid points
          for i=1:nin
            
            if isempty(z)
              z_tmp = [];
            else
              z_tmp = z(inds);
            end
            % Compute conditional covariance matrices and mean vector
            if ~predictive
              mu = mf+cji'./cii.*(fvec(i)-Ef(ind(i1)));
            else
              mu = mf+cji'./cii.*(fvec(i)-Ef2(ind(i1)));
            end
            W = -diag(gplik.fh.llg2(gplik, y_tmp, mu, 'latent', z_tmp));
            deriv = gplik.fh.llg(gplik, y_tmp, mu, 'latent', z_tmp);
            logll = gplik.fh.ll(gplik,y_tmp, mu, z_tmp);
            
            % Computation of correction term by integrating the second order Taylor
            % expansion of product of global Gaussian approximation conditioned on latent
            % value x_i, q(x_-i|x_i), and t_-i(x_-i)/ttilde_-i(x_-i)
            mu1=mu-f_mode_tmp;
            lnZ = lnZ0 + logll - mu1'*llg_mode - 0.5*mu1'*llg2_mode*mu1;
            mu2=deriv-llg_mode-(mu1'*llg2_mode)';
            lnZ = lnZ - (0.5*mu2'/(icW - W))*mu2;
            lnZ = lnZ  - evaluate_q(diag(W+llg2_mode), ci);
            
            lc(i,i1) = lnZ;
            %p(i,i1) = fh_p(fvec(i,i1));
            
          end
          lp(:,i1) = fh_p(fvecm2(:,i1));
          lp2(:,i1) = fh_p(fvecm(:,i1));
          lp(:,i1) = lp(:,i1)-max(lp(:,i1));
          lp2(:,i1) = lp2(:,i1)-max(lp2(:,i1));       
          
        end
    end
    
end

for i1=1:length(ind)
  fvec = fvecm(:,i1);
  
  % Interpolate correction to these grid points 
  % using piecewise cubic Hermite interpolation
  fvec2 = fvecm2(:,i1);
  lc(:,i1)=lc(:,i1)-lc(6,i1);
  
  % Check that the corrected distribution has decreasing tails
  lctmp=lc(:,i1);
  lptmp=(lp2(:,i1)+lc(:,i1));
  fvectmp=fvec;
  while lptmp(end)-lptmp(end-1) > 0
    lctmp=lctmp(1:end-1);
    lptmp=lptmp(1:end-1);
    fvectmp=fvectmp(1:end-1);
  end
  while lptmp(1)-lptmp(2) > 0
    lctmp=lctmp(2:end);
    lptmp=lptmp(2:end);
    fvectmp=fvectmp(2:end);
  end
    
  
  if (sum(isnan(lctmp))>0)
    warning('NaNs in moment computations')
    lctmp(isnan(lctmp))=0;
  end
  lc2(:,i1) = interp1(fvectmp, lctmp, fvec2, 'pchip');

  % Make correction
  pc(:,i1)=exp(lc2(:,i1) + lp(:,i1));
  
  if any(isnan(pc(:,i1))) || any(isinf(pc(:,i1))) 
    warning('NaNs in moment computations')
    pc(isnan(pc(:,i1)),i1)=0;
    pc(isinf(pc(:,i1)),i1)=0;
  end
  if any(pc(:,i1)<eps)
    pc(pc(:,i1)<eps,i1)=0;
    iz1=find(pc(1:ceil(0.5*ng),i1)==0,1);
    iz2=ceil(0.5*ng)+find(pc(ceil(0.5*ng):ng,i1)==0,1,'last');
    pc([1:iz1 iz2:ng],i1)=0;
  end
  
  % Form corrected distribution & normalize
  p(:,i1) = exp(lp(:,i1))./trapz(fvec2,exp(lp(:,i1)));
  pc(:,i1) = pc(:,i1)./trapz(fvec2, pc(:,i1));
  c(:,i1) = pc(:,i1)./p(:,i1);
  
end
if isequal(fcorr, 'lr')
    pc=p;
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
