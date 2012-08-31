function [Ef, Covf, lpyt, Ey, Vary] = gpla2_pred(gp, x, y, varargin)
%GPLA_ND_PRED Predictions with Gaussian Process Laplace
%             approximation with non-diagonal likelihoods
%
%  Description
%    [EFT, COVFT] = GPLA_ND_PRED(GP, X, Y, XT, OPTIONS) takes
%    a GP structure GP together with a matrix XT of input vectors,
%    matrix X of training inputs and vector Y of training targets,
%    and evaluates the predictive distribution at inputs XT. Returns
%    a posterior mean EFT and covariance COVFT of latent variables.
%
%    [EF, COVF, LPYT] = GPLA_ND_PRED(GP, X, Y, XT, 'yt', YT, OPTIONS)
%    returns also logarithm of the predictive density LPYT of the
%    observations YT at input locations XT. This can be used for
%    example in the cross-validation. Here Y has to be a vector.
%
%    [EF, VARF, LPYT, EYT, VARYT] = GPLA_ND_PRED(GP, X, Y, XT, OPTIONS)
%    returns also the posterior predictive mean EYT and variance VARYT.
%
%    [EF, VARF, LPY, EY, VARY] = GPLA_ND_PRED(GP, X, Y, OPTIONS)
%    evaluates the predictive distribution at training inputs X
%    and logarithm of the predictive density LPY of the training
%    observations Y.
%
%     OPTIONS is optional parameter-value pair
%       predcf - is index vector telling which covariance functions are
%                used for prediction. Default is all (1:gpcfn). See
%                additional information below.
%       tstind - is a vector/cell array defining, which rows of X belong
%                to which training block in *IC type sparse models. Deafult
%                is []. In case of PIC, a cell array containing index
%                vectors specifying the blocking structure for test data.
%                IN FIC and CS+FIC a vector of length n that points out the
%                test inputs that are also in the training set (if none,
%                set TSTIND = []).
%       yt     - is optional observed yt in test points
%       z      - optional observed quantity in triplet (x_i,y_i,z_i)
%                Some likelihoods may use this. For example, in
%                case of Poisson likelihood we have z_i=E_i, that
%                is, expected value for ith case.
%       zt     - optional observed quantity in triplet (xt_i,yt_i,zt_i)
%                Some likelihoods may use this. For example, in
%                case of Poisson likelihood we have zt_i=Et_i, that
%                is, expected value for ith case.
%
%  See also
%    GPLA_ND_E, GPLA_ND_G, GP_PRED, DEMO_MULTICLASS
%
% Copyright (c) 2010 Jaakko Riihim�ki

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

ip=inputParser;
ip.FunctionName = 'GPLA_ND_PRED';
ip.addRequired('gp', @isstruct);
ip.addRequired('x', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addRequired('y', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addOptional('xt', [], @(x) isempty(x) || (isreal(x) && all(isfinite(x(:)))))
ip.addParamValue('yt', [], @(x) isreal(x) && all(isfinite(x(:))))
ip.addParamValue('z', [], @(x) isreal(x) && all(isfinite(x(:))))
ip.addParamValue('zt', [], @(x) isreal(x) && all(isfinite(x(:))))
%ip.addParamValue('predcf', [], @(x) isempty(x) || iscell(x) && isvector(x))
ip.addParamValue('predcf', [], @(x) isempty(x) || ...
  isvector(x) && isreal(x) && all(isfinite(x)&x>0))
ip.addParamValue('tstind', [], @(x) isempty(x) || iscell(x) ||...
  (isvector(x) && isreal(x) && all(isfinite(x)&x>0)))
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

Ey=[];
Vary=[];

[tn, nout] = size(y);

switch gp.type
  % ============================================================
  % FULL
  % ============================================================
  case 'FULL'
    
    [e, edata, eprior, f, L, a, E, M] = gpla2_e(gp_pak(gp), gp, x, y, 'z', z);
    
    switch gp.lik.type
      
      case {'LGP', 'LGPC'}
        
        W=-gp.lik.fh.llg2(gp.lik, y, f, 'latent', z);
        
        ntest=size(xt,1);
        nl=tn;
        nlt=ntest;
        nlp=length(nl); % number of latent processes
        
        if isfield(gp.latent_opt, 'kron') && gp.latent_opt.kron==1
          
          gptmp=gp; gptmp.jitterSigma2=0;
          Ka = gp_trcov(gptmp, unique(x(:,1)));
          wtmp=gp_pak(gptmp); wtmp(1)=0; gptmp=gp_unpak(gptmp,wtmp);
          Kb = gp_trcov(gptmp, unique(x(:,2)));
          clear gptmp
          n1=size(Ka,1);
          n2=size(Kb,1);
          
          [Va,Da]=eig(Ka); [Vb,Db]=eig(Kb);
          
          % eigenvalues of K matrix
          Dtmp=kron(diag(Da),diag(Db));
          [sDtmp,istmp]=sort(Dtmp,'descend');
          
          n = size(y,1);
          % Form the low-rank approximation.  Exclude eigenvalues
          % smaller than gp.latent_opt.eig_tol or take
          % gp.latent_opt.eig_prct*n eigenvalues at most.
          nlr=min([sum(sDtmp>gp.latent_opt.eig_tol) round(gp.latent_opt.eig_prct*n)]);
          sDtmp=sDtmp+gp.jitterSigma2;
          
          itmp1=meshgrid(1:n1,1:n2);
          itmp2=meshgrid(1:n2,1:n1)';
          ind=[itmp1(:) itmp2(:)];
          
          % included eigenvalues
          Dlr=sDtmp(1:nlr);
          % included eigenvectors
          Vlr=zeros(n,nlr);
          for i1=1:nlr
            Vlr(:,i1)=kron(Va(:,ind(istmp(i1),1)),Vb(:,ind(istmp(i1),2)));
          end
        else
          K_nf = gp_cov(gp,xt,x,predcf);
          K = gp_trcov(gp, x);
        end
        
        if isfield(gp,'meanf')
          [H,b_m,B_m Hs]=mean_prep(gp,x,xt);
          if ~(isfield(gp.latent_opt, 'kron') && gp.latent_opt.kron==1)
            K_nf=K_nf + Hs'*B_m*H;
            %K = gp_trcov(gp, x);
            K = K+H'*B_m*H;
          end
        end
        
        % Evaluate the mean
        if isfield(gp.latent_opt, 'kron') && gp.latent_opt.kron==1
          Eft=f;
        else
          deriv = feval(gp.lik.fh.llg, gp.lik, y, f, 'latent', z);
          Eft = K_nf*deriv;
          if isfield(gp,'meanf')
            Eft=Eft + K_nf*(K\H'*b_m);
            %Eft=Eft + K_nf*(K\Hs'*b_m);
          end
        end
        
        if nargout > 1
          
          if isfield(gp.latent_opt, 'kron') && gp.latent_opt.kron==1
            
            Lb=gp_trvar(gp,x)-sum(bsxfun(@times,Vlr.*Vlr,Dlr'),2);
            if isfield(gp,'meanf')
              Dt=[Dlr; diag(B_m)];
              Vt=[Vlr H'];
            else
              Dt=Dlr;
              Vt=Vlr;
            end
            
            g2 = feval(gp.lik.fh.llg2, gp.lik, y, f, 'latent', z);
            Lbt=sum(y)*(g2)+1./Lb;
            
            St=[diag(1./Dt)+Vt'*bsxfun(@times,1./Lb,Vt) zeros(size(Dt,1),1); ...
              zeros(1,size(Dt,1)) 1];
            Pt=[bsxfun(@times,1./Lb,Vt) sqrt(sum(y))*g2];
            Ptt=bsxfun(@times,1./sqrt(Lbt),Pt);
            
            StL=chol(St-Ptt'*Ptt,'lower');
            iStL=StL\(bsxfun(@times,Pt',1./Lbt'));
            
            Covfd=1./Lbt;
            Covfu=iStL;
            Covf{1}=Covfd; Covf{2}=Covfu;
          else
            
            % Evaluate the variance
            if isempty(predcf)
              kstarstarfull = gp_trcov(gp,xt);
            else
              kstarstarfull = gp_trcov(gp,xt,predcf);
            end
            
            if isfield(gp,'meanf')
              kstarstarfull = kstarstarfull + Hs'*B_m*Hs;
            end
            
            if strcmpi(gp.lik.type,'LGPC')
              g2 = feval(gp.lik.fh.llg2, gp.lik, y, f, 'latent', z);
              g2sq=sqrt(g2);
              n1=gp.lik.gridn(1); n2=gp.lik.gridn(2);
              ny2=sum(reshape(y,fliplr(gp.lik.gridn)));
              
              R=zeros(tn);
              for k1=1:n1
                R((1:n2)+(k1-1)*n2,(1:n2)+(k1-1)*n2)=sqrt(ny2(k1))*(diag(g2sq((1:n2)+(k1-1)*n2))-g2((1:n2)+(k1-1)*n2)*g2sq((1:n2)+(k1-1)*n2)');
                %RKR(:,(1:n2)+(k1-1)*n2)=RKR(:,(1:n2)+(k1-1)*n2)*R((1:n2)+(k1-1)*n2,(1:n2)+(k1-1)*n2);
              end
              KR=K*R;
              RKR=R'*KR;
              RKR(1:(size(K,1)+1):end)=RKR(1:(size(K,1)+1):end)+1;
              [L,notpositivedefinite] = chol(RKR,'lower');
              K_nfR=K_nf*R;
              Ltmp=L\K_nfR';
              Covf=kstarstarfull-(Ltmp'*Ltmp);
            else
              g2 = feval(gp.lik.fh.llg2, gp.lik, y, f, 'latent', z);
              g2sq = sqrt(g2);
              ny=sum(y);
              
              KR=bsxfun(@times,K,g2sq')-(K*g2)*g2sq';
              RKR=ny*(bsxfun(@times,g2sq,KR)-g2sq*(g2'*KR));
              RKR(1:(size(K,1)+1):end)=RKR(1:(size(K,1)+1):end)+1;
              [L,notpositivedefinite] = chol(RKR,'lower');
              
              K_nfR=bsxfun(@times,K_nf,g2sq')-(K_nf*g2)*g2sq';
              Ltmp=L\K_nfR';
              Covf=kstarstarfull-ny*(Ltmp'*Ltmp);
            end
          end
        end
        Ef=Eft;
        
      case {'Softmax', 'Multinom'}
        
        if isfield(gp, 'comp_cf')  % own covariance for each ouput component
          multicf = true;
          if length(gp.comp_cf) ~= nout && nout > 1
            error('GPLA_ND_E: the number of component vectors in gp.comp_cf must be the same as number of outputs.')
          end
          if ~isempty(predcf)
            if ~iscell(predcf) || length(predcf)~=nout && nout > 1
              error(['GPLA_ND_PRED: if own covariance for each output component is used,'...
                'predcf has to be cell array and contain nout (vector) elements.   '])
            end
          else
            predcf = gp.comp_cf;
          end
        else
          multicf = false;
          for i1=1:nout
            predcf2{i1} = predcf;
          end
          predcf=predcf2;
        end
        
        ntest=size(xt,1);
        
        % K_nf is 3-D covariance matrix where each slice corresponds to
        % each latent process (output)
        K_nf = zeros(ntest,tn,nout);
        if multicf
          for i1=1:nout
            K_nf(:,:,i1) = gp_cov(gp,xt,x,predcf{i1});
          end
        else
          for i1=1:nout
            K_nf(:,:,i1) = gp_cov(gp,xt,x,predcf{i1});
          end
        end
        
        nout=size(y,2);
        f2=reshape(f,tn,nout);
        
        llg_vec = gp.lik.fh.llg(gp.lik, y, f2, 'latent', z);
        llg = reshape(llg_vec,size(y));
        
        %mu_star = K_nf*reshape(a,tn,nout);
        a=reshape(a,size(y));
        for i1 = 1:nout
          %   Ef(:,i1) = K_nf(:,:,i1)*llg(:,i1);
          Ef(:,i1) = K_nf(:,:,i1)*a(:,i1);
        end
        
        if nargout > 1
          [pi2_vec, pi2_mat] = gp.lik.fh.llg2(gp.lik, y, f2, 'latent', z);
          % W = -diag(pi2_vec) + pi2_mat*pi2_mat', where
          % W_ij = -d^2(log(p(y|f)))/(df_i)(df_j)
          Covf=zeros(nout, nout, ntest);
          
          R=(repmat(1./pi2_vec,1,tn).*pi2_mat);
          for i1=1:nout
            b=E(:,:,i1)*K_nf(:,:,i1)';
            c_cav = R((1:tn)+(i1-1)*tn,:)*(M\(M'\(R((1:tn)+(i1-1)*tn,:)'*b)));
            
            for j1=1:nout
              c=E(:,:,j1)*c_cav;
              Covf(i1,j1,:)=sum(c.*K_nf(:,:,j1)');
            end
            
            kstarstar = gp_trvar(gp,xt,predcf{i1});
            Covf(i1,i1,:) = squeeze(Covf(i1,i1,:)) + kstarstar - sum(b.*K_nf(:,:,i1)')';
          end
        end
        
      otherwise       
        
        ntest=size(xt,1);
        if isfield(gp.lik,'xtime')
          xtime=gp.lik.xtime;
          ntime = size(xtime,1);
          nl=[ntime tn];
          nlt=[ntime ntest];
        else
          nl=repmat(tn,1, length(gp.comp_cf));
          nlt=repmat(ntest, 1, length(gp.comp_cf));
        end
        nlp=length(nl); % number of latent processes
        
        if isfield(gp.lik,'xtime')
          [llg2diag, llg2mat] = feval(gp.lik.fh.llg2, gp.lik, y, f, 'latent', z);
          % W = [diag(Wdiag(1:ntime)) Wmat; Wmat' diag(Wdiag(ntime+1:end))]
          Wdiag=-llg2diag; Wmat=-llg2mat;
        else
          Wvec=-gp.lik.fh.llg2(gp.lik, y, f, 'latent', z);
          % W = [diag(Wvec(1:n,1)) diag(Wvec(1:n,2)); diag(Wvec(n+1:end,1)) diag(Wvec(n+1:end,2))]
          Wdiag=[Wvec(1:nl(1),1); Wvec(nl(1)+(1:nl(2)),2)];
        end
        
        % K_nf is K(x,xt) covariance matrix where blocks correspond to
        % latent processes
        K_nf = zeros(sum(nlt),sum(nl));
        if isempty(predcf)
          K_nf((1:nlt(2))+nlt(1),(1:nl(2))+nl(1)) = gp_cov(gp,xt,x, gp.comp_cf{2});
          if isfield(gp.lik, 'xtime')
            K_nf(1:nlt(1),1:nl(1)) = gp_cov(gp,xtime, xtime, gp.comp_cf{1});
          else
            K_nf(1:nlt(1),1:nl(1)) = gp_cov(gp,xt,x, gp.comp_cf{1});
          end
        else
          K_nf((1:nlt(2))+nlt(1),(1:nl(2))+nl(1)) = gp_cov(gp,xt,x, intersect(gp.comp_cf{2}, predcf));
          if isfield(gp.lik, 'xtime')
            K_nf(1:nlt(1),1:nl(1)) = gp_cov(gp,xtime, xtime, intersect(gp.comp_cf{1}, predcf));
          else
            K_nf(1:nlt(1),1:nl(1)) = gp_cov(gp,xt,x, intersect(gp.comp_cf{1}, predcf));
          end
        end
        
        if isfield(gp,'meanf')
          [H,b_m,B_m Hs]=mean_prep(gp,x,xt);
          if ~(isfield(gp.latent_opt, 'kron') && gp.latent_opt.kron==1)
            K_nf=K_nf + Hs'*B_m*H;
            K = gp_trcov(gp, x);
            K = K+H'*B_m*H;
          end
        end
        
        deriv = feval(gp.lik.fh.llg, gp.lik, y, f, 'latent', z);
        Eft = K_nf*deriv;
        if isfield(gp,'meanf')
          Eft=Eft + K_nf*(K\H'*b_m);
          %Eft=Eft + K_nf*(K\Hs'*b_m);
        end
        
        if nargout > 1
          
          % Evaluate the variance
          
          % Kss is K(X*,X*) covariance matrix between test points, where
          % each block corresponds to latent processes
          Kss = zeros(sum(nlt));
          if isempty(predcf)
            Kss((1:nlt(2))+nlt(1),(1:nlt(2))+nlt(1)) = gp_trcov(gp,xt,gp.comp_cf{2});
            if isfield(gp.lik,'xtime')
              Kss(1:nlt(1),1:nlt(1)) = gp_trcov(gp,xtime,gp.comp_cf{1});
            else
              Kss(1:nlt(1),1:nlt(1)) = gp_trcov(gp,xt,gp.comp_cf{1});
            end
          else
            Kss((1:nlt(2))+nlt(1),(1:nlt(2))+nlt(1)) = gp_trcov(gp,xt,intersect(gp.comp_cf{2}, predcf));
            if isfield(gp.lik,'xtime')
              Kss(1:nlt(1),1:nlt(1)) = gp_trcov(gp,xtime,intersect(gp.comp_cf{1}, predcf));
            else
              Kss(1:nlt(1),1:nlt(1)) = gp_trcov(gp,xt,intersect(gp.comp_cf{1}, predcf));
            end
          end
          
          if isfield(gp,'meanf')
            Kss = Kss + Hs'*B_m*Hs;
          end
                    
          % iB = inv(I + W*K)
          iB=L\eye(sum(nl));
          
          iBW11=bsxfun(@times, iB(1:nl(1),1:nl(1)),Wdiag(1:nl(1))');
          iBW12=bsxfun(@times, iB(1:nl(1),nl(1)+(1:nl(2))), Wdiag(nl(1)+(1:nl(2)))');
          iBW22=bsxfun(@times, iB(nl(1)+(1:nl(2)),nl(1)+(1:nl(2))),Wdiag(nl(1)+(1:nl(2)))');
          if isfield(gp.lik,'xtime')
            iBW11=iBW11 + iB(1:nl(1),nl(1)+(1:nl(2)))*Wmat';
            iBW12=iBW12 + iB(1:nl(1),1:nl(1))*Wmat;
            iBW22=iBW22 + iB(nl(1)+(1:nl(2)),1:nl(1))*Wmat;
          else
            iBW11=iBW11 + bsxfun(@times,iB(1:nl(1),nl(1)+(1:nl(2))),Wvec(nl(1)+(1:nl(2)),1)');
            iBW12=iBW12 + bsxfun(@times,iB(1:nl(1),1:nl(1)),Wvec(1:nl(1),2)');
            iBW22=iBW22 + bsxfun(@times,iB(nl(1)+(1:nl(2)),1:nl(1)),Wvec(1:nl(1),2)');
          end
          iBW=[iBW11 iBW12; iBW12' iBW22];
          
          KiBWK=K_nf*iBW*K_nf';
          
          % Covf = K(X*,X*) - K(X,X*)*inv(I+WK)*W*K(X*,X)
          Covf=Kss-KiBWK;
        end
        Ef=Eft;
        
    end
end

% ============================================================
% Evaluate also the predictive mean and variance of new observation(s)
% ============================================================
if nargout > 2 && isempty(yt)
  error('yt has to be provided to get lpyt.')
end
if nargout > 3
  [lpyt, Ey, Vary] = gp.lik.fh.predy(gp.lik, Ef, Covf, yt, zt);
elseif nargout > 2
  lpyt = gp.lik.fh.predy(gp.lik, Ef, Covf, yt, zt);
end



end

