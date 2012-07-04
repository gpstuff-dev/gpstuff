function [Ef, Covf, lpyt, Ey, Vary] = gpla_nd_pred(gp, x, y, varargin)
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
      
      if isfield(gp.lik, 'structW') && ~gp.lik.structW
        % FULL
        % ============================================================
        [e, edata, eprior, f, L, a, W, p] = gpla_nd_e(gp_pak(gp), gp, x, y, 'z', z);
        if isfield(gp.lik,'xtime')
          [Wdiag, Wmat] = feval(gp.lik.fh.llg2, gp.lik, y, f, 'latent', z);
          Wdiag=-Wdiag; Wmat=-Wmat;
        else
          W=-gp.lik.fh.llg2(gp.lik, y, f, 'latent', z);
        end
        
        ntest=size(xt,1);
        if isfield(gp.lik, 'fullW') && gp.lik.fullW
          nl=tn;
          nlt=ntest;
        else
          if isfield(gp.lik,'xtime')
            xtime=gp.lik.xtime;
            ntime = size(xtime,1);
            nl=[ntime tn];
            nlt=[ntime ntest];
          else
            nl=[tn tn];
            nlt=[ntest ntest];
          end
        end
        nlp=length(nl); % number of latent processes
        
        if isfield(gp.lik, 'fullW') && gp.lik.fullW
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
        else
          K_nf = zeros(sum(nlt),sum(nl));
          if isfield(gp.lik,'xtime')
            if isempty(predcf)
              K_nf(1:ntime,1:ntime) = gp_cov(gp, xtime, xtime, gp.comp_cf{1});
              K_nf((1:ntest)+ntime,(1:tn)+ntime) = gp_cov(gp,xt,x, gp.comp_cf{2});
            else
              K_nf(1:ntime,1:ntime) = gp_cov(gp,xtime,xtime, intersect(gp.comp_cf{1}, predcf));
              K_nf((1:ntest)+ntime,(1:tn)+ntime) = gp_cov(gp,xt,x, intersect(gp.comp_cf{2}, predcf));
            end
          else
            if isempty(predcf)
              for i1=1:nlp
                K_nf((1:ntest)+(i1-1)*ntest,(1:tn)+(i1-1)*tn) = gp_cov(gp,xt,x, gp.comp_cf{i1});
              end
            else
              for i1=1:nlp
                K_nf((1:ntest)+(i1-1)*ntest,(1:tn)+(i1-1)*tn) = gp_cov(gp,xt,x, intersect(gp.comp_cf{i1}, predcf));
              end
            end
          end
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
          if issparse(K_nf) && issparse(L)
            deriv = feval(gp.lik.fh.llg, gp.lik, y(p), f, 'latent', z(p));
            Eft = K_nf(:,p)*deriv;
          else
            deriv = feval(gp.lik.fh.llg, gp.lik, y, f, 'latent', z);
            Eft = K_nf*deriv;
            if isfield(gp,'meanf')
              Eft=Eft + K_nf*(K\H'*b_m);
              %Eft=Eft + K_nf*(K\Hs'*b_m);
            end
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
              if isfield(gp.lik, 'fullW') && gp.lik.fullW
                kstarstarfull = gp_trcov(gp,xt);
              else
                kstarstarfull = zeros(sum(nlt));
                if isfield(gp.lik,'xtime')
                  kstarstarfull(1:ntime,1:ntime) = gp_trcov(gp,xtime,gp.comp_cf{1});
                  kstarstarfull((1:ntest)+ntime,(1:ntest)+ntime) = gp_trcov(gp,xt,gp.comp_cf{2});
                else
                  for i1=1:nlp
                    kstarstarfull((1:ntest)+(i1-1)*ntest,(1:ntest)+(i1-1)*ntest) = gp_trcov(gp,xt,gp.comp_cf{i1});
                  end
                end
              end
            else
              if isfield(gp.lik, 'fullW') && gp.lik.fullW
                kstarstarfull = gp_trcov(gp,xt,predcf);
              else
                kstarstarfull = zeros(sum(nlt));
                if isfield(gp.lik,'xtime')
                  kstarstarfull(1:ntime,1:ntime) = gp_trcov(gp,xtime,intersect(gp.comp_cf{1}, predcf));
                  kstarstarfull((1:ntest)+ntime,(1:ntest)+ntime) = gp_trcov(gp,xt,intersect(gp.comp_cf{2}, predcf));
                else
                  for i1=1:nlp
                    kstarstarfull((1:ntest)+(i1-1)*ntest,(1:ntest)+(i1-1)*ntest) = gp_trcov(gp,xt,intersect(gp.comp_cf{i1}, predcf));
                  end
                end
              end
            end
            
            if isfield(gp,'meanf')
              kstarstarfull = kstarstarfull + Hs'*B_m*Hs;
            end
            %           if W >= 0
            %               % This is the usual case where likelihood is log concave
            %               % for example, Poisson and probit
            %               if issparse(K_nf) && issparse(L)
            %                   % If compact support covariance functions are used
            %                   % the covariance matrix will be sparse
            %                   sqrtW = sqrt(W);
            %                   sqrtWKfn = sqrtW*K_nf(:,p)';
            %                   V = ldlsolve(L,sqrtWKfn);
            %                   Varft = kstarstar - sum(sqrtWKfn.*V,1)';
            %               else
            %                   W = diag(W);
            %                   V = L\(sqrt(W)*K_nf');
            %                   Varft = kstarstar - sum(V'.*V',2);
            %               end
            %           else
            %               % We may end up here if the likelihood is not log concace
            %               % For example Student-t likelihood
            %               V = L*diag(W);
            %               R = diag(W) - V'*V;
            %               Varft = kstarstar - sum(K_nf.*(R*K_nf')',2);
            %           end
            
            
            if isfield(gp.lik, 'fullW') && gp.lik.fullW
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
            else
              n=size(x,1);
              iWK=L\eye(sum(nl));
              %iWKW=iWK*[diag(W(1:n,1)) diag(W((n+1):(2*n),1)); diag(W(1:n,2)) diag(W((n+1):(2*n),2))];
              if isfield(gp.lik,'xtime')
                iWKW=zeros(n+ntime);
                iWKW(1:ntime,1:ntime)=bsxfun(@times, iWK(1:ntime,1:ntime),Wdiag(1:ntime)') + iWK(1:ntime,ntime+(1:n))*Wmat';
                iWKW(1:ntime,ntime+(1:n))=iWK(1:ntime,1:ntime)*Wmat + bsxfun(@times, iWK(1:ntime,ntime+(1:n)), Wdiag(ntime+(1:n))');
                iWKW(ntime+(1:n),1:ntime)=bsxfun(@times,iWK(ntime+(1:n),1:ntime),Wdiag(1:ntime)') + iWK(ntime+(1:n),ntime+(1:n))*Wmat';
                iWKW(ntime+(1:n),ntime+(1:n))=iWK(ntime+(1:n),1:ntime)*Wmat + bsxfun(@times, iWK(ntime+(1:n),ntime+(1:n)),Wdiag(ntime+(1:n))');
              else
                iWKW11=bsxfun(@times,iWK(1:n,1:n),W(1:n,1)')+bsxfun(@times,iWK(1:n,(1:n)+n),W((1:n)+n,1)');
                iWKW12=bsxfun(@times,iWK(1:n,1:n),W(1:n,2)')+bsxfun(@times,iWK(1:n,(1:n)+n),W((1:n)+n,2)');
                iWKW21=bsxfun(@times,iWK((1:n)+n,1:n),W(1:n,1)')+bsxfun(@times,iWK((1:n)+n,(1:n)+n),W((1:n)+n,1)');
                iWKW22=bsxfun(@times,iWK((1:n)+n,1:n),W(1:n,2)')+bsxfun(@times,iWK((1:n)+n,(1:n)+n),W((1:n)+n,2)');
                iWKW=[iWKW11 iWKW12; iWKW21 iWKW22];
              end
              
              KiWKWK=K_nf*iWKW*K_nf';
              Covf=kstarstarfull-KiWKWK;
            end
          end
        end
        Ef=Eft;
        
      else
        
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
        
        [e, edata, eprior, f, L, a, E, M, p] = gpla_nd_e(gp_pak(gp), gp, x, y, 'z', z);
        
        ntest=size(xt,1);
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
      end
      
      % ============================================================
      % FIC
      % ============================================================
    case 'FIC'        % Predictions with FIC sparse approximation for GP
      % ============================================================
      % PIC
      % ============================================================
    case {'PIC' 'PIC_BLOCK'}        % Predictions with PIC sparse approximation for GP
      % ============================================================
      % CS+FIC
      % ============================================================
    case 'CS+FIC'        % Predictions with CS+FIC sparse approximation for GP
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