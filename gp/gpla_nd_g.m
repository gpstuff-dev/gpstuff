function [g, gdata, gprior] = gpla_nd_g(w, gp, x, y, varargin)
%GPLA_SOFTMAX_G   Evaluate gradient of Laplace approximation's marginal
%         log posterior estimate for softmax likelihood (GPLA_SOFTMAX_E)
%
%  Description
%    G = GPLA_SOFTMAX_G(W, GP, X, Y, OPTIONS) takes a full GP
%    hyper-parameter vector W, structure GP a matrix X of
%    input vectors and a matrix Y of target vectors, and evaluates
%    the gradient G of EP's marginal log posterior estimate . Each
%    row of X corresponds to one input vector and each row of Y
%    corresponds to one target vector.
%
%    [G, GDATA, GPRIOR] = GPLA_SOFTMAX_G(W, GP, X, Y, OPTIONS) also
%    returns the data and prior contributions to the gradient.
%
%    OPTIONS is optional parameter-value pair
%      z - optional observed quantity in triplet (x_i,y_i,z_i)
%          Some likelihoods may use this. For example, in case of
%          Poisson likelihood we have z_i=E_i, that is, expected
%          value for ith case.
%
%  See also
%    GPLA_SOFTMAX_E, GPLA_E, GPLA_SOFTMAX_PRED

% Copyright (c) 2010 Jaakko Riihim�ki, Pasi Jyl�nki

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.

ip=inputParser;
ip.FunctionName = 'GPLA_ND_G';
ip.addRequired('w', @(x) isvector(x) && isreal(x) && all(isfinite(x)));
ip.addRequired('gp',@isstruct);
ip.addRequired('x', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addRequired('y', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addParamValue('z', [], @(x) isreal(x) && all(isfinite(x(:))))
ip.parse(w, gp, x, y, varargin{:});
z=ip.Results.z;

gp = gp_unpak(gp, w);       % unpak the parameters
ncf = length(gp.cf);
n=size(x,1);
nout=size(y,2);

g = [];
gdata = [];
gprior = [];

% First Evaluate the data contribution to the error
switch gp.type
  % ============================================================
  % FULL
  % ============================================================
  case 'FULL'   % A full GP
    
    if isfield(gp, 'comp_cf')  % own covariance for each ouput component
      multicf = true;
      if length(gp.comp_cf) ~= nout && nout > 1
        error('GPLA_ND_G: the number of component vectors in gp.comp_cf must be the same as number of outputs.')
      end
    else
      multicf = false;
    end
    
    if isfield(gp.lik, 'structW') && ~gp.lik.structW
      
      if isfield(gp.lik, 'fullW') && gp.lik.fullW
        nl=n;
      else
        if isfield(gp.lik,'xtime')
          xtime=gp.lik.xtime;
          ntime = size(xtime,1);
          nl=[ntime n];
        else
          nl=[n n];
        end
      end
      nlp=length(nl); % number of latent processes
      
      K = zeros(sum(nl));

      if isfield(gp.lik, 'fullW') && gp.lik.fullW
        K = gp_trcov(gp,x);
      else
        if isfield(gp.lik,'xtime')
          K(1:ntime,1:ntime)=gp_trcov(gp, xtime, gp.comp_cf{1});
          K((1:n)+ntime,(1:n)+ntime) = gp_trcov(gp, x, gp.comp_cf{2});
        else
          for i1=1:nlp
            K((1:n)+(i1-1)*n,(1:n)+(i1-1)*n) = gp_trcov(gp, x, gp.comp_cf{i1});
          end
        end
      end
      
      if isfield(gp,'meanf')
        [H,b_m,B_m]=mean_prep(gp,x,[]);
        K=K+H'*B_m*H;
        Hb_m=H'*b_m;
        iKHb_m=K\Hb_m;
       % [H,b_m,B_m]=mean_prep(gp,x,[]);
       % K=K+H'*B_m*H;
      end
      
      [e, edata, eprior, f, L, a, W, p] = gpla_nd_e(gp_pak(gp), gp, x, y, 'z', z);
      if isnan(e)
        return
      end
      
      if isfield(gp.lik, 'fullW') && gp.lik.fullW
        [g2d,g2u] = feval(gp.lik.fh.llg2, gp.lik, y, f, 'latent', z);
        Wd=-g2d; Wu=g2u;
        W=-(g2u*g2u'+diag(g2d));
        g3=feval(gp.lik.fh.llg3, gp.lik, y, f, 'latent', z);
        %g3i1 = n*(-diag(g3d(:,i1)) + bsxfun(@times,g3,g3d(:,i1)') + bsxfun(@times,g3d(:,i1),g3'));
        
        KW=-(K*g2u)*g2u'- bsxfun(@times, K, g2d');
        KW(1:(n+1):end)=KW(1:(n+1):end)+1;
        iKW=KW\eye(n);
        A=iKW*K;
        
        ny=sum(y);
        const1=( 0.5*ny*(sum(A(1:(n+1):end).*g3'))-ny*sum(sum(A.*(g3*g3'))) );
        const2=sum(bsxfun(@times,A,g3));
        s2=const1.*g3 - 0.5*ny*diag(A).*g3 + ny*const2'.*g3;
      else
        KW=zeros(sum(nl));
        if isfield(gp.lik,'xtime')
          [Wdiag, Wmat] = feval(gp.lik.fh.llg2, gp.lik, y, f, 'latent', z);
          Wdiag=-Wdiag; Wmat=-Wmat;
          KW=zeros(ntime+n);
          KW(1:ntime,1:ntime)=bsxfun(@times, K(1:ntime,1:ntime), Wdiag(1:ntime)');
          KW(1:ntime,ntime+(1:n))=K(1:ntime,1:ntime)*Wmat;
          KW(ntime+(1:n),1:ntime)=K(ntime+(1:n),ntime+(1:n))*Wmat';
          KW(ntime+(1:n),ntime+(1:n))=bsxfun(@times,K(ntime+(1:n),ntime+(1:n)), Wdiag(ntime+(1:n))');
          KW(1:(ntime+n+1):end)=KW(1:(ntime+n+1):end)+1;
        else
          for il=1:nlp
            KW((1:n)+(il-1)*n,1:n)=bsxfun(@times, K((1:n)+(il-1)*n,(1:n)+(il-1)*n), W(1:n,il)');
            KW((1:n)+(il-1)*n,(n+1):(2*n))=bsxfun(@times, K((1:n)+(il-1)*n,(1:n)+(il-1)*n), W((n+1):(2*n),il)');
          end
          KW(1:(2*n+1):end)=KW(1:(2*n+1):end)+1;
        end
        
        iKW=KW\eye(sum(nl));
        A=iKW*K;
        s2=zeros(sum(nl),1);
        
        if isfield(gp.lik,'xtime')
          A_diag=diag(A);
          A_mat=A(1:ntime,ntime+(1:n));
          for i1=1:sum(nl)
            [dw_diag,dw_mat]=feval(gp.lik.fh.llg3, gp.lik, y, f, 'latent', z, i1);
            s2(i1) = 0.5*(sum(A_diag.*dw_diag)+2*sum(sum(A_mat.*dw_mat)));
          end
        else
          dw_mat = feval(gp.lik.fh.llg3, gp.lik, y, f, 'latent', z);
          for i1=1:n
            s2(i1) = 0.5*trace(A(i1:n:(i1+n),i1:n:(i1+n))*dw_mat(:,:,1,i1));
            s2(i1+n) = 0.5*trace(A(i1:n:(i1+n),i1:n:(i1+n))*dw_mat(:,:,2,i1));
          end
        end
      end
      
      % =================================================================
      % Gradient with respect to covariance function parameters
      if ~isempty(strfind(gp.infer_params, 'covariance'))
        % Evaluate the gradients from covariance functions
        for i=1:ncf
          
          i1=0;
          if ~isempty(gprior)
            i1 = length(gprior);
          end
          
          gpcf = gp.cf{i};
          if isfield(gp.lik, 'fullW') && gp.lik.fullW
            DKff = feval(gpcf.fh.cfg, gpcf, x);
          else
            % check in which components the covariance function is present
            do = false(nlp,1);
            for z1=1:nlp
              if any(gp.comp_cf{z1}==i)
                do(z1) = true;
              end
            end
            
            if isfield(gp.lik,'xtime')
              if ~isempty(intersect(gp.comp_cf{1},i))
                DKff = feval(gpcf.fh.cfg, gpcf, xtime);
              else
                DKff = feval(gpcf.fh.cfg, gpcf, x);
              end
            else
              DKff = feval(gpcf.fh.cfg, gpcf, x);
            end
          end
          gprior_cf = -feval(gpcf.fh.lpg, gpcf);
          g1 = feval(gp.lik.fh.llg, gp.lik, y, f, 'latent', z);
          
          
          if isfield(gp.lik, 'fullW') && gp.lik.fullW
            for i2 = 1:length(DKff)
              i1 = i1+1;
              if ~isfield(gp,'meanf')
                %s1 = 0.5 * a'*DKff{i2}*a - 0.5*trace(iKWDKff*W);
                %s1 = 0.5 * a'*DKff{i2}*a - 0.5*trace(iKWDKff*(-g2u*g2u'-diag(g2d)));
                %s1 = 0.5 * a'*DKff{i2}*a - 0.5*((-iKWDKff*g2u)'*g2u) + 0.5*sum(iKWDKff(1:(n+1):end).*g2d');
                %s1 = 0.5 * a'*DKff{i2}*a - 0.5*((-iKW*(DKff{i2}*g2u))'*g2u) + 0.5*sum(iKWDKff(1:(n+1):end).*g2d');
                s1 = 0.5 * a'*DKff{i2}*a - 0.5*((-iKW*(DKff{i2}*g2u))'*g2u) + 0.5*sum(sum(iKW'.*DKff{i2}).*g2d');
                %s1 = 0.5 * a'*DKff{i2}*a - 0.5*trace(inv(eye(n)+K*W)*DKff{i2}*W);
                %s1 = 0.5 * a'*DKff{i2}*a - 0.5*sum(sum(R.*DKff{i2}));
              else
                %s1 = 0.5 * (a-K\(H'*b_m))'*DKff{i2}*(a-K\(H'*b_m)) - 0.5*sum(sum(R.*DKff{i2}));
                s1 = 0.5 * (a-iKHb_m)'*DKff{i2}*(a-iKHb_m) - 0.5*((-iKW*(DKff{i2}*g2u))'*g2u) + 0.5*sum(sum(iKW'.*DKff{i2}).*g2d');
              end
              %b = DKff{i2} * g1;
              if issparse(K)
                s3 = b - K*(sqrtW*ldlsolve(L,sqrtW*b));
              else
                
                %s3=iKWDKff*g1;
                s3=iKW*(DKff{i2}*g1);
                
                %s3=inv(eye(n)+K*W)*DKff{i2} * g1;
                %              s3 = b - K*(R*b);
                %              b = DKff{i2} * g1;
                %s3 = (1./W).*(R*b);
              end
              gdata(i1) = -(s1 + s2'*s3);
              gprior(i1) = gprior_cf(i2);
            end
          else
            %IKW=(eye(2*n)+K*W);
            %WK=(eye(2*n)+K*W)\W;
            %WK=W/IKW;
            
            %WiKW=[diag(W(1:n,1)) diag(W((n+1):(2*n),1)); diag(W(1:n,2)) diag(W((n+1):(2*n),2))]*iKW;
            if isfield(gp.lik,'xtime')
              WiKW=zeros(sum(nl));
              WiKW(1:ntime,1:ntime)=bsxfun(@times, Wdiag(1:ntime),iKW(1:ntime,1:ntime)) + Wmat*iKW(ntime+(1:n),1:ntime);
              WiKW(1:ntime,ntime+(1:n))=bsxfun(@times, Wdiag(1:ntime),iKW(1:ntime,ntime+(1:n))) + Wmat*iKW(ntime+(1:n),ntime+(1:n));
              WiKW(ntime+(1:n),1:ntime)=Wmat'*iKW(1:ntime,1:ntime) + bsxfun(@times, Wdiag(ntime+(1:n)),iKW(ntime+(1:n),1:ntime));
              WiKW(ntime+(1:n),ntime+(1:n))=bsxfun(@times, Wdiag(ntime+(1:n)),iKW(ntime+(1:n),ntime+(1:n))) + Wmat'*iKW(1:ntime,ntime+(1:n));
            else
              WiKW11=bsxfun(@times,W(1:n,1),iKW(1:n,1:n))+bsxfun(@times,W(1:n,2),iKW((1:n)+n,1:n));
              WiKW12=bsxfun(@times,W(1:n,1),iKW(1:n,(1:n)+n))+bsxfun(@times,W(1:n,2),iKW((1:n)+n,(1:n)+n));
              WiKW21=bsxfun(@times,W((1:n)+n,1),iKW(1:n,1:n))+bsxfun(@times,W((1:n)+n,2),iKW((1:n)+n,1:n));
              WiKW22=bsxfun(@times,W((1:n)+n,1),iKW(1:n,(1:n)+n))+bsxfun(@times,W((1:n)+n,2),iKW((1:n)+n,(1:n)+n));
              WiKW=[WiKW11 WiKW12; WiKW21 WiKW22];
            end
            %WK=W*inv(eye(2*n)+K*W);
            %WK=W-W*inv(eye(2*n)+K*W)*K*W;
            
            for i2 = 1:length(DKff)
              i1 = i1+1;
              if ~isfield(gp,'meanf')
                
                %s1 = 0.5 * a'*DKff{i2}*a - 0.5*sum(sum(R.*DKff{i2}));
                dKnl = zeros(sum(nl));
                
                if isfield(gp.lik,'xtime')
                  if ~isempty(intersect(gp.comp_cf{1},i)) %do(indnl)
                    dKnl(1:ntime,1:ntime) = DKff{i2};
                    %end
                  else
                    %if do(indnl)
                    dKnl(ntime+(1:n),ntime+(1:n)) = DKff{i2};
                    %end
                  end
                else
                  for indnl=1:nlp
                    if do(indnl)
                      dKnl((1:n)+(indnl-1)*n,(1:n)+(indnl-1)*n) = DKff{i2};
                    end
                  end
                end
                
                s1 = 0.5 * a'*dKnl*a - 0.5*sum(sum((dKnl.*WiKW)));
                %s1 = 0.5 * a'*dKnl*a - 0.5*trace((eye(2*n)+K*W)\dKnl*W);
                %s1 = 0.5 * a'*dKnl*a - 0.5*trace(WK*dKnl);
              else
                %s1 = 0.5 * (a-K\(H'*b_m))'*DKff{i2}*(a-K\(H'*b_m)) - 0.5*sum(sum(R.*DKff{i2}));
              end
              b = dKnl*g1;
              %b = DKff{i2} * g1;
              if issparse(K)
                s3 = b - K*(sqrtW*ldlsolve(L,sqrtW*b));
              else
                %s3 = KW\b;
                s3=iKW*b;
                %s3 = b - K*(R*b);
                %b = DKff{i2} * g1;
                
                %s3 = (1./W).*(R*b);
              end
              
              gdata(i1) = -(s1 + s2'*s3);
              %gdata(i1) = -(s1);
              gprior(i1) = gprior_cf(i2);
            end
          end
          % Set the gradients of hyperparameter
          if length(gprior_cf) > length(DKff)
            for i2=length(DKff)+1:length(gprior_cf)
              i1 = i1+1;
              gdata(i1) = 0;
              gprior(i1) = gprior_cf(i2);
            end
          end
        end
      end
      
      % =================================================================
      % Gradient with respect to likelihood function parameters
      if ~isempty(strfind(gp.infer_params, 'likelihood')) ...
          && ~isempty(gp.lik.fh.pak(gp.lik))
        
        gdata_lik = 0;
        lik = gp.lik;
        
        g_logPrior = -feval(lik.fh.lpg, lik);
        if ~isempty(g_logPrior)
          
          DW_sigma = feval(lik.fh.llg3, lik, y, f, 'latent2+param', z);
          DL_sigma = feval(lik.fh.llg, lik, y, f, 'param', z);
          b = K * feval(lik.fh.llg2, lik, y, f, 'latent+param', z);
          %s3 = b - K*(R*b);
          %s3 = KW\b;
          s3 = iKW*b;
          %nl= size(DW_sigma,2);
          
          gdata_lik = - DL_sigma - 0.5.*sum(sum((A.*DW_sigma))) - s2'*s3;
          %gdata_lik = - DL_sigma - 0.5.*trace(A*DW_sigma) - s2'*s3;
          %gdata_lik = - DL_sigma - 0.5.*trace((eye(2*n)+K*W)\K*DW_sigma) - s2'*s3;
          %gdata_lik = - DL_sigma - 0.5.*sum(repmat(C2,1,nl).*DW_sigma) - s2'*s3;
          
          % set the gradients into vectors that will be returned
          gdata = [gdata gdata_lik];
          gprior = [gprior g_logPrior];
          i1 = length(g_logPrior);
          i2 = length(gdata_lik);
          if i1  > i2
            gdata = [gdata zeros(1,i1-i2)];
          end
        end
      end
      
      % g = gdata + gprior;
      
    else
      
      K = zeros(n,n,nout);
      if multicf
        for i1=1:nout
          K(:,:,i1) = gp_trcov(gp, x, gp.comp_cf{i1});
        end
      else
        Ktmp=gp_trcov(gp, x);
        for i1=1:nout
          K(:,:,i1) = Ktmp;
        end
      end
      
      [e, edata, eprior, f, L, a, E, M, p] = gpla_nd_e(gp_pak(gp), gp, x, y, 'z', z);
      
      % softmax
      f2=reshape(f,n,nout);
      
      llg = gp.lik.fh.llg(gp.lik, y, f2, 'latent', z);
      [pi2_vec, pi2_mat] = gp.lik.fh.llg2(gp.lik, y, f2, 'latent', z);
      R = repmat(1./pi2_vec,1,n).*pi2_mat;
      RE = zeros(n,n*nout);
      for i1=1:nout
        RE(:,(1:n)+(i1-1)*n) = R((1:n)+(i1-1)*n,:)'*E(:,:,i1);
      end
      
      inv_iWK=zeros(n,n,nout);
      
      % Matrices for computing the derivative of determinant term w.r.t. f
      A=zeros(nout, nout, n);
      Minv=M\(M'\eye(n));
      Minv=(Minv+Minv')./2;
      for cc1=1:nout
        EMinv=RE(:,(1:n)+(cc1-1)*n)'*Minv;
        KEMinv=K(:,:,cc1)*EMinv;
        for cc2=1:nout
          if cc2>=cc1
            if cc1==cc2
              EMtmp = - EMinv*RE(:,(1:n)+(cc2-1)*n);
              EMtmp = EMtmp + E(:,:,cc1);
              inv_iWK(:,:,cc1) = EMtmp;
              A(cc1,cc1,:) = diag(K(:,:,cc1))-sum((K(:,:,cc1)*EMtmp).*K(:,:,cc1),2);
            else
              EMtmp = - KEMinv*RE(:,(1:n)+(cc2-1)*n);
              A(cc1,cc2,:) = -sum(EMtmp.*K(:,:,cc2),2);
              A(cc2,cc1,:) = -sum(EMtmp.*K(:,:,cc2),2);
            end
          end
        end
      end
      
      % Derivative of determinant term w.r.t. f
      s2=zeros(n*nout,1);
      dw_mat = gp.lik.fh.llg3(gp.lik, y, f, 'latent', z);
      for cc3=1:nout
        for ii1=1:n
          s2(ii1+(cc3-1)*n) = -0.5*trace(A(:,:,ii1)*dw_mat(:,:,cc3,ii1));
        end
      end
      
      % Loop over the covariance functions
      for i=1:ncf
        DKllg=zeros(size(a));
        EDKllg=zeros(size(a));
        DKffba=zeros(n*nout,1);
        
        % check in which components the covariance function is present
        do = false(nout,1);
        if multicf
          for z1=1:nout
            if any(gp.comp_cf{z1}==i)
              do(z1) = true;
            end
          end
        else
          do = true(nout,1);
        end
        
        i1=0;
        if ~isempty(gprior)
          i1 = length(gprior);
        end
        
        % Gradients from covariance functions
        gpcf = gp.cf{i};
        DKff = gpcf.fh.cfg(gpcf, x);
        gprior_cf = -gpcf.fh.lpg(gpcf);
        
        for i2 = 1:length(DKff)
          i1 = i1+1;
          DKffb=DKff{i2};
          
          % Derivative of explicit terms
          trace_sum_tmp=0;
          for z1=1:nout
            if do(z1)
              DKffba((1:n)+(z1-1)*n)=DKffb*a((1:n)+(z1-1)*n);
              trace_sum_tmp = trace_sum_tmp + sum(sum( inv_iWK(:,:,z1) .* DKffb ));
            end
          end
          s1 = 0.5 * a'*DKffba - 0.5.*trace_sum_tmp;
          
          % Derivative of f w.r.t. theta
          for z1=1:nout
            if do(z1)
              DKllg((1:n)+(z1-1)*n)=DKffb*llg((1:n)+(z1-1)*n);
              EDKllg((1:n)+(z1-1)*n)=E(:,:,z1)*DKllg((1:n)+(z1-1)*n);
            end
          end
          s3 = EDKllg - RE'*(M\(M'\(RE*DKllg)));
          for z1=1:nout
            s3((1:n)+(z1-1)*n)=K(:,:,z1)*s3((1:n)+(z1-1)*n);
          end
          s3 = DKllg - s3;
          
          gdata(i1) = -(s1 + s2'*s3);
          gprior(i1) = gprior_cf(i2);
          
        end
        
        % Set the gradients of hyper-hyperparameter
        if length(gprior_cf) > length(DKff)
          for i2=length(DKff)+1:length(gprior_cf)
            i1 = i1+1;
            gdata(i1) = 0;
            gprior(i1) = gprior_cf(i2);
          end
        end
      end
      
    end
    
    %         % =================================================================
    %         % Gradient with respect to likelihood function parameters
    %         if ~isempty(strfind(gp.infer_params, 'likelihood')) ...
    %             && ~isempty(gp.lik.fh.pak(gp.lik))
    %
    %             gdata_likelih = 0;
    %             lik = gp.lik;
    %
    %             g_logPrior = feval(lik.fh.gprior, lik);
    %             if ~isempty(g_logPrior)
    %
    %                 DW_sigma = feval(lik.fh.llg3, lik, y, f, 'latent2+hyper', z);
    %                 DL_sigma = feval(lik.fh.llg, lik, y, f, 'hyper', z);
    %                 b = K * feval(lik.fh.llg2, lik, y, f, 'latent+hyper', z);
    %                 s3 = b - K*(R*b);
    %                 nl= size(DW_sigma,2);
    %
    %                 gdata_lik = - DL_sigma - 0.5.*sum(repmat(C2,1,nl).*DW_sigma) - s2'*s3;
    %
    %                 % set the gradients into vectors that will be returned
    %                 gdata = [gdata gdata_lik];
    %                 gprior = [gprior g_logPrior];
    %                 i1 = length(g_logPrior);
    %                 i2 = length(gdata_lik);
    %                 if i1  > i2
    %                     gdata = [gdata zeros(1,i1-i2)];
    %                 end
    %             end
    %         end
    
    g = gdata + gprior;
    
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
    
end

assert(isreal(gdata))
assert(isreal(gprior))
end
