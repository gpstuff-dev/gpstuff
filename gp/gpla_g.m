function [g, gdata, gprior] = gpla_g(w, gp, x, y, varargin)
%GPLA_G   Evaluate gradient of Laplace approximation's marginal 
%         log posterior estimate (GPLA_E)
%
%     Description
%	G = GPLA_G(W, GP, X, Y, OPTIONS) takes a full GP hyper-parameter
%        vector W, data structure GP a matrix X of input vectors
%        and a matrix Y of target vectors, and evaluates the
%        gradient G of EP's marginal log posterior estimate . Each
%        row of X corresponds to one input vector and each row of Y
%        corresponds to one target vector.
%
%	[G, GDATA, GPRIOR] = GPLA_G(W, GP, X, Y, OPTIONS) also returns 
%        the data and prior contributions to the gradient.
%
%     OPTIONS is optional parameter-value pair
%       'z'    is optional observed quantity in triplet (x_i,y_i,z_i)
%              Some likelihoods may use this. For example, in case of 
%              Poisson likelihood we have z_i=E_i, that is, expected 
%              value for ith case. 
%  
%	See also   
%       GPLA_E, LA_PRED

% Copyright (c) 2007-2010 Jarno Vanhatalo
% Copyright (c) 2010 Aki Vehtari

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

  ip=inputParser;
  ip.FunctionName = 'GPLA_G';
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

    g = [];
    gdata = [];
    gprior = [];

    % First Evaluate the data contribution to the error
    switch gp.type
        % ============================================================
        % FULL
        % ============================================================
      case 'FULL'   % A full GP
                    % Calculate covariance matrix and the site parameters
        K = gp_trcov(gp,x);
        
        [e, edata, eprior, f, L, a, W, p] = gpla_e(gp_pak(gp), gp, x, y, 'z', z);
        
        if W >= 0              % This is the usual case where likelihood is log concave
                               % for example, Poisson and probit
            if issparse(K)                               % use sparse matrix routines

                % permute
                y = y(p);
                x = x(p,:);
                K = K(p,p);
                if ~isempty(z)
                    z = z(p,:);
                end

                sqrtW = sqrt(W);
                
                R = sqrtW*spinv(L,1)*sqrtW;
                sqrtWK = sqrtW*K;
                C = ldlsolve(L,sqrtWK);
                C2 = diag(K) - sum(sqrtWK.*C,1)';
                s2 = 0.5*C2.*feval(gp.likelih.fh_g3, gp.likelih, y, f, 'latent', z);
            else                                         % evaluate with full matrices
                sqrtW = diag(sqrt(W));
                R = sqrtW*(L'\(L\sqrtW));
                C2 = diag(K) - sum((L\(sqrtW*K)).^2,1)' ;
                s2 = 0.5*C2.*feval(gp.likelih.fh_g3, gp.likelih, y, f, 'latent', z);
            end
        else                         % We might end up here if the likelihood is not log concace
                                     % For example Student-t likelihood. 
            C = L;
            V = L*diag(W);
            R = diag(W) - V'*V;
            C2 = sum(C.^2,1)';
            s2 = 0.5*C2.*feval(gp.likelih.fh_g3, gp.likelih, y, f, 'latent', z);
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
                [DKff, gprior_cf] = feval(gpcf.fh_ghyper, gpcf, x);
                g1 = feval(gp.likelih.fh_g, gp.likelih, y, f, 'latent', z);
                for i2 = 1:length(DKff)
                    i1 = i1+1;
                    
                    s1 = 0.5 * a'*DKff{i2}*a - 0.5*sum(sum(R.*DKff{i2}));
                    b = DKff{i2} * g1;
                    if issparse(K)
                        s3 = b - K*(sqrtW*ldlsolve(L,sqrtW*b));
                    else
                        s3 = b - K*(R*b);
                        b = DKff{i2} * g1;
                        %s3 = (1./W).*(R*b);
                    end
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
            
            % Evaluate the gradient from noise functions
            if isfield(gp, 'noise')
                nn = length(gp.noise);
                for i=1:nn
                    i1 = i1+1;
                    
                    noise = gp.noise{i};
                    [DCff, gprior_cf] = feval(noise.fh_ghyper, noise, x);
                    
                    for i2 = 1:length(DCff)
                        i1 = i1+1;
                        
                        Bdm = b'*(DKff{i2}*b);
                        Cdm = sum(sum(invC.*DCff{i2})); 
                        gdata(i1) = 0.5.*(Cdm - Bdm);
                        gprior(i1) = gprior_cf(i2);
                    end
                    
                    % Set the gradients of hyper-hyperparameter
                    if length(gprior_cf) > length(DCff)
                        for i2=length(DCff)+1:length(gprior_cf)
                            i1 = i1+1;
                            gdata(i1) = 0;
                            gprior(i1) = gprior_cf(i2);
                        end
                    end
                    
                end
            end
        end
        
        % =================================================================
        % Gradient with respect to likelihood function parameters
        if ~isempty(strfind(gp.infer_params, 'likelihood')) ...
            && ~isempty(gp.likelih.fh_pak(gp.likelih))
            
            gdata_likelih = 0;
            likelih = gp.likelih;
            
            g_logPrior = feval(likelih.fh_priorg, likelih);
            if ~isempty(g_logPrior)
            
                DW_sigma = feval(likelih.fh_g3, likelih, y, f, 'latent2+hyper', z);
                DL_sigma = feval(likelih.fh_g, likelih, y, f, 'hyper', z);
                b = K * feval(likelih.fh_g2, likelih, y, f, 'latent+hyper', z);
                s3 = b - K*(R*b);
                nl= size(DW_sigma,2);
                
                gdata_likelih = - DL_sigma - 0.5.*sum(repmat(C2,1,nl).*DW_sigma) - s2'*s3;

                % set the gradients into vectors that will be returned
                gdata = [gdata gdata_likelih];
                gprior = [gprior g_logPrior];
                i1 = length(g_logPrior);
                i2 = length(gdata_likelih);
                if i1  > i2
                    gdata = [gdata zeros(1,i1-i2)];
                end
            end
        end
        
        g = gdata + gprior;

        % ============================================================
        % FIC
        % ============================================================
      case 'FIC'
        g_ind = zeros(1,numel(gp.X_u));
        gdata_ind = zeros(1,numel(gp.X_u));
        gprior_ind = zeros(1,numel(gp.X_u));

        u = gp.X_u;
        m = size(u,1);

        [e, edata, eprior, f, L, a, La1] = gpla_e(gp_pak(gp), gp, x, y, 'z', z);

        K_fu = gp_cov(gp, x, u);         % f x u
        K_uu = gp_trcov(gp, u);          % u x u, noiseles covariance K_uu
        Luu = chol(K_uu);
        B=Luu'\(K_fu');       % u x f
        iKuuKuf = Luu\B;
        
        W = -feval(gp.likelih.fh_g2, gp.likelih, y, f, 'latent', z);
        sqrtW = sqrt(W);
        
        % Components for trace( inv(inv(W) + K) * dK) )
        Lah = 1 + sqrtW.*La1.*sqrtW;
        sWKfu = (repmat(sqrtW,1,m).*K_fu);
        B3 = repmat(Lah,1,m).\sWKfu;
        A = K_uu + sWKfu'*B3; A=(A+A')/2;
        L2 = repmat(sqrtW,1,m).*(B3/chol(A));
        iLa2W = sqrtW.*(Lah.\sqrtW);
                
        LL = sum(L2.*L2,2);
        BB = sum(B.^2)';
        
        % Evaluate s2
        C1 = L2'*B'*B;
        C2 = L2'.*repmat(La1',m,1);
        
        s2t = La1 + BB;
        s2t = s2t - (La1.*iLa2W.*La1 - sum(C2.^2)' + sum(B'.*((B*(repmat(iLa2W,1,m).*B'))*B)',2)...
                    - sum(C1.^2)' + 2*La1.*iLa2W.*BB - 2*La1.*sum(L2.*C1',2));

        s2 = 0.5*s2t.*feval(gp.likelih.fh_g3, gp.likelih, y, f, 'latent', z);
        b3 = feval(gp.likelih.fh_g, gp.likelih, y, f, 'latent', z);
        
        
        % =================================================================
        % Gradient with respect to covariance function parameters
        if ~isempty(strfind(gp.infer_params, 'covariance'))
            for i=1:ncf            
                i1=0;
                if ~isempty(gprior)
                    i1 = length(gprior);
                end
                
                % Get the gradients of the covariance matrices 
                % and gprior from gpcf_* structures
                gpcf = gp.cf{i};
                [DKff, gprior_cf] = feval(gpcf.fh_ghyper, gpcf, x, [], 1); 
                DKuu = feval(gpcf.fh_ghyper, gpcf, u); 
                DKuf = feval(gpcf.fh_ghyper, gpcf, u, x);
                
                for i2 = 1:length(DKuu)
                    i1 = i1+1;
                    
                    % 0.5* a'*dK*a, where a = K\f
                    KfuiKuuKuu = iKuuKuf'*DKuu{i2};
                    gdata(i1) = -0.5.*((2.*a'*DKuf{i2}'-(a'*KfuiKuuKuu))*(iKuuKuf*a) + (a'.*DKff{i2}')*a...
                                      - (2.*a'.*sum(DKuf{i2}'.*iKuuKuf',2)'*a-a'.*sum(KfuiKuuKuu.*iKuuKuf',2)'*a) );
                    
                    % trace( inv(inv(W) + K) * dQ) )
                    gdata(i1) = gdata(i1) - 0.5.*sum(sum(L2'.*(2.*L2'*DKuf{i2}'*iKuuKuf - L2'*KfuiKuuKuu*iKuuKuf)));
                    gdata(i1) = gdata(i1) + 0.5.*sum(DKff{i2}.*iLa2W - LL.*DKff{i2});
                    gdata(i1) = gdata(i1) + 0.5.*(2.*sum(LL.*sum(DKuf{i2}'.*iKuuKuf',2)) - sum(LL.*sum(KfuiKuuKuu.*iKuuKuf',2)));
                    
                    b = (2*DKuf{i2}' - KfuiKuuKuu)*(iKuuKuf*b3) + DKff{i2}.*b3 - sum((2.*DKuf{i2}'- KfuiKuuKuu).*iKuuKuf',2).*b3;
                    bb = sqrtW.*(Lah.\(sqrtW.*b)) - L2*(L2'*b);
                    s3 = b - (La1.*bb + B'*(B*bb));
                    gdata(i1) = gdata(i1) - s2'*s3;
                    
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
            
            % =================================================================
            % Gradient with respect to noise function parameters
            if isfield(gp, 'noise')
                nn = length(gp.noise);
                for i=1:nn
                    i1 = i1+1;
                    
                    gpcf = gp.noise{i};
                    
                    [DCff, gprior_cf] = feval(gpcf.fh_ghyper, gpcf, x);
                    for i2 = 1:length(DCff)
                        gdata(i1)= -0.5*DCff.*a'*a;
                        gdata(i1)= gdata(i1) + 0.5*sum((1./La-LL).*DCff{i2});
                        gprior(i1) = gprior_cf(i2);
                    end
                end
            end            
        end
        
        % =================================================================
        % Gradient with respect to inducing inputs
        
        if ~isempty(strfind(gp.infer_params, 'inducing'))
            if isfield(gp.p, 'X_u') && ~isempty(gp.p.X_u)
                m = size(gp.X_u,2);
                st=0;
                if ~isempty(gprior)
                    st = length(gprior);
                end
                
                gdata(st+1:st+length(gp.X_u(:))) = 0;
                i1 = st+1;
                for i = 1:size(gp.X_u,1)
                    if iscell(gp.p.X_u) % Own prior for each inducing input
                        pr = gp.p.X_u{i};
                        gprior(i1:i1+m) = feval(pr.fh_g, gp.X_u(i,:), pr);
                    else % One prior for all inducing inputs
                        gprior(i1:i1+m-1) = feval(gp.p.X_u.fh_g, gp.X_u(i,:), gp.p.X_u);
                    end
                    i1 = i1 + m;
                end
                
                for i=1:ncf
                    i1=st;
                    
                    gpcf = gp.cf{i};
                    DKuu = feval(gpcf.fh_ginput, gpcf, u);
                    DKuf = feval(gpcf.fh_ginput, gpcf, u, x);
                    
                    for i2 = 1:length(DKuu)
                        i1 = i1+1;
                        
                        % 0.5* a'*dK*a, where a = K\f
                        KfuiKuuKuu = iKuuKuf'*DKuu{i2};
                        gdata(i1) = gdata(i1) -0.5.*((2.*a'*DKuf{i2}'-(a'*KfuiKuuKuu))*(iKuuKuf*a) + ...
                                                     - (2.*a'.*sum(DKuf{i2}'.*iKuuKuf',2)'*a-a'.*sum(KfuiKuuKuu.*iKuuKuf',2)'*a) );
                        
                        % trace( inv(inv(W) + K) * dQ) )
                        gdata(i1) = gdata(i1) - 0.5.*(sum(sum(L2'.*(2.*L2'*DKuf{i2}'*iKuuKuf - L2'*KfuiKuuKuu*iKuuKuf))));
                        gdata(i1) = gdata(i1) + 0.5.*(2.*sum(LL.*sum(DKuf{i2}'.*iKuuKuf',2)) - sum(LL.*sum(KfuiKuuKuu.*iKuuKuf',2)));
                        
                        
                        % b2*dK*b3
                        b = (2*DKuf{i2}'-KfuiKuuKuu)*(iKuuKuf*b3)  - sum((2.*DKuf{i2}'- KfuiKuuKuu).*iKuuKuf',2).*b3;
                        bb = (iLa2W.*b - L2*(L2'*b));
                        s3 = b - (La1.*bb + B'*(B*bb));
                        gdata(i1) = gdata(i1) - s2'*s3;
                    end
                end
            end
        end
        
        % =================================================================
        % Gradient with respect to likelihood function parameters
        
        if ~isempty(strfind(gp.infer_params, 'likelihood')) && ~isempty(gp.likelih.fh_pak(gp.likelih))
            gdata_likelih = 0;
            likelih = gp.likelih;

            
            DW_sigma = feval(likelih.fh_g3, likelih, y, f, 'latent2+hyper', z);
            DL_sigma = feval(likelih.fh_g, likelih, y, f, 'hyper', z);
            DL_f_sigma = feval(likelih.fh_g2, likelih, y, f, 'latent+hyper', z);
            b = La1.*DL_f_sigma + B'*(B*DL_f_sigma);            
            bb = (iLa2W.*b - L2*(L2'*b));
            s3 = b - (La1.*bb + B'*(B*bb));            

            gdata_likelih = - DL_sigma - 0.5.*sum(s2t.*DW_sigma) - s2'*s3;
            
            % evaluate prior contribution for the gradient
            if isfield(gp.likelih, 'p')
                g_logPrior = -feval(likelih.fh_priorg, likelih);
            else
                g_logPrior = zeros(size(gdata_likelih));
            end
            % set the gradients into vectors that will be returned
            gdata = [gdata gdata_likelih];
            gprior = [gprior g_logPrior];
            i1 = length(gdata);
        end

        g = gdata + gprior;

        % ============================================================
        % PIC
        % ============================================================
      case {'PIC' 'PIC_BLOCK'}
        g_ind = zeros(1,numel(gp.X_u));
        gdata_ind = zeros(1,numel(gp.X_u));
        gprior_ind = zeros(1,numel(gp.X_u));

        u = gp.X_u;
        m = size(u,1);
        ind = gp.tr_index;

        [e, edata, eprior, f, L, a, La1] = gpla_e(gp_pak(gp), gp, x, y, 'z', z);

        K_fu = gp_cov(gp, x, u);         % f x u
        K_uu = gp_trcov(gp, u);          % u x u, noiseles covariance K_uu
        K_uu = (K_uu+K_uu')./2;          % ensure the symmetry of K_uu
        Luu = chol(K_uu);
        iKuuKuf = Luu\(Luu'\K_fu');
        B=Luu'\(K_fu');       % u x f

        W = -feval(gp.likelih.fh_g2, gp.likelih, y, f, 'latent', z);
        sqrtW = sqrt(W);
        
        % Components for trace( inv(inv(W) + K) * dK) )
        B2 = (repmat(sqrtW,1,m).*K_fu);
        for i=1:length(ind)
            La2{i} = eye(size(La1{i})) + diag(sqrtW(ind{i}))*La1{i}*diag(sqrtW(ind{i}));
            LLa2{i} = chol(La2{i});
            B3(ind{i},:) = LLa2{i}\(LLa2{i}'\B2(ind{i},:));
        end
        A2 = K_uu + B2'*B3; A2=(A2+A2')/2;
        L2 = repmat(sqrtW,1,m).*B3/chol(A2);
        for i=1:length(ind)
            iLa2W{i} = diag(sqrtW(ind{i}))*(LLa2{i}\(LLa2{i}'\diag(sqrtW(ind{i}))));
        end
        
        LL = sum(L2.*L2,2);
        BB = sum(B.^2)';
                
        % Evaluate s2
        C1 = L2'*B'*B;
        s2t = BB;
        for i=1:length(ind)
            C2(:,ind{i}) = L2(ind{i},:)'*La1{i};
            s2t1(ind{i},:) = diag(La1{i}*iLa2W{i}*La1{i});
            s2t2(ind{i},:) = La1{i}*iLa2W{i}*B(:,ind{i})';
            s2t3(ind{i},:) = La1{i}*L2(ind{i},:);
            Bt(ind{i},:) = iLa2W{i}*B(:,ind{i})';
            s2t(ind{i}) = s2t(ind{i}) + diag(La1{i});
        end
        
        s2t = s2t - (s2t1 - sum(C2.^2)' + sum(B'.*((B*Bt)*B)',2)...
                    - sum(C1.^2)' + 2*sum(s2t2.*B',2) - 2*sum(s2t3.*C1',2));

        s2 = 0.5*s2t.*feval(gp.likelih.fh_g3, gp.likelih, y, f, 'latent', z);
        b3 = feval(gp.likelih.fh_g, gp.likelih, y, f, 'latent', z);

        % =================================================================
        % Gradient with respect to covariance function parameters
        if ~isempty(strfind(gp.infer_params, 'covariance'))
            for i=1:ncf
                i1=0;
                if ~isempty(gprior)
                    i1 = length(gprior);
                end
                
                % Get the gradients of the covariance matrices 
                % and gprior from gpcf_* structures
                gpcf = gp.cf{i};
                [DKuu, gprior_cf] = feval(gpcf.fh_ghyper, gpcf, u); 
                DKuf = feval(gpcf.fh_ghyper, gpcf, u, x); 
                for kk = 1:length(ind)
                    DKff{kk} = feval(gpcf.fh_ghyper, gpcf, x(ind{kk},:));                 
                end
                
                for i2 = 1:length(DKuu)
                    i1 = i1+1;
                    
                    KfuiKuuKuu = iKuuKuf'*DKuu{i2};
                    gdata(i1) = -0.5.*((2.*a'*DKuf{i2}'-(a'*KfuiKuuKuu))*(iKuuKuf*a) );
                    gdata(i1) = gdata(i1) - 0.5.*(sum(sum(L2'.*(2.*L2'*DKuf{i2}'*iKuuKuf - L2'*KfuiKuuKuu*iKuuKuf))));

                    b = (2*DKuf{i2}'-KfuiKuuKuu)*(iKuuKuf*b3);
                    for kk=1:length(ind)
                        gdata(i1) = gdata(i1) -0.5.*(a(ind{kk})'*DKff{kk}{i2}*a(ind{kk})...
                                                     - (2.*a(ind{kk})'*DKuf{i2}(:,ind{kk})'*iKuuKuf(:,ind{kk})*a(ind{kk})...
                                                        -a(ind{kk})'*KfuiKuuKuu(ind{kk},:)*iKuuKuf(:,ind{kk})*a(ind{kk})) );
                        
                        % trace( inv(inv(W) + K) * dQ) )                        
                        gdata(i1) = gdata(i1) + 0.5.*(sum(sum(iLa2W{kk}.*DKff{kk}{i2})) - sum(sum(L2(ind{kk},:)'.*(L2(ind{kk},:)'*DKff{kk}{i2}))));
                        gdata(i1) = gdata(i1) + 0.5.*(2.*sum(sum(L2(ind{kk},:)'.*(L2(ind{kk},:)'*DKuf{i2}(:,ind{kk})'*iKuuKuf(:,ind{kk})))) - ...
                                                      sum(sum(L2(ind{kk},:)'.*((L2(ind{kk},:)'*KfuiKuuKuu(ind{kk},:))*iKuuKuf(:,ind{kk})))));
                        
                        b(ind{kk}) = b(ind{kk}) + DKff{kk}{i2}*b3(ind{kk})...
                            - (2.*DKuf{i2}(:,ind{kk})'- KfuiKuuKuu(ind{kk},:))*iKuuKuf(:,ind{kk})*b3(ind{kk});
                        bbt(ind{kk},:) = iLa2W{kk}*b(ind{kk});
                    end
                    
                    % b2*dK*b3
                    bb = (bbt - L2*(L2'*b));
                    for kk=1:length(ind)
                        s3t(ind{kk},:) = La1{kk}*bb(ind{kk});
                    end
                    s3 = b - (s3t + B'*(B*bb));
                    gdata(i1) = gdata(i1) - s2'*s3;
                    
                    gprior(i1) = gprior_cf(i2);
                end
                
                % Set the gradients of hyper-hyperparameter
                if length(gprior_cf) > length(DKuu)
                    for i2=length(DKuu)+1:length(gprior_cf)
                        i1 = i1+1;
                        gdata(i1) = 0;
                        gprior(i1) = gprior_cf(i2);
                    end
                end
                
                
            end
            
            % Evaluate the gradient from noise functions
            if isfield(gp, 'noise')
                nn = length(gp.noise);
                for i=1:nn
                    gpcf = gp.noise{i};
                    [DCff, gprior_cf] = feval(gpcf.fh_ghyper, gpcf, x);
                    for i2 = 1:length(DCff)
                        i1 = i1+1;
                        gdata(i1)= -0.5*DCff{i2}.*b*b';
                        ind = gpcf.tr_index;
                        for kk=1:length(ind)
                            gdata(i1)= gdata(i1) + 0.5*trace((inv(La{kk})-L(ind{kk},:)*L(ind{kk},:)')).*DCff{i2};
                        end
                        gprior(i1) = gprior_cf(i2);
                    end
                end
            end
            
        end
        
        % =================================================================
        % Gradient with respect to inducing inputs
        
        if ~isempty(strfind(gp.infer_params, 'inducing'))
            if isfield(gp.p, 'X_u') && ~isempty(gp.p.X_u)
                m = size(gp.X_u,2);
                
                st=0;
                if ~isempty(gprior)
                    st = length(gprior);
                end
                gdata(st+1:st+length(gp.X_u(:))) = 0;
                
                i1 = st+1;
                for i = 1:size(gp.X_u,1)
                    if iscell(gp.p.X_u) % Own prior for each inducing input
                        pr = gp.p.X_u{i};
                        gprior(i1:i1+m) = feval(pr.fh_g, gp.X_u(i,:), pr);
                    else % One prior for all inducing inputs
                        gprior(i1:i1+m-1) = feval(gp.p.X_u.fh_g, gp.X_u(i,:), gp.p.X_u);
                    end
                    i1 = i1 + m;
                end
                
                % Loop over the  covariance functions
                for i=1:ncf            
                    i1=st;
                    gpcf = gp.cf{i};
                    DKuu = feval(gpcf.fh_ginput, gpcf, u);
                    DKuf = feval(gpcf.fh_ginput, gpcf, u, x);
                    
                    for i2 = 1:length(DKuu)
                        i1 = i1+1;
                        
                        
                        KfuiKuuKuu = iKuuKuf'*DKuu{i2};
                        gdata(i1) = -0.5.*((2.*a'*DKuf{i2}'-(a'*KfuiKuuKuu))*(iKuuKuf*a) );
                        gdata(i1) = gdata(i1) - 0.5.*(sum(sum(L2'.*(2.*L2'*DKuf{i2}'*iKuuKuf - L2'*KfuiKuuKuu*iKuuKuf))));
                        
                        b = (2*DKuf{i2}'-KfuiKuuKuu)*(iKuuKuf*b3);
                        for kk=1:length(ind)
                            gdata(i1) = gdata(i1) -0.5.*(- (2.*a(ind{kk})'*DKuf{i2}(:,ind{kk})'*iKuuKuf(:,ind{kk})*a(ind{kk})...
                                                            -a(ind{kk})'*KfuiKuuKuu(ind{kk},:)*iKuuKuf(:,ind{kk})*a(ind{kk})) );
                            
                            % trace( inv(inv(W) + K) * dQ) )                        
                            gdata(i1) = gdata(i1) + 0.5.*(2.*sum(sum(L2(ind{kk},:)'.*(L2(ind{kk},:)'*DKuf{i2}(:,ind{kk})'*iKuuKuf(:,ind{kk})))) - ...
                                                          sum(sum(L2(ind{kk},:)'.*((L2(ind{kk},:)'*KfuiKuuKuu(ind{kk},:))*iKuuKuf(:,ind{kk})))));
                            
                            b(ind{kk}) = b(ind{kk}) + ...
                                - (2.*DKuf{i2}(:,ind{kk})'- KfuiKuuKuu(ind{kk},:))*iKuuKuf(:,ind{kk})*b3(ind{kk});
                            bbt(ind{kk},:) = iLa2W{kk}*b(ind{kk});
                        end
                        
                        % b2*dK*b3
                        bb = (bbt - L2*(L2'*b));
                        for kk=1:length(ind)
                            s3t(ind{kk},:) = La1{kk}*bb(ind{kk});
                        end
                        s3 = b - (s3t + B'*(B*bb));
                        gdata(i1) = gdata(i1) - s2'*s3;
                    end
                end
            end
        end
        
        % =================================================================
        % Gradient with respect to likelihood function parameters
        
        if ~isempty(strfind(gp.infer_params, 'likelihood')) && ~isempty(gp.likelih.fh_pak(gp.likelih))
            gdata_likelih = 0;
            likelih = gp.likelih;
            
            DW_sigma = feval(likelih.fh_g3, likelih, y, f, 'latent2+hyper', z);
            DL_sigma = feval(likelih.fh_g, likelih, y, f, 'hyper', z);
            DL_f_sigma = feval(likelih.fh_g2, likelih, y, f, 'latent+hyper', z);
            b = B'*(B*DL_f_sigma);
            for kk=1:length(ind)
                b(ind{kk}) = b(ind{kk}) + La1{kk}*DL_f_sigma(ind{kk});
                bbt(ind{kk},:) = iLa2W{kk}*b(ind{kk});
            end
            bb = (bbt - L2*(L2'*b));
            for kk=1:length(ind)
                s3t(ind{kk},:) = La1{kk}*bb(ind{kk});
            end
            s3 = b - (s3t + B'*(B*bb));

            gdata_likelih = - DL_sigma - 0.5.*sum(s2t.*DW_sigma) - s2'*s3;

            % evaluate prior contribution for the gradient
            if isfield(gp.likelih, 'p')
                g_logPrior = -feval(likelih.fh_priorg, likelih);
            else
                g_logPrior = zeros(size(gdata_likelih));
            end
            % set the gradients into vectors that will be returned
            gdata = [gdata gdata_likelih];
            gprior = [gprior g_logPrior];
            i1 = length(gdata);
        end

        g = gdata + gprior;        

        % ============================================================
        % CS+FIC
        % ============================================================        
      case 'CS+FIC'
        u = gp.X_u;
        m = size(u,1);

        [e, edata, eprior, f, L, a, La1] = gpla_e(gp_pak(gp), gp, x, y, 'z', z);

        cf_orig = gp.cf;

        cf1 = {};
        cf2 = {};
        j = 1;
        k = 1;
        for i = 1:ncf
            if ~isfield(gp.cf{i},'cs')
                cf1{j} = gp.cf{i};
                j = j + 1;
            else
                cf2{k} = gp.cf{i};
                k = k + 1;
            end
        end
        gp.cf = cf1;

        % First evaluate needed covariance matrices
        % v defines that parameter is a vector
        [Kv_ff, Cv_ff] = gp_trvar(gp, x);  % f x 1  vector
        K_fu = gp_cov(gp, x, u);         % f x u
        K_uu = gp_trcov(gp, u);    % u x u, noiseles covariance K_uu
        K_uu = (K_uu+K_uu')./2;     % ensure the symmetry of K_uu
        gp.cf = cf_orig;
        
        W = -feval(gp.likelih.fh_g2, gp.likelih, y, f, 'latent', z);
        
        % Find fill reducing permutation and permute all the
        % matrices
        p = analyze(La1);
        if ~isempty(z)
            z = z(p,:);
        end
        f = f(p);
        y = y(p);
        La1 = La1(p,p);
        K_fu = K_fu(p,:);
        L = L(p,:);
        x = x(p,:);
        W = W(p);
        a = a(p);
        
        Luu = chol(K_uu)';
        B=Luu\(K_fu');       % u x f
        iKuuKuf = Luu'\B;
        sW = sqrt(W);
        sqrtW = sparse(1:n,1:n,sW,n,n);
        Inn = sparse(1:n,1:n,1,n,n);
        
        % Components for trace( inv(inv(W) + K) * dK) )
        Lah = Inn + sqrtW*La1*sqrtW;
        LD2 = ldlchol(Lah);
        B2 = (repmat(sW,1,m).*K_fu);
        %B3 = Lah\B2;
        B3 = ldlsolve(LD2,B2);
        A2 = K_uu + B2'*B3; A2=(A2+A2')/2;
        L2 = repmat(sW,1,m).*B3/chol(A2);

        siLa2 = spinv(LD2,1);
        dsiLa2 = diag(siLa2);
        
        LL = sum(L2.*L2,2);
        BB = sum(B.^2)';
                
        % Evaluate s2
        C1 = L2'*B'*B;
        C2 = L2'*La1;
        C3 = repmat(sW,1,m).*ldlsolve(LD2,repmat(sW,1,m).*B');
        
        s2t = diag(La1) + BB;        
        %diag(La1*sqrtW*ldlsolve(LD2,sqrtW*La1))
        s2t = s2t - (diag(La1) - sum(La1*sqrtW.*siLa2',2)./sW - sum(C2.^2)' + sum(B'.*(B*C3*B)',2)...
                    - sum(C1.^2)' + 2*sum((La1*C3).*B',2) - 2*sum(C2'.*C1',2));
        
        s2 = 0.5*s2t.*feval(gp.likelih.fh_g3, gp.likelih, y, f, 'latent', z);
        
        b3 = feval(gp.likelih.fh_g, gp.likelih, y, f, 'latent', z);
        
        % =================================================================
        % Gradient with respect to covariance function parameters
        if ~isempty(strfind(gp.infer_params, 'covariance'))    
            for i=1:ncf
                i1=0;
                if ~isempty(gprior)
                    i1 = length(gprior);
                end
                
                gpcf = gp.cf{i};
                
                % Evaluate the gradient for FIC covariance functions
                if ~isfield(gpcf,'cs')
                    % Get the gradients of the covariance matrices 
                    % and gprior from gpcf_* structures
                    [DKff, gprior_cf] = feval(gpcf.fh_ghyper, gpcf, x, [], 1); 
                    DKuu = feval(gpcf.fh_ghyper, gpcf, u); 
                    DKuf = feval(gpcf.fh_ghyper, gpcf, u, x); 
                    
                    for i2 = 1:length(DKuu)
                        i1 = i1+1;
 
                        % 0.5* a'*dK*a, where a = K\f
                        KfuiKuuKuu = iKuuKuf'*DKuu{i2};
                        gdata(i1) = -0.5.*((2.*a'*DKuf{i2}'-(a'*KfuiKuuKuu))*(iKuuKuf*a) + (a'.*DKff{i2}')*a...
                                           - (2.*a'.*sum(DKuf{i2}'.*iKuuKuf',2)'*a-a'.*sum(KfuiKuuKuu.*iKuuKuf',2)'*a) );
                    
                        % trace( inv(inv(W) + K) * dQ) )
                        gdata(i1) = gdata(i1) - 0.5.*(sum(sum(L2'.*(2.*L2'*DKuf{i2}'*iKuuKuf - L2'*KfuiKuuKuu*iKuuKuf))));
                        gdata(i1) = gdata(i1) + 0.5.*(sum(DKff{i2}.*dsiLa2.*W - LL.*DKff{i2}));
                        gdata(i1) = gdata(i1) + 0.5.*(2.*sum(LL.*sum(DKuf{i2}'.*iKuuKuf',2)) - sum(LL.*sum(KfuiKuuKuu.*iKuuKuf',2)));
                        
                        gdata(i1) = gdata(i1) + 0.5.*sum(sum(sqrtW*ldlsolve(LD2,repmat(sW,1,m).*(2.*DKuf{i2}' - KfuiKuuKuu)).*iKuuKuf',2));
                        gdata(i1) = gdata(i1) - 0.5.*sum(sW.*dsiLa2.*sW.*sum((2.*DKuf{i2}' - KfuiKuuKuu).*iKuuKuf',2) ); 
                                                
                        % b2*dK*b3
                        b = (2*DKuf{i2}'-KfuiKuuKuu)*(iKuuKuf*b3) + DKff{i2}.*b3 - sum((2.*DKuf{i2}'- KfuiKuuKuu).*iKuuKuf',2).*b3;
                        bb = (sW.*ldlsolve(LD2,sW.*b) - L2*(L2'*b));
                        s3 = b - (La1*bb + B'*(B*bb));
                        gdata(i1) = gdata(i1) - s2'*s3;
                    
                        gprior(i1) = gprior_cf(i2);
                    end
                    
                    % Evaluate the gradient for compact support covariance functions
                else
                    % Get the gradients of the covariance matrices 
                    % and gprior from gpcf_* structures
                    [DKff,gprior_cf] = feval(gpcf.fh_ghyper, gpcf, x);
                    
                    for i2 = 1:length(DKff)
                        i1 = i1+1;
                        
                        % Evaluate the gradient with respect to magnSigma
                        gdata(i1) = 0.5*(sum(sW.*sum(siLa2.*(sqrtW*DKff{i2})',2)) - sum(sum(L2.*(L2'*DKff{i2}')')) - a'*DKff{i2}*a);
                        b = DKff{i2}*b3;
                        bb = (sW.*ldlsolve(LD2,sW.*b) - L2*(L2'*b));
                        s3 = b - (La1*bb + B'*(B*bb));
                        gdata(i1) = gdata(i1) - s2'*s3;
                        gprior(i1) = gprior_cf(i2);

                    end
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
            
            % Evaluate the gradient from noise functions
            if isfield(gp, 'noise')
                nn = length(gp.noise);
                for i=1:nn
                    gpcf = gp.noise{i};       
                    % Get the gradients of the covariance matrices 
                    % and gprior from gpcf_* structures
                    [DCff, gprior_cf] = feval(gpcf.fh_ghyper, gpcf, x);
                    for i2 = 1:length(DCff)
                        i1 = i1+1;
                        gdata(i1)= -0.5*DCff{i2}.*b*b';
                        gdata(i1)= gdata(i1) + 0.5*sum(idiagLa-LL).*DCff{i2};
                        gprior(i1) = gprior_cf(i2);
                    end

                    % Set the gradients of hyper-hyperparameter
                    if length(gprior_cf) > length(DCff)
                        for i2=length(DCff)+1:length(gprior_cf)
                            i1 = i1+1;
                            gdata(i1) = 0;
                            gprior(i1) = gprior_cf(i2);
                        end
                    end
                end
            end               
        end
        % =================================================================
        % Gradient with respect to inducing inputs
        
        if ~isempty(strfind(gp.infer_params, 'inducing'))
            if isfield(gp.p, 'X_u') && ~isempty(gp.p.X_u)
                m = size(gp.X_u,2);
                st=0;
                if ~isempty(gprior)
                    st = length(gprior);
                end
                
                gdata(st+1:st+length(gp.X_u(:))) = 0;
                i1 = st+1;
                for i = 1:size(gp.X_u,1)
                    if iscell(gp.p.X_u) % Own prior for each inducing input
                        pr = gp.p.X_u{i};
                        gprior(i1:i1+m) = feval(pr.fh_g, gp.X_u(i,:), pr);
                    else % One prior for all inducing inputs
                        gprior(i1:i1+m-1) = feval(gp.p.X_u.fh_g, gp.X_u(i,:), gp.p.X_u);
                    end
                    i1 = i1 + m;
                end
                
                for i=1:ncf
                    i1=st;
                    gpcf = gp.cf{i};            
                    if ~isfield(gpcf,'cs')
                        DKuu = feval(gpcf.fh_ginput, gpcf, u);
                        DKuf = feval(gpcf.fh_ginput, gpcf, u, x);
                        
                        for i2 = 1:length(DKuu)
                            i1=i1+1;
                            
                            % 0.5* a'*dK*a, where a = K\f
                            KfuiKuuKuu = iKuuKuf'*DKuu{i2};
                            gdata(i1) = -0.5.*((2.*a'*DKuf{i2}'-(a'*KfuiKuuKuu))*(iKuuKuf*a) ...
                                               - (2.*a'.*sum(DKuf{i2}'.*iKuuKuf',2)'*a-a'.*sum(KfuiKuuKuu.*iKuuKuf',2)'*a) );
                            
                            % trace( inv(inv(W) + K) * dQ) )
                            gdata(i1) = gdata(i1) - 0.5.*(sum(sum(L2'.*(2.*L2'*DKuf{i2}'*iKuuKuf - L2'*KfuiKuuKuu*iKuuKuf))));
                            gdata(i1) = gdata(i1) + 0.5.*(2.*sum(LL.*sum(DKuf{i2}'.*iKuuKuf',2)) - sum(LL.*sum(KfuiKuuKuu.*iKuuKuf',2)));
                            
                            gdata(i1) = gdata(i1) + 0.5.*sum(sum(sqrtW*ldlsolve(LD2,repmat(sW,1,m).*(2.*DKuf{i2}' - KfuiKuuKuu)).*iKuuKuf',2));
                            gdata(i1) = gdata(i1) - 0.5.*( sum(sW.*dsiLa2.*sW.*sum((2.*DKuf{i2}' - KfuiKuuKuu).*iKuuKuf',2)) ); 
                            
                            % b2*dK*b3
                            b = (2*DKuf{i2}'-KfuiKuuKuu)*(iKuuKuf*b3) - sum((2.*DKuf{i2}'- KfuiKuuKuu).*iKuuKuf',2).*b3;
                            bb = (sW.*ldlsolve(LD2,sW.*b) - L2*(L2'*b));
                            s3 = b - (La1*bb + B'*(B*bb));
                            gdata(i1) = gdata(i1) - s2'*s3;
                        end
                    end
                end
            end
        end
        % =================================================================
        % Gradient with respect to likelihood function parameters
        
        if ~isempty(strfind(gp.infer_params, 'likelihood')) && ~isempty(gp.likelih.fh_pak(gp.likelih))
            gdata_likelih = 0;
            likelih = gp.likelih;
            
            DW_sigma = feval(likelih.fh_g3, likelih, y, f, 'latent2+hyper', z);
            DL_sigma = feval(likelih.fh_g, likelih, y, f, 'hyper', z);
            DL_f_sigma = feval(likelih.fh_g2, likelih, y, f, 'latent+hyper', z);
            b = La1*DL_f_sigma + B'*(B*DL_f_sigma);            
            bb = (sW.*ldlsolve(LD2,sW.*b) - L2*(L2'*b));
            s3 = b - (La1*bb + B'*(B*bb));            
            
            gdata_likelih = - DL_sigma - 0.5.*sum(s2t.*DW_sigma) - s2'*s3;
            
            % evaluate prior contribution for the gradient
            if isfield(gp.likelih, 'p')
                g_logPrior = -feval(likelih.fh_priorg, likelih);
            else
                g_logPrior = zeros(size(gdata_likelih));
            end
            % set the gradients into vectors that will be returned
            gdata = [gdata gdata_likelih];
            gprior = [gprior g_logPrior];
            i1 = length(gdata);
        end

        g = gdata + gprior;
        
    end
    
    assert(isreal(gdata))
    assert(isreal(gprior))
end
