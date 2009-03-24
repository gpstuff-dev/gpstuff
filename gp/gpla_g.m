function [g, gdata, gprior] = gpla_g(w, gp, x, y, param, varargin)
%GPLA_G   Evaluate gradient of Laplace approximation's marginal log posterior estimate 
%
%	Description
%	G = GPLA_G(W, GP, X, Y) takes a full GP hyper-parameter vector W, 
%       data structure GP a matrix X of input vectors and a matrix Y
%       of target vectors, and evaluates the gradient G of EP's marginal 
%       log posterior estimate . Each row of X corresponds to one input
%       vector and each row of Y corresponds to one target vector. 
%
%	G = GPLA_G(W, GP, P, Y, PARAM) in case of sparse model takes also  
%       string PARAM defining the parameters to take the gradients with 
%       respect to. Possible parameters are 'hyper' = hyperparameters and 
%      'inducing' = inducing inputs, 'hyper+inducing' = hyper+inducing parameters.
%
%	[G, GDATA, GPRIOR] = GPLA_G(GP, X, Y) also returns separately  the
%	data and prior contributions to the gradient.
%
%       NOTE! The CS+FIC model is not supported 
%
%	See also   
%       GPLA_E, LA_PRED

% Copyright (c) 2007-2008      Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.
    
    gp=gp_unpak(gp, w, param);       % unpak the parameters
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
        [e, edata, eprior, f, L, La2, b] = gpla_e(gp_pak(gp, param), gp, x, y, param, varargin{:});

        if issparse(K)
            W = La2;
            I = sparse(1:n,1:n,1,n,n);
            w1 = K\f;
            sqrtW = sqrt(W);
            sinvB = spinv(L,1);
            isqrtWsinBsqrtW = sqrtW\sinvB*sqrtW;
            w2 = sum( isqrtWsinBsqrtW .* K,2) .* feval(gp.likelih.fh_g3, gp.likelih, y, f, 'latent');
            w2 = - (sqrtW*ldlsolve(L, sqrtW \ w2))'; 
            w3 = b;
            invB = sqrtW*sinvB*sqrtW;
        else
            W = La2;
            I = eye(size(K));
            w1 = K\f;
            if W >= 0
                sqrtW = sqrt(W);
                w2 = diag(K*sqrtW/L'/L/sqrtW) .* feval(gp.likelih.fh_g3, gp.likelih, y, f, 'latent');
                invB = sqrtW/L'/L*sqrtW;
            else
                w2 = diag(inv(inv(K) + W)) .* feval(gp.likelih.fh_g3, gp.likelih, y, f, 'latent');
                invB = inv( inv(W) + K );
            end
            w2 = - w2' / (I + K*W);
            w3 = b;
        end

        % Hyperparameters
        % --------------------
        if strcmp(param,'hyper') || strcmp(param,'hyper+likelih')
            % Evaluate the gradients from covariance functions
            for i=1:ncf
                i1=0;
                if ~isempty(gprior)
                    i1 = length(gprior);
                end
            
                gpcf = gp.cf{i};
                [DKff, gprior_cf] = feval(gpcf.fh_ghyper, gpcf, x);
                
                for i2 = 1:length(DKff)
                    i1 = i1+1;
                    
                    Bdm = w1'*(DKff{i2}*w1);
                    Bdm = Bdm + w2*(DKff{i2}*w3);
                    Cdm = sum(sum(invB.*DKff{i2}));
                    gdata(i1) = 0.5.*(Cdm - Bdm);
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
        
        % likelihood parameters
        %--------------------------------------
        if strcmp(param,'likelih') || strcmp(param,'hyper+likelih')
            gdata_likelih = 0;
            likelih = gp.likelih;
            
            DW_sigma = feval(likelih.fh_g3, likelih, y, f, 'latent2+hyper');
            DL_sigma = feval(likelih.fh_g, likelih, y, f, 'hyper');            
            w3 = K * feval(likelih.fh_hessian, likelih, y, f, 'latent+hyper');
            
            gdata_likelih = - DL_sigma - 0.5.*sum(diag(inv(inv(K) + W)).*DW_sigma) - 0.5.*w2*w3;
           
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
        % FIC
        % ============================================================
      case 'FIC'
        g_ind = zeros(1,numel(gp.X_u));
        gdata_ind = zeros(1,numel(gp.X_u));
        gprior_ind = zeros(1,numel(gp.X_u));

        u = gp.X_u;
        m = size(u,1);

        [e, edata, eprior, f, L, La1, b] = gpla_e(gp_pak(gp, param), gp, x, y, param, varargin{:});

        K_fu = gp_cov(gp, x, u);         % f x u
        K_uu = gp_trcov(gp, u);          % u x u, noiseles covariance K_uu
        K_uu = (K_uu+K_uu')./2;          % ensure the symmetry of K_uu
        iKuuKuf = K_uu\K_fu';

        W = -feval(gp.likelih.fh_hessian, gp.likelih, y, f, 'latent');
        sqrtW = sqrt(W);
        b = f'./La1' - (f'*L)*L';

        La2 = 1 + W.*La1;
        La3 = 1./La1 + W;
        B2 = (repmat(sqrtW,1,m).*K_fu);

        % Components for
        B3 = repmat(La2,1,m).\B2;
        A2 = K_uu + B2'*B3; A2=(A2+A2')/2;
        L2 = B3/chol(A2);

        % Evaluate diag(La3 - L3'*L3)
        B4 = repmat(La3,1,m).\L;
        A3 = eye(size(K_uu)) - L'*B4; A3 = (A3+A3')./2;
        L3 = B4/chol(A3);
        dA3L3tL3 = -1./La3' - sum(L3.*L3,2)';
        dA3L3tL3 = dA3L3tL3.*feval(gp.likelih.fh_g3, gp.likelih, y, f, 'latent')';

        KufW = K_fu'.*repmat(W',m,1);
        iLa2Kfu = repmat(La2,1,m).\K_fu;
        A4 = K_uu + KufW*iLa2Kfu; A4 = (A4+A4')./2;
        L4 = iLa2Kfu/chol(A4);
        L5 = chol(A4)'\(KufW./repmat(La2',m,1));

        % Set the parameters for the actual gradient evaluation
        b2 = (dA3L3tL3./La2' - dA3L3tL3*L4*L5);
        b3 = feval(gp.likelih.fh_g, gp.likelih, y, f, 'latent');
        L = repmat(sqrtW,1,m).*L2;
        La = La2./W;
        
        LL = sum(L.*L,2);
        
        % =================================================================
        % Evaluate the gradients from covariance functions
        % =================================================================
        if strcmp(param,'hyper') || strcmp(param,'hyper+inducing') || strcmp(param,'hyper+likelih') || strcmp(param,'all')
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
                    
                    KfuiKuuKuu = iKuuKuf'*DKuu{i2};
                    gdata(i1) = -0.5.*((2*b*DKuf{i2}'-(b*KfuiKuuKuu))*(iKuuKuf*b') + 2.*sum(sum(L'.*(L'*DKuf{i2}'*iKuuKuf))) - ...
                                       sum(sum(L'.*((L'*KfuiKuuKuu)*iKuuKuf))));
                    
                    gdata(i1) = gdata(i1) - 0.5.*(b.*DKff{i2}')*b';
                    gdata(i1) = gdata(i1) + 0.5.*(2.*b.*sum(DKuf{i2}'.*iKuuKuf',2)'*b'- b.*sum(KfuiKuuKuu.*iKuuKuf',2)'*b');
                    gdata(i1) = gdata(i1) + 0.5.*(sum(DKff{i2}./La - LL.*DKff{i2}));
                    gdata(i1) = gdata(i1) + 0.5.*(2.*sum(LL.*sum(DKuf{i2}'.*iKuuKuf',2)) - sum(LL.*sum(KfuiKuuKuu.*iKuuKuf',2)));
                    
                    gdata(i1) = gdata(i1) - 0.5.*(2*b2*DKuf{i2}'-(b2*KfuiKuuKuu))*(iKuuKuf*b3);
                    gdata(i1) = gdata(i1) - 0.5.*(b2.*DKff{i2}')*b3;
                    gdata(i1) = gdata(i1) + 0.5.*(2.*b2.*sum(DKuf{i2}'.*iKuuKuf',2)'*b3- b2.*sum(KfuiKuuKuu.*iKuuKuf',2)'*b3);
                    
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
            % Evaluate the gradient from noise functions
            % =================================================================
            if isfield(gp, 'noise')
                nn = length(gp.noise);
                for i=1:nn
                    i1 = i1+1;
                    
                    gpcf = gp.noise{i};
                    
                    [DCff, gprior_cf] = feval(gpcf.fh_ghyper, gpcf, x);
                    for i2 = 1:length(DCff)
                        gdata(i1)= -0.5*DCff.*b*b';
                        gdata(i1)= gdata(i1) + 0.5*sum((1./La-LL).*DCff{i2});
                        gprior(i1) = gprior_cf(i2);
                    end
                end
            end            
        end
        
        if strcmp(param,'inducing') || strcmp(param,'hyper+inducing') || strcmp(param,'all')
            st=0;
            if ~isempty(gprior)
                st = length(gprior);
            end
            gdata(st+1:st+length(gp.X_u(:))) = 0;
            
            for i=1:ncf
                i1=st;
                
                gpcf = gp.cf{i};
                [DKuu, gprior_ind] = feval(gpcf.fh_ginput, gpcf, u);
                [DKuf] = feval(gpcf.fh_ginput, gpcf, u, x);
                
                for i2 = 1:length(DKuu)
                    i1 = i1+1;
                    KfuiKuuKuu = iKuuKuf'*DKuu{i2};
                    
                    gdata(i1) = gdata(i1) - 0.5.*((2*b*DKuf{i2}'-(b*KfuiKuuKuu))*(iKuuKuf*b') + ...
                                         2.*sum(sum(L'.*(L'*DKuf{i2}'*iKuuKuf))) - sum(sum(L'.*((L'*KfuiKuuKuu)*iKuuKuf))));
                    gdata(i1) = gdata(i1) + 0.5.*(2.*b.*sum(DKuf{i2}'.*iKuuKuf',2)'*b'- b.*sum(KfuiKuuKuu.*iKuuKuf',2)'*b');
                    gdata(i1) = gdata(i1) + 0.5.*(2.*sum(LL.*sum(DKuf{i2}'.*iKuuKuf',2)) - ...
                                                  sum(LL.*sum(KfuiKuuKuu.*iKuuKuf',2)));                    
                    
                    gdata(i1) = gdata(i1) -0.5.*(2*b2*DKuf{i2}'-(b2*KfuiKuuKuu))*(iKuuKuf*b3);
                    gdata(i1) = gdata(i1) + 0.5.*(2.*b2.*sum(DKuf{i2}'.*iKuuKuf',2)'*b3- b2.*sum(KfuiKuuKuu.*iKuuKuf',2)'*b3);
                    
                    gprior(i1) = gprior_ind(i2);
                end
            end
        end
        
        % likelihood parameters
        %--------------------------------------
        if strcmp(param,'likelih') || strcmp(param,'hyper+likelih')
            gdata_likelih = 0;
            likelih = gp.likelih;
            
            DW_sigma = feval(likelih.fh_g3, likelih, y, f, 'latent2+hyper');
            DL_sigma = feval(likelih.fh_g, likelih, y, f, 'hyper');            
            DL_f_sigma = feval(likelih.fh_hessian, likelih, y, f, 'latent+hyper');
            b3 = K_fu*(iKuuKuf*DL_f_sigma) + La1.*DL_f_sigma;
            
            gdata_likelih = - DL_sigma - 0.5.*sum((1./La3 + sum(L3.*L3,2)).*DW_sigma) - 0.5.*b2*b3;
           
            
            
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

        [e, edata, eprior, f, L, La1, b] = gpla_e(gp_pak(gp, param), gp, x, y, param, varargin{:});

        K_fu = gp_cov(gp, x, u);         % f x u
        K_uu = gp_trcov(gp, u);          % u x u, noiseles covariance K_uu
        K_uu = (K_uu+K_uu')./2;          % ensure the symmetry of K_uu
        iKuuKuf = K_uu\K_fu';

        W = -feval(gp.likelih.fh_hessian, gp.likelih, y, f, 'latent');
        sqrtW = sqrt(W);
        fiLa = zeros(size(f'));
        for i=1:length(ind)
            fiLa(ind{i}) = f(ind{i})'/La1{i};
            La{i} = diag(sqrtW(ind{i}))*La1{i}*diag(sqrtW(ind{i}));
            Lahat{i} = eye(size(La{i})) + La{i};
            La2{i} = eye(size(La1{i})) + La1{i}*diag(W(ind{i}));
            La3{i} = inv(La1{i}) + diag(W(ind{i}));
        end
        b = fiLa - (f'*L)*L';
        B2 = (repmat(sqrtW,1,m).*K_fu);

        % Components for
        B3 = zeros(size(K_fu));
        B4 = zeros(size(L));
        diLa3 = zeros(1,n);
        for i=1:length(ind)
            B3(ind{i},:) = Lahat{i}\B2(ind{i},:);
            B4(ind{i},:) = La3{i}\L(ind{i},:);
            diLa3(ind{i}) = diag(inv(La3{i}));
        end
        A2 = K_uu + B2'*B3; A2=(A2+A2')/2;
        L2 = B3/chol(A2);

        % Evaluate diag(La3 - L3'*L3)
        A3 = eye(size(K_uu)) - L'*B4; A3 = (A3+A3')./2;
        L3 = B4/chol(A3);
        dA3L3tL3 = diLa3 + sum(L3.*L3,2)';
        dA3L3tL3 = -dA3L3tL3.*feval(gp.likelih.fh_g3, gp.likelih, y, f, 'latent')';

        KufW = K_fu'.*repmat(W',m,1);
        iLa2Kfu = zeros(size(K_fu));
        KufWiLa2 = zeros(size(K_fu'));
        for i=1:length(ind)
            iLa2Kfu(ind{i},:) = La2{i}\K_fu(ind{i},:);
            KufWiLa2(:,ind{i}) = KufW(:,ind{i})/La2{i};
        end
        A4 = K_uu + KufW*iLa2Kfu; A4 = (A4+A4')./2;
        L4 = iLa2Kfu/chol(A4);
        L5 = chol(A4)'\KufWiLa2;

        % Set the parameters for the actual gradient evaluation
        b3 = feval(gp.likelih.fh_g, gp.likelih, y, f, 'latent');
        L = repmat(sqrtW,1,m).*L2;
        b2 = zeros(1,n);
        for i=1:length(ind)
            La{i} = diag(sqrtW(ind{i}))\Lahat{i}/diag(sqrtW(ind{i}));
            b2(ind{i}) = dA3L3tL3(ind{i})/La2{i};
        end
        b2 = (b2 - dA3L3tL3*L4*L5);
        
        % =================================================================
        % Evaluate the gradients from covariance functions
        if strcmp(param,'hyper') || strcmp(param,'hyper+inducing') || strcmp(param,'hyper+likelih') || strcmp(param,'all')
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
                    %            H = (2*K_uf'- KfuiKuuKuu)*iKuuKuf;
                    % Here we evaluate  gdata = -0.5.* (b*H*b' + trace(L*L'H)
                    gdata(i1) = -0.5.*((2*b*DKuf{i2}'-(b*KfuiKuuKuu))*(iKuuKuf*b') + 2.*sum(sum(L'.*(L'*DKuf{i2}'*iKuuKuf))) - ...
                                       sum(sum(L'.*((L'*KfuiKuuKuu)*iKuuKuf))));
                    gdata(i1) = gdata(i1) -0.5.*(2*b2*DKuf{i2}'-(b2*KfuiKuuKuu))*(iKuuKuf*b3);
                    
                    for kk=1:length(ind)
                        gdata(i1) = gdata(i1) ...
                            + 0.5.*(-b(ind{kk})*DKff{kk}{i2}*b(ind{kk})' ...
                                    + 2.*b(ind{kk})*DKuf{i2}(:,ind{kk})'*iKuuKuf(:,ind{kk})*b(ind{kk})'- ...
                                    b(ind{kk})*KfuiKuuKuu(ind{kk},:)*iKuuKuf(:,ind{kk})*b(ind{kk})' ...
                                    + trace(La{kk}\DKff{kk}{i2})...
                                    - trace(L(ind{kk},:)*(L(ind{kk},:)'*DKff{kk}{i2})) ...               
                                    + 2.*sum(sum(L(ind{kk},:)'.*(L(ind{kk},:)'*DKuf{i2}(:,ind{kk})'*iKuuKuf(:,ind{kk})))) - ...
                                    sum(sum(L(ind{kk},:)'.*((L(ind{kk},:)'*KfuiKuuKuu(ind{kk},:))*iKuuKuf(:,ind{kk})))));                
                        
                        gdata(i1) = gdata(i1) ...
                            + 0.5.*(-b2(ind{kk})*DKff{kk}{i2}*b3(ind{kk}) ...
                                    + 2.*b2(ind{kk})*DKuf{i2}(:,ind{kk})'*iKuuKuf(:,ind{kk})*b3(ind{kk})- ...
                                    b2(ind{kk})*KfuiKuuKuu(ind{kk},:)*iKuuKuf(:,ind{kk})*b3(ind{kk}));
                    end
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

        if strcmp(param,'inducing') || strcmp(param,'hyper+inducing') || strcmp(param,'all')
            st=0;
            if ~isempty(gprior)
                st = length(gprior);
            end
            gdata(st+1:st+length(gp.X_u(:))) = 0;
                        
            % Loop over the  covariance functions
            for i=1:ncf            
                i1=st;
                gpcf = gp.cf{i};
                [DKuu, gprior_ind] = feval(gpcf.fh_ginput, gpcf, u);
                [DKuf] = feval(gpcf.fh_ginput, gpcf, u, x);
                
                for i2 = 1:length(DKuu)
                    i1 = i1+1;
                    
                    KfuiKuuDKuu_u = iKuuKuf'*DKuu{i2};
                    gdata(i1) = gdata(i1) -0.5.*((2*b*DKuf{i2}'-(b*KfuiKuuDKuu_u))*(iKuuKuf*b') + 2.*sum(sum(L'.*((L'*DKuf{i2}')*iKuuKuf))) - ...
                                        sum(sum(L'.*((L'*KfuiKuuDKuu_u)*iKuuKuf))));
                    gdata(i1) = gdata(i1) -0.5.*(2*b2*DKuf{i2}'-(b2*KfuiKuuDKuu_u))*(iKuuKuf*b3);
                    
                    for kk=1:length(ind)
                        gdata(i1) = gdata(i1) + 0.5.*(2.*b(ind{kk})*DKuf{i2}(:,ind{kk})'*iKuuKuf(:,ind{kk})*b(ind{kk})'- ...
                                                      b(ind{kk})*KfuiKuuDKuu_u(ind{kk},:)*iKuuKuf(:,ind{kk})*b(ind{kk})' ...
                                                      + 2.*sum(sum(L(ind{kk},:)'.*(L(ind{kk},:)'*DKuf{i2}(:,ind{kk})'*iKuuKuf(:,ind{kk})))) - ...
                                                      sum(sum(L(ind{kk},:)'.*((L(ind{kk},:)'*KfuiKuuDKuu_u(ind{kk},:))*iKuuKuf(:,ind{kk})))));
                        gdata(i1) = gdata(i1) + 0.5.*(2.*b2(ind{kk})*DKuf{i2}(:,ind{kk})'*iKuuKuf(:,ind{kk})*b3(ind{kk})- ...
                                                      b2(ind{kk})*KfuiKuuDKuu_u(ind{kk},:)*iKuuKuf(:,ind{kk})*b3(ind{kk}));
                    end
                    gprior(i1) = gprior_ind(i2);
                end
            end
        end
        
        % likelihood parameters
        %--------------------------------------
        if strcmp(param,'likelih') || strcmp(param,'hyper+likelih') || strcmp(param,'all')
            gdata_likelih = 0;
            likelih = gp.likelih;
            
            DW_sigma = feval(likelih.fh_g3, likelih, y, f, 'latent2+hyper');
            DL_sigma = feval(likelih.fh_g, likelih, y, f, 'hyper');            
            DL_f_sigma = feval(likelih.fh_hessian, likelih, y, f, 'latent+hyper');
            b3 = K_fu*(iKuuKuf*DL_f_sigma);
            for i=1:length(ind)
                b3(ind{i}) = b3(ind{i}) + La1{i}*DL_f_sigma(ind{i});
            end
                        
            gdata_likelih = - DL_sigma - 0.5.*sum((diLa3' + sum(L3.*L3,2)).*DW_sigma) - 0.5.*b2*b3; 
            
            
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
        g_ind = zeros(1,numel(gp.X_u));
        gdata_ind = zeros(1,numel(gp.X_u));
        gprior_ind = zeros(1,numel(gp.X_u));

        u = gp.X_u;
        m = size(u,1);

        [e, edata, eprior, f, L, La1, b] = gpla_e(gp_pak(gp, param), gp, x, y, param, varargin{:});

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
        
        % Find fill reducing permutation and permute all the
        % matrices
        p = analyze(La1);
        r(p) = 1:n;
        gp.likelih = feval(gp.likelih.fh_permute, gp.likelih, p);
        f = f(p);
        y = y(p);
        La1 = La1(p,p);
        K_fu = K_fu(p,:);
        L = L(p,:);
        x = x(p,:);

        % Help matrices
        iKuuKuf = K_uu\K_fu';
        Inn = sparse(1:n,1:n,1,n,n);
        Wd = -feval(gp.likelih.fh_hessian, gp.likelih, y, f, 'latent');
        sqrtW = sqrt(Wd);
        W = sparse(1:n,1:n,Wd,n,n);
        sqrtW = sparse(1:n,1:n,sqrtW,n,n);
        
        % b=f'*(Qff+La1)^{-1}
        b = f'/La1 - (f'*L)*L';

        % Help matrices for trace component
        sqrtWLa1 = sqrtW*La1;
        Lahat = Inn + sqrtWLa1*sqrtW;
        LDh = ldlchol(Lahat);
        B2 = sqrtW*K_fu;
        %        B3 = Lahat\B2;
        B3 = ldlsolve(LDh,B2);
        A2 = K_uu + B2'*B3; A2=(A2+A2')/2;
        L2 = B3/chol(A2);
        
        % Help matrices for b2 set 1
        %        L3 = La1*L-sqrtWLa1'*(Lahat\(sqrtWLa1*L));
        L3 = La1*L - sqrtWLa1'*ldlsolve(LDh,(sqrtWLa1*L));
        AA2 = eye(size(K_uu)) - L'*L3; AA2 = (AA2 + AA2')./2;
        L3 = L3/chol(AA2);
        
        % Evaluate diag(La3^{-1} + L3'*L3).*thirdgrad
        %b2 = diag(La1) - sum((sqrtWLa1'/chol(Lahat)).^2,2) + sum(L3.*L3,2);
        La = (sqrtW\Lahat)/sqrtW;
        LD = ldlchol(La);
        siLa = spinv(LD,1);
        
        b2 = (sum(La1.*siLa,2)./Wd  + sum(L3.*L3,2)).*feval(gp.likelih.fh_g3, gp.likelih, y, f, 'latent');
        
        %b2 = (b2'*W)';
        
        % Help matrices for b2 set 2 
        La2 = W + W*La1*W;
        KufW = K_fu'*W;
        LD2 = ldlchol(La2);
        %iLa2WKfu = La2\(W*K_fu);
        iLa2WKfu = ldlsolve(LD2,KufW');
        A4 = K_uu + KufW*iLa2WKfu; A4 = (A4+A4')./2;
        L4 = iLa2WKfu/chol(A4);
        
        % Evaluate rest of b2
        %b2 = b2'/La2 - b2'*L4*L4';
        b2 = ldlsolve(LD2,b2)' - b2'*L4*L4';
        b2 = -b2*W;

        % Set the parameters for the actual gradient evaluation
        b3 = feval(gp.likelih.fh_g, gp.likelih, y, f, 'latent');
        L = sqrtW*L2;
        idiagLa = diag(siLa);        
        LL = sum(L.*L,2);
        
        % =================================================================
        if strcmp(param,'hyper') || strcmp(param,'hyper+inducing') || strcmp(param,'hyper+likelih')
            % Evaluate the gradients from covariance functions
            
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
                        
                        % Evaluate the gradient with respect to magnSigma
                        KfuiKuuKuu = iKuuKuf'*DKuu{i2};
                        gdata(i1) = -0.5.*((2*b*DKuf{i2}'-(b*KfuiKuuKuu))*(iKuuKuf*b') + 2.*sum(sum(L'.*(L'*DKuf{i2}'*iKuuKuf))) - ...
                                           sum(sum(L'.*((L'*KfuiKuuKuu)*iKuuKuf))));
                        
                        gdata(i1) = gdata(i1) - 0.5.*(b.*DKff{i2}')*b';
                        gdata(i1) = gdata(i1) + 0.5.*(2.*b.*sum(DKuf{i2}'.*iKuuKuf',2)'*b'- b.*sum(KfuiKuuKuu.*iKuuKuf',2)'*b');
                        gdata(i1) = gdata(i1) + 0.5.*sum(idiagLa.*DKff{i2} - LL.*DKff{i2});
                        gdata(i1) = gdata(i1) + 0.5.*(2.*sum(LL.*sum(DKuf{i2}'.*iKuuKuf',2)) - sum(LL.*sum(KfuiKuuKuu.*iKuuKuf',2)));
                        
                        %gdata(i1) = gdata(i1) + 0.5.*sum(sum(La\((2.*K_uf') - KfuiKuuKuu).*iKuuKuf',2));
                        gdata(i1) = gdata(i1) + 0.5.*sum(sum(ldlsolve(LD,2.*DKuf{i2}' - KfuiKuuKuu).*iKuuKuf',2));
                        gdata(i1) = gdata(i1) - 0.5.*( idiagLa'*(sum((2.*DKuf{i2}' - KfuiKuuKuu).*iKuuKuf',2)) ); 
                        
                        gdata(i1) = gdata(i1) - 0.5.*(2*b2*DKuf{i2}'-(b2*KfuiKuuKuu))*(iKuuKuf*b3);
                        gdata(i1) = gdata(i1) - 0.5.*(b2.*DKff{i2}')*b3;
                        gdata(i1) = gdata(i1) + 0.5.*(2.*b2.*sum(DKuf{i2}'.*iKuuKuf',2)'*b3 - b2.*sum(KfuiKuuKuu.*iKuuKuf',2)'*b3);
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
                        gdata(i1) = 0.5*(sum(sum(siLa.*DKff{i2}',2)) - sum(sum(L.*(L'*DKff{i2}')')) - b*DKff{i2}*b');
                        gdata(i1) = gdata(i1) - 0.5.*b2*DKff{i2}*b3;
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

        if strcmp(param,'inducing') || strcmp(param,'hyper+inducing') || strcmp(param,'all')
            st=0;
            if ~isempty(gprior)
                st = length(gprior);
            end
            gdata(st+1:st+length(gp.X_u(:))) = 0;
                        
            for i=1:ncf
                i1=st;
                gpcf = gp.cf{i};            
                if ~isfield(gpcf,'cs')
                    [DKuu, gprior_ind] = feval(gpcf.fh_ginput, gpcf, u);
                    [DKuf] = feval(gpcf.fh_ginput, gpcf, u, x);
                    
                    for i2 = 1:length(DKuu)
                        i1=i1+1;
                        
                        KfuiKuuKuu = iKuuKuf'*DKuu{i2};
                        
                        gdata(i1) = gdata(i1) -0.5.*((2*b*DKuf{i2}'-(b*KfuiKuuKuu))*(iKuuKuf*b') + ...
                                                             2.*sum(sum(L'.*(L'*DKuf{i2}'*iKuuKuf))) - sum(sum(L'.*((L'*KfuiKuuKuu)*iKuuKuf))));
                        gdata(i1) = gdata(i1) + 0.5.*(2.*b.*sum(DKuf{i2}'.*iKuuKuf',2)'*b'- b.*sum(KfuiKuuKuu.*iKuuKuf',2)'*b');
                        gdata(i1) = gdata(i1) + 0.5.*(2.*sum(LL.*sum(DKuf{i2}'.*iKuuKuf',2)) - ...
                                                              sum(LL.*sum(KfuiKuuKuu.*iKuuKuf',2)));
                        
                        gdata(i1) = gdata(i1) + 0.5.*sum(sum(ldlsolve(LD,(2.*DKuf{i2}') - KfuiKuuKuu).*iKuuKuf',2));
                        gdata(i1) = gdata(i1) - 0.5.*( idiagLa'*(sum((2.*DKuf{i2}' - KfuiKuuKuu).*iKuuKuf',2)) ); % corrected
                        
                        gdata(i1) = gdata(i1) -0.5.*(2*b2*DKuf{i2}'-(b2*KfuiKuuKuu))*(iKuuKuf*b3);
                        gdata(i1) = gdata(i1) + 0.5.*(2.*b2.*sum(DKuf{i2}'.*iKuuKuf',2)'*b3- b2.*sum(KfuiKuuKuu.*iKuuKuf',2)'*b3);
                        gprior(i1) = gprior_ind(i2);
                    end
                end
            end
        end
        
        % likelihood parameters
        %--------------------------------------
        if strcmp(param,'likelih') || strcmp(param,'hyper+likelih') || strcmp(param,'all')
            gdata_likelih = 0;
            likelih = gp.likelih;
            
            DW_sigma = feval(likelih.fh_g3, likelih, y, f, 'latent2+hyper');
            DL_sigma = feval(likelih.fh_g, likelih, y, f, 'hyper');            
            DL_f_sigma = feval(likelih.fh_hessian, likelih, y, f, 'latent+hyper');
            b3 = K_fu*(iKuuKuf*DL_f_sigma) + La1*DL_f_sigma;
                        
            gdata_likelih = - DL_sigma - 0.5.*sum((sum(La1.*siLa,2)./Wd + sum(L3.*L3,2)).*DW_sigma) - 0.5.*b2*b3;
            
            
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
    
end
