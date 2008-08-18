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

        W = La2;
        der_f = b;
        ntest=size(x,1);

        I = eye(size(K));
        sqrtW = sqrt(W);
        C = sqrtW*K;
        Z = (L\sqrtW);
        Z = Z'*Z;          %Z = sqrtW*((I + C*sqrtW)\sqrtW);

        CC = C*diag(feval(gp.likelih.fh_g3, gp.likelih, y, f, 'latent')./diag(sqrtW));
        s2 = -0.5*diag(L'\(L\(CC + CC')));       %s2 = -0.5*diag((I + C*sqrtW)\(CC + CC'));

        b = K\f;
        B = eye(size(K)) + K*W;
        invC = Z + der_f*(s2'/B);
        invCv = invC(:);

        % Evaluate the gradients from covariance functions
        for i=1:ncf
            i1=0;
            if ~isempty(gprior)
                i1 = length(gprior);
            end
            
            gpcf = gp.cf{i};
            gpcf.GPtype = gp.type;
            [gprior, DKff] = feval(gpcf.fh_ghyper, gpcf, x, y, g, gdata, gprior);
            
            i1 = i1+1;
            i2 = 1;
            
            % Evaluate the gradient with respect to magnSigma
            Bdm = b'*(DKff{i2}*b);
            Cdm = sum(invCv.*DKff{i2}(:)); % help argument for magnSigma2
            gdata(i1) = 0.5.*(Cdm - Bdm);

            if isfield(gpcf.p.lengthScale, 'p') && ~isempty(gpcf.p.lengthScale.p)
                i1 = i1+1;
                if any(strcmp(fieldnames(gpcf.p.lengthScale.p),'nu'))
                    i1 = i1+1;
                end
            end
            
            % Evaluate the gradient with respect to lengthScale
            for i2 = 2:length(DKff)
                i1 = i1+1;                
                Bdl = b'*(DKff{i2}*b);
                Cdl = sum(invCv.*DKff{i2}(:)); % help arguments for lengthScale
                gdata(i1)=0.5.*(Cdl - Bdl);
            end
        end

        % Evaluate the gradient from noise functions
        if isfield(gp, 'noise')
            nn = length(gp.noise);
            for i=1:nn
                i1 = i1+1;
                
                noise = gp.noise{i};
                noise.type = gp.type;
                [g, gdata, gprior] = feval(noise.fh_ghyper, noise, x, y, g, gdata, gprior, invC, B);
                
                B = trace(invC);
                C=b'*b;    
                gdata(i1)=0.5.*DCff.*(B - C); 
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

        [e, edata, eprior, f, L, La1, b] = gpla_e(gp_pak(gp, param), gp, x, y, param, varargin{:});

        K_fu = gp_cov(gp, x, u);         % f x u
        K_uu = gp_trcov(gp, u);          % u x u, noiseles covariance K_uu
        K_uu = (K_uu+K_uu')./2;          % ensure the symmetry of K_uu
        iKuuKuf = K_uu\K_fu';

        W = -feval(gp.likelih.fh_hessian, gp.likelih, y, f, 'latent');
        sqrtW = sqrt(W);
        b = f'./La1' - (f'*L)*L';

        La = W.*La1;
        Lahat = 1 + La;
        La2 = Lahat;
        La3 = 1./La1 + W;
        B2 = (repmat(sqrtW,1,m).*K_fu);

        % Components for
        B3 = repmat(Lahat,1,m).\B2;
        A2 = K_uu + B2'*B3; A2=(A2+A2)/2;
        L2 = B3/chol(A2);

        % Evaluate diag(La3 - L3'*L3)
        B4 = repmat(La3,1,m).\L;
        A3 = eye(size(K_uu)) - L'*B4; A3 = (A3+A3')./2;
        L3 = B4/chol(A3);
        dA3L3tL3 = 1./La3' + sum(L3.*L3,2)';
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
        La = Lahat./W;
        
        % =================================================================
        % Evaluate the gradients from covariance functions
        % =================================================================
        for i=1:ncf            
            i1=0;
            if ~isempty(gprior)
                i1 = length(gprior);
            end
            
            gpcf = gp.cf{i};
            gpcf.GPtype = gp.type;
            gpcf.X_u = gp.X_u;
            if strcmp(param,'hyper') || strcmp(param,'all')
                [gprior, DKff, DKuu, DKuf] = feval(gpcf.fh_ghyper, gpcf, x, y, g, gdata, gprior); 
                i1 = i1+1;
                i2 = 1;
                
                % Evaluate the gradient with respect to magnSigma
                KfuiKuuKuu = iKuuKuf'*DKuu{i2};
                gdata(i1) = -0.5.*((2*b*DKuf{i2}'-(b*KfuiKuuKuu))*(iKuuKuf*b') + 2.*sum(sum(L'.*(L'*DKuf{i2}'*iKuuKuf))) - ...
                    sum(sum(L'.*((L'*KfuiKuuKuu)*iKuuKuf))));

                gdata(i1) = gdata(i1) - 0.5.*(b.*DKff')*b';
                gdata(i1) = gdata(i1) + 0.5.*(2.*b.*sum(DKuf{i2}'.*iKuuKuf',2)'*b'- b.*sum(KfuiKuuKuu.*iKuuKuf',2)'*b');
                gdata(i1) = gdata(i1) + 0.5.*(sum(DKff./La) - sum(sum(L.*L)).*gpcf.magnSigma2);
                gdata(i1) = gdata(i1) + 0.5.*(2.*sum(sum(L.*L,2).*sum(DKuf{i2}'.*iKuuKuf',2)) - sum(sum(L.*L,2).*sum(KfuiKuuKuu.*iKuuKuf',2)));

                gdata(i1) = gdata(i1) - 0.5.*(2*b2*DKuf{i2}'-(b2*KfuiKuuKuu))*(iKuuKuf*b3);
                gdata(i1) = gdata(i1) - 0.5.*(b2.*DKff')*b3;
                gdata(i1) = gdata(i1) + 0.5.*(2.*b2.*sum(DKuf{i2}'.*iKuuKuf',2)'*b3- b2.*sum(KfuiKuuKuu.*iKuuKuf',2)'*b3);
                
                if isfield(gpcf.p.lengthScale, 'p') && ~isempty(gpcf.p.lengthScale.p)
                    i1 = i1+1;
                    if any(strcmp(fieldnames(gpcf.p.lengthScale.p),'nu'))
                        i1 = i1+1;
                    end
                end

                % Evaluate the gradient with respect to lengthScale
                for i2 = 2:length(DKuu)
                    i1 = i1+1;
                    KfuiKuuKuu = iKuuKuf'*DKuu{i2};
                    gdata(i1) = -0.5.*((2*b*DKuf{i2}'-(b*KfuiKuuKuu))*(iKuuKuf*b') + 2.*sum(sum(L'.*(L'*DKuf{i2}'*iKuuKuf))) - ...
                                       sum(sum(L'.*((L'*KfuiKuuKuu)*iKuuKuf))));
                    gdata(i1) = gdata(i1) + 0.5.*(2.*b.*sum(DKuf{i2}'.*iKuuKuf',2)'*b'- b.*sum(KfuiKuuKuu.*iKuuKuf',2)'*b');
                    gdata(i1) = gdata(i1) + 0.5.*(2.*sum(sum(L.*L,2).*sum(DKuf{i2}'.*iKuuKuf',2)) - sum(sum(L.*L,2).*sum(KfuiKuuKuu.*iKuuKuf',2)));
                    
                    gdata(i1) = gdata(i1) -0.5.*(2*b2*DKuf{i2}'-(b2*KfuiKuuKuu))*(iKuuKuf*b3);
                    gdata(i1) = gdata(i1) + 0.5.*(2.*b2.*sum(DKuf{i2}'.*iKuuKuf',2)'*b3 - b2.*sum(KfuiKuuKuu.*iKuuKuf',2)'*b3);
                end
            end
            if strcmp(param,'inducing') || strcmp(param,'all')                
                [gprior_ind, DKuu, DKuf] = feval(gpcf.fh_gind, gpcf, x, y, g_ind, gdata_ind, gprior_ind);
                
                for i2 = 1:length(DKuu)
                    KfuiKuuKuu = iKuuKuf'*DKuu{i2};
                    
                    gdata_ind(i2) = gdata_ind(i2) - 0.5.*((2*b*DKuf{i2}'-(b*KfuiKuuKuu))*(iKuuKuf*b') + ...
                                                          2.*sum(sum(L'.*(L'*DKuf{i2}'*iKuuKuf))) - sum(sum(L'.*((L'*KfuiKuuKuu)*iKuuKuf))));
                    gdata_ind(i2) = gdata_ind(i2) + 0.5.*(2.*b.*sum(DKuf{i2}'.*iKuuKuf',2)'*b'- b.*sum(KfuiKuuKuu.*iKuuKuf',2)'*b');
                    gdata_ind(i2) = gdata_ind(i2) + 0.5.*(2.*sum(sum(L.*L,2).*sum(DKuf{i2}'.*iKuuKuf',2)) - ...
                                                          sum(sum(L.*L,2).*sum(KfuiKuuKuu.*iKuuKuf',2)));                    
                   
                    gdata_ind(i2) = gdata_ind(i2) -0.5.*(2*b2*DKuf{i2}'-(b2*KfuiKuuKuu))*(iKuuKuf*b3);
                    gdata_ind(i2) = gdata_ind(i2) + 0.5.*(2.*b2.*sum(DKuf{i2}'.*iKuuKuf',2)'*b3- b2.*sum(KfuiKuuKuu.*iKuuKuf',2)'*b3);
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
                gpcf.GPtype = gp.type;
                gpcf.X_u = gp.X_u;
                if strcmp(param,'hyper') || strcmp(param,'all')
                    [gprior, DCff] = feval(gpcf.fh_ghyper, gpcf, x, y, g, gdata, gprior);
                    gdata(i1)= -0.5*DCff.*b*b';
                    gdata(i1)= gdata(i1) + 0.5*sum(1./La-sum(L.*L,2)).*DCff;
                end
            end
        end
        g = gdata + gprior;

        % ============================================================
        % PIC
        % ============================================================
    case 'PIC_BLOCK'
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
        A2 = K_uu + B2'*B3; A2=(A2+A2)/2;
        L2 = B3/chol(A2);

        % Evaluate diag(La3 - L3'*L3)
        A3 = eye(size(K_uu)) - L'*B4; A3 = (A3+A3')./2;
        L3 = B4/chol(A3);
        dA3L3tL3 = diLa3 + sum(L3.*L3,2)';
        dA3L3tL3 = dA3L3tL3.*feval(gp.likelih.fh_g3, gp.likelih, y, f, 'latent')';

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
        for i=1:ncf            
            i1=0;
            if ~isempty(gprior)
                i1 = length(gprior);
            end
            
            gpcf = gp.cf{i};
            gpcf.GPtype = gp.type;
            gpcf.X_u = gp.X_u;
            gpcf.tr_index = gp.tr_index;
            if strcmp(param,'hyper') || strcmp(param,'all')
                [gprior, DKff, DKuu, DKuf] = feval(gpcf.fh_ghyper, gpcf, x, y, g, gdata, gprior); 
                i1 = i1+1;
                i2 = 1;                
           
                % Evaluate the gradient with respect to magnSigma
                K_ff = DKff{i2};
                KfuiKuuKuu = iKuuKuf'*DKuu{i2};
                %            H = (2*K_uf'- KfuiKuuKuu)*iKuuKuf;
                % Here we evaluate  gdata = -0.5.* (b*H*b' + trace(L*L'H)
                gdata(i1) = -0.5.*((2*b*DKuf{i2}'-(b*KfuiKuuKuu))*(iKuuKuf*b') + 2.*sum(sum(L'.*(L'*DKuf{i2}'*iKuuKuf))) - ...
                                   sum(sum(L'.*((L'*KfuiKuuKuu)*iKuuKuf))));
                gdata(i1) = gdata(i1) -0.5.*(2*b2*DKuf{i2}'-(b2*KfuiKuuKuu))*(iKuuKuf*b3);
                
                for kk=1:length(K_ff)
                    gdata(i1) = gdata(i1) ...
                        + 0.5.*(-b(ind{kk})*K_ff{kk}*b(ind{kk})' ...
                        + 2.*b(ind{kk})*DKuf{i2}(:,ind{kk})'*iKuuKuf(:,ind{kk})*b(ind{kk})'- ...
                        b(ind{kk})*KfuiKuuKuu(ind{kk},:)*iKuuKuf(:,ind{kk})*b(ind{kk})' ...
                        + trace(La{kk}\K_ff{kk})...
                        - trace(L(ind{kk},:)*(L(ind{kk},:)'*K_ff{kk})) ...               
                        + 2.*sum(sum(L(ind{kk},:)'.*(L(ind{kk},:)'*DKuf{i2}(:,ind{kk})'*iKuuKuf(:,ind{kk})))) - ...
                        sum(sum(L(ind{kk},:)'.*((L(ind{kk},:)'*KfuiKuuKuu(ind{kk},:))*iKuuKuf(:,ind{kk})))));                
                    
                    gdata(i1) = gdata(i1) ...
                        + 0.5.*(-b2(ind{kk})*K_ff{kk}*b3(ind{kk}) ...
                                + 2.*b2(ind{kk})*DKuf{i2}(:,ind{kk})'*iKuuKuf(:,ind{kk})*b3(ind{kk})- ...
                                b2(ind{kk})*KfuiKuuKuu(ind{kk},:)*iKuuKuf(:,ind{kk})*b3(ind{kk}));
                end
                
                if isfield(gpcf.p.lengthScale, 'p') && ~isempty(gpcf.p.lengthScale.p)
                    i1 = i1+1;
                    if any(strcmp(fieldnames(gpcf.p.lengthScale.p),'nu'))
                        i1 = i1+1;
                    end
                end
                
                % Evaluate the gradient with respect to lengthScale
                for i2 = 2:length(DKuu)                 
                    i1 = i1+1;

                    DKff_l = DKff{i2};
                    KfuiKuuDKuu_l = iKuuKuf'*DKuu{i2};
                    % H = (2*DKuf_l'- KfuiKuuDKuu_l)*iKuuKuf;
                    % Here we evaluate  gdata = -0.5.* (b*H*b' + trace(L*L'H)
                    gdata(i1) = -0.5.*((2*b*DKuf{i2}'-(b*KfuiKuuDKuu_l))*(iKuuKuf*b') + 2.*sum(sum(L'.*((L'*DKuf{i2}')*iKuuKuf))) - ...
                                       sum(sum(L'.*((L'*KfuiKuuDKuu_l)*iKuuKuf))));
                    gdata(i1) = gdata(i1) -0.5.*(2*b2*DKuf{i2}'-(b2*KfuiKuuDKuu_l))*(iKuuKuf*b3);

                    for kk=1:length(K_ff)
                        gdata(i1) = gdata(i1) ...
                            + 0.5.*(-b(ind{kk})*DKff_l{kk}*b(ind{kk})' ...
                                    + 2.*b(ind{kk})*DKuf{i2}(:,ind{kk})'*iKuuKuf(:,ind{kk})*b(ind{kk})'- ...
                                    b(ind{kk})*KfuiKuuDKuu_l(ind{kk},:)*iKuuKuf(:,ind{kk})*b(ind{kk})' ...
                                    + trace(La{kk}\DKff_l{kk})...
                                    - trace(L(ind{kk},:)*(L(ind{kk},:)'*DKff_l{kk})) ...
                                    + 2.*sum(sum(L(ind{kk},:)'.*(L(ind{kk},:)'*DKuf{i2}(:,ind{kk})'*iKuuKuf(:,ind{kk})))) - ...
                                    sum(sum(L(ind{kk},:)'.*((L(ind{kk},:)'*KfuiKuuDKuu_l(ind{kk},:))*iKuuKuf(:,ind{kk})))));
                        
                        gdata(i1) = gdata(i1) ...
                            + 0.5.*(-b2(ind{kk})*DKff_l{kk}*b3(ind{kk}) ...
                                    + 2.*b2(ind{kk})*DKuf{i2}(:,ind{kk})'*iKuuKuf(:,ind{kk})*b3(ind{kk})- ...
                                    b2(ind{kk})*KfuiKuuDKuu_l(ind{kk},:)*iKuuKuf(:,ind{kk})*b3(ind{kk}));
                    end
                end
            end
            if strcmp(param,'inducing') || strcmp(param,'all')
                [gprior_ind, DKuu, DKuf] = feval(gpcf.fh_gind, gpcf, x, y, g_ind, gdata_ind, gprior_ind);
                           
                for i2 = 1:length(DKuu)
                    KfuiKuuDKuu_u = iKuuKuf'*DKuu{i2};
                    
                    gdata_ind(i2) = gdata_ind(i2) -0.5.*((2*b*DKuf{i2}'-(b*KfuiKuuDKuu_u))*(iKuuKuf*b') + 2.*sum(sum(L'.*((L'*DKuf{i2}')*iKuuKuf))) - ...
                                           sum(sum(L'.*((L'*KfuiKuuDKuu_u)*iKuuKuf))));
                    gdata_ind(i2) = gdata_ind(i2) -0.5.*(2*b2*DKuf{i2}'-(b2*KfuiKuuDKuu_u))*(iKuuKuf*b3);

                    for kk=1:length(ind)
                        gdata_ind(i2) = gdata_ind(i2) + 0.5.*(2.*b(ind{kk})*DKuf{i2}(:,ind{kk})'*iKuuKuf(:,ind{kk})*b(ind{kk})'- ...
                                                              b(ind{kk})*KfuiKuuDKuu_u(ind{kk},:)*iKuuKuf(:,ind{kk})*b(ind{kk})' ...
                                                              + 2.*sum(sum(L(ind{kk},:)'.*(L(ind{kk},:)'*DKuf{i2}(:,ind{kk})'*iKuuKuf(:,ind{kk})))) - ...
                                                              sum(sum(L(ind{kk},:)'.*((L(ind{kk},:)'*KfuiKuuDKuu_u(ind{kk},:))*iKuuKuf(:,ind{kk})))));
                        gdata_ind(i2) = gdata_ind(i2) + 0.5.*(2.*b2(ind{kk})*DKuf{i2}(:,ind{kk})'*iKuuKuf(:,ind{kk})*b3(ind{kk})- ...
                                    b2(ind{kk})*KfuiKuuDKuu_u(ind{kk},:)*iKuuKuf(:,ind{kk})*b3(ind{kk}));
                    end
                end
            end
        end

        % Evaluate the gradient from noise functions
        if isfield(gp, 'noise')
            nn = length(gp.noise);
            for i=1:nn
                i1 = i1+1;
                
                gpcf = gp.noise{i};
                gpcf.GPtype = gp.type;
                gpcf.X_u = gp.X_u;
                gpcf.tr_index = gp.tr_index;
                if strcmp(param,'hyper') || strcmp(param,'all')
                    [gprior, DCff] = feval(gpcf.fh_ghyper, gpcf, x, y, g, gdata, gprior);
                    gdata(i1)= -0.5*DCff.*b*b';
                    ind = gpcf.tr_index;
                    for kk=1:length(ind)
                        gdata(i1)= gdata(i1) + 0.5*trace((inv(La{kk})-L(ind{kk},:)*L(ind{kk},:)')).*DCff;
                    end                    
                end
            end
        end
        g = gdata + gprior;        
        
        
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

        % Help matrices
        iKuuKuf = K_uu\K_fu';
        Inn = sparse(1:n,1:n,1,n,n);
        Wd = -feval(gp.likelih.fh_hessian, gp.likelih, y, f, 'latent');
        sqrtW = sqrt(Wd);
        W = sparse(1:n,1:n,Wd,n,n);
        sqrtW = sparse(1:n,1:n,sqrtW,n,n);
        
        % b=f'*(Qff+La1)^{-1}*f
        b = f'/La1 - (f'*L)*L';

        % Help matrices for trace component
        sqrtWLa1 = sqrtW*La1;
        Lahat = Inn + sqrtWLa1*sqrtW;
        LDh = ldlchol(Lahat);
        B2 = sqrtW*K_fu;
        %        B3 = Lahat\B2;
        B3 = ldlsolve(LDh,B2);
        A2 = K_uu + B2'*B3; A2=(A2+A2)/2;
        L2 = B3/chol(A2);
        
        % Help matrices for b2 set 1
        %        L3 = La1*L-sqrtWLa1'*(Lahat\(sqrtWLa1*L));
        L3 = La1*L-sqrtWLa1'*ldlsolve(LDh,(sqrtWLa1*L));
        L3 = L3/chol(eye(size(K_uu)) - L'*L3);
                
        % Evaluate diag(La3^{-1} + L3'*L3).*thirdgrad
        %b2 = diag(La1) - sum((sqrtWLa1'/chol(Lahat)).^2,2) + sum(L3.*L3,2);
        La = (sqrtW\Lahat)/sqrtW;
        LD = ldlchol(La);
        siLa = sinv(La);
        b2 = sum(La1.*siLa,2)./Wd  + sum(L3.*L3,2);
        %        b2 = sum(La1.*sinv(sqrtW\Lahat/sqrtW),2)./Wd  + sum(L3.*L3,2); 
        b2 = b2.*feval(gp.likelih.fh_g3, gp.likelih, y, f, 'latent');
        
            
        % Help matrices for b2 set 2 
        La2 = W + W*La1*W;
        KufW = K_fu'*W;
        LD2 = ldlchol(La2);
        %iLa2WKfu = La2\(W*K_fu);
        iLa2WKfu = ldlsolve(LD2,(W*K_fu));
        A4 = K_uu + KufW*iLa2WKfu; A4 = (A4+A4')./2;
        L4 = iLa2WKfu/chol(A4);
        
        % Evaluate rest of b2
        %b2 = b2'/La2 - b2'*L4*L4';
        b2 = ldlsolve(LD2,b2)' - b2'*L4*L4';

        % Set the parameters for the actual gradient evaluation
        b2 = -b2*W;
        b3 = feval(gp.likelih.fh_g, gp.likelih, y, f, 'latent');
        L = sqrtW*L2;
        idiagLa = diag(siLa);
        
        % =================================================================
        % Evaluate the gradients from covariance functions
        % =================================================================
        for i=1:ncf            
            i1=0;
            if ~isempty(gprior)
                i1 = length(gprior);
            end
            
            gpcf = gp.cf{i};
            gpcf.GPtype = gp.type;
            gpcf.X_u = gp.X_u;
            if strcmp(param,'hyper') || strcmp(param,'all')
                % Evaluate the gradient for full support covariance functions
                if ~isfield(gpcf,'cs')
                    [gprior, DKff, DKuu, DKuf] = feval(gpcf.fh_ghyper, gpcf, x, y, g, gdata, gprior); 
                    i1 = i1+1;
                    i2 = 1;

                    % Evaluate the gradient with respect to magnSigma
                    KfuiKuuKuu = iKuuKuf'*DKuu{i2};
                    gdata(i1) = -0.5.*((2*b*DKuf{i2}'-(b*KfuiKuuKuu))*(iKuuKuf*b') + 2.*sum(sum(L'.*(L'*DKuf{i2}'*iKuuKuf))) - ...
                                       sum(sum(L'.*((L'*KfuiKuuKuu)*iKuuKuf))));
                    
                    gdata(i1) = gdata(i1) - 0.5.*(b.*DKff')*b';
                    gdata(i1) = gdata(i1) + 0.5.*(2.*b.*sum(DKuf{i2}'.*iKuuKuf',2)'*b'- b.*sum(KfuiKuuKuu.*iKuuKuf',2)'*b');
                    gdata(i1) = gdata(i1) + 0.5.*(idiagLa'*DKff - sum(sum(L.*L)).*gpcf.magnSigma2);
                    gdata(i1) = gdata(i1) + 0.5.*(2.*sum(sum(L.*L,2).*sum(DKuf{i2}'.*iKuuKuf',2)) - sum(sum(L.*L,2).*sum(KfuiKuuKuu.*iKuuKuf',2)));
                    
                    %gdata(i1) = gdata(i1) + 0.5.*sum(sum(La\((2.*K_uf') - KfuiKuuKuu).*iKuuKuf',2));
                    gdata(i1) = gdata(i1) + 0.5.*sum(sum(ldlsolve(LD,2.*DKuf{i2}' - KfuiKuuKuu).*iKuuKuf',2));
                    gdata(i1) = gdata(i1) - 0.5.*( idiagLa'*(sum((2.*DKuf{i2}' - KfuiKuuKuu).*iKuuKuf',2)) ); 
                    
                    gdata(i1) = gdata(i1) - 0.5.*(2*b2*DKuf{i2}'-(b2*KfuiKuuKuu))*(iKuuKuf*b3);
                    gdata(i1) = gdata(i1) - 0.5.*(b2.*DKff')*b3;
                    gdata(i1) = gdata(i1) + 0.5.*(2.*b2.*sum(DKuf{i2}'.*iKuuKuf',2)'*b3- b2.*sum(KfuiKuuKuu.*iKuuKuf',2)'*b3);

                    if isfield(gpcf.p.lengthScale, 'p') && ~isempty(gpcf.p.lengthScale.p)
                        i1 = i1+1;
                        if any(strcmp(fieldnames(gpcf.p.lengthScale.p),'nu'))
                            i1 = i1+1;
                        end
                    end
                    
                    % Evaluate the gradient with respect to lengthScale
                    for i2 = 2:length(DKuu)
                        i1 = i1+1;
                        
                        KfuiKuuKuu = iKuuKuf'*DKuu{i2};
                        gdata(i1) = -0.5.*((2*b*DKuf{i2}'-(b*KfuiKuuKuu))*(iKuuKuf*b') + 2.*sum(sum(L'.*(L'*DKuf{i2}'*iKuuKuf))) - ...
                                           sum(sum(L'.*((L'*KfuiKuuKuu)*iKuuKuf))));
                        
                        gdata(i1) = gdata(i1) + 0.5.*(2.*b.*sum(DKuf{i2}'.*iKuuKuf',2)'*b'- b.*sum(KfuiKuuKuu.*iKuuKuf',2)'*b');
                        gdata(i1) = gdata(i1) + 0.5.*(2.*sum(sum(L.*L,2).*sum(DKuf{i2}'.*iKuuKuf',2)) - sum(sum(L.*L,2).*sum(KfuiKuuKuu.*iKuuKuf',2)));
                        
                        %gdata(i1) = gdata(i1) + 0.5.*sum(sum(La\(2.*DKuf_l{i2}').*iKuuKuf',2) - sum(La\KfuiKuuKuu.*iKuuKuf',2));
                        gdata(i1) = gdata(i1) + 0.5.*sum(sum(ldlsolve(LD,2.*DKuf{i2}' - KfuiKuuKuu).*iKuuKuf',2));
                        gdata(i1) = gdata(i1) - 0.5.*( idiagLa'*(sum(2.*DKuf{i2}'.*iKuuKuf',2) - sum(KfuiKuuKuu.*iKuuKuf',2)) );
                        
                        gdata(i1) = gdata(i1) - 0.5.*(2*b2*DKuf{i2}'-(b2*KfuiKuuKuu))*(iKuuKuf*b3);
                        gdata(i1) = gdata(i1) + 0.5.*(2.*b2.*sum(DKuf{i2}'.*iKuuKuf',2)'*b3- b2.*sum(KfuiKuuKuu.*iKuuKuf',2)'*b3);
                    end
                % Evaluate the gradient for compact support covariance functions
                else
                    [gprior, DKff] = feval(gpcf.fh_ghyper, gpcf, x, y, g, gdata, gprior);
                    i1 = i1+1;
                    i2 = 1;
                    
                    % Evaluate the gradient with respect to magnSigma
                    gdata(i1) = 0.5*(sum(sum(siLa.*DKff{i2}',2)) - sum(sum(L.*(L'*DKff{i2}')')) - b*DKff{i2}*b');
                    gdata(i1) = gdata(i1) + 0.5.*b2*DKff{i2}*b3;

                    % Evaluate the gradient with respect to lengthScale
                    for i2 = 2:length(DKff)
                        i1 = i1+1;
                        gdata(i1) = 0.5*(sum(sum(siLa.*DKff{i2}',2)) - sum(sum(L.*(L'*DKff{i2}')')) - b*DKff{i2}*b');
                        gdata(i1) = gdata(i1) + 0.5.*b2*DKff{i2}*b3;
                    end
                end
            end
            if strcmp(param,'inducing') || strcmp(param,'all')
                [gprior_ind, DKuu, DKuf] = feval(gpcf.fh_gind, gpcf, x, y, g_ind, gdata_ind, gprior_ind);
                
                for i2 = 1:length(DKuu)
                    KfuiKuuKuu = iKuuKuf'*DKuu{i2};
                    
                    gdata_ind(i2) = gdata_ind(i2) -0.5.*((2*b*DKuf{i2}'-(b*KfuiKuuKuu))*(iKuuKuf*b') + ...
                                           2.*sum(sum(L'.*(L'*DKuf{i2}'*iKuuKuf))) - sum(sum(L'.*((L'*KfuiKuuKuu)*iKuuKuf))));
                    gdata_ind(i2) = gdata_ind(i2) + 0.5.*(2.*b.*sum(DKuf{i2}'.*iKuuKuf',2)'*b'- b.*sum(KfuiKuuKuu.*iKuuKuf',2)'*b');
                    gdata_ind(i2) = gdata_ind(i2) + 0.5.*(2.*sum(sum(L.*L,2).*sum(DKuf{i2}'.*iKuuKuf',2)) - ...
                                                          sum(sum(L.*L,2).*sum(KfuiKuuKuu.*iKuuKuf',2)));
                    
                    gdata_ind(i2) = gdata_ind(i2) + 0.5.*sum(sum(ldlsolve(LD,(2.*DKuf{i2}') - KfuiKuuKuu).*iKuuKuf',2));
                    gdata_ind(i2) = gdata_ind(i2) - 0.5.*( idiagLa'*(sum((2.*DKuf{i2}' - KfuiKuuKuu).*iKuuKuf',2)) ); % corrected
                    
                    gdata_ind(i2) = gdata_ind(i2) -0.5.*(2*b2*DKuf{i2}'-(b2*KfuiKuuKuu))*(iKuuKuf*b3);
                    gdata_ind(i2) = gdata_ind(i2) + 0.5.*(2.*b2.*sum(DKuf{i2}'.*iKuuKuf',2)'*b3- b2.*sum(KfuiKuuKuu.*iKuuKuf',2)'*b3);                    
                end
            end
        end

        % Evaluate the gradient from noise functions
        if isfield(gp, 'noise')
            nn = length(gp.noise);
            for i=1:nn
                i1 = i1+1;
                
                gpcf = gp.noise{i};
                gpcf.GPtype = gp.type;
                gpcf.X_u = gp.X_u;
                if strcmp(param,'inducing') || strcmp(param,'all')
                    [gprior, DCff] = feval(gpcf.fh_ghyper, gpcf, x, y, g, gdata, gprior);
                    gdata(i1)= -0.5*DCff.*b*b';
                    gdata(i1)= gdata(i1) + 0.5*sum(idiagLa-sum(L.*L,2)).*DCff;
                end
            end
        end
        g = gdata + gprior;
        
end
switch param
    case 'inducing'
        g = gdata_ind;
    case 'all'
        g = [g gdata_ind];
end

% ==============================================================
% Begin of the nested functions
% ==============================================================

    function deriv = derivative(f, likelihood)
        switch likelihood
            case 'probit'
                deriv = y.*normpdf(f)./normcdf(y.*f);
            case 'poisson'
                deriv = y - gp.avgE.*exp(f);
        end
    end
    function Hessian = hessian(f, likelihood)
        switch likelihood
            case 'probit'
                z = y.*f;
                Hessian = (normpdf(f)./normcdf(z)).^2 + z.*normpdf(f)./normcdf(z);
            case 'poisson'
                Hessian = gp.avgE.*exp(f);
        end
    end
    function thir_grad = thirdgrad(f,likelihood)
        switch likelihood
            case 'probit'
                z2 = normpdf(f)./normcdf(y.*f);
                thir_grad = 2.*y.*z2.^3 + 3.*f.*z2.^2 - z2.*(y-y.*f.^2);
            case 'poisson'
                thir_grad = - gp.avgE.*exp(f);
        end
    end
end
