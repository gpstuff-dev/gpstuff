function [g, gdata, gprior] = gp_g(w, gp, x, t, param, varargin)
%GP_G   Evaluate gradient of energy for Gaussian Process
%
%	Description
%	G = GP_G(W, GP, X, Y) takes a full GP hyper-parameter vector W,
%       data structure GP a matrix X of input vectors and a matrix Y
%       of target vectors, and evaluates the gradient G of the energy function. 
%	Each row of X corresponds to one input vector and each row of Y 
%       corresponds to one target vector. NOTE! This parametrization works 
%       only for full GP!
%
%	G = GP_G(W, GP, P, Y, PARAM) in case of sparse model takes also
%       string PARAM defining the parameters to take the gradients with
%       respect to. Possible parameters are 
%       'hyper'          = hyperparameters
%       'inducing'       = inducing inputs 
%       'hyper+inducing' = hyperparameters and inducing inputs
%
%	[G, GDATA, GPRIOR] = GP_G(GP, X, Y, VARARGIN) also returns separately
%	the data and prior contributions to the gradient.
%
%	See also
%       GP_E, GP_PAK, GP_UNPAK, GPCF_*

% Copyright (c) 2006      Helsinki University of Technology (author) Jarno Vanhatalo
% Copyright (c) 2007-2008 Jarno Vanhatalo

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.

gp=gp_unpak(gp, w, param);       % unpak the parameters
ncf = length(gp.cf);
n=size(x,1);

g = [];
gdata = [];
gprior = [];

% ============================================================
% FULL
% ============================================================
switch gp.type
    case 'FULL'   % A full GP
        % Evaluate covariance
        [K, C] = gp_trcov(gp,x);
        
        if issparse(C)
            invC = sinv(C);       % evaluate the sparse inverse
            LD = ldlchol(C);
            b = ldlsolve(LD,t);
        else
            invC = inv(C);        % evaluate the full inverse
            b = C\t;
        end

        % Get the gradients of the covariance matrices and gprior
        % from gpcf_* structures and evaluate the gradients
        for i=1:ncf
            i1=0;
            if ~isempty(gprior)
                i1 = length(gprior);
            end
            
            gpcf = gp.cf{i};
            gpcf.GPtype = gp.type;
            [gprior, DKff] = feval(gpcf.fh_ghyper, gpcf, x, t, g, gdata, gprior);
            i1 = i1+1;
            i2 = 1;
            
            % Evaluate the gradient with respect to magnSigma
            Bdm = b'*(DKff{i2}*b);
            Cdm = sum(sum(invC.*DKff{i2})); % help argument for magnSigma2
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
                Cdl = sum(sum(invC.*DKff{i2})); % help arguments for lengthScale
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
                [gprior, DCff] = feval(noise.fh_ghyper, noise, x, t, g, gdata, gprior);
                
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
        DKuu_u = 0;
        DKuf_u = 0;

        % First evaluate the needed covariance matrices
        % v defines that parameter is a vector
        [Kv_ff, Cv_ff] = gp_trvar(gp, x);  % 1 x f  vector
        K_fu = gp_cov(gp, x, u);         % f x u
        K_uu = gp_trcov(gp, u);          % u x u, noiseles covariance K_uu
        K_uu = (K_uu+K_uu')./2;          % ensure the symmetry of K_uu
        Luu = chol(K_uu)';
        % Evaluate the Lambda (La)
        % Q_ff = K_fu*inv(K_uu)*K_fu'
        % Here we need only the diag(Q_ff), which is evaluated below
        B=Luu\(K_fu');
        Qv_ff=sum(B.^2)';
        Lav = Cv_ff-Qv_ff;   % 1 x f, Vector of diagonal elements
        % iLaKfu = diag(inv(Lav))*K_fu = inv(La)*K_fu
        iLaKfu = zeros(size(K_fu));  % f x u,
        for i=1:n
            iLaKfu(i,:) = K_fu(i,:)./Lav(i);  % f x u
        end
        % ... then evaluate some help matrices.
        % A = K_uu+K_uf*inv(La)*K_fu
        A = K_uu+K_fu'*iLaKfu;
        A = (A+A')./2;               % Ensure symmetry
        A = chol(A);
        L = iLaKfu/A;
        b = t'./Lav' - (t'*L)*L';
        iKuuKuf = K_uu\K_fu';
        La = Lav;
        
        
        % =================================================================
        % Loop over the covariance functions
        for i=1:ncf            
            i1=0;
            if ~isempty(gprior)
                i1 = length(gprior);
            end
            
            gpcf = gp.cf{i};
            gpcf.GPtype = gp.type;
            gpcf.X_u = gp.X_u;
            if strcmp(param,'hyper') || strcmp(param,'hyper+inducing')
                % Get the gradients of the covariance matrices 
                % and gprior from gpcf_* structures
                [gprior, DKff, DKuu, DKuf] = feval(gpcf.fh_ghyper, gpcf, x, t, g, gdata, gprior); 
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
                end
            end
            if strcmp(param,'inducing') || strcmp(param,'hyper+inducing')                
                [gprior_ind, DKuu, DKuf] = feval(gpcf.fh_gind, gpcf, x, t, g_ind, gdata_ind, gprior_ind);
                
                for i2 = 1:length(DKuu)
                    KfuiKuuKuu = iKuuKuf'*DKuu{i2};
                
                    gdata_ind(i2) = gdata_ind(i2) - 0.5.*((2*b*DKuf{i2}'-(b*KfuiKuuKuu))*(iKuuKuf*b') + ...
                                            2.*sum(sum(L'.*(L'*DKuf{i2}'*iKuuKuf))) - sum(sum(L'.*((L'*KfuiKuuKuu)*iKuuKuf))));
                    gdata_ind(i2) = gdata_ind(i2) + 0.5.*(2.*b.*sum(DKuf{i2}'.*iKuuKuf',2)'*b'- b.*sum(KfuiKuuKuu.*iKuuKuf',2)'*b');
                    gdata_ind(i2) = gdata_ind(i2) + 0.5.*(2.*sum(sum(L.*L,2).*sum(DKuf{i2}'.*iKuuKuf',2)) - ...
                                                            sum(sum(L.*L,2).*sum(KfuiKuuKuu.*iKuuKuf',2)));                    
                end
            end
        end

        % Loop over the noise functions
        if isfield(gp, 'noise')
            nn = length(gp.noise);
            for i=1:nn
                i1 = i1+1;
                
                gpcf = gp.noise{i};
                gpcf.GPtype = gp.type;
                gpcf.X_u = gp.X_u;
                if strcmp(param,'hyper') || strcmp(param,'hyper+inducing')
                    % Get the gradients of the covariance matrices 
                    % and gprior from gpcf_* structures
                    [gprior, DCff] = feval(gpcf.fh_ghyper, gpcf, x, t, g, gdata, gprior);
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
        ind = gp.tr_index;
        DKuu_u = 0;
        DKuf_u = 0;

        % First evaluate the needed covariance matrices
        % if they are not in the memory
        % v defines that parameter is a vector
        K_fu = gp_cov(gp, x, u);         % f x u
        K_uu = gp_trcov(gp, u);          % u x u, noiseles covariance K_uu
        K_uu = (K_uu+K_uu')./2;          % ensure the symmetry of K_uu
        Luu = chol(K_uu)';
        % Evaluate the Lambda (La)
        % Q_ff = K_fu*inv(K_uu)*K_fu'
        % Here we need only the diag(Q_ff), which is evaluated below
        %B=K_fu/Luu;
        B=Luu\K_fu';
        iLaKfu = zeros(size(K_fu));  % f x u
        for i=1:length(ind)
            Qbl_ff = B(:,ind{i})'*B(:,ind{i});
            [Kbl_ff, Cbl_ff] = gp_trcov(gp, x(ind{i},:));
            La{i} = Cbl_ff - Qbl_ff;
            iLaKfu(ind{i},:) = La{i}\K_fu(ind{i},:);
        end

        % ... then evaluate some help matrices.
        % A = chol(K_uu+K_uf*inv(La)*K_fu))
        A = K_uu+K_fu'*iLaKfu;
        A = (A+A')./2;            % Ensure symmetry

        L = iLaKfu/chol(A);
        b = zeros(1,n);
        b_apu=(t'*L)*L';
        for i=1:length(ind)
            b(ind{i}) = t(ind{i})'/La{i} - b_apu(ind{i});
        end

        iKuuKuf = K_uu\K_fu';                % L, b, iKuuKuf, La
        
        % =================================================================
        % Loop over the  covariance functions
        for i=1:ncf            
            i1=0;
            if ~isempty(gprior)
                i1 = length(gprior);
            end
            
            gpcf = gp.cf{i};
            gpcf.GPtype = gp.type;
            gpcf.X_u = gp.X_u;
            gpcf.tr_index = gp.tr_index;
            if strcmp(param,'hyper') || strcmp(param,'hyper+inducing')
                % Get the gradients of the covariance matrices 
                % and gprior from gpcf_* structures
                [gprior, DKff, DKuu, DKuf] = feval(gpcf.fh_ghyper, gpcf, x, t, g, gdata, gprior); 
                i1 = i1+1;
                i2 = 1;                
           
                % Evaluate the gradient with respect to magnSigma
                K_ff = DKff{i2};
                KfuiKuuKuu = iKuuKuf'*DKuu{i2};
                %            H = (2*K_uf'- KfuiKuuKuu)*iKuuKuf;
                % Here we evaluate  gdata = -0.5.* (b*H*b' + trace(L*L'H)
                gdata(i1) = -0.5.*((2*b*DKuf{i2}'-(b*KfuiKuuKuu))*(iKuuKuf*b') + 2.*sum(sum(L'.*(L'*DKuf{i2}'*iKuuKuf))) - ...
                    sum(sum(L'.*((L'*KfuiKuuKuu)*iKuuKuf))));
                
                for kk=1:length(K_ff)
                        gdata(i1) = gdata(i1) ...
                            + 0.5.*(-b(ind{kk})*K_ff{kk}*b(ind{kk})' ...
                                    + 2.*b(ind{kk})*DKuf{i2}(:,ind{kk})'*iKuuKuf(:,ind{kk})*b(ind{kk})'- ...
                                    b(ind{kk})*KfuiKuuKuu(ind{kk},:)*iKuuKuf(:,ind{kk})*b(ind{kk})' ... 
                                    + trace(La{kk}\K_ff{kk})...
                                    - trace(L(ind{kk},:)*(L(ind{kk},:)'*K_ff{kk})) ...
                                    + 2.*sum(sum(L(ind{kk},:)'.*(L(ind{kk},:)'*DKuf{i2}(:,ind{kk})'*iKuuKuf(:,ind{kk})))) - ...
                                    sum(sum(L(ind{kk},:)'.*((L(ind{kk},:)'*KfuiKuuKuu(ind{kk},:))*iKuuKuf(:,ind{kk})))));
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
                    for kk=1:length(K_ff)
                        gdata(i1) = gdata(i1) ... 
                            + 0.5.*(-b(ind{kk})*DKff_l{kk}*b(ind{kk})' ...
                                    + 2.*b(ind{kk})*DKuf{i2}(:,ind{kk})'*iKuuKuf(:,ind{kk})*b(ind{kk})'- ...
                                    b(ind{kk})*KfuiKuuDKuu_l(ind{kk},:)*iKuuKuf(:,ind{kk})*b(ind{kk})' ... 
                                    + trace(La{kk}\DKff_l{kk})...
                                    - trace(L(ind{kk},:)*(L(ind{kk},:)'*DKff_l{kk})) ...
                                    + 2.*sum(sum(L(ind{kk},:)'.*(L(ind{kk},:)'*DKuf{i2}(:,ind{kk})'*iKuuKuf(:,ind{kk})))) - ...
                                    sum(sum(L(ind{kk},:)'.*((L(ind{kk},:)'*KfuiKuuDKuu_l(ind{kk},:))*iKuuKuf(:,ind{kk})))));
                    end
                end
            end
            
            if strcmp(param,'inducing') || strcmp(param,'hyper+inducing')
                [gprior_ind, DKuu, DKuf] = feval(gpcf.fh_gind, gpcf, x, t, g_ind, gdata_ind, gprior_ind);
                           
                for i2 = 1:length(DKuu)
                    KfuiKuuDKuu_u = iKuuKuf'*DKuu{i2};
                    
                    gdata_ind(i2) = gdata_ind(i2) -0.5.*((2*b*DKuf{i2}'-(b*KfuiKuuDKuu_u))*(iKuuKuf*b') + 2.*sum(sum(L'.*((L'*DKuf{i2}')*iKuuKuf))) - ...
                                           sum(sum(L'.*((L'*KfuiKuuDKuu_u)*iKuuKuf))));

                    for kk=1:length(ind)
                        gdata_ind(i2) = gdata_ind(i2) + 0.5.*(2.*b(ind{kk})*DKuf{i2}(:,ind{kk})'*iKuuKuf(:,ind{kk})*b(ind{kk})'- ...
                                                              b(ind{kk})*KfuiKuuDKuu_u(ind{kk},:)*iKuuKuf(:,ind{kk})*b(ind{kk})' ...
                                                              + 2.*sum(sum(L(ind{kk},:)'.*(L(ind{kk},:)'*DKuf{i2}(:,ind{kk})'*iKuuKuf(:,ind{kk})))) - ...
                                                              sum(sum(L(ind{kk},:)'.*((L(ind{kk},:)'*KfuiKuuDKuu_u(ind{kk},:))*iKuuKuf(:,ind{kk})))));
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
                if strcmp(param,'hyper') || strcmp(param,'hyper+inducing')
                    [gprior, DCff] = feval(gpcf.fh_ghyper, gpcf, x, t, g, gdata, gprior);
                    gdata(i1)= -0.5*DCff.*b*b';
                    ind = gpcf.tr_index;
                    for kk=1:length(ind)
                        gdata(i1)= gdata(i1) + 0.5*trace((inv(La{kk})-L(ind{kk},:)*L(ind{kk},:)')).*DCff;
                    end                    
                end
            end
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
        DKuu_u = 0;
        DKuf_u = 0;

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

        % First evaluate the needed covariance matrices
        % if they are not in the memory
        % v defines that parameter is a vector
        [Kv_ff, Cv_ff] = gp_trvar(gp, x);  % 1 x f  vector
        K_fu = gp_cov(gp, x, u);         % f x u
        K_uu = gp_trcov(gp, u);          % u x u, noiseles covariance K_uu
        K_uu = (K_uu+K_uu')./2;          % ensure the symmetry of K_uu
        Luu = chol(K_uu)';
        % Evaluate the Lambda (La)
        % Q_ff = K_fu*inv(K_uu)*K_fu'
        % Here we need only the diag(Q_ff), which is evaluated below
        B=Luu\(K_fu');
        Qv_ff=sum(B.^2)';
        Lav = Cv_ff-Qv_ff;   % 1 x f, Vector of diagonal elements

        gp.cf = cf2;
        K_cs = gp_trcov(gp,x);
        La = sparse(1:n,1:n,Lav,n,n) + K_cs;
        gp.cf = cf_orig;

        LD = ldlchol(La);
        %        iLaKfu = La\K_fu;
        iLaKfu = ldlsolve(LD, K_fu);

        % ... then evaluate some help matrices.
        % A = chol(K_uu+K_uf*inv(La)*K_fu))
        A = K_uu+K_fu'*iLaKfu;
        A = (A+A')./2;            % Ensure symmetry
        L = iLaKfu/chol(A);
        %b = t'/La - (t'*L)*L';
        b = ldlsolve(LD,t)' - (t'*L)*L';
        
        siLa = sinv(La);
        idiagLa = diag(siLa);
        iKuuKuf = K_uu\K_fu';
        LL = L.*L;
        
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
            if strcmp(param,'hyper') || strcmp(param,'hyper+inducing')
                % Evaluate the gradient for FIC covariance functions
                if ~isfield(gpcf,'cs')
                    % Get the gradients of the covariance matrices 
                    % and gprior from gpcf_* structures
                    [gprior, DKff, DKuu, DKuf] = feval(gpcf.fh_ghyper, gpcf, x, t, g, gdata, gprior); 
                    i1 = i1+1;
                    i2 = 1;

                    % Evaluate the gradient with respect to magnSigma
                    KfuiKuuKuu = iKuuKuf'*DKuu{i2};
                    gdata(i1) = -0.5.*((2*b*DKuf{i2}'-(b*KfuiKuuKuu))*(iKuuKuf*b') + 2.*sum(sum(L'.*(L'*DKuf{i2}'*iKuuKuf))) - ...
                                       sum(sum(L'.*((L'*KfuiKuuKuu)*iKuuKuf))));
                    
                    temp1 = sum(KfuiKuuKuu.*iKuuKuf',2);
                    temp2 = sum(DKuf{i2}'.*iKuuKuf',2);
                    temp3 = 2.*DKuf{i2}' - KfuiKuuKuu;
                    gdata(i1) = gdata(i1) - 0.5.*(b.*DKff')*b';
                    gdata(i1) = gdata(i1) + 0.5.*(2.*b.*temp2'*b'- b.*temp1'*b');
                    gdata(i1) = gdata(i1) + 0.5.*(idiagLa'*DKff - sum(sum(LL)).*gpcf.magnSigma2);   % corrected
                    gdata(i1) = gdata(i1) + 0.5.*(2.*sum(sum(LL,2).*temp2) - sum(sum(LL,2).*temp1));
                    
                    %gdata(i1) = gdata(i1) + 0.5.*sum(sum(La\((2.*K_uf') - KfuiKuuKuu).*iKuuKuf',2));
                    gdata(i1) = gdata(i1) + 0.5.*sum(sum(ldlsolve(LD,temp3).*iKuuKuf',2));
                    gdata(i1) = gdata(i1) - 0.5.*( idiagLa'*(sum(temp3.*iKuuKuf',2)) ); % corrected                

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
                        
                        temp1 = sum(KfuiKuuKuu.*iKuuKuf',2);
                        temp2 = sum(DKuf{i2}'.*iKuuKuf',2);
                        temp3 = 2.*DKuf{i2}' - KfuiKuuKuu;
                        gdata(i1) = gdata(i1) + 0.5.*(b.*(2.*temp2'-temp1')*b');
                        gdata(i1) = gdata(i1) + 0.5.*(2.*sum(sum(LL,2).*temp2) - sum(sum(LL,2).*temp1));
                        
                        %gdata(i1) = gdata(i1) + 0.5.*sum(sum(La\(2.*DKuf_l{i2}').*iKuuKuf',2) - sum(La\KfuiKuuKuu.*iKuuKuf',2));
                        gdata(i1) = gdata(i1) + 0.5.*sum(sum(ldlsolve(LD,temp3).*iKuuKuf',2));
                        gdata(i1) = gdata(i1) - 0.5.*( idiagLa'*sum(temp3.*iKuuKuf',2) );
                    end
                % Evaluate the gradient for compact support covariance functions
                else
                    % Get the gradients of the covariance matrices 
                    % and gprior from gpcf_* structures
                    [gprior, DKff] = feval(gpcf.fh_ghyper, gpcf, x, t, g, gdata, gprior);
                    i1 = i1+1;
                    i2 = 1;
                    
                    % Evaluate the gradient with respect to magnSigma
                    gdata(i1) = 0.5*(sum(sum(siLa.*DKff{i2}',2)) - sum(sum(L.*(L'*DKff{i2}')')) - b*DKff{i2}*b');
                    
                    if isfield(gpcf.p.lengthScale, 'p') && ~isempty(gpcf.p.lengthScale.p)
                        i1 = i1+1;
                        if any(strcmp(fieldnames(gpcf.p.lengthScale.p),'nu'))
                            i1 = i1+1;
                        end
                    end

                    % Evaluate the gradient with respect to lengthScale
                    for i2 = 2:length(DKff)
                        i1 = i1+1;
                        gdata(i1) = 0.5*(sum(sum(siLa.*DKff{i2}',2)) - sum(sum(L.*(L'*DKff{i2}')')) - b*DKff{i2}*b');
                    end
                end
            end
            if strcmp(param,'inducing') || strcmp(param,'hyper+inducing')
                % Get the gradients of the covariance matrices 
                % and gprior from gpcf_* structures
                [gprior_ind, DKuu, DKuf] = feval(gpcf.fh_gind, gpcf, x, t, g_ind, gdata_ind, gprior_ind);
                
                for i2 = 1:length(DKuu)
                    KfuiKuuKuu = iKuuKuf'*DKuu{i2};
                    
                    gdata_ind(i2) = gdata_ind(i2) -0.5.*((2*b*DKuf{i2}'-(b*KfuiKuuKuu))*(iKuuKuf*b') + ...
                                           2.*sum(sum(L'.*(L'*DKuf{i2}'*iKuuKuf))) - sum(sum(L'.*((L'*KfuiKuuKuu)*iKuuKuf))));
                    gdata_ind(i2) = gdata_ind(i2) + 0.5.*(2.*b.*sum(DKuf{i2}'.*iKuuKuf',2)'*b'- b.*sum(KfuiKuuKuu.*iKuuKuf',2)'*b');
                    gdata_ind(i2) = gdata_ind(i2) + 0.5.*(2.*sum(sum(L.*L,2).*sum(DKuf{i2}'.*iKuuKuf',2)) - ...
                                                          sum(sum(L.*L,2).*sum(KfuiKuuKuu.*iKuuKuf',2)));
                    
                    gdata_ind(i2) = gdata_ind(i2) + 0.5.*sum(sum(ldlsolve(LD,(2.*DKuf{i2}') - KfuiKuuKuu).*iKuuKuf',2));
                    gdata_ind(i2) = gdata_ind(i2) - 0.5.*( idiagLa'*(sum((2.*DKuf{i2}' - KfuiKuuKuu).*iKuuKuf',2)) ); % corrected
                    
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
                if strcmp(param,'hyper') || strcmp(param,'hyper+inducing')
                    % Get the gradients of the covariance matrices 
                    % and gprior from gpcf_* structures
                    [gprior, DCff] = feval(gpcf.fh_ghyper, gpcf, x, t, g, gdata, gprior);
                    gdata(i1)= -0.5*DCff.*b*b';
                    gdata(i1)= gdata(i1) + 0.5*sum(idiagLa-sum(LL,2)).*DCff;
                end
            end
        end
        
        g = gdata + gprior;
end

switch param
    case 'inducing'
        % Evaluate here the gradient from prior
        g = gdata_ind;
    case 'hyper+inducing'
        % Evaluate here the gradient from prior
        g = [g gdata_ind];
end
end
