function [g, gdata, gprior] = gpep_g(w, gp, x, y, param, varargin)
%GPEP_G   Evaluate gradient of EP's marginal log posterior estimate 
%
%	Description
%	G = GPEP_G(W, GP, X, Y) takes a full GP hyper-parameter vector W, 
%       data structure GP a matrix X of input vectors and a matrix Y
%       of target vectors, and evaluates the gradient G of EP's marginal 
%       log posterior estimate . Each row of X corresponds to one input
%       vector and each row of Y corresponds to one target vector. 
%
%	G = GPEP_G(W, GP, P, Y, PARAM) in case of sparse model takes also  
%       string PARAM defining the parameters to take the gradients with 
%       respect to. Possible parameters are 'hyper' = hyperparameters and 
%      'inducing' = inducing inputs, 'hyper+inducing' = hyper+inducing parameters.
%
%	[G, GDATA, GPRIOR] = GPEP_G(GP, X, Y) also returns separately  the
%	data and prior contributions to the gradient.
%
%       NOTE! The CS+FIC model is not supported 
%
%	See also   
%       GPEP_E, EP_PRED

% Copyright (c) 2007-2008  Jarno Vanhatalo

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
        [K, C] = gp_trcov(gp,x);
        [e, edata, eprior, tautilde, nutilde, L] = gpep_e(w, gp, x, y, param, varargin);

        Stildesqroot=diag(sqrt(tautilde));
        
        % logZep; nutilde; tautilde;
        b=nutilde-Stildesqroot*(L'\(L\(Stildesqroot*(C*nutilde))));
        invC = Stildesqroot*(L'\(L\Stildesqroot));
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
                [gprior, DCff] = feval(noise.fh_ghyper, noise, x, t, g, gdata, gprior);
                
                B = trace(invC);
                C=b'*b;    
                gdata(i1)=0.5.*DCff.*(B - C); 
            end
        end
        % likelihood parameters
        %--------------------------------------
        if strcmp(param,'likelih') || strcmp(param,'hyper+likelih')
            [Ef, Varf] = ep_pred(gp, x, y, x, param);                
            gdata_likelih = 0;
            likelih = gp.likelih;
            for k1 = 1:length(y)
                sigm2_i = Varf(k1) ;
                myy_i = Ef(k1);
                gdata_likelih = gdata_likelih - feval(likelih.fh_siteDeriv, likelih, y, k1, sigm2_i, myy_i);
            end
        end

        g = gdata + gprior;
        
        % ============================================================
        % SPARSE MODELS
        % ============================================================
      case {'FIC'}
        g_ind = zeros(1,numel(gp.X_u));
        gdata_ind = zeros(1,numel(gp.X_u));
        gprior_ind = zeros(1,numel(gp.X_u));
        
        u = gp.X_u;
        DKuu_u = 0;
        DKuf_u = 0;

        [e, edata, eprior, tautilde, nutilde, L, La, b] = gpep_e(w, gp, x, y, param, varargin);

        K_fu = gp_cov(gp, x, u);         % f x u
        K_uu = gp_trcov(gp, u);          % u x u, noiseles covariance K_uu
        K_uu = (K_uu+K_uu')./2;          % ensure the symmetry of K_uu        
        iKuuKuf = K_uu\K_fu';
        
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
            % Covariance function hyperparameters
            %--------------------------------------
            if strcmp(param,'hyper') || strcmp(param,'hyper+inducing') || strcmp(param,'hyper+likelih')
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
            % Inducing inputs
            %--------------------------------------
            if strcmp(param,'inducing') || strcmp(param,'hyper+inducing')                
                [gprior_ind, DKuu, DKuf] = feval(gpcf.fh_gind, gpcf, x, y, g_ind, gdata_ind, gprior_ind);
                
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

        % Evaluate the gradient from noise functions
        if isfield(gp, 'noise')
            nn = length(gp.noise);
            for i=1:nn
                i1 = i1+1;
                
                gpcf = gp.noise{i};
                gpcf.GPtype = gp.type;
                gpcf.X_u = gp.X_u;
                gpcf.tr_index = gp.tr_index;
                if strcmp(param,'hyper') || strcmp(param,'hyper+inducing') || strcmp(param,'hyper+likelih')
                    [gprior, DCff] = feval(gpcf.fh_ghyper, gpcf, x, y, g, gdata, gprior);
                    gdata(i1)= -0.5*DCff.*b*b';
                    gdata(i1)= gdata(i1) + 0.5*sum(1./La-sum(L.*L,2)).*DCff;
                end
            end
        end
        
        % likelihood parameters
        %--------------------------------------
        if strcmp(param,'likelih') || strcmp(param,'hyper+likelih')
            [Ef, Varf] = ep_pred(gp, x, y, x, param);                
            gdata_likelih = 0;
            likelih = gp.likelih;
            for k1 = 1:length(y)
                sigm2_i = Varf(k1) ;
                myy_i = Ef(k1);
                gdata_likelih = gdata_likelih - feval(likelih.fh_siteDeriv, likelih, y, k1, sigm2_i, myy_i);
            end
        end
        g = gdata + gprior;
        
      case {'PIC_BLOCK'}
        g_ind = zeros(1,numel(gp.X_u));
        gdata_ind = zeros(1,numel(gp.X_u));
        gprior_ind = zeros(1,numel(gp.X_u));
        
        u = gp.X_u;
        ind = gp.tr_index;
        DKuu_u = 0;
        DKuf_u = 0;

        [e, edata, eprior, tautilde, nutilde, L, La, b] = gpep_e(w, gp, x, y, param, varargin);

        K_fu = gp_cov(gp, x, u);         % f x u
        K_uu = gp_trcov(gp, u);          % u x u, noiseles covariance K_uu
        K_uu = (K_uu+K_uu')./2;          % ensure the symmetry of K_uu        
        iKuuKuf = K_uu\K_fu';
        
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
            if strcmp(param,'hyper') || strcmp(param,'hyper+inducing') || strcmp(param,'hyper+likelih')
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
                [gprior_ind, DKuu, DKuf] = feval(gpcf.fh_gind, gpcf, x, y, g_ind, gdata_ind, gprior_ind);
                
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
                if strcmp(param,'hyper') || strcmp(param,'hyper+inducing') || strcmp(param,'hyper+likelih')
                    [gprior, DCff] = feval(gpcf.fh_ghyper, gpcf, x, y, g, gdata, gprior);
                    gdata(i1)= -0.5*DCff.*b*b';
                    ind = gpcf.tr_index;
                    for kk=1:length(ind)
                        gdata(i1)= gdata(i1) + 0.5*trace((inv(La{kk})-L(ind{kk},:)*L(ind{kk},:)')).*DCff;
                    end                    
                end
            end
        end
        % likelihood parameters
        %--------------------------------------
        if strcmp(param,'likelih') || strcmp(param,'hyper+likelih')
            [Ef, Varf] = ep_pred(gp, x, y, x, param);                
            gdata_likelih = 0;
            likelih = gp.likelih;
            for k1 = 1:length(y)
                sigm2_i = Varf(k1) ;
                myy_i = Ef(k1);
                gdata_likelih = gdata_likelih - feval(likelih.fh_siteDeriv, likelih, y, k1, sigm2_i, myy_i);
            end
        end
        
        g = gdata + gprior;
        
     case {'CS+FIC'}
        g_ind = zeros(1,numel(gp.X_u));
        gdata_ind = zeros(1,numel(gp.X_u));
        gprior_ind = zeros(1,numel(gp.X_u));

        u = gp.X_u;
        DKuu_u = 0;
        DKuf_u = 0;

        [e, edata, eprior, tautilde, nutilde, L, La, b] = gpep_e(w, gp, x, y, param, varargin);

        m = length(u);
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
        iKuuKuf = K_uu\K_fu';
        gp.cf = cf_orig;
        
        LD = ldlchol(La);
        siLa = sinv(La);
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
            if strcmp(param,'hyper') || strcmp(param,'hyper+inducing') || strcmp(param,'hyper+likelih')
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
                    gdata(i1) = gdata(i1) + 0.5.*(idiagLa'*DKff - sum(sum(L.*L)).*gpcf.magnSigma2);   % corrected
                    gdata(i1) = gdata(i1) + 0.5.*(2.*sum(sum(L.*L,2).*sum(DKuf{i2}'.*iKuuKuf',2)) - sum(sum(L.*L,2).*sum(KfuiKuuKuu.*iKuuKuf',2)));
                    
                    %gdata(i1) = gdata(i1) + 0.5.*sum(sum(La\((2.*K_uf') - KfuiKuuKuu).*iKuuKuf',2));
                    gdata(i1) = gdata(i1) + 0.5.*sum(sum(ldlsolve(LD,2.*DKuf{i2}' - KfuiKuuKuu).*iKuuKuf',2));
                    gdata(i1) = gdata(i1) - 0.5.*( idiagLa'*(sum((2.*DKuf{i2}' - KfuiKuuKuu).*iKuuKuf',2)) ); % corrected                

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
                    end
                % Evaluate the gradient for compact support covariance functions
                else
                    [gprior, DKff] = feval(gpcf.fh_ghyper, gpcf, x, y, g, gdata, gprior);
                    i1 = i1+1;
                    i2 = 1;
                    
                    % Evaluate the gradient with respect to magnSigma
                    gdata(i1) = 0.5*(sum(sum(siLa.*DKff{i2}',2)) - sum(sum(L.*(L'*DKff{i2}')')) - b*DKff{i2}*b');

                    % Evaluate the gradient with respect to lengthScale
                    for i2 = 2:length(DKff)
                        i1 = i1+1;
                        gdata(i1) = 0.5*(sum(sum(siLa.*DKff{i2}',2)) - sum(sum(L.*(L'*DKff{i2}')')) - b*DKff{i2}*b');
                    end
                end
            end
            if strcmp(param,'inducing') || strcmp(param,'hyper+inducing')
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
                if strcmp(param,'inducing') || strcmp(param,'hyper+inducing')
                    [gprior, DCff] = feval(gpcf.fh_ghyper, gpcf, x, y, g, gdata, gprior);
                    gdata(i1)= -0.5*DCff.*b*b';
                    gdata(i1)= gdata(i1) + 0.5*sum(idiagLa-sum(L.*L,2)).*DCff;                    
                end
            end
        end
        % likelihood parameters
        %--------------------------------------
        if strcmp(param,'likelih') || strcmp(param,'hyper+likelih')
            [Ef, Varf] = ep_pred(gp, x, y, x, param);                
            gdata_likelih = 0;
            likelih = gp.likelih;
            for k1 = 1:length(y)
                sigm2_i = Varf(k1) ;
                myy_i = Ef(k1);
                gdata_likelih = gdata_likelih - feval(likelih.fh_siteDeriv, likelih, y, k1, sigm2_i, myy_i);
            end
        end

        g = gdata + gprior;
    end

    switch param
      case 'inducing'
        g = gdata_ind;
      case 'hyper+inducing'
        g = [g gdata_ind];
      case 'hyper+likelih'
        g = [g gdata_likelih];
    end
end
