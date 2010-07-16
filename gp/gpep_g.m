function [g, gdata, gprior] = gpep_g(w, gp, x, y, varargin)
%GPEP_G   Evaluate gradient of EP's marginal log posterior estimate (GPEP_E)
%
%	Description
%	G = GPEP_G(W, GP, X, Y, OPTIONS) takes a full GP hyper-parameter 
%       vector W, data structure GP a matrix X of input vectors and a 
%       matrix Y of target vectors, and evaluates the gradient G of EP's 
%       marginal log posterior estimate (gpep_e). Each row of X 
%       corresponds to one input vector and each row of Y corresponds to 
%       one target vector. 
%
%	[G, GDATA, GPRIOR] = GPEP_G(GP, X, Y, OPTIONS) also returns 
%	separately the data and prior contributions to the gradient.
%    
%     OPTIONS is optional parameter-value pair
%       'z'    is optional observed quantity in triplet (x_i,y_i,z_i)
%              Some likelihoods may use this. For example, in case of 
%              Poisson likelihood we have z_i=E_i, that is, expected 
%              value for ith case. 
%
%	See also   
%       GPEP_E, EP_PRED

% Copyright (c) 2007-2010  Jarno Vanhatalo
% Copyright (c) 2010       Heikki Peura
    
% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.
        
    ip=inputParser;
    ip.FunctionName = 'GPEP_G';
    ip.addRequired('w', @(x) isvector(x) && isreal(x) && all(isfinite(x)));
    ip.addRequired('gp',@isstruct);
    ip.addRequired('x', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
    ip.addRequired('y', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
    ip.addParamValue('z', [], @(x) isreal(x) && all(isfinite(x(:))))
    ip.parse(w, gp, x, y, varargin{:});
    z=ip.Results.z;

    
    gp=gp_unpak(gp, w);       % unpak the parameters
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
        
        if issparse(C)          % If compact support covariance functions are used 
                                % the covariance matrix will be sparse
            [e, edata, eprior, tautilde, nutilde, LD, La2, b] = gpep_e(w, gp, x, y, 'z', z);
            Stildesqroot = sparse(1:n,1:n,sqrt(tautilde),n,n);
            
            b = nutilde - Stildesqroot*ldlsolve(LD,Stildesqroot*(C*nutilde));
            invC = spinv(LD,1);       % evaluate the sparse inverse
            invC = Stildesqroot*invC*Stildesqroot;
        else
            [e, edata, eprior, tautilde, nutilde, L, La2, b] = gpep_e(w, gp, x, y, 'z', z);
            
            
            if tautilde > 0             % This is the usual case where likelihood is log concave
                                        % for example, Poisson and probit
                % Stildesqroot=diag(sqrt(tautilde));
                % logZep; nutilde; tautilde;
                % b=nutilde-Stildesqroot*(L'\(L\(Stildesqroot*(C*nutilde))));
                % invC = Stildesqroot*(L'\(L\Stildesqroot));
                Stildesqroot=sqrt(tautilde);
                temp=L\diag(Stildesqroot);
                invC = temp'*temp;
                b=nutilde-Stildesqroot.*(L'\(L\(Stildesqroot.*(C*nutilde))));
            else                         % We might end up here if the likelihood is not log concace
                                         % For example Student-t likelihood.
                                         % NOTE! This does not work reliably yet
                S = diag(tautilde);
                b = nutilde - tautilde.*(L'*L*(nutilde));
                invC = S*L';
                invC = S - invC*invC';
            end           
            
            if tautilde > 0             % This is the usual case where likelihood is log concave
                                        % for example, Poisson and probit
                Stildesqroot=diag(sqrt(tautilde));
                
                % logZep; nutilde; tautilde;
                b=nutilde-Stildesqroot*(L'\(L\(Stildesqroot*(C*nutilde))));
                invC = Stildesqroot*(L'\(L\Stildesqroot));
            else                         % We might end up here if the likelihood is not log concace
                                         % For example Student-t likelihood. 
                                         % NOTE! This does not work reliably yet
                S = diag(tautilde);
                b = nutilde - tautilde.*(L'*L*(nutilde));
                invC = S*L';
                invC = S - invC*invC';
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
                [DKff, gprior_cf] = feval(gpcf.fh_ghyper, gpcf, x);
                
                
                for i2 = 1:length(DKff)
                    i1 = i1+1;                
                    Bdl = b'*(DKff{i2}*b);
                    Cdl = sum(sum(invC.*DKff{i2})); % help arguments for lengthScale
                    gdata(i1)=0.5.*(Cdl - Bdl);
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
                    i1=0;
                    if ~isempty(gprior)
                        i1 = length(gprior);
                    end
                    
                    noise = gp.noise{i};
                    [DCff, gprior_cf] = feval(noise.fh_ghyper, noise, x);
                    
                    for i2 = 1:length(DCff)
                        i1 = i1+1;
                        
                        B = trace(invC);
                        C=b'*b;    
                        gdata(i1)=0.5.*DCff{i2}.*(B - C); 
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
        if ~isempty(strfind(gp.infer_params, 'likelihood')) && isfield(gp.likelih, 'fh_siteDeriv')
            [Ef, Varf] = ep_pred(gp, x, y, x, 'z', z);
            
            sigm2_i = (Varf.^-1 - tautilde).^-1;
            mu_i = sigm2_i.*(Ef./Varf - nutilde);
            
            gdata_likelih = 0;
            likelih = gp.likelih;
            for k1 = 1:length(y)
                gdata_likelih = gdata_likelih - feval(likelih.fh_siteDeriv, likelih, y, k1, sigm2_i(k1), mu_i(k1), z);
            end
            % evaluate prior contribution for the gradient
            if isfield(gp.likelih, 'p')
                g_logPrior = feval(likelih.fh_priorg, likelih);
            else
                g_logPrior = zeros(size(gdata_likelih));
            end
            % set the gradients into vectors that will be returned
            gdata = [gdata gdata_likelih];
            gprior = [gprior g_logPrior];
            i1 = length(gdata);
        end
                
        % ============================================================
        % FIC
        % ============================================================
      case {'FIC'}
        g_ind = zeros(1,numel(gp.X_u));
        gdata_ind = zeros(1,numel(gp.X_u));
        gprior_ind = zeros(1,numel(gp.X_u));
        
        u = gp.X_u;
        DKuu_u = 0;
        DKuf_u = 0;

        [e, edata, eprior, tautilde, nutilde, L, La, b] = gpep_e(w, gp, x, y, 'z', z);

        K_fu = gp_cov(gp, x, u);         % f x u
        K_uu = gp_trcov(gp, u);          % u x u, noiseles covariance K_uu
        K_uu = (K_uu+K_uu')./2;          % ensure the symmetry of K_uu        
        iKuuKuf = K_uu\K_fu';
        
        LL = sum(L.*L,2);
        
        % =================================================================
        % Gradient with respect to covariance function parameters
        if ~isempty(strfind(gp.infer_params, 'covariance'))
            for i=1:ncf            
                i1=0;
                if ~isempty(gprior)
                    i1 = length(gprior);
                end
                
                gpcf = gp.cf{i};
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
                    gdata(i1) = gdata(i1) + 0.5.*(sum(DKff{i2}./La) - sum(LL.*DKff{i2}));
                    gdata(i1) = gdata(i1) + 0.5.*(2.*sum(LL.*sum(DKuf{i2}'.*iKuuKuf',2)) - sum(LL.*sum(KfuiKuuKuu.*iKuuKuf',2)));

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
                    gpcf = gp.noise{i};
                    
                    [DCff, gprior_cf] = feval(gpcf.fh_ghyper, gpcf, x);
                    for i2 = 1:length(DCff)
                        i1 = i1+1;
                        gdata(i1)= -0.5*DCff{i2}.*b*b';
                        gdata(i1)= gdata(i1) + 0.5*sum(1./La-LL).*DCff{i2};
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
                        
                        KfuiKuuKuu = iKuuKuf'*DKuu{i2};
                        
                        gdata(i1) = gdata(i1) - 0.5.*((2*b*DKuf{i2}'-(b*KfuiKuuKuu))*(iKuuKuf*b') + ...
                                                          2.*sum(sum(L'.*(L'*DKuf{i2}'*iKuuKuf))) - sum(sum(L'.*((L'*KfuiKuuKuu)*iKuuKuf))));
                        gdata(i1) = gdata(i1) + 0.5.*(2.*b.*sum(DKuf{i2}'.*iKuuKuf',2)'*b'- b.*sum(KfuiKuuKuu.*iKuuKuf',2)'*b');
                        gdata(i1) = gdata(i1) + 0.5.*(2.*sum(LL.*sum(DKuf{i2}'.*iKuuKuf',2)) - ...
                                                      sum(LL.*sum(KfuiKuuKuu.*iKuuKuf',2)));
                    end
                end
            end
        end
        
        % =================================================================
        % Gradient with respect to a likelihood function parameters        
        if ~isempty(strfind(gp.infer_params, 'likelihood')) && isfield(gp.likelih, 'fh_siteDeriv')
            [Ef, Varf] = ep_pred(gp, x, y, x, 'tstind', 1:n, 'z', z);
            sigm2_i = (Varf.^-1 - tautilde).^-1;
            mu_i = sigm2_i.*(Ef./Varf - nutilde);
            
            gdata_likelih = 0;
            likelih = gp.likelih;
            for k1 = 1:length(y)
                gdata_likelih = gdata_likelih - feval(likelih.fh_siteDeriv, likelih, y, k1, sigm2_i(k1), mu_i(k1), z);
            end

            % evaluate prior contribution for the gradient
            if isfield(gp.likelih, 'p')
                g_logPrior = feval(likelih.fh_priorg, likelih);
            else
                g_logPrior = zeros(size(gdata_likelih));
            end
            % set the gradients into vectors that will be returned
            gdata = [gdata gdata_likelih];
            gprior = [gprior g_logPrior];
            i1 = length(gdata);
        end
        % ============================================================
        % PIC
        % ============================================================
      case {'PIC' 'PIC_BLOCK'}
        g_ind = zeros(1,numel(gp.X_u));
        gdata_ind = zeros(1,numel(gp.X_u));
        gprior_ind = zeros(1,numel(gp.X_u));
        
        u = gp.X_u;
        ind = gp.tr_index;
        DKuu_u = 0;
        DKuf_u = 0;

        [e, edata, eprior, tautilde, nutilde, L, La, b] = gpep_e(w, gp, x, y, 'z', z);

        K_fu = gp_cov(gp, x, u);         % f x u
        K_uu = gp_trcov(gp, u);          % u x u, noiseles covariance K_uu
        K_uu = (K_uu+K_uu')./2;          % ensure the symmetry of K_uu        
        iKuuKuf = K_uu\K_fu';
        
        % =================================================================
        % Gradient with respect to covariance function parameters
        if ~isempty(strfind(gp.infer_params, 'covariance'))
            
            % Evaluate the gradients from covariance functions
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
                    
                    for kk=1:length(ind)
                        gdata(i1) = gdata(i1) ...
                            + 0.5.*(-b(ind{kk})*DKff{kk}{i2}*b(ind{kk})' ...
                                    + 2.*b(ind{kk})*DKuf{i2}(:,ind{kk})'*iKuuKuf(:,ind{kk})*b(ind{kk})'- ...
                                    b(ind{kk})*KfuiKuuKuu(ind{kk},:)*iKuuKuf(:,ind{kk})*b(ind{kk})' ...
                                    + trace(La{kk}\DKff{kk}{i2})...
                                    - trace(L(ind{kk},:)*(L(ind{kk},:)'*DKff{kk}{i2})) ...
                                    + 2.*sum(sum(L(ind{kk},:)'.*(L(ind{kk},:)'*DKuf{i2}(:,ind{kk})'*iKuuKuf(:,ind{kk})))) - ...
                                    sum(sum(L(ind{kk},:)'.*((L(ind{kk},:)'*KfuiKuuKuu(ind{kk},:))*iKuuKuf(:,ind{kk})))));                
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
                        KfuiKuuDKuu_u = iKuuKuf'*DKuu{i2};
                        
                        gdata(i1) = gdata(i1) - 0.5.*((2*b*DKuf{i2}'-(b*KfuiKuuDKuu_u))*(iKuuKuf*b') + 2.*sum(sum(L'.*((L'*DKuf{i2}')*iKuuKuf))) - ...
                                                      sum(sum(L'.*((L'*KfuiKuuDKuu_u)*iKuuKuf))));
                        
                        for kk=1:length(ind)
                            gdata(i1) = gdata(i1) + 0.5.*(2.*b(ind{kk})*DKuf{i2}(:,ind{kk})'*iKuuKuf(:,ind{kk})*b(ind{kk})'- ...
                                                          b(ind{kk})*KfuiKuuDKuu_u(ind{kk},:)*iKuuKuf(:,ind{kk})*b(ind{kk})' ...
                                                          + 2.*sum(sum(L(ind{kk},:)'.*(L(ind{kk},:)'*DKuf{i2}(:,ind{kk})'*iKuuKuf(:,ind{kk})))) - ...
                                                          sum(sum(L(ind{kk},:)'.*((L(ind{kk},:)'*KfuiKuuDKuu_u(ind{kk},:))*iKuuKuf(:,ind{kk})))));
                        end
                    end
                end
            end
        end
        
        % =================================================================
        % Gradient with respect to likelihood function parameters
        
        if ~isempty(strfind(gp.infer_params, 'likelihood')) && isfield(gp.likelih, 'fh_siteDeriv')

            [Ef, Varf] = ep_pred(gp, x, y, x, 'tstind', gp.tr_index, 'z', z);
            
            sigm2_i = (Varf.^-1 - tautilde).^-1;
            mu_i = sigm2_i.*(Ef./Varf - nutilde);
            
            gdata_likelih = 0;
            likelih = gp.likelih;
            for k1 = 1:length(y)
                gdata_likelih = gdata_likelih - feval(likelih.fh_siteDeriv, likelih, y, k1, sigm2_i(k1), mu_i(k1), z);
            end

            % evaluate prior contribution for the gradient
            if isfield(gp.likelih, 'p')
                g_logPrior = feval(likelih.fh_priorg, likelih);
            else
                g_logPrior = zeros(size(gdata_likelih));
            end
            % set the gradients into vectors that will be returned
            gdata = [gdata gdata_likelih];
            gprior = [gprior g_logPrior];
            i1 = length(gdata);
        end
        

        % ============================================================
        % CS+FIC
        % ============================================================
     case {'CS+FIC'}
        g_ind = zeros(1,numel(gp.X_u));
        gdata_ind = zeros(1,numel(gp.X_u));
        gprior_ind = zeros(1,numel(gp.X_u));

        u = gp.X_u;
        DKuu_u = 0;
        DKuf_u = 0;

        [e, edata, eprior, tautilde, nutilde, L, La, b] = gpep_e(w, gp, x, y, 'z', z);

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
        siLa = spinv(LD,1);
% $$$         siLa = spinv(La);
        idiagLa = diag(siLa);
        LL = sum(L.*L,2);
        
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

                        KfuiKuuKuu = iKuuKuf'*DKuu{i2};
                        gdata(i1) = -0.5.*((2*b*DKuf{i2}'-(b*KfuiKuuKuu))*(iKuuKuf*b') + 2.*sum(sum(L'.*(L'*DKuf{i2}'*iKuuKuf))) - ...
                                           sum(sum(L'.*((L'*KfuiKuuKuu)*iKuuKuf))));
                        
                        gdata(i1) = gdata(i1) - 0.5.*(b.*DKff{i2}')*b';
                        gdata(i1) = gdata(i1) + 0.5.*(2.*b.*sum(DKuf{i2}'.*iKuuKuf',2)'*b'- b.*sum(KfuiKuuKuu.*iKuuKuf',2)'*b');
                        gdata(i1) = gdata(i1) + 0.5.*sum(idiagLa.*DKff{i2} - LL.*DKff{i2});   % corrected
                        gdata(i1) = gdata(i1) + 0.5.*(2.*sum(LL.*sum(DKuf{i2}'.*iKuuKuf',2)) - sum(LL.*sum(KfuiKuuKuu.*iKuuKuf',2)));
                        
                        %gdata(i1) = gdata(i1) + 0.5.*sum(sum(La\((2.*K_uf') - KfuiKuuKuu).*iKuuKuf',2));
                        gdata(i1) = gdata(i1) + 0.5.*sum(sum(ldlsolve(LD,2.*DKuf{i2}' - KfuiKuuKuu).*iKuuKuf',2));
                        gdata(i1) = gdata(i1) - 0.5.*( idiagLa'*(sum((2.*DKuf{i2}' - KfuiKuuKuu).*iKuuKuf',2)) ); % corrected                
                        gprior(i1) = gprior_cf(i2);                    
                    end                        

                % Evaluate the gradient for compact support covariance functions
                else
                    % Get the gradients of the covariance matrices 
                    % and gprior from gpcf_* structures
                    [DKff,gprior_cf] = feval(gpcf.fh_ghyper, gpcf, x);
                    
                    for i2 = 1:length(DKff)
                        i1 = i1+1;
                        gdata(i1) = 0.5*(sum(sum(siLa.*DKff{i2}',2)) - sum(sum(L.*(L'*DKff{i2}')')) - b*DKff{i2}*b');
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
                        gdata(i1)= gdata(i1) + 0.5*sum(idiagLa-LL.*DCff{i2});
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
                        [DKuu, gprior_ind] = feval(gpcf.fh_ginput, gpcf, u);
                        [DKuf] = feval(gpcf.fh_ginput, gpcf, u, x);
                        
                        for i2 = 1:length(DKuu)
                            i1 = i1 + 1;
                            KfuiKuuKuu = iKuuKuf'*DKuu{i2};
                            
                            gdata(i1) = gdata(i1) - 0.5.*((2*b*DKuf{i2}'-(b*KfuiKuuKuu))*(iKuuKuf*b') + ...
                                                          2.*sum(sum(L'.*(L'*DKuf{i2}'*iKuuKuf))) - sum(sum(L'.*((L'*KfuiKuuKuu)*iKuuKuf))));
                            gdata(i1) = gdata(i1) + 0.5.*(2.*b.*sum(DKuf{i2}'.*iKuuKuf',2)'*b'- b.*sum(KfuiKuuKuu.*iKuuKuf',2)'*b');
                            gdata(i1) = gdata(i1) + 0.5.*(2.*sum(LL.*sum(DKuf{i2}'.*iKuuKuf',2)) - ...
                                                          sum(LL.*sum(KfuiKuuKuu.*iKuuKuf',2)));
                            
                            gdata(i1) = gdata(i1) + 0.5.*sum(sum(ldlsolve(LD,(2.*DKuf{i2}') - KfuiKuuKuu).*iKuuKuf',2));
                            gdata(i1) = gdata(i1) - 0.5.*( idiagLa'*(sum((2.*DKuf{i2}' - KfuiKuuKuu).*iKuuKuf',2)) ); % corrected
                            gprior(i1) = gprior_ind(i2);
                        end
                    end
                end
            end
        end
        

        % =================================================================
        % Gradient with respect to likelihood function parameters
        
        if ~isempty(strfind(gp.infer_params, 'likelihood')) && isfield(gp.likelih, 'fh_siteDeriv')
            [Ef, Varf] = ep_pred(gp, x, y, x, 'tstind', 1:n, 'z', z);
            sigm2_i = (Varf.^-1 - tautilde).^-1;
            mu_i = sigm2_i.*(Ef./Varf - nutilde);
            
            gdata_likelih = 0;
            likelih = gp.likelih;
            for k1 = 1:length(y)
                gdata_likelih = gdata_likelih - feval(likelih.fh_siteDeriv, likelih, y, k1, sigm2_i(k1), mu_i(k1), z);
            end
            % evaluate prior contribution for the gradient
            if isfield(gp.likelih, 'p')
                g_logPrior = feval(likelih.fh_priorg, likelih);
            else
                g_logPrior = zeros(size(gdata_likelih));
            end
            % set the gradients into vectors that will be returned
            gdata = [gdata gdata_likelih];
            gprior = [gprior g_logPrior];
            i1 = length(gdata);
        end
        
        % ============================================================
        % DTC/VAR
        % ============================================================        
      case {'DTC' 'SOR'}
        g_ind = zeros(1,numel(gp.X_u));
        gdata_ind = zeros(1,numel(gp.X_u));
        gprior_ind = zeros(1,numel(gp.X_u));
        
        u = gp.X_u;
        DKuu_u = 0;
        DKuf_u = 0;

        [e, edata, eprior, tautilde, nutilde, L, La, b] = gpep_e(w, gp, x, y, 'z', z);

        K_fu = gp_cov(gp, x, u);         % f x u
        K_uu = gp_trcov(gp, u);          % u x u, noiseles covariance K_uu
        K_uu = (K_uu+K_uu')./2;          % ensure the symmetry of K_uu        
        iKuuKuf = K_uu\K_fu';
        
        LL = sum(L.*L,2);
        iLav=1./La;
        
        % =================================================================
        % Gradient with respect to covariance function parameters
        if ~isempty(strfind(gp.infer_params, 'covariance'))
            for i=1:ncf            
                i1=0;
                if ~isempty(gprior)
                    i1 = length(gprior);
                end
                
                gpcf = gp.cf{i};
                % Get the gradients of the covariance matrices 
                % and gprior from gpcf_* structures
                gpcf = gp.cf{i};
                [DKff, gprior_cf] = feval(gpcf.fh_ghyper, gpcf, x, [], 1); 
                DKuu = feval(gpcf.fh_ghyper, gpcf, u); 
                DKuf = feval(gpcf.fh_ghyper, gpcf, u, x); 
                
                for i2 = 1:length(DKuu)
                    i1 = i1+1;
                    
                    KfuiKuuKuu = iKuuKuf'*DKuu{i2};
                    gdata(i1) = -0.5.*((2*b*DKuf{i2}'-(b*KfuiKuuKuu))*(iKuuKuf*b'));
                    gdata(i1) = gdata(i1) + 0.5.*(2.*(sum(iLav'*sum(DKuf{i2}'.*iKuuKuf',2))-sum(sum(L'.*(L'*DKuf{i2}'*iKuuKuf))))...
                    - sum(iLav'*sum(KfuiKuuKuu.*iKuuKuf',2))+ sum(sum(L'.*((L'*KfuiKuuKuu)*iKuuKuf))));
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
                    gpcf = gp.noise{i};
                    
                    [DCff, gprior_cf] = feval(gpcf.fh_ghyper, gpcf, x);
                    for i2 = 1:length(DCff)
                        i1 = i1+1;
                        gdata(i1)= -0.5*DCff{i2}.*b*b';
                        gdata(i1)= gdata(i1) + 0.5*sum(1./La-LL).*DCff{i2};
                        if strcmp(gp.type, 'VAR')
                            gdata(i1)= gdata(i1) + 0.5*(sum((Kv_ff-Qv_ff)./La));
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
                
                for i=1:ncf
                    i1=st;
                    
                    gpcf = gp.cf{i};
                    DKuu = feval(gpcf.fh_ginput, gpcf, u);
                    DKuf = feval(gpcf.fh_ginput, gpcf, u, x);
                
                    for i2 = 1:length(DKuu)
                        i1 = i1+1;
                        
                        KfuiKuuKuu = iKuuKuf'*DKuu{i2};
                        gdata(i1) = gdata(i1) - 0.5.*((2*b*DKuf{i2}'-(b*KfuiKuuKuu))*(iKuuKuf*b'));
                        gdata(i1) = gdata(i1) + 0.5.*(2.*(sum(iLav'*sum(DKuf{i2}'.*iKuuKuf',2))-sum(sum(L'.*(L'*DKuf{i2}'*iKuuKuf))))...
                        - sum(iLav'*sum(KfuiKuuKuu.*iKuuKuf',2))+ sum(sum(L'.*((L'*KfuiKuuKuu)*iKuuKuf))));

                        if strcmp(gp.type, 'VAR')
                            gdata(i1) = gdata(i1) + 0.5.*(0-2.*sum(iLav'*sum(DKuf{i2}'.*iKuuKuf',2)) + ...
                                sum(iLav'*sum(KfuiKuuKuu.*iKuuKuf',2)));
                        end
                    end
                end
            end
        end
        
        % =================================================================
        % Gradient with respect to likelihood function parameters
        
        if ~isempty(strfind(gp.infer_params, 'likelihood')) && isfield(gp.likelih, 'fh_siteDeriv')
            [Ef, Varf] = ep_pred(gp, x, y, x, 'tstind', 1:n, 'z', z);
            gdata_likelih = 0;
            likelih = gp.likelih;
            for k1 = 1:length(y)
                sigm2_i = Varf(k1) ;
                myy_i = Ef(k1);
                gdata_likelih = gdata_likelih - feval(likelih.fh_siteDeriv, likelih, y, k1, sigm2_i, myy_i, z);
            end
            % evaluate prior contribution for the gradient
            if isfield(gp.likelih, 'p')
                g_logPrior = feval(likelih.fh_priorg, likelih);
            else
                g_logPrior = zeros(size(gdata_likelih));
            end
            % set the gradients into vectors that will be returned
            gdata = [gdata gdata_likelih];
            gprior = [gprior g_logPrior];
            i1 = length(gdata);
        end
        
        % ============================================================
        % VAR
        % ============================================================        
        
        % NOTE! Not properly implemented as no analytical result has been
        % derived. Not suitable for large data sets.
        
       case 'VAR'
           epsilon = 1.0e-6;
           
           func = fcnchk(@gpep_e, 3);
           %grad = fcnchk(grad, length(varargin));
           
           % Treat
           nparams = length(w);
           deltaf = zeros(1, nparams);
           step = zeros(1, nparams);
           for i = 1:nparams
               % Move a small way in the ith coordinate of w
               step(i) = 1.0;
               func = fcnchk(func, 3);
               fplus = feval(func, w+epsilon.*step, gp,x,y);
               fminus = feval(func, w-epsilon.*step, gp,x,y);
               %   fplus  = feval('linef_test', epsilon, func, w, step, varargin{:});
               %   fminus = feval('linef_test', -epsilon, func, w, step, varargin{:});
               % Use central difference formula for approximation
               deltaf(i) = 0.5*(fplus - fminus)/epsilon;
               step(i) = 0.0;
           end
           % gradient = feval(grad, w, varargin{:});
           % fprintf(1, 'Checking gradient ...\n\n');
           % fprintf(1, '   analytic   diffs     delta\n\n');
           % disp([gradient', deltaf', gradient' - deltaf'])
           
           %delta = gradient' - deltaf';
           gdata=deltaf;
           
           %gdata=numgrad_test(gp_pak(gp), @gpep_e, gp, x, y);
           gprior=0;

        % ============================================================
        % SSGP
        % ============================================================        
      case 'SSGP'        
        
        [e, edata, eprior, tautilde, nutilde, L, S, b] = gpep_e(w, gp, x, y, 'z', z);

        Phi = gp_trcov(gp, x);         % f x u
        m = size(Phi,2);

        SPhi = repmat(S,1,m).*Phi;                        
        % =================================================================
        % Evaluate the gradients from covariance functions
        for i=1:ncf            
            i1=0;
            if ~isempty(gprior)
                i1 = length(gprior);
            end
            
            gpcf = gp.cf{i};
            gpcf.GPtype = gp.type;
            % Covariance function hyperparameters
            %--------------------------------------
            if ~isempty(strfind(gp.infer_params, 'covariance'))
                % Get the gradients of the covariance matrices 
                % and gprior from gpcf_* structures
                [gprior, DKff] = feval(gpcf.fh_ghyper, gpcf, x, y, g, gdata, gprior); 
                i1 = i1+1;
                i2 = 1;
                
                % Evaluate the gradient with respect to magnSigma
                SDPhi = repmat(S,1,m).*DKff{i2};                
                
                gdata(i1) = 0.5*( sum(sum(SDPhi.*Phi,2)) + sum(sum(SPhi.*DKff{i2},2)) );
                gdata(i1) = gdata(i1) - 0.5*( sum(sum(L'.*(L'*DKff{i2}*Phi' + L'*Phi*DKff{i2}'),1)) );
                gdata(i1) = gdata(i1) - 0.5*(b*DKff{i2}*Phi' + b*Phi*DKff{i2}')*b';
                
                
                if isfield(gpcf.p.lengthScale, 'p') && ~isempty(gpcf.p.lengthScale.p)
                    i1 = i1+1;
                    if any(strcmp(fieldnames(gpcf.p.lengthScale.p),'nu'))
                        i1 = i1+1;
                    end
                end

                % Evaluate the gradient with respect to lengthScale
                for i2 = 2:length(DKff)
                    i1 = i1+1;
                    SDPhi = repmat(S,1,m).*DKff{i2};
                
                    gdata(i1) = 0.5*( sum(sum(SDPhi.*Phi,2)) + sum(sum(SPhi.*DKff{i2},2)) );
                    gdata(i1) = gdata(i1) - 0.5*( sum(sum(L'.*(L'*DKff{i2}*Phi' + L'*Phi*DKff{i2}'),1)) );
                    gdata(i1) = gdata(i1) - 0.5*(b*DKff{i2}*Phi' + b*Phi*DKff{i2}')*b';
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
                if ~isempty(strfind(gp.infer_params, 'covariance'))
                    [gprior, DCff] = feval(gpcf.fh_ghyper, gpcf, x, y, g, gdata, gprior);
                    gdata(i1)= -0.5*DCff.*b*b';
                    gdata(i1)= gdata(i1) + 0.5*sum(1./La-sum(L.*L,2)).*DCff;
                end
            end
        end
        
        % likelihood parameters
        %--------------------------------------
        if ~isempty(strfind(gp.infer_params, 'likelih')) && isfield(gp.likelih, 'fh_siteDeriv')
            [Ef, Varf] = ep_pred(gp, x, y, x, param);
            gdata_likelih = 0;
            likelih = gp.likelih;
            for k1 = 1:length(y)
                sigm2_i = Varf(k1) ;
                myy_i = Ef(k1);
                gdata_likelih = gdata_likelih + feval(likelih.fh_siteDeriv, likelih, y, k1, sigm2_i, myy_i, z);
            end
            % evaluate prior contribution for the gradient
            if isfield(gp.likelih, 'p')
                g_logPrior = feval(likelih.fh_priorg, likelih);
            else
                g_logPrior = zeros(size(gdata_likelih));
            end
            % set the gradients into vectors that will be returned
            gdata = [gdata gdata_likelih];
            gprior = [gprior g_logPrior];
            i1 = length(gdata);
        end
        
     
    end
    
    g = gdata + gprior;
   
end
