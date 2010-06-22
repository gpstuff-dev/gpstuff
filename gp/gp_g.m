function [g, gdata, gprior] = gp_g(w, gp, x, y, varargin)
%GP_G   Evaluate gradient of energy (GP_E) for Gaussian Process
%
%	Description
%	G = GP_G(W, GP, X, Y, OPTIONS) takes a full GP hyper-parameter
%        vector W, data structure GP a matrix X of input vectors
%        and a matrix Y of target vectors, and evaluates the
%        gradient G of the energy function (gp_e). Each row of X
%        corresponds to one input vector and each row of Y NOTE! 
%        This parametrization works only for full GP!
%
%	[G, GDATA, GPRIOR] = GP_G(W, GP, X, Y, OPTIONS) also returns
%        separately the data and prior contributions to the gradient.
%
%     OPTIONS is optional parameter-value pair
%       No applicable options
%
%	See also
%       GP_E, GP_PAK, GP_UNPAK, GPCF_*

% Copyright (c) 2007-2010 Jarno Vanhatalo
% Copyright (c) 2010 Aki Vehtari
% Copyright (c) 2010 Heikki Peura

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.

ip=inputParser;
ip.FunctionName = 'GP_E';
ip.addRequired('w', @(x) isvector(x) && isreal(x) && all(isfinite(x)));
ip.addRequired('gp',@isstruct);
ip.addRequired('x', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addRequired('y', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.parse(w, gp, x, y, varargin{:});

gp=gp_unpak(gp, w);       % unpak the parameters
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
        invC = spinv(C);       % evaluate the sparse inverse
        LD = ldlchol(C);
        b = ldlsolve(LD,y);
    else
        invC = inv(C);        % evaluate the full inverse
        b = C\y;
    end

    % =================================================================
    % Gradient with respect to covariance function parameters
    if ~isempty(strfind(gp.infer_params, 'covariance'))
        for i=1:ncf
            i1=0;
            if ~isempty(gprior)
                i1 = length(gprior);
            end
            
            gpcf = gp.cf{i};
            [DKff, gprior_cf] = feval(gpcf.fh_ghyper, gpcf, x);
            
            % Evaluate the gradient with respect to covariance function parameters
            for i2 = 1:length(DKff)
                i1 = i1+1;  
                Bdl = b'*(DKff{i2}*b);
                Cdl = sum(sum(invC.*DKff{i2})); % help arguments
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
                noise = gp.noise{i};
                [DCff, gprior_cf] = feval(noise.fh_ghyper, noise, x);
                
                for i2 = 1:length(DCff)
                    i1 = i1+1;
                    if size(DCff{i2}) > 1
                        Bdl = b'*(DCff{i2}*b);
                        Cdl = sum(sum(invC.*DCff{i2})); % help arguments
                        gdata(i1)=0.5.*(Cdl - Bdl);
                    else
                        B = trace(invC);
                        C=b'*b;
                        gdata(i1)=0.5.*DCff{i2}.*(B - C); 
                    end
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
    Luu = chol(K_uu,'lower');
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
    A = chol(A,'upper');
    L = iLaKfu/A;
    b = y'./Lav' - (y'*L)*L';
    iKuuKuf = Luu'\(Luu\K_fu');
    La = Lav;
    LL = sum(L.*L,2);
    
    % =================================================================
    % Gradient with respect to covariance function parameters
    if ~isempty(strfind(gp.infer_params, 'covariance'))    
        % Loop over the covariance functions
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
        
        % =================================================================
        % Gradient with respect to noise function parameters
        
        % Loop over the noise functions
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
                    gdata(i1)= gdata(i1) + 0.5*sum(DCff{i2}./La-sum(L.*L,2).*DCff{i2});
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
            
            % Loop over the covariance functions
            for i=1:ncf
                i1 = st;
                gpcf = gp.cf{i};
                DKuu = feval(gpcf.fh_ginput, gpcf, u);
                DKuf = feval(gpcf.fh_ginput, gpcf, u, x);
                
                for i2 = 1:length(DKuu)
                    i1=i1+1;
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
    
    g = gdata + gprior;
    
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

    % First evaluate the needed covariance matrices
    % if they are not in the memory
    % v defines that parameter is a vector
    K_fu = gp_cov(gp, x, u);         % f x u
    K_uu = gp_trcov(gp, u);          % u x u, noiseles covariance K_uu
    K_uu = (K_uu+K_uu')./2;          % ensure the symmetry of K_uu
    Luu = chol(K_uu,'lower');
    % Evaluate the Lambda (La)
    % Q_ff = K_fu*inv(K_uu)*K_fu'
    % Here we need only the diag(Q_ff), which is evaluated below
    %B=K_fu/Luu;
    B=Luu\K_fu';
    iLaKfu = zeros(size(K_fu));  % f x u
    for i=1:length(ind)
        Qbl_ff = B(:,ind{i})'*B(:,ind{i});
        [Kbl_ff, Cbl_ff] = gp_trcov(gp, x(ind{i},:));
        la = Cbl_ff - Qbl_ff;
        La{i} = (la + la')./2;
        iLaKfu(ind{i},:) = La{i}\K_fu(ind{i},:);
    end
    % ... then evaluate some help matrices.
    % A = chol(K_uu+K_uf*inv(La)*K_fu))
    A = K_uu+K_fu'*iLaKfu;
    A = (A+A')./2;            % Ensure symmetry

    L = iLaKfu/chol(A,'upper');
    b = zeros(1,n);
    b_apu=(y'*L)*L';
    for i=1:length(ind)
        b(ind{i}) = y(ind{i})'/La{i} - b_apu(ind{i});
    end
    iKuuKuf = Luu'\(Luu\K_fu');
    
    % =================================================================
    % Gradient with respect to covariance function parameters

    if ~isempty(strfind(gp.infer_params, 'covariance'))
        % Loop over the  covariance functions
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
                                +trace(La{kk}\DKff{kk}{i2})...                                
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
        
    
        % =================================================================
        % Gradient with respect to noise function parameters
        
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
            
            % Loop over the  covariance functions
            for i=1:ncf            
                i1=st;
                gpcf = gp.cf{i};
                DKuu = feval(gpcf.fh_ginput, gpcf, u);
                DKuf = feval(gpcf.fh_ginput, gpcf, u, x);
                
                for i2 = 1:length(DKuu)
                    i1 = i1+1;
                    KfuiKuuDKuu_u = iKuuKuf'*DKuu{i2};                
                    gdata(i1) = gdata(i1) -0.5.*((2*b*DKuf{i2}'-(b*KfuiKuuDKuu_u))*(iKuuKuf*b') + 2.*sum(sum(L'.*((L'*DKuf{i2}')*iKuuKuf))) - ...
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
    Luu = chol(K_uu,'lower');
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
    L = iLaKfu/chol(A,'upper');
    %b = y'/La - (y'*L)*L';
    b = ldlsolve(LD,y)' - (y'*L)*L';
    
    siLa = spinv(La);
    idiagLa = diag(siLa);
    iKuuKuf = K_uu\K_fu';
    LL = sum(L.*L,2);
    
    % =================================================================
    % Gradient with respect to covariance function parameters
    if ~isempty(strfind(gp.infer_params, 'covariance'))
        % Loop over covariance functions 
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
                    
                    temp1 = sum(KfuiKuuKuu.*iKuuKuf',2);
                    temp2 = sum(DKuf{i2}'.*iKuuKuf',2);
                    temp3 = 2.*DKuf{i2}' - KfuiKuuKuu;
                    gdata(i1) = gdata(i1) - 0.5.*(b.*DKff{i2}')*b';
                    gdata(i1) = gdata(i1) + 0.5.*(2.*b.*temp2'*b'- b.*temp1'*b');
                    gdata(i1) = gdata(i1) + 0.5.*(sum(idiagLa.*DKff{i2} - LL.*DKff{i2}));   % corrected
                    gdata(i1) = gdata(i1) + 0.5.*(2.*sum(LL.*temp2) - sum(LL.*temp1));
                    
                    %gdata(i1) = gdata(i1) + 0.5.*sum(sum(La\((2.*K_uf') - KfuiKuuKuu).*iKuuKuf',2));
                    gdata(i1) = gdata(i1) + 0.5.*sum(sum(ldlsolve(LD,temp3).*iKuuKuf',2));
                    gdata(i1) = gdata(i1) - 0.5.*( idiagLa'*(sum(temp3.*iKuuKuf',2)) ); % corrected                
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
        
        % =================================================================
        % Gradient with respect to noise function parameters
        
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
                        i1 = i1+1;
                        KfuiKuuKuu = iKuuKuf'*DKuu{i2};
                        
                        gdata(i1) = gdata(i1) - 0.5.*((2*b*DKuf{i2}'-(b*KfuiKuuKuu))*(iKuuKuf*b') + ...
                                                      2.*sum(sum(L'.*(L'*DKuf{i2}'*iKuuKuf))) - sum(sum(L'.*((L'*KfuiKuuKuu)*iKuuKuf))));
                        gdata(i1) = gdata(i1) + 0.5.*(2.*b.*sum(DKuf{i2}'.*iKuuKuf',2)'*b'- b.*sum(KfuiKuuKuu.*iKuuKuf',2)'*b');
                        gdata(i1) = gdata(i1) + 0.5.*(2.*sum(sum(L.*L,2).*sum(DKuf{i2}'.*iKuuKuf',2)) - ...
                                                      sum(sum(L.*L,2).*sum(KfuiKuuKuu.*iKuuKuf',2)));
                        
                        gdata(i1) = gdata(i1) + 0.5.*sum(sum(ldlsolve(LD,(2.*DKuf{i2}') - KfuiKuuKuu).*iKuuKuf',2));
                        gdata(i1) = gdata(i1) - 0.5.*( idiagLa'*(sum((2.*DKuf{i2}' - KfuiKuuKuu).*iKuuKuf',2)) ); % corrected
                    end
                end
            end
        end
    end
    
    g = gdata + gprior;
    
    % ============================================================
    % DTC/VAR
    % ============================================================
  case {'DTC' 'VAR' 'SOR'}
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
    Lav = Cv_ff-Kv_ff;   % 1 x f, Vector of diagonal elements
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
    b = y'./Lav' - (y'*L)*L';
    iKuuKuf = Luu'\(Luu\K_fu');
    La = Lav;
    LL = sum(L.*L,2);
    iLav=1./Lav;
    
    LL1=iLav-LL;
    
    % =================================================================
    
    if ~isempty(strfind(gp.infer_params, 'covariance'))
        % Loop over the covariance functions
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
                gdata(i1) = -0.5.*((2*b*DKuf{i2}'-(b*KfuiKuuKuu))*(iKuuKuf*b'));
                gdata(i1) = gdata(i1) + 0.5.*(2.*(sum(iLav'*sum(DKuf{i2}'.*iKuuKuf',2))-sum(sum(L'.*(L'*DKuf{i2}'*iKuuKuf))))...
                    - sum(iLav'*sum(KfuiKuuKuu.*iKuuKuf',2))+ sum(sum(L'.*((L'*KfuiKuuKuu)*iKuuKuf))));
                
                if strcmp(gp.type, 'VAR')
                    gdata(i1) = gdata(i1) + 0.5.*(sum(iLav.*DKff{i2})-2.*sum(iLav'*sum(DKuf{i2}'.*iKuuKuf',2)) + ...
                    sum(iLav'*sum(KfuiKuuKuu.*iKuuKuf',2))); % trace-term derivative
                end
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
        
        % Loop over the noise functions
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
                    gdata(i1)= gdata(i1) + 0.5*sum(DCff{i2}./La-sum(L.*L,2).*DCff{i2});
                    if strcmp(gp.type, 'VAR')
                        gdata(i1)= gdata(i1) + 0.5*(sum((Kv_ff-Qv_ff)./La));
                    end
                    
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
            
            % Loop over the covariance functions
            for i=1:ncf
                i1 = st;
                gpcf = gp.cf{i};
                DKuu = feval(gpcf.fh_ginput, gpcf, u);
                DKuf = feval(gpcf.fh_ginput, gpcf, u, x);
                
                for i2 = 1:length(DKuu)
                    i1=i1+1;
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
    
    g = gdata + gprior;  
    
    % ============================================================
    % SSGP
    % ============================================================
  case 'SSGP'        % Predictions with sparse spectral sampling approximation for GP
                     % The approximation is proposed by M. Lazaro-Gredilla, J. Quinonero-Candela and A. Figueiras-Vidal
                     % in Microsoft Research technical report MSR-TR-2007-152 (November 2007)
                     % NOTE! This does not work at the moment.

    % First evaluate the needed covariance matrices
    % v defines that parameter is a vector
    [Phi, S] = gp_trcov(gp, x);        % n x m and nxn sparse matrices
    Sv = diag(S);
    
    m = size(Phi,2);
    
    A = eye(m,m) + Phi'*(S\Phi);
    A = chol(A,'lower');
    L = (S\Phi)/A';

    b = y'./Sv' - (y'*L)*L';
    iSPhi = S\Phi;
    
    % =================================================================
    if strcmp(param,'hyper') || strcmp(param,'hyper+inducing')
        % Loop over the covariance functions
        for i=1:ncf
            i1=0;
            if ~isempty(gprior)
                i1 = length(gprior);
            end
            
            gpcf = gp.cf{i};
            
            
            % Get the gradients of the covariance matrices 
            % and gprior from gpcf_* structures
            [DKff, gprior_cf] = feval(gpcf.fh_ghyper, gpcf, x); 

            % Evaluate the gradient with respect to lengthScale
            for i2 = 1:length(DKff)
                i1 = i1+1;
                iSDPhi = S\DKff{i2};
                
                gdata(i1) = 0.5*( sum(sum(iSDPhi.*Phi,2)) + sum(sum(iSPhi.*DKff{i2},2)) );
                gdata(i1) = gdata(i1) - 0.5*( sum(sum(L'.*(L'*DKff{i2}*Phi' + L'*Phi*DKff{i2}'),1)) );
                gdata(i1) = gdata(i1) - 0.5*(b*DKff{i2}*Phi' + b*Phi*DKff{i2}')*b';
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
        % Loop over the noise functions
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
                    gdata(i1)= gdata(i1) + 0.5*sum(1./Sv-sum(L.*L,2)).*DCff{i2};
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
        
    if strcmp(param,'inducing') || strcmp(param,'hyper+inducing')                
        for i=1:ncf
            i1=0;
            if ~isempty(gprior)
                i1 = length(gprior);
            end
            
            gpcf = gp.cf{i};
            
            gpcf.GPtype = gp.type;        
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
    
    g = gdata + gprior;

end

end
