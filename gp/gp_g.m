function [g, gdata, gprior] = gp_g(w, gp, x, t, param, varargin)
%GP_G   Evaluate gradient of error for Gaussian Process.
%
%	Description
%	G = GP_G(W, GP, X, Y) takes a full GP hyper-parameter vector W, 
%       data structure GP a matrix X of input vectors and a matrix Y
%       of target vectors, and evaluates the error gradient G. Each row of X
%	corresponds to one input vector and each row of Y corresponds
%       to one target vector. Works only for full GP.
%
%	G = GP_G(W, GP, P, Y, PARAM) in case of sparse model takes also  
%       string PARAM defining the parameters to take the gradients with 
%       respect to. Possible parameters are 'hyper' = hyperparameters and 
%      'inducing' = inducing inputs, 'all' = all parameters.
%
%	[G, GDATA, GPRIOR] = GP_G(GP, X, Y) also returns separately  the
%	data and prior contributions to the gradient.
%
%	See also   
%

% Copyright (c) 2006      Jarno Vanhatalo

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
                    % Calculate covariance

        [K, C] = gp_trcov(gp,x);
        invC = inv(C);
        B = C\t;
        
        % Evaluate the gradients from covariance functions
        for i=1:ncf
            gpcf = gp.cf{i};
            gpcf.type = gp.type;
            [g, gdata, gprior] = feval(gpcf.fh_ghyper, gpcf, x, t, g, gdata, gprior, invC, B);
        end
        
        % Evaluate the gradient from noise functions
        if isfield(gp, 'noise')
            nn = length(gp.noise);
            for i=1:nn
                noise = gp.noise{i};
                noise.type = gp.type;
                [g, gdata, gprior] = feval(noise.fh_ghyper, noise, x, t, g, gdata, gprior, invC, B);
            end
        end
        % Do not go further
        return;
        
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
            %            Qbl_ff2(ind{i},ind{i}) = B(:,ind{i})'*B(:,ind{i});
            [Kbl_ff, Cbl_ff] = gp_trcov(gp, x(ind{i},:));
            La{i} = Cbl_ff - Qbl_ff;
            iLaKfu(ind{i},:) = La{i}\K_fu(ind{i},:);    % Check if works by changing inv(La{i})!!!
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

        % ============================================================
        % CS+PIC
        % ============================================================
      case 'CS+PIC'
        g_ind = zeros(1,numel(gp.X_u));
        gdata_ind = zeros(1,numel(gp.X_u));
        gprior_ind = zeros(1,numel(gp.X_u));
        
        u = gp.X_u;
        ind = gp.tr_index;
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
        K_fu = gp_cov(gp, x, u);         % f x u
        K_uu = gp_trcov(gp, u);          % u x u, noiseles covariance K_uu
        K_uu = (K_uu+K_uu')./2;          % ensure the symmetry of K_uu
        Luu = chol(K_uu)';
        % Evaluate the Lambda (La)
        % Q_ff = K_fu*inv(K_uu)*K_fu'
        % Here we need only the diag(Q_ff), which is evaluated below        
        %B=K_fu/Luu;
        B=Luu\K_fu';
        %iLaKfu = zeros(size(K_fu));  % f x u

        [I,J]=find(tril(sparse(gp.tr_indvec(:,1),gp.tr_indvec(:,2),1,n,n),-1));
        q_ff = sum(B(:,I).*B(:,J));
        q_ff = sparse(I,J,q_ff,n,n);
        c_ff = gp_covvec(gp, x(I,:), x(J,:))';
        c_ff = sparse(I,J,c_ff,n,n);
        [Kv_ff, Cv_ff] = gp_trvar(gp,x);
        La = c_ff + c_ff' - q_ff - q_ff' + sparse(1:n,1:n, Cv_ff-sum(B.^2,1)',n,n);
        
        gp.cf = cf2;        
        K_cs = gp_trcov(gp,x);
        La = La + K_cs;
        gp.cf = cf_orig;
        
        iLaKfu = La\K_fu;
        
        % ... then evaluate some help matrices.
        % A = chol(K_uu+K_uf*inv(La)*K_fu))
        A = K_uu+K_fu'*iLaKfu;
        A = (A+A')./2;            % Ensure symmetry
        L = iLaKfu/chol(A);
        b = t'/La - (t'*L)*L';
        
        %iKuuKuf = inv(K_uu)*K_fu';                % L, b, iKuuKuf, La
        iKuuKuf = K_uu\K_fu';
        
        % ============================================================
        % PIC_BAND
        % ============================================================
      case 'PIC_BAND'
        % Do nothing
        u = gp.X_u;
        ind = gp.tr_index;
        nzmax = size(ind,1);
        
        % First evaluate the needed covariance matrices
        % if they are not in the memory
        % v defines that parameter is a vector
        K_fu = gp_cov(gp, x, u);         % f x u
        K_uu = gp_trcov(gp, u);          % u x u, noiseles covariance K_uu
        K_uu = (K_uu+K_uu')./2;          % ensure the symmetry of K_uu
        Luu = chol(K_uu)';
        % Evaluate the Lambda (La)
        % Q_ff = K_fu*inv(K_uu)*K_fu'
        B=Luu\K_fu';        
        [I,J]=find(tril(sparse(ind(:,1),ind(:,2),1,n,n),-1));
        q_ff = sum(B(:,I).*B(:,J));
        q_ff = sparse(I,J,q_ff,n,n);
        c_ff = gp_covvec(gp, x(I,:), x(J,:))';
        c_ff = sparse(I,J,c_ff,n,n);
        [Kv_ff, Cv_ff] = gp_trvar(gp,x);
        La = c_ff + c_ff' - q_ff - q_ff' + sparse(1:n,1:n, Cv_ff-sum(B.^2,1)',n,n);
                
        iLaKfu = La\K_fu;
        
        % ... then evaluate some help matrices.
        % A = chol(K_uu+K_uf*inv(La)*K_fu))
        A = K_uu+K_fu'*iLaKfu;
        A = (A+A')./2;            % Ensure symmetry
        
        L = iLaKfu/chol(A);
        b = t'/La - (t'*L)*L';
        iKuuKuf = inv(K_uu)*K_fu';
    end
    
    % =================================================================
    % Evaluate the gradients from covariance functions
    for i=1:ncf
        gpcf = gp.cf{i};
        gpcf.type = gp.type;
        if isfield(gp, 'X_u')
            gpcf.X_u = gp.X_u;
        end
        if isfield(gp, 'tr_index')
            gpcf.tr_index = gp.tr_index;            
        end
        if isfield(gp, 'tr_indvec')
            gpcf.tr_indvec = gp.tr_indvec;
        end
        switch param
          case 'hyper'
            [g, gdata, gprior] = feval(gpcf.fh_ghyper, gpcf, x, t, g, gdata, gprior, L, b, iKuuKuf, La); %, L2, b2, Labl2
          case 'inducing'
            [g_ind, gdata_ind, gprior_ind] = feval(gpcf.fh_gind, gpcf, x, t, g_ind, gdata_ind, gprior_ind, L, b, iKuuKuf, La);
          case 'all'
            [g, gdata, gprior] = feval(gpcf.fh_ghyper, gpcf, x, t, g, gdata, gprior, L, b, iKuuKuf, La); %, L2, b2, Labl2
            [g_ind, gdata_ind, gprior_ind] = feval(gpcf.fh_gind, gpcf, x, t, g_ind, gdata_ind, gprior_ind, L, b, iKuuKuf, La);
          otherwise
            error('Unknown parameter to take the gradient with respect to! \n')
        end
    end
        
    % Evaluate the gradient from noise functions
    if isfield(gp, 'noise')
        nn = length(gp.noise);
        for i=1:nn            
            gpcf = gp.noise{i};
            gpcf.type = gp.type;
            if isfield(gp, 'X_u')
                gpcf.X_u = gp.X_u;
            end
            if isfield(gp, 'tr_index')
                gpcf.tr_index = gp.tr_index;
            end
            if isfield(gp, 'tr_indvec')
                gpcf.tr_indvec = gp.tr_indvec;
            end
            switch param
              case 'hyper'
                [g, gdata, gprior] = feval(gpcf.fh_ghyper, gpcf, x, t, g, gdata, gprior, L, b, iKuuKuf, La);
              case 'inducing'
                [g_ind, gdata_ind, gprior_ind] = feval(gpcf.fh_gind, gpcf, x, t, g_ind, gdata_ind, gprior_ind, L, b, iKuuKuf, La);
              case 'all'
                [g, gdata, gprior] = feval(gpcf.fh_ghyper, gpcf, x, t, g, gdata, gprior, L, b, iKuuKuf, La); 
                [g_ind, gdata_ind, gprior_ind] = feval(gpcf.fh_gind, gpcf, x, t, g_ind, gdata_ind, gprior_ind, L, b, iKuuKuf, La);
            end
        end
    end
    switch param
      case 'inducing'
        % Evaluate here the gradient from prior
        g = g_ind;
      case 'all'
        % Evaluate here the gradient from prior
        g = [g g_ind];
    end
end
