function [g, gdata, gprior] = gpep_g(w, gp, x, y, param, varargin)
%GP_G   Evaluate gradient of error for Gaussian Process.
%
%	Description
%	G = GPEP_G(W, GP, X, Y) takes a full GP hyper-parameter vector W, 
%       data structure GP a matrix X of input vectors and a matrix Y
%       of target vectors, and evaluates the error gradient G. Each row of X
%	corresponds to one input vector and each row of Y corresponds
%       to one target vector. Works only for full GP.
%
%	G = GPEP_G(W, GP, P, Y, PARAM) in case of sparse model takes also  
%       string PARAM defining the parameters to take the gradients with 
%       respect to. Possible parameters are 'hyper' = hyperparameters and 
%      'inducing' = inducing inputs, 'all' = all parameters.
%
%	[G, GDATA, GPRIOR] = GP_G(GP, X, Y) also returns separately  the
%	data and prior contributions to the gradient.
%
%	See also   
%

% Copyright (c) 2007      Jarno Vanhatalo, Jaakko Riihimäki

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
      case 'FULL'   % A full GP
        % Calculate covariance matrix and the site parameters
        [K, C] = gp_trcov(gp,x);
        [e, edata, eprior, tautilde, nutilde, L] = gpep_e(w, gp, x, y, param, varargin);

        Stildesqroot=diag(sqrt(tautilde));
        
        % logZep; nutilde; tautilde;
        b=nutilde-Stildesqroot*(L'\(L\(Stildesqroot*(C*nutilde))));
        invC = Stildesqroot*(L'\(L\Stildesqroot));
        
        % Evaluate the gradients from covariance functions
        for i=1:ncf
            gpcf = gp.cf{i};
            gpcf.type = gp.type;
            [g, gdata, gprior] = feval(gpcf.fh_ghyper, gpcf, x, y, g, gdata, gprior, invC, b);
        end
        
        % Evaluate the gradient from noise functions
        if isfield(gp, 'noise')
            nn = length(gp.noise);
            for i=1:nn
                noise = gp.noise{i};
                noise.type = gp.type;
                [g, gdata, gprior] = feval(noise.fh_ghyper, noise, x, y, g, gdata, gprior, invC, B);
            end
        end
        % Do not go further
        return;
      %================================================================  
      case 'FIC'
        u = gp.X_u;
        DKuu_u = 0;
        DKuf_u = 0;
        [e, edata, eprior, tautilde, nutilde, iLaKfu, La] = gpep_e(w, gp, x, y, param, varargin);
        
        y = nutilde./tautilde;
        
        % First evaluate the needed covariance matrices
        [Kv_ff, Cv_ff] = gp_trvar(gp, x);  % 1 x f  vector
        K_fu = gp_cov(gp, x, u);         % f x u
        K_uu = gp_trcov(gp, u);          % u x u, noiseles covariance K_uu
        K_uu = (K_uu+K_uu')./2;          % ensure the symmetry of K_uu

        A = K_uu+K_fu'*iLaKfu;
        A = (A+A')./2;     % Ensure symmetry

        b = (y'*iLaKfu)/A;
        C = inv(A) + b'*b;
        C = (C+C')/2;
        
        % Evaluate R = mask(inv(La)*J*inv(La) , diag(n,n)), where J = H - K_fu*C*K_uf;
        %        H = diag(La) - t*t' + 2*K_fu*inv(A)*K_fu'*diag(1./Lav)*t*t';
        %        J = H - K_fu*C*K_fu';
        %        R = diag(diag(1./Lav)*J*diag(1./Lav));
        R = 1./La - (y./La).^2 + 2.*(iLaKfu*b').*(y./La) -  sum((iLaKfu*chol(C)').^2,2);
        % iKuuKufR = inv(K_uu)*K_uf*R
        iKuuKuf = K_uu\K_fu';
        for i=1:n
            iKuuKufR(:,i) = iKuuKuf(:,i).*R(i);  % f x u 
        end
        %        iKuuKufR = iKuuKuf*diag(R);        
        
        DE_Kuu = 0.5*( C - inv(K_uu) + iKuuKufR*iKuuKuf');  % These are here in matrix form, but
        DE_Kuf = C*iLaKfu' - iKuuKufR - b'*(y./La)';       % should be used as vectors DE_Kuu(:) in gpcf_*_g functions
        
        DE_Kff = 0.5*R;
        L = DE_Kuu; b = DE_Kuf; iKuuKuf = DE_Kff;
      %================================================================        
      case 'PIC_BLOCK'
        u = gp.X_u;
        ind = gp.tr_index;
        DKuu_u = 0;
        DKuf_u = 0;
        [e, edata, eprior, tautilde, nutilde, L, La] = gpep_e(w, gp, x, y, param, varargin);

        y = nutilde./tautilde;
        
        % First evaluate the needed covariance matrices
        % if they are not in the memory
        % v defines that parameter is a vector
        K_fu = gp_cov(gp, x, u);         % f x u
        K_uu = gp_trcov(gp, u);          % u x u, noiseles covariance K_uu
        K_uu = (K_uu+K_uu')./2;          % ensure the symmetry of K_uu

        b = zeros(1,n);
        for i=1:length(ind)
            b(ind{i}) = (La{i}\y(ind{i}))';
        end
        b = b - (y'*L)*L';
        
        iKuuKuf = K_uu\K_fu';
        
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
        switch param
          case 'hyper'
            [g, gdata, gprior] = feval(gpcf.fh_ghyper, gpcf, x, y, g, gdata, gprior, L, b, iKuuKuf, La); %, L2, b2, Labl2
          case 'inducing'
            [D1, D2] = feval(gpcf.fh_gind, gpcf, x, y);
            DKuu_u = DKuu_u + D1;
            DKuf_u = DKuf_u + D2;
          case 'all'
            [g, gdata, gprior] = feval(gpcf.fh_ghyper, gpcf, x, y, g, gdata, gprior, DE_Kuu, DE_Kuf, DE_Kff);            
            [D1, D2] = feval(gpcf.fh_gind, gpcf, x, y);
            DKuu_u = DKuu_u + D1;
            DKuf_u = DKuf_u + D2;
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
            switch param
              case 'hyper'
                [g, gdata, gprior] = feval(gpcf.fh_ghyper, gpcf, x, y, g, gdata, gprior, L, b, iKuuKuf, La);
              case 'inducing'
                [D1, D2] = feval(gpcf.fh_gind, gpcf, x, y);
                DKuu_u = DKuu_u + D1;
                DKuf_u = DKuf_u + D2;
              case 'all'
                [g, gdata, gprior] = feval(gpcf.fh_ghyper, gpcf, x, y, g, gdata, gprior, DE_Kuu, DE_Kuf, DE_Kff);            
                [D1, D2] = feval(gpcf.fh_gind, gpcf, x, y);
                DKuu_u = DKuu_u + D1;
                DKuf_u = DKuf_u + D2;
            end
        end
    end
    switch param
      case 'inducing'
        % The prior gradient has to be implemented here, whenever the prior is defined
        g = DE_Kuu(:)'*DKuu_u + DE_Kuf(:)'*DKuf_u;
      case 'all'
        g2 = DE_Kuu(:)'*DKuu_u + DE_Kuf(:)'*DKuf_u;
        g = [g g2];
    end
end
