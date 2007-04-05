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
        
    
    ncf = length(gp.cf);
    n=length(x);
    gp=gp_unpak(gp, w, param);       % unpak the parameters
    
    g = [];
    gdata = [];
    gprior = [];
    
    % First Evaluate the data contribution to the error    
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
        
      case 'FIC'
        u = gp.X_u;
        
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
        % A = chol(K_uu+K_uf*inv(La)*K_fu))
        A = K_uu+K_fu'*iLaKfu;
        A = (A+A')./2;               % Ensure symmetry
        b = (t'*iLaKfu)*inv(A);
        C = inv(A) + b'*b;
        C = (C+C')/2;
               
        % Evaluate R = mask(inv(La)*J*inv(La) , diag(n,n)), where J = H - K_fu*C*K_uf;
        %H = diag(Lav) -t*t'+2*K_fu*pdinv(A)*K_fu'*diag(1./Lav)*t*t';
        %J = H - K_fu*C*K_fu';
        %R = diag(diag(1./Lav)*J*diag(1./Lav));
        R = 1./Lav - (t./Lav).^2 + 2.*(iLaKfu*b').*(t./Lav) -  sum((iLaKfu*chol(C)').^2,2); % diag(iLaKfu*C*iLaKfu'); %
        % iKuuKufR = inv(K_uu)*K_uf*R
        iKuuKuf = K_uu\K_fu';
        for i=1:n
            iKuuKufR(:,i) = iKuuKuf(:,i).*R(i);  % f x u 
        end
                
        DE_Kuu = 0.5*( C - inv(K_uu) + iKuuKufR*iKuuKuf');      % These are here in matrix form, but
        DE_Kuf = C*iLaKfu' - iKuuKufR - b'*(t./Lav)';              % should be used as vectors DE_Kuu(:) 
        %DE_Kuf = 2*DE_Kuf;                                         % in gpcf_*_g functions        
      case 'PIC_BLOCK'
            
      case 'PIC_BAND'
            
    end
    
    % Evaluate the gradients from covariance functions
    for i=1:ncf
        gpcf = gp.cf{i};
        gpcf.type = gp.type;
        if isfield(gp, 'X_u')
            gpcf.X_u = gp.X_u;
        end
        switch param
          case 'hyper'
            [g, gdata, gprior] = feval(gpcf.fh_ghyper, gpcf, x, t, g, gdata, gprior, DE_Kuu, DE_Kuf, 0.5*R);            
          case 'inducing'
            [D1, D2] = feval(gpcf.fh_gind, gpcf, x, t);
            DKuu_u = DKuu_u + D1;
            DKuf_u = DKuf_u + D2;
          case 'all'
            [g, gdata, gprior] = feval(gpcf.fh_ghyper, gpcf, x, t, g, gdata, gprior, DE_Kuu, DE_Kuf, 0.5*R);            
            [D1, D2] = feval(gpcf.fh_gind, gpcf, x, t);
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
            switch param
              case 'hyper'
                [g, gdata, gprior] = feval(gpcf.fh_ghyper, gpcf, x, t, g, gdata, gprior, DE_Kuu, DE_Kuf, 0.5*R);            
              case 'inducing'
                [D1, D2] = feval(gpcf.fh_gind, gpcf, x, t);
                DKuu_u = DKuu_u + D1;
                DKuf_u = DKuf_u + D2;
              case 'all'
                [g, gdata, gprior] = feval(gpcf.fh_ghyper, gpcf, x, t, g, gdata, gprior, DE_Kuu, DE_Kuf, 0.5*R);            
                [D1, D2] = feval(gpcf.fh_gind, gpcf, x, t);
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
