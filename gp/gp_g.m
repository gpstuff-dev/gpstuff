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
      %================================================================  
      case 'FIC'
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
        b = (t'*iLaKfu)/A;
        C = inv(A) + b'*b;
        C = (C+C')/2;
        
        % Evaluate R = mask(inv(La)*J*inv(La) , diag(n,n)), where J = H - K_fu*C*K_uf;
        %        H = diag(Lav) - t*t'+2*K_fu*inv(A)*K_fu'*diag(1./Lav)*t*t';
        %        J = H - K_fu*C*K_fu';
        %        R = diag(diag(1./Lav)*J*diag(1./Lav));
        R = 1./Lav - (t./Lav).^2 + 2.*(iLaKfu*b').*(t./Lav) -  sum((iLaKfu*chol(C)').^2,2);
        % iKuuKufR = inv(K_uu)*K_uf*R
        iKuuKuf = K_uu\K_fu';
        for i=1:n
            iKuuKufR(:,i) = iKuuKuf(:,i).*R(i);  % f x u 
        end
        %        iKuuKufR = iKuuKuf*diag(R);        
        
        DE_Kuu = 0.5*( C - inv(K_uu) + iKuuKufR*iKuuKuf');  % These are here in matrix form, but
        DE_Kuf = C*iLaKfu' - iKuuKufR - b'*(t./Lav)';       % should be used as vectors DE_Kuu(:) in gpcf_*_g functions
        
        DE_Kff = 0.5*R;
        L = DE_Kuu; b = DE_Kuf; iKuuKuf = DE_Kff; La = Lav;
      %================================================================        
      case 'PIC_BLOCK'
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

% $$$         mask = gp.mask;
% $$$         %        Q_ff2 = B'*B;
% $$$         Q_ff2 = B'*B;
% $$$         [Kbl_ff2, Cbl_ff2] = gp_trcov(gp, x);
% $$$         Labl2 = mask.*(Cbl_ff2 - Q_ff2);
% $$$         iLaKfu2 = Labl2\K_fu;
% $$$         A2 = K_uu+K_fu'*iLaKfu2;
% $$$         A2 = (A2+A2')/2;
% $$$         L2 = iLaKfu2*chol(inv(A2))';
% $$$         b2 = t'/Labl2 - (t'*L2)*L2';
        
        
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
        
        % inv(Labl2) - inv(Q_ff2 + Labl2)
        %inv(mask.*(Cbl_ff2-Q_ff2)) - inv(Q_ff2 + mask.*(Cbl_ff2-Q_ff2))
        %================================================================
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
        switch param
          case 'hyper'
            [g, gdata, gprior] = feval(gpcf.fh_ghyper, gpcf, x, t, g, gdata, gprior, L, b, iKuuKuf, La); %, L2, b2, Labl2
          case 'inducing'
            [D1, D2] = feval(gpcf.fh_gind, gpcf, x, t);
            DKuu_u = DKuu_u + D1;
            DKuf_u = DKuf_u + D2;
          case 'all'
            [g, gdata, gprior] = feval(gpcf.fh_ghyper, gpcf, x, t, g, gdata, gprior, DE_Kuu, DE_Kuf, DE_Kff);            
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
            if isfield(gp, 'tr_index')
                gpcf.tr_index = gp.tr_index;
            end
            switch param
              case 'hyper'
                [g, gdata, gprior] = feval(gpcf.fh_ghyper, gpcf, x, t, g, gdata, gprior, L, b, iKuuKuf, La);
              case 'inducing'
                [D1, D2] = feval(gpcf.fh_gind, gpcf, x, t);
                DKuu_u = DKuu_u + D1;
                DKuf_u = DKuf_u + D2;
              case 'all'
                [g, gdata, gprior] = feval(gpcf.fh_ghyper, gpcf, x, t, g, gdata, gprior, DE_Kuu, DE_Kuf, DE_Kff);            
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








% $$$         C = inv(A) + b'*b;
% $$$         C = (C+C')/2;
% $$$         
% $$$         % Evaluate R = mask(inv(La)*J*inv(La) , diag(n,n)), where J = H - K_fu*C*K_uf;
% $$$         H = Labl - t*t' + (2*K_fu*inv(A)*K_fu'*inv(Labl)*t*t')';
% $$$         H = H + H' - diag(diag(H));
% $$$         J = H - K_fu*C*K_fu';
% $$$         %J= J + J' -diag(diag(J));
% $$$         R = mask.*(inv(Labl)*J*inv(Labl));
% $$$         %        R = R + R' - diag(diag(R));
% $$$         iKuuKuf = K_uu\K_fu';
% $$$         iKuuKufR = iKuuKuf*R;
% $$$         DE_Kuf = b'*(t'/Labl);
% $$$         
% $$$ % $$$         iKuuKuf = K_uu\K_fu';
% $$$ % $$$         DE_Kuf = zeros(size(K_fu'));
% $$$ % $$$         iKuuKufR = zeros(size(iKuuKuf));
% $$$ % $$$ 
% $$$ % $$$         for i=1:length(ind)
% $$$ % $$$             iLat = Labl{i}\t(ind{i},:);
% $$$ % $$$             iLaKfubt = (iLaKfu(ind{i},:)*b');
% $$$ % $$$             R{i} = inv(Labl{i}) - iLat*iLat' + 2.*iLaKfubt*iLat' -  iLaKfu(ind{i},:)*C*iLaKfu(ind{i},:)';
% $$$ % $$$             % iKuuKufR = inv(K_uu)*K_uf*R
% $$$ % $$$             iKuuKufR(:,ind{i}) = iKuuKuf(:,ind{i}')*R{i};  % u x f  
% $$$ % $$$             DE_Kuf(:,ind{i}) = b'*(t(ind{i},:)'/Labl{i});
% $$$ % $$$         end
% $$$                 
% $$$         DE_Kuu = 0.5*( C - inv(K_uu) + iKuuKufR*iKuuKuf'); % These are here in matrix form, but
% $$$         DE_Kuf = C*iLaKfu' - iKuuKufR - DE_Kuf;            % should be used as vectors DE_Kuu(:) in gpcf_*_g functions
% $$$         
% $$$         for i=1:length(ind)
% $$$             DE_Kff{i} = 0.5.*R(ind{i},ind{i});
% $$$         end
% $$$ % $$$         for i=1:length(ind)
% $$$ % $$$             DE_Kff{i} = 0.5.*R{i};
% $$$ % $$$         end
% $$$                
