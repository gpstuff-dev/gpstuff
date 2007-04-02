function [g, gdata, gprior] = gp_g(w, gp, x, t, varargin)
%GP_G   Evaluate gradient of error for Gaussian Process.
%
%	Description
%	G = GP_G(W, GP, X, T) takes a gp hyper-parameter vector W, 
%       data structure GP a matrix X of input vectors and a matrix T 
%       of target vectors, and evaluates the error gradient G. Each row of X
%	corresponds to one input vector and each row of T corresponds
%       to one target vector.
%
%	G = GP_G(W, GP, P, T, U) in case of sparse model takes also inducing 
%       points U.
%
%	[G, GDATA, GPRIOR] = GP_G(GP, P, T) also returns separately  the
%	data and prior contributions to the gradient.
%
%	See also   
%

% Copyright (c) 2006      Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.


    gp=gp_unpak(gp, w);
    ncf = length(gp.cf);
    n=length(x);

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
            [g, gdata, gprior] = feval(gpcf.fh_g, gpcf, x, t, g, gdata, gprior, invC, B);
        end
        
        % Evaluate the gradient from noise functions
        if isfield(gp, 'noise')
            nn = length(gp.noise);
            for i=1:nn
                noise = gp.noise{i};
                noise.type = gp.type;
                [g, gdata, gprior] = feval(noise.fh_g, noise, x, t, g, gdata, gprior, invC, B);
            end
        end
        % Do not go further
        return;
        
      case 'FIC'
        u = gp.X_u;
        
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
% $$$         Lav = Cv_ff - diag(K_fu*pdinv(K_uu)*K_fu');
% $$$         iLaKfu= diag(1./Lav)*K_fu;
        % ... then evaluate some help matrices.
        % A = chol(K_uu+K_uf*inv(La)*K_fu))
        A = K_uu+K_fu'*iLaKfu;
        A = (A+A')./2;               % Ensure symmetry
        b = (t'*iLaKfu)*pdinv(A);
        %        C = 2*inv(A)-diag(diag(inv(A))) - 2*b'*b + diag(diag(b'*b));
        C = inv(A) + b'*b;
        %C = A - b'*b;
        %C = (C+C')/2;
               
        % Evaluate R = mask(inv(La)*J*inv(La) , diag(n,n)), where J = H - K_fu*C*K_uf;
        H = diag(Lav) -t*t'+2*K_fu*pdinv(A)*K_fu'*diag(1./Lav)*t*t';
        J = H - K_fu*C*K_fu';
        R = diag(diag(1./Lav)*J*diag(1./Lav));
        %        R = 1./Lav - (t./Lav).^2 + 2.*(iLaKfu*b').*(t./Lav) -  diag(iLaKfu*C*iLaKfu'); %sum((iLaKfu*chol(C)'),2); %
        % iKuuKufR = inv(K_uu)*K_uf*R
        iKuuKuf = K_uu\K_fu';
% $$$         for i=1:n
% $$$             iKuuKufR(:,i) = iKuuKuf(:,i).*R(i);  % f x u 
% $$$         end
        iKuuKufR = iKuuKuf*diag(R);
        
        DE_Kuu = 0.5*( C - pdinv(K_uu) + iKuuKufR*iKuuKuf');      % These are here in matrix form, but
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
        [g, gdata, gprior] = feval(gpcf.fh_g, gpcf, x, t, g, gdata, gprior, DE_Kuu, DE_Kuf, 0.5*R);
    end
        
    % Evaluate the gradient from noise functions
    if isfield(gp, 'noise')
        nn = length(gp.noise);
        for i=1:nn
            noise = gp.noise{i};
            noise.type = gp.type;
            [g, gdata, gprior] = feval(noise.fh_g, noise, x, t, g, gdata, gprior, DE_Kuu, DE_Kuf, 0.5*R);
        end
    end
end



































% $$$ 
% $$$ if strcmp(w, 'init')
% $$$     if isfield(gp,'sparse')
% $$$         % Initialize help matrices and similarity checkers
% $$$         b=[]; A=[]; W=[]; iKuuKuf=[];
% $$$         uu=[]; ww=[]; xx =[]; tt=[];
% $$$         g = @gp_gnest;
% $$$     else
% $$$         invC=[]; B=[];
% $$$         ww=[]; xx =[]; tt=[];
% $$$         g = @gp_gnest;
% $$$     end
% $$$     return
% $$$ end
% $$$ uu=[]; ww=[]; xx =[]; tt=[];
% $$$ [g, gdata, gprior] = gp_gnest(w, gp, x, t, varargin{:});
% $$$ 
% $$$     function [g, gdata, gprior] = gp_gnest(w, gp, x, t, varargin)
% $$$     gp=gp_unpak(gp, w);
% $$$     ncf = length(gp.cf);
% $$$     n=length(x);
% $$$     
% $$$     g = [];
% $$$     gdata = [];
% $$$     gprior = [];
% $$$     
% $$$     % First check if sparse Gaussian process is used... 
% $$$     if isfield(gp, 'sparse')
% $$$         u = varargin{1};
% $$$         if ~issame(uu,u) && ~issame(ww,w) && ~issame(xx,x) && ~issame(tt,t)
% $$$             % First evaluate the needed covariance matrices
% $$$             % if they are not in the memory
% $$$             % v defines that parameter is a vector
% $$$             [Kv_ff, Cv_ff] = gp_trvar(gp, x);  % 1 x f  vector
% $$$             K_fu = gp_cov(gp, x, u);         % f x u
% $$$             K_uu = gp_trcov(gp, u);    % u x u, noiseles covariance K_uu
% $$$             K_uu = (K_uu+K_uu')./2;     % ensure the symmetry of K_uu
% $$$             Luu = chol(K_uu)';
% $$$             % Evaluate the Lambda (La) for specific model
% $$$             switch gp.sparse
% $$$               case 'FIC'
% $$$                 % Q_ff = K_fu*inv(K_uu)*K_fu'
% $$$                 % Here we need only the diag(Q_ff), which is evaluated below
% $$$                 B=Luu\(K_fu');
% $$$                 Qv_ff=sum(B.^2)';
% $$$                 Lav = Cv_ff-Qv_ff;   % 1 x f, Vector of diagonal elements
% $$$                 % iLaKfu = diag(inv(Lav))*K_fu = inv(La)*K_fu
% $$$                 iLaKfu = zeros(size(K_fu));  % f x u, 
% $$$                 for i=1:n
% $$$                     iLaKfu(i,:) = K_fu(i,:)./Lav(i);  % f x u 
% $$$                 end
% $$$             end
% $$$             % ... then evaluate some help matrices.
% $$$             % iKuuKuf = inv(K_uu)*K_uf
% $$$             iKuuKuf = K_uu\K_fu';
% $$$             c = K_uu+K_fu'*iLaKfu; 
% $$$             c = (c+c')./2;          % ensure symmetry
% $$$             c = chol(c)';   % u x u, 
% $$$             ic = inv(c);
% $$$             % b = t'*inv(Q_ff+La)
% $$$             %   = t'*La - t'*inv(La)*K_fu*inv(K_uu+K_uf*inv(La)*K_fu)*K_uf*inv(La)
% $$$             c = iLaKfu*ic';
% $$$             b = (t./Lav)' - (t'*c)*c';
% $$$             cc = iLaKfu/K_uu;
% $$$             % A = inv(K_uu)*K_uf*inv(Q_ff + La)
% $$$             A = (cc - c*ic*(K_fu'*cc))';
% $$$             W = 1./Lav-sum(c.^2, 2);
% $$$         end
% $$$         
% $$$         % Evaluate the gradients from covariance functions
% $$$         for i=1:ncf
% $$$             gpcf = gp.cf{i};
% $$$             gpcf.sparse = gp.sparse;
% $$$             [g, gdata, gprior] = feval(gpcf.fh_g, gpcf, x, t, g, gdata, gprior, b, A, u, W, iKuuKuf);
% $$$         end
% $$$         
% $$$         % Evaluate the gradient from noise functions
% $$$         if isfield(gp, 'noise')
% $$$             nn = length(gp.noise);
% $$$             for i=1:nn
% $$$                 noise = gp.noise{i};
% $$$                 noise.sparse = gp.sparse;
% $$$                 [g, gdata, gprior] = feval(noise.fh_g, noise, x, t, g, gdata, gprior, b, A, u, W, iKuuKuf);
% $$$             end
% $$$         end
% $$$         
% $$$     else
% $$$         % Calculate covariance
% $$$         [K, C] = gp_trcov(gp,x);
% $$$         
% $$$         invC = inv(C);
% $$$         B = C\t;
% $$$         
% $$$         % Evaluate the gradients from covariance functions
% $$$         for i=1:ncf
% $$$             gpcf = gp.cf{i};
% $$$             [g, gdata, gprior] = feval(gpcf.fh_g, gpcf, x, t, g, gdata, gprior, invC, B);
% $$$         end
% $$$         
% $$$         % Evaluate the gradient from noise functions
% $$$         if isfield(gp, 'noise')
% $$$             nn = length(gp.noise);
% $$$             for i=1:nn
% $$$                 noise = gp.noise{i};
% $$$                 [g, gdata, gprior] = feval(noise.fh_g, noise, x, t, g, gdata, gprior, invC, B);
% $$$             end
% $$$         end
% $$$     end
% $$$     end
% $$$ end
% $$$ 
% $$$ 
