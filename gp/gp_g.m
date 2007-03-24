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

if strcmp(w, 'init')
    if isfield(gp,'sparse')
        % Initialize help matrices and similarity checkers
        b=[]; A=[]; W=[]; iKuuKuf=[];
        uu=[]; ww=[]; xx =[]; tt=[];
        g = @gp_gnest;
    else
        invC=[]; B=[];
        ww=[]; xx =[]; tt=[];
        g = @gp_gnest;
    end
    return
end
uu=[]; ww=[]; xx =[]; tt=[];
[g, gdata, gprior] = gp_gnest(w, gp, x, t, varargin{:});

    function [g, gdata, gprior] = gp_gnest(w, gp, x, t, varargin)
    gp=gp_unpak(gp, w);
    ncf = length(gp.cf);
    n=length(x);
    
    g = [];
    gdata = [];
    gprior = [];
    
    % First check if sparse Gaussian process is used... 
    if isfield(gp, 'sparse')
        u = varargin{1};
        if ~issame(uu,u) && ~issame(ww,w) && ~issame(xx,x) && ~issame(tt,t)
            % First evaluate the needed covariance matrices
            % if they are not in the memory
            % v defines that parameter is a vector
            [Kv_ff, Cv_ff] = gp_trvar(gp, x);  % 1 x f  vector
            K_fu = gp_cov(gp, x, u);         % f x u
            K_uu = gp_trcov(gp, u);    % u x u, noiseles covariance K_uu
            K_uu = (K_uu+K_uu')./2;     % ensure the symmetry of K_uu
            Luu = chol(K_uu)';
            % Evaluate the Lambda (La) for specific model
            switch gp.sparse
              case 'FITC'
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
            end
            % ... then evaluate some help matrices.
            % iKuuKuf = inv(K_uu)*K_uf
            iKuuKuf = K_uu\K_fu';
            c = K_uu+K_fu'*iLaKfu; 
            c = (c+c')./2;          % ensure symmetry
            c = chol(c)';   % u x u, 
            ic = inv(c);
            % b = t'*inv(Q_ff+La)
            %   = t'*La - t'*inv(La)*K_fu*inv(K_uu+K_uf*inv(La)*K_fu)*K_uf*inv(La)
            c = iLaKfu*ic';
            b = (t./Lav)' - (t'*c)*c';
            cc = iLaKfu/K_uu;
            % A = inv(K_uu)*K_uf*inv(Q_ff + La)
            A = (cc - c*ic*(K_fu'*cc))';
            W = 1./Lav-sum(c.^2, 2);
        end
        
        % Evaluate the gradients from covariance functions
        for i=1:ncf
            gpcf = gp.cf{i};
            gpcf.sparse = gp.sparse;
            [g, gdata, gprior] = feval(gpcf.fh_g, gpcf, x, t, g, gdata, gprior, b, A, u, W, iKuuKuf);
        end
        
        % Evaluate the gradient from noise functions
        if isfield(gp, 'noise')
            nn = length(gp.noise);
            for i=1:nn
                noise = gp.noise{i};
                noise.sparse = gp.sparse;
                [g, gdata, gprior] = feval(noise.fh_g, noise, x, t, g, gdata, gprior, b, A, u, W, iKuuKuf);
            end
        end
        
    else
        % Calculate covariance
        [K, C] = gp_trcov(gp,x);
        
        invC = inv(C);
        B = C\t;
        
        % Evaluate the gradients from covariance functions
        for i=1:ncf
            gpcf = gp.cf{i};
            [g, gdata, gprior] = feval(gpcf.fh_g, gpcf, x, t, g, gdata, gprior, invC, B);
        end
        
        % Evaluate the gradient from noise functions
        if isfield(gp, 'noise')
            nn = length(gp.noise);
            for i=1:nn
                noise = gp.noise{i};
                [g, gdata, gprior] = feval(noise.fh_g, noise, x, t, g, gdata, gprior, invC, B);
            end
        end
    end
    end
end


