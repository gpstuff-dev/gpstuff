function [e, edata, eprior] = gp_e(w, gp, x, t, varargin)
%GP2_E	Evaluate error function for Gaussian Process.
%
%	Description
%	E = GP_E(W, GP, P, T) takes a gp data structure GP together
%	with a matrix P of input vectors and a matrix T of target vectors,
%	and evaluates the error function E.  Each row of P
%	corresponds to one input vector and each row of T corresponds to one
%	target vector.
%
%	E = GP_E(W, GP, P, T, U) in case of sparse model takes also inducing 
%       points U
%
%	[E, EDATA, EPRIOR] = GP2R_E(W, GP, P, T) also returns the data and
%	prior components of the total error.
%
%	See also
%	GP2, GP2PAK, GP2UNPAK, GP2FWD, GP2R_G
%

% Copyright (c) 2006      Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

if strcmp(w, 'init')
    if isfield(gp,'sparse')
        % Initialize help matrices and similarity checkers
        c=[]; ic=[]; Lav=[]; Luu=[]; iLaKfu=[];
        uu=[]; ww=[]; xx =[]; tt=[];
        e = @gp_enest;
    else
        invC=[]; B=[];
        ww=[]; xx =[]; tt=[];
        e = @gp_enest;
    end
    return
end
uu=[]; ww=[]; xx =[]; tt=[];
[e, edata, eprior] = gp_enest(w, gp, x, t, varargin{:});

    function [e, edata, eprior] = gp_enest(w, gp, x, t, varargin)
    
    gp=gp_unpak(gp, w);
    ncf = length(gp.cf);
    n=length(x);
    
    % First check if sparse Gaussian process is used, if so evaluate
    % the data contribution to the error of sparse GP.
    if isfield(gp, 'sparse')
       
        u = varargin{1};
        if ~issame(uu,u) && ~issame(ww,w) && ~issame(xx,x) && ~issame(tt,t)
            % First evaluate needed covariance matrices
            % v defines that parameter is a vector
            [Kv_ff, Cv_ff] = gp_trvar(gp, x);  % f x 1  vector
            K_fu = gp_cov(gp, x, u);         % f x u
            K_uu = gp_trcov(gp, u);    % u x u, noiseles covariance K_uu
            K_uu = (K_uu+K_uu')./2;     % ensure the symmetry of K_uu
            Luu = chol(K_uu)';
            % Evaluate the Lambda (La) for specific model
            switch gp.sparse
              case 'FIC'
                % Q_ff = K_fu*inv(K_uu)*K_fu'
                % Here we need only the diag(Q_ff), which is evaluated below
                B=Luu\(K_fu');       % u x f
                Qv_ff=sum(B.^2)';
                Lav = Cv_ff-Qv_ff;   % f x 1, Vector of diagonal elements
                                     % iLaKfu = diag(iLav)*K_fu = inv(La)*K_fu
                iLaKfu = zeros(size(K_fu));  % f x u, 
                for i=1:n
                    iLaKfu(i,:) = K_fu(i,:)./Lav(i);  % f x u 
                end
            end
            % The data contribution to the error is 
            % E = n/2*log(2*pi) + 0.5*log(det(Q_ff+La)) + 0.5*t'inv(Q_ff+La)t
            %   = n/2nlog(2*pi) + ldet + linv, 
            
            % First some help matrices...
            % c = chol(K_uu+K_uf*inv(La)*K_fu))
            c = K_uu+K_fu'*iLaKfu; 
            c = (c+c')./2;          % ensure symmetry
            c = chol(c)';   % u x u, 
            ic = inv(c);
        end
        % The actual error evaluation
        % log(det(K)) = sum(log(diag(L))), where L = chol(K). NOTE! chol(K) is upper triangular
        ldet = 0.5*sum(log(Lav))+sum(log(diag(inv(Luu))))+sum(log(diag(c)));
        % Here on two following rows we evaluate actually  
        % linv = 0.5*t'*inv(La)*t-0.5*t'*(inv(La)*K_fu*(K_uu+Kuf*inv(La)*K_fu)*K_fu'*inv(La))*t;
        b = (t'*iLaKfu)*ic';  % 1 x u
        linv = 0.5*(t./Lav)'*t-0.5*b*b';
        edata = 0.5*n.*log(2*pi) + ldet + linv;
        
    else   % Here evaluate the data contribution to the error of full Gaussian process
        [K, C] = gp_trcov(gp, x);
        % 0.5*log(det(C)) = sum(log(diag(L)))
        L = chol(C)';
        b=L\t;
        edata = 0.5*n.*log(2*pi) + sum(log(diag(L))) + 0.5*b'*b;
    end    
    
    % Evaluate the prior contribution to the error from covariance functions
    eprior = 0;
    for i=1:ncf
        gpcf = gp.cf{i};
        eprior = eprior + feval(gpcf.fh_e, gpcf, x, t);
    end
    
    % Evaluate the prior contribution to the error from noise functions
    if isfield(gp, 'noise')
        nn = length(gp.noise);
        for i=1:nn
            noise = gp.noise{i};
            eprior = eprior + feval(noise.fh_e, noise, x, t);
        end
    end
    
    e = edata + eprior;
    end
end