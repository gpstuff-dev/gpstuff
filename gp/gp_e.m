function [e, edata, eprior] = gp_e(w, gp, x, t, param, varargin)
%GP_E	Evaluate energy function for Gaussian Process 
%
%	Description
%	E = GP_E(W, GP, X, Y, PARAM) takes a Gaussian process data structure GP 
%	together with a matrix X of input vectors and a matrix Y of targets,
%	and evaluates the energy function E.  Each row of X corresponds to one 
%       input vector and each row of Y corresponds to one target vector.
%
%	[E, EDATA, EPRIOR] = GP2R_E(W, GP, P, T, PARAM) also returns the data and
%	prior components of the total energy.
%
%       The energy is minus log posterior cost function:
%            E = EDATA + EPRIOR 
%              = - log p(Y|X, th) - log p(th),
%       where th represents the hyperparameters (lengthScale, magnSigma2...), X is
%       inputs and Y is observations (regression) or latent values (non-Gaussian
%       likelihood).
%
%	See also
%	GP_G, GPCF_*, GP_INIT, GP_PAK, GP_UNPAK, GP_FWD
%

% Copyright (c) 2006      Helsinki University of Technology (author) Jarno Vanhatalo
% Copyright (c) 2007-2008 Jarno Vanhatalo

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.


gp=gp_unpak(gp, w, param);
ncf = length(gp.cf);
n=length(x);

% First Evaluate the data contribution to the error
switch gp.type
    % ============================================================
    % FULL GP (and compact support GP)
    % ============================================================
  case 'FULL'   % A full GP
    [K, C] = gp_trcov(gp, x);

    if issparse(C)
        LD = ldlchol(C);
        edata = 0.5*(n.*log(2*pi) + sum(log(diag(LD))) + t'*ldlsolve(LD,t));
    else
        L = chol(C)';
        b=L\t;
        edata = 0.5*n.*log(2*pi) + sum(log(diag(L))) + 0.5*b'*b;
    end
   
    % ============================================================
    % FIC
    % ============================================================
  case 'FIC'
    % The eguations in FIC are implemented as by Neil (2006)
    % See also Snelson and Ghahramani (2006) and Vanhatalo and Vehtari (2007)

    % First evaluate needed covariance matrices
    % v defines that parameter is a vector
    u = gp.X_u;
    [Kv_ff, Cv_ff] = gp_trvar(gp, x);  % n x 1  vector
    K_fu = gp_cov(gp, x, u);         % n x m
    K_uu = gp_trcov(gp, u);          % m x m, noiseles covariance K_uu
    K_uu = (K_uu+K_uu')./2;          % ensure the symmetry of K_uu
    Luu = chol(K_uu)';
    % Evaluate the Lambda (La)
    % Q_ff = K_fu*inv(K_uu)*K_fu'
    % Here we need only the diag(Q_ff), which is evaluated below
    B=Luu\(K_fu');       % m x n
    Qv_ff=sum(B.^2)';
    Lav = Cv_ff-Qv_ff;   % n x 1, Vector of diagonal elements
                         % iLaKfu = diag(iLav)*K_fu = inv(La)*K_fu
    iLaKfu = zeros(size(K_fu));  % f x u,
    for i=1:n
        iLaKfu(i,:) = K_fu(i,:)./Lav(i);  % f x u
    end
    % The data contribution to the error is
    % E = n/2*log(2*pi) + 0.5*log(det(Q_ff+La)) + 0.5*t'inv(Q_ff+La)*t
    %   = + 0.5*log(det(La)) + 0.5*trace(iLa*t*t') - 0.5*log(det(K_uu))
    %     + 0.5*log(det(A)) - 0.5*trace(inv(A)*iLaKfu'*t*t'*iLaKfu)

    % First some help matrices...
    % A = chol(K_uu+K_uf*inv(La)*K_fu))
    A = K_uu+K_fu'*iLaKfu;
    A = (A+A')./2;     % Ensure symmetry
    A = chol(A);
    % The actual error evaluation
    % 0.5*log(det(K)) = sum(log(diag(L))), where L = chol(K). NOTE! chol(K) is upper triangular
    b = (t'*iLaKfu)/A;
    edata = sum(log(Lav)) + t'./Lav'*t - 2*sum(log(diag(Luu))) + 2*sum(log(diag(A))) - b*b';
    edata = .5*(edata + n*log(2*pi));
    % ============================================================
    % PIC
    % ============================================================
  case 'PIC_BLOCK'
    % First evaluate needed covariance matrices
    % v defines that parameter is a vector
    u = gp.X_u;
    ind = gp.tr_index;
    K_fu = gp_cov(gp, x, u);         % n x m
    K_uu = gp_trcov(gp, u);    % m x m, noiseles covariance K_uu
    K_uu = (K_uu+K_uu')./2;     % ensure the symmetry of K_uu
    Luu = chol(K_uu)';

    % Evaluate the Lambda (La)
    % Q_ff = K_fu*inv(K_uu)*K_fu'
    % Here we need only the blockdiag(Q_ff), which is evaluated below
    B=Luu\(K_fu');       % u x f  and B'*B = K_fu*K_uu*K_uf
    iLaKfu = zeros(size(K_fu));  % f x u
    edata = 0;
    for i=1:length(ind)
        Qbl_ff = B(:,ind{i})'*B(:,ind{i});
        [Kbl_ff, Cbl_ff] = gp_trcov(gp, x(ind{i},:));
        Labl{i} = Cbl_ff - Qbl_ff;
        iLaKfu(ind{i},:) = Labl{i}\K_fu(ind{i},:);
        edata = edata + 2*sum(log(diag(chol(Labl{i})))) + t(ind{i},:)'*(Labl{i}\t(ind{i},:));
    end
    % The data contribution to the error is
    % E = n/2*log(2*pi) + 0.5*log(det(Q_ff+La)) + 0.5*t'inv(Q_ff+La)t

    % First some help matrices...
    % A = chol(K_uu+K_uf*inv(La)*K_fu))
    A = K_uu+K_fu'*iLaKfu;
    A = (A+A')./2;     % Ensure symmetry
    A = chol(A)';
    % The actual error evaluation
    % 0.5*log(det(K)) = sum(log(diag(L))), where L = chol(K). NOTE! chol(K) is upper triangular
    b = (t'*iLaKfu)*inv(A)';
    edata = edata - 2*sum(log(diag(Luu))) + 2*sum(log(diag(A))) - b*b';
    edata = .5*(edata + n*log(2*pi));
    % ============================================================
    % CS+FIC
    % ============================================================
  case 'CS+FIC'
    u = gp.X_u;

    % Separate the FIC and CS covariance functions
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

    % Evaluate the covariance matrices needed for FIC part
    [Kv_ff, Cv_ff] = gp_trvar(gp, x);  % n x 1  vector
    K_fu = gp_cov(gp, x, u);         % n x m
    K_uu = gp_trcov(gp, u);    % m x m, noiseles covariance K_uu
    K_uu = (K_uu+K_uu')./2;     % ensure the symmetry of K_uu
    Luu = chol(K_uu)';

    % Evaluate the Lambda (La)
    % Q_ff = K_fu*inv(K_uu)*K_fu'
    B=Luu\(K_fu');       % m x n
    Qv_ff=sum(B.^2)';
    Lav = Cv_ff-Qv_ff;   % n x 1, Vector of diagonal elements

    % Evaluate the CS covariance matrix
    gp.cf = cf2;
    K_cs = gp_trcov(gp,x);
    La = sparse(1:n,1:n,Lav,n,n) + K_cs;
    gp.cf = cf_orig;     % Set the original covariance functions in the GP structure

    LD = ldlchol(La);
    
    %        iLaKfu = La\K_fu;
    iLaKfu = ldlsolve(LD,K_fu);
    edata = sum(log(diag(LD))) + t'*ldlsolve(LD,t);
    % The data contribution to the error is
    % E = n/2*log(2*pi) + 0.5*log(det(Q_ff+La)) + 0.5*t'inv(Q_ff+La)t

    % First some help matrices...
    % A = chol(K_uu+K_uf*inv(La)*K_fu))
    A = K_uu+K_fu'*iLaKfu;
    A = (A+A')./2;     % Ensure symmetry
    A = chol(A);
    % The actual error evaluation
    % 0.5*log(det(K)) = sum(log(diag(L))), where L = chol(K). NOTE! chol(K) is upper triangular
    %b = (t'*iLaKfu)*inv(A)';
    b = (t'*iLaKfu)/A;
    edata = edata - 2*sum(log(diag(Luu))) + 2*sum(log(diag(A))) - b*b';
    edata = .5*(edata + n*log(2*pi));
  otherwise
    error('Unknown type of Gaussian process!')
end

% ============================================================
% Evaluate the prior contribution to the error from covariance functions
% ============================================================
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
