function [e, edata, eprior] = gp_e(w, gp, x, t, param, varargin)
%GP2_E	Evaluate error function for Gaussian Process.
%
%	Description
%	E = GP_E(W, GP, P, T, PARAM) takes a gp data structure GP together
%	with a matrix P of input vectors and a matrix T of target vectors,
%	and evaluates the error function E.  Each row of P
%	corresponds to one input vector and each row of T corresponds to one
%	target vector.
%
%	[E, EDATA, EPRIOR] = GP2R_E(W, GP, P, T, PARAM) also returns the data and
%	prior components of the total error.
%
%	See also
%	GP2, GP2PAK, GP2UNPAK, GP2FWD, GP2R_G
%

% Copyright (c) 2006      Jarno Vanhatalo

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.


% $$$     if strcmp(w, 'init')
% $$$         switch gp.type
% $$$           case 'FULL'
% $$$             invC=[]; B=[];
% $$$             ww=[]; xx =[]; tt=[];
% $$$             e = @gp_enest;
% $$$           case 'FIC'
% $$$             % Initialize help matrices and similarity checkers
% $$$
% $$$             e = @gp_enest;
% $$$           case 'PIC_BLOCK'
% $$$
% $$$           case 'PIC_BAND'
% $$$
% $$$         end
% $$$         return
% $$$     end
% $$$     uu=[]; ww=[]; xx =[]; tt=[];
% $$$     [e, edata, eprior] = gp_enest(w, gp, x, t, varargin{:});
% $$$
% $$$     function [e, edata, eprior] = gp_enest(w, gp, x, t, varargin)


gp=gp_unpak(gp, w, param);
ncf = length(gp.cf);
n=length(x);

% First Evaluate the data contribution to the error
switch gp.type
    % ============================================================
    % FULL
    % ============================================================
    case 'FULL'   % A full GP
        [K, C] = gp_trcov(gp, x);
        % 0.5*log(det(C)) = sum(log(diag(L)))
        L = chol(C)';
        b=L\t;
        edata = 0.5*n.*log(2*pi) + sum(log(diag(L))) + 0.5*b'*b;
        % ============================================================
        % FIC
        % ============================================================
    case 'FIC'
        % The eguations in FIC are implemented as in Neil, 2006
        u = gp.X_u;

        % First evaluate needed covariance matrices
        % v defines that parameter is a vector
        [Kv_ff, Cv_ff] = gp_trvar(gp, x);  % f x 1  vector
        K_fu = gp_cov(gp, x, u);         % f x u
        K_uu = gp_trcov(gp, u);    % u x u, noiseles covariance K_uu
        K_uu = (K_uu+K_uu')./2;     % ensure the symmetry of K_uu
        Luu = chol(K_uu)';
        % Evaluate the Lambda (La)
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
        u = gp.X_u;
        ind = gp.tr_index;

        % First evaluate needed covariance matrices
        % v defines that parameter is a vector
        K_fu = gp_cov(gp, x, u);         % f x u
        K_uu = gp_trcov(gp, u);    % u x u, noiseles covariance K_uu
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
        Luu = chol(K_uu)';

        % Evaluate the Lambda (La)
        % Q_ff = K_fu*inv(K_uu)*K_fu'
        B=Luu\(K_fu');       % u x f
        Qv_ff=sum(B.^2)';
        Lav = Cv_ff-Qv_ff;   % f x 1, Vector of diagonal elements

        iLaKfu = zeros(size(K_fu));  % f x u

        gp.cf = cf2;
        K_cs = gp_trcov(gp,x);
        La = sparse(1:n,1:n,Lav,n,n) + K_cs;
        gp.cf = cf_orig;

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
        % ============================================================
        % CS+PIC
        % ============================================================

    case 'CS+PIC'
        u = gp.X_u;
        ind = gp.tr_index;

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
        K_fu = gp_cov(gp, x, u);         % f x u
        K_uu = gp_trcov(gp, u);    % u x u, noiseles covariance K_uu
        K_uu = (K_uu+K_uu')./2;     % ensure the symmetry of K_uu
        Luu = chol(K_uu)';


        %K_cs = feval(gp.cs.fh_trcov,gp,x);

        % Evaluate the Lambda (La)
        % Q_ff = K_fu*inv(K_uu)*K_fu'
        % Here we need only the blockdiag(Q_ff), which is evaluated below
        B=Luu\(K_fu');       % u x f  and B'*B = K_fu*K_uu*K_uf
        iLaKfu = zeros(size(K_fu));  % f x u
        edata = 0;
        % $$$             for i=1:length(ind)
        % $$$                 Qbl_ff = B(:,ind{i})'*B(:,ind{i});
        % $$$                 [Kbl_ff, Cbl_ff] = gp_trcov(gp, x(ind{i},:));
        % $$$                 Labl{i} = Cbl_ff - Qbl_ff;
        % $$$                 iLaKfu(ind{i},:) = Labl{i}\K_fu(ind{i},:);
        % $$$                 edata = edata + 2*sum(log(diag(chol(Labl{i})))) + t(ind{i},:)'*(Labl{i}\t(ind{i},:));
        % $$$             end

        % $$$             Labl = sparse(1:n,1:n,0,n,n);
        % $$$             for i=1:length(ind)
        % $$$                 Qbl_ff = B(:,ind{i})'*B(:,ind{i});
        % $$$                 [Kbl_ff, Cbl_ff] = gp_trcov(gp, x(ind{i},:));
        % $$$                 Labl(ind{i},ind{i}) = Cbl_ff - Qbl_ff;
        % $$$             end

        [I,J]=find(tril(sparse(gp.tr_indvec(:,1),gp.tr_indvec(:,2),1,n,n),-1));
        q_ff = sum(B(:,I).*B(:,J));
        q_ff = sparse(I,J,q_ff,n,n);
        c_ff = gp_covvec(gp, x(I,:), x(J,:))';
        c_ff = sparse(I,J,c_ff,n,n);
        [Kv_ff, Cv_ff] = gp_trvar(gp,x);
        Labl = c_ff + c_ff' - q_ff - q_ff' + sparse(1:n,1:n, Cv_ff-sum(B.^2,1)',n,n);

        % $$$             I = gp.tr_indvec(:,1);
        % $$$             J = gp.tr_indvec(:,2);
        % $$$             q_ff = sum(B(:,I).*B(:,J));
        % $$$             q_ff = sparse(I,J,q_ff,n,n);
        % $$$             c_ff = gp_covvec(gp, x(I,:), x(J,:))';
        % $$$             c_ff = sparse(I,J,c_ff,n,n);
        % $$$             Labl = c_ff-q_ff;

        gp.cf = cf2;
        K_cs = gp_trcov(gp,x);
        Labl = Labl + K_cs;
        gp.cf = cf_orig;

        iLaKfu = Labl\K_fu;
        edata = edata + 2*sum(log(diag(chol(Labl)))) + t'*(Labl\t);
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
        % PIC_BAND
        % ============================================================
    case  'PIC_BAND'
        u = gp.X_u;
        ind = gp.tr_index;

        % First evaluate needed covariance matrices
        % v defines that parameter is a vector
        K_fu = gp_cov(gp, x, u);         % f x u
        K_uu = gp_trcov(gp, u);    % u x u, noiseles covariance K_uu
        K_uu = (K_uu+K_uu')./2;     % ensure the symmetry of K_uu
        Luu = chol(K_uu)';
        % Evaluate the Lambda (La)
        % Q_ff = K_fu*inv(K_uu)*K_fu'
        % Here we need only the blockdiag(Q_ff), which is evaluated below
        B=Luu\(K_fu');       % u x f  and B'*B = K_fu*K_uu*K_uf
        %            q_ff = zeros(1,size(ind,1));
        % $$$             for i = 1:size(ind,1)
        % $$$                 q_ff(i) = B(:,ind(i,1))'*B(:,ind(i,2));
        % $$$                 %    c_ff(i) = gp_cov(gp, x(ind(i,1),:), x(ind(i,2),:));
        % $$$             end
        % $$$             c_ff = gp_covvec(gp, x(ind(:,1),:), x(ind(:,2),:))';
        % $$$             [Kv_ff, Cv_ff] = gp_trvar(gp,x);
        % $$$             La = sparse(ind(:,1),ind(:,2),c_ff-q_ff,n,n) + sparse(1:n,1:n, Cv_ff-Kv_ff,n,n);

        [I,J]=find(tril(sparse(ind(:,1),ind(:,2),1,n,n),-1));
        % $$$             for i = 1:length(J)
        % $$$                 q_ff(i) = B(:,I(i))'*B(:,J(i));
        % $$$             end
        q_ff = sum(B(:,I).*B(:,J));
        q_ff = sparse(I,J,q_ff,n,n);
        c_ff = gp_covvec(gp, x(I,:), x(J,:))';
        c_ff = sparse(I,J,c_ff,n,n);
        [Kv_ff, Cv_ff] = gp_trvar(gp,x);
        La = c_ff + c_ff' - q_ff - q_ff' + sparse(1:n,1:n, Cv_ff-sum(B.^2,1)',n,n);


        %cputime - t

        iLaKfu = La\K_fu;

        % The data contribution to the error is
        % E = n/2*log(2*pi) + 0.5*log(det(Q_ff+La)) + 0.5*t'inv(Q_ff+La)t
        %   = + 0.5*log(det(La)) + 0.5*trace(iLa*t*t') - 0.5*log(det(K_uu))
        %     + 0.5*log(det(A)) - 0.5*trace(inv(A)*iLaKfu'*t*t'*iLaKfu)

        % First some help matrices...
        % A = chol(K_uu+K_uf*inv(La)*K_fu))
        A = K_uu+K_fu'*iLaKfu;
        A = (A+A')./2;     % Ensure symmetry
        % The actual error evaluation
        % 0.5*log(det(K)) = sum(log(diag(L))), where L = chol(K). NOTE! chol(K) is upper triangular
        b = (t'*iLaKfu)/chol(A);

        edata = 2*sum(log(diag(chol(La)))) + t'*(La\t);
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

% $$$ end
