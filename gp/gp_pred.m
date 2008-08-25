function [y, VarY, noisyY] = gp_pred(gp, tx, ty, x, varargin)
%GP_PRED    Make predictions for Gaussian process
%
%	Description
%	F = GP_PRED(GP, TX, TY, X) takes a gp data structure GP together with a
%	matrix X of input vectors, Matrix TX of training inputs and vector TY of
%       training targets, and returns a vector F of predictions (mean(Y|X, TX, TY)). 
%       Each row of X corresponds to one input vector and each row of Y corresponds 
%       to one output vector.
%
%	[F, VarF] = GP_PRED(GP, TX, TY, X) returns also the predictive (noiseles) 
%       variances of F (1xn vector). NOTE! VarF contains the variance of the latent 
%       function, that is  diag(K_fy - K_fy*(Kyy+s^2I)^(-1)*K_yf. If you want predictive 
%       variance of observations add gp.noise{1}.noiseSigmas2 to VarF.
%
%	[F, VarF, sampY] = GP_PRED(GP, TX, TY, X) returns also a sample from predictive 
%       distribution of observation sampY. This is needed for example in the Student-t 
%       noise model.
%       NOTE! in FIC/PIC the prediction NoisyY is made at the training points!
%
%	See also
%	GP_PREDS, GP_PAK, GP_UNPAK
%

% Copyright (c) 2006      Helsinki University of Technology (author Jarno Vanhatalo)
% Copyright (c) 2007-2008 Jarno Vanhatalo

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.

tn = size(tx,1);

% Evaluate this if sparse model is used
switch gp.type
  case 'FULL'
    K=gp_cov(gp,tx,x);
    if nargin > 4
        y=K'*(invC*ty);
    else
        [c, C]=gp_trcov(gp,tx);
        if issparse(C)
            LD = ldlchol(C);
            y = K'*ldlsolve(LD,ty);
        else
            L = chol(C)';
            %    y=K'*(C\ty);
            a = L'\(L\ty);
            y = K'*a;
        end
    end
    if nargout > 1
        
        if issparse(C)
            V = gp_trvar(gp,x);
            VarY = V - diag(K'*ldlsolve(LD,K));
        else
            v = L\K;
            V = gp_trvar(gp,x);
            % Vector of diagonal elements of covariance matrix
            % b = L\K;
            % VarY = V - sum(b.^2)';
            VarY = V - diag(v'*v);
        end
    end
    if nargout > 2
        K2 = gp_trcov(gp,x);
        predcov = chol(K2-v'*v)';
        noisyY = y + predcov*randn(size(y));
    end
  case 'FIC'
    u = gp.X_u;
    % Turn the inducing vector on right direction
    if size(u,2) ~= size(tx,2)
        u=u';
    end
    % Calculate some help matrices
    [Kv_ff, Cv_ff] = gp_trvar(gp, tx);  % 1 x f  vector
    K_fu = gp_cov(gp, tx, u);         % f x u
    K_nu = gp_cov(gp, x, u);         % n x u
    K_uu = gp_trcov(gp, u);    % u x u, noiseles covariance K_uu
    Luu = chol(K_uu)';
    % Evaluate the Lambda (La) for specific model
    % Q_ff = K_fu*inv(K_uu)*K_fu'
    % Here we need only the diag(Q_ff), which is evaluated below
    B=Luu\(K_fu');
    Qv_ff=sum(B.^2)';
    Lav = Cv_ff-Qv_ff;   % 1 x f, Vector of diagonal elements
                         % iLaKfu = diag(inv(Lav))*K_fu = inv(La)*K_fu
    iLaKfu = zeros(size(K_fu));  % f x u,
    for i=1:length(tx)
        iLaKfu(i,:) = K_fu(i,:)./Lav(i);  % f x u
    end
    A = K_uu+K_fu'*iLaKfu;
    A = (A+A')./2;

    L = iLaKfu/chol(A);
    p = ty./Lav - L*(L'*ty);

    %p2 = ty./Lav - iLaKfu*(A\(iLaKfu'*ty));
    %    Knf = K_nu*(K_uu\K_fu');
    y = K_nu*(K_uu\(K_fu'*p));

    if nargout > 1
        Knn_v = gp_trvar(gp,x);
        B2=Luu\(K_nu');
        VarY = Knn_v - sum(B2'.*(B*(repmat(Lav,1,size(K_uu,1)).\B')*B2)',2)  + sum((K_nu*(K_uu\(K_fu'*L))).^2, 2);
    end
    if nargout > 2
        % Sigma_post = Qnn + La_n - Qnf*(Qff+La_f)^(-1)*Qfn
        %            = B'*(I-B*La_f^(-1)*B' + B*L*L'*B')*B + La_n
        Lav_n = Kv_ff-Qv_ff;
        BL = B*L;
        Sigm_mm = eye(size(K_uu)) - B*(repmat(Lav,1,size(K_uu,1)).\B') + BL*BL';
        noisyY = y + B'*(chol(Sigm_mm)'*randn(size(K_uu,1),1)) + randn(size(y)).*sqrt(Lav_n);
    end
  case 'PIC_BLOCK'
    u = gp.X_u;
    ind = gp.tr_index;
    tstind = varargin{1};   % An array containing information
                            % in which block each of the test inputs belongs
                            % Turn the inducing vector on right direction
    if size(u,2) ~= size(tx,2)
        u=u';
    end

    % Calculate some help matrices
    [Kv_ff, Cv_ff] = gp_trvar(gp, tx);  % 1 x f  vector
    K_fu = gp_cov(gp, tx, u);         % f x u
    K_nu = gp_cov(gp, x, u);         % n x u
    K_uu = gp_trcov(gp, u);    % u x u, noiseles covariance K_uu
    Luu = chol(K_uu)';
    % Evaluate the Lambda (La) for specific model
    % Q_ff = K_fu*inv(K_uu)*K_fu'
    % Here we need only the diag(Q_ff), which is evaluated below
    B=Luu\K_fu';
    iLaKfu = zeros(size(K_fu));  % f x u
    for i=1:length(ind)
        Qbl_ff = B(:,ind{i})'*B(:,ind{i});
        [Kbl_ff, Cbl_ff] = gp_trcov(gp, tx(ind{i},:));
        La{i} = Cbl_ff - Qbl_ff;
        iLaKfu(ind{i},:) = La{i}\K_fu(ind{i},:);    
    end
    A = K_uu+K_fu'*iLaKfu;
    A = (A+A')./2;            % Ensure symmetry
    L = iLaKfu/chol(A);

    tyy = ty;
    % From this on evaluate the prediction
    % See Snelson and Ghahramani (2007) for details
    p=iLaKfu*(A\(iLaKfu'*tyy));
    for i=1:length(ind)
        p2(ind{i},:) = La{i}\tyy(ind{i},:);
    end
    p= p2-p;

    iKuuKuf = K_uu\K_fu';
    
    w_bu=zeros(length(x),length(u));
    w_n=zeros(length(x),1);
    for i=1:length(ind)
        w_bu(tstind{i},:) = repmat((iKuuKuf(:,ind{i})*p(ind{i},:))', length(tstind{i}),1);
        K_nf = gp_cov(gp, x(tstind{i},:), tx(ind{i},:));              % n x u
        w_n(tstind{i},:) = K_nf*p(ind{i},:);
    end
    
    y = K_nu*(iKuuKuf*p) - sum(K_nu.*w_bu,2) + w_n;
    

    if nargout > 1
        
        kstarstar = gp_trvar(gp, x);
        KnuiKuu = K_nu/K_uu;
        KufiLaKfu = K_fu'*iLaKfu;
        QnfL = KnuiKuu*(K_fu'*L);
        Varf1 = zeros(size(x,1),1);
        Varf2 = zeros(size(x,1),1);
        Varf3 = zeros(size(x,1),1);
        for i=1:length(ind)
            KubiLaKbu = K_fu(ind{i},:)'/La{i}*K_fu(ind{i},:);
            nonblock = KufiLaKfu - KubiLaKbu;
            Varf1(tstind{i}) = diag(KnuiKuu(tstind{i},:)*nonblock*KnuiKuu(tstind{i},:)');
            
            Knb = gp_cov(gp, x(tstind{i},:), tx(ind{i},:));
            Varf2(tstind{i}) = diag(Knb/La{i}*Knb');
            
            
            KnbL = Knb*L(ind{i},:);
            QnbL = KnuiKuu(tstind{i},:)*(K_fu(ind{i},:)'*L(ind{i},:));
            %Varf3(tstind{i}) = sum(QnfL(tstind{i},:) - QnbL + KnbL,2);
            Varf3(tstind{i}) = diag((QnfL(tstind{i},:) - QnbL + KnbL)*(QnfL(tstind{i},:) - QnbL + KnbL)');
        end
        
        VarY = kstarstar - (Varf1 + Varf2 - Varf3);
    end
    if nargout > 2
        noisyY = y;
    end
  case 'CS+FIC'
    u = gp.X_u;
    ncf = length(gp.cf);
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
    [Kv_ff, Cv_ff] = gp_trvar(gp, tx);  % f x 1  vector
    K_fu = gp_cov(gp, tx, u);         % f x u
    K_uu = gp_trcov(gp, u);    % u x u, noiseles covariance K_uu
    K_uu = (K_uu+K_uu')./2;     % ensure the symmetry of K_uu
    Luu = chol(K_uu)';
    K_nu = gp_cov(gp, x, u);         % n x u

    % Evaluate the Lambda (La)
    % Q_ff = K_fu*inv(K_uu)*K_fu'
    B=Luu\(K_fu');       % u x f
    Qv_ff=sum(B.^2)';
    Lav = Cv_ff-Qv_ff;   % f x 1, Vector of diagonal elements

    gp.cf = cf2;
    K_cs = gp_trcov(gp,tx);
    Kcs_nf = gp_cov(gp, x, tx);
    La = sparse(1:tn,1:tn,Lav,tn,tn) + K_cs;
    gp.cf = cf_orig;

    iLaKfu = La\K_fu;
    A = K_uu+K_fu'*iLaKfu;
    A = (A+A')./2;     % Ensure symmetry
    L = iLaKfu/chol(A);
    
    p = La\ty - L*(L'*ty);

    %p2 = ty./Lav - iLaKfu*(A\(iLaKfu'*ty));
    %    Knf = K_nu*(K_uu\K_fu');
    y = K_nu*(K_uu\(K_fu'*p)) + Kcs_nf*p;
    
    if nargout > 1
        Knn_v = gp_trvar(gp,x);
        B2=Luu\(K_nu');
        VarY = Knn_v - sum(B2'.*(B*(La\B')*B2)',2)  + sum((K_nu*(K_uu\(K_fu'*L))).^2, 2) - sum((Kcs_nf/chol(La)).^2,2) + sum((Kcs_nf*L).^2, 2);
        %VarY = VarY - 2.*diag((Kcs_nf*iLaKfu)*(K_uu\K_nu')) + 2.*diag((Kcs_nf*L)*(L'*K_fu*(K_uu\K_nu')));
        VarY = VarY - 2.*sum((Kcs_nf*iLaKfu).*(K_uu\K_nu')',2) + 2.*sum((Kcs_nf*L).*(L'*K_fu*(K_uu\K_nu'))' ,2);
    end
    if nargout > 2
        error('gp_fwd with three output arguments is not implemented for CS+FIC!')
    end    
  case 'SSGP'
    [Phi_f, S] = gp_trcov(gp, tx);
    Phi_a = gp_trcov(gp, x);
    m = size(Phi_f,2);
    ns = eye(m,m)*S(1,1);
    
    L = chol(Phi_f'*Phi_f + ns)';
    y = Phi_a*(L'\(L\(Phi_f'*ty)));

    
    if nargout > 1
        VarY = sum(Phi_a/L',2)*S(1,1);
    end
    if nargout > 2
        error('gp_pred with three output arguments is not implemented for SSGP!')
    end
end