function [y, VarY, noisyY] = gp_fwd(gp, tx, ty, x, varargin)
%GP2FWD	Forward propagation through Gaussian Process
%
%	Description
%	Y = GP_FWD(GP, TX, TY, X) takes a gp data structure GP together with a
%	matrix X of input vectors, Matrix TX of training inputs and vector TY of 
%       training targets, and forward propagates the inputs through the gp to generate 
%       a matrix Y of (noiseless) output vectors (mean(Y|X)). Each row of X 
%       corresponds to one input vector and each row of Y corresponds to one output 
%       vector.
%
%	Y = GP_FWD(GP, TX, TY, X, U) in case of sparse model takes also inducing 
%       points U.
%
%	[Y, VarY] = GP_FWD(GP, TX, TY, X) returns also the variances of Y 
%       (1xn vector).
%
%	[Y, VarY, NoisyY] = GP_FWD(GP, TX, TY, X) returns also the noisy prediction 
%       for Y (1xn vector). These are needed for example in the Student-t noise model.
%
%	See also
%	GP, GP_PAK, GP_UNPAK
%

% Copyright (c) 2000 Aki Vehtari
% Copyright (c) 2006 Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

% Evaluate this if sparse model is used  
switch gp.type
  case 'FULL'
    K=gp_cov(gp,tx,x);
    if nargin > 4
        y=K'*(invC*ty);
    else
        [c, C]=gp_trcov(gp,tx);
        L = chol(C)';
        %    y=K'*(C\ty);
        a = L'\(L\ty);
        y = K'*a;
    end  
    if nargout > 1
        v = L\K;
        V = gp_trvar(gp,x);
        % Vector of diagonal elements of covariance matrix
        % b = L\K;
        % VarY = V - sum(b.^2)';
        VarY = V - diag(v'*v);
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
        KnfiLa = zeros(size(Knf));
        for i = 1:length(tx)
            KnfiLa(:,i) = Knf(:,i)./sqrt(Lav(i));
        end
        Knn_v = gp_trvar(gp,x);
        VarY = Knn_v - sum(KnfiLa.*KnfiLa, 2) + sum((Knf*L).^2, 2);
    end
    if nargout > 2
        randn('state', 100);
        random_vector = randn(size(y));
        K_nn = K_nu*inv(K_uu)*K_nu' + diag(Knn_v - diag(K_nu*inv(K_uu)*K_nu'));
        noisyY2 = y + chol(K_nn - KnfiLa*KnfiLa' + Knf*L*L'*Knf')' * random_vector;

        
        B=Luu\(K_nu');
        Qv_ff=sum(B.^2)';
        Lav = Cv_ff-Qv_ff;

        randn('state', 100);
        random_vector = randn(size(y));
        noisyY = y + B'*random_vector + random_vector.*sqrt(Lav) - KnfiLa'*random_vector + L'*(Knf'*random_vector);
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
        %            Qbl_ff2(ind{i},ind{i}) = B(:,ind{i})'*B(:,ind{i});
        [Kbl_ff, Cbl_ff] = gp_trcov(gp, tx(ind{i},:));
        La{i} = Cbl_ff - Qbl_ff;
        iLaKfu(ind{i},:) = La{i}\K_fu(ind{i},:);    % Check if works by changing inv(La{i})!!!
    end
    A = K_uu+K_fu'*iLaKfu;
    A = (A+A')./2;            % Ensure symmetry

    tyy = ty;
    % From this on evaluate the prediction
    % See Snelson and Ghahramani (2007) for details 
    p=iLaKfu*(A\(iLaKfu'*tyy));
    for i=1:length(ind)
        p2(ind{i},:) = La{i}\tyy(ind{i},:);
    end
    p= p2-p;
    
    %iKuuKuf = K_uu\K_fu';
    w_u = K_uu\(K_fu'*p);
    
    w_bu=zeros(length(x),length(u));
    w_n=zeros(length(x),1);
    for i=1:length(ind)
        w_bu(tstind{i},:) = repmat((K_uu\(K_fu(ind{i},:)'*p(ind{i},:)))', length(tstind{i}),1);
        K_nf = gp_cov(gp, x(tstind{i},:), tx(ind{i},:));              % n x u
        w_n(tstind{i},:) = K_nf*p(ind{i},:);
    end
    
    y = K_nu*w_u - sum(K_nu.*w_bu,2) + w_n;
    %    VarY = p;
    
    if nargout > 1
        error('The variaance is not implemented for PIC yet! \n')
% $$$         % VarY = Knn - Qnn + Knu*S*Kun
% $$$         B=Luu\(K_nu');
% $$$         Qv_nn=sum(B.^2)';
% $$$         % Vector of diagonal elements of covariance matrix
% $$$         L = chol(K_uu+K_fu'*iLaKfu)';
% $$$         b = L\K_nu';
% $$$         Kv_nn = gp_trvar(gp,x);
% $$$         VarY = Kv_nn - Qv_nn + sum(b.^2)';
    end
  case 'CS+PIC'
    % Calculate some help matrices  
    u = reshape(Gp.X_u,length(Gp.X_u)/nin,nin);
    ind = varargin{1};           % block indeces for training points
    tstind = varargin{2};        % block indeces for test points
    
    n = size(tx,1);
    
    cf_orig = Gp.cf;
    
    cf1 = {};
    cf2 = {};
    j = 1;
    k = 1;
    for i = 1:length(cf_orig) 
        if ~isfield(Gp.cf{i},'cs')
            cf1{j} = Gp.cf{i};
            j = j + 1;
        else
            cf2{k} = Gp.cf{i};
            k = k + 1;
        end         
    end
    
    Gp.cf = cf1;
    K_nu = gp_cov(Gp, x, u);            % n x u   
    [Kv_ff, Cv_ff] = gp_trvar(Gp, tx);  % 1 x f  vector
    K_fu = gp_cov(Gp, tx, u);           % f x u
    K_uu = gp_trcov(Gp, u);             % u x u, noiseles covariance K_uu
    Luu = chol(K_uu)';
    %K_nf = gp_cov(Gp,x,tx);
    
    % Evaluate the Lambda (La) for specific model
    % Q_ff = K_fu*inv(K_uu)*K_fu'
    B=Luu\K_fu';

% $$$     La = sparse(1:n,1:n,0,n,n);
% $$$     for i=1:length(ind)
% $$$         Qbl_ff = B(:,ind{i})'*B(:,ind{i});
% $$$         [Kbl_ff, Cbl_ff] = gp_trcov(Gp, tx(ind{i},:));
% $$$         La(ind{i},ind{i}) =  Cbl_ff - Qbl_ff;
% $$$     end
    
    [I,J]=find(tril(sparse(gp.tr_indvec(:,1),gp.tr_indvec(:,2),1,n,n),-1));
    q_ff = sum(B(:,I).*B(:,J));
    q_ff = sparse(I,J,q_ff,n,n);
    c_ff = gp_covvec(gp, x(I,:), x(J,:))';
    c_ff = sparse(I,J,c_ff,n,n);
    [Kv_ff, Cv_ff] = gp_trvar(gp,x);
    La = c_ff + c_ff' - q_ff - q_ff' + sparse(1:n,1:n, Cv_ff-sum(B.^2,1)',n,n);
    
    
    % Add the compact support cf to lambda
    Gp.cf = cf2;
    K_cs = gp_trcov(Gp,tx);
    K_cs_nf = gp_cov(Gp,x,tx);
    %K_cs_nf = gp_cov(Gp,x,tx);
    La = La + K_cs;
    Gp.cf = cf_orig;
    
    iLaKfu = La\K_fu;
    A = K_uu+K_fu'*iLaKfu;
    A = (A+A')./2;            % Ensure symmetry
    
    %L = iLaKfu/chol(A);        
    %K_ff = gp_trcov(Gp,tx);
    %iKff = inv(La)-L*L';
    
    %Q_nf = K_nf*iKff*K_ff;        
    
    Gp.cf = cf_orig;
    
    if size(ty,2)>1
        tyy = ty(:,i1);
    else    % Here latent values are not present
        tyy = ty;
    end
    
    
    
    % From this on evaluate the prediction
    % See Snelson and Ghahramani (2007) for details 
    p=iLaKfu*(A\(iLaKfu'*tyy));
% $$$         for i=1:length(ind)
% $$$             p2(ind{i},:) = La(ind{i},ind{i})\tyy(ind{i},:);
% $$$         end
    p2 = La\tyy;
    p= p2-p;
    
    %iKuuKuf = K_uu\K_fu';
    %w_u = K_uu\(K_fu'*p);
    Gp.cf = cf1;
    w_bu=zeros(length(x),length(u));
    %w_bu=zeros(length(x),1);
    w_n=zeros(length(x),1);
    for i=1:length(ind)
        w_bu(tstind{i},:) = repmat((K_uu\(K_fu(ind{i},:)'*p(ind{i},:)))', length(tstind{i}),1);
        %w_bu(tstind{i},:) = repmat(K_uu\(K_fu(ind{i},:)'*p(ind{i},:)))', length(tstind{i}),1);
        %w_bu = w_bu + K_nu*(K_uu\(K_fu(ind{i},:)'))*p(ind{i});
        K_nb = gp_cov(Gp, x(tstind{i},:), tx(ind{i},:));              % n x u
        w_n(tstind{i},:) = K_nb*p(ind{i},:);
    end
    %    [max(- sum(K_nu.*w_bu,2) + w_n), mean(- sum(K_nu.*w_bu,2) + w_n), min(- sum(K_nu.*w_bu,2) + w_n)]
    %y = K_nu*w_u - sum(K_nu.*w_bu,2) + w_n;
    %y = (K_nu*(K_uu\K_fu')+K_cs_nf)*p - w_bu + w_n;
    y = (K_nu*(K_uu\K_fu')+K_cs_nf)*p - sum(K_nu.*w_bu,2) + w_n;
    Gp.cf = cf_orig;
    if nargout > 1   
        error('Variance is not yet implemented for PIC! \n')
% $$$             B=Luu\(K_nu');
% $$$             Qv_nn=sum(B.^2)';
% $$$             % Vector of diagonal elements of covariance matrix
% $$$             L = chol(S)';
% $$$             b = L\K_nu';
% $$$             [Kv_nn, Cv_nn] = gp_trvar(Gp,x);
% $$$             VarY(:,:,i1) = Kv_nn - Qv_nn + sum(b.^2)';
    end
end