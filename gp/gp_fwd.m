function [y, VarY] = gp_fwd(gp, tx, ty, x, varargin)
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
%       BUGS: - only 1 output allowed
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
    Sinv = K_uu+K_fu'*iLaKfu;
    
    %y = K_nu*(Sinv\(K_fu'*(ty./Lav)));
    iKuuKufiLa = K_uu\iLaKfu';
    y = K_nu*(iKuuKufiLa*ty + iKuuKufiLa*(K_fu*(Sinv\(K_fu'*(ty./Lav)))));
    if nargout > 1
        % VarY = Knn - Qnn + Knu*S*Kun
        B=Luu\(K_nu');
        Qv_nn=sum(B.^2)';
        % Vector of diagonal elements of covariance matrix
        L = chol(K_uu+K_fu'*iLaKfu)';
        b = L\K_nu';
        Kv_nn = gp_trvar(gp,x);
        VarY = Kv_nn - Qv_nn + sum(b.^2)';
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
        [Kbl_ff, Cbl_ff] = gp_trcov(gp, x(ind{i},:));
        La{i} = Cbl_ff - Qbl_ff;
        iLaKfu(ind{i},:) = La{i}\K_fu(ind{i},:);    % Check if works by changing inv(La{i})!!!
    end
    A = K_uu+K_fu'*iLaKfu;
    A = (A+A')./2;            % Ensure symmetry

    %    y = K_nu*(A\(K_fu'*(ty./Lav)));   
    p=zeros(size(ty));
    for i=1:length(ind)
        iLaty = La{i}\ty(ind{i});
        p(ind{i}) = (iLaty + La{i}\(K_fu(ind{i},:)*(A\(K_fu(ind{i},:)'*(iLaty)))));
    end
    iKuuKuf = K_uu\K_fu';
    y=zeros(size(x));
    for i=1:length(tstind)
        K_nf = gp_cov(gp, x(tstind{i},:), tx(tstind{i},:));         % n x u
        notinblock = [];
        for j=1:length(tstind)
            if j~=i
                notinblock = [notinblock ; tstind{j}];
            end
        end
        K_nu = gp_cov(gp, x(notinblock,:), u);         % n x u
        y(tstind{i}) = K_nf*p(tstind{i});
        y(notinblock) = K_nu*iKuuKuf(:,notinblock)*p(notinblock);
    end
    if nargout > 1
        error('The variaance is not implemented for PIC_BLOCK yet! \n')
% $$$         % VarY = Knn - Qnn + Knu*S*Kun
% $$$         B=Luu\(K_nu');
% $$$         Qv_nn=sum(B.^2)';
% $$$         % Vector of diagonal elements of covariance matrix
% $$$         L = chol(K_uu+K_fu'*iLaKfu)';
% $$$         b = L\K_nu';
% $$$         Kv_nn = gp_trvar(gp,x);
% $$$         VarY = Kv_nn - Qv_nn + sum(b.^2)';
    end
end