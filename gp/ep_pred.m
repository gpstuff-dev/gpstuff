function [Ef, Varf, p1] = ep_pred(gp, tx, ty, x, varargin)
%EP_PRED	Predictions with Gaussian Process EP
%
%	Description
%	Y = EP_PRED(GP, TX, TY, X) takes a gp data structure GP together with a
%	matrix X of input vectors, Matrix TX of training inputs and vector TY of 
%       training targets, and forward propagates the inputs through the gp to generate 
%       a matrix Y of (noiseless) output vectors (mean(Y|X)). Each row of X 
%       corresponds to one input vector and each row of Y corresponds to one output 
%       vector.
%
%	Y = EP_PRED(GP, TX, TY, X, U) in case of sparse model takes also inducing 
%       points U.
%
%	[Y, VarY] = EP_PRED(GP, TX, TY, X) returns also the variances of Y 
%       (1xn vector).
%
%       BUGS: - only 1 output allowed
%
%	See also
%	GP, GP_PAK, GP_UNPAK
%
% Copyright (c) 2006 Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

    [tn, tnin] = size(tx);
    
    switch gp.type
      case 'FULL'
        [K, C]=gp_trcov(gp,tx);
        
        [e, edata, eprior, tautilde, nutilde, L] = gpep_e(gp_pak(gp,'hyper'), gp, tx, ty, 'hyper', varargin{:});

        sqrttautilde = sqrt(tautilde);
        Stildesqroot = diag(sqrttautilde);
        
        z=Stildesqroot*(L'\(L\(Stildesqroot*(C*nutilde))));
        
        kstarstar=gp_trvar(gp, x);

        ntest=size(x,1);
        
        K_nf=gp_cov(gp,x,tx);
        V = (L\Stildesqroot)*K_nf';
        for i1=1:ntest
            % Compute covariance between observations
            Ef(i1,1)=K_nf(i1,:)*(nutilde-z);
            Varf(i1,1)=kstarstar(i1)-V(:,i1)'*V(:,i1);
            switch gp.likelih
              case 'probit'
                p1(i1,1)=normcdf(Ef(i1,1)/sqrt(1+Varf(i1))); % Probability p(y_new=1)
              case 'poisson'
                p1 = [];
            end
        end
        
      case 'FIC'
        
        u = gp.X_u;
        K_fu = gp_cov(gp, tx, u);         % f x u
        K_uu = gp_trcov(gp, u);          % u x u, noiseles covariance K_uu
        K_uu = (K_uu+K_uu')./2;          % ensure the symmetry of K_uu

        [e, edata, eprior, tautilde, nutilde, iLaKfu, La] = gpep_e(gp_pak(gp,'hyper'), gp, tx, ty, 'hyper', varargin);

        myytilde = nutilde./tautilde;
        
        Luu = chol(K_uu)';
        % Evaluate the Lambda (La) 
        % Q_ff = K_fu*inv(K_uu)*K_fu'
        % Here we need only the diag(Q_ff), which is evaluated below
        B=Luu\(K_fu');       % u x f
        Qv_ff=sum(B.^2)';
        iLaKfu = zeros(size(K_fu));  % f x u, 
        for i=1:length(La)
            iLaKfu(i,:) = K_fu(i,:)./La(i);  % f x u 
        end
        A = K_uu+K_fu'*iLaKfu;
        A = (A+A')./2;     % Ensure symmetry
        A = chol(A)';
        L = iLaKfu/A';

% $$$         A = K_uu+K_fu'*iLaKfu;
% $$$         A = (A+A')./2;               % Ensure symmetry
% $$$         L = iLaKfu/chol(A);
        p = myytilde./La - L*(L'*myytilde);
        
        kstarstar=gp_trvar(gp, x);

        ntest=size(x,1);
        
        K_nu=gp_cov(gp,x,u);
        Knf = K_nu*(K_uu\K_fu');
        Ef = Knf*p;
        for i1=1:ntest
            % Compute covariance between observations
            Varf(i1,1)=kstarstar(i1) - ((Knf(i1,:)./La')*Knf(i1,:)' - Knf(i1,:)*L*L'*Knf(i1,:)');
            switch gp.likelih
              case 'probit'
                p1(i1,1)=normcdf(Ef(i1,1)/sqrt(1+Varf(i1))); % Probability p(y_new=1)
              case 'poisson'
                p1 = [];
            end
        end
        
      case 'PIC_BLOCK'
        
    end
end
