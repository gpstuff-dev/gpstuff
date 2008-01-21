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
        
        kstarstar = gp_trvar(gp, x);

        ntest=size(x,1);
        
        K_nf=gp_cov(gp,x,tx);
        Ef=K_nf*(nutilde-z);
        
        if nargout > 1
            V = (L\Stildesqroot)*K_nf';
            for i1=1:ntest
                % Compute covariance between observations
                Varf(i1,1)=kstarstar(i1)-V(:,i1)'*V(:,i1);
                switch gp.likelih
                  case 'probit'
                    p1(i1,1)=normcdf(Ef(i1,1)/sqrt(1+Varf(i1))); % Probability p(y_new=1)
                  case 'poisson'
                    p1 = NaN;
                end
            end
        end
        
      case 'FIC'
        u = gp.X_u;
        K_fu = gp_cov(gp, tx, u);         % f x u
        K_uu = gp_trcov(gp, u);          % u x u, noiseles covariance K_uu
        K_uu = (K_uu+K_uu')./2;          % ensure the symmetry of K_uu
        kstarstar=gp_trvar(gp, x);
        
        if length(varargin) < 1
            error('The argument telling the optimzed/sampled parameters has to be provided.') 
        end

        [e, edata, eprior, tautilde, nutilde, L, La, b] = gpep_e(gp_pak(gp, varargin{:}), gp, tx, ty, varargin{:});

        p = b';
        
        ntest=size(x,1);
        
        K_nu=gp_cov(gp,x,u);
        %Knf = K_nu*(K_uu\K_fu'); 
        %Ef = Knf*p;
        Ef = K_nu*(K_uu\(K_fu'*p));
        
        if nargout > 1
            % Compute variances of predictions
            %Varf(i1,1)=kstarstar(i1) - (sum(Knf(i1,:).^2./La') - sum((Knf(i1,:)*L).^2));
            Luu = chol(K_uu)';
            B=Luu\(K_fu');   
            B2=Luu\(K_nu');   
            Varf = kstarstar - sum(B2'.*(B*(repmat(La,1,size(K_uu,1)).\B')*B2)',2)  + sum((K_nu*(K_uu\(K_fu'*L))).^2, 2);
            for i1=1:ntest
                switch gp.likelih
                  case 'probit'
                    p1(i1,1)=normcdf(Ef(i1,1)/sqrt(1+Varf(i1))); % Probability p(y_new=1)
                  case 'poisson'
                    p1 = NaN;
                end
            end
        end
        
      case 'PIC_BLOCK'
        % Calculate some help matrices  
        u = gp.X_u;
        ind = gp.tr_index;
        param = varargin{1};
        tstind = varargin{2};        % block indeces for test points
        
        K_fu = gp_cov(gp, tx, u);         % f x u
        K_nu = gp_cov(gp, x, u);         % n x u   
        K_uu = gp_trcov(gp, u);    % u x u, noiseles covariance K_uu
        kstarstar = gp_trvar(gp, x);
        
        if length(varargin) < 1
            error('The argument telling the optimzed/sampled parameters has to be provided.') 
        end
        
        [e, edata, eprior, tautilde, nutilde, L, La, b] = gpep_e(gp_pak(gp, param), gp, tx, ty, param);

        % From this on evaluate the prediction
        % See Snelson and Ghahramani (2007) for details 
        %        p=iLaKfu*(A\(iLaKfu'*myytilde));
        p = b';

        iKuuKuf = K_uu\K_fu';
        
        w_bu=zeros(length(x),length(u));
        w_n=zeros(length(x),1);
        for i=1:length(ind)
            w_bu(tstind{i},:) = repmat((iKuuKuf(:,ind{i})*p(ind{i},:))', length(tstind{i}),1);
            K_nf = gp_cov(gp, x(tstind{i},:), tx(ind{i},:));              % n x u
            w_n(tstind{i},:) = K_nf*p(ind{i},:);
        end
        
        Ef = K_nu*(iKuuKuf*p) - sum(K_nu.*w_bu,2) + w_n;

        if nargout > 1
            KnfL = K_nu*(iKuuKuf*L);
            Varf = zeros(length(x),1);
            for i=1:length(ind)
                v_n = gp_cov(gp, x(tstind{i},:), tx(ind{i},:));              % n x u
                v_bu = K_nu(tstind{i},:)*iKuuKuf(:,ind{i});
                KnfLa = K_nu*(iKuuKuf(:,ind{i})/chol(La{i}));
                KnfLa(tstind{i},:) = KnfLa(tstind{i},:) - (v_bu + v_n)/chol(La{i});
                Varf = Varf + sum((KnfLa).^2,2);
                KnfL(tstind{i},:) = KnfL(tstind{i},:) - v_bu*L(ind{i},:) + v_n*L(ind{i},:);
            end
            Varf = kstarstar - (Varf - sum((KnfL).^2,2));  
            
%             v_bu = zeros(length(x),length(tx));
%             %v_bu = zeros(size(iKuuKuf));
%             v_n = zeros(length(x),length(tx));
%             for i=1:length(ind)
%                 K_nf = gp_cov(gp, x(tstind{i},:), tx(ind{i},:));              % n x u
%                 v_bu(tstind{i},ind{i}) = K_nu(tstind{i},:)*iKuuKuf(:,ind{i});
%                 v_n(tstind{i},ind{i}) = K_nf;
%             end
%             K_nf = K_nu*iKuuKuf - v_bu + v_n;
%             
%             ntest=size(x,1);
%             Varf = zeros(ntest,1);
%             %Varf = zeros(ntest,ntest);
%             for i=1:length(ind)
%                 Varf = Varf + sum((K_nf(:,ind{i})/chol(La{i})).^2,2);
% % $$$                 max(max(inv(La{i})))
% % $$$                 min(min(inv(La{i})))
%                 %Varf = Varf + (K_nf(:,ind{i})/La{i})*K_nf(:,ind{i})' - K_nf(:,ind{i})*L(ind{i},:)*L(ind{i},:)'*K_nf(:,ind{i})';
%             end
%             %Varf = kstarstar - diag(Varf);
%             Varf = kstarstar - (Varf - sum((K_nf*L).^2,2));

            ntest=size(x,1);
            for i1=1:ntest
                switch gp.likelih
                  case 'probit'
                    p1(i1,1)=normcdf(Ef(i1,1)/sqrt(1+Varf(i1))); % Probability p(y_new=1)
                  case 'poisson'
                    p1 = NaN;
                end
            end
        end
    end
end