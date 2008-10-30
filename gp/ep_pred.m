function [Ef, Varf, p1] = ep_pred(gp, tx, ty, x, varargin)
%EP_PRED	Predictions with Gaussian Process EP
%
%	Description
%	Y = EP_PRED(GP, TX, TY, X) takes a gp data structure GP together with a
%	matrix X of input vectors, Matrix TX of training inputs and vector TY of 
%       training targets, and evaluates the predictive distribution at inputs. 
%       Returns a matrix Y of (noiseless) output vectors (mean(Y|X, TX, TY)). Each 
%       row of X corresponds to one input vector and each row of Y corresponds to 
%       one output vector.
%
%	Y = EP_PRED(GP, TX, TY, X, 'PARAM') in case of sparse model takes also 
%       string defining, which parameters have been optimized.
%
%	[Y, VarY] = EP_PRED(GP, TX, TY, X, VARARGIN) returns also the variances of Y 
%       (1xn vector).
%
%	See also
%	GPEP_E, GPEP_G, GP_PRED
%
% Copyright (c) 2007-2008 Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

    [tn, tnin] = size(tx);
    
    switch gp.type
      case 'FULL'
        [K, C]=gp_trcov(gp,tx);
        
        [e, edata, eprior, tautilde, nutilde, L] = gpep_e(gp_pak(gp, varargin{:}), gp, tx, ty, varargin{:});

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
                switch gp.likelih.type
                  case 'probit'
                    p1(i1,1)=normcdf(Ef(i1,1)/sqrt(1+Varf(i1))); % Probability p(y_new=1)
                  case 'poisson'
                    p1 = NaN;
                end
            end
        end
        
      case 'FIC'
        param = varargin{1};
        % Here tstind = 1 if the prediction is made for the training set 
        if nargin > 1
            tstind = varargin{2};
        end
        
        u = gp.X_u;
        K_fu = gp_cov(gp, tx, u);         % f x u
        K_uu = gp_trcov(gp, u);          % u x u, noiseles covariance K_uu
        K_uu = (K_uu+K_uu')./2;          % ensure the symmetry of K_uu
        kstarstar=gp_trvar(gp, x);
        
        if length(varargin) < 1
            error('The argument telling the optimzed/sampled parameters has to be provided.') 
        end

        [e, edata, eprior, tautilde, nutilde, L, La, b] = gpep_e(gp_pak(gp, param), gp, tx, ty, param);

        % From this on evaluate the prediction
        % See Snelson and Ghahramani (2007) for details 
        %        p=iLaKfu*(A\(iLaKfu'*myytilde));
        p = b';
        
        ntest=size(x,1);
        
        K_nu=gp_cov(gp,x,u);
        Ef = K_nu*(K_uu\(K_fu'*p));
        
        % 
        if tstind == 1
            [Kv_ff, Cv_ff] = gp_trvar(gp, x);
            Luu = chol(K_uu)';
            B=Luu\(K_fu');
            Qv_ff=sum(B.^2)';
            Lav = Cv_ff-Qv_ff;
            Ef = Ef + Lav.*p;
        end
        
        if nargout > 1
            % Compute variances of predictions
            %Varf(i1,1)=kstarstar(i1) - (sum(Knf(i1,:).^2./La') - sum((Knf(i1,:)*L).^2));
            Luu = chol(K_uu)';
            B=Luu\(K_fu');   
            B2=Luu\(K_nu');   
            Varf = kstarstar - sum(B2'.*(B*(repmat(La,1,size(K_uu,1)).\B')*B2)',2)  + sum((K_nu*(K_uu\(K_fu'*L))).^2, 2);

            if tstind == 1
                La = diag(La);
                Lav = diag(Lav);
                Qff = B2'*B;
                Varf = Varf - diag( Qff/La*Lav - Qff*L*L'*Lav + Lav/La*Qff + Lav/La*Lav - Lav*L*L'*Qff - Lav*L*L'*Lav);
            end
            
            for i1=1:ntest
                switch gp.likelih.type
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
            kstarstar = gp_trvar(gp, x);
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

            ntest=size(x,1);
            for i1=1:ntest
                switch gp.likelih.type
                  case 'probit'
                    p1(i1,1)=normcdf(Ef(i1,1)/sqrt(1+Varf(i1))); % Probability p(y_new=1)
                  case 'poisson'
                    p1 = NaN;
                end
            end
        end
        % ============================================================
        % CS+FIC
        % ============================================================
      case 'CS+FIC'
        param = varargin{1};
        % Here tstind = 1 if the prediction is made for the training set 
        if nargin > 1
            tstind = varargin{2};
        end

        u = gp.X_u;
        m = length(u);
        n = size(tx,1);
        cf_orig = gp.cf;
        ncf = length(gp.cf);
        
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

        K_fu = gp_cov(gp, tx, u);   % f x u
        K_uu = gp_trcov(gp, u);     % u x u, noiseles covariance K_uu
        K_uu = (K_uu+K_uu')./2;     % ensure the symmetry of K_uu
        K_nu=gp_cov(gp,x,u);
        % evaluate also Lav if the prediction is made for training set
        if tstind == 1
            [Kv_ff, Cv_ff] = gp_trvar(gp, x);
            Luu = chol(K_uu)';
            B=Luu\(K_fu');
            Qv_ff=sum(B.^2)';
            Lav = Cv_ff-Qv_ff;            
        end
        gp.cf = cf2;         
        Kcs_nf = gp_cov(gp, x, tx);
        gp.cf = cf_orig;
        
        if tstind == 1
            Kcs_nf = Kcs_nf + sparse(1:n,1:n,Lav,n,n);
        end

        if length(varargin) < 1
            error('The argument telling the optimized/sampled parameters has to be provided.')
        end

        [e, edata, eprior, tautilde, nutilde, L, La, b] = gpep_e(gp_pak(gp, param), gp, tx, ty, param);

        p = b';
        ntest=size(x,1);
        Ef = K_nu*(K_uu\(K_fu'*p)) + Kcs_nf*p;
        
        % Evaluate the variance
        if nargout > 1
            Luu = chol(K_uu)';
            B=Luu\(K_fu');   
            Knn_v = gp_trvar(gp,x);
            B2=Luu\(K_nu');
            Varf = Knn_v - sum(B2'.*(B*(La\B')*B2)',2)  + sum((B2'*(B*L)).^2, 2);
            p = amd(La);
            Varf = Varf - sum((Kcs_nf(:,p)/chol(La(p,p))).^2,2);
            Varf = Varf + sum((Kcs_nf*L).^2, 2);
            Varf = Varf - 2.*sum((Kcs_nf*(La\K_fu)).*(K_uu\K_nu')',2) + 2.*sum((Kcs_nf*L).*(L'*K_fu*(K_uu\K_nu'))' ,2);
            
            % Test
% $$$             Varf = Knn_v - diag( Kcs_nf*(La\Kcs_nf') - Kcs_nf*L*L'*Kcs_nf' + Kcs_nf*(La\B'*B2) - Kcs_nf*L*L'*B'*B2 ...
% $$$                                  + B2'*B*(La\Kcs_nf') + B2'*B*(La\B'*B2) - B2'*B*L*L'*Kcs_nf' - B2'*B*L*L'*B'*B2 );
% $$$             P = K_nu/K_uu*K_fu';
% $$$             Varf = Knn_v - diag( P/La*P' + P/La*Kcs_nf' - P*L*L'*Kcs_nf' - P*L*L'*P'...
% $$$                                  + Kcs_nf/La*P' + Kcs_nf/La*Kcs_nf' - Kcs_nf*L*L'*P' - Kcs_nf*L*L'*Kcs_nf');

            

% $$$             Varf = diag(K_nu*(K_uu\K_fu') + Kcs_nf) - diag( Kcs_nf*(La\Kcs_nf') - Kcs_nf*L*L'*Kcs_nf' + Kcs_nf*(La\B'*B2) - Kcs_nf*L*L'*B'*B2 ...
% $$$                                  + B2'*B*(La\Kcs_nf') + B2'*B*(La\B'*B2) - B2'*B*L*L'*Kcs_nf' - B2'*B*L*L'*B'*B2 );
            
            
            for i1=1:ntest
                switch gp.likelih.type
                    case 'probit'
                        p1(i1,1)=normcdf(Ef(i1,1)/sqrt(1+Varf(i1))); % Probability p(y_new=1)
                    case 'poisson'
                        p1 = NaN;
                end
            end
        end
      case 'SSGP'
        if length(varargin) < 1
            error('The argument telling the optimzed/sampled parameters has to be provided.') 
        end
        param = varargin{1};

        [e, edata, eprior, tautilde, nutilde, L, S, b] = gpep_e(gp_pak(gp, param), gp, tx, ty, param);
        
        Phi_f = gp_trcov(gp, tx);
        Phi_a = gp_trcov(gp, x);

        m = size(Phi_f,2);
        ntest=size(x,1);
        
        Ef = Phi_a*(Phi_f'*b');
        
        if nargout > 1
            % Compute variances of predictions
            %Varf(i1,1)=kstarstar(i1) - (sum(Knf(i1,:).^2./La') - sum((Knf(i1,:)*L).^2));
            Varf = sum(Phi_a.^2,2) - sum(Phi_a.*((Phi_f'*(repmat(S,1,m).*Phi_f))*Phi_a')',2) + sum((Phi_a*(Phi_f'*L)).^2,2);
            for i1=1:ntest
                switch gp.likelih.type
                  case 'probit'
                    p1(i1,1)=normcdf(Ef(i1,1)/sqrt(1+Varf(i1))); % Probability p(y_new=1)
                  case 'poisson'
                    p1 = NaN;
                end
            end
        end
    end
end

