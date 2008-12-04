function [Ef, Varf, p1] = la_pred(gp, tx, ty, x, varargin)
%LA_PRED	Predictions with Gaussian Process Laplace approximation
%
%	Description
%	Y = LA_PRED(GP, TX, TY, X) takes a gp data structure GP together with a
%	matrix X of input vectors, Matrix TX of training inputs and vector TY of 
%       training targets, and evaluates the predictive distribution at inputs. 
%       Returns a matrix Y of (noiseless) output vectors (mean(Y|X, TX, TY)). Each 
%       row of X corresponds to one input vector and each row of Y corresponds to 
%       one output vector.
%
%	Y = LA_PRED(GP, TX, TY, X, 'PARAM') in case of sparse model takes also 
%       string defining, which parameters have been optimized.
%
%	[Y, VarY] = LA_PRED(GP, TX, TY, X, VARARGIN) returns also the variances of Y 
%       (1xn vector).
%
%	See also
%	GPLA_E, GPLA_G, GP_PRED
%
% Copyright (c) 2007-2008 Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

    [tn, tnin] = size(tx);

    switch gp.type
      case 'FULL'
        [e, edata, eprior, f, L, La2, b] = gpla_e(gp_pak(gp,'hyper'), gp, tx, ty, 'hyper', varargin{:});

        W = La2;
        deriv = b;
        ntest=size(x,1);

        % Evaluate the expectation
        K_nf = gp_cov(gp,x,tx);
        Ef = K_nf*deriv;

        % Evaluate the variance
        if nargout > 1
            kstarstar = gp_trvar(gp,x);
            V = L\(sqrt(W)*K_nf');
            for i1=1:ntest
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
        if nargin > 5
            tstind = varargin{2};
            if length(tstind) ~= size(tx,1)
                error('tstind (if provided) has to be of same lenght as tx.')
            end
        else
             tstind = [];
        end

        u = gp.X_u;
        K_fu = gp_cov(gp, tx, u);         % f x u
        K_uu = gp_trcov(gp, u);          % u x u, noiseles covariance K_uu
        K_uu = (K_uu+K_uu')./2;          % ensure the symmetry of K_uu

        m = size(u,1);

        if length(varargin) < 1
            error('The argument telling the optimized/sampled parameters has to be provided.')
        end

        [e, edata, eprior, f, L, La2, b] = gpla_e(gp_pak(gp, param), gp, tx, ty, param);

        deriv = b;
        ntest=size(x,1);

        K_nu=gp_cov(gp,x,u);
        Ef = K_nu*(K_uu\(K_fu'*deriv));

        % if the prediction is made for training set, evaluate Lav also for prediction points
        if ~isempty(tstind)
            [Kv_ff, Cv_ff] = gp_trvar(gp, x(tstind,:));
            Luu = chol(K_uu)';
            B=Luu\(K_fu');
            Qv_ff=sum(B.^2)';
            Lav = zeros(size(La));
            Lav(tstind) = Cv_ff-Qv_ff;
            Ef = Ef + Lav.*p;
        end

        
        % Evaluate the variance
        if nargout > 1
            W = -feval(gp.likelih.fh_hessian, gp.likelih, ty, f, 'latent');
            kstarstar = gp_trvar(gp,x);
            Luu = chol(K_uu)';
            La = W.*La2;
            Lahat = 1 + La;
            B = (repmat(sqrt(W),1,m).*K_fu);

            % Components for (I + W^(1/2)*(Qff + La2)*W^(1/2))^(-1) = Lahat^(-1) - L2*L2'
            B2 = repmat(Lahat,1,m).\B;
            A2 = K_uu + B'*B2; A2=(A2+A2)/2;
            L2 = B2/chol(A2);

            % Set params for K_nf
            BB=Luu\(B');
            BB2=Luu\(K_nu');
            Varf = kstarstar - sum(BB2'.*(BB*(repmat(Lahat,1,size(K_uu,1)).\BB')*BB2)',2)  + sum((K_nu*(K_uu\(B'*L2))).^2, 2);
            
            % if the prediction is made for training set, evaluate Lav also for prediction points
            if ~isempty(tstind)
                Varf(tstind) = Varf(tstind) - 2.*sum( BB2(:,tstind)'.*(repmat((La.\Lav(tstind)),1,m).*BB'),2) ...
                    + 2.*sum( BB2(:,tstind)'*(BB*L).*(repmat(Lav(tstind),1,m).*L), 2)  ...
                    - Lav(tstind)./La.*Lav(tstind) + sum((repmat(Lav(tstind),1,m).*L).^2,2);                
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
        u = gp.X_u;
        K_fu = gp_cov(gp, tx, u);         % f x u
        K_uu = gp_trcov(gp, u);          % u x u, noiseles covariance K_uu
        K_uu = (K_uu+K_uu')./2;          % ensure the symmetry of K_uu
        K_nu=gp_cov(gp,x,u);

        ind = gp.tr_index;
        param = varargin{1};
        tstind = varargin{2};
        ntest = size(x,1);
        m = size(u,1);

        if length(varargin) < 1
            error('The argument telling the optimized/sampled parameters has to be provided.')
        end

        [e, edata, eprior, f, L, La2, b] = gpla_e(gp_pak(gp, param), gp, tx, ty, param);

        deriv = b;

        iKuuKuf = K_uu\K_fu';
        w_bu=zeros(length(x),length(u));
        w_n=zeros(length(x),1);
        for i=1:length(ind)
            w_bu(tstind{i},:) = repmat((iKuuKuf(:,ind{i})*deriv(ind{i},:))', length(tstind{i}),1);
            K_nf = gp_cov(gp, x(tstind{i},:), tx(ind{i},:));              % n x u
            w_n(tstind{i},:) = K_nf*deriv(ind{i},:);
        end

        Ef = K_nu*(iKuuKuf*deriv) - sum(K_nu.*w_bu,2) + w_n;

        % Evaluate the variance
        if nargout > 1
            W = -feval(gp.likelih.fh_hessian, gp.likelih, ty, f, 'latent');
            kstarstar = gp_trvar(gp,x);
            sqrtW = sqrt(W);
            % Components for (I + W^(1/2)*(Qff + La2)*W^(1/2))^(-1) = Lahat^(-1) - L2*L2'
            for i=1:length(ind)
                La{i} = diag(sqrtW(ind{i}))*La2{i}*diag(sqrtW(ind{i}));
                Lahat{i} = eye(size(La{i})) + La{i};
            end
            B = (repmat(sqrt(W),1,m).*K_fu);
            for i=1:length(ind)
                B2(ind{i},:) = Lahat{i}\B(ind{i},:);
            end
            A2 = K_uu + B'*B2; A2=(A2+A2)/2;
            L2 = B2/chol(A2);

            iKuuB = K_uu\B';
            KnfL2 = K_nu*(iKuuB*L2);
            Varf = zeros(length(x),1);
            for i=1:length(ind)
                v_n = gp_cov(gp, x(tstind{i},:), tx(ind{i},:)).*repmat(sqrtW(ind{i},:)',length(tstind{i}),1);              % n x u
                v_bu = K_nu(tstind{i},:)*iKuuB(:,ind{i});
                KnfLa = K_nu*(iKuuB(:,ind{i})/chol(Lahat{i}));
                KnfLa(tstind{i},:) = KnfLa(tstind{i},:) - (v_bu + v_n)/chol(Lahat{i});
                Varf = Varf + sum((KnfLa).^2,2);
                KnfL2(tstind{i},:) = KnfL2(tstind{i},:) - v_bu*L2(ind{i},:) + v_n*L2(ind{i},:);
            end
            Varf = kstarstar - (Varf - sum((KnfL2).^2,2));

            for i1=1:ntest
                switch gp.likelih.type
                  case 'probit'
                    p1(i1,1)=normcdf(Ef(i1,1)/sqrt(1+Varf(i1))); % Probability p(y_new=1)
                  case 'poisson'
                    p1 = NaN;
                end
            end
        end
      case 'CS+FIC'
        param = varargin{1};
        % Here tstind = 1 if the prediction is made for the training set 
        if nargin > 5
            tstind = varargin{2};
            if length(tstind) ~= size(tx,1)
                error('tstind (if provided) has to be of same lenght as tx.')
            end
        else
             tstind = [];
        end

        n = size(tx,1);
        n2 = size(x,1);
        u = gp.X_u;
        m = length(u);
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

        [Kv_ff, Cv_ff] = gp_trvar(gp, tx);  % f x 1  vector
        K_fu = gp_cov(gp, tx, u);         % f x u
        K_uu = gp_trcov(gp, u);    % u x u, noiseles covariance K_uu
        K_uu = (K_uu+K_uu')./2;     % ensure the symmetry of K_uu
        K_nu=gp_cov(gp,x,u);
        % evaluate also Lav if the prediction is made for training set
        if ~isempty(tstind)
            [Kv_ff, Cv_ff] = gp_trvar(gp, x(tstind,:));
            Luu = chol(K_uu)';
            B=Luu\(K_fu');
            Qv_ff=sum(B.^2)';
            Lav = Cv_ff-Qv_ff;
        end
        
        gp.cf = cf2;
        Kcs_nf = gp_cov(gp, x, tx);
        gp.cf = cf_orig;

        if ~isempty(tstind)
            Kcs_nf = Kcs_nf + sparse(tstind,1:n,Lav,n2,n);
        end

        if length(varargin) < 1
            error('The argument telling the optimized/sampled parameters has to be provided.')
        end

        [e, edata, eprior, f, L, La2, b] = gpla_e(gp_pak(gp, param), gp, tx, ty, param);

        deriv = b;
        ntest=size(x,1);

        % Knf = K_nu*(K_uu\K_fu');
        % Ef = Knf*p;
        Ef = K_nu*(K_uu\(K_fu'*deriv)) + Kcs_nf*deriv;

        % Evaluate the variance
        if nargout > 1
            W = -feval(gp.likelih.fh_hessian, gp.likelih, ty, f, 'latent');
            sqrtW = sparse(1:tn,1:tn,sqrt(W),tn,tn);
            kstarstar = gp_trvar(gp,x);
            Luu = chol(K_uu)';
            Lahat = sparse(1:tn,1:tn,1,tn,tn) + sqrtW*La2*sqrtW;
            B = sqrtW*K_fu;

            % Components for (I + W^(1/2)*(Qff + La2)*W^(1/2))^(-1) = Lahat^(-1) - L2*L2'
            B2 = Lahat\B;
            A2 = K_uu + B'*B2; A2=(A2+A2)/2;
            L2 = B2/chol(A2);

            % Set params for K_nf
            BB=Luu\(B)';    % sqrtW*K_fu
            BB2=Luu\(K_nu');
            KcssW = Kcs_nf*sqrtW;
            Varf = kstarstar - sum(BB2'.*(BB*(Lahat\BB')*BB2)',2);
            Varf = Varf + sum((K_nu*(K_uu\(B'*L2))).^2, 2);
            %Varf = Varf - sum((KcssW/chol(Lahat)).^2,2);
            m = amd(Lahat);
            tmp = sum((KcssW(:,m)/chol(Lahat(m,m))).^2,2);
            Varf = Varf - tmp;
            
            Varf = Varf + sum((KcssW*L2).^2, 2);
            %VarY = VarY - 2.*diag((Kcs_nf*iLaKfu)*(K_uu\K_nu')) + 2.*diag((Kcs_nf*L)*(L'*K_fu*(K_uu\K_nu')));
            Varf = Varf - 2.*sum((KcssW*(Lahat\B)).*(K_uu\K_nu')',2);
            Varf = Varf + 2.*sum((KcssW*L2).*(L2'*B*(K_uu\K_nu'))' ,2);
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