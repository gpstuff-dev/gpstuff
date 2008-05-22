function [Ef, Varf, p1] = la_pred(gp, tx, ty, x, varargin)
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
        u = gp.X_u;
        K_fu = gp_cov(gp, tx, u);         % f x u
        K_uu = gp_trcov(gp, u);          % u x u, noiseles covariance K_uu
        K_uu = (K_uu+K_uu')./2;          % ensure the symmetry of K_uu

        m = size(u,1);

        if length(varargin) < 1
            error('The argument telling the optimized/sampled parameters has to be provided.')
        end

        [e, edata, eprior, f, L, La2, b] = gpla_e(gp_pak(gp, varargin{:}), gp, tx, ty, varargin{:});

        deriv = b;
        ntest=size(x,1);

        ntest=size(x,1);

        K_nu=gp_cov(gp,x,u);
        % Knf = K_nu*(K_uu\K_fu');
        % Ef = Knf*p;
        Ef = K_nu*(K_uu\(K_fu'*deriv));

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

        [e, edata, eprior, f, L, La2, b] = gpla_e(gp_pak(gp,'hyper'), gp, tx, ty, 'hyper', varargin{:});

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
        gp.cf = cf2;
        Kcs_nf = gp_cov(gp, x, tx);
        gp.cf = cf_orig;

        if length(varargin) < 1
            error('The argument telling the optimized/sampled parameters has to be provided.')
        end

        [e, edata, eprior, f, L, La2, b] = gpla_e(gp_pak(gp,'hyper'), gp, tx, ty, 'hyper', varargin{:});

        deriv = b;
        ntest=size(x,1);

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
%
% ==============================================================
% Begin of the nested functions
% ==============================================================
%
    function Hessian = hessian(f, likelihood)
        switch likelihood
            case 'probit'
                z = ty.*f;
                Hessian = (normpdf(f)./normcdf(z)).^2 + z.*normpdf(f)./normcdf(z);
            case 'poisson'
                Hessian = gp.avgE.*exp(f);
        end
    end
end