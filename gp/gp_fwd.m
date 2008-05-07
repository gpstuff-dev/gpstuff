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
%	[Y, VarY, NoisyY] = GP_FWD(GP, TX, TY, X) returns the noisy prediction for
%       Y (1xn vector). These are needed for example in the Student-t noise model.
%       NOTE! in FIC/PIC the prediction NoisyY is made at the training points!
%
%	See also
%	GP, GP_PAK, GP_UNPAK
%

% Copyright (c) 2000 Aki Vehtari
% Copyright (c) 2006 Jarno Vanhatalo

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
            
%             QnfL = KnuiKuu*(K_fu'*L);
%             dQnfiLaQfn = sum(KnuiKuu.*(K_fu'*iLaKfu*KnuiKuu')',2);
%             
%             VarY = kstarstar - (dQnfiLaQfn - Varf2 +Varf1 - sum(QnfL-QnbL + KnbL,2) );
            
%             kstarstar = gp_trvar(gp, x); 
%             KnfL = K_nu*(iKuuKuf*L);
%             Varf = zeros(length(x),1);
%             for i=1:length(ind)
%                 v_n = gp_cov(gp, x(tstind{i},:), tx(ind{i},:));              % n x u
%                 v_bu = K_nu(tstind{i},:)*iKuuKuf(:,ind{i});
%                 KnfLa = K_nu*(iKuuKuf(:,ind{i})/chol(La{i}));
%                 KnfLa(tstind{i},:) = KnfLa(tstind{i},:) - (v_bu + v_n)/chol(La{i});
%                 Varf = Varf + sum((KnfLa).^2,2);
%                 KnfL(tstind{i},:) = KnfL(tstind{i},:) - v_bu*L(ind{i},:) + v_n*L(ind{i},:);
%             end
%             VarY = kstarstar - (Varf - sum((KnfL).^2,2));  
            

%             Knn_v = gp_trvar(gp,x);
%             iKuuKuf = K_uu\K_fu';
%             v_bu = zeros(length(x),length(tx));
%             v_n = zeros(length(x),length(tx));
%             for i=1:length(ind)
%                 K_nf = gp_cov(gp, x(tstind{i},:), tx(ind{i},:));              % n x u
%                 v_bu(tstind{i},ind{i}) = K_nu(tstind{i},:)*iKuuKuf(:,ind{i});
%                 v_n(tstind{i},ind{i}) = K_nf;
%             end
%             K_nf = K_nu*iKuuKuf - v_bu + v_n;
% 
%             ntest=size(x,1);
%             VarY = zeros(ntest,1);
%             %Varf = zeros(ntest,ntest);
%             for i=1:length(ind)
%                 VarY = VarY + sum((K_nf(:,ind{i})/chol(La{i})).^2,2);
%                 %Varf = Varf + (K_nf(:,ind{i})/La{i})*K_nf(:,ind{i})' - K_nf(:,ind{i})*L(ind{i},:)*L(ind{i},:)'*K_nf(:,ind{i})';
%             end
%             %Varf = kstarstar - diag(Varf);
% 
%             VarY = Knn_v - (VarY - sum((K_nf*L).^2,2));
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