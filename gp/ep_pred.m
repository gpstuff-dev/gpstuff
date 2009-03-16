function [Ef, Varf, p1] = ep_pred(gp, tx, ty, x, param, predcf, tstind)
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
    if nargin < 5
        error('The argument telling the optimzed/sampled parameters has to be provided.') 
    end
    
    if nargin < 6
        predcf = [];
    end
    
    switch gp.type
      case 'FULL'
        [e, edata, eprior, tautilde, nutilde, L] = gpep_e(gp_pak(gp, param), gp, tx, ty, param);
        
        [K, C]=gp_trcov(gp,tx,predcf);
        [K, C]=gp_trcov(gp,tx);
 
        sqrttautilde = sqrt(tautilde);
        Stildesqroot = diag(sqrttautilde);
        
        z=Stildesqroot*(L'\(L\(Stildesqroot*(C*nutilde))));
        
        kstarstar = gp_trvar(gp, x, predcf);

        ntest=size(x,1);
        
        K_nf=gp_cov(gp,x,tx,predcf);
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
        [e, edata, eprior, tautilde, nutilde, L, La, b] = gpep_e(gp_pak(gp, param), gp, tx, ty, param);

        % Here tstind = 1 if the prediction is made for the training set 
        if nargin > 6
            %tstind = varargin{2};
            if length(tstind) ~= size(tx,1)
                error('tstind (if provided) has to be of same lenght as tx.')
            end
        else
             tstind = [];
        end
        
        u = gp.X_u;
        m = size(u,1);
        
        K_fu = gp_cov(gp,tx,u,predcf);         % f x u
        K_nu=gp_cov(gp,x,u,predcf);
        K_uu = gp_trcov(gp,u,predcf);          % u x u, noiseles covariance K_uu
        K_uu = (K_uu+K_uu')./2;          % ensure the symmetry of K_uu

        kstarstar=gp_trvar(gp,x,predcf);        

        % From this on evaluate the prediction
        % See Snelson and Ghahramani (2007) for details 
        %        p=iLaKfu*(A\(iLaKfu'*myytilde));
        p = b';
        
        ntest=size(x,1);
        
        Ef = K_nu*(K_uu\(K_fu'*p));
        
        % if the prediction is made for training set, evaluate Lav also for prediction points
        if ~isempty(tstind)
            [Kv_ff, Cv_ff] = gp_trvar(gp, x(tstind,:), predcf);
            Luu = chol(K_uu)';
            B=Luu\(K_fu');
            Qv_ff=sum(B.^2)';
            Lav = Kv_ff-Qv_ff;
            Ef(tstind) = Ef(tstind) + Lav.*p;
        end
        
        if nargout > 1
            % Compute variances of predictions
            %Varf(i1,1)=kstarstar(i1) - (sum(Knf(i1,:).^2./La') - sum((Knf(i1,:)*L).^2));
            Luu = chol(K_uu)';
            B=Luu\(K_fu');   
            B2=Luu\(K_nu');   
            Varf = kstarstar - sum(B2'.*(B*(repmat(La,1,m).\B')*B2)',2)  + sum((K_nu*(K_uu\(K_fu'*L))).^2, 2);

            % if the prediction is made for training set, evaluate Lav also for prediction points
            if ~isempty(tstind)
                Varf(tstind) = Varf(tstind) - 2.*sum( B2(:,tstind)'.*(repmat((La.\Lav),1,m).*B'),2) ...
                    + 2.*sum( B2(:,tstind)'*(B*L).*(repmat(Lav,1,m).*L), 2)  ...
                    - Lav./La.*Lav + sum((repmat(Lav,1,m).*L).^2,2);
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
        
      case {'PIC' 'PIC_BLOCK'}
        % Calculate some help matrices  
        u = gp.X_u;
        ind = gp.tr_index;
        [e, edata, eprior, tautilde, nutilde, L, La, b] = gpep_e(gp_pak(gp, param), gp, tx, ty, param);
        
        K_fu = gp_cov(gp, tx, u, predcf);         % f x u
        K_nu = gp_cov(gp, x, u, predcf);         % n x u   
        K_uu = gp_trcov(gp, u, predcf);    % u x u, noiseles covariance K_uu

        % From this on evaluate the prediction
        % See Snelson and Ghahramani (2007) for details 
        %        p=iLaKfu*(A\(iLaKfu'*myytilde));
        p = b';

        iKuuKuf = K_uu\K_fu';
        
        w_bu=zeros(length(x),length(u));
        w_n=zeros(length(x),1);
        for i=1:length(ind)
            w_bu(tstind{i},:) = repmat((iKuuKuf(:,ind{i})*p(ind{i},:))', length(tstind{i}),1);
            K_nf = gp_cov(gp, x(tstind{i},:), tx(ind{i},:), predcf);              % n x u
            w_n(tstind{i},:) = K_nf*p(ind{i},:);
        end
        
        Ef = K_nu*(iKuuKuf*p) - sum(K_nu.*w_bu,2) + w_n;

        if nargout > 1
            kstarstar = gp_trvar(gp, x, predcf);
            KnfL = K_nu*(iKuuKuf*L);
            Varf = zeros(length(x),1);
            for i=1:length(ind)
                v_n = gp_cov(gp, x(tstind{i},:), tx(ind{i},:), predcf);              % n x u
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
        % Here tstind = 1 if the prediction is made for the training set 
        if nargin > 6 
            if length(tstind) ~= size(tx,1)
                error('tstind (if provided) has to be of same lenght as tx.')
            end
        else
            tstind = [];
        end
        
        u = gp.X_u;
        m = length(u);
        n = size(tx,1);
        n2 = size(x,1);
                
        [e, edata, eprior, tautilde, nutilde, L, La, b] = gpep_e(gp_pak(gp, param), gp, tx, ty, param);

        % Indexes to all non-compact support and compact support covariances.
        cf1 = [];
        cf2 = [];
        % Indexes to non-CS and CS covariances, which are used for predictions
        predcf1 = [];
        predcf2 = [];    
        
        ncf = length(gp.cf);
        % Loop through all covariance functions
        for i = 1:ncf        
            % Non-CS covariances
            if ~isfield(gp.cf{i},'cs') 
                cf1 = [cf1 i];
                % If used for prediction
                if ~isempty(find(predcf==i))
                    predcf1 = [predcf1 i]; 
                end
                % CS-covariances
        else
            cf2 = [cf2 i];           
            % If used for prediction
            if ~isempty(find(predcf==i))
                predcf2 = [predcf2 i]; 
            end
            end
        end
        if isempty(predcf1) && isempty(predcf2)
            predcf1 = cf1;
            predcf2 = cf2;
        end
        
        % Determine the types of the covariance functions used
        % in making the prediction.
        if ~isempty(predcf1) && isempty(predcf2)       % Only non-CS covariances
            ptype = 1;
            predcf2 = cf2;
        elseif isempty(predcf1) && ~isempty(predcf2)   % Only CS covariances
            ptype = 2;
            predcf1 = cf1;
        else                                           % Both non-CS and CS covariances
            ptype = 3;
        end
           
        K_fu = gp_cov(gp,tx,u,predcf1);   % f x u
        K_uu = gp_trcov(gp,u,predcf1);     % u x u, noiseles covariance K_uu
        K_uu = (K_uu+K_uu')./2;     % ensure the symmetry of K_uu
        K_nu=gp_cov(gp,x,u,predcf1);
        
        Kcs_nf = gp_cov(gp, x, tx, predcf2);
        
        p = b';
        ntest=size(x,1);
                
        % Calculate the predictive mean according to the type of
        % covariance functions used for making the prediction
        if ptype == 1
            Ef = K_nu*(K_uu\(K_fu'*p));
        elseif ptype == 2
            Ef = Kcs_nf*p;
        else 
            Ef = K_nu*(K_uu\(K_fu'*p)) + Kcs_nf*p;        
        end

        % evaluate also Lav if the prediction is made for training set
        if ~isempty(tstind)
            [Kv_ff, Cv_ff] = gp_trvar(gp, x(tstind,:), predcf1);
            Luu = chol(K_uu)';
            B=Luu\(K_fu');
            Qv_ff=sum(B.^2)';
            Lav = Kv_ff-Qv_ff;
        end
        
        % Add also Lav if the prediction is made for training set
        % and non-CS covariance function is used for prediction
        if ~isempty(tstind) && (ptype == 1 || ptype == 3)
            Ef(tstind) = Ef(tstind) + Lav.*p;
        end

        % Evaluate the variance
        if nargout > 1
            Knn_v = gp_trvar(gp,x,predcf);
            Luu = chol(K_uu)';
            B=Luu\(K_fu');   
            B2=Luu\(K_nu');
            p = amd(La);
            iLaKfu = La\K_fu;
            % Calculate the predictive variance according to the type
            % covariance functions used for making the prediction
            if ptype == 1 || ptype == 3                            
                % FIC part of the covariance
                Varf = Knn_v - sum(B2'.*(B*(La\B')*B2)',2) + sum((K_nu*(K_uu\(K_fu'*L))).^2, 2);
                % Add Lav2 if the prediction is made for the training set
                if  ~isempty(tstind)
                    % Non-CS covariance
                    if ptype == 1
                        Kcs_nf = sparse(tstind,1:n,Lav,n2,n);
                        % Non-CS and CS covariances
                    else
                        Kcs_nf = Kcs_nf + sparse(tstind,1:n,Lav,n2,n);
                    end
                    % Add Lav2 inside Kcs_nf
                    Varf = Varf - sum((Kcs_nf(:,p)/chol(La(p,p))).^2,2) + sum((Kcs_nf*L).^2, 2) ...
                           - 2.*sum((Kcs_nf*iLaKfu).*(K_uu\K_nu')',2) + 2.*sum((Kcs_nf*L).*(L'*K_fu*(K_uu\K_nu'))' ,2);                
                    % In case of both non-CS and CS prediction covariances add 
                    % only Kcs_nf if the prediction is not done for the training set 
                elseif ptype == 3
                    Varf = Varf - sum((Kcs_nf(:,p)/chol(La(p,p))).^2,2) + sum((Kcs_nf*L).^2, 2) ...
                           - 2.*sum((Kcs_nf*iLaKfu).*(K_uu\K_nu')',2) + 2.*sum((Kcs_nf*L).*(L'*K_fu*(K_uu\K_nu'))' ,2);
                end
            % Prediction with only CS covariance
            elseif ptype == 2
                Varf = Knn_v - sum((Kcs_nf(:,p)/chol(La(p,p))).^2,2) + sum((Kcs_nf*L).^2, 2) ;
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
      case 'SSGP'
        [e, edata, eprior, tautilde, nutilde, L, S, b] = gpep_e(gp_pak(gp, param), gp, tx, ty, param);
        %param = varargin{1};

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

