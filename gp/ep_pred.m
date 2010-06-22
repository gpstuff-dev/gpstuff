function [Ef, Varf, Ey, Vary, Py] = ep_pred(gp, x, y, xt, varargin)
%EP_PRED       Predictions with Gaussian Process EP approximation
%
%     Description
%	[EF, VARF, EY, VARY] = EP_PRED(GP, X, Y, XT, OPTIONS) takes a GP 
%        data structure GP together with a matrix XT of input vectors,
%        matrix X of training inputs and vector Y of training
%        targets, and evaluates the predictive distribution at
%        inputs X. Returns a posterior mean EF and variance VARF of
%        latent variables and the posterior predictive mean EY and
%        variance VARY of obervations at input locations X. 
%
%     OPTIONS is optional parameter-value pair
%       'predcf' is index vector telling which covariance functions are 
%                used for prediction. Default is all (1:gpcfn). See 
%                additional information below.
%       'tstind' is a vector/cell array defining, which rows of X belong 
%                to which training block in *IC type sparse models. Deafult 
%                is []. In case of PIC, a cell array containing index 
%                vectors specifying the blocking structure for test data.
%                IN FIC and CS+FIC a vector of length n that points out the 
%                test inputs that are also in the training set (if none,
%                set TSTIND = [])
%       'yt'     is optional observed yt in test points (see below)
%       'z'      is optional observed quantity in triplet (x_i,y_i,z_i)
%                Some likelihoods may use this. For example, in case of 
%                Poisson likelihood we have z_i=E_i, that is, expected value 
%                for ith case. 
%       'zt'     is optional observed quantity in triplet (xt_i,yt_i,zt_i)
%                Some likelihoods may use this. For example, in case of 
%                Poisson likelihood we have z_i=E_i, that is, the expected 
%                value for the ith case. 
%
%	[EF, VARF, EY, VARY, PYT] = EP_PRED(GP, X, Y, XT, 'yt', YT, OPTIONS) 
%        returns also the predictive density PYT of the test observations 
%        YT at input locations XT. This can be used for example in the
%        cross-validation.
%
%       NOTE! In case of FIC and PIC sparse approximation the
%       prediction for only some PREDCF covariance functions is
%       just an approximation since the covariance functions are
%       coupled in the approximation and are not strictly speaking
%       additive anymore.
%
%       For example, if you use covariance such as K = K1 + K2 your
%       predictions Ef1 = ep_pred(GP, X, Y, X, 'predcf', 1) and 
%       Ef2 = ep_pred(gp, x, y, x, 'predcf', 2) should sum up to 
%       Ef = ep_pred(gp, x, y, x). That is Ef = Ef1 + Ef2. With 
%       FULL model this is true but with FIC and PIC this is true only 
%       approximately. That is Ef \approx Ef1 + Ef2.
%
%       With CS+FIC the predictions are exact if the PREDCF
%       covariance functions are all in the FIC part or if they are
%       CS covariances.
%
%       NOTE! When making predictions with a subset of covariance
%       functions with FIC approximation the predictive variance
%       can in some cases be ill-behaved i.e. negative or
%       unrealistically small. This may happen because of the
%       approximative nature of the prediction.
%  
%	See also
%	GPEP_E, GPEP_G, GP_PRED, DEMO_SPATIAL, DEMO_CLASSIFIC
%
% Copyright (c) 2007-2010 Jarno Vanhatalo
% Copyright (c) 2010      Heikki Peura

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

    ip=inputParser;
    ip.FunctionName = 'EP_PRED';
    ip.addRequired('gp',@isstruct);
    ip.addRequired('x', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
    ip.addRequired('y', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
    ip.addRequired('xt', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
    ip.addParamValue('yt', [], @(x) isreal(x) && all(isfinite(x(:))))
    ip.addParamValue('z', [], @(x) isreal(x) && all(isfinite(x(:))))
    ip.addParamValue('zt', [], @(x) isreal(x) && all(isfinite(x(:))))
    ip.addParamValue('predcf', [], @(x) isempty(x) || ...
                     isvector(x) && isreal(x) && all(isfinite(x)&x>0))
    ip.addParamValue('tstind', [], @(x) isempty(x) || iscell(x) ||...
                 (isvector(x) && isreal(x) && all(isfinite(x)&x>0)))
    ip.parse(gp, x, y, xt, varargin{:});
    yt=ip.Results.yt;
    z=ip.Results.z;
    zt=ip.Results.zt;
    predcf=ip.Results.predcf;
    tstind=ip.Results.tstind;

    
    [tn, tnin] = size(x);
        
    switch gp.type
        % ============================================================
        % FULL
        % ============================================================
      case 'FULL'        % Predictions with FULL GP model
        [e, edata, eprior, tautilde, nutilde, L] = gpep_e(gp_pak(gp), gp, x, y, 'z', z);
        
        [K, C]=gp_trcov(gp,x);
        kstarstar = gp_trvar(gp, xt, predcf);
        ntest=size(xt,1);
        K_nf=gp_cov(gp,xt,x,predcf);
        [n,nin] = size(x);
 
        if tautilde > 0             % This is the usual case where likelihood is log concave
                                    % for example, Poisson and probit
            sqrttautilde = sqrt(tautilde);
            Stildesqroot = sparse(1:n, 1:n, sqrttautilde, n, n);
                            
            if issparse(L)          % If compact support covariance functions are used 
                                    % the covariance matrix will be sparse
                z=Stildesqroot*ldlsolve(L,Stildesqroot*(C*nutilde));
            else
                z=Stildesqroot*(L'\(L\(Stildesqroot*(C*nutilde))));
            end
            Ef=K_nf*(nutilde-z);    % The mean

            % Compute variance
            if nargout > 1
                if issparse(L)
                    V = ldlsolve(L, Stildesqroot*K_nf');
                    Varf = kstarstar - sum(K_nf.*(Stildesqroot*V)',2);
                else
                    V = (L\Stildesqroot)*K_nf';
                    Varf = kstarstar - sum(V.^2)';
                end
            end
        else                         % We might end up here if the likelihood is not log concace
                                     % For example Student-t likelihood. 
                                     % NOTE! This does not work reliably yet
            z=tautilde.*(L'*(L*nutilde));
            Ef=K_nf*(nutilde-z);
            
            if nargout > 1
                S = diag(tautilde);
                V = K_nf*S*L';
                Varf = kstarstar - sum((K_nf*S).*K_nf,2) + sum(V.^2,2);
            end
        end
        % ============================================================
        % FIC
        % ============================================================        
      case 'FIC'        % Predictions with FIC sparse approximation for GP
        [e, edata, eprior, tautilde, nutilde, L, La, b] = gpep_e(gp_pak(gp), gp, x, y, 'z', z);

        % Here tstind = 1 if the prediction is made for the training set 
        if nargin > 6
            if ~isempty(tstind) && length(tstind) ~= size(x,1)
                error('tstind (if provided) has to be of same lenght as x.')
            end
        else
             tstind = [];
        end
        
        u = gp.X_u;
        m = size(u,1);
        
        K_fu = gp_cov(gp,x,u,predcf);          % f x u
        K_nu=gp_cov(gp,xt,u,predcf);
        K_uu = gp_trcov(gp,u,predcf);          % u x u, noiseles covariance K_uu
        K_uu = (K_uu+K_uu')./2;                % ensure the symmetry of K_uu

        kstarstar=gp_trvar(gp,xt,predcf);        

        % From this on evaluate the prediction
        % See Snelson and Ghahramani (2007) for details 
        %        p=iLaKfu*(A\(iLaKfu'*myytilde));
        p = b';
        
        ntest=size(xt,1);
        
        Ef = K_nu*(K_uu\(K_fu'*p));
        
        % if the prediction is made for training set, evaluate Lav also for prediction points
        if ~isempty(tstind)
            [Kv_ff, Cv_ff] = gp_trvar(gp, xt(tstind,:), predcf);
            Luu = chol(K_uu)';
            B=Luu\(K_fu');
            Qv_ff=sum(B.^2)';
            Lav = Kv_ff-Qv_ff;
            Ef(tstind) = Ef(tstind) + Lav.*p;
        end
        
        % Compute variance
        if nargout > 1
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
        end
        % ============================================================
        % PIC
        % ============================================================
      case {'PIC' 'PIC_BLOCK'}        % Predictions with PIC sparse approximation for GP
        % Calculate some help matrices  
        u = gp.X_u;
        ind = gp.tr_index;
        [e, edata, eprior, tautilde, nutilde, L, La, b] = gpep_e(gp_pak(gp), gp, x, y, 'z', z);
        
        K_fu = gp_cov(gp, x, u, predcf);         % f x u
        K_nu = gp_cov(gp, xt, u, predcf);         % n x u   
        K_uu = gp_trcov(gp, u, predcf);    % u x u, noiseles covariance K_uu

        % From this on evaluate the prediction
        % See Snelson and Ghahramani (2007) for details 
        %        p=iLaKfu*(A\(iLaKfu'*myytilde));
        p = b';

        iKuuKuf = K_uu\K_fu';
        
        w_bu=zeros(length(xt),length(u));
        w_n=zeros(length(xt),1);
        for i=1:length(ind)
            w_bu(tstind{i},:) = repmat((iKuuKuf(:,ind{i})*p(ind{i},:))', length(tstind{i}),1);
            K_nf = gp_cov(gp, xt(tstind{i},:), x(ind{i},:), predcf);              % n x u
            w_n(tstind{i},:) = K_nf*p(ind{i},:);
        end
        
        Ef = K_nu*(iKuuKuf*p) - sum(K_nu.*w_bu,2) + w_n;

        % Compute variance
        if nargout > 1
            kstarstar = gp_trvar(gp, xt, predcf);
            KnfL = K_nu*(iKuuKuf*L);
            Varf = zeros(length(xt),1);
            for i=1:length(ind)
                v_n = gp_cov(gp, xt(tstind{i},:), x(ind{i},:), predcf);              % n x u
                v_bu = K_nu(tstind{i},:)*iKuuKuf(:,ind{i});
                KnfLa = K_nu*(iKuuKuf(:,ind{i})/chol(La{i}));
                KnfLa(tstind{i},:) = KnfLa(tstind{i},:) - (v_bu + v_n)/chol(La{i});
                Varf = Varf + sum((KnfLa).^2,2);
                KnfL(tstind{i},:) = KnfL(tstind{i},:) - v_bu*L(ind{i},:) + v_n*L(ind{i},:);
            end
            Varf = kstarstar - (Varf - sum((KnfL).^2,2));  

        end
        % ============================================================
        % CS+FIC
        % ============================================================
      case 'CS+FIC'        % Predictions with CS+FIC sparse approximation for GP
        % Here tstind = 1 if the prediction is made for the training set 
        if nargin > 6 
            if ~isempty(tstind) && length(tstind) ~= size(x,1)
                error('tstind (if provided) has to be of same lenght as x.')
            end
        else
            tstind = [];
        end
        
        u = gp.X_u;
        m = length(u);
        n = size(x,1);
        n2 = size(xt,1);
                
        [e, edata, eprior, tautilde, nutilde, L, La, b] = gpep_e(gp_pak(gp), gp, x, y, 'z', z);

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
           
        K_fu = gp_cov(gp,x,u,predcf1);   % f x u
        K_uu = gp_trcov(gp,u,predcf1);     % u x u, noiseles covariance K_uu
        K_uu = (K_uu+K_uu')./2;     % ensure the symmetry of K_uu
        K_nu=gp_cov(gp,xt,u,predcf1);
        
        Kcs_nf = gp_cov(gp, xt, x, predcf2);
        
        p = b';
        ntest=size(xt,1);
                
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
            [Kv_ff, Cv_ff] = gp_trvar(gp, xt(tstind,:), predcf1);
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
            Knn_v = gp_trvar(gp,xt,predcf);
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
        end
        % ============================================================
        % DTC/(VAR)
        % ============================================================
      case {'DTC' 'VAR' 'SOR'}        % Predictions with DTC or variational sparse approximation for GP
        [e, edata, eprior, tautilde, nutilde, L, La, b] = gpep_e(gp_pak(gp), gp, x, y, 'z', z);

        % Here tstind = 1 if the prediction is made for the training set 
        if nargin > 6
            if ~isempty(tstind) && length(tstind) ~= size(x,1)
                error('tstind (if provided) has to be of same lenght as x.')
            end
        else
             tstind = [];
        end
        
        u = gp.X_u;
        m = size(u,1);
        
        K_fu = gp_cov(gp,x,u,predcf);         % f x u
        K_nu=gp_cov(gp,xt,u,predcf);
        K_uu = gp_trcov(gp,u,predcf);          % u x u, noiseles covariance K_uu
        K_uu = (K_uu+K_uu')./2;          % ensure the symmetry of K_uu

        kstarstar=gp_trvar(gp,xt,predcf);        

        % From this on evaluate the prediction
        p = b';
        
        ntest=size(xt,1);
        
        Ef = K_nu*(K_uu\(K_fu'*p));
        
        % if the prediction is made for training set, evaluate Lav also for prediction points
        if ~isempty(tstind)
            [Kv_ff, Cv_ff] = gp_trvar(gp, xt(tstind,:), predcf);
            Luu = chol(K_uu)';
            B=Luu\(K_fu');
            Qv_ff=sum(B.^2)';
            Lav = Kv_ff-Cv_ff;
            Ef(tstind) = Ef(tstind);% + Lav.*p;
        end
        
        if nargout > 1
            % Compute variances of predictions
            %Varf(i1,1)=kstarstar(i1) - (sum(Knf(i1,:).^2./La') - sum((Knf(i1,:)*L).^2));
            Luu = chol(K_uu)';
            B=Luu\(K_fu');   
            B2=Luu\(K_nu');   

            Varf = sum(B2'.*(B*(repmat(La,1,m).\B')*B2)',2)  + sum((K_nu*(K_uu\(K_fu'*L))).^2, 2);
            switch gp.type
              case {'VAR' 'DTC'}
                Varf = kstarstar - Varf;
              case 'SOR'
                Varf = sum(B2.^2,1)' - Varf;
            end
        end
        % ============================================================
        % SSGP
        % ============================================================
      case 'SSGP'        % Predictions with sparse spectral sampling approximation for GP
                         % The approximation is proposed by M. Lazaro-Gredilla, J. Quinonero-Candela and A. Figueiras-Vidal
                         % in Microsoft Research technical report MSR-TR-2007-152 (November 2007)
                         % NOTE! This does not work at the moment.
        [e, edata, eprior, tautilde, nutilde, L, S, b] = gpep_e(gp_pak(gp), gp, x, y, 'z', z);
        %param = varargin{1};

        Phi_f = gp_trcov(gp, x);
        Phi_a = gp_trcov(gp, xt);

        m = size(Phi_f,2);
        ntest=size(xt,1);
        
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
    
    
    % ============================================================
    % Evaluate also the predictive mean and variance of new observation(s)
    % ============================================================    
    if nargout > 2
        if isempty(yt)
            [Ey, Vary] = feval(gp.likelih.fh_predy, gp.likelih, Ef, Varf, [], zt);
        else
            [Ey, Vary, Py] = feval(gp.likelih.fh_predy, gp.likelih, Ef, Varf, yt, zt);
        end
    end
end

