function [Ef, Varf, Ey, Vary, Pyt] = la_pred(gp, x, y, xt, varargin)
%LA_PRED	Predictions with Gaussian Process Laplace approximation
%
%     Description
%	[EF, VARF, EY, VARY] = LA_PRED(GP, X, Y, XT, OPTIONS) takes a GP 
%        data structure GP together with a matrix X of input vectors,
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
%                set TSTIND = []).
%       'yt'     is optional observed yt in test points (see below)
%       'z'      is optional observed quantity in triplet (x_i,y_i,z_i)
%                Some likelihoods may use this. For example, in case of 
%                Poisson likelihood we have z_i=E_i, that is, expected value 
%                for ith case. 
%       'zt'     is optional observed quantity in triplet (xt_i,yt_i,zt_i)
%                Some likelihoods may use this. For example, in case of 
%                Poisson likelihood we have z_i=E_i, that is, expected value 
%                for ith case. 
%
%	[EF, VARF, EY, VARY, PYT] = LA_PRED(GP, X, Y, XT, 'yt', YT) 
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
%       predictions Ef1 = la_pred(GP, X, Y, X, 'predcf', 1) and 
%       Ef2 = la_pred(gp, x, y, x, 'predcf', 2) should sum up to 
%       Ef = la_pred(gp, x, y, x). That is Ef = Ef1 + Ef2. With 
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
%	GPLA_E, GPLA_G, GP_PRED, DEMO_SPATIAL, DEMO_CLASSIFIC
%
% Copyright (c) 2007-2010 Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

  ip=inputParser;
  ip.FunctionName = 'LA_PRED';
  ip.addRequired('gp', @isstruct);
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
      case 'FULL'
        [e, edata, eprior, f, L, a, W, p] = gpla_e(gp_pak(gp), gp, x, y, 'z', z);

        ntest=size(xt,1);
        K_nf = gp_cov(gp,xt,x,predcf);

        % Evaluate the variance
        if nargout > 1
            kstarstar = gp_trvar(gp,xt,predcf);
            if W >= 0             % This is the usual case where likelihood is log concave
                                  % for example, Poisson and probit
                if issparse(K_nf) && issparse(L)          % If compact support covariance functions are used 
                                                          % the covariance matrix will be sparse
                    deriv = feval(gp.likelih.fh_g, gp.likelih, y(p), f, 'latent', z(p));
                    Ef = K_nf(:,p)*deriv;
                    sqrtW = sqrt(W);
                    sqrtWKfn = sqrtW*K_nf(:,p)';
                    V = ldlsolve(L,sqrtWKfn);
                    Varf = kstarstar - sum(sqrtWKfn.*V,1)';
                else
                    deriv = feval(gp.likelih.fh_g, gp.likelih, y, f, 'latent', z);
                    Ef = K_nf*deriv;
                    W = diag(W);
                    V = L\(sqrt(W)*K_nf');
                    Varf = kstarstar - sum(V'.*V',2);
                end
            else                  % We may end up here if the likelihood is not log concace
                                  % For example Student-t likelihood. 
                deriv = feval(gp.likelih.fh_g, gp.likelih, y, f, 'latent', z);
                Ef = K_nf*deriv;
                V = L*diag(W);
                R = diag(W) - V'*V;
                Varf = kstarstar - sum(K_nf.*(R*K_nf')',2);
            end
        end
        % ============================================================
        % FIC
        % ============================================================    
      case 'FIC'        % Predictions with FIC sparse approximation for GP
        % Here tstind = 1 if the prediction is made for the training set 
        if nargin > 6
            if ~isempty(tstind) && length(tstind) ~= size(x,1)
                error('tstind (if provided) has to be of same lenght as x.')
            end
        else
             tstind = [];
        end

        u = gp.X_u;
        K_fu = gp_cov(gp, x, u, predcf);         % f x u
        K_uu = gp_trcov(gp, u, predcf);          % u x u, noiseles covariance K_uu
        K_uu = (K_uu+K_uu')./2;                  % ensure the symmetry of K_uu
        Luu = chol(K_uu)';

        m = size(u,1);

        [e, edata, eprior, f, L, a, La2] = gpla_e(gp_pak(gp), gp, x, y, 'z', z);

        deriv = feval(gp.likelih.fh_g, gp.likelih, y, f, 'latent', z);
        ntest=size(xt,1);

        K_nu=gp_cov(gp,xt,u,predcf);
        Ef = K_nu*(Luu'\(Luu\(K_fu'*deriv)));

        % if the prediction is made for training set, evaluate Lav also for prediction points
        if ~isempty(tstind)
            [Kv_ff, Cv_ff] = gp_trvar(gp, xt(tstind,:), predcf);
            B=Luu\(K_fu');
            Qv_ff=sum(B.^2)';
            %Lav = zeros(size(La));
            %Lav(tstind) = Kv_ff-Qv_ff;
            Lav = Kv_ff-Qv_ff;            
            Ef(tstind) = Ef(tstind) + Lav.*deriv;
        end

        
        % Evaluate the variance
        if nargout > 1
            % re-evaluate matrices with training components
            Kfu_tr = gp_cov(gp, x, u);
            Kuu_tr = gp_trcov(gp, u);
            Kuu_tr = (K_uu+K_uu')./2;
            
            W = -feval(gp.likelih.fh_g2, gp.likelih, y, f, 'latent', z);
            kstarstar = gp_trvar(gp,xt,predcf);
            La = W.*La2;
            Lahat = 1 + La;
            B = (repmat(sqrt(W),1,m).*Kfu_tr);

            % Components for (I + W^(1/2)*(Qff + La2)*W^(1/2))^(-1) = Lahat^(-1) - L2*L2'
            B2 = repmat(Lahat,1,m).\B;
            A2 = Kuu_tr + B'*B2; A2=(A2+A2)/2;
            L2 = B2/chol(A2);

            % Set params for K_nf
            BB=Luu\(B');
            BB2=Luu\(K_nu');
            Varf = kstarstar - sum(BB2'.*(BB*(repmat(Lahat,1,m).\BB')*BB2)',2)  + sum((K_nu*(K_uu\(B'*L2))).^2, 2);
            
            % if the prediction is made for training set, evaluate Lav also for prediction points
            if ~isempty(tstind)
                LavsW = Lav.*sqrt(W);
                    Varf(tstind) = Varf(tstind) - (LavsW./sqrt(Lahat)).^2 + sum((repmat(LavsW,1,m).*L2).^2, 2) ...
                           - 2.*sum((repmat(LavsW,1,m).*(repmat(Lahat,1,m).\B)).*(K_uu\K_nu(tstind,:)')',2)...
                           + 2.*sum((repmat(LavsW,1,m).*L2).*(L2'*B*(K_uu\K_nu(tstind,:)'))' ,2);
            end
        end
        % ============================================================
        % PIC
        % ============================================================
      case {'PIC' 'PIC_BLOCK'}        % Predictions with PIC sparse approximation for GP
        u = gp.X_u;
        K_fu = gp_cov(gp, x, u, predcf);         % f x u
        K_uu = gp_trcov(gp, u, predcf);          % u x u, noiseles covariance K_uu
        K_uu = (K_uu+K_uu')./2;                  % ensure the symmetry of K_uu
        K_nu=gp_cov(gp,xt,u,predcf);

        ind = gp.tr_index;
        ntest = size(xt,1);
        m = size(u,1);

        [e, edata, eprior, f, L, a, La2] = gpla_e(gp_pak(gp), gp, x, y, 'z', z);

        deriv = feval(gp.likelih.fh_g, gp.likelih, y, f, 'latent', z);

        iKuuKuf = K_uu\K_fu';
        w_bu=zeros(length(xt),length(u));
        w_n=zeros(length(xt),1);
        for i=1:length(ind)
            w_bu(tstind{i},:) = repmat((iKuuKuf(:,ind{i})*deriv(ind{i},:))', length(tstind{i}),1);
            K_nf = gp_cov(gp, xt(tstind{i},:), x(ind{i},:), predcf);              % n x u
            w_n(tstind{i},:) = K_nf*deriv(ind{i},:);
        end

        Ef = K_nu*(iKuuKuf*deriv) - sum(K_nu.*w_bu,2) + w_n;

        % Evaluate the variance
        if nargout > 1
            W = -feval(gp.likelih.fh_g2, gp.likelih, y, f, 'latent', z);
            kstarstar = gp_trvar(gp,xt,predcf);
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
            Varf = zeros(length(xt),1);
            for i=1:length(ind)
                v_n = gp_cov(gp, xt(tstind{i},:), x(ind{i},:),predcf).*repmat(sqrtW(ind{i},:)',length(tstind{i}),1);              % n x u
                v_bu = K_nu(tstind{i},:)*iKuuB(:,ind{i});
                KnfLa = K_nu*(iKuuB(:,ind{i})/chol(Lahat{i}));
                KnfLa(tstind{i},:) = KnfLa(tstind{i},:) - (v_bu + v_n)/chol(Lahat{i});
                Varf = Varf + sum((KnfLa).^2,2);
                KnfL2(tstind{i},:) = KnfL2(tstind{i},:) - v_bu*L2(ind{i},:) + v_n*L2(ind{i},:);
            end
            Varf = kstarstar - (Varf - sum((KnfL2).^2,2));
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

        n = size(x,1);
        n2 = size(xt,1);
        u = gp.X_u;
        m = length(u);

        [e, edata, eprior, f, L, a, La2] = gpla_e(gp_pak(gp), gp, x, y, 'z', z);
        
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
        
        K_fu = gp_cov(gp,x,u,predcf1);         % f x u
        K_uu = gp_trcov(gp,u,predcf1);    % u x u, noiseles covariance K_uu
        K_uu = (K_uu+K_uu')./2;     % ensure the symmetry of K_uu
        K_nu=gp_cov(gp,xt,u,predcf1);

        Kcs_nf = gp_cov(gp, xt, x, predcf2);

        deriv = feval(gp.likelih.fh_g, gp.likelih, y, f, 'latent', z);
        ntest=size(xt,1);

        % Calculate the predictive mean according to the type of
        % covariance functions used for making the prediction
        if ptype == 1
            Ef = K_nu*(K_uu\(K_fu'*deriv));
        elseif ptype == 2
            Ef = Kcs_nf*deriv;
        else 
            Ef = K_nu*(K_uu\(K_fu'*deriv)) + Kcs_nf*deriv;        
        end

        % evaluate also Lav if the prediction is made for training set
        if ~isempty(tstind)
            [Kv_ff, Cv_ff] = gp_trvar(gp, xt(tstind,:), predcf1);
            Luu = chol(K_uu)';
            B=Luu\(K_fu');
            Qv_ff=sum(B.^2)';
            %Lav = zeros(size(Ef));
            %Lav(tstind) = Kv_ff-Qv_ff;
            Lav = Kv_ff-Qv_ff;
        end

        % Add also Lav if the prediction is made for training set
        % and non-CS covariance function is used for prediction
        if ~isempty(tstind) && (ptype == 1 || ptype == 3)
            Ef(tstind) = Ef(tstind) + Lav.*deriv;
        end

        
        % Evaluate the variance
        if nargout > 1
            W = -feval(gp.likelih.fh_g2, gp.likelih, y, f, 'latent', z);
            sqrtW = sparse(1:tn,1:tn,sqrt(W),tn,tn);
            kstarstar = gp_trvar(gp,xt,predcf);
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
            
            m = amd(Lahat);
            % Calculate the predictive variance according to the type
            % covariance functions used for making the prediction
            if ptype == 1 || ptype == 3                            
                % FIC part of the covariance
                Varf = kstarstar - sum(BB2'.*(BB*(Lahat\BB')*BB2)',2) + sum((K_nu*(K_uu\(B'*L2))).^2, 2);
                % Add Lav to Kcs_nf if the prediction is made for the training set
                if  ~isempty(tstind)
                    % Non-CS covariance
                    if ptype == 1         
                        Kcs_nf = sparse(tstind,1:n,Lav,n2,n);                    
                    % Non-CS and CS covariances
                    else                  
                        Kcs_nf = Kcs_nf + sparse(tstind,1:n,Lav,n2,n);
                    end
                    KcssW = Kcs_nf*sqrtW;                    
                    Varf = Varf - sum((KcssW(:,m)/chol(Lahat(m,m))).^2,2) + sum((KcssW*L2).^2, 2) ...
                           - 2.*sum((KcssW*(Lahat\B)).*(K_uu\K_nu')',2) + 2.*sum((KcssW*L2).*(L2'*B*(K_uu\K_nu'))' ,2);
                % In case of both non-CS and CS prediction covariances add 
                % only Kcs_nf if the prediction is not done for the training set 
                elseif ptype == 3
                    KcssW = Kcs_nf*sqrtW;
                    Varf = Varf - sum((KcssW(:,m)/chol(Lahat(m,m))).^2,2) + sum((KcssW*L2).^2, 2) ...
                           - 2.*sum((KcssW*(Lahat\B)).*(K_uu\K_nu')',2) + 2.*sum((KcssW*L2).*(L2'*B*(K_uu\K_nu'))' ,2);
                end
            % Prediction with only CS covariance
            elseif ptype == 2
                KcssW = Kcs_nf*sqrtW;
                Varf = kstarstar - sum((KcssW(:,m)/chol(Lahat(m,m))).^2,2) + sum((KcssW*L2).^2, 2);
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
            [Ey, Vary, Pyt] = feval(gp.likelih.fh_predy, gp.likelih, Ef, Varf, yt, zt);
        end
    end
end