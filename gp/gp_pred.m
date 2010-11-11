function [Eft, Varft, Eyt, Varyt, pyt] = gp_pred(gp, x, y, xt, varargin)
%GP_PRED  Make predictions with Gaussian process 
%
%  Description
%    [EFT, VARFT, EYT, VARYT] = GP_PRED(GP, X, Y, XT, OPTIONS)
%    takes a GP data structure together with matrix X of training
%    inputs and vector Y of training targets, and evaluates the
%    predictive distribution at test inputs XT. Returns a posterior
%    mean EFT and variance VARFT of latent variables and the
%    posterior predictive mean EYT and variance VARYT.
%
%        Eft =  E[f | xt,x,y,th]  = K_fy*(Kyy+s^2I)^(-1)*y
%      Varft = Var[f | xt,x,y,th] = diag(K_fy - K_fy*(Kyy+s^2I)^(-1)*K_yf). 
%
%    Each row of X corresponds to one input vector and each row of
%    Y corresponds to one output vector.
%
%    [EFT, VARFT, EYT, VARYT, PYT] = GP_PRED(GP, X, Y, XT, 'yt', YT, ...)
%    returns also the predictive density PYT of the observations YT
%    at test input locations XT. This can be used for example in
%    the cross-validation. Here Y has to be vector.
%
%    OPTIONS is optional parameter-value pair
%      predcf - an index vector telling which covariance functions are 
%                 used for prediction. Default is all (1:gpcfn). 
%                 See additional information below.
%      tstind - a vector/cell array defining, which rows of X belong 
%               to which training block in *IC type sparse models. 
%               Default is []. In case of PIC, a cell array
%               containing index vectors specifying the blocking
%               structure for test data. IN FIC and CS+FIC a
%               vector of length n that points out the test inputs
%               that are also in the training set (if none, set
%               TSTIND = [])
%      yt     - optional observed yt in test points (see below)
%
%    NOTE! In case of FIC and PIC sparse approximation the
%    prediction for only some PREDCF covariance functions is just
%    an approximation since the covariance functions are coupled in
%    the approximation and are not strictly speaking additive
%    anymore.
%
%    For example, if you use covariance such as K = K1 + K2 your
%    predictions Ef1 = gp_pred(GP, X, Y, X, 'predcf', 1) and Ef2 =
%    gp_pred(gp, x, y, x, 'predcf', 2) should sum up to Ef =
%    gp_pred(gp, x, y, x). That is Ef = Ef1 + Ef2. With FULL model
%    this is true but with FIC and PIC this is true only
%    approximately. That is Ef \approx Ef1 + Ef2.
%
%    With CS+FIC the predictions are exact if the PREDCF covariance
%    functions are all in the FIC part or if they are CS
%    covariances.
%
%    NOTE! When making predictions with a subset of covariance
%    functions with FIC approximation the predictive variance can
%    in some cases be ill-behaved i.e. negative or
%    unrealistically small. This may happen because of the
%    approximative nature of the prediction.
%
%  See also
%    GP_SET, GP_OPTIM, DEMO_REGRESSION*
%

% Copyright (c) 2007-2010 Jarno Vanhatalo
% Copyright (c) 2008 Jouni Hartikainen
% Copyright (c) 2010 Aki Vehtari

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.

if iscell(gp) || numel(gp.jitterSigma2)>1 || isfield(gp,'latent_method')
  % use inference specific methods
  if iscell(gp)
    fh_pred=@ia_pred;
  elseif numel(gp.jitterSigma2)>1
    fh_pred=@mc_pred;
  elseif isfield(gp,'latent_method')
    switch gp.latent_method
      case 'Laplace'
        switch gp.lik.type
          case 'Softmax'
            fh_pred=@la_softmax_pred;
          otherwise
            fh_pred=@la_pred;
        end
      case 'EP'
        fh_pred=@ep_pred;
      case 'MCMC'
        fh_pred=@mc_pred;
    end
  else
    error('Logical error by coder of this function!')
  end
  switch nargout
    case 1
      [Eft] = fh_pred(gp, x, y, xt, varargin{:});
    case 2
      [Eft, Varft] = fh_pred(gp, x, y, xt, varargin{:});
    case 3
      [Eft, Varft, Eyt] = fh_pred(gp, x, y, xt, varargin{:});
    case 4
      [Eft, Varft, Eyt, Varyt] = fh_pred(gp, x, y, xt, varargin{:});
    case 5
      [Eft, Varft, Eyt, Varyt, pyt] = fh_pred(gp, x, y, xt, varargin{:});
  end
  return
end

ip=inputParser;
ip.FunctionName = 'GP_PRED';
ip.addRequired('gp',@isstruct);
ip.addRequired('x', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addRequired('y', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addRequired('xt',  @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addParamValue('yt', [], @(x) isreal(x) && all(isfinite(x(:))))
ip.addParamValue('predcf', [], @(x) isempty(x) || ...
                 isvector(x) && isreal(x) && all(isfinite(x)&x>0))
ip.addParamValue('tstind', [], @(x) isempty(x) || iscell(x) ||...
                 (isvector(x) && isreal(x) && all(isfinite(x)&x>0)))
ip.parse(gp, x, y, xt, varargin{:});
yt=ip.Results.yt;
predcf=ip.Results.predcf;
tstind=ip.Results.tstind;

tn = size(x,1);

if nargout > 4 && isempty(yt)
    error('GP_PRED -> To compute PYT, the YT has to be provided.')
end

% Evaluate this if sparse model is used
switch gp.type
  case 'FULL'
    [c, C]=gp_trcov(gp,x);
    K=gp_cov(gp,x,xt,predcf);

    % Are there specified mean functions
    if  ~isfield(gp,'meanf')         % zero mean
        if issparse(C)
            LD = ldlchol(C);
            Eft = K'*ldlsolve(LD,y);
        else
            L = chol(C)';
            %    y=K'*(C\y);
            a = L'\(L\y);
            Eft = K'*a;
        end
    else
        if isempty(C)
            C=0;
            K=0;
            L=[];
            Kyy=zeros(length(y),1);
        else
            L = chol(C)';
            Kyy = L'\(L\y);                 
        end
        
        [RB RAR] = mean_predf(gp,x,xt,K,L,Kyy,'gaussian',[]);    % terms with non-zero mean -prior
        
        Eft_zm = K'*Kyy;                       % mean with zero mean -prior
        Eft = Eft_zm + RB;
    end

    if nargout > 1
        % Are there specified mean functions
        if  ~isfield(gp,'meanf')
            if issparse(C)
                V = gp_trvar(gp,xt,predcf);
                Varft = V - diag(K'*ldlsolve(LD,K));
            else
                v = L\K;
                V = gp_trvar(gp,xt,predcf);
                % Vector of diagonal elements of covariance matrix
                % b = L\K;
                % Varft = V - sum(b.^2)';
                Varft = V - diag(v'*v);
            end
        else
            V = gp_trvar(gp,xt,predcf);
            if issparse(C)
                Varft_zm = V - diag(K'*ldlsolve(LD,K)); 
            else
                v = L\K;
                Varft_zm= V - diag(v'*v);
            end
           
            Varft = Varft_zm + RAR; 
        end
    end
    if nargout > 2
        % Scale mixture model in lik_smt is a special case 
        % handle it separately
        if ~strcmp(gp.lik.type, 'lik_smt') 
          % normal case
            [V, Cv] = gp_trvar(gp,xt,predcf);
            Eyt = Eft;
            Varyt = Varft + Cv - V;
            if nargout > 4
                pyt = norm_pdf(yt, Eyt, sqrt(Varyt));
            end
        else 
          % scale mixture case
            nu = gp.lik.nu;
            sigma2 = gp.lik.tau2.*gp.lik.alpha.^2;
            sigma = sqrt(sigma2);
            
            Eyt = Eft;
            Varyt = (nu./(nu-2).*sigma2);
            
            if nargout > 4
                for i2 = 1:length(Eft)
                    mean_app = Eft(i2);
                    sigm_app = sqrt(Varft(i2));
                    
                    pd = @(f) t_pdf(yt(i2), nu, f, sigma).*norm_pdf(f,Eft(i2),sqrt(Varft(i2)));
                    pyt(i2) = quadgk(pd, mean_app - 12*sigm_app, mean_app + 12*sigm_app);
                end
            end           
        end
    end
  case 'FIC'
    % Check the tstind vector
    if nargin > 5
        if ~isempty(tstind) && length(tstind) ~= size(x,1)
            error('tstind (if provided) has to be of same lenght as x.')
        end
    else
        tstind = [];
    end
    
    u = gp.X_u;
    m = size(u,1);
    % Turn the inducing vector on right direction
    if size(u,2) ~= size(x,2)
        u=u';
    end
    % Calculate some help matrices
    [Kv_ff, Cv_ff] = gp_trvar(gp, x);  % 1 x f  vector
    K_fu = gp_cov(gp, x, u);   % f x u
    K_uu = gp_trcov(gp, u);     % u x u, noiseles covariance K_uu
    K_nu = gp_cov(gp,xt,u);       % n x u
    Luu = chol(K_uu)';
    
    % Evaluate the Lambda (La) for specific model
    % Q_ff = K_fu*inv(K_uu)*K_fu'
    % Here we need only the diag(Q_ff), which is evaluated below
    B=Luu\(K_fu');
    Qv_ff=sum(B.^2)';
    Lav = Cv_ff-Qv_ff;   % 1 x f, Vector of diagonal elements
                         % iLaKfu = diag(inv(Lav))*K_fu = inv(La)*K_fu
    iLaKfu = zeros(size(K_fu));  % f x u,
    for i=1:length(x)
        iLaKfu(i,:) = K_fu(i,:)./Lav(i);  % f x u
    end
    A = K_uu+K_fu'*iLaKfu;
    A = (A+A')./2;

    L = iLaKfu/chol(A);
    p = y./Lav - L*(L'*y);

    % Prediction matrices formed with only subset of cf's.
    if ~isempty(predcf)
        K_fu = gp_cov(gp, x, u, predcf);   % f x u
        K_uu = gp_trcov(gp, u, predcf);     % u x u, noiseles covariance K_uu
        K_nu = gp_cov(gp,xt,u,predcf);       % n x u
    end
    Eft = K_nu*(K_uu\(K_fu'*p));

    % if the prediction is made for training set, evaluate Lav also for prediction points
    if ~isempty(tstind)
        [Kv_ff, Cv_ff] = gp_trvar(gp, xt(tstind,:), predcf);
        Luu = chol(K_uu)';
        B=Luu\(K_fu');
        Qv_ff=sum(B.^2)';
        Lav2 = zeros(size(Eft));
        Lav2(tstind) = Kv_ff-Qv_ff;
        Eft(tstind) = Eft(tstind) + Lav2(tstind).*p;
    end

    if nargout > 1
        [Knn_v, Cnn_v] = gp_trvar(gp,xt,predcf);
        Luu = chol(K_uu)';
        B=Luu\(K_fu');
        B2=Luu\(K_nu');
        
        Varft = Knn_v - sum(B2'.*(B*(repmat(Lav,1,size(K_uu,1)).\B')*B2)',2)  + sum((K_nu*(K_uu\(K_fu'*L))).^2, 2);

        % if the prediction is made for training set, evaluate Lav also for prediction points
        if ~isempty(tstind)
            Varft(tstind) = Varft(tstind)...
                - 2.*sum( B2(:,tstind)'.*(repmat((Lav.\Lav2(tstind)),1,m).*B'),2) ...
                + 2.*sum( B2(:,tstind)'*(B*L).*(repmat(Lav2(tstind),1,m).*L), 2)  ...
                - Lav2(tstind)./Lav.*Lav2(tstind) + sum((repmat(Lav2(tstind),1,m).*L).^2,2);
        end
    end
        
    
    if nargout > 2
        Eyt = Eft;
        Varyt = Varft + Cnn_v - Knn_v;
    end
    if nargout > 4
        pyt = norm_pdf(yt, Eyt, sqrt(Varyt));
    end
    
  case {'PIC' 'PIC_BLOCK'}
    u = gp.X_u;
    ind = gp.tr_index;
    if size(u,2) ~= size(x,2)
        u=u';
    end

    % Calculate some help matrices
    [Kv_ff, Cv_ff] = gp_trvar(gp, x);  % 1 x f  vector
    K_fu = gp_cov(gp, x, u);         % f x u
    K_nu = gp_cov(gp, xt, u);         % n x u
    K_uu = gp_trcov(gp, u);    % u x u, noiseles covariance K_uu
    Luu = chol(K_uu)';

    % Evaluate the Lambda (La) for specific model
    % Q_ff = K_fu*inv(K_uu)*K_fu'
    % Here we need only the diag(Q_ff), which is evaluated below
    B=Luu\K_fu';
    iLaKfu = zeros(size(K_fu));  % f x u
    for i=1:length(ind)
        Qbl_ff = B(:,ind{i})'*B(:,ind{i});
        [Kbl_ff, Cbl_ff] = gp_trcov(gp, x(ind{i},:));
        La{i} = Cbl_ff - Qbl_ff;
        iLaKfu(ind{i},:) = La{i}\K_fu(ind{i},:);    
    end
    A = K_uu+K_fu'*iLaKfu;
    A = (A+A')./2;            % Ensure symmetry
    L = iLaKfu/chol(A);

    tyy = y;
    % From this on evaluate the prediction
    % See Snelson and Ghahramani (2007) for details
    p=iLaKfu*(A\(iLaKfu'*tyy));
    for i=1:length(ind)
        p2(ind{i},:) = La{i}\tyy(ind{i},:);
    end
    p= p2-p;
    
    % Prediction matrices formed with only subsetof cf's.
    if ~isempty(predcf)
        K_fu = gp_cov(gp, x, u, predcf);        % f x u
        K_nu = gp_cov(gp, xt, u, predcf);         % n x u
        K_uu = gp_trcov(gp, u, predcf);          % u x u, noiseles covariance K_uu
    end
        
    iKuuKuf = K_uu\K_fu';    
    w_bu=zeros(length(xt),length(u));
    w_n=zeros(length(xt),1);
    for i=1:length(ind)
        w_bu(tstind{i},:) = repmat((iKuuKuf(:,ind{i})*p(ind{i},:))', length(tstind{i}),1);
        K_nf = gp_cov(gp, xt(tstind{i},:), x(ind{i},:),predcf);              % n x u
        w_n(tstind{i},:) = K_nf*p(ind{i},:);
    end
    
    Eft = K_nu*(iKuuKuf*p) - sum(K_nu.*w_bu,2) + w_n;
    

    if nargout > 1        
        % Form iLaKfu again if a subset of cf's is used for making predictions
        if ~isempty(predcf)
            iLaKfu = zeros(size(K_fu));  % f x u
            for i=1:length(ind)
                iLaKfu(ind{i},:) = La{i}\K_fu(ind{i},:);    
            end
        end
        
        kstarstar = gp_trvar(gp, xt, predcf);
        KnuiKuu = K_nu/K_uu;
        KufiLaKfu = K_fu'*iLaKfu;
        QnfL = KnuiKuu*(K_fu'*L);
        Varft1 = zeros(size(xt,1),1);
        Varft2 = zeros(size(xt,1),1);
        Varft3 = zeros(size(xt,1),1);
        for i=1:length(ind)
            KubiLaKbu = K_fu(ind{i},:)'/La{i}*K_fu(ind{i},:);
            nonblock = KufiLaKfu - KubiLaKbu;
            Varft1(tstind{i}) = diag(KnuiKuu(tstind{i},:)*nonblock*KnuiKuu(tstind{i},:)');
            
            Knb = gp_cov(gp, xt(tstind{i},:), x(ind{i},:), predcf);
            Varft2(tstind{i}) = diag(Knb/La{i}*Knb');
            
            KnbL = Knb*L(ind{i},:);
            QnbL = KnuiKuu(tstind{i},:)*(K_fu(ind{i},:)'*L(ind{i},:));
            %Varft3(tstind{i}) = sum(QnfL(tstind{i},:) - QnbL + KnbL,2);
            Varft3(tstind{i}) = diag((QnfL(tstind{i},:) - QnbL + KnbL)*(QnfL(tstind{i},:) - QnbL + KnbL)');
        end        
        Varft = kstarstar - (Varft1 + Varft2 - Varft3);
    end
    
    
% $$$     B2=Luu\(K_nu');
% $$$     C = B'*B;
% $$$     KKnn = gp_trcov(gp,xt,predcf);
% $$$     Knn = B2'*B2;
% $$$     Knf = B2'*B;
% $$$     KKnf = gp_cov(gp,xt,x,predcf);
% $$$     for i=1:length(ind)
% $$$         C(ind{i},ind{i}) = C(ind{i},ind{i}) + La{i};
% $$$         Knn(ind{i},ind{i}) = Knn(ind{i},ind{i}) + KKnn(tstind{i},tstind{i}) - B2(:,tstind{i})'*B2(:,tstind{i});
% $$$         Knf(tstind{i},ind{i}) = Knf(tstind{i},ind{i}) + KKnf(tstind{i},ind{i}) - B2(:,tstind{i})'*B(:,ind{i});
% $$$     end
% $$$     
% $$$     L = chol(C)';
% $$$     %    y=K'*(C\y);
% $$$     a = L'\(L\y);
% $$$     Eft = Knf*a;
% $$$         
% $$$     v = L\Knf';
% $$$     Varft = diag(Knn) - diag(v'*v);
    
    if nargout > 2
        Eyt = Eft;
        [Knn_v, Cnn_v] = gp_trvar(gp,xt,predcf);
        Varyt = Varft + Cnn_v - Knn_v;
    end
    if nargout > 4
        pyt = norm_pdf(yt, Eyt, sqrt(Varyt));
    end
  case 'CS+FIC'
    % Here tstind = 1 if the prediction is made for the training set 
    if nargin > 5
        if ~isempty(tstind) && length(tstind) ~= size(x,1)
            error('tstind (if provided) has to be of same lenght as x.')
        end
    else
        tstind = [];
    end
    
    n = size(x,1);
    n2 = size(xt,1);

    u = gp.X_u;
    m = size(u,1);
    ncf = length(gp.cf);
    
    % Indexes to all non-compact support and compact support covariances.
    cf1 = [];
    cf2 = [];
    % Indexes to non-CS and CS covariances, which are used for predictions
    predcf1 = [];
    predcf2 = [];    

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
    
    % First evaluate needed covariance matrices
    % v defines that parameter is a vector
    [Kv_ff, Cv_ff] = gp_trvar(gp, x, cf1);  % f x 1  vector    
    K_fu = gp_cov(gp, x, u, cf1);         % f x u
    K_uu = gp_trcov(gp, u, cf1);    % u x u, noiseles covariance K_uu
    K_uu = (K_uu+K_uu')./2;     % ensure the symmetry of K_uu

    Luu  = chol(K_uu)';
    K_nu = gp_cov(gp, xt, u, cf1);         % n x u

    % Evaluate the Lambda (La)
    % Q_ff = K_fu*inv(K_uu)*K_fu'
    B=Luu\(K_fu');       % u x f
    Qv_ff=sum(B.^2)';
    Lav = Cv_ff-Qv_ff;   % f x 1, Vector of diagonal elements

    K_cs = gp_trcov(gp,x,cf2);
    Kcs_nf = gp_cov(gp, xt, x, predcf2);
    La = sparse(1:tn,1:tn,Lav,tn,tn) + K_cs;
    
    iLaKfu = La\K_fu;
    A = K_uu+K_fu'*iLaKfu;
    A = (A+A')./2;     % Ensure symmetry
    L = iLaKfu/chol(A);
    
    p = La\y - L*(L'*y);

    %p2 = y./Lav - iLaKfu*(A\(iLaKfu'*y));
    %    Knf = K_nu*(K_uu\K_fu');

    K_fu = gp_cov(gp, x, u, predcf1);       % f x u
    K_uu = gp_trcov(gp, u, predcf1);         % u x u, noiseles covariance K_uu
    K_uu = (K_uu+K_uu')./2;                  % ensure the symmetry of K_uu
    K_nu = gp_cov(gp, xt, u, predcf1);        % n x u    

    % Calculate the predictive mean according to the type of
    % covariance functions used for making the prediction
    if ptype == 1
        Eft = K_nu*(K_uu\(K_fu'*p));
    elseif ptype == 2
        Eft = Kcs_nf*p;
    else 
        Eft = K_nu*(K_uu\(K_fu'*p)) + Kcs_nf*p;
    end
    
    % evaluate also Lav2 if the prediction is made for training set
    if ~isempty(tstind)
        [Kv_ff, Cv_ff] = gp_trvar(gp, xt(tstind,:), predcf1);
        Luu = chol(K_uu)';
        B=Luu\(K_fu');
        Qv_ff=sum(B.^2)';
        Lav2 = zeros(size(Eft));
        Lav2(tstind) = Kv_ff-Qv_ff;
    end  

    % Add also Lav2 if the prediction is made for training set
    % and non-CS covariance function is used for prediction
    if ~isempty(tstind) && (ptype == 1 || ptype == 3)
        Eft(tstind) = Eft(tstind) + Lav2(tstind).*p;
    end
    
    if nargout > 1
        [Knn_v, Cnn_v] = gp_trvar(gp,xt,predcf);
        Luu = chol(K_uu)';
        B=Luu\(K_fu');
        B2=Luu\(K_nu');
        iLaKfu = La\K_fu;
        
        % Calculate the predictive variance according to the type
        % covariance functions used for making the prediction
        if ptype == 1 || ptype == 3                            
            % FIC part of the covariance
            Varft = Knn_v - sum(B2'.*(B*(La\B')*B2)',2) + sum((K_nu*(K_uu\(K_fu'*L))).^2, 2);
            % Add Lav2 if the prediction is made for the training set
            if  ~isempty(tstind)
                % Non-CS covariance
                if ptype == 1
                    Kcs_nf = sparse(tstind,1:n,Lav2(tstind),n2,n);
                % Non-CS and CS covariances
                else
                    Kcs_nf = Kcs_nf + sparse(tstind,1:n,Lav2(tstind),n2,n);
                end
                % Add Lav2 and possibly Kcs_nf
                Varft = Varft - sum((Kcs_nf/chol(La)).^2,2) + sum((Kcs_nf*L).^2, 2) ...
                       - 2.*sum((Kcs_nf*iLaKfu).*(K_uu\K_nu')',2) + 2.*sum((Kcs_nf*L).*(L'*K_fu*(K_uu\K_nu'))' ,2);                
            % In case of both non-CS and CS prediction covariances add 
            % only Kcs_nf if the prediction is not done for the training set 
            elseif ptype == 3
                Varft = Varft - sum((Kcs_nf/chol(La)).^2,2) + sum((Kcs_nf*L).^2, 2) ...
                       - 2.*sum((Kcs_nf*iLaKfu).*(K_uu\K_nu')',2) + 2.*sum((Kcs_nf*L).*(L'*K_fu*(K_uu\K_nu'))' ,2);
            end
        % Prediction with only CS covariance
        elseif ptype == 2
            Varft = Knn_v - sum((Kcs_nf/chol(La)).^2,2) + sum((Kcs_nf*L).^2, 2) ;
        end        
    end
    
% $$$     Lav_pr = Kv_ff-Qv_ff;
% $$$     K2 = B'*B + K_cs + diag(Lav_pr);
% $$$     C = B'*B + K_cs + diag(Lav);
% $$$     K = B'*B + Kcs_nf + diag(Lav_pr); 
% $$$     
% $$$     L = chol(C)';
% $$$     %    y=K'*(C\y);
% $$$     a = L'\(L\y);
% $$$     Eft = K'*a;
% $$$ 
% $$$     v = L\K;
% $$$     V = gp_trvar(gp,xt,predcf);
% $$$     Varft = V - diag(v'*v);

    if nargout > 2
        Eyt = Eft;
        Varyt = Varft + Cnn_v - Knn_v;
    end
    if nargout > 4
        pyt = norm_pdf(yt, Eyt, sqrt(Varyt));
    end
    
  case {'VAR' 'DTC' 'SOR'}
    % Check the tstind vector
    if nargin > 5
        if ~isempty(tstind) && length(tstind) ~= size(tx,1)
            error('tstind (if provided) has to be of same lenght as tx.')
        end
    else
        tstind = [];
    end
    
    u = gp.X_u;
    m = size(u,1);
    % Turn the inducing vector on right direction
    if size(u,2) ~= size(x,2)
        u=u';
    end
    % Calculate some help matrices
    [Kv_ff, Cv_ff] = gp_trvar(gp, x);  % 1 x f  vector
    K_fu = gp_cov(gp, x, u);   % f x u
    K_uu = gp_trcov(gp, u);     % u x u, noiseles covariance K_uu
    K_nu = gp_cov(gp,xt,u);       % n x u
    Luu = chol(K_uu)';
    
    % Evaluate the Lambda (La) for specific model
    % Q_ff = K_fu*inv(K_uu)*K_fu'
    % Here we need only the diag(Q_ff), which is evaluated below
    B=Luu\(K_fu');
    Qv_ff=sum(B.^2)';
    Lav = Cv_ff-Kv_ff;   % 1 x f, Vector of diagonal elements
                         % iLaKfu = diag(inv(Lav))*K_fu = inv(La)*K_fu
    iLaKfu = zeros(size(K_fu));  % f x u,
    for i=1:length(x)
        iLaKfu(i,:) = K_fu(i,:)./Lav(i);  % f x u
    end
    A = K_uu+K_fu'*iLaKfu;
    A = (A+A')./2;

    L = iLaKfu/chol(A);
    p = y./Lav - L*(L'*y);

    % Prediction matrices formed with only subset of cf's.
    if ~isempty(predcf)
        K_fu = gp_cov(gp, x, u, predcf);   % f x u
        K_uu = gp_trcov(gp, u, predcf);     % u x u, noiseles covariance K_uu
        K_nu = gp_cov(gp,xt,u,predcf);       % n x u
    end
    Eft = K_nu*(K_uu\(K_fu'*p));


    if nargout > 1
        [Knn_v, Cnn_v] = gp_trvar(gp,xt,predcf);
        Luu = chol(K_uu)';
        B=Luu\(K_fu');
        B2=Luu\(K_nu');
        
        Varftr = sum(B2'.*(B*bsxfun(@ldivide,Lav,B')*B2)',2) - sum((K_nu*(K_uu\(K_fu'*L))).^2, 2);
        switch gp.type
          case {'VAR' 'DTC'}
            Varft = Knn_v - Varftr;
          case  'SOR'
            Varft = sum(B2.^2,1)' - Varftr;
        end

    end
    if nargout > 2
        Eyt = Eft;
        switch gp.type
          case {'VAR' 'DTC'}
            Varyt = Varft + Cnn_v - Knn_v;
          case 'SOR'
            Varyt = Varft + Cnn_v - sum(B2.^2,1)';
        end
    end
    if nargout > 4
        pyt = norm_pdf(y, Eyt, sqrt(Varyt));
    end  
    
  case 'SSGP'
    if nargin > 4
        error(['Prediction with a subset of original ' ...
               'covariance functions not currently implemented with SSGP']);
    end

    [Phi_f, S] = gp_trcov(gp, x);
    Phi_a = gp_trcov(gp, xt);
    m = size(Phi_f,2);
    ns = eye(m,m)*S(1,1);
    
    L = chol(Phi_f'*Phi_f + ns)';
    Eft = Phi_a*(L'\(L\(Phi_f'*y)));

    
    if nargout > 1
        Varft = sum(Phi_a/L',2)*S(1,1);
    end
    if nargout > 2
        error('gp_pred with three output arguments is not implemented for SSGP!')
    end
end