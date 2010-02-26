function [e, edata, eprior, f, L, a, La2] = gpla_e(w, gp, x, y, param, varargin)
%GPLA_E Conduct LAplace approximation and return marginal log posterior estimate
%
%	Description
%	E = GPLA_E(W, GP, X, Y, PARAM) takes a gp data structure GP together
%	with a matrix X of input vectors and a matrix Y of target vectors,
%	and finds the Laplace approximation for the conditional posterior p(Y|X, th), 
%       where th is the hyperparameters. Returns the energy E at th. Each row 
%       of X corresponds to one input vector and each row of Y corresponds to 
%       one target vector.
%
%	[E, EDATA, EPRIOR] = GPLA_E(W, GP, P, T, PARAM) also returns the data and
%	prior components of the total error.
%
%       The energy is minus log posterior cost function:
%            E = EDATA + EPRIOR 
%              = - log p(Y|X, th) - log p(th),
%       where th represents the hyperparameters (lengthScale, magnSigma2...), X is
%       inputs and Y is observations (regression) or latent values (non-Gaussian
%       likelihood).
%
%       NOTE! The CS+FIC model is not supported 
%
%	See also
%       GPLA_G, LA_PRED, GP_E
%
%

% Copyright (c) 2007-2009      Jarno Vanhatalo
% 
% The Newtons iteration is implemented as described in 
% Rasmussen and Williams (2006).

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.

    if strcmp(w, 'init')
        w0 = rand(size(gp_pak(gp, param)));
        e0=[];
        edata0= inf;
        eprior0=[];
        W = zeros(size(y));
        W0 = W;
        f0 = zeros(size(y));
        L0 = [];
        f = zeros(size(y));
        n0 = size(x,1);
        La20 = [];
        a0 = 0;

        laplace_algorithm(gp_pak(gp,param), gp, x, y, param, varargin);

        gp.fh_e = @laplace_algorithm;
        e = gp;
    else
        [e, edata, eprior, f, L, a, La2] = feval(gp.fh_e, w, gp, x, y, param, varargin);
    end

    function [e, edata, eprior, f, L, a, La2] = laplace_algorithm(w, gp, x, y, param, varargin)
        
        if abs(w-w0) < 1e-8 % 1e-8
               % The covariance function parameters haven't changed so just
               % return the Energy and the site parameters that are saved
            e = e0;
            edata = edata0;
            eprior = eprior0;
            f = f0;
            L = L0;
            La2 = La20;
            W = W0;
            a = a0;
        else

            gp=gp_unpak(gp, w, param);
            ncf = length(gp.cf);
            n = length(x);

            % Begin optimization from the old f if it is better than the new
            %if edata0 < 
            %f = f0;
            f = zeros(size(f0));

            % =================================================
            % First Evaluate the data contribution to the error
            switch gp.type
                % ============================================================
                % FULL
                % ============================================================
              case 'FULL'
                K = gp_trcov(gp, x);

                % If K is sparse, permute all the inputs so that evaluations are more efficient
                if issparse(K)
% $$$                     p = analyze(K);
% $$$                     r(p) = 1:n;
% $$$                     gp.likelih = feval(gp.likelih.fh_permute, gp.likelih, p);
% $$$                     y = y(p);
% $$$                     K = K(p,p);
                    LD = ldlchol(K);
                else
                    LD = chol(K);
                end
                
                switch gp.laplace_opt.optim_method
                    % find the mode by fminunc large scale method
                  case 'fminunc_large'
                    if ~isfield(gp.laplace_opt, 'fminunc_opt')
                        opt=optimset('GradObj','on');
                        opt=optimset(opt,'Hessian','on');
                        if issparse(K)
                            fhm = @(W, f, varargin) (ldlsolve(LD,f) + repmat(W,1,size(f,2)).*f);  % W*f; %
                        else
                            fhm = @(W, f, varargin) (LD\(LD'\f) + repmat(W,1,size(f,2)).*f);  % W*f; %
                        end                            
                        opt=optimset(opt,'HessMult', fhm);
                        opt=optimset(opt,'TolX', 1e-12);
                        opt=optimset(opt,'TolFun', 1e-12);
                        opt=optimset(opt,'LargeScale', 'on');
                        opt=optimset(opt,'Display', 'off'); % 'iter'
                    else
                        opt = gp.laplace_opt.fminunc_opt;
                    end
               
                    if issparse(K)
                        fe = @(f, varargin) (0.5*f*(ldlsolve(LD,f')) - feval(gp.likelih.fh_e, gp.likelih, y, f'));
                        fg = @(f, varargin) (ldlsolve(LD,f') - feval(gp.likelih.fh_g, gp.likelih, y, f', 'latent'))';
                        fh = @(f, varargin) (-feval(gp.likelih.fh_g2, gp.likelih, y, f', 'latent')); %inv(K) + diag(g2(f', gp.likelih)) ; %
                    else
                        fe = @(f, varargin) (0.5*f*(LD\(LD'\f')) - feval(gp.likelih.fh_e, gp.likelih, y, f'));
                        fg = @(f, varargin) (LD\(LD'\f') - feval(gp.likelih.fh_g, gp.likelih, y, f', 'latent'))';
                        fh = @(f, varargin) (-feval(gp.likelih.fh_g2, gp.likelih, y, f', 'latent')); %inv(K) + diag(g2(f', gp.likelih)) ; %
                    end
                    
                    mydeal = @(varargin)varargin{1:nargout};
                    [f,fval,exitflag,output] = fminunc(@(ww) mydeal(fe(ww), fg(ww), fh(ww)), f', opt);
                    f = f';

                    if issparse(K)
                        a = ldlsolve(LD,f);
                    else
                        a = LD\(LD'\f);
                    end

                  case 'newton'
                    tol = 1e-12;
                    a = f;
                    W = -feval(gp.likelih.fh_g2, gp.likelih, y, f, 'latent');
                    dlp = feval(gp.likelih.fh_g, gp.likelih, y, f, 'latent');
                    lp_new = feval(gp.likelih.fh_e, gp.likelih, y, f);
                    lp_old = -Inf;
                    
                    while lp_new - lp_old > tol                                
                        lp_old = lp_new; a_old = a; 
                        sW = sqrt(W);    
                        if issparse(K)
                            sW = sparse(1:n, 1:n, sW, n, n);
                            L = ldlchol( speye(n)+sW*K*sW );
                        else
                            L = chol(eye(n)+sW*sW'.*K);                        % L'*L=B=eye(n)+sW*K*sW
                        end
                        b = W.*f+dlp;
                        if issparse(K)
                            a = b - sW*ldlsolve(L,sW*(K*b));
                        else
                            a = b - sW.*(L\(L'\(sW.*(K*b))));
                        end
                        f = K*a;
                        W = -feval(gp.likelih.fh_g2, gp.likelih, y, f, 'latent');
                        dlp = feval(gp.likelih.fh_g, gp.likelih, y, f, 'latent');
                        lp = feval(gp.likelih.fh_e, gp.likelih, y, f);
                        lp_new = -a'*f/2 + lp;
                        i = 0;
                        while i < 10 && lp_new < lp_old                       % if objective didn't increase
                            a = (a_old+a)/2;                                  % reduce step size by half
                            f = K*a;
                            W = -feval(gp.likelih.fh_g2, gp.likelih, y, f, 'latent');
                            lp = feval(gp.likelih.fh_e, gp.likelih, y, f);
                            lp_new = -a'*f/2 + lp;
                            i = i+1;
                        end 
                    end
                    
                  case 'likelih_specific'                            
                    [f, a] = feval(gp.likelih.fh_optimizef, gp, y, K);
                end
                
                % evaluate the approximate log marginal likelihood
                W = -feval(gp.likelih.fh_g2, gp.likelih, y, f, 'latent');
                logZ = 0.5 * f'*a - feval(gp.likelih.fh_e, gp.likelih, y, f);
                if min(W) >= 0
                    if issparse(K)
                        W = sparse(1:n,1:n, -feval(gp.likelih.fh_g2, gp.likelih, y, f, 'latent'), n,n);
                        sqrtW = sqrt(W);
                        B = sparse(1:n,1:n,1,n,n) + sqrtW*K*sqrtW;
                        L = ldlchol(B);

                        % Note that here we use LDL cholesky
                        edata = logZ + 0.5.*sum(log(diag(L))); % 0.5*log(det(eye(size(K)) + K*W)) ; %
                        
% $$$                         % Reorder some of the returned and stored values
% $$$                         f = f(r);
                    else
                        sW = sqrt(W);
                        B = eye(size(K)) + sW*sW'.*K;
                        L = chol(B)';
                        edata = logZ + sum(log(diag(L))); % 0.5*log(det(eye(size(K)) + K*W)) ; %
                    end
                else
                    [W2,I] = sort(W, 1, 'descend');

                    L = chol(K);
                    L1 = L;
                    for jj=1:size(K,1)
                        i = I(jj);
                        ll = sum(L(:,i).^2);
                        l = L'*L(:,i);
                        upfact = W(i)./(1 + W(i).*ll);
                        
                        % Check that Cholesky factorization will remain positive definite
                        if 1./ll + W(i) < 0 %1 + W(i).*ll <= 0 | abs(upfact) > abs(1./ll) %upfact > 1./ll
                            warning('gpla_e: 1./Sigma(i,i) + W(i) < 0')
                            
                            ind = 1:i-1;
                            mu = K(i,ind)*feval(gp.likelih.fh_g, gp.likelih, y(I(ind)), f(I(ind)), 'latent');
                            upfact = feval(gp.likelih.fh_upfact, gp, y(I(i)), mu, ll);
                        end
                        if upfact > 0
                            L = cholupdate(L, l.*sqrt(upfact), '-');
                        else
                            L = cholupdate(L, l.*sqrt(-upfact));
                        end
                    end
                    edata = logZ + sum(log(diag(L1))) - sum(log(diag(L)));
                end
                                                                
                La2 = W;

                % ============================================================
                % FIC
                % ============================================================
              case 'FIC'
                u = gp.X_u;
                m = length(u);

                % First evaluate needed covariance matrices
                % v defines that parameter is a vector
                [Kv_ff, Cv_ff] = gp_trvar(gp, x);  % f x 1  vector
                K_fu = gp_cov(gp, x, u);         % f x u                
                K_uu = gp_trcov(gp, u);    % u x u, noiseles covariance K_uu
                Luu = chol(K_uu)';
                % Evaluate the Lambda (La)
                % Q_ff = K_fu*inv(K_uu)*K_fu'
                % Here we need only the diag(Q_ff), which is evaluated below
                B=Luu\(K_fu');       % u x f
                Qv_ff=sum(B.^2)';
                Lav = Cv_ff-Qv_ff;   % f x 1, Vector of diagonal elements
                iLaKfu = repmat(Lav,1,m).\K_fu;  % f x u
                A = K_uu+K_fu'*iLaKfu;  A = (A+A')./2;     % Ensure symmetry
                A = chol(A);
                L = iLaKfu/A;
            
                switch gp.laplace_opt.optim_method
                    % find the mode by fminunc large scale method
                  case 'fminunc_large'
                    if ~isfield(gp.laplace_opt, 'fminunc_opt')
                        opt=optimset('GradObj','on');
                        opt=optimset(opt,'Hessian','on');
                        fhm = @(W, f, varargin) (f./repmat(Lav,1,size(f,2)) - L*(L'*f)  + repmat(W,1,size(f,2)).*f);  % hessian*f; %
                        opt=optimset(opt,'HessMult', fhm);
                        opt=optimset(opt,'TolX', 1e-8);
                        opt=optimset(opt,'TolFun', 1e-8);
                        opt=optimset(opt,'LargeScale', 'on');
                        opt=optimset(opt,'Display', 'off');   % 'iter'
                    else
                        opt = gp.laplace_opt.fminunc_opt;
                    end

                    fe = @(f, varargin) (0.5*f*(f'./repmat(Lav,1,size(f',2)) - L*(L'*f')) - feval(gp.likelih.fh_e, gp.likelih, y, f'));
                    fg = @(f, varargin) (f'./repmat(Lav,1,size(f',2)) - L*(L'*f') - feval(gp.likelih.fh_g, gp.likelih, y, f', 'latent'))';
                    fh = @(f, varargin) (-feval(gp.likelih.fh_g2, gp.likelih, y, f', 'latent'));
                    mydeal = @(varargin)varargin{1:nargout};
                    [f,fval,exitflag,output] = fminunc(@(ww) mydeal(fe(ww), fg(ww), fh(ww)), f', opt);
                    f = f';

                    a = f./Lav - L*L'*f;
                    
                    % find the mode by newton method
                  case 'newton'
                    tol = 1e-12;
                    a = f;
                    W = -feval(gp.likelih.fh_g2, gp.likelih, y, f, 'latent');
                    dlp = feval(gp.likelih.fh_g, gp.likelih, y, f, 'latent');
                    lp_new = feval(gp.likelih.fh_e, gp.likelih, y, f);
                    lp_old = -Inf;
                    
                    while lp_new - lp_old > tol                        % begin Newton's iterations
                        lp_old = lp_new; a_old = a; 
                        sW = sqrt(W);
                        
                        Lah = 1 + sW.*Lav.*sW;
                        sWKfu = repmat(sW,1,m).*K_fu;
                        A = K_uu + sWKfu'*(repmat(Lah,1,m).\sWKfu);   A = (A+A')./2;
                        Lb = (repmat(Lah,1,m).\sWKfu)/chol(A);
                        b = W.*f+dlp;
                        b2 = sW.*(Lav.*b + B'*(B*b));
                        a = b - sW.*(b2./Lah - Lb*(Lb'*b2));
                        
                        f = Lav.*a + B'*(B*a);
                        W = -feval(gp.likelih.fh_g2, gp.likelih, y, f, 'latent');
                        dlp = feval(gp.likelih.fh_g, gp.likelih, y, f, 'latent');
                        lp = feval(gp.likelih.fh_e, gp.likelih, y, f);
                        lp_new = -a'*f/2 + lp;
                        i = 0;
                        while i < 10 && lp_new < lp_old                       % if objective didn't increase
                            a = (a_old+a)/2;                                  % reduce step size by half
                            f = Lav.*a + B'*(B*a);
                            W = -feval(gp.likelih.fh_g2, gp.likelih, y, f, 'latent');
                            lp = feval(gp.likelih.fh_e, gp.likelih, y, f);
                            lp_new = -a'*f/2 + lp;
                            i = i+1;
                        end 
                    end                                                    % end Newton's iterations 
                  case 'likelih_specific'
                    [f, a] = feval(gp.likelih.fh_optimizef, gp, y, K_uu, Lav, K_fu);
                  otherwise 
                    error('gpla_e: Unknown optimization method !')
                end
                               
                W = -feval(gp.likelih.fh_g2, gp.likelih, y, f, 'latent');
                logZ = 0.5*f'*a - feval(gp.likelih.fh_e, gp.likelih, y, f);
                
                if W >= 0
                    sqrtW = sqrt(W);
                    
                    Lah = 1 + sqrtW.*Lav.*sqrtW;
                    sWKfu = repmat(sqrtW,1,m).*K_fu;
                    A = K_uu + sWKfu'*(repmat(Lah,1,m).\sWKfu);   A = (A+A')./2;
                    A = chol(A);
                    edata = sum(log(Lah)) - 2*sum(log(diag(Luu))) + 2*sum(log(diag(A)));
                    edata = logZ + 0.5*edata;
                else
                    K = diag(Lav) + B'*B;
% $$$                         [W,I] = sort(W, 1, 'descend');
% $$$                         K = K(I,I);
                    [W2,I] = sort(W, 1, 'descend');
                        
                    L = chol(K);
                    L1 = L;
                    for jj=1:size(K,1)
                        i = I(jj);
                        ll = sum(L(:,i).^2);
                        l = L'*L(:,i);
                        upfact = W(i)./(1 + W(i).*ll);
                        
                        % Check that Cholesky factorization will remain positive definite
                        if 1 + W(i).*ll <= 0 | upfact > 1./ll
                            warning('gpla_e: 1 + W(i).*ll < 0')
                            
                            ind = 1:i-1;
                            mu = K(i,ind)*feval(gp.likelih.fh_g, gp.likelih, y(I(ind)), f(I(ind)), 'latent');
                            upfact = feval(gp.likelih.fh_upfact, gp, y(I(i)), mu, ll);
                            
    % $$$                                 W2 = -1./(ll+1e-3);
    % $$$                                 upfact = W2./(1 + W2.*ll);
                        end
                        if upfact > 0
                            L = cholupdate(L, l.*sqrt(upfact), '-');
                        else
                            L = cholupdate(L, l.*sqrt(-upfact));
                        end
                    end
                    edata = logZ + sum(log(diag(L1))) - sum(log(diag(L)));  % sum(log(diag(chol(K)))) + sum(log(diag(chol((inv(K) + W)))));
                end
                    
                    
                La2 = Lav;

                % ============================================================
                % PIC
                % ============================================================
              case {'PIC' 'PIC_BLOCK'}
                ind = gp.tr_index;
                u = gp.X_u;
                m = length(u);

                % First evaluate needed covariance matrices
                % v defines that parameter is a vector
                K_fu = gp_cov(gp, x, u);         % f x u
                K_uu = gp_trcov(gp, u);    % u x u, noiseles covariance K_uu
                K_uu = (K_uu+K_uu')./2;     % ensure the symmetry of K_uu
                Luu = chol(K_uu)';
                % Evaluate the Lambda (La)
                % Q_ff = K_fu*inv(K_uu)*K_fu'
                % Here we need only the diag(Q_ff), which is evaluated below
                B=Luu\(K_fu');       % u x f

                % First some helper parameters
                iLaKfu = zeros(size(K_fu));  % f x u
                for i=1:length(ind)
                    Qbl_ff = B(:,ind{i})'*B(:,ind{i});
                    [Kbl_ff, Cbl_ff] = gp_trcov(gp, x(ind{i},:));
                    Labl{i} = Cbl_ff - Qbl_ff;
                    LLabl{i} = chol(Labl{i});
                    iLaKfu(ind{i},:) = LLabl{i}\(LLabl{i}'\K_fu(ind{i},:));
                end
                A = K_uu+K_fu'*iLaKfu;
                A = (A+A')./2;     % Ensure symmetry
                A = chol(A);
                L = iLaKfu/A;
                % Begin optimization
                switch gp.laplace_opt.optim_method
                    % find the mode by fminunc large scale method
                  case 'fminunc_large'
                    if ~isfield(gp.laplace_opt, 'fminunc_opt')
                        opt=optimset('GradObj','on');
                        opt=optimset(opt,'Hessian','on');
                        fhm = @(W, f, varargin) (iKf(f)  + repmat(W,1,size(f,2)).*f);
                        opt=optimset(opt,'HessMult', fhm);
                        opt=optimset(opt,'TolX', 1e-8);
                        opt=optimset(opt,'TolFun', 1e-8);
                        opt=optimset(opt,'LargeScale', 'on');
                        opt=optimset(opt,'Display', 'off');   % 'iter'
                    else
                        opt = gp.laplace_opt.fminunc_opt;
                    end

                    [f,fval,exitflag,output] = fminunc(@(ww) egh(ww), f', opt);
                    f = f';
                    
                    a = iKf(f);
                                        
                    % find the mode by Scaled conjugate gradient method
                  case 'newton'
                    tol = 1e-12;
                    a = f;
                    W = -feval(gp.likelih.fh_g2, gp.likelih, y, f, 'latent');
                    dlp = feval(gp.likelih.fh_g, gp.likelih, y, f, 'latent');
                    lp_new = feval(gp.likelih.fh_e, gp.likelih, y, f);
                    lp_old = -Inf;
                    
                    while lp_new - lp_old > tol                        % begin Newton's iterations
                        lp_old = lp_new; a_old = a;
                        sW = sqrt(W);

                        V = repmat(sW,1,m).*K_fu;
                        for i=1:length(ind)
                            Lah{i} = eye(size(Labl{i})) + diag(sW(ind{i}))*Labl{i}*diag(sW(ind{i}));
                            LLah{i} = chol(Lah{i});
                            V2(ind{i},:) = LLah{i}\(LLah{i}'\V(ind{i},:));
                        end                        
                        
                        A = K_uu + V'*V2;   A = (A+A')./2;
                        Lb = V2/chol(A);
                        b = W.*f+dlp;
                        b2 = B'*(B*b);
                        bt = zeros(size(b2));
                        for i=1:length(ind)
                            b2(ind{i}) = sW(ind{i}).*(Labl{i}*b(ind{i}) + b2(ind{i})); 
                            bt(ind{i}) = LLah{i}\(LLah{i}'\b2(ind{i}));
                        end
                        a = b - sW.*(bt - Lb*(Lb'*b2));

                        f = B'*(B*a);
                        for i=1:length(ind)
                            f(ind{i}) = Labl{i}*a(ind{i}) + f(ind{i}) ;
                        end
                        W = -feval(gp.likelih.fh_g2, gp.likelih, y, f, 'latent');
                        dlp = feval(gp.likelih.fh_g, gp.likelih, y, f, 'latent');
                        lp = feval(gp.likelih.fh_e, gp.likelih, y, f);
                        lp_new = -a'*f/2 + lp;
                        i = 0;
                        while i < 10 && lp_new < lp_old                       % if objective didn't increase
                            a = (a_old+a)/2;                                  % reduce step size by half                            
                            f = B'*(B*a);
                            for i=1:length(ind)
                                f(ind{i}) = Labl{i}*a(ind{i}) + f(ind{i}) ;
                            end
                            W = -feval(gp.likelih.fh_g2, gp.likelih, y, f, 'latent');
                            lp = feval(gp.likelih.fh_e, gp.likelih, y, f);
                            lp_new = -a'*f/2 + lp;
                            i = i+1;
                        end 
                    end                                                    % end Newton's iterations 
                    
                end
                
                W = -feval(gp.likelih.fh_g2, gp.likelih, y, f, 'latent');
                sqrtW = sqrt(W);
               
                logZ = 0.5*f'*a - feval(gp.likelih.fh_e, gp.likelih, y, f);
                
                WKfu = repmat(sqrtW,1,m).*K_fu;
                edata = 0;
                for i=1:length(ind)
                    Lahat = eye(size(Labl{i})) + diag(sqrtW(ind{i}))*Labl{i}*diag(sqrtW(ind{i}));
                    LLahat = chol(Lahat);
                    iLahatWKfu(ind{i},:) = LLahat\(LLahat'\WKfu(ind{i},:));
                    edata = edata + 2.*sum(log(diag(LLahat)));
                end
                A = K_uu + WKfu'*iLahatWKfu;   A = (A+A')./2;
                A = chol(A);
                edata =  edata - 2*sum(log(diag(Luu))) + 2*sum(log(diag(A)));
                edata = logZ + 0.5*edata;

                La2 = Labl;              
                
                % ============================================================
                % CS+FIC
                % ============================================================
              case 'CS+FIC'
                u = gp.X_u;
                m = length(u);
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
                [Kv_ff, Cv_ff] = gp_trvar(gp, x);  % f x 1  vector
                K_fu = gp_cov(gp, x, u);         % f x u
                K_uu = gp_trcov(gp, u);    % u x u, noiseles covariance K_uu
                K_uu = (K_uu+K_uu')./2;     % ensure the symmetry of K_uu
                Luu = chol(K_uu)';

                % Evaluate the Lambda (La)
                % Q_ff = K_fu*inv(K_uu)*K_fu'
                B=Luu\(K_fu');       % u x f
                Qv_ff=sum(B.^2)';
                Lav = Cv_ff-Qv_ff;   % f x 1, Vector of diagonal elements
                
                gp.cf = cf2;
                K_cs = gp_trcov(gp,x);
                La = sparse(1:n,1:n,Lav,n,n) + K_cs;
                gp.cf = cf_orig;
                
                % Find fill reducing permutation and permute all the
                % matrices
                p = analyze(La);
                r(p) = 1:n;
                gp.likelih = feval(gp.likelih.fh_permute, gp.likelih, p);                
                f = f(p);
                y = y(p);
                La = La(p,p);
                K_fu = K_fu(p,:);
                B = B(:,p);
                VD = ldlchol(La);
                
                iLaKfu = ldlsolve(VD,K_fu);
                %iLaKfu = La\K_fu;

                A = K_uu+K_fu'*iLaKfu;  A = (A+A')./2;     % Ensure symmetry
                A = chol(A);
                L = iLaKfu/A;
                % Begin optimization
                switch gp.laplace_opt.optim_method
                    % find the mode by fminunc large scale method
                  case 'fminunc_large'
                    if ~isfield(gp.laplace_opt, 'fminunc_opt')
                        opt=optimset('GradObj','on');
                        opt=optimset(opt,'Hessian','on');
                        fhm = @(W, f, varargin) (ldlsolve(VD,f) - L*(L'*f)  + repmat(W,1,size(f,2)).*f);  % Hessian*f; % La\f
                        %fhm = @(W, f, ikf) (W{1}  + repmat(W{2},1,size(f,2)).*f);  % Hessian*f; % La\f
                        opt=optimset(opt,'HessMult', fhm);
                        opt=optimset(opt,'TolX', 1e-8);
                        opt=optimset(opt,'TolFun', 1e-8);
                        opt=optimset(opt,'LargeScale', 'on');
                        opt=optimset(opt,'Display', 'off');   % 'iter'
                    else
                        opt = gp.laplace_opt.fminunc_opt;
                    end

% $$$                     fe = @(f, varargin) (0.5*f*(ldlsolve(VD,f') - L*(L'*f')) - feval(gp.likelih.fh_e, gp.likelih, y, f'));
% $$$                     fg = @(f, varargin) (ldlsolve(VD,f') - L*(L'*f') - feval(gp.likelih.fh_g, gp.likelih, y, f', 'latent'))';
% $$$                     fh = @(f, varargin) (-feval(gp.likelih.fh_hessian, gp.likelih, y, f', 'latent'));
% $$$                     mydeal = @(varargin)varargin{1:nargout};
% $$$                     [f,fval,exitflag,output] = fminunc(@(ww) mydeal(fe(ww), fg(ww), fh(ww)), f', opt);
% $$$                     f = f';
                    [f,fval,exitflag,output] = fminunc(@(ww) egh(ww), f', opt);
                    f = f';
                    
                    a = ldlsolve(VD,f) - L*L'*f;
                  case 'newton'
                    tol = 1e-8;
                    a = f;
                    W = -feval(gp.likelih.fh_g2, gp.likelih, y, f, 'latent');
                    dlp = feval(gp.likelih.fh_g, gp.likelih, y, f, 'latent');
                    lp_new = feval(gp.likelih.fh_e, gp.likelih, y, f);
                    lp_old = -Inf;
                    I = sparse(1:n,1:n,1,n,n);
                    
                    while lp_new - lp_old > tol                        % begin Newton's iterations
                        lp_old = lp_new; a_old = a; 
                        sW = sqrt(W);
                        sqrtW = sparse(1:n,1:n,sW,n,n);
                        
                        Lah = I + sqrtW*La*sqrtW; 
                        VDh = ldlchol(Lah);
                        V = repmat(sW,1,m).*K_fu;
                        Vt = ldlsolve(VDh,V);
                        A = K_uu + V'*Vt;   A = (A+A')./2;
                        Lb = Vt/chol(A);
                        b = W.*f+dlp;
                        b2 = sW.*(La*b + B'*(B*b));
                        a = b - sW.*(ldlsolve(VDh,b2) - Lb*(Lb'*b2) );

                        f = La*a + B'*(B*a);
                        W = -feval(gp.likelih.fh_g2, gp.likelih, y, f, 'latent');
                        dlp = feval(gp.likelih.fh_g, gp.likelih, y, f, 'latent');
                        lp = feval(gp.likelih.fh_e, gp.likelih, y, f);
                        lp_new = -a'*f/2 + lp;
                        i = 0;
                        while i < 10 && lp_new < lp_old                       % if objective didn't increase
                            a = (a_old+a)/2;                                  % reduce step size by half
                            f = La*a + B'*(B*a);
                            W = -feval(gp.likelih.fh_g2, gp.likelih, y, f, 'latent');
                            lp = feval(gp.likelih.fh_e, gp.likelih, y, f);
                            lp_new = -a'*f/2 + lp;
                            i = i+1;
                        end
                    end                                                    % end Newton's iterations 
                end
                
                
                W = -feval(gp.likelih.fh_g2, gp.likelih, y, f, 'latent');
                sqrtW = sqrt(W);
                
                logZ = 0.5*f'*a - feval(gp.likelih.fh_e, gp.likelih, y, f);
                    
                WKfu = repmat(sqrtW,1,m).*K_fu;
                sqrtW = sparse(1:n,1:n,sqrtW,n,n);
                Lahat = sparse(1:n,1:n,1,n,n) + sqrtW*La*sqrtW;
                LDh = ldlchol(Lahat);
                A = K_uu + WKfu'*ldlsolve(LDh,WKfu);   A = (A+A')./2;
                A = chol(A);
                edata = sum(log(diag(LDh))) - 2*sum(log(diag(Luu))) + 2*sum(log(diag(A)));
                edata = logZ + 0.5*edata;
                
                La2 = La;
                
                % Reorder all the returned and stored values
                a = a(r);
                L = L(r,:);
                La2 = La2(r,r);
                y = y(r);
                f = f(r);
                W = W(r);
                gp.likelih = feval(gp.likelih.fh_permute, gp.likelih, r);
                
                % ============================================================
                % SSGP
                % ============================================================
              case 'SSGP'
                % First evaluate needed covariance matrices
                % v defines that parameter is a vector
                [Phi, S] = gp_trcov(gp, x);        % n x m matrix and nxn sparse matrix
                Sv = diag(S);
                
                m = size(Phi,2);
                
                A = eye(m,m) + Phi'*(S\Phi);
                A = chol(A)';
                L = (S\Phi)/A';
                
                switch gp.laplace_opt.optim_method
                    % find the mode by fminunc large scale method
                  case 'fminunc_large'
                    if ~isfield(gp.laplace_opt, 'fminunc_opt')
                        opt=optimset('GradObj','on');
                        opt=optimset(opt,'Hessian','on');
                        fhm = @(W, f, varargin) (f./repmat(Sv,1,size(f,2)) - L*(L'*f)  + repmat(W,1,size(f,2)).*f);  % Hessian*f; %
                        opt=optimset(opt,'HessMult', fhm);
                        opt=optimset(opt,'TolX', 1e-8);
                        opt=optimset(opt,'TolFun', 1e-8);
                        opt=optimset(opt,'LargeScale', 'on');
                        opt=optimset(opt,'Display', 'off');   % 'iter'
                    else
                        opt = gp.laplace_opt.fminunc_opt;
                    end

                    fe = @(f, varargin) (0.5*f*(f'./repmat(Sv,1,size(f',2)) - L*(L'*f')) - feval(gp.likelih.fh_e, gp.likelih, y, f'));
                    fg = @(f, varargin) (f'./repmat(Sv,1,size(f',2)) - L*(L'*f') - feval(gp.likelih.fh_g, gp.likelih, y, f', 'latent'))';
                    fh = @(f, varargin) (-feval(gp.likelih.fh_g2, gp.likelih, y, f', 'latent'));
                    mydeal = @(varargin)varargin{1:nargout};
                    [f,fval,exitflag,output] = fminunc(@(ww) mydeal(fe(ww), fg(ww), fh(ww)), f', opt);
                    f = f';

                    W = -feval(gp.likelih.fh_g2, gp.likelih, y, f, 'latent');
                    sqrtW = sqrt(W);

                    b = L'*f;
                    logZ = 0.5*(f'*(f./Sv) - b'*b) - feval(gp.likelih.fh_e, gp.likelih, y, f);
                  case 'Newton'
                    error('The Newton algorithm is not implemented for FIC!\n')
                end
                WPhi = repmat(sqrtW,1,m).*Phi;
                A = eye(m,m) + WPhi'./repmat((1+Sv.*W)',m,1)*WPhi;   A = (A+A')./2;
                A = chol(A);
                edata = sum(log(1+Sv.*W)) + 2*sum(log(diag(A)));
                edata = logZ + 0.5*edata;

                La2 = Sv;

              otherwise
                error('Unknown type of Gaussian process!')
            end

            % ======================================================================
            % Evaluate the prior contribution to the error from covariance functions
            % ======================================================================
            eprior = 0;
            for i=1:ncf
                gpcf = gp.cf{i};
                eprior = eprior + feval(gpcf.fh_e, gpcf, x, y);
            end

            % Evaluate the prior contribution to the error from noise functions
            if isfield(gp, 'noise')
                nn = length(gp.noise);
                for i=1:nn
                    noise = gp.noise{i};
                    eprior = eprior + feval(noise.fh_e, noise, x, y);
                end
            end
            
            % Evaluate the prior contribution to the error from likelihood function
            if isfield(gp, 'likelih') && isfield(gp.likelih, 'p')
                likelih = gp.likelih;
                eprior = eprior + feval(likelih.fh_priore, likelih);
            end

            e = edata + eprior;
            
            w0 = w;
            e0 = e;
            edata0 = edata;
            eprior0 = eprior;
            f0 = f;
            L0 = L;
            W0 = W;
            n0 = size(x,1);
            La20 = La2;
            a0 = a;
        end
        
        %
        % ==============================================================
        % Begin of the nested functions
        % ==============================================================
        %        
        function [e, g, h] = egh(f, varargin)
            ikf = iKf(f');
            e = 0.5*f*ikf - feval(gp.likelih.fh_e, gp.likelih, y, f');
            g = (ikf - feval(gp.likelih.fh_g, gp.likelih, y, f', 'latent'))';
            h = -feval(gp.likelih.fh_g2, gp.likelih, y, f', 'latent');
        end
        function ikf = iKf(f, varargin)
            
            switch gp.type
              case {'PIC' 'PIC_BLOCK'}
                iLaf = zeros(size(f));
                for i=1:length(ind)
                    iLaf(ind{i},:) = LLabl{i}\(LLabl{i}'\f(ind{i},:));
                end
                ikf = iLaf - L*(L'*f);
              case 'CS+FIC'
                ikf = ldlsolve(VD,f) - L*(L'*f);
            end
        end
    end
end