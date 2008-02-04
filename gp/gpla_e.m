function [e, edata, eprior, f, L, La2, b] = gpla_e(w, gp, x, y, param, varargin)
%GP2_E	Evaluate error function for Gaussian Process.
%
%	Description
%	E = GPEP_E(W, GP, P, T, PARAM) takes a gp data structure GP together
%	with a matrix P of input vectors and a matrix T of target vectors,
%	and evaluates the error function E.  Each row of P
%	corresponds to one input vector and each row of T corresponds to one
%	target vector.
%
%	[E, EDATA, EPRIOR] = GPEP_E(W, GP, P, T, PARAM) also returns the data and
%	prior components of the total error.
%
%	See also
%	
%

% Copyright (c) 2007      Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.
    
    if strcmp(w, 'init')
        w0 = rand(size(gp_pak(gp, param)));
        e0=[];
        edata0= inf; 
        eprior0=[];

        W = zeros(size(y));
        f0 = zeros(size(y));
        L0 = [];
        f = zeros(size(y));
        n0 = size(x,1);
        La20 = [];
        b0 = 0;
        
        % Create a table of constants that are needed in 
        % the function evaluations
        switch gp.likelih
          case 'probit'
            const_table = [];
          case 'poisson'
            const_table = gammaln(y+1);
            const_table(:,2) = y./gp.avgE;
            %            const_table(:,2) = 
        end
    
        laplace_algorithm(gp_pak(gp,param), gp, x, y, param, varargin);
        
        gp.fh_e = @laplace_algorithm;
        e = gp;
    else
        [e, edata, eprior, f, L, La2, b] = feval(gp.fh_e, w, gp, x, y, param, varargin);
    end
    
    function [e, edata, eprior, f, L, La2, b] = laplace_algorithm(w, gp, x, y, param, varargin)

        if abs(w-w0) < 1e-8 % 1e-8
            % The covariance function parameters haven't changed so just 
            % return the Energy and the site parameters that are saved
            e = e0;
            edata = edata0;
            eprior = eprior0;
            f = f0;
            L = L0;
            La2 = La20;
            b = b0;
        else
            
            gp=gp_unpak(gp, w, param);
            ncf = length(gp.cf);
            n = length(x);

            % laplace iteration parameters
            iter=1;
            maxiter = gp.laplace_opt.maxiter;
            tol = gp.laplace_opt.tol;
            logZ_tmp=0; logZ=Inf;
            f = f0; 
            
            % =================================================
            % First Evaluate the data contribution to the error    
            switch gp.type
                % ============================================================
                % FULL
                % ============================================================
              case 'FULL'   
                K = gp_trcov(gp, x);
                
                switch gp.laplace_opt.optim_method
                    % find the mode by fminunc large scale method
                  case 'fminunc_large'
                    
                    if ~isfield(gp.laplace_opt, 'fminunc_opt')
                        opt=optimset('GradObj','on');
                        opt=optimset(opt,'Hessian','on');
                        fhm = @(W, f, varargin) (K\f + repmat(W,1,size(f,2)).*f);  % W*f; %
                        opt=optimset(opt,'HessMult', fhm);
                        opt=optimset(opt,'TolX', 1e-8);
                        opt=optimset(opt,'TolFun', 1e-8);
                        opt=optimset(opt,'LargeScale', 'on');
                        opt=optimset(opt,'Display', 'off'); % 'iter'
                    else
                        opt = gp.laplace_opt.fminunc_opt;
                    end
                                        
                    fe = @(f, varargin) (0.5*f*(K\f') - loglikelihood(f', gp.likelih));
                    fg = @(f, varargin) (K\f' - derivative(f', gp.likelih))';
                    fh = @(f, varargin) (hessian(f', gp.likelih)); %inv(K) + diag(hessian(f', gp.likelih)) ; %
                    mydeal = @(varargin)varargin{1:nargout};
                    [f,fval,exitflag,output] = fminunc(@(ww) mydeal(fe(ww), fg(ww), fh(ww)), f', opt);
                    f = f';
                    
                    W = diag(hessian(f, gp.likelih));
                    sqrtW = sqrt(W);
                    B = eye(size(K)) + sqrtW*K*sqrtW;
                    L = chol(B)';
                    a = K\f;
                    logZ = 0.5 * f'*a - loglikelihood(f, gp.likelih);
                    
                    % find the mode by Scaled conjugate gradient method
                  case 'SCG'
                    if ~isfield(gp.laplace_opt, 'scg_opt')
                        opt(1) = 0;
                        opt(2) = 1e-8;
                        opt(3) = 3e-8;
                        opt(9) = 0;
                        opt(10) = 0;
                        opt(11) = 0;
                        opt(14) = 500;
                    else
                        opt = gp.laplace_opt.scg_opt;
                    end
                    
                    fe = @(f, varargin) (0.5*f*(K\f') - loglikelihood(f', gp.likelih));
                    fg = @(f, varargin) (K\f' - derivative(f', gp.likelih))';
                    [f, opt, flog]=scg(fe, f', opt, fg);
                    f = f';
                    
                    W = diag(hessian(f, gp.likelih));
                    sqrtW = sqrt(W);
                    B = eye(size(K)) + sqrtW*K*sqrtW;
                    L = chol(B)';
                    a = K\f;
                    logZ = 0.5 * f'*a - loglikelihood(f, gp.likelih);
                                        
                    % find the mode by Newton iteration
                  case 'Newton'
                    while iter<=maxiter & abs(logZ_tmp-logZ)>tol % logZ_tmp-logZ > tol   % 
                        logZ_tmp=logZ;
                        
                        % Evaluate the minus Hessian 
                        W = diag(hessian(f, gp.likelih));
                        sqrtW = sqrt(W);
                        B = eye(size(K)) + sqrtW*K*sqrtW;
                        L = chol(B)';
                        
                        % Evaluate the derivative with respect to f
                        der_f = derivative(f, gp.likelih);
                        b = W*f + der_f;
                        a = b - sqrtW*L'\(L\(sqrtW*(K*b)));
                        ft = K*a;
                        
                        % Evaluate the error criteria (=minus log marginal likelihood)
                        logZ = 0.5 * a'*ft - loglikelihood(ft, gp.likelih);
                        f = ft;
                        iter=iter+1;
                        fprintf('%.8f, iter: %d \n', logZ, iter-1)
                    end
                end
                edata = logZ + sum(log(diag(L))); % 0.5*log(det(eye(size(K)) + K*W)) ; %
                % Set something into La2
                La2 = W;
                b = derivative(f, gp.likelih);
                
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
                K_uu = (K_uu+K_uu')./2;     % ensure the symmetry of K_uu
                Luu = chol(K_uu)';
                % Evaluate the Lambda (La) 
                % Q_ff = K_fu*inv(K_uu)*K_fu'
                % Here we need only the diag(Q_ff), which is evaluated below
                B=Luu\(K_fu');       % u x f
                Qv_ff=sum(B.^2)';    
                Lav = Cv_ff-Qv_ff;   % f x 1, Vector of diagonal elements
                iLaKfu = zeros(size(K_fu));  % f x u, 
                for i=1:n
                    iLaKfu(i,:) = K_fu(i,:)./Lav(i);  % f x u 
                end
                A = K_uu+K_fu'*iLaKfu;  A = (A+A')./2;     % Ensure symmetry
                A = chol(A);
                L = iLaKfu/A;

                switch gp.laplace_opt.optim_method
                    % find the mode by fminunc large scale method
                  case 'fminunc_large'
                    if ~isfield(gp.laplace_opt, 'fminunc_opt')
                        opt=optimset('GradObj','on');
                        opt=optimset(opt,'Hessian','on');
                        fhm = @(W, f, varargin) (f./repmat(Lav,1,size(f,2)) - L*(L'*f)  + repmat(W,1,size(f,2)).*f);  % Hessian*f; %
                        opt=optimset(opt,'HessMult', fhm);
                        opt=optimset(opt,'TolX', 1e-8);
                        opt=optimset(opt,'TolFun', 1e-8);
                        opt=optimset(opt,'LargeScale', 'on');
                        opt=optimset(opt,'Display', 'off');   % 'iter'
                    else
                        opt = gp.laplace_opt.fminunc_opt;
                    end
                                        
                    fe = @(f, varargin) (0.5*f*(f'./repmat(Lav,1,size(f',2)) - L*(L'*f')) - loglikelihood(f', gp.likelih));
                    fg = @(f, varargin) (f'./repmat(Lav,1,size(f',2)) - L*(L'*f') - derivative(f', gp.likelih))';
                    fh = @(f, varargin) (hessian(f', gp.likelih)); 
                    mydeal = @(varargin)varargin{1:nargout};
                    [f,fval,exitflag,output] = fminunc(@(ww) mydeal(fe(ww), fg(ww), fh(ww)), f', opt);
                    f = f';
                    
                    W = hessian(f, gp.likelih);
                    sqrtW = sqrt(W);
% $$$                     B = eye(size(K)) + sqrtW*K*sqrtW;
% $$$                     L = chol(B)';
                    
                    b = L'*f;
                    logZ = 0.5*(f'*(f./Lav) - b'*b) - loglikelihood(f, gp.likelih);
                    
                    % find the mode by Scaled conjugate gradient method
                  case 'SCG'
                    error('The SCG algorithm is not implemented for FIC!\n') 
                    % find the mode by Newton iteration
                  case 'Newton'
                   error('The Newton algorithm is not implemented for FIC!\n') 
                end
                WKfu = repmat(sqrtW,1,m).*K_fu;
                A = K_uu + WKfu'./repmat((1+Lav.*W)',m,1)*WKfu;   A = (A+A')./2;
                A = chol(A);
                edata = sum(log(1+Lav.*W)) - 2*sum(log(diag(Luu))) + 2*sum(log(diag(A)));
                edata = logZ + 0.5*edata;
                
                La2 = Lav;
                b = derivative(f, gp.likelih);
                                
                % ============================================================
                % PIC
                % ============================================================
              case 'PIC_BLOCK'
                ind = gp.tr_index;
                u = gp.X_u;
                m = length(u);
                
                % First evaluate needed covariance matrices
                % v defines that parameter is a vector
                [Kv_ff, Cv_ff] = gp_trvar(gp, x);  % f x 1  vector
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
                    iLabl{i} = inv(Labl{i});
                    iLaKfu(ind{i},:) = Labl{i}\K_fu(ind{i},:);
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
                    
                    W = hessian(f, gp.likelih);
                    sqrtW = sqrt(W);
                    
                    ikf = iKf(f);
                    logZ = 0.5*f'*ikf - loglikelihood(f, gp.likelih);
                    
                    % find the mode by Scaled conjugate gradient method
                  case 'SCG'
                    if ~isfield(gp.laplace_opt, 'scg_opt')
                        opt(1) = 0;
                        opt(2) = 1e-6;
                        opt(3) = 3e-6;
                        opt(9) = 0;
                        opt(10) = 0;
                        opt(11) = 0;
                        opt(14) = 500;
                    else
                        opt = gp.laplace_opt.scg_opt;
                    end
                    
                    fe = @(f, varargin) (0.5*f*iKf(f') - loglikelihood(f', gp.likelih));
                    fg = @(f, varargin) (iKf(f') - derivative(f', gp.likelih))';
                    [f, opt, flog]=scg(fe, f', opt, fg);
                    f = f';
                    
                    W = hessian(f, gp.likelih);
                    sqrtW = sqrt(W);
                    
                    ikf = iKf(f);
                    logZ = 0.5*f'*ikf - loglikelihood(f, gp.likelih);
                    % find the mode by Quasi-Newton iteration
                  case 'quasiNewton'
                    if ~isfield(gp.laplace_opt, 'scg_opt')
                        opt(1) = 0;
                        opt(2) = 1e-5;
                        opt(3) = 3e-5;
                        opt(9) = 0;
                        opt(10) = 0;
                        opt(11) = 0;
                        opt(14) = 500;
                        opt(15) = 1e-4;
                        opt(18)=0;
                    else
                        opt = gp.laplace_opt.scg_opt;
                    end
                    
                    fe = @(f, varargin) (0.5*f*iKf(f') - loglikelihood(f', gp.likelih));
                    fg = @(f, varargin) (iKf(f') - derivative(f', gp.likelih))';
                    [f, opt, flog]=quasinew(fe, f', opt, fg);
                    f = f';
                    
                    W = hessian(f, gp.likelih);
                    sqrtW = sqrt(W);
                    
                    ikf = iKf(f);
                    logZ = 0.5*f'*ikf - loglikelihood(f, gp.likelih);

                    % find the mode by Newton iteration
                  case 'Newton'
                   error('The Newton algorithm is not implemented for PIC!\n') 
                end
                WKfu = repmat(sqrtW,1,m).*K_fu;
                edata = 0;
                for i=1:length(ind)
                    Lahat = eye(size(Labl{i})) + diag(sqrtW(ind{i}))*Labl{i}*diag(sqrtW(ind{i}));
                    iLahatWKfu(ind{i},:) = Lahat\WKfu(ind{i},:);
                    edata = edata + 2.*sum(log(diag(chol(Lahat)')));
                end
                A = K_uu + WKfu'*iLahatWKfu;   A = (A+A')./2;
                A = chol(A);
                edata =  edata - 2*sum(log(diag(Luu))) + 2*sum(log(diag(A)));
                edata = logZ + 0.5*edata;
                
                La2 = Labl;
                b = derivative(f, gp.likelih);
                
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
            
            % The last things to do
            if isfield(gp.laplace_opt, 'display') && gp.laplace_opt.display == 1
                fprintf('   Number of Newton iterations in Laplace: %d \n', iter-1)
            end
            
            e = edata + eprior;
            
            w0 = w;
            e0 = e;
            edata0 = edata;
            eprior0 = eprior;
            f0 = f;
            L0 = L;
            n0 = size(x,1);
            La20 = La2;
            b0 = b;
        end
%
% ==============================================================
% Begin of the nested functions
% ==============================================================
%
        function loglikelih = loglikelihood(f, likelihood)
            switch likelihood
              case 'probit'
                loglikelih = sum(log(normcdf(y.*f)));
              case 'poisson'
                lambda = gp.avgE.*exp(f);
                gamlny = const_table(:,1);
                loglikelih =  sum(-lambda + y.*log(lambda) - gamlny);
            end
        end
        function deriv = derivative(f, likelihood)
            switch likelihood
              case 'probit'
                deriv = y.*normpdf(f)./normcdf(y.*f);
              case 'poisson'
                deriv = y - gp.avgE.*exp(f);
            end
        end
        function Hessian = hessian(f, likelihood)
            switch likelihood
              case 'probit'
                z = y.*f;
                Hessian = (normpdf(f)./normcdf(z)).^2 + z.*normpdf(f)./normcdf(z);
              case 'poisson'
                Hessian = gp.avgE.*exp(f);
            end
        end
        function [e, g, h] = egh(f, varargin)
            ikf = iKf(f');
            e = 0.5*f*ikf - loglikelihood(f', gp.likelih);
            g = (ikf - derivative(f', gp.likelih))';
            h = hessian(f', gp.likelih);
        end        
        function ikf = iKf(f, varargin)
            iLaf = zeros(size(f));
            for i=1:length(ind)
                iLaf(ind{i},:) = iLabl{i}*f(ind{i},:);
            end
            ikf = iLaf - L*(L'*f);
        end        
    end
end
