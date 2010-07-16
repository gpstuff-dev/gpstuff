function [e, edata, eprior, site_tau, site_nu, L, La2, b] = gpep_e(w, gp, x, y, varargin)
%GPEP_E     Conduct Expectation propagation and return marginal 
%           log posterior estimate
%
%     Description
%	GP = GPEP_E('init', GP, X, Y, OPTIONS) takes a GP data structure
%        GP together with a matrix X of input vectors and a matrix Y
%        of target vectors, and initializes required fiels for the
%        EP algorithm.
%
%	E = GPEP_E(W, GP, X, Y, OPTIONS) takes a GP data structure GP
%        together with a matrix X of input vectors and a matrix Y of
%        target vectors, and finds the EP approximation for the
%        conditional posterior p(Y | X, th), where th is the
%        hyperparameters. Returns the energy at th (see below).  Each
%        row of X corresponds to one input vector and each row of Y
%        corresponds to one target vector.
%
%	[E, EDATA, EPRIOR] = GPEP_E(W, GP, X, Y, OPTIONS) returns also 
%        the data and prior components of the total energy.
%
%       The energy is minus log posterior cost function for th:
%            E = EDATA + EPRIOR 
%              = - log p(Y|X, th) - log p(th),
%       where th represents the hyperparameters (lengthScale, magnSigma2...), 
%       X is inputs and Y is observations.
%
%       OPTIONS is optional parameter-value pair
%        'z'    is optional observed quantity in triplet (x_i,y_i,z_i)
%               Some likelihoods may use this. For example, in case of 
%               Poisson likelihood we have z_i=E_i, that is, expected 
%               value for ith case. 
%
%	See also
%       GPEP_G, EP_PRED, GP_E

    
% Copyright (c) 2007           Jaakko Riihimäki
% Copyright (c) 2007-2010      Jarno Vanhatalo
% Copyright (c) 2010           Heikki Peura

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.
    
% Check the inputs
    
    ip=inputParser;
    ip.FunctionName = 'GPEP_E';
    ip.addRequired('w', @(x) ...
                   (ischar(x) && strcmp(w, 'init')) || ...
                   isvector(x) && isreal(x) && all(isfinite(x)));
    ip.addRequired('gp',@isstruct);
    ip.addRequired('x', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
    ip.addRequired('y', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
    ip.addParamValue('z', [], @(x) isreal(x) && all(isfinite(x(:))))
    ip.parse(w, gp, x, y, varargin{:});
    z=ip.Results.z;

    
    if strcmp(w, 'init')
        w0 = rand(size(gp_pak(gp)));
        e0=[];
        edata0= inf;
        eprior0=[];
        nutilde0 = zeros(size(y));
        tautilde0 = zeros(size(y));
        myy0 = zeros(size(y));
        L0 = [];
        myy=zeros(size(y));
        n0 = size(x,1);
        La20 = [];
        b0 = 0;
        
        ep_algorithm(gp_pak(gp), gp, x, y, z);

        gp.fh_e = @ep_algorithm;
        e = gp;
    else
        [e, edata, eprior, site_tau, site_nu, L, La2, b] = feval(gp.fh_e, w, gp, x, y, z);

    end

    function [e, edata, eprior, tautilde, nutilde, L, La2, b] = ep_algorithm(w, gp, x, y, z)

        if abs(w-w0) < 1e-8
            % The covariance function parameters haven't changed so just
            % return the Energy and the site parameters that are saved
            e = e0;
            edata = edata0;
            eprior = eprior0;
            nutilde = nutilde0;
            myy = myy0;
            tautilde = tautilde0;
            L = L0;
            La2 = La20;
            b = b0;
        else
            % Conduct evaluation for the energy and the site parameters
            gp=gp_unpak(gp, w);
            ncf = length(gp.cf);
            n = length(x);

            % ep iteration parameters
            iter=1;
            maxiter = gp.ep_opt.maxiter;
            tol = gp.ep_opt.tol;
            nutilde = zeros(size(y));
            tautilde = zeros(size(y));
            %tautilde = gp.likelih.sigma2^-1 *ones(size(y));
            myy = zeros(size(y));
            logZep_tmp=0; logZep=Inf;

            M0 = [];
            
            % =================================================
            % First Evaluate the data contribution to the error
            switch gp.type
                % ============================================================
                % FULL
                % ============================================================
              case 'FULL'   % A full GP
                [K,C] = gp_trcov(gp, x);

                % The EP algorithm for full support covariance function
                %------------------------------------------------------
                if ~issparse(C)
                    Sigm = C;
                    Ls = chol(Sigm);
                    Stildesqroot=zeros(n);
                    
                    % If Student-t likelihood is used sort
                    % the update order so that the problematic updates
                    % are left for last
                    if strcmp(gp.likelih.type,'Student-t')
                      f=feval(gp.likelih.fh_optimizef,gp,y,K);
                      W=-feval(gp.likelih.fh_g2,gp.likelih,y,f,'latent');
                      [foo,I]=sort(W,'descend');
                    else
                      I=1:n;
                    end
                    
                    % The EP -algorithm
                    while iter<=maxiter && abs(logZep_tmp-logZep)>tol
                        
                        logZep_tmp=logZep;
                        muvec_i = zeros(n,1); sigm2vec_i = zeros(n,1);
                        
                        for ii=1:n
                            i1 = I(ii);
                            % Algorithm utilizing Cholesky updates
                            % This is numerically more stable but slower
% $$$                             % approximate cavity parameters
% $$$                             S11 = sum(Ls(:,i1).^2);
% $$$                             S1 = Ls'*Ls(:,i1);
% $$$                             tau_i=S11^-1-tautilde(i1);
% $$$                             vee_i=S11^-1*myy(i1)-nutilde(i1);
% $$$                             
% $$$                             myy_i=vee_i/tau_i;
% $$$                             sigm2_i=tau_i^-1;
% $$$                             
% $$$                             if sigm2_i < 0
% $$$                                 [ii i1]
% $$$                             end
% $$$                             
% $$$                             % marginal moments
% $$$                             [M0(i1), muhati, sigm2hati] = feval(gp.likelih.fh_tiltedMoments, gp.likelih, y, i1, sigm2_i, myy_i, z);
% $$$                             
% $$$                             % update site parameters
% $$$                             deltatautilde = sigm2hati^-1-tau_i-tautilde(i1);
% $$$                             tautilde(i1) = tautilde(i1)+deltatautilde;
% $$$                             nutilde(i1) = sigm2hati^-1*muhati-vee_i;
% $$$                             
% $$$                             upfact = 1./(deltatautilde^-1+S11);
% $$$                             if upfact > 0
% $$$                                 Ls = cholupdate(Ls, S1.*sqrt(upfact), '-');
% $$$                             else
% $$$                                 Ls = cholupdate(Ls, S1.*sqrt(-upfact));
% $$$                             end
% $$$                             Sigm = Ls'*Ls;
% $$$                             myy=Sigm*nutilde;
% $$$                             
% $$$                             muvec_i(i1,1)=myy_i;
% $$$                             sigm2vec_i(i1,1)=sigm2_i;

                            % Algorithm as in Rasmussen and Williams 2006
                            % approximate cavity parameters
                            tau_i=Sigm(i1,i1)^-1-tautilde(i1);
                            vee_i=Sigm(i1,i1)^-1*myy(i1)-nutilde(i1);
                            
                            if tau_i < 0
                                tautilde(find(tautilde<0)) = 0;
                               
                                Stilde=tautilde;
                                Stildesqroot=diag(sqrt(tautilde));
                                B=eye(n)+Stildesqroot*C*Stildesqroot;
                                L=chol(B,'lower');
                                V=(L\Stildesqroot)*C;
                                Sigm=C-V'*V;                                 
                                nutilde=Sigm\myy;
                                                               
                                tau_i=Sigm(i1,i1)^-1-tautilde(i1);
                                vee_i=Sigm(i1,i1)^-1*myy(i1)-nutilde(i1);
                            
                                disp(sprintf('negative cavity at site %d', i1))
                            end
                            myy_i=vee_i/tau_i;
                            sigm2_i=tau_i^-1;
                            
                            % marginal moments
                            [M0(i1), muhati, sigm2hati] = feval(gp.likelih.fh_tiltedMoments, gp.likelih, y, i1, sigm2_i, myy_i, z);
                            
                            % update site parameters
                            deltatautilde=sigm2hati^-1-tau_i-tautilde(i1);
                            tautilde(i1)=tautilde(i1)+deltatautilde;
                            nutilde(i1)=sigm2hati^-1*muhati-vee_i;
                            
                            apu = deltatautilde/(1+deltatautilde*Sigm(i1,i1));
                            Sigm = Sigm - apu*(Sigm(:,i1)*Sigm(:,i1)');
                            
                            % The below is how Rasmussen and Williams
                            % (2006) do the update. The above version is
                            % more robust.
                            %apu = deltatautilde^-1+Sigm(i1,i1);
                            %apu = (Sigm(:,i1)/apu)*Sigm(:,i1)';
                            %Sigm = Sigm - apu;
                            %Sigm=Sigm-(deltatautilde^-1+Sigm(i1,i1))^-1*(Sigm(:,i1)*Sigm(:,i1)');
                            myy=Sigm*nutilde;
                            
                            muvec_i(i1,1)=myy_i;
                            sigm2vec_i(i1,1)=sigm2_i;
                        end
                        % Recompute the approximate posterior parameters
                        if tautilde > 0             % This is the usual case where likelihood is log concave
                                                    % for example, Poisson and probit 
                            
                            Stilde=tautilde;
                            Stildesqroot=diag(sqrt(tautilde));
                            
                            % NOTICE! upper triangle matrix! cf. to                    
                            % line 13 in the algorithm 3.5, p. 58.
                            
                            B=eye(n)+Stildesqroot*C*Stildesqroot;
                            L=chol(B,'lower');
                            
                            V=(L\Stildesqroot)*C;
                            Sigm=C-V'*V; myy=Sigm*nutilde;
                            Ls = chol(Sigm);
                            
                            % Compute the marginal likelihood
                            % Direct formula (3.65):
                            % Sigmtilde=diag(1./tautilde);
                            % mutilde=inv(Stilde)*nutilde;
                            %
                            % logZep=-0.5*log(det(Sigmtilde+K))-0.5*mutilde'*inv(K+Sigmtilde)*mutilde+
                            %         sum(log(normcdf(y.*muvec_i./sqrt(1+sigm2vec_i))))+
                            %         0.5*sum(log(sigm2vec_i+1./tautilde))+
                            %         sum((muvec_i-mutilde).^2./(2*(sigm2vec_i+1./tautilde)))
                            
                            % 4. term & 1. term
                            term41=0.5*sum(log(1+tautilde.*sigm2vec_i))-sum(log(diag(L)));
                            
                            % 5. term (1/2 element) & 2. term
                            T=1./sigm2vec_i;
                            Cnutilde = C*nutilde;
                            L2 = V*nutilde;
                            term52 = nutilde'*Cnutilde - L2'*L2 - (nutilde'./(T+Stilde)')*nutilde;
                            term52 = term52.*0.5;
                            
                            % 5. term (2/2 element)
                            term5=0.5*muvec_i'.*(T./(Stilde+T))'*(Stilde.*muvec_i-2*nutilde);
                            
                            % 3. term
                            term3 = sum(log(M0));
                            
                            logZep = -(term41+term52+term5+term3);
                            iter=iter+1;
                        else                         % We might end up here if the likelihood is not log concace
                                                     % For example Student-t likelihood. 
                                                     % NOTE! This does not work reliably yet
                            Stilde=tautilde;
                            Ls = chol(Sigm);
                            myy=Sigm*nutilde;
                            
                            % Compute the marginal likelihood
                            % 4. term & 1. term
                            term41 = 0.5*sum(log(1+tautilde.*sigm2vec_i)) - sum(log(diag(chol(C)))) + sum(log(diag(Ls)));
                            
                            % 5. term (1/2 element) & 2. term
                            T=1./sigm2vec_i;
                            term52 = nutilde'*(Ls'*(Ls*nutilde)) - (nutilde'./(T+Stilde)')*nutilde;
                            term52 = term52.*0.5;
                            
                            % 5. term (2/2 element)
                            term5=0.5*muvec_i'.*(T./(Stilde+T))'*(Stilde.*muvec_i-2*nutilde);
                            
                            % 3. term
                            term3 = sum(log(M0));
                            
                            logZep = -(term41+term52+term5+term3);
                            iter=iter+1;
                            B=Ls;
                            L=Ls;                            
                        end
                    end
                % EP algorithm for compactly supported covariance function (that is C is sparse)
                %---------------------------------------------------------------------------
                else
                    p = analyze(K);
                    r(p) = 1:n;
                    if ~isempty(z)
                        z = z(p,:);
                    end
                    y = y(p);
                    K = K(p,p);
                    
                    Inn = sparse(1:n,1:n,1,n,n);
                    sqrtS = sparse(1:n,1:n,0,n,n);
                    myy = zeros(size(y));
                    sigm2 = zeros(size(y));
                    gamma = zeros(size(y));
                    VD = ldlchol(Inn);
                    
                    
                    % The EP -algorithm
                    while iter<=maxiter && abs(logZep_tmp-logZep)>tol
                        
                        logZep_tmp=logZep;
                        muvec_i = zeros(n,1); sigm2vec_i = zeros(n,1);
                        for i1=1:n
                            % approximate cavity parameters
                            Ki1 = K(:,i1);
                            sqrtSKi1 = ssmult(sqrtS, Ki1);
                            tttt = ldlsolve(VD,sqrtSKi1);
                            sigm2(i1) = Ki1(i1) - sqrtSKi1'*tttt;
                            myy(i1) = gamma(i1) - tttt'*sqrtS*gamma;
                            
                            tau_i=sigm2(i1)^-1-tautilde(i1);
                            vee_i=sigm2(i1)^-1*myy(i1)-nutilde(i1);
                            
                            myy_i=vee_i/tau_i;
                            sigm2_i=tau_i^-1;
                            
                            % marginal moments
                            [M0(i1), muhati, sigm2hati] = feval(gp.likelih.fh_tiltedMoments, gp.likelih, y, i1, sigm2_i, myy_i, z);
                            
                            % update site parameters
                            tautilde_old = tautilde(i1);
                            deltatautilde=sigm2hati^-1-tau_i-tautilde(i1);
                            tautilde(i1)=tautilde(i1)+deltatautilde;
                            nutilde_old = nutilde(i1);
                            nutilde(i1)=sigm2hati^-1*muhati-vee_i;
                            deltanutilde = nutilde(i1) - nutilde_old;
                            
                            % Update the LDL decomposition
                            sqrtS(i1,i1) = sqrt(tautilde(i1));
                            sqrtSKi1(i1) = sqrt(tautilde(i1)).*Ki1(i1);
                            D2_n = sqrtSKi1.*sqrtS(i1,i1) + Inn(:,i1);
                                                        
                            if tautilde_old == 0
                                VD = ldlrowupdate(i1,VD,VD(:,i1),'-');
                                VD = ldlrowupdate(i1,VD,D2_n,'+');
                            else
                                VD = ldlrowmodify(VD, D2_n, i1);
                            end                                                        
                            gamma = gamma + Ki1.*deltanutilde;
                            
                            muvec_i(i1,1)=myy_i;
                            sigm2vec_i(i1,1)=sigm2_i;
                        end
                        % Recompute the approximate posterior parameters
                        sqrtS = sparse(1:n,1:n,sqrt(tautilde),n,n);
                        KsqrtS = ssmult(K,sqrtS);
                        B = ssmult(sqrtS,KsqrtS) + Inn;
                        VD = ldlchol(B);
                        Knutilde = K*nutilde;
                        myy = Knutilde - KsqrtS*ldlsolve(VD,sqrtS*Knutilde);
                        
                        % Compute the marginal likelihood                        
                        % 4. term & 1. term
                        term41=0.5*sum(log(1+tautilde.*sigm2vec_i)) - 0.5.*sum(log(diag(VD)));
                        
                        % 5. term (1/2 element) & 2. term
                        T=1./sigm2vec_i;
                        term52 = nutilde'*myy - (nutilde'./(T+tautilde)')*nutilde;
                        term52 = term52.*0.5;
                        
                        % 5. term (2/2 element)
                        term5=0.5*muvec_i'.*(T./(tautilde+T))'*(tautilde.*muvec_i-2*nutilde);
                        
                        % 3. term
                        term3 = sum(log(M0));
                        
                        logZep = -(term41+term52+term5+term3);
                        
                        iter=iter+1;
                    end
                    % Reorder all the returned and stored values
                    B = B(r,r);
                    nutilde = nutilde(r);
                    tautilde = tautilde(r);
                    myy = myy(r);
                    y = y(r);
                    if ~isempty(z)
                        z = z(r,:);
                    end
                    L = ldlchol(B);
                end
                edata = logZep;
                % Set something into La2
                La2 = B;
                b = 0;
                
              % ============================================================
              % FIC
              % ============================================================
              case 'FIC'
                u = gp.X_u;
                m = size(u,1);

                % First evaluate needed covariance matrices
                % v defines that parameter is a vector
                [Kv_ff, Cv_ff] = gp_trvar(gp, x);  % f x 1  vector
                K_fu = gp_cov(gp, x, u);           % f x u
                K_uu = gp_trcov(gp, u);     % u x u, noiseles covariance K_uu
                K_uu = (K_uu+K_uu')./2;     % ensure the symmetry of K_uu
                Luu = chol(K_uu)';
                % Evaluate the Lambda (La)
                % Q_ff = K_fu*inv(K_uu)*K_fu'
                % Here we need only the diag(Q_ff), which is evaluated below
                B=Luu\(K_fu');       % u x f
                Qv_ff=sum(B.^2)';
                Lav = Cv_ff-Qv_ff;   % f x 1, Vector of diagonal elements
                                     % iLaKfu = diag(iLav)*K_fu = inv(La)*K_fu
                                     % First some helper parameters
                iLaKfu = zeros(size(K_fu));  % f x u,
                for i=1:n
                    iLaKfu(i,:) = K_fu(i,:)./Lav(i);  % f x u
                end
                A = K_uu+K_fu'*iLaKfu;  A = (A+A')./2;     % Ensure symmetry
                A = chol(A);
                L = iLaKfu/A;
                Lahat = 1./Lav;
                I = eye(size(K_uu));

                R0 = chol(inv(K_uu));
                R = R0;
                P = K_fu;
                myy = zeros(size(y));
                eta = zeros(size(y));
                gamma = zeros(size(K_uu,1),1);
                D_vec = Lav;
                Ann=0;

                while iter<=maxiter && abs(logZep_tmp-logZep)>tol

                    logZep_tmp=logZep;
                    muvec_i = zeros(n,1); sigm2vec_i = zeros(n,1);
                    for i1=1:n
                        % approximate cavity parameters
                        pn = P(i1,:)';
                        Ann = D_vec(i1) + sum((R*pn).^2);
                        tau_i = Ann^-1-tautilde(i1);
                        myy(i1) = eta(i1) + pn'*gamma;
                        vee_i = Ann^-1*myy(i1)-nutilde(i1);

                        myy_i=vee_i/tau_i;
                        sigm2_i=tau_i^-1;

                        % marginal moments
                        [M0(i1), muhati, sigm2hati] = feval(gp.likelih.fh_tiltedMoments, gp.likelih, y, i1, sigm2_i, myy_i, z);
                        
                        % update site parameters
                        deltatautilde = sigm2hati^-1-tau_i-tautilde(i1);
                        tautilde(i1) = tautilde(i1)+deltatautilde;
                        deltanutilde = sigm2hati^-1*muhati-vee_i - nutilde(i1);
                        nutilde(i1) = sigm2hati^-1*muhati-vee_i;

                        % Update the parameters
                        dn = D_vec(i1);
                        D_vec(i1) = D_vec(i1) - deltatautilde.*D_vec(i1).^2 ./ (1+deltatautilde.*D_vec(i1));
                        P(i1,:) = pn' - (deltatautilde.*dn ./ (1+deltatautilde.*dn)).*pn';
                        updfact = deltatautilde./(1 + deltatautilde.*Ann);
                        if updfact > 0
                            RtRpnU = R'*(R*pn).*sqrt(updfact);
                            R = cholupdate(R, RtRpnU, '-');
                        elseif updfact < 0
                            RtRpnU = R'*(R*pn).*sqrt(abs(updfact));
                            R = cholupdate(R, RtRpnU, '+');
                        end
                        eta(i1) = eta(i1) + (deltanutilde - deltatautilde.*eta(i1)).*dn./(1+deltatautilde.*dn);
                        gamma = gamma + (deltanutilde - deltatautilde.*myy(i1))./(1+deltatautilde.*dn) * R'*(R*pn);
                        %                            myy = eta + P*gamma;

                        % Store cavity parameters
                        muvec_i(i1,1)=myy_i;
                        sigm2vec_i(i1,1)=sigm2_i;
                    end

                    % Re-evaluate the parameters
                    temp1 = (1+Lav.*tautilde).^(-1);
                    D_vec = temp1.*Lav;
                    R0P0t = R0*K_fu';
                    temp2 = zeros(size(R0P0t));
                    for i2 = 1:length(temp1)
                        P(i2,:) = temp1(i2).*K_fu(i2,:);
                        temp2(:,i2) = R0P0t(:,i2).*tautilde(i2).*temp1(i2);
                    end
                    R = chol(inv(eye(size(R0)) + temp2*R0P0t')) * R0;
                    eta = D_vec.*nutilde;
                    gamma = R'*(R*(P'*nutilde));
                    myy = eta + P*gamma;

                    % Compute the marginal likelihood, see FULL model for
                    % details about equations
                    Lahat = 1./Lav + tautilde;
                    Lhat = bsxfun(@rdivide,L,Lahat);
                    H = I-L'*Lhat;
                    B = H\L';
                    Bhat = B./repmat(Lahat',m,1);

                    % 4. term & 1. term
                    Stildesqroot=sqrt(tautilde);
                    D = Stildesqroot.*Lav.*Stildesqroot + 1;
                    SsqrtKfu = K_fu.*repmat(Stildesqroot,1,m);
                    AA = K_uu + (SsqrtKfu'./repmat(D',m,1))*SsqrtKfu; AA = (AA+AA')/2;
                    AA = chol(AA,'lower');
                    term41 = - 0.5*sum(log(1+tautilde.*sigm2vec_i)) - sum(log(diag(Luu))) + sum(log(diag(AA))) + 0.5.*sum(log(D));

                    % 5. term (1/2 element) & 2. term
                    T=1./sigm2vec_i;
                    term52 = -0.5*( (nutilde./Lahat)'*nutilde + (nutilde'*Lhat)*(Bhat*nutilde) - (nutilde./(T+tautilde))'*nutilde);

                    % 5. term (2/2 element)
                    term5 = - 0.5*muvec_i'.*(T./(tautilde+T))'*(tautilde.*muvec_i-2*nutilde);

                    % 3. term
                    term3 = -sum(log(M0));
                    
                    logZep = term41+term52+term5+term3;

                    iter=iter+1;
                end
                edata = logZep;
                %L = iLaKfu;
                
                % b'  = (La + Kfu*iKuu*Kuf + 1./S)*1./S * nutilde
                %     = (S - S * (iLa - L*L' + S)^(-1) * S) * 1./S
                %     = I - S * (Lahat - L*L')^(-1)
                % L   = S*Kfu * (Lav + 1./S)^(-1) / chol(K_uu + SsqrtKfu'*(Lav + 1./S)^(-1)*SsqrtKfu) 
                % La2 = D./S = Lav + 1./S, 
                %
                % The way evaluations are done is numerically more stable
                % See equations (3.71) and (3.72) in Rasmussen and Williams (2006)
                b = nutilde'.*(1 - Stildesqroot./Lahat.*Stildesqroot)' - (nutilde'*Lhat)*Bhat.*tautilde';    % part of eq. (3.71)
                L = ((repmat(Stildesqroot,1,m).*SsqrtKfu)./repmat(D',m,1)')/AA';                             % part of eq. (3.72)
                La2 = 1./(Stildesqroot./D.*Stildesqroot);                                                    % part of eq. (3.72)
                D = D_vec;
                
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
                    Llabl = chol(Labl{i});
                    iLaKfu(ind{i},:) = Llabl\(Llabl'\K_fu(ind{i},:));
                end
                A = K_uu+K_fu'*iLaKfu;
                A = (A+A')./2;     % Ensure symmetry
                A = chol(A);
                L = iLaKfu/A;
                I = eye(size(K_uu));

                R0 = chol(inv(K_uu));
                R = R0;
                P = K_fu;
                R0P0t = R0*K_fu';
                myy = zeros(size(y));
                eta = zeros(size(y));
                gamma = zeros(size(K_uu,1),1);
                D = Labl;
                Ann=0;

                while iter<=maxiter && abs(logZep_tmp-logZep)>tol

                    logZep_tmp=logZep;
                    muvec_i = zeros(n,1); sigm2vec_i = zeros(n,1);
                    for bl=1:length(ind)
                        bl_ind = ind{bl};
                        for in=1:length(bl_ind)
                            i1 = bl_ind(in);
                            % approximate cavity parameters
                            Dbl = D{bl}; dn = Dbl(in,in); pn = P(i1,:)';
                            Ann = dn + sum((R*pn).^2);
                            tau_i = Ann^-1-tautilde(i1);
                            myy(i1) = eta(i1) + pn'*gamma;
                            vee_i = Ann^-1*myy(i1)-nutilde(i1);

                            myy_i=vee_i/tau_i;
                            sigm2_i=tau_i^-1;
                            
                            % marginal moments
                            [M0(i1), muhati, sigm2hati] = feval(gp.likelih.fh_tiltedMoments, gp.likelih, y, i1, sigm2_i, myy_i, z);

                            % update site parameters
                            deltatautilde = sigm2hati^-1-tau_i-tautilde(i1);
                            tautilde(i1) = tautilde(i1)+deltatautilde;
                            deltanutilde = sigm2hati^-1*muhati-vee_i - nutilde(i1);
                            nutilde(i1) = sigm2hati^-1*muhati-vee_i;

                            % Update the parameters
                            Dblin = Dbl(:,in);
                            Dbl = Dbl - deltatautilde ./ (1+deltatautilde.*dn) * Dblin*Dblin';
                            %Dbl = inv(inv(Dbl) + diag(tautilde(bl_ind)));
                            P(bl_ind,:) = P(bl_ind,:) - ((deltatautilde ./ (1+deltatautilde.*dn)).* Dblin)*pn';
                            updfact = deltatautilde./(1 + deltatautilde.*Ann);
                            if updfact > 0
                                RtRpnU = R'*(R*pn).*sqrt(updfact);
                                R = cholupdate(R, RtRpnU, '-');
                            elseif updfact < 0
                                RtRpnU = R'*(R*pn).*sqrt(abs(updfact));
                                R = cholupdate(R, RtRpnU, '+');
                            end
                            eta(bl_ind) = eta(bl_ind) + (deltanutilde - deltatautilde.*eta(i1))./(1+deltatautilde.*dn).*Dblin;
                            gamma = gamma + (deltanutilde - deltatautilde.*myy(i1))./(1+deltatautilde.*dn) * (R'*(R*pn));
                            %myy = eta + P*gamma;

                            D{bl} = Dbl;
                            % Store cavity parameters
                            muvec_i(i1,1)=myy_i;
                            sigm2vec_i(i1,1)=sigm2_i;                                
                        end
                    end
                    % Re-evaluate the parameters
                    temp2 = zeros(size(R0P0t));
                    
                    Stildesqroot=sqrt(tautilde);
                    for i=1:length(ind)
                        sdtautilde = diag(Stildesqroot(ind{i}));
                        Dhat = sdtautilde*Labl{i}*sdtautilde + eye(size(Labl{i}));
                        Ldhat{i} = chol(Dhat);
                        D{i} = Labl{i} - Labl{i}*sdtautilde*(Ldhat{i}\(Ldhat{i}'\sdtautilde*Labl{i}));
                        P(ind{i},:) = D{i}*(Labl{i}\K_fu(ind{i},:));
                        
                        temp2(:,ind{i}) = R0P0t(:,ind{i})*sdtautilde/Dhat*sdtautilde;
                        eta(ind{i}) = D{i}*nutilde(ind{i});
                    end
                    R = chol(inv(eye(size(R0)) + temp2*R0P0t')) * R0;
                    gamma = R'*(R*(P'*nutilde));
                    myy = eta + P*gamma;

                    % Compute the marginal likelihood, see FULL model for
                    % details about equations
                    %
                    % First some helper parameters
                    for i = 1:length(ind)
                        Lhat(ind{i},:) = D{i}*L(ind{i},:);
                    end
                    H = I-L'*Lhat;
                    B = H\L';

                    % Compute the marginal likelihood, see FULL model for
                    % details about equations
                    term41 = 0; term52 = 0;
                    for i=1:length(ind)
                        Bhat(:,ind{i}) = B(:,ind{i})*D{i};
                        SsqrtKfu(ind{i},:) = bsxfun(@times,K_fu(ind{i},:),Stildesqroot(ind{i}));
                        %SsqrtKfu(ind{i},:) = gtimes(K_fu(ind{i},:),Stildesqroot(ind{i}));
                        iDSsqrtKfu(ind{i},:) = Ldhat{i}\(Ldhat{i}'\SsqrtKfu(ind{i},:));
                        term41 = term41 + sum(log(diag(Ldhat{i})));
                        term52 = term52 + nutilde(ind{i})'*(D{i}*nutilde(ind{i}));
                        
                    end
                    AA = K_uu + SsqrtKfu'*iDSsqrtKfu; AA = (AA+AA')/2;
                    AA = chol(AA,'lower');
                    term41 = term41 - 0.5*sum(log(1+tautilde.*sigm2vec_i)) - sum(log(diag(Luu))) + sum(log(diag(AA)));

                    % 5. term (1/2 element) & 2. term
                    T=1./sigm2vec_i;
                    term52 = -0.5*( term52 + (nutilde'*Lhat)*(Bhat*nutilde) - (nutilde./(T+tautilde))'*nutilde);

                    % 5. term (2/2 element)
                    term5 = - 0.5*muvec_i'.*(T./(tautilde+T))'*(tautilde.*muvec_i-2*nutilde);

                    % 3. term
                    term3 = -sum(log(M0));
                    
                    logZep = term41+term52+term5+term3;

                    iter=iter+1;
                end
                edata = logZep;
                
                b = zeros(1,n);
                for i=1:length(ind)
                    b(ind{i}) = nutilde(ind{i})'*D{i};
                    La2{i} = inv(diag(Stildesqroot(ind{i}))*(Ldhat{i}\(Ldhat{i}'\diag(Stildesqroot(ind{i})))));
                end
                b = nutilde' - ((b + (nutilde'*Lhat)*Bhat).*tautilde');

                L = (repmat(Stildesqroot,1,m).*iDSsqrtKfu)/AA';

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
                K_fu = gp_cov(gp, x, u);           % f x u
                K_uu = gp_trcov(gp, u);            % u x u, noiseles covariance K_uu
                K_uu = (K_uu+K_uu')./2;            % ensure the symmetry of K_uu
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
                
                % clear unnecessary variables
                clear K_cs; clear Qv_ff; clear Kv_ff; clear Cv_ff; clear Lav;
                
                % Find fill reducing permutation and permute all the
                % matrices
                p = analyze(La);
                r(p) = 1:n;
                if ~isempty(z)
                    z = z(p,:);
                end
                y = y(p);
                La = La(p,p);
                K_fu = K_fu(p,:);
                
                VD = ldlchol(La);
                iLaKfu = ldlsolve(VD,K_fu);
                A = K_uu+K_fu'*iLaKfu; A = (A+A')./2;     % Ensure symmetry
                L = iLaKfu/chol(A);
                
                I = eye(size(K_uu));

                Inn = sparse(1:n,1:n,1,n,n);
                sqrtS = sparse(1:n,1:n,0,n,n);
                R0 = chol(inv(K_uu));
                R = R0;
                P = K_fu;
                R0P0t = R0*K_fu';
                myy = zeros(size(y));
                eta = zeros(size(y));
                gamma = zeros(size(K_uu,1),1);
                Ann=0;
                LasqrtS = La*sqrtS;                
                VD = ldlchol(Inn);
                while iter<=maxiter && abs(logZep_tmp-logZep)>tol

                    logZep_tmp=logZep;
                    muvec_i = zeros(n,1); sigm2vec_i = zeros(n,1);
                    for i1=1:n
                        % approximate cavity parameters
                        tttt = ldlsolve(VD,ssmult(sqrtS,La(:,i1)));
                        Di1 =  La(:,i1) - ssmult(LasqrtS,tttt);
                                                
                        dn = Di1(i1);
                        pn = P(i1,:)';
                        Ann = dn + sum((R*pn).^2);
                        tau_i = Ann^-1-tautilde(i1);
                        myy(i1) = eta(i1) + pn'*gamma;
                        vee_i = Ann^-1*myy(i1)-nutilde(i1);

                        myy_i=vee_i/tau_i;
                        sigm2_i= tau_i^-1;  % 1./tau_i;  % 

                        % marginal moments
                        [M0(i1), muhati, sigm2hati] = feval(gp.likelih.fh_tiltedMoments, gp.likelih, y, i1, sigm2_i, myy_i, z);

                        % update site parameters
                        deltatautilde = sigm2hati^-1-tau_i-tautilde(i1);
                        tautilde(i1) = tautilde(i1)+deltatautilde;
                        deltanutilde = sigm2hati^-1*muhati-vee_i - nutilde(i1);
                        nutilde(i1) = sigm2hati^-1*muhati-vee_i;
                        
                        % Update the parameters
                        P = P - ((deltatautilde ./ (1+deltatautilde.*dn)).* Di1)*pn';
                        updfact = deltatautilde./(1 + deltatautilde.*Ann);
                        if updfact > 0
                            RtRpnU = R'*(R*pn).*sqrt(updfact);
                            R = cholupdate(R, RtRpnU, '-');
                        elseif updfact < 0
                            RtRpnU = R'*(R*pn).*sqrt(abs(updfact));
                            R = cholupdate(R, RtRpnU, '+');
                        end
                        eta = eta + (deltanutilde - deltatautilde.*eta(i1))./(1+deltatautilde.*dn).*Di1;
                        gamma = gamma + (deltanutilde - deltatautilde.*myy(i1))./(1+deltatautilde.*dn) * (R'*(R*pn));

                        % Store cavity parameters
                        muvec_i(i1,1)=myy_i;
                        sigm2vec_i(i1,1)=sigm2_i;
                                                
                        D2_o = ssmult(sqrtS,LasqrtS(:,i1)) + Inn(:,i1);
                        sqrtS(i1,i1) = sqrt(tautilde(i1));
                        LasqrtS(:,i1) = La(:,i1).*sqrtS(i1,i1);
                        D2_n = ssmult(sqrtS,LasqrtS(:,i1)) + Inn(:,i1);
                        
                        if tautilde(i1) - deltatautilde == 0
                            VD = ldlrowupdate(i1,VD,VD(:,i1),'-');
                            VD = ldlrowupdate(i1,VD,D2_n,'+');
                        else
                            VD = ldlrowmodify(VD, D2_n, i1);
                        end
                    end
                    % Re-evaluate the parameters
                    sqrtS = sparse(1:n,1:n,sqrt(tautilde),n,n);
                    sqrtSLa = ssmult(sqrtS,La);
                    D2 = ssmult(sqrtSLa,sqrtS) + Inn;
                    LasqrtS = ssmult(La,sqrtS);
                    VD = ldlchol(D2);

                    SsqrtKfu = sqrtS*K_fu;
                    iDSsqrtKfu = ldlsolve(VD,SsqrtKfu);
                    P = K_fu - sqrtSLa'*iDSsqrtKfu;
                    R = chol(inv( eye(size(R0)) + R0P0t*sqrtS*ldlsolve(VD,sqrtS*R0P0t'))) * R0;
                    eta = La*nutilde - sqrtSLa'*ldlsolve(VD,sqrtSLa*nutilde);
                    gamma = R'*(R*(P'*nutilde));
                    myy = eta + P*gamma;
                    
                    % Compute the marginal likelihood,
                    Lhat = La*L - sqrtSLa'*ldlsolve(VD,sqrtSLa*L);                    
                    H = I-L'*Lhat;
                    B = H\L';
                    Bhat = B*La - ldlsolve(VD,sqrtSLa*B')'*sqrtSLa;

                    % 4. term & 1. term                    
                    AA = K_uu + SsqrtKfu'*iDSsqrtKfu; AA = (AA+AA')/2;
                    AA = chol(AA,'lower');
                    term41 = - 0.5*sum(log(1+tautilde.*sigm2vec_i)) - sum(log(diag(Luu))) + sum(log(diag(AA))) + 0.5*sum(log(diag(VD)));
                    
                    % 5. term (1/2 element) & 2. term
                    T=1./sigm2vec_i;
                    term52 = -0.5*( nutilde'*(eta) + (nutilde'*Lhat)*(Bhat*nutilde) - (nutilde./(T+tautilde))'*nutilde);
                                        
                    % 5. term (2/2 element)
                    term5 = - 0.5*muvec_i'.*(T./(tautilde+T))'*(tautilde.*muvec_i-2*nutilde);

                    % 3. term
                    term3 = -sum(log(M0));

                    logZep = term41+term52+term5+term3;

                    iter=iter+1;
                end
                edata = logZep;
                
                % b'  = (K_fu/K_uu*K_fu' + La + diag(1./tautilde)) \ (tautilde.\nutilde)
                % L   = S*Kfu * (Lav + 1./S)^(-1) / chol(K_uu + SsqrtKfu'*(Lav + 1./S)^(-1)*SsqrtKfu) 
                % La2 = D./S = Lav + 1./S, 
                %
                % The way evaluations are done is numerically more stable than with inversion of S (tautilde)
                % See equations (3.71) and (3.72) in Rasmussen and Williams (2006)               
                b = nutilde' - ((eta' + (nutilde'*Lhat)*Bhat).*tautilde');
                                
                L = (sqrtS*iDSsqrtKfu)/AA';
                La2 = sqrtS\D2/sqrtS;
                
                % Reorder all the returned and stored values
                b = b(r);
                L = L(r,:);
                La2 = La2(r,r);
                D = La(r,r);
                nutilde = nutilde(r);
                tautilde = tautilde(r);
                myy = myy(r);
                P = P(r,:);
                y = y(r);
                if ~isempty(z)
                    z = z(r,:);
                end
                % ============================================================
                % DTC,VAR
                % ============================================================
              case {'DTC' 'VAR' 'SOR'}
                % First evaluate needed covariance matrices
                % v defines that parameter is a vector
                u = gp.X_u;
                m = size(u,1);

                % First evaluate needed covariance matrices
                % v defines that parameter is a vector
                [Kv_ff, Cv_ff] = gp_trvar(gp, x);  % f x 1  vector
                K_fu = gp_cov(gp, x, u);           % f x u
                K_uu = gp_trcov(gp, u);     % u x u, noiseles covariance K_uu
                K_uu = (K_uu+K_uu')./2;     % ensure the symmetry of K_uu
                Luu = chol(K_uu)';
                % Evaluate the Lambda (La)
                % Q_ff = K_fu*inv(K_uu)*K_fu'
                % Here we need only the diag(Q_ff), which is evaluated below
                B=Luu\(K_fu');       % u x f

                
                Phi = B';
                m = size(Phi,2);
                
                R = eye(m,m);
                P = Phi;
                myy = zeros(size(y));                
                gamma = zeros(m,1);
                Ann=0;

                while iter<=maxiter && abs(logZep_tmp-logZep)>tol

                    logZep_tmp=logZep;
                    muvec_i = zeros(n,1); sigm2vec_i = zeros(n,1);
                    for i1=1:n
                        % approximate cavity parameters
                        phi = Phi(i1,:)';
                        Ann = sum((R*phi).^2);
                        tau_i = Ann^-1-tautilde(i1);
                        myy(i1) = phi'*gamma;
                        vee_i = Ann^-1*myy(i1)-nutilde(i1);

                        myy_i=vee_i/tau_i;
                        sigm2_i=tau_i^-1;

                        % marginal moments
                        [M0(i1), muhati, sigm2hati] = feval(gp.likelih.fh_tiltedMoments, gp.likelih, y, i1, sigm2_i, myy_i, z);
                        
                        % update site parameters
                        deltatautilde = sigm2hati^-1-tau_i-tautilde(i1);
                        tautilde(i1) = tautilde(i1)+deltatautilde;
                        deltanutilde = sigm2hati^-1*muhati-vee_i - nutilde(i1);
                        nutilde(i1) = sigm2hati^-1*muhati-vee_i;

                        % Update the parameters
                        lnn = sum((R*phi).^2);
                        updfact = deltatautilde/(1 + deltatautilde*lnn);
                        if updfact > 0
                            RtLphiU = R'*(R*phi).*sqrt(updfact);
                            R = cholupdate(R, RtLphiU, '-');
                        elseif updfact < 0
                            RtLphiU = R'*(R*phi).*sqrt(updfact);
                            R = cholupdate(R, RtLphiU, '+');
                        end
                        gamma = gamma - R'*(R*phi)*(deltatautilde*myy(i1)-deltanutilde);
                        
                        % Store cavity parameters
                        muvec_i(i1,1)=myy_i;
                        sigm2vec_i(i1,1)=sigm2_i;
                    end
                    
                    SS = Phi/(eye(m,m)+Phi'*diag(tautilde)*Phi)*Phi' ;
                    
                    % Re-evaluate the parameters
                    R = chol(inv(eye(m,m) + Phi'*(repmat(tautilde,1,m).*Phi)));
                    gamma = R'*(R*(Phi'*nutilde));
                    myy = Phi*gamma;

                    % Compute the marginal likelihood, see FULL model for
                    % details about equations
                    % 4. term & 1. term
                    Stildesqroot=sqrt(tautilde);
                    SsqrtPhi = Phi.*repmat(Stildesqroot,1,m);
                    AA = eye(m,m) + SsqrtPhi'*SsqrtPhi; AA = (AA+AA')/2;
                    AA = chol(AA,'lower');
                    term41 = - 0.5*sum(log(1+tautilde.*sigm2vec_i)) + sum(log(diag(AA)));

                    % 5. term (1/2 element) & 2. term
                    T=1./sigm2vec_i;
                    bb = nutilde'*Phi;
                    bb2 = bb*SsqrtPhi';
                    bb3 = bb2*SsqrtPhi/AA';
                    term52 = -0.5*( bb*bb' - bb2*bb2' + bb3*bb3' - (nutilde./(T+tautilde))'*nutilde);

                    % 5. term (2/2 element)
                    term5 = - 0.5*muvec_i'.*(T./(tautilde+T))'*(tautilde.*muvec_i-2*nutilde);

                    % 3. term
                    term3 = -sum(log(M0));
                    
                    logZep = term41+term52+term5+term3;

                    iter=iter+1;
                end
                edata = logZep;
                %L = iLaKfu;
                if strcmp(gp.type,'VAR')
                   Qv_ff = sum(B.^2)';
                   edata = edata + 0.5*sum((Kv_ff-Qv_ff).*tautilde);
                end

                temp = Phi*(SsqrtPhi'*(SsqrtPhi*bb'));
                %                b = Phi*bb' - temp + Phi*(SsqrtPhi'*(SsqrtPhi*(AA'\(AA\temp))));
                
                b = nutilde - bb2'.*Stildesqroot + repmat(tautilde,1,m).*Phi*(AA'\bb3');
                b = b';
                
%                 StildeKfu = zeros(size(K_fu));  % f x u,
%                 for i=1:n
%                     StildeKfu(i,:) = K_fu(i,:).*tautilde(i);  % f x u
%                 end
%                 A = K_uu+K_fu'*StildeKfu;  A = (A+A')./2;     % Ensure symmetry
%                 A = chol(A);
%                 L = StildeKfu/A;
                L = repmat(tautilde,1,m).*Phi/AA';
                %L = repmat(tautilde,1,m).*K_fu/AA';
                mu=nutilde./tautilde;
                %b = nutilde - mu'*L*L'*mu;
                %b=b';
                La2 = 1./tautilde;
                D = 0;
                
                % ============================================================
                % SSGP
                % ============================================================
              case 'SSGP'        % Predictions with sparse spectral sampling approximation for GP
                                 % The approximation is proposed by M. Lazaro-Gredilla, J. Quinonero-Candela and A. Figueiras-Vidal
                                 % in Microsoft Research technical report MSR-TR-2007-152 (November 2007)
                                 % NOTE! This does not work at the moment.
                
                % First evaluate needed covariance matrices
                % v defines that parameter is a vector
                Phi = gp_trcov(gp, x);        % n x m matrix and nxn sparse matrix                
                m = size(Phi,2);
                
                R = eye(m,m);
                P = Phi;
                myy = zeros(size(y));                
                gamma = zeros(m,1);
                Ann=0;

                while iter<=maxiter && abs(logZep_tmp-logZep)>tol

                    logZep_tmp=logZep;
                    muvec_i = zeros(n,1); sigm2vec_i = zeros(n,1);
                    for i1=1:n
                        % approximate cavity parameters
                        phi = Phi(i1,:)';
                        Ann = sum((R*phi).^2);
                        tau_i = Ann^-1-tautilde(i1);
                        myy(i1) = phi'*gamma;
                        vee_i = Ann^-1*myy(i1)-nutilde(i1);

                        myy_i=vee_i/tau_i;
                        sigm2_i=tau_i^-1;

                        % marginal moments
                        [M0(i1), muhati, sigm2hati] = feval(gp.likelih.fh_tiltedMoments, gp.likelih, y, i1, sigm2_i, myy_i, z);
                        
                        % update site parameters
                        deltatautilde = sigm2hati^-1-tau_i-tautilde(i1);
                        tautilde(i1) = tautilde(i1)+deltatautilde;
                        deltanutilde = sigm2hati^-1*muhati-vee_i - nutilde(i1);
                        nutilde(i1) = sigm2hati^-1*muhati-vee_i;

                        % Update the parameters
                        lnn = sum((R*phi).^2);
                        updfact = deltatautilde/(1 + deltatautilde*lnn);
                        if updfact > 0
                            RtLphiU = R'*(R*phi).*sqrt(updfact);
                            R = cholupdate(R, RtLphiU, '-');
                        elseif updfact < 0
                            RtLphiU = R'*(R*phi).*sqrt(updfact);
                            R = cholupdate(R, RtLphiU, '+');
                        end
                        gamma = gamma - R'*(R*phi)*(deltatautilde*myy(i1)-deltanutilde);
                        
                        % Store cavity parameters
                        muvec_i(i1,1)=myy_i;
                        sigm2vec_i(i1,1)=sigm2_i;
                    end
                    
                    SS = Phi/(eye(m,m)+Phi'*diag(tautilde)*Phi)*Phi' ;
                    
                    % Re-evaluate the parameters
                    R = chol(inv(eye(m,m) + Phi'*(repmat(tautilde,1,m).*Phi)));
                    gamma = R'*(R*(Phi'*nutilde));
                    myy = Phi*gamma;

                    % Compute the marginal likelihood, see FULL model for
                    % details about equations
                    % 4. term & 1. term
                    Stildesqroot=sqrt(tautilde);
                    SsqrtPhi = Phi.*repmat(Stildesqroot,1,m);
                    AA = eye(m,m) + SsqrtPhi'*SsqrtPhi; AA = (AA+AA')/2;
                    AA = chol(AA,'lower');
                    term41 = - 0.5*sum(log(1+tautilde.*sigm2vec_i)) + sum(log(diag(AA)));

                    % 5. term (1/2 element) & 2. term
                    T=1./sigm2vec_i;
                    bb = nutilde'*Phi;
                    bb2 = bb*SsqrtPhi';
                    bb3 = bb2*SsqrtPhi/AA';
                    term52 = -0.5*( bb*bb' - bb2*bb2' + bb3*bb3' - (nutilde./(T+tautilde))'*nutilde);

                    % 5. term (2/2 element)
                    term5 = - 0.5*muvec_i'.*(T./(tautilde+T))'*(tautilde.*muvec_i-2*nutilde);

                    % 3. term
                    term3 = -sum(log(M0));
                    
                    logZep = term41+term52+term5+term3;

                    iter=iter+1;
                end
                edata = logZep;
                %L = iLaKfu;

                temp = Phi*(SsqrtPhi'*(SsqrtPhi*bb'));
                %                b = Phi*bb' - temp + Phi*(SsqrtPhi'*(SsqrtPhi*(AA'\(AA\temp))));
                
                b = nutilde - bb2'.*Stildesqroot + repmat(tautilde,1,m).*Phi*(AA'\bb3');
                b = b';
                
                L = repmat(tautilde,1,m).*Phi/AA';
                La2 = tautilde;
                D = 0;
              otherwise
                error('Unknown type of Gaussian process!')
            end

            % =====================================================================================
            % Evaluate the prior contribution to the error from covariance functions and likelihood
            % =====================================================================================

            % Evaluate the prior contribution to the error from covariance functions
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
            
            % Evaluate the prior contribution to the error from likelihood functions
            if isfield(gp, 'likelih') && isfield(gp.likelih, 'p')
                likelih = gp.likelih;
                eprior = eprior + feval(likelih.fh_priore, likelih);
            end

            % The last things to do
            if isfield(gp.ep_opt, 'display') && gp.ep_opt.display == 1
                fprintf('   Number of iterations in EP: %d \n', iter-1)
            end

            e = edata + eprior;

            w0 = w;
            e0 = e;
            edata0 = edata;
            eprior0 = eprior;
            nutilde0 = nutilde;
            tautilde0 = tautilde;
            myy0 = myy;
            L0 = L;
            n0 = size(x,1);
            La20 = La2;
            b0 = b;
        end
    end
end
