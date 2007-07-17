function [e, edata, eprior, site_tau, site_nu, L] = gpep_e(w, gp, x, y, param, varargin)
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

% Copyright (c) 2007      Jarno Vanhatalo, Jaakko Riihimäki

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.
    
    if strcmp(w, 'init')
        w0 = rand(size(gp_pak(gp, param)));
        e0=[]; 
        edata0= inf; 
        eprior0=[];
        nutilde0 = zeros(size(y));
        tautilde0 = zeros(size(y));
        myy0 = zeros(size(y));;
        L0 = [];
        myy=zeros(size(y));
    
        ep_algorithm(gp_pak(gp,param), gp, x, y, param, varargin);
        
        gp.fh_e = @ep_algorithm;
        e = gp;
    else
        
        [e, edata, eprior, site_tau, site_nu, L] = feval(gp.fh_e, w, gp, x, y, param, varargin);
        
    end
    
    function [e, edata, eprior, tautilde, nutilde, L, B] = ep_algorithm(w, gp, x, y, param, varargin)

        if abs(w-w0) < 1e-6
            % The covariance function parameters haven't changed so just 
            % return the Energy and the site parameters that are saved
            e = e0;
            edata = edata0;
            eprior = eprior0;
            nutilde = nutilde0;
            myy = myy0;
            tautilde = tautilde0;
            L = L0;
            %                fprintf('palauta vanhat \n')
        else
            % Conduct evaluation for the energy and the site parameters
            gp=gp_unpak(gp, w, param);
            ncf = length(gp.cf);
            n=length(x);

            % ep iteration parameters
            iter=1;
            maxiter = gp.ep_opt.maxiter;
            tol = gp.ep_opt.tol;
            
            % ep initialisation
% $$$             logZep_tmp = edata0;
% $$$             nutilde = nutilde0;
% $$$             tautilde = tautilde0;
% $$$             myy = myy0;
            nutilde = zeros(size(y));
            tautilde = zeros(size(y));
            myy = zeros(size(y));
            logZep_tmp=0; logZep=Inf;
            
            % =================================================
            % First Evaluate the data contribution to the error    
            switch gp.type
              case 'FULL'   % A full GP
                [K,C] = gp_trcov(gp, x);
                Sigm = C;
                Stildesqroot=zeros(n);
                
                % The EP -algorithm
                while iter<=maxiter & abs(logZep_tmp-logZep)>tol
                    
                    logZep_tmp=logZep;
                    muvec_i = zeros(n,1); sigm2vec_i = zeros(n,1);
                    for i1=1:n
                        % approximate cavity parameters
                        tau_i=Sigm(i1,i1)^-1-tautilde(i1);
                        vee_i=Sigm(i1,i1)^-1*myy(i1)-nutilde(i1);

                        myy_i=vee_i/tau_i;
                        sigm2_i=tau_i^-1;

                        % marginal moments
                        %[muhati, sigm2hati] = marginalMoments(gp.likelih);
                        zi=y(i1)*myy_i/sqrt(1+sigm2_i);
                        normp_zi = normpdf(zi);
                        normc_zi = normcdf(zi);
                        muhati=myy_i+(y(i1)*sigm2_i*normp_zi)/(normc_zi*sqrt(1+sigm2_i));
                        sigm2hati=sigm2_i-(sigm2_i^2*normp_zi)/((1+sigm2_i)*normc_zi)*(zi+normp_zi/normc_zi);
                
                        
                        % update site parameters
                        deltatautilde=sigm2hati^-1-tau_i-tautilde(i1); tautilde(i1)=tautilde(i1)+deltatautilde;
                        nutilde(i1)=sigm2hati^-1*muhati-vee_i;

                        apu = deltatautilde^-1+Sigm(i1,i1);
                        apu = (Sigm(:,i1)/apu)*Sigm(:,i1)';
                        Sigm = Sigm - apu;
                        %Sigm=Sigm-(deltatautilde^-1+Sigm(i1,i1))^-1*(Sigm(:,i1)*Sigm(:,i1)');
                        myy=Sigm*nutilde;

                        muvec_i(i1,1)=myy_i;
                        sigm2vec_i(i1,1)=sigm2_i;
                    end

                    Stilde=tautilde;
% $$$                     Stilde=diag(tautilde);
                    Stildesqroot=diag(sqrt(tautilde));

                    % NOTICE! upper triangle matrix! cf. to
                    % line 13 in the algorithm 3.5, p. 58.
                    B=eye(n)+Stildesqroot*C*Stildesqroot;
                    L=chol(B,'lower');

                    V=L\Stildesqroot*C;
                    Sigm=C-V'*V; myy=Sigm*nutilde;

                    % Direct formula (3.65):
                    % Sigmtilde=diag(1./tautilde);
                    % mutilde=inv(Stilde)*nutilde;
                    %
                    % logZep=-0.5*log(det(Sigmtilde+K))-0.5*mutilde'*inv(K+Sigmtilde)*mutilde+sum(log(normcdf(y.*muvec_i./sqrt(1+sigm2vec_i))))+0.5*sum(log(sigm2vec_i+1./tautilde))+sum((muvec_i-mutilde).^2./(2*(sigm2vec_i+1./tautilde)))

                    % 4. term & 1. term
                    term41=0.5*sum(log(1+tautilde.*sigm2vec_i))-sum(log(diag(L)));

                    % 5. term (1/2 element) & 2. term
% $$$                     T=diag(1./sigm2vec_i);
% $$$                     term52=0.5*nutilde'*(C-C*Stildesqroot*inv(B)*Stildesqroot*C-inv(T+Stilde))*nutilde;
                    T=1./sigm2vec_i;
                    Cnutilde = C*nutilde;
                    L2 = V*nutilde;
                    term52 = nutilde'*Cnutilde - L2'*L2 - (nutilde'./(T+Stilde)')*nutilde;
                    term52 = term52.*0.5;

                    % 5. term (2/2 element)
% $$$                     term5=0.5*muvec_i'*T*inv(Stilde+T)*(Stilde*muvec_i-2*nutilde);
                    term5=0.5*muvec_i'.*(T./(Stilde+T))'*(Stilde.*muvec_i-2*nutilde);

                    % 3. term
                    term3=sum(log(normcdf(y.*muvec_i./sqrt(1+sigm2vec_i))));

                    logZep=-(term41+term52+term5+term3);

                    iter=iter+1;
                end
                
                if isfield(gp.ep_opt, 'display') && gp.ep_opt.display == 1
                    fprintf('   Number of iterations in EP: %d \n', iter)
                end
                edata = logZep;
                                
              case 'FIC'
                % The eguations in FIC are implemented as in Neil, 2006
                u = gp.X_u;
                
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
                                     % iLaKfu = diag(iLav)*K_fu = inv(La)*K_fu
                iLaKfu = zeros(size(K_fu));  % f x u, 
                for i=1:n
                    iLaKfu(i,:) = K_fu(i,:)./Lav(i);  % f x u 
                end
                % The data contribution to the error is 
                % E = n/2*log(2*pi) + 0.5*log(det(Q_ff+La)) + 0.5*t'inv(Q_ff+La)t
                %   = + 0.5*log(det(La)) + 0.5*trace(iLa*t*t') - 0.5*log(det(K_uu)) 
                %     + 0.5*log(det(A)) - 0.5*trace(inv(A)*iLaKfu'*t*t'*iLaKfu)
                
                % First some help matrices...
                % A = chol(K_uu+K_uf*inv(La)*K_fu))
                A = K_uu+K_fu'*iLaKfu;
                A = (A+A')./2;     % Ensure symmetry
                A = chol(A)';
                % The actual error evaluation
                % 0.5*log(det(K)) = sum(log(diag(L))), where L = chol(K). NOTE! chol(K) is upper triangular
                b = (t'*iLaKfu)*inv(A)';
                edata = sum(log(Lav)) + t'./Lav'*t - 2*sum(log(diag(Luu))) + 2*sum(log(diag(A))) - b*b';
                edata = .5*(edata + n*log(2*pi));
              case 'PIC_BLOCK'
                u = gp.X_u;
                ind = gp.tr_index;
                
                % First evaluate needed covariance matrices
                % v defines that parameter is a vector
                K_fu = gp_cov(gp, x, u);         % f x u
                K_uu = gp_trcov(gp, u);    % u x u, noiseles covariance K_uu
                K_uu = (K_uu+K_uu')./2;     % ensure the symmetry of K_uu
                Luu = chol(K_uu)';
                % Evaluate the Lambda (La) 
                % Q_ff = K_fu*inv(K_uu)*K_fu'
                % Here we need only the blockdiag(Q_ff), which is evaluated below
                B=Luu\(K_fu');       % u x f  and B'*B = K_fu*K_uu*K_uf
                iLaKfu = zeros(size(K_fu));  % f x u
                edata = 0;
                for i=1:length(ind)
                    Qbl_ff = B(:,ind{i})'*B(:,ind{i});
                    [Kbl_ff, Cbl_ff] = gp_trcov(gp, x(ind{i},:));
                    Labl{i} = Cbl_ff - Qbl_ff;
                    iLaKfu(ind{i},:) = Labl{i}\K_fu(ind{i},:);
                    edata = edata + 2*sum(log(diag(chol(Labl{i})))) + t(ind{i},:)'*(Labl{i}\t(ind{i},:));
                end
                % The data contribution to the error is 
                % E = n/2*log(2*pi) + 0.5*log(det(Q_ff+La)) + 0.5*t'inv(Q_ff+La)t
                
                % First some help matrices...
                % A = chol(K_uu+K_uf*inv(La)*K_fu))
                A = K_uu+K_fu'*iLaKfu;
                A = (A+A')./2;     % Ensure symmetry
                A = chol(A)';
                % The actual error evaluation
                % 0.5*log(det(K)) = sum(log(diag(L))), where L = chol(K). NOTE! chol(K) is upper triangular
                b = (t'*iLaKfu)*inv(A)';
                edata = edata - 2*sum(log(diag(Luu))) + 2*sum(log(diag(A))) - b*b';
                edata = .5*(edata + n*log(2*pi));
              case  'PIC_BAND'
                u = gp.X_u;
                ind = gp.tr_index;
                
                % First evaluate needed covariance matrices
                % v defines that parameter is a vector
                K_fu = gp_cov(gp, x, u);         % f x u
                K_uu = gp_trcov(gp, u);    % u x u, noiseles covariance K_uu
                K_uu = (K_uu+K_uu')./2;     % ensure the symmetry of K_uu
                Luu = chol(K_uu)';
                % Evaluate the Lambda (La) 
                % Q_ff = K_fu*inv(K_uu)*K_fu'
                % Here we need only the blockdiag(Q_ff), which is evaluated below
                B=Luu\(K_fu');       % u x f  and B'*B = K_fu*K_uu*K_uf
                                     %            q_ff = zeros(1,size(ind,1));
    % $$$             for i = 1:size(ind,1)
    % $$$                 q_ff(i) = B(:,ind(i,1))'*B(:,ind(i,2));
    % $$$                 %    c_ff(i) = gp_cov(gp, x(ind(i,1),:), x(ind(i,2),:));
    % $$$             end
    % $$$             c_ff = gp_covvec(gp, x(ind(:,1),:), x(ind(:,2),:))';
    % $$$             [Kv_ff, Cv_ff] = gp_trvar(gp,x);
    % $$$             La = sparse(ind(:,1),ind(:,2),c_ff-q_ff,n,n) + sparse(1:n,1:n, Cv_ff-Kv_ff,n,n);
                
                [I,J]=find(tril(sparse(ind(:,1),ind(:,2),1,n,n),-1));
    % $$$             for i = 1:length(J)
    % $$$                 q_ff(i) = B(:,I(i))'*B(:,J(i));
    % $$$             end
                q_ff = sum(B(:,I).*B(:,J));
                q_ff = sparse(I,J,q_ff,n,n);
                c_ff = gp_covvec(gp, x(I,:), x(J,:))';
                c_ff = sparse(I,J,c_ff,n,n);
                [Kv_ff, Cv_ff] = gp_trvar(gp,x);
                La = c_ff + c_ff' - q_ff - q_ff' + sparse(1:n,1:n, Cv_ff-sum(B.^2,1)',n,n);

                
                %cputime - t
                
                iLaKfu = La\K_fu;

                % The data contribution to the error is 
                % E = n/2*log(2*pi) + 0.5*log(det(Q_ff+La)) + 0.5*t'inv(Q_ff+La)t
                %   = + 0.5*log(det(La)) + 0.5*trace(iLa*t*t') - 0.5*log(det(K_uu)) 
                %     + 0.5*log(det(A)) - 0.5*trace(inv(A)*iLaKfu'*t*t'*iLaKfu)
                
                % First some help matrices...
                % A = chol(K_uu+K_uf*inv(La)*K_fu))
                A = K_uu+K_fu'*iLaKfu;
                A = (A+A')./2;     % Ensure symmetry
                                   % The actual error evaluation
                                   % 0.5*log(det(K)) = sum(log(diag(L))), where L = chol(K). NOTE! chol(K) is upper triangular
                b = (t'*iLaKfu)/chol(A);
                
                edata = 2*sum(log(diag(chol(La)))) + t'*(La\t);
                edata = edata - 2*sum(log(diag(Luu))) + 2*sum(log(diag(A))) - b*b';
                edata = .5*(edata + n*log(2*pi));
              otherwise
                error('Unknown type of Gaussian process!')
            end
            
            % ======================================================================
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
            e = edata + eprior;
            
            w0 = w;
            e0 = e;
            edata0 = edata;
            eprior0 = eprior;
            nutilde0 = nutilde;
            tautilde0 = tautilde;
            myy0 = myy;
            L0 = L;

        end
        
        % Evaluate the marginal moments
        function [muhati1, sigm2hati1] = marginalMoments(likelihood)
            switch likelihood
              case 'probit'
                zi=y(i1)*myy_i/sqrt(1+sigm2_i);
                muhati1=myy_i+(y(i1)*sigm2_i*normpdf(zi))/(normcdf(zi)*sqrt(1+sigm2_i));
                sigm2hati1=sigm2_i-(sigm2_i^2*normpdf(zi))/((1+sigm2_i)*normcdf(zi))*(zi+normpdf(zi)/normcdf(zi));
              case 'poisson'
                
            end
        end
    end
end
