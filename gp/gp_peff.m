function p_eff = gp_peff(gp, tx, ty, param);
%GP_PEFF	The efective number of parameters in GP model with focus on latent variables.
%
%	Description
%	P_EFF = EP_PEFF(GP, TX, TY, PARAM) Takes the Gaussian process data structure GP, 
%       training inputs TX and training outputs and returns the efective number of 
%       parameters as defined by Spiegelhalter et.al. (2002). PARAM is a string defining, 
%       which parameters have been optimized.	
%
%       NOTE!
%       The effective number of parameters is evaluated with focus on latent variable f. 
%       This means that the hyperparameters th (parameters of covariance function and 
%       likelihood) are considered fixed. (See Spiegelhalter et.al. (2002) for discussion 
%       on the parameters in focus in Bayesian model). Thus, the returned p_eff tells the 
%       effective number of latent variables. This statistics is important for example when 
%       assessing the goodness of Laplace or EP approximation in case of non-Gaussian 
%       likelihood (See Vanhatalo et. al. for discussion). 
%
%       If you want to evaluate the efective number of hyperparameters see gp_dic.
%
%       The effective number of parameters is approximated as follows:
%
%               p_eff = n - trace( K\C ),
%
%       where K is the prior covariance matrix and C the posterior covariance matrix. 
%       This approximation is introduced by Spiegelhalter et.al. (2002) in equation (16). 
%
%	See also
%         gp_dic	
%   
%       References: 
%         Spiegelhalter, Best, Carlin and van der Linde (2002). Bayesian measures
%         of model complexity and fit. J. R. Statist. Soc. B, 64, 583-639.
%         
%         Vanhatalo, Pietilä, Vehtari (20XX). Approximate inference for disease mapping 
%         with sparse Gaussian processes.
%   
% Copyright (c) 2007-2008 Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.


    tn = size(tx,1);
    
    if ~isstruct(gp.likelih)    % a regression model
        switch gp.type
          case 'FULL'
            
            
            [K, C] = gp_trcov(gp, tx);
            L = chol(C);
            p_eff = trace( L\(L'\K) );
            
          case 'FIC'
            u = gp.X_u;
            m = size(u,1);
            % Turn the inducing vector on right direction
            if size(u,2) ~= size(tx,2)
                u=u';
            end
            % Calculate some help matrices
            [Kv_ff, Cv_ff] = gp_trvar(gp, tx);  % 1 x f  vector
            K_fu = gp_cov(gp, tx, u);   % f x u
            K_uu = gp_trcov(gp, u);     % u x u, noiseles covariance K_uu
            Luu = chol(K_uu)';
            
            % Evaluate the Lambda (La) for specific model
            % Q_ff = K_fu*inv(K_uu)*K_fu'
            % Here we need only the diag(Q_ff), which is evaluated below
            B=Luu\(K_fu');
            Qv_ff=sum(B.^2)';
            Lav = Kv_ff-Qv_ff;
            Lav2 = Cv_ff-Qv_ff;

            iLaKfu = zeros(size(K_fu));  % f x u,
            for i=1:length(tx)
                iLaKfu(i,:) = K_fu(i,:)./Lav2(i);  % f x u
            end
            A = K_uu+K_fu'*iLaKfu;
            A = (A+A')./2;

            L = iLaKfu/chol(A);

            p_eff = sum(Lav./Lav2) + sum(sum( repmat(Lav2,1,m).\B'.*B' - L.*(L'*B'*B)' - L.*(L'.*repmat(Lav',m,1))', 2));
            
          case {'PIC' 'PIC_BLOCK'}
            u = gp.X_u;
            ind = gp.tr_index;
            if size(u,2) ~= size(tx,2)
                u=u';
            end
            
            % Calculate some help matrices
            [Kv_ff, Cv_ff] = gp_trvar(gp, tx);  % 1 x f  vector
            K_fu = gp_cov(gp, tx, u);         % f x u
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
                La{i} = Kbl_ff - Qbl_ff;
                La2{i} = Cbl_ff - Qbl_ff;
                iLaKfu(ind{i},:) = La2{i}\K_fu(ind{i},:);    
            end
            A = K_uu+K_fu'*iLaKfu;
            A = (A+A')./2;            % Ensure symmetry
            L = iLaKfu/chol(A);
            
            p_eff = sum(sum(- L.*(L'*B'*B)',2));
            for i=1:length(ind)
                LLa2 = chol(La2{i});
                p_eff = p_eff + trace(LLa2\(LLa2'\La{i})) + trace( LLa2\(LLa2'\B(:,ind{i})'*B(:,ind{i})) - L(ind{i},:)*L(ind{i},:)'*La{i} );
            end
            
          case 'CS+FIC'
            
        end
    
    else                        % A non Gaussian observation model
        
        switch gp.type
          case 'FULL'
            
          case 'FIC'
            u = gp.X_u;
            
            m = size(u,1);
            % Turn the inducing vector on right direction
            if size(u,2) ~= size(tx,2)
                u=u';
            end
            % Calculate some help matrices
            [Kv_ff, Cv_ff] = gp_trvar(gp, tx);  % 1 x f  vector
            K_fu = gp_cov(gp, tx, u);   % f x u
            K_uu = gp_trcov(gp, u);     % u x u, noiseles covariance K_uu
            Luu = chol(K_uu)';
            
            % Evaluate the Lambda (La) for specific model
            % Q_ff = K_fu*inv(K_uu)*K_fu'
            % Here we need only the diag(Q_ff), which is evaluated below
            B=Luu\(K_fu');
            Qv_ff=sum(B.^2)';
            Lav = Kv_ff-Qv_ff;
            
            switch gp.latent_method
              case 'EP'

                
              case 'Laplace'
                [e, edata, eprior, f, L, a, La2] = gpla_e(gp_pak(gp, param), gp, tx, ty, param);
                
                W = -feval(gp.likelih.fh_g2, gp.likelih, ty, f, 'latent');
                La = W.*Lav;
                Lahat = 1 + La;
                sqrtW = sqrt(W);
                B = (repmat(sqrtW,1,m).*K_fu);
                
                % Components for (I + W^(1/2)*(Qff + La2)*W^(1/2))^(-1) = Lahat^(-1) - L2*L2'
                B2 = repmat(Lahat,1,m).\B;
                A2 = K_uu + B'*B2; A2=(A2+A2)/2;
                L2 = B2/chol(A2);
                
                BB=Luu\(K_fu');
                BB2=B/Luu';
                
                p_eff = sum(W./Lahat.*Lav) + sum(sqrtW .* (sum((repmat(Lahat,1,m).\BB2).*BB',2) - sum(L2.*(L2'.*repmat(sqrtW'.*La2',m,1))',2) - sum(L2.*(L2'*BB2*BB)',2)) );
                
          case {'PIC' 'PIC_BLOCK'}
          
          
          case 'CS+FIC'
            u = gp.X_u;
            
            switch gp.latent_method
              case 'EP'

                
              case 'Laplace'
                [e, edata, eprior, f, L, a, La2] = gpla_e(gp_pak(gp, param), gp, tx, ty, param);
                                
                % Indexes to all non-compact support and compact support covariances.
                cf1 = [];
                cf2 = [];
                
                ncf = length(gp.cf);
                % Loop through all covariance functions
                for i = 1:ncf        
                    % Non-CS covariances
                    if ~isfield(gp.cf{i},'cs') 
                        cf1 = [cf1 i];
                        % CS-covariances
                    else
                        cf2 = [cf2 i];           
                    end
                end

                K_fu = gp_cov(gp,tx,u,cf1);         % f x u
                K_uu = gp_trcov(gp,u,cf1);    % u x u, noiseles covariance K_uu
                K_uu = (K_uu+K_uu')./2;     % ensure the symmetry of K_uu
                
                Kcs = gp_trcov(gp, tx, cf2);
                Luu = chol(K_uu)';

                W = -feval(gp.likelih.fh_g2, gp.likelih, ty, f, 'latent');
                sqrtW = sparse(1:tn,1:tn,sqrt(W),tn,tn);
                Lahat = sparse(1:tn,1:tn,1,tn,tn) + sqrtW*La2*sqrtW;
                B = sqrtW*K_fu;

                % Components for (I + W^(1/2)*(Qff + La2)*W^(1/2))^(-1) = Lahat^(-1) - L2*L2'
                B2 = Lahat\B;
                A2 = K_uu + B'*B2; A2=(A2+A2)/2;
                L2 = B2/chol(A2);

                BB = Luu\(K_fu');
                BB2 = B/Luu';

                VD = ldlchol(Lahat);
                spiLahat = spinv(VD,1);
                
                p_eff = sum(sum(sqrtW*spiLahat*sqrtW.*La2,2)) + sum(sqrtW * (sum(ldlsolve(VD, BB2).*BB',2) - sum(L2.*(L2'*sqrtW*La2)',2) - sum(L2.*(L2'*BB2*BB)',2)) );
                                
            end

            
            
            
        end
    end
    
    
end