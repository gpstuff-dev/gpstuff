function [Y, VarY] = gp_preds(gp, tx, ty, x, predcf, tstind)
%GP_PREDS	(Multible) Predictions of Gaussian Processes.
%
%	Description
%	F = GP_PREDS(RECGP, TX, TY, X) takes a Gaussian processes record structure 
%       RECGP together with a matrix X of input vectors, matrix TX of training inputs 
%       and vector TY of training targets, and makes predictions with Gaussian 
%       processes stored in RECGP to generate a matrix F of outputs. Each row of X 
%       corresponds to one input vector and each row of F corresponds to one output
%       vector related to the row of X.
%       
%	[F, VarF] = GP_PREDS(GPCF, TX, TY, X) returns also the predictive (noiseles) 
%       variances of F (1xn vector). NOTE! Each column of VarF contains the variances 
%       of the latent function, that is  diag(K_fy - K_fy*(Kyy+s^2I)^(-1)*K_yf. If 
%       you want predictive variance of observations add gp.noise{1}.noiseSigmas2 to VarF.
%
%       LATENT VALUES
%       TY can be also matrix of latent values in which case it is of size N x NMC, where 
%       NMC is the number of Monte Carlo samples stored in the RECGP structure. In this case
%       TY is handled as all the other sampled parameters.
%
%	See also
%	GP, GP_PAK, GP_UNPAK, GP_PRED

% Copyright (c) 2000      Aki Vehtari
% Copyright (c) 2006      Helsinki University of Technology (author Jarno Vanhatalo)
% Copyright (c) 2007-2008 Jarno Vanhatalo

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.

    tn = size(tx,1);
    if nargin < 4
        error('Requires at least 4 arguments');
    end
    if nargin < 5
        predcf = [];
    end  

    nin  = gp.nin;
    nout = gp.nout;

    nmc=size(gp.hmcrejects,1);
    Y = zeros(size(x,1),nmc);

    if strcmp(gp.type, 'PIC_BLOCK') || strcmp(gp.type, 'PIC')
        ind = gp.tr_index;           % block indeces for training points
        gp = rmfield(gp,'tr_index');
    end

    % Check for CAR covariance functions and remove adjancency
    % matrices from them
    A    = cell(1,length(gp.cf));
    Aexp = cell(1,length(gp.cf)); 
    for i1 = 1:length(gp.cf)
        if strcmp(gp.cf{i1}.type, 'gpcf_car')
            % Save matrices
            A{i1} = gp.cf{i1}.A;
            Aexp{i1} = gp.cf{i1}.Aexp;            
            
            % Delete before taking the nth sample
            gp.cf{i1} = rmfield(gp.cf{i1},'A');
            gp.cf{i1} = rmfield(gp.cf{i1},'Aexp');
        end
    end
    
    % loop over all samples
    for i1=1:nmc
        Gp = take_nth(gp,i1);

        % Put the adjancency matrices back to CAR covariances
        for i2 = 1:length(Gp.cf)
            if strcmp(Gp.cf{i2}.type, 'gpcf_car')
                Gp.cf{i2}.A = A{i2};
                Gp.cf{i2}.Aexp = Aexp{i2};            
            end
        end

        
        switch gp.type            
            % --------------------------------------------
            %  FULL GP                                   
            % --------------------------------------------
          case 'FULL'         % Do following if full model is used    
            [c,C]=gp_trcov(Gp, tx);
            K=gp_cov(Gp, tx, x, predcf);            
            
            % This is used only in the case of latent values. 
            if size(ty,2)>1
                if issparse(C)
                    LD = ldlchol(C);
                    y = K'*ldlsolve(LD,ty(:,i1));
                else
                    L = chol(C)';
                    a = L'\(L\ty(:,i1));
                    y = K'*a;
                end
            else    % Here latent values are not present
                if issparse(C)
                    LD = ldlchol(C);
                    y = K'*ldlsolve(LD,ty);
                else
                    L = chol(C)';
                    a = L'\(L\ty);
                    y = K'*a;
                end
            end
            
            if nargout>1
                if issparse(C)
                    V = gp_trvar(Gp,x,predcf);
                    VarY(:,i1) = V - diag(K'*ldlsolve(LD,K));
                else
                    v = L\K;
                    V = gp_trvar(Gp,x,predcf);
                    % Vector of diagonal elements of covariance matrix
                    % b = L\K;
                    % VarY = V - sum(b.^2)';
                    VarY(:,i1) = V - diag(v'*v);
                end
            end
            Y(:,i1) = y;
            % --------------------------------------------
            %  FIC GP                                   
            % --------------------------------------------
          case 'FIC'        % Do following if FIC sparse model is used
            if nargin > 5
                if length(tstind) ~= size(tx,1)
                    error('tstind (if provided) has to be of same lenght as tx.')
                end
            else
                tstind = [];
            end
            
            % Calculate some help matrices  
            u = reshape(Gp.X_u,length(Gp.X_u)/nin,nin);
            m = size(u,1);
            [Kv_ff, Cv_ff] = gp_trvar(Gp, tx);  % 1 x f  vector
            K_fu = gp_cov(Gp, tx, u);           % f x u
            K_nu = gp_cov(Gp, x, u);            % n x u
            K_uu = gp_trcov(Gp, u);             % u x u, noiseles covariance K_uu
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

            % Prediction matrices formed with only subsetof cf's.
            if ~isempty(predcf)
                K_fu = gp_cov(Gp,tx,u,predcf);     % f x u
                K_uu = gp_trcov(Gp,u,predcf);      % u x u, noiseles covariance K_uu
                K_nu = gp_cov(Gp,x,u,predcf);      % n x u
            end

            if size(ty,2)>1
                p = ty(:,i1)./Lav - L*(L'*ty(:,i1)); 
            else                         % Here latent values are not present
                p = ty./Lav - L*(L'*ty); 
            end
            y = K_nu*(K_uu\(K_fu'*p));
            
            % If the prediction is made for training set, evaluate Lav also for prediction points
            if ~isempty(tstind)
                [Kv_ff, Cv_ff] = gp_trvar(Gp, x(tstind,:), predcf);
                Luu = chol(K_uu)';
                B=Luu\(K_fu');
                Qv_ff=sum(B.^2)';
                Lav2 = zeros(size(Lav));
                Lav2(tstind) = Kv_ff-Qv_ff;
                y = y + Lav2.*p;
            end

            if nargout > 1   % see Quinonera-Candela&Rasmussen (2005)
                Knn_v = gp_trvar(Gp,x,predcf);
                Luu = chol(K_uu)';
                B=Luu\(K_fu');
                B2=Luu\(K_nu');        
                K_fu = gp_cov(Gp, tx, u, predcf);         % f x u
                K_uu = gp_trcov(Gp, u, predcf);    % u x u, noiseles covariance K_uu
                VarY(:,i1) = Knn_v - sum(B2'.*(B*(repmat(Lav,1,size(K_uu,1)).\B')*B2)',2)  + sum((K_nu*(K_uu\(K_fu'*L))).^2, 2);
                % if the prediction is made for training set, evaluate Lav also for prediction points
                if ~isempty(tstind)
                    VarY(tstind,i1) = VarY(tstind,i1) - 2.*sum( B2(:,tstind)'.*(repmat((Lav.\Lav2(tstind)),1,m).*B'),2) ...
                        + 2.*sum( B2(:,tstind)'*(B*L).*(repmat(Lav2(tstind),1,m).*L), 2)  ...
                        - Lav2(tstind)./Lav.*Lav2(tstind) + sum((repmat(Lav2(tstind),1,m).*L).^2,2);                
                end                
            end
            Y(:,i1) = y;
            % --------------------------------------------
            %  PIC                                   
            % --------------------------------------------            
          case {'PIC' 'PIC_BLOCK'}  % Do following if FIC sparse model is used
                                    % Calculate some help matrices  
            u = reshape(Gp.X_u,length(Gp.X_u)/nin,nin);
            
            [Kv_ff, Cv_ff] = gp_trvar(Gp, tx);  % 1 x f  vector
            K_fu = gp_cov(Gp,tx,u);         % f x u
            K_nu = gp_cov(Gp,x,u);         % n x u   
            K_uu = gp_trcov(Gp,u);    % u x u, noiseles covariance K_uu
            Luu = chol(K_uu)';
            
            % Evaluate the Lambda (La) for specific model
            % Q_ff = K_fu*inv(K_uu)*K_fu'
            % Here we need only the diag(Q_ff), which is evaluated below
            B=Luu\K_fu';
            iLaKfu = zeros(size(K_fu));  % f x u
            for i=1:length(ind)
                Qbl_ff = B(:,ind{i})'*B(:,ind{i});
                %            Qbl_ff2(ind{i},ind{i}) = B(:,ind{i})'*B(:,ind{i});
                [Kbl_ff, Cbl_ff] = gp_trcov(Gp, tx(ind{i},:));
                La{i} = Cbl_ff - Qbl_ff;
                iLaKfu(ind{i},:) = La{i}\K_fu(ind{i},:);    
            end
            A = K_uu+K_fu'*iLaKfu;
            A = (A+A')./2;            % Ensure symmetry
            L = iLaKfu/chol(A);
            
            if size(ty,2)>1
                tyy = ty(:,i1);
            else    % Here latent values are not present
                tyy = ty;
            end

            % From this on evaluate the prediction
            % See Snelson and Ghahramani (2007) for details 
            p=iLaKfu*(A\(iLaKfu'*tyy));
            for i=1:length(ind)
                p2(ind{i},:) = La{i}\tyy(ind{i},:);
            end
            p= p2-p;        
            
            % Prediction matrices formed with only subsetof cf's.
            if ~isempty(predcf)
                K_fu = gp_cov(Gp,tx,u,predcf);        % f x u
                K_nu = gp_cov(Gp,x,u,predcf);         % n x u
                K_uu = gp_trcov(Gp,u,predcf);          % u x u, noiseles covariance K_uu
            end

            iKuuKuf = K_uu\K_fu';
            w_u = K_uu\(K_fu'*p);
            
            w_bu=zeros(length(x),length(u));
            w_n=zeros(length(x),1);
            for i=1:length(ind)
                w_bu(tstind{i},:) = repmat((iKuuKuf(:,ind{i})*p(ind{i},:))',length(tstind{i}),1);
                K_nb = gp_cov(Gp, x(tstind{i},:), tx(ind{i},:), predcf);              % n x u
                w_n(tstind{i},:) = K_nb*p(ind{i},:);
            end
            %    [max(- sum(K_nu.*w_bu,2) + w_n), mean(- sum(K_nu.*w_bu,2) + w_n), min(- sum(K_nu.*w_bu,2) + w_n)]
            y = K_nu*w_u - sum(K_nu.*w_bu,2) + w_n;
            
            if nargout > 1
                % Form iLaKfu again if a subset of cf's is used for making predictions
                if ~isempty(predcf)
                    iLaKfu = zeros(size(K_fu));  % f x u
                    for i=1:length(ind)
                        iLaKfu(ind{i},:) = La{i}\K_fu(ind{i},:);    
                    end
                end
                
                kstarstar = gp_trvar(Gp, x, predcf);
                %iKuuKuf = K_uu\K_fu';
                KnfL = K_nu*(iKuuKuf*L);
                Varf = zeros(length(x),1);
                for i=1:length(ind)
                    v_n = gp_cov(Gp, x(tstind{i},:), tx(ind{i},:), predcf);              % n x u
                    v_bu = K_nu(tstind{i},:)*iKuuKuf(:,ind{i});
                    KnfLa = K_nu*(iKuuKuf(:,ind{i})/chol(La{i}));
                    KnfLa(tstind{i},:) = KnfLa(tstind{i},:) - (v_bu + v_n)/chol(La{i});
                    Varf = Varf + sum((KnfLa).^2,2);
                    KnfL(tstind{i},:) = KnfL(tstind{i},:) - v_bu*L(ind{i},:) + v_n*L(ind{i},:);
                end
                Varf = kstarstar - (Varf - sum((KnfL).^2,2));  
                
                VarY(:,i1) = Varf;
            end
            
            Y(:,i1) = y;
            % --------------------------------------------
            %  CS+FIC                                   
            % --------------------------------------------
          case 'CS+FIC'    
            % Here tstind = 1 if the prediction is made for the training set 
            if nargin > 5
                %tstind = varargin{2};
                if length(tstind) ~= size(tx,1)
                    error('tstind (if provided) has to be of same lenght as tx.')
                end
            else
                tstind = [];
            end
            
            u = reshape(Gp.X_u,length(Gp.X_u)/nin,nin);
            ncf = length(Gp.cf);
            cf_orig = Gp.cf;
            
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
            [Kv_ff, Cv_ff] = gp_trvar(Gp,tx,cf1);  % f x 1  vector
            K_fu = gp_cov(Gp,tx,u,cf1);         % f x u
            K_uu = gp_trcov(Gp,u,cf1);    % u x u, noiseles covariance K_uu
            K_uu = (K_uu+K_uu')./2;     % ensure the symmetry of K_uu
            Luu = chol(K_uu)';
            K_nu = gp_cov(Gp,x,u,cf1);         % n x u

            % Evaluate the Lambda (La)
            % Q_ff = K_fu*inv(K_uu)*K_fu'
            B=Luu\(K_fu');       % u x f
            Qv_ff=sum(B.^2)';
            Lav = Cv_ff-Qv_ff;   % f x 1, Vector of diagonal elements

            K_cs = gp_trcov(Gp,tx,cf2);
            Kcs_nf = gp_cov(Gp, x, tx, predcf2);
            La = sparse(1:tn,1:tn,Lav,tn,tn) + K_cs;
            
            % evaluate also Lav2 if the prediction is made for training set
            if ~isempty(tstind)
                [Kv_ff, Cv_ff] = gp_trvar(gp,x(tstind,:),predcf1);
                Luu = chol(K_uu)';
                B=Luu\(K_fu');
                Qv_ff=sum(B.^2)';
                Lav2 = zeros(size(x,1));
                Lav2(tstind) = Kv_ff-Qv_ff;
            end

            iLaKfu = La\K_fu;
            A = K_uu+K_fu'*iLaKfu;
            A = (A+A')./2;     % Ensure symmetry
            L = iLaKfu/chol(A);
            
            if size(ty,2)>1
                p = La\ty(:,i1) - L*(L'*ty(:,i1));
            else
                p = La\ty - L*(L'*ty);
            end

            % Calculate the predictive mean according to the type of
            % covariance functions used for making the prediction
            if ptype == 1
                Ef = K_nu*(K_uu\(K_fu'*p));
            elseif ptype == 2
                Ef = Kcs_nf*p;
            else 
                Ef = K_nu*(K_uu\(K_fu'*p)) + Kcs_nf*p;        
            end
            
            % Add also Lav2 if the prediction is made for training set
            % and non-CS covariance function is used for prediction
            if ~isempty(tstind) && (ptype == 1 || ptype == 3)
                Ef = Ef + Lav2.*p;
            end
    
            if nargout > 1
                Knn_v = gp_trvar(Gp,x,predcf);
                Luu = chol(K_uu)';
                B=Luu\(K_fu');
                B2=Luu\(K_nu');
                
                % Calculate the predictive variance according to the type
                % covariance functions used for making the prediction
                if ptype == 1 || ptype == 3                            
                    % FIC part of the covariance
                    Varf = Knn_v - sum(B2'.*(B*(La\B')*B2)',2) + sum((K_nu*(K_uu\(K_fu'*L))).^2, 2);
                    % Add Lav2 if the prediction is made for the training set
                    if  ~isempty(tstind)
                        % Non-CS covariance
                        if ptype == 1
                            Kcs_nf = sparse(tstind,1:n,Lav2,n2,n);
                            % Non-CS and CS covariances
                        else
                            Kcs_nf = Kcs_nf + sparse(tstind,1:n,Lav2,n2,n);
                        end
                        % Add Lav2 and possibly Kcs_nf
                        Varf = Varf - sum((Kcs_nf/chol(La)).^2,2) + sum((Kcs_nf*L).^2, 2) ...
                               - 2.*sum((Kcs_nf*iLaKfu).*(K_uu\K_nu')',2) + 2.*sum((Kcs_nf*L).*(L'*K_fu*(K_uu\K_nu'))' ,2);                
                        % In case of both non-CS and CS prediction covariances add 
                        % only Kcs_nf if the prediction is not done for the training set 
                    elseif ptype == 3
                        Varf = Varf - sum((Kcs_nf/chol(La)).^2,2) + sum((Kcs_nf*L).^2, 2) ...
                               - 2.*sum((Kcs_nf*iLaKfu).*(K_uu\K_nu')',2) + 2.*sum((Kcs_nf*L).*(L'*K_fu*(K_uu\K_nu'))' ,2);
                    end
                    % Prediction with only CS covariance
                elseif ptype == 2
                    Varf = Knn_v - sum((Kcs_nf/chol(La)).^2,2) + sum((Kcs_nf*L).^2, 2) ;
                end   
                VarY(:,i1) = Varf;
            end
            Y(:,i1) = y;
          case 'SSGP'
            error('gp_pred: SSGP is not implemented yet!')
            
        end %switch gp.type
        
    end

function x = take_nth(x,nth)
%TAKE_NTH    Take n'th parameters from MCMC-chains
%
%   x = take_nth(x,n) returns chain containing only
%   n'th simulation sample 
%
%   See also
%     THIN, JOIN

% Copyright (c) 1999 Simo S�rkk�
% Copyright (c) 2000 Aki Vehtari
% Copyright (c) 2006 Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

    if nargin < 2
        n = 1;
    end

    [m,n]=size(x);

    if isstruct(x)
        if (m>1 | n>1)
            % array of structures
            for i=1:(m*n)
                x(i) = take_nth(x(i),n);
            end
        else
            % single structure
            names = fieldnames(x);
            for i=1:size(names,1)
                value = getfield(x,names{i});
                if length(value) > 1
                    x = setfield(x,names{i},take_nth(value,nth));
                elseif iscell(value)
                    x = setfield(x,names{i},{take_nth(value{1},nth)});
                end
            end
        end
    elseif iscell(x)
        % cell array
        for i=1:(m*n)
            x{i} = take_nth(x{i},nth);
        end
    elseif m > 1
        x = x(nth,:);
    end
