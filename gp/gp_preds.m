function [Y, VarY] = gp_preds(gp, tx, ty, x, varargin)
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

    nin  = gp.nin;
    nout = gp.nout;

    nmc=size(gp.hmcrejects,1);
    Y = zeros(size(x,1),nmc);

    if strcmp(gp.type, 'PIC_BLOCK')
        ind = gp.tr_index;           % block indeces for training points
        gp = rmfield(gp,'tr_index');
    end

    % loop over all samples
    for i1=1:nmc
        Gp = take_nth(gp,i1);
        
        switch gp.type            
            % --------------------------------------------
            %  FULL GP                                   
            % --------------------------------------------
          case 'FULL'         % Do following if full model is used    
            [c,C]=gp_trcov(Gp, tx);
            K=gp_cov(Gp, tx, x);
            L = chol(C)';
            
            % This is used only in the case of latent values. 
            if size(ty,2)>1
                if issparse(C)
                    LD = ldlchol(C);
                    y = K'*ldlsolve(LD,ty(:,i1));
                else
                    a = L'\(L\ty(:,i1));
                    y = K'*a;
                end
            else    % Here latent values are not present
                if issparse(C)
                    LD = ldlchol(C);
                    y = K'*ldlsolve(LD,ty);
                else
                    a = L'\(L\ty);
                    y = K'*a;
                end
            end
            
            if nargout>1
                if issparse(C)
                    V = gp_trvar(Gp,x);
                    VarY(:,i1) = V - diag(K'*ldlsolve(LD,K));
                else
                    v = L\K;
                    V = gp_trvar(Gp,x);
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
                            % Calculate some help matrices  
            u = reshape(Gp.X_u,length(Gp.X_u)/nin,nin);
            [Kv_ff, Cv_ff] = gp_trvar(Gp, tx);  % 1 x f  vector
            K_fu = gp_cov(Gp, tx, u);         % f x u
            K_nu = gp_cov(Gp, x, u);         % n x u
            K_uu = gp_trcov(Gp, u);    % u x u, noiseles covariance K_uu
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
            
            if size(ty,2)>1
                p = ty(:,i1)./Lav - L*(L'*ty(:,i1)); 
                y = K_nu*(K_uu\(K_fu'*p));
                %            y = K_nu*(S*(K_fu'*(ty(:,i1)./Lav)));
                %  y=K_nu*S*K_uf*diag(1./La)*ty(:,i1);
            else    % Here latent values are not present
                p = ty./Lav - L*(L'*ty); 
                y = K_nu *(K_uu\(K_fu'*p));
                %y = K_nu*(iKuuKufiLa*ty + iKuuKufiLa*(K_fu*(Sinv\(K_fu'*(ty./Lav)))));
                %y = K_nu*(Sinv\(K_fu'*(ty./Lav)));
                %y=K_nu*Sinv*K_uf*diag(1./La)*ty;
            end
            if nargout > 1   % see Quinonera-Candela&Rasmussen (2005)
                Knn_v = gp_trvar(Gp,x);
                B2=Luu\(K_nu');        
                VarY(:,i1) = Knn_v - sum(B2'.*(B*(repmat(Lav,1,size(K_uu,1)).\B')*B2)',2)  + sum((K_nu*(K_uu\(K_fu'*L))).^2, 2);
            end
            Y(:,i1) = y;
            % --------------------------------------------
            %  PIC                                   
            % --------------------------------------------            
          case 'PIC_BLOCK'        % Do following if FIC sparse model is used
                                  % Calculate some help matrices  
            u = reshape(Gp.X_u,length(Gp.X_u)/nin,nin);
            tstind = varargin{1};        % block indeces for test points
            
            [Kv_ff, Cv_ff] = gp_trvar(Gp, tx);  % 1 x f  vector
            K_fu = gp_cov(Gp, tx, u);         % f x u
            K_nu = gp_cov(Gp, x, u);         % n x u   
            K_uu = gp_trcov(Gp, u);    % u x u, noiseles covariance K_uu
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
            
            %iKuuKuf = K_uu\K_fu';
            w_u = K_uu\(K_fu'*p);
            
            w_bu=zeros(length(x),length(u));
            w_n=zeros(length(x),1);
            for i=1:length(ind)
                w_bu(tstind{i},:) = repmat((K_uu\(K_fu(ind{i},:)'*p(ind{i},:)))', length(tstind{i}),1);
                K_nb = gp_cov(Gp, x(tstind{i},:), tx(ind{i},:));              % n x u
                w_n(tstind{i},:) = K_nb*p(ind{i},:);
            end
            %    [max(- sum(K_nu.*w_bu,2) + w_n), mean(- sum(K_nu.*w_bu,2) + w_n), min(- sum(K_nu.*w_bu,2) + w_n)]
            y = K_nu*w_u - sum(K_nu.*w_bu,2) + w_n;
            
            if nargout > 1
                kstarstar = gp_trvar(Gp, x);
                iKuuKuf = K_uu\K_fu';
                KnfL = K_nu*(iKuuKuf*L);
                Varf = zeros(length(x),1);
                for i=1:length(ind)
                    v_n = gp_cov(Gp, x(tstind{i},:), tx(ind{i},:));              % n x u
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
            u = reshape(Gp.X_u,length(Gp.X_u)/nin,nin);
            ncf = length(Gp.cf);
            cf_orig = Gp.cf;
            
            cf1 = {};
            cf2 = {};
            j = 1;
            k = 1;
            for i = 1:ncf
                if ~isfield(Gp.cf{i},'cs')
                    cf1{j} = Gp.cf{i};
                    j = j + 1;
                else
                    cf2{k} = Gp.cf{i};
                    k = k + 1;
                end
            end
            Gp.cf = cf1;

            % First evaluate needed covariance matrices
            % v defines that parameter is a vector
            [Kv_ff, Cv_ff] = gp_trvar(Gp, tx);  % f x 1  vector
            K_fu = gp_cov(Gp, tx, u);         % f x u
            K_uu = gp_trcov(Gp, u);    % u x u, noiseles covariance K_uu
            K_uu = (K_uu+K_uu')./2;     % ensure the symmetry of K_uu
            Luu = chol(K_uu)';
            K_nu = gp_cov(Gp, x, u);         % n x u

            % Evaluate the Lambda (La)
            % Q_ff = K_fu*inv(K_uu)*K_fu'
            B=Luu\(K_fu');       % u x f
            Qv_ff=sum(B.^2)';
            Lav = Cv_ff-Qv_ff;   % f x 1, Vector of diagonal elements

            Gp.cf = cf2;
            K_cs = gp_trcov(Gp,tx);
            Kcs_nf = gp_cov(Gp, x, tx);
            La = sparse(1:tn,1:tn,Lav,tn,tn) + K_cs;
            Gp.cf = cf_orig;

            iLaKfu = La\K_fu;
            A = K_uu+K_fu'*iLaKfu;
            A = (A+A')./2;     % Ensure symmetry
            L = iLaKfu/chol(A);
            
            if size(ty,2)>1
                p = La\ty(:,i1) - L*(L'*ty(:,i1));
            else
                p = La\ty - L*(L'*ty);
            end

            %p2 = ty./Lav - iLaKfu*(A\(iLaKfu'*ty));
            %    Knf = K_nu*(K_uu\K_fu');
            y = K_nu*(K_uu\(K_fu'*p)) + Kcs_nf*p;
            
            if nargout > 1
                Knn_v = gp_trvar(Gp,x);
                B2=Luu\(K_nu');
                VarY = Knn_v - sum(B2'.*(B*(La\B')*B2)',2)  + sum((K_nu*(K_uu\(K_fu'*L))).^2, 2) - sum((Kcs_nf/chol(La)).^2,2) + sum((Kcs_nf*L).^2, 2);
                %VarY = VarY - 2.*diag((Kcs_nf*iLaKfu)*(K_uu\K_nu')) + 2.*diag((Kcs_nf*L)*(L'*K_fu*(K_uu\K_nu')));
                VarY = VarY - 2.*sum((Kcs_nf*iLaKfu).*(K_uu\K_nu')',2) + 2.*sum((Kcs_nf*L).*(L'*K_fu*(K_uu\K_nu'))' ,2);
                
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
