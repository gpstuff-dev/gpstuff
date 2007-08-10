function [Y, VarY] = gp_fwds(gp, tx, ty, x, varargin)
%GP_FWDS	Forward propagation through Gaussian Processes.
%
%	Description
%	Y = GP_FWDS(GP, TX, TY, X) takes a Gaussian processes data structure GP
%       together with a matrix X of input vectors, matrix TX of training inputs 
%       and vector TY of training targets, and forward propagates the inputs 
%       through the Gaussian processes to generate a vector Y of outputs.
%       TX is of size MxN and TY is size Mx1. Each row of X corresponds to one
%       input vector and each row of Y corresponds to one output vector. 
%       
%	Y = GP_FWD(GP, TX, TY, X, U) in case of sparse model takes also inducing 
%       points U.
%
%       LATENT VALUES
%       TY can be also matrix of latent values in which case it is of size MxNMC. 
%       In this case TY is handled as all the other sampled parameters.
%
%	[Y, CovY] = GP_FWDS(GP, TX, TY, X) returns also covariance of Y.
%
%       BUGS: - only 1 output allowed
%
%	See also
%	GP, GP_PAK, GP_UNPAK, GP_FWD

% Copyright (c) 2006      Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.


if nargin < 4
  error('Requires at least 4 arguments');
end

nin  = gp.nin;
nout = gp.nout;

nmc=size(gp.hmcrejects,1);
Y = zeros(size(x,1),nout,nmc);

% loop over all samples
for i1=1:nmc
    Gp = take_nth(gp,i1);
   
    switch gp.type
      case 'FULL'         % Do following if full model is used    
        [c,C]=gp_trcov(Gp, tx);
        K=gp_cov(Gp, tx, x);
    
        % This is used only in the case of latent values. 
        if size(ty,2)>1
            y=K'*(C\ty(:,i1));
        else    % Here latent values are not present
            y=K'*(C\ty);
        end
        
        if nargout>1
% $$$             [V, CC] = gp_trvar(Gp,x);
% $$$             % Vector of diagonal elements of covariance matrix
% $$$             L = chol(C)';
% $$$             b = L\K;
% $$$             VarY(:,:,i1) = V - sum(b.^2);
            
% $$$       [c,CC] = gp_trcov(Gp,x);
% $$$       VarY(:,:,i1) = CC - K'*(C\K);
        end
        Y(:,:,i1) = y;
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
        
        if size(ty,2)>1
            p = ty(:,i1)./Lav - iLaKfu*(A\(iLaKfu'*ty(:,i1))); 
            y = K_nu*(K_uu\(K_fu'*p));
            %            y = K_nu*(S*(K_fu'*(ty(:,i1)./Lav)));
            %  y=K_nu*S*K_uf*diag(1./La)*ty(:,i1);
        else    % Here latent values are not present
            p = ty./Lav - iLaKfu*(A\(iLaKfu'*ty)); 
            y = K_nu *(K_uu\(K_fu'*p));
            %y = K_nu*(iKuuKufiLa*ty + iKuuKufiLa*(K_fu*(Sinv\(K_fu'*(ty./Lav)))));
                %y = K_nu*(Sinv\(K_fu'*(ty./Lav)));
                %y=K_nu*Sinv*K_uf*diag(1./La)*ty;
        end
        if nargout > 1   % see Quinonera-Candela&Rasmussen (2005)
            error('Variance is not yet implemented for FIC! \n')
% $$$             B=Luu\(K_nu');
% $$$             Qv_nn=sum(B.^2)';
% $$$             % Vector of diagonal elements of covariance matrix
% $$$             L = chol(S)';
% $$$             b = L\K_nu';
% $$$             [Kv_nn, Cv_nn] = gp_trvar(Gp,x);
% $$$             VarY(:,:,i1) = Kv_nn - Qv_nn + sum(b.^2)';
        end
        Y(:,i1) = y;
        
      case 'PIC_BLOCK'        % Do following if FIC sparse model is used
        % Calculate some help matrices  
        u = reshape(Gp.X_u,length(Gp.X_u)/nin,nin);
        ind = varargin{1};           % block indeces for training points
        tstind = varargin{2};        % block indeces for test points
        
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
            error('Variance is not yet implemented for PIC! \n')
% $$$             B=Luu\(K_nu');
% $$$             Qv_nn=sum(B.^2)';
% $$$             % Vector of diagonal elements of covariance matrix
% $$$             L = chol(S)';
% $$$             b = L\K_nu';
% $$$             [Kv_nn, Cv_nn] = gp_trvar(Gp,x);
% $$$             VarY(:,:,i1) = Kv_nn - Qv_nn + sum(b.^2)';
        end
        
        Y(:,:,i1) = y;
      case 'CS+PIC'
        % Calculate some help matrices  
        u = reshape(Gp.X_u,length(Gp.X_u)/nin,nin);
        ind = varargin{1};           % block indeces for training points
        tstind = varargin{2};        % block indeces for test points
        
        n = size(tx,1);
        
        cf_orig = Gp.cf;
        
        cf1 = {};
        cf2 = {};
        j = 1;
        k = 1;
        for i = 1:length(cf_orig) 
            if ~isfield(Gp.cf{i},'cs')
                cf1{j} = Gp.cf{i};
                j = j + 1;
            else
                cf2{k} = Gp.cf{i};
                k = k + 1;
            end         
        end

        Gp.cf = cf1;
        K_nu = gp_cov(Gp, x, u);            % n x u   
        [Kv_ff, Cv_ff] = gp_trvar(Gp, tx);  % 1 x f  vector
        K_fu = gp_cov(Gp, tx, u);           % f x u
        K_uu = gp_trcov(Gp, u);             % u x u, noiseles covariance K_uu
        Luu = chol(K_uu)';
        %K_nf = gp_cov(Gp,x,tx);

        % Evaluate the Lambda (La) for specific model
        % Q_ff = K_fu*inv(K_uu)*K_fu'
        B=Luu\K_fu';
% $$$         La = sparse(1:n,1:n,0,n,n);
% $$$         for i=1:length(ind)
% $$$             Qbl_ff = B(:,ind{i})'*B(:,ind{i});
% $$$             [Kbl_ff, Cbl_ff] = gp_trcov(Gp, tx(ind{i},:));
% $$$             La(ind{i},ind{i}) =  Cbl_ff - Qbl_ff;
% $$$         end
        
        [I,J]=find(tril(sparse(gp.tr_indvec(:,1),gp.tr_indvec(:,2),1,n,n),-1));
        q_ff = sum(B(:,I).*B(:,J));
        q_ff = sparse(I,J,q_ff,n,n);
        c_ff = gp_covvec(gp, x(I,:), x(J,:))';
        c_ff = sparse(I,J,c_ff,n,n);
        [Kv_ff, Cv_ff] = gp_trvar(gp,x);
        La = c_ff + c_ff' - q_ff - q_ff' + sparse(1:n,1:n, Cv_ff-sum(B.^2,1)',n,n);
        
        
        % Add the compact support cf to lambda
        Gp.cf = cf2;
        K_cs = gp_trcov(Gp,tx);
        K_cs_nf = gp_cov(Gp,x,tx);
        %K_cs_nf = gp_cov(Gp,x,tx);
        La = La + K_cs;
        Gp.cf = cf_orig;

        iLaKfu = La\K_fu;
        A = K_uu+K_fu'*iLaKfu;
        A = (A+A')./2;            % Ensure symmetry
        
        %L = iLaKfu/chol(A);        
        %K_ff = gp_trcov(Gp,tx);
        %iKff = inv(La)-L*L';
        
        %Q_nf = K_nf*iKff*K_ff;        

        Gp.cf = cf_orig;
        
        if size(ty,2)>1
            tyy = ty(:,i1);
        else    % Here latent values are not present
            tyy = ty;
        end

        
        
        % From this on evaluate the prediction
        % See Snelson and Ghahramani (2007) for details 
        p=iLaKfu*(A\(iLaKfu'*tyy));
% $$$         for i=1:length(ind)
% $$$             p2(ind{i},:) = La(ind{i},ind{i})\tyy(ind{i},:);
% $$$         end
        p2 = La\tyy;
        p= p2-p;
        
        %iKuuKuf = K_uu\K_fu';
        %w_u = K_uu\(K_fu'*p);
        Gp.cf = cf1;
                
        w_bu=zeros(length(x),length(u));
        %w_bu=zeros(length(x),1);
        w_n=zeros(length(x),1);
        for i=1:length(ind)
            w_bu(tstind{i},:) = repmat((K_uu\(K_fu(ind{i},:)'*p(ind{i},:)))', length(tstind{i}),1);
            %w_bu(tstind{i},:) = repmat(K_uu\(K_fu(ind{i},:)'*p(ind{i},:)))', length(tstind{i}),1);
            %w_bu = w_bu + K_nu*(K_uu\(K_fu(ind{i},:)'))*p(ind{i});
            K_nb = gp_cov(Gp, x(tstind{i},:), tx(ind{i},:));              % n x u
            w_n(tstind{i},:) = K_nb*p(ind{i},:);
        end
        %    [max(- sum(K_nu.*w_bu,2) + w_n), mean(- sum(K_nu.*w_bu,2) + w_n), min(- sum(K_nu.*w_bu,2) + w_n)]
        %y = K_nu*w_u - sum(K_nu.*w_bu,2) + w_n;
        %y = (K_nu*(K_uu\K_fu')+K_cs_nf)*p - w_bu + w_n;
        y = (K_nu*(K_uu\K_fu')+K_cs_nf)*p - sum(K_nu.*w_bu,2) + w_n;
        Gp.cf = cf_orig;
        if nargout > 1   
            error('Variance is not yet implemented for PIC! \n')
% $$$             B=Luu\(K_nu');
% $$$             Qv_nn=sum(B.^2)';
% $$$             % Vector of diagonal elements of covariance matrix
% $$$             L = chol(S)';
% $$$             b = L\K_nu';
% $$$             [Kv_nn, Cv_nn] = gp_trvar(Gp,x);
% $$$             VarY(:,:,i1) = Kv_nn - Qv_nn + sum(b.^2)';
        end
        
        Y(:,:,i1) = y;
    end
end

function x = take_nth(x,nth)
%TAKE_NTH    Take n'th parameters from MCMC-chains
%
%   x = take_nth(x,n) returns chain containing only
%   n'th simulation sample 
%
%   See also
%     THIN, JOIN

% Copyright (c) 1999 Simo Särkkä
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
