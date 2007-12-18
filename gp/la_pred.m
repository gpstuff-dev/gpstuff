function [Ef, Varf, p1] = la_pred(gp, tx, ty, x, varargin)
%EP_PRED	Predictions with Gaussian Process EP
%
%	Description
%	Y = EP_PRED(GP, TX, TY, X) takes a gp data structure GP together with a
%	matrix X of input vectors, Matrix TX of training inputs and vector TY of 
%       training targets, and forward propagates the inputs through the gp to generate 
%       a matrix Y of (noiseless) output vectors (mean(Y|X)). Each row of X 
%       corresponds to one input vector and each row of Y corresponds to one output 
%       vector.
%
%	Y = EP_PRED(GP, TX, TY, X, U) in case of sparse model takes also inducing 
%       points U.
%
%	[Y, VarY] = EP_PRED(GP, TX, TY, X) returns also the variances of Y 
%       (1xn vector).
%
%       BUGS: - only 1 output allowed
%
%	See also
%	GP, GP_PAK, GP_UNPAK
%
% Copyright (c) 2006 Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

    [tn, tnin] = size(tx);
    
    switch gp.type
      case 'FULL'        
        [e, edata, eprior, f, L, La2, b] = gpla_e(gp_pak(gp,'hyper'), gp, tx, ty, 'hyper', varargin{:});

        W = La2;
        deriv = b;
        ntest=size(x,1);
        
        % Evaluate the expectation        
        K_nf = gp_cov(gp,x,tx);
        Ef = K_nf*deriv;

        % Evaluate the variance
        if nargout > 1
            kstarstar = gp_trvar(gp,x);
            V = L\(sqrt(W)*K_nf');
            for i1=1:ntest
                Varf(i1,1)=kstarstar(i1)-V(:,i1)'*V(:,i1);
                switch gp.likelih
                  case 'probit'
                    p1(i1,1)=normcdf(Ef(i1,1)/sqrt(1+Varf(i1))); % Probability p(y_new=1)
                  case 'poisson'
                    p1 = NaN;
                end
            end
        end
        
      case 'FIC'
        u = gp.X_u;
        K_fu = gp_cov(gp, tx, u);         % f x u
        K_uu = gp_trcov(gp, u);          % u x u, noiseles covariance K_uu
        K_uu = (K_uu+K_uu')./2;          % ensure the symmetry of K_uu
        
        m = size(u,1);
        
        if length(varargin) < 1
            error('The argument telling the optimzed/sampled parameters has to be provided.') 
        end

        [e, edata, eprior, f, L, La2, b] = gpla_e(gp_pak(gp,'hyper'), gp, tx, ty, 'hyper', varargin{:});
        
        deriv = b;
        ntest=size(x,1);
        
        ntest=size(x,1);
                
        K_nu=gp_cov(gp,x,u);
        % Knf = K_nu*(K_uu\K_fu'); 
        % Ef = Knf*p;
        Ef = K_nu*(K_uu\(K_fu'*deriv));
                
        % Evaluate the variance
        if nargout > 1
            W = hessian(f, gp.likelih);
            kstarstar = gp_trvar(gp,x);
            Luu = chol(K_uu)';
            La = W.*La2;
            Lahat = 1 + La;
            B = (repmat(sqrt(W),1,m).*K_fu);
            
            % Components for (I + W^(1/2)*(Qff + La2)*W^(1/2))^(-1) = Lahat^(-1) - L2*L2'
            B2 = repmat(Lahat,1,m).\B;
            A2 = K_uu + B'*B2; A2=(A2+A2)/2;
            L2 = B2/chol(A2);             
            
            % Set params for K_nf
            BB=Luu\(B');
            BB2=Luu\(K_nu');
            Varf = kstarstar - sum(BB2'.*(BB*(repmat(Lahat,1,size(K_uu,1)).\BB')*BB2)',2)  + sum((K_nu*(K_uu\(B'*L2))).^2, 2);
            for i1=1:ntest
                switch gp.likelih
                  case 'probit'
                    p1(i1,1)=normcdf(Ef(i1,1)/sqrt(1+Varf(i1))); % Probability p(y_new=1)
                  case 'poisson'
                    p1 = NaN;
                end
            end
        end
        
      case 'PIC_BLOCK'
        
    end
%
% ==============================================================
% Begin of the nested functions
% ==============================================================
%    
    function Hessian = hessian(f, likelihood)
        switch likelihood
          case 'probit'
            z = ty.*f;
            Hessian = (normpdf(f)./normcdf(z)).^2 + z.*normpdf(f)./normcdf(z);
          case 'poisson'
            Hessian = gp.avgE.*exp(f);
        end
    end
end