function [g, gdata, gprior] = gpla_g(w, gp, x, y, param, varargin)
%GP_G   Evaluate gradient of error for Gaussian Process.
%
%	Description
%	G = GPEP_G(W, GP, X, Y) takes a full GP hyper-parameter vector W, 
%       data structure GP a matrix X of input vectors and a matrix Y
%       of target vectors, and evaluates the error gradient G. Each row of X
%	corresponds to one input vector and each row of Y corresponds
%       to one target vector. Works only for full GP.
%
%	G = GPEP_G(W, GP, P, Y, PARAM) in case of sparse model takes also  
%       string PARAM defining the parameters to take the gradients with 
%       respect to. Possible parameters are 'hyper' = hyperparameters and 
%      'inducing' = inducing inputs, 'all' = all parameters.
%
%	[G, GDATA, GPRIOR] = GP_G(GP, X, Y) also returns separately  the
%	data and prior contributions to the gradient.
%
%	See also   
%

% Copyright (c) 2007      Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.
        
    gp=gp_unpak(gp, w, param);       % unpak the parameters
    ncf = length(gp.cf);
    n=size(x,1);
    
    g = [];
    gdata = [];
    gprior = [];
    
    % First Evaluate the data contribution to the error    
    switch gp.type
        % ============================================================
        % FULL
        % ============================================================
      case 'FULL'   % A full GP
        % Calculate covariance matrix and the site parameters
        K = gp_trcov(gp,x);
        [e, edata, eprior, f, L, La2, b] = gpla_e(gp_pak(gp, param), gp, x, y, param, varargin{:});

        W = La2;
        der_f = b;
        ntest=size(x,1);
        
        I = eye(size(K));
        sqrtW = sqrt(W);
        C = sqrtW*K;
        Z = (L\sqrtW);
        Z = Z'*Z;          %Z = sqrtW*((I + C*sqrtW)\sqrtW);

        CC = C*diag(thirdgrad(f, gp.likelih)./diag(sqrtW));
        s2 = -0.5*diag(L'\(L\(CC + CC')));       %s2 = -0.5*diag((I + C*sqrtW)\(CC + CC'));
                
        b = K\f;
        B = eye(size(K)) + K*W;
        invC = Z + der_f*(s2'/B);
               
        % Evaluate the gradients from covariance functions
        for i=1:ncf
            gpcf = gp.cf{i};
            gpcf.type = gp.type;
            [g, gdata, gprior] = feval(gpcf.fh_ghyper, gpcf, x, y, g, gdata, gprior, invC, b);
        end
        
        % Evaluate the gradient from noise functions
        if isfield(gp, 'noise')
            nn = length(gp.noise);
            for i=1:nn
                noise = gp.noise{i};
                noise.type = gp.type;
                [g, gdata, gprior] = feval(noise.fh_ghyper, noise, x, y, g, gdata, gprior, invC, B);
            end
        end
        % Do not go further
        return;
        % ============================================================
        % FIC
        % ============================================================
      case 'FIC'
        g_ind = zeros(1,numel(gp.X_u));
        gdata_ind = zeros(1,numel(gp.X_u));
        gprior_ind = zeros(1,numel(gp.X_u));
        
        u = gp.X_u;
        m = size(u,1);

        [e, edata, eprior, f, L, La2, b] = gpla_e(gp_pak(gp, param), gp, x, y, param, varargin{:});

        K_fu = gp_cov(gp, x, u);         % f x u
        K_uu = gp_trcov(gp, u);          % u x u, noiseles covariance K_uu
        K_uu = (K_uu+K_uu')./2;          % ensure the symmetry of K_uu        
        iKuuKuf = K_uu\K_fu';
        
        W = hessian(f, gp.likelih);
        sqrtW = sqrt(W);
        b = f'./La2' - (f'*L)*L';
        
        La = W.*La2;
        Lahat = 1 + La;
        B = (repmat(sqrtW,1,m).*K_fu);
        
        % Components for (I + W^(1/2)*(Qff + La2)*W^(1/2))^(-1) = Lahat^(-1) - L2*L2'
        B2 = repmat(Lahat,1,m).\B;
        A2 = K_uu + B'*B2; A2=(A2+A2)/2;
        L2 = B2/chol(A2);                
        
        % components for (I+(Qff+La2)*W))^(-1) = Lahat^(-1) - L3a*L3b
        B3 = repmat(Lahat,1,m).\K_fu;
        L3b = (K_fu'.*repmat(W',m,1))./repmat(Lahat',m,1);
        A3 = K_uu + L3b*K_fu; A3=(A3+A3)/2;
        A3 = chol(A3);
        L3a = B3/A3;
        L3b = A3'\L3b;
        
        % Components for  W^(1/2)*(Qff + La2)*W^(1/2)*dW/dth
        %              =  La + B*K_uu^(-1)*B'*thirdg_f 
        %              =  La + B*K_uu^(-1)*C 
        % aim is to evaluate 
        % s2 = 0.5*diag((Lahat^(-1) - L2*L2' )*(La + B*K_uu\C + (La + B*K_uu\C)' )*(Lahat^(-1) - L3a*L3b))';
        C = B'.*repmat(thirdgrad(f, gp.likelih)',m,1);

% $$$         s2 = - 0.5*diag((diag(1./Lahat) - L2*L2' )*(diag(La) + B*(K_uu\C) + (diag(La) + B*(K_uu\C))' )*(diag(1./Lahat) - L3a*L3b))';

        LaLa = (Lahat.\La);
        LaB = repmat(Lahat,1,m).\B;
        KuuC = K_uu\C;
        L2LaLa = L2'.*repmat(LaLa',m,1);
        L2BKuuC = L2'*B*KuuC;
        
        
        s2 = (LaLa./Lahat...
              - sum((repmat(LaLa,1,m).*L3a).*L3b',2) ...
              + sum(LaB.*(KuuC./repmat(Lahat',m,1))',2) ...
              - sum(LaB.*((KuuC*L3a)*L3b)',2)) ...
             - (...
                 sum(L2.*L2LaLa',2)...
                 - sum(L2.*(((L2'.*repmat(La',m,1))*L3a)*L3b)', 2) ...
                 + sum(L2.*L2BKuuC',2)./Lahat...
                 - sum(L2.*((L2BKuuC*L3a)*L3b)',2));
        s2 = s2';

        
      
        % Set the parameters for the actual gradient evaluation
        b2 = s2; 
        b3 = derivative(f, gp.likelih);
        L = repmat(sqrtW,1,m).*L2;
        La = Lahat./W;
        
        % ============================================================
        % PIC
        % ============================================================
      case 'PIC_BLOCK'
        g_ind = zeros(1,numel(gp.X_u));
        gdata_ind = zeros(1,numel(gp.X_u));
        gprior_ind = zeros(1,numel(gp.X_u));
        
        u = gp.X_u;
        m = size(u,1);
        ind = gp.tr_index;

        [e, edata, eprior, f, L, La2, b] = gpla_e(gp_pak(gp, param), gp, x, y, param, varargin{:});

        K_fu = gp_cov(gp, x, u);         % f x u
        K_uu = gp_trcov(gp, u);          % u x u, noiseles covariance K_uu
        K_uu = (K_uu+K_uu')./2;          % ensure the symmetry of K_uu        
        iKuuKuf = K_uu\K_fu';
        
        W = hessian(f, gp.likelih);
        sqrtW = sqrt(W);
        fiLa = zeros(size(f'));
        for i=1:length(ind)
            fiLa(ind{i}) = f(ind{i})'/La2{i};
            La{i} = diag(sqrtW(ind{i}))*La2{i}*diag(sqrtW(ind{i}));
            Lahat{i} = eye(size(La{i})) + La{i};
            Lahat2{i} = eye(size(La{i})) + La2{i}*diag(W(ind{i}));
        end
        b = fiLa - (f'*L)*L';
        B = (repmat(sqrtW,1,m).*K_fu);
        
        % Components for (I + W^(1/2)*(Qff + La2)*W^(1/2))^(-1) = Lahat^(-1) - L2*L2'
        for i=1:length(ind)
            B2(ind{i},:) = Lahat{i}\B(ind{i},:);
        end
        A2 = K_uu + B'*B2; A2=(A2+A2)/2;
        L2 = B2/chol(A2);
        
        % components for (I+(Qff+La2)*W))^(-1) = Lahat^(-1) - L3a*L3b
        L3b = K_fu'.*repmat(W',m,1);
        for i=1:length(ind)
            B3(ind{i},:) = Lahat{i}\K_fu(ind{i},:);
            L3b(:,ind{i}) = L3b(:,ind{i})/Lahat2{i};
        end
        A3 = K_uu + L3b*K_fu; A3=(A3+A3)/2;
        A3 = chol(A3);
        L3a = B3/A3;
        L3b = A3'\L3b;
                                
        % Components for  W^(1/2)*(Qff + La2)*W^(1/2)*dW/dth
        %              =  La + B*K_uu^(-1)*B'*thirdg_f 
        %              =  La + B*K_uu^(-1)*C 
        % aim is to evaluate 
        % s2 = 0.5*diag((Lahat^(-1) - L2*L2' )*(La + B*K_uu\C + (La + B*K_uu\C)' )*(Lahat^(-1) - L3a*L3b))';
        C = B'.*repmat(thirdgrad(f, gp.likelih)',m,1);

% $$$         s2 = - 0.5*diag((Lahat^(-1) - L2*L2' )*(La + B*(K_uu\C) + (La + B*(K_uu\C))' )*(Lahat^(-1) - L3a*L3b))';

        KuuC = K_uu\C;
        L2BKuuC = L2'*B*KuuC;
        LaLaLahat = zeros(n,1);
        for i=1:length(ind)
            LaLa{i} = Lahat{i}\La{i};
            LaLaLahat(ind{i}) = diag(LaLa{i}/Lahat{i});
            LaB(ind{i},:) = Lahat{i}\B(ind{i},:);
            L2LaLa(:,ind{i}) = L2(ind{i},:)'*LaLa{i}';
            LaLaL3a(ind{i},:) = LaLa{i}*L3a(ind{i},:);
            L2La(:,ind{i}) = L2(ind{i},:)'*La{i};
            KuuCiLahat(:,ind{i}) = KuuC(:,ind{i})/Lahat{i};
            L2BKuuCiLahat(:,ind{i}) = L2BKuuC(:,ind{i})/Lahat{i};
        end

        s2 = (LaLaLahat...
              - sum(LaLaL3a.*L3b',2) ...
              + sum(LaB.*KuuCiLahat',2) ...
              - sum(LaB.*((KuuC*L3a)*L3b)',2)) ...
             - (...
                 sum(L2.*L2LaLa',2)...
                 - sum(L2.*((L2La*L3a)*L3b)', 2) ...
                 + sum(L2.*L2BKuuCiLahat',2)...
                 - sum(L2.*((L2BKuuC*L3a)*L3b)',2));
        s2 = s2';
      
        % Set the parameters for the actual gradient evaluation
        b2 = s2; 
        b3 = derivative(f, gp.likelih);
        L = repmat(sqrtW,1,m).*L2;
        for i=1:length(ind)
            La{i} = La2{i} + diag(1./W(ind{i}));
        end
    end
    % =================================================================
    % Evaluate the gradients from covariance functions
    for i=1:ncf
        gpcf = gp.cf{i};
        gpcf.type = gp.type;
        if isfield(gp, 'X_u')
            gpcf.X_u = gp.X_u;
        end
        if isfield(gp, 'tr_index')
            gpcf.tr_index = gp.tr_index;
        end
        switch param
          case 'hyper'
            [g, gdata, gprior] = feval(gpcf.fh_ghyper, gpcf, x, y, g, gdata, gprior, L, b, iKuuKuf, La, b2, b3); %, L2, b2, Labl2
          case 'inducing'
            [g_ind, gdata_ind, gprior_ind] = feval(gpcf.fh_gind, gpcf, x, y, g_ind, gdata_ind, gprior_ind, L, b, iKuuKuf, La, b2, b3);
          case 'all'
            [g, gdata, gprior] = feval(gpcf.fh_ghyper, gpcf, x, y, g, gdata, gprior, L, b, iKuuKuf, La); %, L2, b2, Labl2
            [g_ind, gdata_ind, gprior_ind] = feval(gpcf.fh_gind, gpcf, x, y, g_ind, gdata_ind, gprior_ind, L, b, iKuuKuf, La, b2, b3);
          otherwise
            error('Unknown parameter to take the gradient with respect to! \n')
        end
    end
        
    % Evaluate the gradient from noise functions
    if isfield(gp, 'noise')
        nn = length(gp.noise);
        for i=1:nn            
            gpcf = gp.noise{i};
            gpcf.type = gp.type;
            if isfield(gp, 'X_u')
                gpcf.X_u = gp.X_u;
            end
            if isfield(gp, 'tr_index')
                gpcf.tr_index = gp.tr_index;
            end
            switch param
              case 'hyper'
                [g, gdata, gprior] = feval(gpcf.fh_ghyper, gpcf, x, y, g, gdata, gprior, L, b, iKuuKuf, La, b2, b3);
              case 'inducing'
                [g_ind, gdata_ind, gprior_ind] = feval(gpcf.fh_gind, gpcf, x, y, g_ind, gdata_ind, gprior_ind, L, b, iKuuKuf, La, b2, b3);
              case 'all'
                [g, gdata, gprior] = feval(gpcf.fh_ghyper, gpcf, x, y, g, gdata, gprior, L, b, iKuuKuf, La, b2, b3);
                [g_ind, gdata_ind, gprior_ind] = feval(gpcf.fh_gind, gpcf, x, y, g_ind, gdata_ind, gprior_ind, L, b, iKuuKuf, La, b2, b3);
            end
        end
    end
    switch param
      case 'inducing'
        % Evaluate here the gradient from prior
        g = g_ind;
      case 'all'
        % Evaluate here the gradient from prior
        g = [g g_ind];
    end
    % 
    %
% ==============================================================
% Begin of the nested functions
% ==============================================================
%
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
    function thir_grad = thirdgrad(f,likelihood)
        switch likelihood
          case 'probit'
            z2 = normpdf(f)./normcdf(y.*f);
            thir_grad = 2.*y.*z2.^3 + 3.*f.*z2.^2 - z2.*(y-y.*f.^2);
          case 'poisson'
            thir_grad = - gp.avgE.*exp(f);     
        end
    end
end
