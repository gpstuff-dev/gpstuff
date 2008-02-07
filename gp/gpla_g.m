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

        [e, edata, eprior, f, L, La1, b] = gpla_e(gp_pak(gp, param), gp, x, y, param, varargin{:});

        K_fu = gp_cov(gp, x, u);         % f x u
        K_uu = gp_trcov(gp, u);          % u x u, noiseles covariance K_uu
        K_uu = (K_uu+K_uu')./2;          % ensure the symmetry of K_uu
        iKuuKuf = K_uu\K_fu';

        W = hessian(f, gp.likelih);
        sqrtW = sqrt(W);
        b = f'./La1' - (f'*L)*L';

        La = W.*La1;
        Lahat = 1 + La;
        La2 = Lahat;
        La3 = 1./La1 + W;
        B2 = (repmat(sqrtW,1,m).*K_fu);

        % Components for
        B3 = repmat(Lahat,1,m).\B2;
        A2 = K_uu + B2'*B3; A2=(A2+A2)/2;
        L2 = B3/chol(A2);

        % Evaluate diag(La3 - L3'*L3)
        B4 = repmat(La3,1,m).\L;
        A3 = eye(size(K_uu)) - L'*B4; A3 = (A3+A3')./2;
        L3 = B4/chol(A3);
        dA3L3tL3 = 1./La3' + sum(L3.*L3,2)';
        dA3L3tL3 = dA3L3tL3.*thirdgrad(f, gp.likelih)';

        KufW = K_fu'.*repmat(W',m,1);
        iLa2Kfu = repmat(La2,1,m).\K_fu;
        A4 = K_uu + KufW*iLa2Kfu; A4 = (A4+A4')./2;
        L4 = iLa2Kfu/chol(A4);
        L5 = chol(A4)'\(KufW./repmat(La2',m,1));

        % Set the parameters for the actual gradient evaluation
        b2 = (dA3L3tL3./La2' - dA3L3tL3*L4*L5);
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

        [e, edata, eprior, f, L, La1, b] = gpla_e(gp_pak(gp, param), gp, x, y, param, varargin{:});

        K_fu = gp_cov(gp, x, u);         % f x u
        K_uu = gp_trcov(gp, u);          % u x u, noiseles covariance K_uu
        K_uu = (K_uu+K_uu')./2;          % ensure the symmetry of K_uu
        iKuuKuf = K_uu\K_fu';

        W = hessian(f, gp.likelih);
        sqrtW = sqrt(W);
        fiLa = zeros(size(f'));
        for i=1:length(ind)
            fiLa(ind{i}) = f(ind{i})'/La1{i};
            La{i} = diag(sqrtW(ind{i}))*La1{i}*diag(sqrtW(ind{i}));
            Lahat{i} = eye(size(La{i})) + La{i};
            La2{i} = eye(size(La1{i})) + La1{i}*diag(W(ind{i}));
            La3{i} = inv(La1{i}) + diag(W(ind{i}));
        end
        b = fiLa - (f'*L)*L';
        B2 = (repmat(sqrtW,1,m).*K_fu);

        % Components for
        B3 = zeros(size(K_fu));
        B4 = zeros(size(L));
        diLa3 = zeros(1,n);
        for i=1:length(ind)
            B3(ind{i},:) = Lahat{i}\B2(ind{i},:);
            B4(ind{i},:) = La3{i}\L(ind{i},:);
            diLa3(ind{i}) = diag(inv(La3{i}));
        end
        A2 = K_uu + B2'*B3; A2=(A2+A2)/2;
        L2 = B3/chol(A2);

        % Evaluate diag(La3 - L3'*L3)
        A3 = eye(size(K_uu)) - L'*B4; A3 = (A3+A3')./2;
        L3 = B4/chol(A3);
        dA3L3tL3 = diLa3 + sum(L3.*L3,2)';
        dA3L3tL3 = dA3L3tL3.*thirdgrad(f, gp.likelih)';

        KufW = K_fu'.*repmat(W',m,1);
        iLa2Kfu = zeros(size(K_fu));
        KufWiLa2 = zeros(size(K_fu'));
        for i=1:length(ind)
            iLa2Kfu(ind{i},:) = La2{i}\K_fu(ind{i},:);
            KufWiLa2(:,ind{i}) = KufW(:,ind{i})/La2{i};
        end
        A4 = K_uu + KufW*iLa2Kfu; A4 = (A4+A4')./2;
        L4 = iLa2Kfu/chol(A4);
        L5 = chol(A4)'\KufWiLa2;

        % Set the parameters for the actual gradient evaluation

        b3 = derivative(f, gp.likelih);
        L = repmat(sqrtW,1,m).*L2;
        b2 = zeros(1,n);
        for i=1:length(ind)
            La{i} = diag(sqrtW(ind{i}))\Lahat{i}/diag(sqrtW(ind{i}));
            b2(ind{i}) = dA3L3tL3(ind{i})/La2{i};
        end
        b2 = (b2 - dA3L3tL3*L4*L5);
    case 'CS+FIC'
        g_ind = zeros(1,numel(gp.X_u));
        gdata_ind = zeros(1,numel(gp.X_u));
        gprior_ind = zeros(1,numel(gp.X_u));

        u = gp.X_u;
        m = size(u,1);

        [e, edata, eprior, f, L, La1, b] = gpla_e(gp_pak(gp, param), gp, x, y, param, varargin{:});

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
        gp.cf = cf_orig;

        iKuuKuf = K_uu\K_fu';

        W = hessian(f, gp.likelih);
        sqrtW = sqrt(W);
        W = sparse(1:n,1:n,W,n,n);
        sqrtW = sparse(1:n,1:n,sqrtW,n,n);
        b = f'/La1 - (f'*L)*L';

        La = sqrtW*La1*sqrtW;
        Lahat = sparse(1:n,1:n,1,n,n) + La;
        La2 = sparse(1:n,1:n,1,n,n) + La1*W;
        %La3 = inv(La1) + W;
        La3 = (sqrtW\Lahat)*sqrtW;
        B2 = sqrtW*K_fu;

        % Components for
        B3 = Lahat\B2;
        A2 = K_uu + B2'*B3; A2=(A2+A2)/2;
        L2 = B3/chol(A2);

        % Evaluate diag(La3 - L3'*L3)
        %B4 = La3\L;
        B4 = La1*(La3\L);
        A3 = eye(size(K_uu)) - L'*B4; A3 = (A3+A3')./2;
        L3 = B4/chol(A3);
        dA3L3tL3 = diag(La1/La3)' + sum(L3.*L3,2)';
        dA3L3tL3 = dA3L3tL3.*thirdgrad(f, gp.likelih)';

        KufW = K_fu'*W;
        iLa2Kfu = La2\K_fu;
        A4 = K_uu + KufW*iLa2Kfu; A4 = (A4+A4')./2;
        L4 = iLa2Kfu/chol(A4);
        L5 = chol(A4)'\(KufW/La2);

        % Set the parameters for the actual gradient evaluation
        b2 = (dA3L3tL3/La2 - dA3L3tL3*L4*L5);
        b3 = derivative(f, gp.likelih);
        L = sqrtW*L2;
        La = (sqrtW\Lahat)/sqrtW;
        
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
