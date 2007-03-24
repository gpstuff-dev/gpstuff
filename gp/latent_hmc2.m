function [z, energ, diagn] = latent_hmc2(f, z, opt, gradf, varargin)
% LATENT_HMC2     HMC sampler for latent values.
%
%
%
%

% Copyright (c) 2006      Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

% Set the state of HMC samler
if isfield(opt, 'rstate')
    if ~isempty(opt.rstate)
        latent_rstate = opt.latent_opt.rstate;
    end
else
    latent_rstate = sum(100*clock);
end

% Set the variables 
gp = varargin{1};
x = varargin{2}; 
y = varargin{3}; 
if nargin > 5
    u = varargin{4}; 
else
    u = [];
end
n=length(y);
L=[];
b=[];
Lav=[];
mincut = -300;
if isfield(gp,'avgE');
    E=gp.avgE(:);
else
    E=1;
end     

% Evaluate the help matrices for covariance matrix
if isfield(gp, 'sparse')
    getL(z, gp, x, y, u);
    % Rotate z towards prior as w = (L\z)';
    % Here we take an advantage of the fact that L = chol(diag(Lav)+b'b)
    % See cholrankup.m for general case of solving a Ax=b system
    w = (L\z)';
else
    getL(z, gp, x, y);
    % Rotate z towards prior
    w = (L\z)';
end

hmc2('state',latent_rstate)
rej = 0;
gradf = @lvpoisson_g2;
f = @lvpoisson_e2;
for li=1:opt.repeat 
    [w, energ, diagn] = hmc2(f, w, opt, gradf, gp, x, y, u, z);
    if li<opt.repeat/2
        if diagn.rej
            opt.stepadj=max(1e-5,opt.stepadj/1.4);
        else
            opt.stepadj=min(1,opt.stepadj*1.02);
        end
    end
    rej=rej+diagn.rej/opt.repeat;
    if isfield(diagn, 'opt')
        opt=diagn.opt;
    end
end
% Rotate w to z
z=L*w';
opt.latent_rstate = hmc2('state');
diagn.opt = opt;
diagn.rej = rej;
diagn.lvs = opt.stepadj;

    function [g, gdata, gprior] = lvpoisson_g2(w, gp, x, y, u, varargin)
    %LVPOISSON_G	Evaluate gradient function for GP latent values with
    %               Poisson likelihood
    
    % Copyright (c) 2006 Aki Vehtari
    
    % Force z and E to be a column vector
    w=w(:);

    z=L*w;
    z = max(z,mincut);
    mu = exp(z).*E;    
    if isfield(gp, 'sparse')
        gdata = (dtrmvt(L,(mu-y)))'; % (  (mu-y) )';
    else
        gdata = (dtrmvt(L,(mu-y)))';   % make the gradient a row vector
    end
    gprior = w';                   % make the gradient a row vector
    g = gdata+gprior;  
    end


    function [e, edata, eprior] = lvpoisson_e2(w, gp, x, t, u, varargin)
    %function [e, edata, eprior] = gp_e(w, gp, x, t, varargin)
    % LVPOISSON_E     Minus log likelihood function for spatial modelling.
    %
    %       E = LVPOISSON_E(X, GP, T, Z) takes.... and returns minus log from 
    
    % The field gp.avgE (if given) contains the information about averige
    % expected number of cases at certain location. The target, t, is 
    % distributed as t ~ poisson(avgE*exp(z))
    
    % Copyright (c) 2006 Aki Vehtari
    
    % This software is distributed under the GNU General Public 
    % License (version 2 or later); please refer to the file 
    % License.txt, included with the software, for details.
    
    % force z and E to be a column vector

    w=w(:);
    z=L*w;
    z = max(z,mincut);
    mu = exp(z).*E;
    edata = sum(mu-t.*log(mu));
    eprior = .5*sum(w.^2);
    e=edata+eprior;    
    end

    function getL(w, gp, x, t, u)
    % Evaluate the cholesky decomposition if needed
    if nargin < 5
        C=gp_trcov(gp, x);
        L=chol(C)';
    else
        [Kv_ff, Cv_ff] = gp_trvar(gp, x);  % f x 1  vector
        K_fu = gp_cov(gp, x, u);         % f x u
        K_uu = gp_trcov(gp, u);    % u x u, noiseles covariance K_uu
        Luu = chol(K_uu)';
        % Evaluate the Lambda (La) for specific model
        switch gp.sparse
          case 'FIC'
            % Q_ff = K_fu*inv(K_uu)*K_fu'
            % Here we need only the diag(Q_ff), which is evaluated below
            b=Luu\(K_fu');       % u x f
            Qv_ff=sum(b.^2)';
            Lav = Cv_ff-Qv_ff;   % f x 1, Vector of diagonal elements
        end
        L = chol(b'*b + diag(Lav))';
    end
    end
end



% $$$ if nargin > 7   % Sparse model
% $$$     [Kv_ff, Cv_ff] = gp_trvar(gp, x);  % f x 1  vector
% $$$     K_fu = gp_cov(gp, x, u);         % f x u
% $$$     K_uu = gp_trcov(gp, u);    % u x u, noiseles covariance K_uu
% $$$     Luu = chol(K_uu)';
% $$$     % Evaluate the Lambda (La) for specific model
% $$$     switch gp.sparse
% $$$       case 'FIC'
% $$$         % Q_ff = K_fu*inv(K_uu)*K_fu'
% $$$         % Here we need only the diag(Q_ff), which is evaluated below
% $$$         b=Luu\(K_fu');       % u x f
% $$$         Qv_ff=sum(b.^2)';
% $$$         Lav = Cv_ff-Qv_ff;   % f x 1, Vector of diagonal elements
% $$$         % iLaKfu = diag(iLav)*K_fu = inv(La)*K_fu
% $$$         iLaKfu = zeros(size(K_fu));  % f x u, 
% $$$         for i=1:n
% $$$             iLaKfu(i,:) = K_fu(i,:)./Lav(i);  % f x u 
% $$$         end
% $$$     end
% $$$ % $$$     iKuuKuf = K_uu\K_fu';
% $$$ % $$$     c = chol(K_uu+K_fu'*iLaKfu);   % u x u, 
% $$$ % $$$     ic = inv(c);
% $$$ 
% $$$     L = chol(b'*b + diag(Lav))';
% $$$ else
% $$$     C=gp_trcov(gp, x);
% $$$     L=chol(C)';
% $$$ end
