function [z, energ, diagn] = latent_hmc(z, opt, varargin)
% LATENT_HMC     HMC sampler for latent values.
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
if nargin > 7
    u = varargin{4}; 
else
    u = [];
end
n=length(y);
Linv=[];
Cinv=[];
iLaKfu=[];
Lav=[];
ic=[];
iLaKfuic = [];
%mincut = -log(1/realmin);
mincut = -300;
if isfield(gp,'avgE');
    E=gp.avgE(:);
else
    E=1;
end

% Evaluate the help matrices for covariance matrix
if isfield(gp,'sparse')
    getL(z, gp, x, y, u);
else
    getL(z, gp, x, y);
end

hmc2('state',latent_rstate)
rej = 0;
gradf = @lvpoisson_g;
f = @lvpoisson_e;
for li=1:opt.repeat 
    [z, energ, diagn] = hmc2(f, z, opt, gradf, gp, x, y, u);
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
opt.latent_rstate = hmc2('state'); 
diagn.opt = opt;
diagn.rej = rej;
diagn.lvs = opt.stepadj;

    function [g, gdata, gprior] = lvpoisson_g(w, gp, x, y, u, varargin)
    %LVPOISSON_G	Evaluate gradient function for GP latent values with
    %               Poisson likelihood
    
    % Copyright (c) 2006 Aki Vehtari
    
    % Force z and E to be a column vector
    z=w(:);
    z = max(z,mincut);
    mu = exp(z).*E;
    gdata = (mu-y)';      % make the gradient a row vector
    
    if isfield(gp, 'sparse')
        %        gprior = ( z./Lav + iLaKfu*(iciciLaKfu*z) )';
        gprior = ( z./Lav)' - z'*iLaKfuic*iLaKfuic';
    else
        %b=Linv*z;
        %gprior=(Linv'*b)'; dsymvr takes advantage of the symmetry of Cinv
        gprior = (dsymvr(Cinv,z))';   % make the gradient a row vector
    end
    g = gdata+gprior;
    end

    function [e, edata, eprior] = lvpoisson_e(w, gp, x, y, u, varargin)
    %function [e, edata, eprior] = gp_e(w, gp, x, t, varargin)
    % LVPOISSON_E     Minus log likelihood function for spatial modelling.
    %
    %       E = LVPOISSON_E(X, GP, T, Z) takes.... and returns minus log from 
    
    % The field gp.avgE (if given) contains the information about averige
    % expected number of cases at certain location. The target, t, is 
    % distributed as t ~ poisson(avgE*exp(z))
    
    % Copyright (c) 2006 Jarno Vanhatalo
    
    % This software is distributed under the GNU General Public 
    % License (version 2 or later); please refer to the file 
    % License.txt, included with the software, for details.
    
    % force z and E to be a column vector
    z=w(:);

    z = max(z,mincut);
    mu = exp(z).*E;
    edata = sum(mu-y.*log(mu));
    if isfield(gp, 'sparse')
        % eprior = 0.5*z'*inv(La)*z-0.5*z'*(inv(La)*K_fu*(K_uu+Kuf*inv(La)*K_fu)*K_fu'*inv(La))*z;
        b = z'*iLaKfuic;  % 1 x u
        eprior = 0.5*(z./Lav)'*z - 0.5*b*b';
    else
        b=Linv*z;
        eprior=.5*b'*b;
    end
    e=edata+eprior;
    end

    function getL(w, gp, x, y, u)
    if nargin < 5   % Full model
        C=gp_trcov(gp, x);
        L=chol(C)';
        Linv=L\eye(n);
        Cinv = Linv'*Linv;
    else
        [Kv_ff, Cv_ff] = gp_trvar(gp, x);  % f x 1  vector
        K_fu = gp_cov(gp, x, u);         % f x u
        K_uu = gp_trcov(gp, u);    % u x u, noiseles covariance K_uu
        Luu = chol(K_uu)';
        % Evaluate the Lambda (La) for specific model
        switch gp.sparse
          case 'FITC'
            % Q_ff = K_fu*inv(K_uu)*K_fu'
            % Here we need only the diag(Q_ff), which is evaluated below
            b=Luu\(K_fu');       % u x f
            Qv_ff=sum(b.^2)';
            Lav = Cv_ff-Qv_ff;   % f x 1, Vector of diagonal elements
                                 % iLaKfu = diag(iLav)*K_fu = inv(La)*K_fu
            iLaKfu = zeros(size(K_fu));  % f x u, 
            for i=1:n
                iLaKfu(i,:) = K_fu(i,:)./Lav(i);  % f x u 
            end
        end        
        % c = chol(K_uu+K_uf*inv(La)*K_fu))
        c = chol(K_uu+K_fu'*iLaKfu)';   % u x u, 
        ic = inv(c);
        iLaKfuic = iLaKfu*ic';
        
        %Linv = chol(diag(1./Lav) - iLaKfu*(ic'*ic)*iLaKfu');
    end
    end
end