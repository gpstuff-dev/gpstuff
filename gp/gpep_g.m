function [g, gdata, gprior] = gpep_g(w, gp, x, y, param, varargin)
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

% Copyright (c) 2007      Jarno Vanhatalo, Jaakko Riihimäki

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
        [K, C] = gp_trcov(gp,x);
        [e, edata, eprior, tautilde, nutilde, L] = gpep_e(w, gp, x, y, param, varargin);

        Stildesqroot=diag(sqrt(tautilde));
        
        % logZep; nutilde; tautilde;
        b=nutilde-Stildesqroot*(L'\(L\(Stildesqroot*(C*nutilde))));
        invC = Stildesqroot*(L'\(L\Stildesqroot));
        
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
        DKuu_u = 0;
        DKuf_u = 0;

        [e, edata, eprior, tautilde, nutilde, L, La, b] = gpep_e(w, gp, x, y, param, varargin);

        K_fu = gp_cov(gp, x, u);         % f x u
        K_uu = gp_trcov(gp, u);          % u x u, noiseles covariance K_uu
        K_uu = (K_uu+K_uu')./2;          % ensure the symmetry of K_uu        
        iKuuKuf = K_uu\K_fu';
        
        % ============================================================
        % PIC
        % ============================================================
      case 'PIC_BLOCK'
        g_ind = zeros(1,numel(gp.X_u));
        gdata_ind = zeros(1,numel(gp.X_u));
        gprior_ind = zeros(1,numel(gp.X_u));
        
        u = gp.X_u;
        ind = gp.tr_index;
        DKuu_u = 0;
        DKuf_u = 0;
        [e, edata, eprior, tautilde, nutilde, L, La, b] = gpep_e(w, gp, x, y, param, varargin);

        K_fu = gp_cov(gp, x, u);         % f x u
        K_uu = gp_trcov(gp, u);          % u x u, noiseles covariance K_uu
        K_uu = (K_uu+K_uu')./2;          % ensure the symmetry of K_uu

        iKuuKuf = K_uu\K_fu';
        
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
            [g, gdata, gprior] = feval(gpcf.fh_ghyper, gpcf, x, y, g, gdata, gprior, L, b, iKuuKuf, La); %, L2, b2, Labl2
          case 'inducing'
            [g_ind, gdata_ind, gprior_ind] = feval(gpcf.fh_gind, gpcf, x, y, g_ind, gdata_ind, gprior_ind, L, b, iKuuKuf, La);
          case 'all'
            [g, gdata, gprior] = feval(gpcf.fh_ghyper, gpcf, x, y, g, gdata, gprior, L, b, iKuuKuf, La); %, L2, b2, Labl2
            [g_ind, gdata_ind, gprior_ind] = feval(gpcf.fh_gind, gpcf, x, y, g_ind, gdata_ind, gprior_ind, L, b, iKuuKuf, La);
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
                [g, gdata, gprior] = feval(gpcf.fh_ghyper, gpcf, x, y, g, gdata, gprior, L, b, iKuuKuf, La);
              case 'inducing'
                [g_ind, gdata_ind, gprior_ind] = feval(gpcf.fh_gind, gpcf, x, y, g_ind, gdata_ind, gprior_ind, L, b, iKuuKuf, La);
              case 'all'
                [g, gdata, gprior] = feval(gpcf.fh_ghyper, gpcf, x, y, g, gdata, gprior, L, b, iKuuKuf, La);
                [g_ind, gdata_ind, gprior_ind] = feval(gpcf.fh_gind, gpcf, x, y, g_ind, gdata_ind, gprior_ind, L, b, iKuuKuf, La);
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
end
