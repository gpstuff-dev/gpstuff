function [dic, p_eff] = gp_dic(gp, x, y, varargin);
%GP_DIC     The DIC statistics and efective number of parameters in a GP model
%
%	Description
%	[DIC, P_EFF] = GP_DIC(GP, X, Y) evaluates DIC and the effective
%        number of parameters as defined by Spiegelhalter et.al. 
%        (2002). The statistics are evaluated with focus on
%        hyperparameters or latent variables depending on the input
%        GP (See Spiegelhalter et.al. (2002) for discussion on the
%        parameters in focus in Bayesian model). X contains
%        training inputs and Y training outputs.
%
%       DIC and p_eff are evaluated as follows:
%        1) GP is a record structure from gp_mc or an array of GPs from gp_ia, 
%
%           In this case the focus is in the hyperparameters (the
%           parameters of the covariance function and the
%           likelihood). The DIC and the effective number of
%           parameters are evaluated as described in equation
%           (6.10) of Bayesian Data Analysis, second edition
%           (Gelman et.al.):
%               p_eff = E[D(y, th)|y] - D(y, E[th|y])
%               DIC   = E[D(y, th)|y] + p_eff
%           where all the expectations are taken over p(th|y) and D(y, th) =
%           -2log(p(y|th). Now in this formulation we first
%           marginalize over the latent variables to obtain p(y|th)
%           = \int p(y|f)(p(f|th) df. If the likelihood is
%           non-Gaussian the marginalization can not be performed
%           exactly. In this case if GP is an MCMC record we use
%           Laplace approximation to approximate p(y|th). If GP is
%           IA array we use the either EP or Laplace approximation
%           depending which has been used in gp_ia.
%
%       2) GP is Gaussian process data structure
%
%          In this case the focus is in the latent variables and
%          the hyperparameters are considered fixed. The mean of
%          the deviance is now evaluated as
%               E[D(y, f)|y] = -2 \int log(p(y|f) p(f|th) df
%
%       3) GP is a record structure from gp_mc or an array of GPs from 
%          gp_ia, but the focus is defined to be both latent-variables 
%          and hyperparameters, 
%               [DIC, P_EFF] = EP_PEFF(GP, X, Y, 'focus', 'all')
%
%          In this case the focus will be the latent variables and
%          hyperparameters. Thus now we will use the posterior p(f,
%          th|y) instead of the conditional posterior p(f|th,y) or
%          posterior marginal p(th|y). The DIC and the effective
%          number of parameters are evaluated as described in
%          equation (6.10) of Bayesian Data Analysis, second
%          edition (Gelman et.al.):
%
%               p_eff = E[D(y, f, th)|y] - D(y, E[f, th|y])
%               DIC   = E[D(y, f, th)|y] + p_eff
%
%          where all the expectations are taken over p(f,th|y).
%       
%	See also
%	     gp_peff
%   
%       References: 
%
%         Spiegelhalter, Best, Carlin and van der Linde (2002). 
%         Bayesian measures of model complexity and fit. J. R. 
%         Statist. Soc. B, 64, 583-639.
%         
%         Gelman, Carlin, Stern and Rubin (2004) Bayesian Data
%         Analysis, second edition. Chapman & Hall / CRC.
%   

% Copyright (c) 2009-2010 Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.xt, included with the software, for details.

    ip=inputParser;
    ip.FunctionName = 'GP_DIC';
    ip.addRequired('gp',@(x) isstruct(x) || iscell(x));
    ip.addRequired('x', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
    ip.addRequired('y', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
    ip.addOptional('focus', 'hyper', @(x) ismember(x,{'hyper','latent','all'}))
    ip.addParamValue('z', [], @(x) isreal(x) && all(isfinite(x(:))))
    ip.parse(gp, x, y, varargin{:});
    focus=ip.Results.focus;
    % pass these forward
    options=struct();
    z = ip.Results.z;
    if ~isempty(ip.Results.z)
      options.zt=ip.Results.z;
      options.z=ip.Results.z;
    end
    
    [tn, nin] = size(x);
    
    % ====================================================
    if isstruct(gp)     % Single GP or MCMC solution
        switch gp.type
          case {'FULL' 'VAR' 'DTC' 'SOR'}
            tstind = [];
          case {'FIC' 'CS+FIC'}
            tstind = 1:tn;
          case 'PIC'
            tstind = gp.tr_index;
        end

        if isfield(gp, 'etr')    % MCMC solution
            if nargin < 4 || isempty(focus)
                focus = 'hyper';
            end
        else                     % A single GP
            focus = 'latent';
        end     
        
        % Define the error and prediction functions
        if isstruct(gp.likelih) && isfield(gp, 'latent_method')
            switch gp.latent_method
              case 'Laplace'
                fh_pred = @la_pred;
              case 'EP'
                fh_pred = @ep_pred;
            end
        else
            fh_pred = @gp_pred;
        end

        
        
        switch focus
            
          case 'hyper'      % An MCMC solution and focus in hyperparameters
            
            % evaluate the mean of the hyperparameters
            if strcmp(gp.type, 'PIC')
                tr_index = gp.tr_index;
                gp = rmfield(gp, 'tr_index');
            else
                tr_index = [];
            end
            for i = 1:length(gp.edata)
                Gp = take_nth(gp,i);
                w(i,:) = gp_pak(Gp);
            end                
            Gp.tr_index = tr_index;
            Gp = gp_unpak(Gp, mean(w,1));
            if strcmp(gp.type, 'FIC') | strcmp(gp.type, 'PIC')  || strcmp(gp.type, 'CS+FIC') || strcmp(gp.type, 'VAR') || strcmp(gp.type, 'DTC') || strcmp(gp.type, 'SOR')
                Gp.X_u = reshape(Gp.X_u,length(Gp.X_u)/nin,nin);
            end

            % a Gaussian regression model
            if ~isstruct(gp.likelih)
                Davg = 2*mean(gp.edata);
                [e, edata] = gp_e(mean(w,1), Gp, x, y);
                Dth = 2*edata;
            else % non-Gaussian likelihood
                
                % For non-Gaussian likelihood we cannot evaluate the marginal likelihood p(y|th) exactly.
                % For this reason we use Laplace approximation to approximate p(y|th)
                
                gp2 = Gp;
                gp2 = gp_init('set', gp2, 'latent_method', {'Laplace', x, y, 'z', z});
                [e, edata] = gpla_e(mean(w,1), gp2, x, y, 'z', z);
                Dth = 2.*edata;
                
                for i1 = 1:length(gp.edata)
                    [e, edata] = gpla_e(w(i1,:), gp2, x, y, 'z', z);
                    Davg(i1) = 2.*edata;
                end
                Davg = mean(Davg);
            end
            
          case 'latent'     % A single GP solution -> focus on latent variables

            [Ef, Varf, Ey, VarY] = feval(fh_pred, gp, x, y, x, 'tstind', tstind, options);
            sampf = gp_rnd(gp, x, y, x, 'tstind', tstind, 'nsamp', 5000, options);
            if ~isstruct(gp.likelih) % a Gaussian regression model
                sigma2 = VarY - Varf;
                Dth = sum(log(2*pi*sigma2)) + sum( (y - Ef).^2./sigma2 );
                Davg = sum(log(2*pi*sigma2)) + mean(sum( (repmat(y,1,5000) - sampf).^2./repmat(sigma2,1,5000), 1));
            else % non-Gaussian likelihood
                Dth = -2.*feval(gp.likelih.fh_e, gp.likelih, y, Ef, z);
                for i1 = 1:size(sampf, 2)
                    Davg(i1) = feval(gp.likelih.fh_e, gp.likelih, y, sampf(:,i1), z);
                end
                Davg = -2.*mean(Davg);
            end
            
          case 'all'        % An MCMC solution and focus on all parameters
            
            nsamples = length(gp.edata);
            if strcmp(gp.type, 'PIC')
                tr_index = gp.tr_index;
                gp = rmfield(gp, 'tr_index');
            else
                tr_index = [];
            end
            for i = 1:nsamples
                Gp = take_nth(gp,i);
                w(i,:) = gp_pak(Gp);
                if  strcmp(gp.type, 'FIC') | strcmp(gp.type, 'PIC')  || strcmp(gp.type, 'CS+FIC') || strcmp(gp.type, 'VAR') || strcmp(gp.type, 'DTC') || strcmp(gp.type, 'SOR')
                    Gp.X_u = reshape(Gp.X_u,length(Gp.X_u)/nin,nin);
                end
                Gp.tr_index = tr_index;
                if ~isstruct(gp.likelih) % a Gaussian regression model
                    sampf(:,i) = gp_rnd(Gp, x, y, x, 'tstind', tstind, options);
                end
                [Ef(:,i), Varf, Ey, VarY] = gp_pred(Gp, x, y, x, 'tstind', tstind);
                sigma2(:,i) = VarY - Varf;
            end
            Ef = mean(Ef, 2);
            
            if ~isstruct(gp.likelih) % a Gaussian regression model
                msigma2 = mean(sigma2,2);
                Dth = sum(log(2*pi*msigma2)) + sum( (y - Ef).^2./msigma2 );
                Davg = mean(sum(log(2*pi*sigma2),1)) + mean(sum( (repmat(y,1,nsamples) - sampf).^2./sigma2, 1));
            else % non-Gaussian likelihood
                Gp = gp_unpak(Gp, mean(w,1));
                Dth = -2.*feval(Gp.likelih.fh_e, Gp.likelih, y, mean(gp.latentValues,1)', z);
                for i1 = 1:nsamples
                    Gp = take_nth(gp,i1);
                    Davg(i1) = feval(Gp.likelih.fh_e, Gp.likelih, y, Gp.latentValues', z);
                end
                Davg = -2.*mean(Davg);
            end
        end       
        
        
        
        
        

    % ====================================================
    elseif iscell(gp)                 % gp_ia solution
        if nargin < 4 || isempty(focus)
            focus = 'hyper';
        end
        if strcmp(focus, 'latent')
            error(['gp_dic: The focus can be ''latent'' only if single GP structure is given. '...
                   'With IA cell array possible options are ''hyper'' and ''all''.            ']);
        end
        
        switch gp{1}.type
          case {'FULL' 'VAR' 'DTC' 'SOR'}
            tstind = [];
          case {'FIC' 'CS+FIC'}
            tstind = 1:tn;
          case 'PIC'
            tstind = gp{1}.tr_index;
        end

        % Define the error and prediction functions
        if isstruct(gp{1}.likelih) && isfield(gp{1}, 'latent_method')
            switch gp{1}.latent_method
              case 'Laplace'
                fh_pred = @la_pred;
                fh_e = @gpla_e;
              case 'EP'
                fh_pred = @ep_pred;
                fh_e = @gpep_e;
            end
        else
            fh_e = @gp_e;
            fh_pred = @gp_pred;
        end
        
        switch focus
            
          case 'hyper'      % An IA solution and focus in hyperparameters
            
            for i = 1:length(gp)
                Gp = gp{i};
                weight(i) = Gp.ia_weight; 
                w(i,:) = gp_pak(Gp);
                [e, edata] = feval(fh_e, w(i,:), Gp, x, y, options);
                energy(i) = edata;
            end
            Davg = 2*sum(energy.*weight);
            wh = sum(w.*repmat(weight',1,size(w,2)),1);
            Gp = gp_unpak(Gp, wh);
            [e, edata] = feval(fh_e,wh, Gp, x, y, options);
            Dth = 2*edata;
            
          case 'all'        % An IA solution and focus on all parameters
            
            nsamples = length(gp);
            for i = 1:nsamples
                Gp = gp{i};
                weight(i) = Gp.ia_weight;
                w(i,:) = gp_pak(Gp);
                [Ef(:,i), Varf(:,i), Ey, VarY] = feval(fh_pred, Gp, x, y, x, 'tstind', tstind, options);
                sigma2(:,i) = VarY - Varf(:,i);
            end
            mEf = sum(Ef.*repmat(weight, size(Ef,1), 1), 2);

            if ~isstruct(gp{1}.likelih) % a Gaussian regression model
                msigma2 = sum(sigma2.*repmat(weight, size(Ef,1), 1), 2);
                Dth = sum(log(2*pi*msigma2)) + sum( (y - mEf).^2./msigma2 );
                deviance = sum(log(2*pi*sigma2),1) + sum((Varf+Ef.^2-2.*repmat(y,1,nsamples).*Ef+repmat(y.^2,1,nsamples))./sigma2,1);
                Davg = sum(deviance.*weight);
            else % non-Gaussian likelihood
                mw = sum(w.*repmat(weight', 1, size(w,2)), 1);
                Gp = gp_unpak(Gp, mw);
                Dth = -2.*feval(Gp.likelih.fh_e, Gp.likelih, y, mEf, z);
                for i1 = 1:nsamples
                    Gp = gp{i1};
                    Davg(i1) = feval(Gp.likelih.fh_e, Gp.likelih, y, Ef(:,i), z);
                end
                Davg = -2.*sum(Davg.*weight);
            end
        end       

        % ====================================================
    else 
        error(['gp_dic: The first input must be a GP structure, a record structure';
               'from gp_mc or an array of GPs form gp_ia.                         ']) 
    end

    dic = 2*Davg - Dth;
    p_eff = Davg - Dth;
    
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
            for i1=1:(m*n)
                x(i1) = take_nth(x(i1),n);
            end
        else
            % single structure
            names = fieldnames(x);
            for i1=1:size(names,1)
                value = getfield(x,names{i1});
                if length(value) > 1
                    x = setfield(x,names{i1},take_nth(value,nth));
                elseif iscell(value)
                    x = setfield(x,names{i1},{take_nth(value{1},nth)});
                end
            end
        end
    elseif iscell(x)
        % cell array
        for i1=1:(m*n)
            x{i1} = take_nth(x{i1},nth);
        end
    elseif m > 1
        x = x(nth,:);
    end
end
