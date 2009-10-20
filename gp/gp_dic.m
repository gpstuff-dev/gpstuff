function [dic, p_eff] = gp_dic(gp, tx, ty, param, focus);
%GP_PEFF	The DIC statistics and efective number of parameters in GP model
%
%	Description
%	[DIC, P_EFF] = EP_PEFF(GP, TX, TY) evaluates DIC and the efective number of 
%       parameters as defined by Spiegelhalter et.al. (2002). The statistics are 
%       evaluated with focus on hyperparameters or latent variables depending on 
%       input GP (See Spiegelhalter et.al. (2002) for discussion on the parameters
%       in focus in Bayesian model). TX contains training inputs and TY training
%       outputs.
%
%       How DIC and p_eff are evaluated:
%   
%       1) GP is a record structure form gp_mc or an array of GPs from gp_ia, 
%    
%         In this case the focus is in the hyperparameters (the parameters of the 
%         covariance function and the likelihood). The DIC and the effective number of 
%         parameters are evaluated as described in equation (6.10) of Bayesian Data 
%         Analysis, second edition (Gelman et.al.):
%
%               p_eff = E[D(y, th)|y] - D(y, E[th|y])
%               DIC   = E[D(y, th)|y] + p_eff
%
%         where all the expectations are taken over p(th|y) and D(y, th) = -2log(p(y|th).
%         Now in this formulation we first marginalize over the latent variables to obtain
%         p(y|th) = \int p(y|f)(p(f|th) df. If the likelihood is non-Gaussian the 
%         marginalization is done via Gaussian quadratures for each of the n likelihood 
%         terms (see fro example likelih_poisson -> likelih_poisson_predprob for more 
%         information).
%
%       2) GP is Gaussian process data structure
%
%         In this case the focus is in the latent variables and the hyperparameters are
%         considered fixed. The efective number of parameters is now approximated as follows:
%
%               p_eff = n - trace( K\C ),
%
%         where K is the prior covariance matrix and C the posterior covariance matrix. 
%         This approximation is introduced by Spiegelhalter et.al. (2002) in equation (16). 
%         The mean of the deviance is evaluated as
%
%               E[D(y, f)|y] = -2 \int log(p(y|f) p(f|th) df
%
%       3) GP is a record structure form gp_mc or an array of GPs from gp_ia, but you define 
%          the focus to be both latent-variables and hyperparameters, 
%          [DIC, P_EFF] = EP_PEFF(GP, TX, TY, 'all')
%
%          In this case the focus will be the latent variables and hyperparameters. Thus now
%          we will use the posterior p(f, th|y) instead of the conditional posterior p(f|th,y) 
%          or posterior marginal p(th|y). The DIC and the effective number of parameters are
%          evaluated as described in equation (6.10) of Bayesian Data Analysis, second edition
%          (Gelman et.al.):
%
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
%         Spiegelhalter, Best, Carlin and van der Linde (2002). Bayesian measures
%         of model complexity and fit. J. R. Statist. Soc. B, 64, 583-639.
%         
%         Gelman, Carlin, Stern and Rubin (2004) Bayesian Data Analysis, second 
%         edition. Chapman & Hall / CRC.
%   
% Copyright (c) 2009 Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

    tn = size(tx,1);
    
    if nargin < 4 | isempty(param)
        param = 'hyper';
    end
    
    % ====================================================
    if isstruct(gp)     % Single GP or MCMC solution
        switch gp.type
          case 'FULL'
            tstind = [];
          case 'FIC'
            tstind = 1:tn;
          case 'PIC'
            tstind = gp.tr_index;
        end

        if isfield(gp, 'etr')    % MCMC solution
            if nargin < 5
                focus = 'hyper';
            end
        else                     % A single GP
            focus = 'latent';
        end

        if ~isstruct(gp.likelih)   % a regression model
        %  ------------------            
            switch focus
                
              case 'hyper'      % An MCMC solution and focus in hyperparameters
                                % Focus is on hyperparameters
                Davg = 2*mean(gp.edata);
                % evaluate the mean of the hyperparameters
                if strcmp(gp.type, 'PIC')
                    tr_index = gp.tr_index;
                    gp = rmfield(gp, 'tr_index');
                else
                    tr_index = [];
                end
                for i = 1:length(gp.edata)                    
                    Gp = take_nth(gp,i);
                    w(i,:) = gp_pak(Gp, param);
                end                
                Gp.tr_index = tr_index;
                Gp = gp_unpak(Gp, mean(w,1), param);
                if strcmp(gp.type, 'FIC') | strcmp(gp.type, 'PIC')
                    nin = Gp.nin;
                    Gp.X_u = reshape(Gp.X_u,length(Gp.X_u)/nin,nin);
                end    
                [e, edata] = gp_e(mean(w,1), Gp, tx, ty, param);
                Dth = 2*edata;
                dic = 2*Davg - Dth;
              case 'latent'     % A single GP solution -> focus on latent variables
                [Ef, Varf, Ey, VarY] = gp_pred(gp, tx, ty, tx, [], tstind);
                sampf = gp_rnd(gp, tx, ty, tx, [], tstind, 1000);
                sigma2 = VarY - Varf;
                Dth = sum(log(2*pi*sigma2)) + sum( (ty - Ef).^2./sigma2 );
                Davg = sum(log(2*pi*sigma2)) + mean(sum( (repmat(ty,1,1000) - sampf).^2./repmat(sigma2,1,1000), 1));
                dic = 2*Davg - Dth;
              case 'all'        % An MCMC solution and focus on all parameters
                                % evaluate the mean of the parameters
                nsamples = length(gp.edata);
                if strcmp(gp.type, 'PIC')
                    tr_index = gp.tr_index;
                    gp = rmfield(gp, 'tr_index');
                else
                    tr_index = [];
                end
                for i = 1:nsamples
                    Gp = take_nth(gp,i);
                    if strcmp(gp.type, 'FIC') | strcmp(gp.type, 'PIC')
                        nin = Gp.nin;
                        Gp.X_u = reshape(Gp.X_u,length(Gp.X_u)/nin,nin);
                    end
                    Gp.tr_index = tr_index;
                    sampf(:,i) = gp_rnd(Gp, tx, ty, tx, [], tstind);
                    [Ef(:,i), Varf, Ey, VarY] = gp_pred(Gp, tx, ty, tx, [], tstind);
                    sigma2(:,i) = VarY - Varf;
                end                
                Ef = mean(Ef, 2);
                msigma2 = mean(sigma2,2);
                
                Dth = sum(log(2*pi*msigma2)) + sum( (ty - Ef).^2./msigma2 );
                Davg = mean(sum(log(2*pi*sigma2),1)) + mean(sum( (repmat(ty,1,nsamples) - sampf).^2./sigma2, 1));
                dic = 2*Davg - Dth;
            end       
        else                        % A non Gaussian observation model
        %  ------------------
            
            
        end
        
        

    % ====================================================
    elseif iscell(gp)                 % gp_ia solution
        if nargin < 5
            focus = 'hyper';
        end
        
        switch gp{1}.type
          case 'FULL'
            tstind = [];
          case 'FIC'
            tstind = 1:tn;
          case 'PIC'
            tstind = gp{1}.tr_index;
        end
        
        if ~isstruct(gp{1}.likelih)   % a regression model
        %  ------------------
            switch focus
                
              case 'hyper'      % An MCMC solution and focus in hyperparameters
                                % Focus is on hyperparameters
                
                for i = 1:length(gp)
                    Gp = gp{i};
                    weight(i) = Gp.ia_weight; 
                    w(i,:) = gp_pak(Gp, param);
                    [e, edata] = gp_e(w(i,:), Gp, tx, ty, param);
                    energy(i) = edata;
                end                
                Davg = 2*sum(energy.*weight);
                wh = sum(w.*repmat(weight',1,size(w,2)),1);
                Gp = gp_unpak(Gp, wh, param);
                [e, edata] = gp_e(wh, Gp, tx, ty, param);
                Dth = 2*edata;
                dic = 2*Davg - Dth;
              case 'all'        % An MCMC solution and focus on all parameters
                                % evaluate the mean of the parameters
                nsamples = length(gp);
                for i = 1:nsamples
                    Gp = gp{i};
                    weight(i) = Gp.ia_weight;
                    
                    [Ef(:,i), Varf(:,i), Ey, VarY] = gp_pred(Gp, tx, ty, tx, [], tstind);
                    sigma2(:,i) = VarY - Varf(:,i);
                end
                mEf = sum(Ef.*repmat(weight, size(Ef,1), 1), 2);
                msigma2 = sum(sigma2.*repmat(weight, size(Ef,1), 1), 2);
                Dth = sum(log(2*pi*msigma2)) + sum( (ty - mEf).^2./msigma2 );
                
                deviance = sum(log(2*pi*sigma2),1) + sum((Varf+Ef.^2-2.*repmat(ty,1,nsamples).*Ef+repmat(ty.^2,1,nsamples))./sigma2,1);
                Davg = sum(deviance.*weight);

                dic = 2*Davg - Dth;
            end       
        else                        % A non Gaussian observation model
        %  ------------------
            
            
        end

    % ====================================================
    else 
        error(['gp_dic: The first input must be a GP structure, a record structure';
               'from gp_mc or an array of GPs form gp_ia.                         ']) 
    end

    
    p_eff = Davg - Dth;
    
    
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
end
