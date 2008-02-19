function [rec, gp, opt] = gp_mc(opt, gp, x, y, xtest, ytest, rec, varargin)
% GP2_MC   Monte Carlo sampling for model GP2R
%
%   REC = GP_MC(OPT, GP, X, T, XX, TT)
%
%   REC = GP_MC(OPT, GP, X, T, XX, TT, REC)
%
%   REC = GP_MC(OPT, GP, X, T, XX, TT, REC, U)
%
%   [REC, GP, OPT] = GP_MC(OPT, GP, X, T, XX, TT, REC, U)
%
%   VARARGIN:n käyttö
%   varargin{:} = u, Linv, ...


%     rec     - record to continue (optional)
%   Returns:
%     rec     - record including hyper-parameters and errors
%     gp      - gp
%     opt     - options structure
%
%   Set default options for GPRMC
%    opt=gp2r_mc;
%      return default options
%    opt=gp2r_mc(opt);
%      fill empty options with default values
%
%   The options and defaults are
%   nsamples (100)
%     the number of samples retained from the Markov chain
%   repeat (1)
%     the number of iterations of basic updates
%   gibbs (0)
%     1 to sample sigmas with gibbs sampling
%   persistence_reset (0)
%     1 to reset persistence after every repeat iterations
%   display (1)
%     1 to display miscallenous data
%     2 to display more miscallenous data
%   plot (1)
%     1 to plot miscallenous data
%

% Copyright (c) 1998-2000 Aki Vehtari
% Copyright (c) 2006      Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

%#function gp_e gp_g
    
% Check arguments
    if nargin < 4
        error('Not enough arguments')
    end

% $$$ % Set empty options to default values
% $$$ opt=gp_mcopt(opt);

    % NOTE ! Here change the initialization of energy and gradient function
    % Initialize the error and gradient functions and get 
    % the function handle to them    
    if isfield(opt, 'fh_e')
        me = opt.fh_e;
        mg = opt.fh_g;
    else
        me = @gp_e;
        mg = @gp_g;
    end

    % Set test data
    if nargin < 6 | isempty(xtest)
        xtest=[];ytest =[];
        if isfield(gp, 'ep_opt')
            xtest = x;       % Set the xtest for EP predictions.
        end
    end

    % Initialize record
    if nargin < 7 | isempty(rec)
        % No old record
        rec=recappend;
    else
        ri=size(rec.etr,1);
    end

    % Set the states of samplers if not given in opt structure
    if isfield(opt, 'hmc_opt')
        if isfield(opt.hmc_opt, 'rstate')
            if ~isempty(opt.hmc_opt.rstate)
                hmc_rstate = opt.hmc_opt.rstate;
            else
                hmc2('state', sum(100*clock))
                hmc_rstate=hmc2('state');
            end
        else
            hmc2('state', sum(100*clock))
            hmc_rstate=hmc2('state');
        end
    end
    if isfield(opt, 'inducing_opt')
        if isfield(opt.inducing_opt, 'rstate')
            if ~isempty(opt.inducing_opt.rstate)
                inducing_rstate = opt.inducing_opt.rstate;
            else
                hmc2('state', sum(100*clock))
                inducing_rstate=hmc2('state');
            end
        else
            hmc2('state', sum(100*clock))
            inducing_rstate=hmc2('state');
        end
    end

    % Set latent values
    if isfield(opt, 'latent_opt')
        z=gp.latentValues';
    else
        z=y;
    end

    % Set handle to gradient and energy functions of negbin-likelihood.
    % Used for sampling the dispersion parameter with slice sampling
    if strcmp(gp.likelih,'negbin')
        nb_me = @nb_e;
    end
    
    % Print labels for sampling information
    if opt.display
        fprintf(' cycle  etr      ');
        if ~isempty(ytest)
            fprintf('etst     ');
        end
        if isfield(opt,'hmc_opt')
            fprintf('hrej     ')              % rejection rate of latent value sampling
        end
        if isfield(opt, 'sls_opt')
            fprintf('slsrej  ');
        end
        if isfield(opt, 'nb_sls_opt')         % Rejection rate of dispersion parameter sampling
            fprintf('rrej  ');
        end
        if isfield(opt,'inducing_opt')
            fprintf('indrej     ')              % rejection rate of latent value sampling
        end
        if isfield(opt,'latent_opt')
            fprintf('lrej ')              % rejection rate of latent value sampling
            if isfield(opt.latent_opt, 'sample_latent_scale') 
                fprintf('    lvScale    ')
            end
        end
        if isfield(opt,'noise_opt')
            fprintf('  nu');
        end      
        fprintf('\n');
    end


    % -------------- Start sampling ----------------------------
    for k=1:opt.nsamples
        
        if opt.persistence_reset
            if isfield(opt, 'hmc_opt')
                hmc_rstate.mom = [];
            end
            if isfield(opt, 'inducing_opt')
                inducing_rstate.mom = [];
            end
            if isfield(opt, 'latent_opt')
                if isfield(opt.latent_opt, 'rstate')
                    opt.latent_opt.rstate.mom = [];
                end
            end
        end
        
        hmcrej = 0;acc=0;
        lrej=0;
        indrej=0;
        for l=1:opt.repeat
            
            % ----------- Sample latent Values  ---------------------
            if isfield(opt,'latent_opt')
                [z, energ, diagnl] = feval(gp.fh_latentmc, z, opt.latent_opt, gp, x, y, varargin{:});
                gp.latentValues = z(:)';
                z = z(:);
                lrej=lrej+diagnl.rej/opt.repeat;
                if isfield(diagnl, 'opt')
                    opt.latent_opt = diagnl.opt;
                end
            end
            
            % ----------- Sample hyperparameters with HMC --------------------- 
            if isfield(opt, 'hmc_opt')
                w = gp_pak(gp, 'hyper');
                hmc2('state',hmc_rstate)              % Set the state
                [w, energies, diagnh] = hmc2(me, w, opt.hmc_opt, mg, gp, x, z, 'hyper', varargin{:});
                hmc_rstate=hmc2('state');             % Save the current state
                hmcrej=hmcrej+diagnh.rej/opt.repeat;
                if isfield(diagnh, 'opt')
                    opt.hmc_opt = diagnh.opt;
                end
                opt.hmc_opt.rstate = hmc_rstate;
                w=w(end,:);
                gp = gp_unpak(gp, w, 'hyper');
            end
            
            % ----------- Sample dispersionparamater with SLS --------------------- 
            if isfield(opt, 'nb_sls_opt')
                w = gp.nb_r;
                [w, energies, diagns] = sls(nb_me, w, opt.nb_sls_opt, [], gp, x, z, y, varargin{:});
                if isfield(diagns, 'opt')
                    opt.nb_sls_opt = diagns.opt;
                end
                w=w(end,:);
                gp.nb_r = w;
            end
            
            % ----------- Sample hyperparameters with SLS --------------------- 
            if isfield(opt, 'sls_opt')
                w = gp_pak(gp, 'hyper');
                [w, energies, diagns] = sls(me, w, opt.sls_opt, mg, gp, x, z, 'hyper', varargin{:});
                if isfield(diagns, 'opt')
                    opt.sls_opt = diagns.opt;
                end
                w=w(end,:);
                gp = gp_unpak(gp, w, 'hyper');
            end

            % ----------- Sample inducing inputs with hmc  ------------ 
            if isfield(opt, 'inducing_opt')
                w = gp_pak(gp, 'inducing');
                hmc2('state',inducing_rstate)         % Set the state
                [w, energies, diagnh] = hmc2(me, w, opt.inducing_opt, mg, gp, x, z, 'inducing');
                inducing_rstate=hmc2('state');        % Save the current state
                indrej=indrej+diagnh.rej/opt.repeat;
                if isfield(diagnh, 'opt')
                    opt.inducing_opt = diagnh.opt;
                end
                opt.inducing_opt.rstate = inducing_rstate;
                w=w(end,:);
                gp = gp_unpak(gp, w, 'inducing');
            end
            
            % ----------- Sample inducing inputs with some other method  ------------ 
% $$$     if isfield(opt, 'inducing_opt')
% $$$         [z, energ, diagnl] = feval(gp.fh_inducingmc, z, opt.inducing_opt, gp, x, y, varargin{:});
% $$$         gp.latentValues = z(:)';
% $$$         z = z(:);
% $$$         slrej=slrej+diagnl.rej/opt.repeat;
% $$$         if isfield(diagnl, 'opt')
% $$$             opt.latent_opt = diagnl.opt;
% $$$         end
% $$$     end
            
            % ------------ Perform sampling for parameters that are not included -------------
            % ------------ in the vector w. Such are, for example, noiseSigmas2  -------------
            % ------------ for gpcf_noiset model
            if isfield(opt, 'noise_opt')
                for nni = 1:length(gp.noise)
                    gp.noise{nni} = feval(gp.noise{nni}.fh_sample, gp, gp.noise{nni}, opt.noise_opt, x, y);
                end
            end
            
            % ----------- Sample inputs  ---------------------
            
            
        end % ------------- for l=1:opt.repeat -------------------------  
        
        % ----------- Set record -----------------------    
        ri=ri+1;

        %    rec=recappend(rec, ri, gp, x, y, xtest, ytest, rejs, varargin{:});
        rec=recappend(rec);
        
        % Display some statistics  THIS COULD BE DONE NICER ALSO...
        if opt.display
            fprintf(' %4d  %.3f  ',ri, rec.etr(ri,1));
            if ~isempty(ytest)
                fprintf('%.3f  ',rec.etst(ri,1));
            end
            if isfield(opt, 'hmc_opt')
                fprintf(' %.1e  ',rec.hmcrejects(ri));
            end
            if isfield(opt, 'sls_opt')
                fprintf('sls  ');
            end
            if isfield(opt, 'inducing_opt')
                fprintf(' %.1e  ',rec.indrejects(ri)); 
            end
            if isfield(opt,'latent_opt')
                fprintf('%.1e',rec.lrejects(ri));
                fprintf('  ');
                if isfield(diagnl, 'lvs')
                    fprintf('%.6f', diagnl.lvs);
                end
            end
            if isfield(opt,'noise_opt')
                fprintf('%.2f', rec.noise{1}.nu(ri));
            end      
            fprintf('\n');
        end
    end
      
    %------------------------------------------------------------------------
    function rec = recappend(rec)
    % RECAPPEND - Record append
    %          Description
    %          REC = RECAPPEND(REC, RI, GP, P, T, PP, TT, REJS, U) takes
    %          old record REC, record index RI, training data P, target
    %          data T, test data PP, test target TT and rejections
    %          REJS. RECAPPEND returns a structure REC containing following
    %          record fields of:
        
        ncf = length(gp.cf);
        nn = length(gp.noise);
        
        if nargin == 0   % Initialize record structure
            rec.nin = gp.nin;
            rec.nout = gp.nout;
            rec.type = gp.type;
            % If sparse model is used save the information about which
            switch gp.type
              case 'FIC'
                rec.X_u = [];
                if isfield(opt, 'inducing_opt')
                    rec.indrejects = 0;
                end
              case 'PIC_BLOCK'
                rec.X_u = [];
                if isfield(opt, 'inducing_opt')
                    rec.indrejects = 0;
                end
                rec.tr_index = gp.tr_index;
              case 'PIC_BAND'
                rec.X_u = [];
                if isfield(opt, 'inducing_opt')
                    rec.indrejects = 0;
                end
                rec.tr_index = gp.tr_index;
              case 'CS+PIC'
                rec.X_u = [];
                if isfield(opt, 'inducing_opt')
                    rec.indrejects = 0;
                end
                rec.tr_index = gp.tr_index;

              otherwise
                % Do nothing
            end
            if isfield(gp,'latentValues')
                rec.latentValues = [];
                rec.lrejects = 0;
            end
            rec.jitterSigmas = [];
            rec.hmcrejects = 0;
            
            if isfield(gp, 'site_tau')
                rec.site_tau = [];
                rec.site_nu = [];
                rec.Ef = [];
                rec.Varf = [];
                rec.p1 = [];
            end

            
            % Initialize the records of covariance functions
            for i=1:ncf
                cf = gp.cf{i};
                rec.cf{i} = feval(cf.fh_recappend, [], gp.nin);
            end
            for i=1:nn
                noise = gp.noise{i};
                rec.noise{i} = feval(noise.fh_recappend, [], gp.nin);
            end
            rec.e = [];
            rec.edata = [];
            rec.eprior = [];
            rec.etr = [];
            ri = 1;
            lrej = 0;
            indrej = 0;
            hmcrej=0;
        end

        % Set the record for every covariance function
        for i=1:ncf
            gpcf = gp.cf{i};
            rec.cf{i} = feval(gpcf.fh_recappend, rec.cf{i}, ri, gpcf);
        end

        % Set the record for every noise function
        for i=1:nn
            noise = gp.noise{i};
            rec.noise{i} = feval(noise.fh_recappend, rec.noise{i}, ri, noise);
        end

        % Set jitterSigmas to record
        if ~isempty(gp.jitterSigmas)
            rec.jitterSigmas(ri,:) = gp.jitterSigmas;
        end

        % Set the latent values to record structure
        if isfield(gp, 'latentValues')
            rec.latentValues(ri,:)=gp.latentValues;
        end
        
        % Set the latent values to record structure
        if isfield(gp, 'site_tau')
            %           fprintf('site_tau '); gp_pak(gp,'hyper')
            switch gp.likelih
              case 'probit'
                [E1, E2, E3, tau, nu] = feval(me, gp_pak(gp,'hyper'), gp, x, y, 'hyper', varargin{:});
                [Ef, Varf, p1] = ep_pred(gp, x, y, xtest);
                rec.site_tau(ri,:)=tau;
                rec.site_nu(ri,:)=nu;
                rec.Ef(ri,:) = Ef';
                rec.Varf(ri,:) = Varf';
                rec.p1(ri,:) = p1';
              case 'poisson'
                [E1, E2, E3, tau, nu] = feval(me, gp_pak(gp,'hyper'), gp, x, y, 'hyper', varargin{:});
                switch gp.type
                  case 'PIC_BLOCK'
                    [Ef] = ep_pred(gp, x, y, xtest, gp.tr_index);
                  otherwise
                    [Ef] = ep_pred(gp, x, y, xtest);
                end
                rec.site_tau(ri,:)=tau;
                rec.site_nu(ri,:)=nu;
                rec.Ef(ri,:) = Ef';
                %rec.Varf(ri,:) = Varf';
            end
        end

        % Set the inducing inputs in the record structure
        switch gp.type
          case {'FIC', 'PIC_BLOCK', 'PIC_BAND', 'CS+PIC'}
            rec.X_u(ri,:) = gp.X_u(:)';
        end
        if isfield(opt, 'inducing_opt')
            rec.indrejects(ri,1)=indrej; 
        end

        % Record training error and rejects
        if isfield(gp,'latentValues')
            [rec.e(ri,:),rec.edata(ri,:),rec.eprior(ri,:)] = feval(me, gp_pak(gp, 'hyper'), gp, x, gp.latentValues', 'hyper', varargin{:});
            rec.etr(ri,:) = rec.e(ri,:);   % feval(gp.likelih_e, gp.latentValues', gp, p, t, varargin{:});
                                           % Set rejects 
            rec.lrejects(ri,1)=lrej;
        else
            %            fprintf('error')
            [rec.e(ri,:),rec.edata(ri,:),rec.eprior(ri,:)] = feval(me, gp_pak(gp, 'hyper'), gp, x, y, 'hyper', varargin{:});
            rec.etr(ri,:) = rec.e(ri,:);
        end
        
        if isfield(opt, 'hmc_opt')
            rec.hmcrejects(ri,1)=hmcrej; 
        end

        % If inputs are sampled set the record which are on at this moment
        if isfield(gp,'inputii')
            rec.inputii(ri,:)=gp.inputii;
        end
        
        % Append the dispersion parameter of NB-likelihood to record
        if strcmp(gp.likelih, 'negbin')
            rec.nb_r(ri,:) = gp.nb_r;
        end
    end
end
