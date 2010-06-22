function [record, gp, opt] = gp_mc(gp, x, y, varargin)
% GP_MC   Markov chain sampling for Gaussian process models
%
%   Description
%   [RECORD, GP, OPT] = GP_MC(GP, X, Y, OPTIONS) Takes the Gaussian 
%    process structure GP, inputs X and outputs Y. Returns record 
%    structure RECORD with parameter samples, the Gaussian process GP
%    at current state of the sampler and an options structure OPT 
%    containing all the options in OPTIONS and information of the
%    current state of the sampler (e.g. the random number seed)
%
%     OPTIONS is optional parameter-value pair
%      'z'          Optional observed quantity in triplet (x_i,y_i,z_i).
%                   Some likelihoods may use this. For example, in case of
%                   Poisson likelihood we have z_i=E_i, that is, expected
%                   value for ith case.
%      'repeat'     Number of iterations between successive sample saves
%                   (that is every repeat'th sample is stored), default 1.
%      'nsamples'   Number of samples to be returned
%      'display'    Defines if sampling information is printed, 1=yes, 0=no.
%                   Default 1.
%      'hmc_opt'    Options structure for HMC sampler (see hmc2_opt). When
%                   this is given the hyperparameters are sampled with hmc2.
%      'sls_opt'    Options structure for slice sampler (see sls_opt). When 
%                   this is given the hyperparameters are sampled with sls.
%      'gibbs_opt'  Options structure for gibbs sampler. Some covariance
%                   function parameters need to be sampled with Gibbs sampling
%                   (such as gpcf_noiset). The gibbs sampler is implemented
%                   in the respective gpcf_* file and this structure is used 
%                   to give the options for it.
%      'latent_opt' Options structure for latent variable sampler. When this 
%                   is given the latent variables are sampled with function 
%                   stored in the gp.fh_mc field in the GP structure. 
%                   See gp_init. 
%      'likelih_hmc_opt'   Options structure for HMC sampler (see hmc2_opt). 
%                          When this is given the hyperparameters of the 
%                          likelihood are sampled with hmc2.
%      'likelih_sls_opt'   Options structure for slice sampler (see sls_opt). 
%                          When this is given the hyperparameters of the 
%                          likelihood are sampled with hmc2.
%      'persistence_reset' Reset the momentum parameter in HMC sampler after 
%                          every repeat'th iteration, default 0.
%      'record'      An old record structure from where the sampling is 
%                    continued
%         
%      The GP_MC function makes nsamples*repeat iterations and stores
%      every repeat'th sample. At each iteration it samples first the
%      latent variables (if 'latent_opt' option is given), then
%      hyperparameters of the covariance function(s) (if 'hmc_opt',
%      'sls_opt' or 'gibbs_opt' option is given), and for last the
%      hyperparameters in the likelihood function (if
%      'likelih_hmc_opt' or 'likelih_sls_opt' option is given). 
%
%  See also:
%  demo_classific1, demo_robustregression

% Copyright (c) 1998-2000 Aki Vehtari
% Copyright (c) 2007-2010 Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

%#function gp_e gp_g
    

    ip=inputParser;
    ip.FunctionName = 'GP_MC';
    ip.addRequired('gp',@isstruct);
    ip.addRequired('x', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
    ip.addRequired('y', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
    ip.addParamValue('z', [], @(x) isreal(x) && all(isfinite(x(:))))
    ip.addParamValue('nsamples', 1, @(x) isreal(x) && all(isfinite(x(:))))
    ip.addParamValue('repeat', 1, @(x) isreal(x) && all(isfinite(x(:))))
    ip.addParamValue('display', 1, @(x) isreal(x) && all(isfinite(x(:))))
    ip.addParamValue('record',[], @isstruct);
    ip.addParamValue('hmc_opt', [], @(x) isstruct(x) || isempty(x));
    ip.addParamValue('sls_opt', [], @(x) isstruct(x) || isempty(x));
    ip.addParamValue('gibbs_opt', [], @(x) isstruct(x) || isempty(x));
    ip.addParamValue('latent_opt', [], @(x) isstruct(x) || isempty(x));
    ip.addParamValue('likelih_hmc_opt', [], @(x) isstruct(x) || isempty(x));
    ip.addParamValue('likelih_sls_opt', [], @(x) isstruct(x) || isempty(x));
    ip.addParamValue('persistence_reset', 0, @(x) ~isempty(x) && isreal(x));
    ip.parse(gp, x, y, varargin{:});
    z=ip.Results.z;
    opt.nsamples=ip.Results.nsamples;
    opt.repeat=ip.Results.repeat;
    opt.display=ip.Results.display;
    record=ip.Results.record;
    opt.hmc_opt = ip.Results.hmc_opt;
    opt.sls_opt = ip.Results.sls_opt;
    opt.gibbs_opt = ip.Results.gibbs_opt;
    opt.latent_opt = ip.Results.latent_opt;
    opt.likelih_hmc_opt = ip.Results.likelih_hmc_opt;
    opt.likelih_sls_opt = ip.Results.likelih_sls_opt;
    opt.persistence_reset = ip.Results.persistence_reset;
    
    % Check arguments
    if nargin < 3
        error('Not enough arguments')
    end

    % NOTE ! Here change the initialization of energy and gradient function
    % Initialize the error and gradient functions and get 
    % the function handle to them    
    me = @gp_e;
    mg = @gp_g;

    % Initialize record
    if isempty(record)
        % No old record
        record=recappend;
    else
        ri=size(record.etr,1);
    end

    % Set the states of samplers if not given in opt structure
    if ~isempty(opt.latent_opt)
        % Set latent values
        f=gp.latentValues';
        if isfield(opt.latent_opt, 'rstate')
            if ~isempty(opt.latent_opt.rstate)
                latent_rstate = opt.latent_opt.rstate;
            else
                hmc2('state', sum(100*clock))
                latent_rstate=hmc2('state');
            end
        else
            hmc2('state', sum(100*clock))
            latent_rstate=hmc2('state');
        end
    else
        f=y;
    end
    if ~isempty(opt.hmc_opt)
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
    if ~isempty(opt.likelih_hmc_opt)
        if isfield(opt.likelih_hmc_opt, 'rstate')
            if ~isempty(opt.likelih_hmc_opt.rstate)
                likelih_hmc_rstate = opt.likelih_hmc_opt.rstate;
            else
                hmc2('state', sum(100*clock))
                likelih_hmc_rstate=hmc2('state');
            end
        else
            hmc2('state', sum(100*clock))
            likelih_hmc_rstate=hmc2('state');
        end        
    end
    
    % Print labels for sampling information
    if opt.display
        fprintf(' cycle  etr      ');
        if ~isempty(opt.hmc_opt)
            fprintf('hrej     ')              % rejection rate of latent value sampling
        end
        if ~isempty(opt.sls_opt)
            fprintf('slsrej  ');
        end
        if ~isempty(opt.likelih_hmc_opt)
            fprintf('likel.rej  ');
        end
        if ~isempty(opt.latent_opt)
            fprintf('lrej ')              % rejection rate of latent value sampling
            if isfield(opt.latent_opt, 'sample_latent_scale') 
                fprintf('    lvScale    ')
            end
        end
        fprintf('\n');
    end


    % -------------- Start sampling ----------------------------
    for k=1:opt.nsamples
        
        if opt.persistence_reset
            if ~isempty(opt.hmc_opt)
                hmc_rstate.mom = [];
            end
            if ~isempty(opt.latent_opt)
                if isfield(opt.latent_opt, 'rstate')
                    opt.latent_opt.rstate.mom = [];
                end
            end
            if ~isempty(opt.likelih_hmc_opt)
                likelih_hmc_rstate.mom = [];
            end
        end
        
        hmcrej = 0;
        likelih_hmcrej = 0;
        lrej=0;
        indrej=0;
        for l=1:opt.repeat
            
            % ----------- Sample latent Values  ---------------------
            if ~isempty(opt.latent_opt)
                [f, energ, diagnl] = feval(gp.fh_mc, f, opt.latent_opt, gp, x, y, z);
                gp.latentValues = f(:)';
                f = f(:);
                lrej=lrej+diagnl.rej/opt.repeat;
                if isfield(diagnl, 'opt')
                    opt.latent_opt = diagnl.opt;
                end
            end
            
            % ----------- Sample covariance function hyperparameters with HMC --------------------- 
            if ~isempty(opt.hmc_opt)
                infer_params = gp.infer_params;
                gp.infer_params = 'covariance';
                w = gp_pak(gp);
                hmc2('state',hmc_rstate)              % Set the state
                [w, energies, diagnh] = hmc2(me, w, opt.hmc_opt, mg, gp, x, f);                
                hmc_rstate=hmc2('state');             % Save the current state
                hmcrej=hmcrej+diagnh.rej/opt.repeat;
                if isfield(diagnh, 'opt')
                    opt.hmc_opt = diagnh.opt;
                end
                opt.hmc_opt.rstate = hmc_rstate;
                w=w(end,:);
                gp = gp_unpak(gp, w);
                gp.infer_params = infer_params;
            end
            
            % ----------- Sample hyperparameters with SLS --------------------- 
            if ~isempty(opt.sls_opt)
                infer_params = gp.infer_params;
                gp.infer_params = 'covariance';
                w = gp_pak(gp);
                [w, energies, diagns] = sls(me, w, opt.sls_opt, mg, gp, x, f);
                if isfield(diagns, 'opt')
                    opt.sls_opt = diagns.opt;
                end
                w=w(end,:);
                gp = gp_unpak(gp, w);
                gp.infer_params = infer_params;
            end

            % ----------- Sample hyperparameters with Gibbs sampling --------------------- 
            if ~isempty(opt.gibbs_opt)
                % loop over the covariance functions
                ncf = length(gp.cf);
                for i1 = 1:ncf
                    gpcf = gp.cf{i1};
                    if isfield(gpcf, 'fh_gibbs')
                        [gpcf, f] = feval(gpcf.fh_gibbs, gp, gpcf, opt.gibbs_opt, x, f);
                        gp.cf{i1} = gpcf;
                    end
                end
                
                % loop over the noise functions                
                nnf = length(gp.noise);
                for i1 = 1:nnf
                    gpcf = gp.noise{i1};
                    if isfield(gpcf, 'fh_gibbs')
                        [gpcf, f] = feval(gpcf.fh_gibbs, gp, gpcf, opt.gibbs_opt, x, f);
                        gp.noise{i1} = gpcf;
                    end
                end
            end
            
            % ----------- Sample hyperparameters of the likelihood with SLS --------------------- 
            if ~isempty(opt.likelih_sls_opt)
                w = gp_pak(gp, 'likelihood');
                fe = @(w, likelih) (-feval(likelih.fh_e,feval(likelih.fh_unpak,w,likelih),y,f,z) + feval(likelih.fh_priore,feval(likelih.fh_unpak,w,likelih)));
                [w, energies, diagns] = sls(fe, w, opt.likelih_sls_opt, [], gp.likelih);
                if isfield(diagns, 'opt')
                    opt.likelih_sls_opt = diagns.opt;
                end
                w=w(end,:);
                gp = gp_unpak(gp, w, 'likelihood');
            end
            
            % ----------- Sample hyperparameters of the likelihood with HMC --------------------- 
            if ~isempty(opt.likelih_hmc_opt)
                infer_params = gp.infer_params;
                gp.infer_params = 'likelihood';
                w = gp_pak(gp);
                fe = @(w, likelih) (-feval(likelih.fh_e,feval(likelih.fh_unpak,w,likelih),y,f,z)+feval(likelih.fh_priore,feval(likelih.fh_unpak,w,likelih)));
                fg = @(w, likelih) (-feval(likelih.fh_g,feval(likelih.fh_unpak,w,likelih),y,f,'hyper',z)+feval(likelih.fh_priorg,feval(likelih.fh_unpak,w,likelih)));
                
                hmc2('state',likelih_hmc_rstate)              % Set the state
                [w, energies, diagnh] = hmc2(fe, w, opt.likelih_hmc_opt, fg, gp.likelih);
                likelih_hmc_rstate=hmc2('state');             % Save the current state
                likelih_hmcrej=likelih_hmcrej+diagnh.rej/opt.repeat;
                if isfield(diagnh, 'opt')
                    opt.likelih_hmc_opt = diagnh.opt;
                end
                opt.likelih_hmc_opt.rstate = likelih_hmc_rstate;
                w=w(end,:);
                gp = gp_unpak(gp, w);
                gp.infer_params = infer_params;
            end        
                        
            
        end % ------------- for l=1:opt.repeat -------------------------  
        
        % ----------- Set record -----------------------    
        ri=ri+1;
        record=recappend(record);
        
        % Display some statistics  THIS COULD BE DONE NICER ALSO...
        if opt.display
            fprintf(' %4d  %.3f  ',ri, record.etr(ri,1));
            if ~isempty(opt.hmc_opt)
                fprintf(' %.1e  ',record.hmcrejects(ri));
            end
            if ~isempty(opt.sls_opt)
                fprintf('sls  ');
            end
            if ~isempty(opt.likelih_hmc_opt)
                fprintf(' %.1e  ',record.likelih_hmcrejects(ri));
            end
            if ~isempty(opt.latent_opt)
                fprintf('%.1e',record.lrejects(ri));
                fprintf('  ');
                if isfield(diagnl, 'lvs')
                    fprintf('%.6f', diagnl.lvs);
                end
            end
            fprintf('\n');
        end
    end
      
    %------------------------------------------------------------------------
    function record = recappend(record)
    % RECAPPEND - Record append
    %          Description
    %          RECORD = RECAPPEND(RECORD, RI, GP, P, T, PP, TT, REJS, U) takes
    %          old record RECORD, record index RI, training data P, target
    %          data T, test data PP, test target TT and rejections
    %          REJS. RECAPPEND returns a structure RECORD containing following
    %          record fields of:
        
        ncf = length(gp.cf);
        nn = length(gp.noise);
        
        if nargin == 0   % Initialize record structure
            record.type = gp.type;
            record.likelih = gp.likelih;
            % If sparse model is used save the information about which
            switch gp.type
              case 'FIC'
                record.X_u = [];
              case {'PIC' 'PIC_BLOCK'}
                record.X_u = [];
                record.tr_index = gp.tr_index;
              case 'CS+FIC'
                record.X_u = [];
              otherwise
                % Do nothing
            end
            if isfield(gp,'latentValues')
                record.latentValues = [];
                record.lrejects = 0;
            end
            record.jitterSigma2 = [];
            record.hmcrejects = 0;
            
            if isfield(gp, 'site_tau')
                record.site_tau = [];
                record.site_nu = [];
                record.Ef = [];
                record.Varf = [];
                record.p1 = [];
            end
            
            % Initialize the records of covariance functions
            for i=1:ncf
                cf = gp.cf{i};
                record.cf{i} = feval(cf.fh_recappend, [], gp.cf{i});
                % Initialize metric structure
                if isfield(cf,'metric')
                    record.cf{i}.metric = feval(cf.metric.recappend, cf.metric, 1);
                end
            end
            for i=1:nn
                noise = gp.noise{i};
                record.noise{i} = feval(noise.fh_recappend, [], gp.noise{i});
            end
            
            % Initialize the recordord for likelihood
            if isstruct(gp.likelih)
                likelih = gp.likelih;
                record.likelih = feval(likelih.fh_recappend, [], gp.likelih);
            end
            
            record.p = gp.p;
            record.infer_params = gp.infer_params;
            record.e = [];
            record.edata = [];
            record.eprior = [];
            record.etr = [];
            ri = 1;
            lrej = 0;
            indrej = 0;
            hmcrej=0;
            likelih_hmcrej=0;
        end

        % Set the record for every covariance function
        for i=1:ncf
            gpcf = gp.cf{i};
            record.cf{i} = feval(gpcf.fh_recappend, record.cf{i}, ri, gpcf);
            % Record metric structure
            if isfield(gpcf,'metric')
                record.cf{i}.metric = feval(record.cf{i}.metric.recappend, record.cf{i}.metric, ri, gpcf.metric);
            end
        end

        % Set the record for every noise function
        for i=1:nn
            noise = gp.noise{i};
            record.noise{i} = feval(noise.fh_recappend, record.noise{i}, ri, noise);
        end

        % Set the record for likelihood
        if isstruct(gp.likelih)
            likelih = gp.likelih;
            record.likelih = feval(likelih.fh_recappend, record.likelih, ri, likelih);
        end

        % Set jitterSigma2 to record
        if ~isempty(gp.jitterSigma2)
            record.jitterSigma2(ri,:) = gp.jitterSigma2;
        end

        % Set the latent values to record structure
        if isfield(gp, 'latentValues')
            record.latentValues(ri,:)=gp.latentValues;
        end

        % Set the inducing inputs in the record structure
        switch gp.type
          case {'FIC', 'PIC', 'PIC_BLOCK', 'CS+FIC'}
            record.X_u(ri,:) = gp.X_u(:)';
        end

        % Record training error and rejects
        if isfield(gp,'latentValues')
            elikelih = feval(gp.likelih.fh_e, gp.likelih, y, gp.latentValues', z);
            [record.e(ri,:),record.edata(ri,:),record.eprior(ri,:)] = feval(me, gp_pak(gp), gp, x, gp.latentValues');
            record.etr(ri,:) = record.e(ri,:) - elikelih;   % 
% $$$             record.edata(ri,:) = elikelih;
                                           % Set rejects 
            record.lrejects(ri,1)=lrej;
        else
            [record.e(ri,:),record.edata(ri,:),record.eprior(ri,:)] = feval(me, gp_pak(gp), gp, x, y);
            record.etr(ri,:) = record.e(ri,:);
        end
        
        if ~isempty(opt.hmc_opt)
            record.hmcrejects(ri,1)=hmcrej; 
        end

        if ~isempty(opt.likelih_hmc_opt)
            record.likelih_hmcrejects(ri,1)=likelih_hmcrej; 
        end

        % If inputs are sampled set the record which are on at this moment
        if isfield(gp,'inputii')
            record.inputii(ri,:)=gp.inputii;
        end
    end
end
