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

% Check arguments
if nargin < 4
    error('Not enough arguments')
end

% $$$ % Set empty options to default values
% $$$ opt=gp_mcopt(opt);

% NOTE ! Here change the initialization of energy and gradient function
% Initialize the error and gradient functions and get 
% the function handle to them
% $$$ me=gp_e('init', gp);
% $$$ mg=gp_g('init', gp);
me = @gp_e;
mg = @gp_g;

% Set test data
if nargin < 6 | isempty(xtest)
  xtest=[];ytest=[];
end

% Initialize record
if nargin < 7 | isempty(rec)
  % No old record
  ri=1;
  rec=recappend([], ri, gp, x, y, [], [], [], varargin{:});
else
  ri=size(rec.etr,1);
end

% Set the states of samplers if not given in opt structure
if isfield(opt, 'hmc_opt')
  if isfield(opt.hmc_opt, 'rstate')
    if ~isempty(opt.hmc_opt.rstate)
      hmc_rstate = opt.hmc_opt.rstate;
    end
  else
    hmc_rstate = sum(100*clock);
  end
end

% Get weight vector
w = gp_pak(gp);

% Set latent values
if isfield(opt, 'latent_opt')
    z=gp.latentValues';
else
    z=y;
end

% Print labels for sampling information
if opt.display
  fprintf(' cycle  etr      ');
  if ~isempty(xtest)
    fprintf('etst     ');
  end
  fprintf('hrej     ')            % rejection rate of hyperparameter sampling
  if isfield(opt,'latent_opt')
    fprintf('lrej ')              % rejection rate of latent value sampling
    if isfield(opt.latent_opt, 'sample_latent_scale') 
      fprintf('    lvScale')
    end
  end
  fprintf('\n');
end


% -------------- Start sampling ----------------------------
for k=1:opt.nsamples
  
  if opt.persistence_reset
    if isfield(opt, 'hmc_opt')
      hmc_rstate.mom = [];
    end
    if isfield(opt, 'latent_opt')
      if isfield(opt.latent_opt, 'rstate')
        opt.latent_opt.rstate.mom = [];
      end
    end
  end
  
  rejects = 0;acc=0;
  slrej=0;
  for l=1:opt.repeat
      
      % ----------- Sample latent Values  ---------------------
      if isfield(opt,'latent_opt')
          [z, energ, diagnl] = feval(gp.fh_latentmc, z, opt.latent_opt, gp, x, y, varargin{:});
          gp.latentValues = z(:)';
          z = z(:);
          slrej=slrej+diagnl.rej/opt.repeat;
          if isfield(diagnl, 'opt')
              opt.latent_opt = diagnl.opt;
          end
      end
      
    % ----------- Sample hyperparameters with HMC --------------------- 
    if isfield(opt, 'hmc_opt')
        w = gp_pak(gp);
        hmc2('state',hmc_rstate)
        [w, energies, diagnh] = hmc2(me, w, opt.hmc_opt, mg, gp, x, z, varargin{:});
        hmc_rstate=hmc2('state');
        rejects=rejects+diagnh.rej/opt.repeat;
        if isfield(diagnh, 'opt')
            opt.hmc_opt = diagnh.opt;
        end
        w=w(end,:);
        gp = gp_unpak(gp, w);
    end
    
    % ----------- Sample hyperparameters with SLS --------------------- 
    if isfield(opt, 'sls_opt')
        w = gp_pak(gp);
        [w, energies, diagns] = sls(me, w, opt.sls_opt, mg, gp, x, z, varargin{:});
        if isfield(diagns, 'opt')
            opt.sls_opt = diagns.opt;
        end
        w=w(end,:);
        gp = gp_unpak(gp, w);
    end

    % ----------- Sample inducing inputs with hmc  ------------ 
    if isfield(opt, 'inducing_opt')
        w = gp_pak(gp);
        hmc2('state',hmc_rstate)
        [w, energies, diagnh] = hmc2(me, w, opt.hmc_opt, mg, gp, x, z, 'inducing', varargin{:});
        hmc_rstate=hmc2('state');
        rejects=rejects+diagnh.rej/opt.repeat;
        if isfield(diagnh, 'opt')
            opt.hmc_opt = diagnh.opt;
        end
        w=w(end,:);
        gp = gp_unpak(gp, w);
    end

    % ----------- Sample inducing inputs with some other method  ------------ 
    if isfield(opt, 'inducing_opt')
        [z, energ, diagnl] = feval(gp.fh_inducingmc, z, opt.inducing_opt, gp, x, y, varargin{:});
        gp.latentValues = z(:)';
        z = z(:);
        slrej=slrej+diagnl.rej/opt.repeat;
        if isfield(diagnl, 'opt')
            opt.latent_opt = diagnl.opt;
        end
    end
    
    % ------------ Sample the noiseSigmas2 for gpcf_noiset model -------------
    % This is not permanent has to change gp.noise{1}. to some more generic
    if isfield(opt, 'noiset_opt')
        gp.noise{1} = feval(gp.noise{1}.fh_sampling, gp, gp.noise{1}, opt.noiset_opt, x, y);
    end
    
    
    
    % ----------- Sample inputs  ---------------------
    
  end % ------------- for l=1:opt.repeat -------------------------  
  
  % ----------- Set record -----------------------
  opt.hmc_opt.rstate = hmc_rstate;
  
  ri=ri+1;
  if isfield(gp,'latentValues')
    rejs.hmcrejs = rejects;
    rejs.slrejs = slrej;
    rec=recappend(rec, ri, gp, x, y, xtest, ytest, rejs, varargin{:});
  else
    rejs.hmcrejs = rejects;
    rec=recappend(rec, ri, gp, x, y, xtest, ytest, rejs, varargin{:});
  end
  
  % Display some statistics  THIS COULD BE DONE NICER ALSO...
  if opt.display
    fprintf(' %4d  %.3f  ',ri, rec.etr(ri,1));
    if ~isempty(xtest)
      fprintf('%.3f  ',rec.etst(ri,1));
    end
    if isfield(opt, 'hmc_opt')
      fprintf(' %.1e  ',rec.hmcrejects(ri));
    end
    if isfield(opt, 'sls_opt')
      fprintf('sls  ');
    end
    if isfield(gp,'latentValues')
      fprintf('%.1e',rec.lrejects(ri));
      fprintf('  ');
      if isfield(diagnl, 'lvs')
        fprintf('%.6f', diagnl.lvs);
      end
    end      
    fprintf('\n');
  end
end

%-----------------------------------------------------------------------------
function rec = recappend(rec, ri, gp, x, y, xtest, ytest, rejs, varargin)
% RECAPPEND - Record append
%          Description
%          REC = RECAPPEND(REC, RI, GP, P, T, PP, TT, REJS, U) takes
%          old record REC, record index RI, training data P, target
%          data T, test data PP, test target TT and rejections
%          REJS. RECAPPEND returns a structure REC containing following
%          record fields of:

ncf = length(gp.cf);
nn = length(gp.noise);

% Initialize record structure
if ri==1
  rec.nin = gp.nin;
  rec.nout = gp.nout;
  % If sparse model is used save the information about which
  rec.type = gp.type;
  switch gp.type
    case 'FIC'
      re.X_u = [];
    otherwise
      % Do nothing
  end
  if isfield(gp, 'fh_likelih_e')
      rec.likelih = gp.likelih_e;
  end
  if isfield(gp, 'fh_likelih_g')
    rec.fh_likelih_g = gp.fh_likelih_g;
  end
  rec.jitterSigmas = [];
  rec.hmcrejects = 0;
  rejs.hmcrejs = 0;
  if isfield(gp,'latentValues')
    rec.fh_latentmc = gp.fh_latentmc;
    rec.latentValues = [];
    rec.lrejects = 0;
    rejs.slrejs = 0;
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
elseif ri==1
  rec.jitterSigmas=[];
end

% Set the latent values to record structure
if isfield(gp, 'latentValues')
  rec.latentValues(ri,:)=gp.latentValues;
end

% Set the inducing inputs in the record structure
switch gp.type
  case 'FIC'
    re.X_u(ri,:) = gp.X_u(:)';
  otherwise
    % Do nothing
end

% Record training error and rejects
if isfield(gp,'latentValues')
    [rec.e(ri,:),rec.edata(ri,:),rec.eprior(ri,:)]=gp_e(gp_pak(gp), gp, p, gp.latentValues', varargin{:});
    rec.etr(ri,:) = rec.e(ri,:);   % feval(gp.likelih_e, gp.latentValues', gp, p, t, varargin{:});
                                   % Set rejects 
    rec.lrejects(ri,1)=rejs.slrejs;
else
    [rec.e(ri,:),rec.edata(ri,:),rec.eprior(ri,:)]=gp_e(gp_pak(gp), gp, p, t, varargin{:});
    rec.etr(ri,:) = rec.e(ri,:);
end

rec.hmcrejects(ri,1)=rejs.hmcrejs; 

% If inputs are sampled set the record which are on at this moment
if isfield(gp,'inputii')
    rec.inputii(ri,:)=gp.inputii;
end
end

end