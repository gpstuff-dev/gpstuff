function opt = gp_mcopt(opt);
%GP_MCOPT     Default options for GP_MC and GP_MC
%
%             Description
%             OPT = GP_MCOPT(OPT) sets default Markov Chain Monte Carlo 
%             sampling options for options structure OPT. 
%
%             The default options are:
%             nsamples          = 1
%             repeat            = 1
%             display           = 1
%             plot              = 1
%             gibbs             = 0
%             hmc_opt           = hmc2_opt
%               hmc_opt.stepsf  = gp2r_steps
%             persistence_reset = 0
%             sample_variances  = 0 
%             sample_latent     = 0

% Copyright (c) 1999 Aki Vehtari

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

if nargin < 1
  opt=[];
end

if ~isfield(opt,'nsamples') | opt.nsamples < 1
  opt.nsamples=1;
end
if ~isfield(opt,'repeat') | opt.repeat < 1
  opt.repeat=1;
end
if ~isfield(opt,'display')
  opt.display=1;
end
if ~isfield(opt,'plot')
  opt.plot=1;
end
if ~isfield(opt,'gibbs')
  opt.gibbs=0;
end
if ~isfield(opt,'hmc_opt')
  opt.hmc_opt=hmc2_opt;
end
if ~isfield(opt,'persistence_reset')
  opt.persistence_reset=0;
end
if ~isfield(opt,'sample_variances')
  opt.sample_variances=0;
end
if ~isfield(opt,'sample_latent')
  opt.sample_latent=0;
end
