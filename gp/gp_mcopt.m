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

% Copyright (c) 1999 Aki Vehtari
% Copyright (c) 2009 Jarno Vanhatalo

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
if ~isfield(opt,'persistence_reset') 
    opt.persistence_reset = 1;
end