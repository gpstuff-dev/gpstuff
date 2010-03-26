function opt = gp_iaopt(opt, method, optmethod);
%GP_IAOPT     Default options for GP_IA
%
%             Description
%             OPT = GP_IAOPT(OPT) sets default options for integration
%             approximation (IA) in the structure OPT. The default integration 
%             method is the importance samping with quasi Monte Carlo:
%    
%             OPT = GP_IAOPT(OPT, METHOD) sets default options for integration 
%             approximation (IA) with user specified method. METHOD is a string 
%             defining the integration method. Possibilities and their
%             parameters are:
%
%             'grid'  = a grid based integration
%                  opt.int_method = 'grid';
%                  opt.step_size   = 1;
%                  opt.threshold  = 2.5;
%                  opt.validate   = 1;    (check that the integration results are valid)
%
%              'is_normal' = an importance sampling with normal proposal distribution
%                  opt.int_method = 'normal';
%                  opt.nsamples   = 40;
%                  opt.validate   = 1;    (check that the integration results are valid)
%             
%              'is_normal_qmc' = an importance sampling with normal proposal distribution
%                              with quasi Monte Carlo sampling
%                  opt.int_method = 'quasi_mc';
%                  opt.nsamples   = 40;
%                  opt.validate   = 1;    (check that the integration results are valid)
%
%              'is_student-t'     = an importance sampling with Student-t proposal distribution
%                  opt.int_method = 'is_student-t';
%                  opt.nsamples   = 40;
%                  opt.nu         = 4; the degrees of freedom
%                  opt.validate   = 1;    (check that the integration results are valid)
%
%              'mcmc_hmc'      = Markov chain Monte Carlo sampling using Hybrid Monte Carlo
%                  opt.int_method = 'mcmc_hmc';
%                  opt.nsamples = 40;
%                  opt.repeat = 1;
%                  opt.display = 1;
%                  opt.validate   = 1;    (check that the integration results are valid)
%
%                  opt.hmc_opt.steps = 3;
%                  opt.hmc_opt.stepadj = 0.01;
%                  opt.hmc_opt.nsamples = 1;
%                  opt.hmc_opt.persistence = 0;
%                  opt.hmc_opt.persistence_reset = 0;
%                  opt.hmc_opt.decay = 0.8;
%
%             OPT = GP_IAOPT(OPT, METHOD, OPTMETHOD) sets also user specified
%             optimization method in the structure OPT. OPTMETHOD is a string defining 
%             the optimization method used to find the posterior mode for hyperparameters. 
%             Possibilities are:
%
%             'scg'      = Scaled conjugate gradient algorithm (the default) 
%             'fminunc'  = Matlabs fminunc algorithm 
%
%             The scaled conjugate
%   
%             For reference on the method see Vanhatalo, Pietilainen and Vehtari (2010)

% Copyright (c) 2009-2010 Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

if nargin < 1
    opt=[];
end

if nargin < 3
    optmethod = 'scg';
end

switch optmethod
  case 'scg'
    opt.scg = scg2_opt;
    opt.scg.tolfun = 1e-3;
    opt.scg.tolx = 1e-3;
    opt.scg.display = -1;    
  case 'fminunc'
    opt.fminunc=optimset('GradObj','on');
    opt.fminunc=optimset(opt.fminunc,'LargeScale', 'on');    
    opt.fminunc=optimset(opt.fminunc,'Display', 'off');
end
    
if nargin < 2
    method = 'is_normal_qmc';
end

switch method
  case 'grid'
    opt.int_method = 'grid';
    opt.step_size = 1;
    opt.threshold = 2.5;
    opt.validate = 0;
  case 'is_normal'
    opt.int_method = 'is_normal';
    opt.nsamples = 40;
    opt.validate = 0;
    opt.qmc = 0;
    opt.improved = 0;
  case 'is_normal_qmc'
    opt.int_method = 'is_normal';
    opt.nsamples = 40;
    opt.validate = 0;
    opt.qmc = 1;
    opt.improved = 0;
  case 'is_student-t'
    opt.int_method = 'is_student-t';
    opt.nsamples = 40;
    opt.nu = 4;
    opt.validate = 0;
    opt.improved = 0;
  case 'mcmc_hmc'
    opt.int_method = 'mcmc_hmc';
    opt.nsamples = 40;
    opt.repeat = 1;
    opt.display = 1;
    opt.validate = 0;
    
    % Set the hmc sampling options
    opt.hmc_opt.steps = 3;
    opt.hmc_opt.stepadj = 0.01;
    opt.hmc_opt.nsamples = 1;
    opt.hmc_opt.persistence = 0;
    opt.hmc_opt.persistence_reset = 0;
    opt.hmc_opt.decay = 0.8;
  case 'mcmc_sls'
    opt.int_method = 'mcmc_sls';
    opt.nsamples = 40;
    opt.repeat = 1;
    opt.display = 1;
    opt.validate = 0;
    
    % Set the sls sampling options
  case 'CCD'
    opt.int_method = 'CCD';
    opt.improved = 0;
    opt.step_size = 1;
    opt.f0 = 1.1;
    opt.validate = 0;
  otherwise
    error('gp_iaopt: unknown type of integration method.')
end    
