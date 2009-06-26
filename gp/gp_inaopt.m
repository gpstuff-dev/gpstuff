function opt = gp_inaopt(opt, method);
%GP_MCOPT     Default options for GP_INA
%
%             Description
%             OPT = GP_INAOPT(OPT) sets default options for integrated nested
%             approximation (INA) in the structure OPT. The default integration 
%             method is the quasi Monte Carlo with importance samping:
%
%    
%             OPT = GP_INAOPT(OPT, METHOD) sets default options for integrated 
%             nested approximation (INA) with user specified method in the 
%             structure OPT. METHOD is a string defining the integration method.
%             Possibilities and their parameters are:
%   
%             'grid_based'  = a grid based integration
%                  opt.int_method = 'grid_based';
%                  opt.stepsize   = 1;
%                  opt.threshold  = 2.5;
%
%              'normal' = an importance sampling with normal proposal distribution
%                  opt.int_method = 'normal';
%                  opt.nsamples   = 40;
%             
%              'quasi_mc' = an importance sampling with normal proposal distribution
%                           with quasi Monte Carlo sampling
%                  opt.int_method = 'quasi_mc';
%                  opt.nsamples   = 40;
%
%             For reference on the method see ...

% Copyright (c) 2009 Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

if nargin < 1
  opt=[];
end

opt.fminunc=optimset('GradObj','on');
opt.fminunc=optimset(opt,'LargeScale', 'on');
opt.fminunc=optimset(opt,'Display', 'iter');

if nargin < 2
    opt.int_method = 'quasi_mc';
    opt.nsamples = 40;
else
    switch method
      case 'grid_based'
        opt.int_method = 'grid_based';
        opt.stepsize = 1;
        opt.threshold = 2.5;
      case 'normal'
        opt.int_method = 'normal';
        opt.nsamples = 40;
      case 'quasi_mc'
        opt.int_method = 'quasi_mc';
        opt.nsamples = 40;        
    end    
end
