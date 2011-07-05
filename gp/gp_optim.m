function [gp, varargout] = gp_optim(gp, x, y, varargin)
%GP_OPTIM  Optimize paramaters of a Gaussian process 
%
%  Description
%    GP = GP_OPTIM(GP, X, Y, OPTIONS) optimises the parameters of a
%    GP structure given matrix X of training inputs and vector
%    Y of training targets.
%
%    [GP, OUTPUT1, OUTPUT2, ...] = GP_OPTIM(GP, X, Y, OPTIONS)
%    optionally returns outputs of the optimization function.
%
%    OPTIONS is optional parameter-value pair
%      z      - optional observed quantity in triplet (x_i,y_i,z_i)
%               Some likelihoods may use this. For example, in case of
%               Poisson likelihood we have z_i=E_i, that is, expected
%               value for ith case.
%      optimf - function handle for an optimization function, which is
%               assumed to have similar input and output arguments
%               as usual fmin*-functions. Default is @fminscg.
%      opt    - options structure for the minimization function. 
%               Use optimset to set these options. By default options
%               'GradObj' is 'on', 'LargeScale' is 'off'.
%
%  See also
%    GP_SET, GP_E, GP_G, GP_EG, FMINSCG, FMINLBFGS, OPTIMSET, DEMO_REGRESSION*
%

%  Experimental
%      energy - 'e' to minimize the marginal posterior energy (default) or
%               'looe' to minimize the leave-one-out energy
%               'kfcve' to minimize the k-fold-cv energy
%      k          - number of folds in CV
%
%  See also
%    GP_SET, GP_E, GP_G, GP_EG, GP_LOOE, GP_LOOG, GP_LOOEG, GPEP_LOOE, 
%    FMINSCG, FMINLBFGS, OPTIMSET, DEMO_REGRESSION*

% Copyright (c) 2010 Aki Vehtari

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.

ip=inputParser;
ip.FunctionName = 'GP_OPTIM';
ip.addRequired('gp',@isstruct);
ip.addRequired('x', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addRequired('y', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addParamValue('z', [], @(x) isreal(x) && all(isfinite(x(:))))
ip.addParamValue('optimf', @fminscg, @(x) isa(x,'function_handle'))
ip.addParamValue('opt', [], @isstruct)
ip.addParamValue('energy', 'e', @(x) ismember(x,{'e', 'looe', 'kfcve'}))
ip.addParamValue('k', 10, @(x) isreal(x) && isscalar(x) && isfinite(x) && x>0)
ip.parse(gp, x, y, varargin{:});
z=ip.Results.z;
optimf=ip.Results.optimf;
opt=ip.Results.opt;
energy=ip.Results.energy;
k=ip.Results.k;

switch energy
  case 'e'
    fh_eg=@(ww) gp_eg(ww, gp, x, y, 'z', z);
    optdefault=struct('GradObj','on','LargeScale','off');
  case 'looe'
    fh_eg=@(ww) gp_looeg(ww, gp, x, y, 'z', z);
    if isfield(gp.lik.fh,'trcov')
      optdefault=struct('GradObj','on','LargeScale','off');
    else
      % EP-LOO and Laplace-LOO do not have yet gradients
      optdefault=struct('GradObj','off','LargeScale','off');
      if ismember('optimf',ip.UsingDefaults)
        optimf=@fminunc;
      end
    end
  case 'kfcve'
    % kfcv does not have yet gradients
    fh_eg=@(ww) gp_kfcve(ww, gp, x, y, 'z', z, 'k', k);
    optdefault=struct('GradObj','off','LargeScale','off');
    if ismember('optimf',ip.UsingDefaults)
      optimf=@fminunc;
    end
end
opt=optimset(optdefault,opt);
w=gp_pak(gp);
switch nargout
  case 6
    [w,fval,exitflag,output,grad,hessian] = optimf(fh_eg, w, opt);
    varargout{:}={fval,exitflag,output,grad,hessian};
  case 5
    [w,fval,exitflag,output,grad] = optimf(fh_eg, w, opt);
    varargout{:}={fval,exitflag,output,grad};
  case 4
    [w,fval,exitflag,output] = optimf(fh_eg, w, opt);
    varargout{:}={fval,exitflag,output};
  case 3
    [w,fval,exitflag] = optimf(fh_eg, w, opt);
    varargout{:}={fval,exitflag};
  case 2
    [w,fval] = optimf(fh_eg, w, opt);
    varargout{:}={fval};
  case 1
    w = optimf(fh_eg, w, opt);
    varargout{:}={};
end
gp=gp_unpak(gp,w);
