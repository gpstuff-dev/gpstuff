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
ip.parse(gp, x, y, varargin{:});
z=ip.Results.z;
optimf=ip.Results.optimf;
opt=ip.Results.opt;

optdefault=struct('GradObj','on','LargeScale','off');
opt=optimset(optdefault,opt);
w=gp_pak(gp);
switch nargout
  case 6
    [w,fval,exitflag,output,grad,hessian] = optimf(@(ww) gp_eg(ww, gp, x, y, 'z', z), w, opt);
    varargout{:}={fval,exitflag,output,grad,hessian};
  case 5
    [w,fval,exitflag,output,grad] = optimf(@(ww) gp_eg(ww, gp, x, y, 'z', z), w, opt);
    varargout{:}={fval,exitflag,output,grad};
  case 4
    [w,fval,exitflag,output] = optimf(@(ww) gp_eg(ww, gp, x, y, 'z', z), w, opt);
    varargout{:}={fval,exitflag,output};
  case 3
    [w,fval,exitflag] = optimf(@(ww) gp_eg(ww, gp, x, y, 'z', z), w, opt);
    varargout{:}={fval,exitflag};
  case 2
    [w,fval] = optimf(@(ww) gp_eg(ww, gp, x, y, 'z', z), w, opt);
    varargout{:}={fval};
  case 1
    w = optimf(@(ww) gp_eg(ww, gp, x, y, 'z', z), w, opt);
    varargout{:}={};
end
gp=gp_unpak(gp,w);
