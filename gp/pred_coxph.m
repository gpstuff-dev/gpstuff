function [Eft1, Eft2, Covf, lpyt] = pred_coxph(gp, x, y, xt, varargin)
% PRED_COXPH Wrapper for returning useful values for coxph likelihood

% Copyright (c) 2012 Ville Tolvanen

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

ip=inputParser;
ip.FunctionName = 'PRED_COXPH';
ip=iparser(ip,'addRequired','gp',@isstruct);
ip=iparser(ip,'addRequired','x', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))));
ip=iparser(ip,'addRequired','y', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))));
ip=iparser(ip,'addRequired','xt',  @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))));
ip=iparser(ip,'addParamValue','yt', [], @(x) isreal(x) && all(isfinite(x(:))));
ip=iparser(ip,'addParamValue','z', [], @(x) isreal(x) && all(isfinite(x(:))));
ip=iparser(ip,'addParamValue','zt', [], @(x) isreal(x) && all(isfinite(x(:))));
ip=iparser(ip,'addParamValue','predcf', [], @(x) isempty(x) || ...
                 isvector(x) && isreal(x) && all(isfinite(x)&x>0));
ip=iparser(ip,'addParamValue','tstind', [], @(x) isempty(x) || iscell(x) ||...
                 (isvector(x) && isreal(x) && all(isfinite(x)&x>0)));
ip=iparser(ip,'parse',gp, x, y, xt, varargin{:});

if ~strcmp(gp.lik.type, 'Coxph')
  error('Likelihood not Coxph')
end
if nargout > 3
  [Ef, Covf, lpyt] = gpla_pred(gp, x, y, xt, varargin{:});
else
  [Ef, Covf] = gpla_pred(gp, x, y, xt, varargin{:});
end
ntime=size(gp.lik.xtime,1);
if isfield(gp.lik, 'stratificationVariables')
  ind_str=gp.lik.stratificationVariables;
  nf1=ntime.*unique([x(:,ind_str); xt(:,ind_str)], 'rows');
else
  nf1=ntime;
end

Eft1 = Ef(1:nf1); Ef(1:nf1) = []; Eft2 = Ef;

end

