function [Eft1, Eft2, Covf, lpyt] = pred_coxph(gp, x, y, xt, varargin)
% PRED_COXPH Wrapper for returning useful values for coxph likelihood

ip=inputParser;
ip.FunctionName = 'PRED_COXPH';
ip.addRequired('gp',@isstruct);
ip.addRequired('x', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addRequired('y', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addRequired('xt',  @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addParamValue('yt', [], @(x) isreal(x) && all(isfinite(x(:))))
ip.addParamValue('z', [], @(x) isreal(x) && all(isfinite(x(:))))
ip.addParamValue('zt', [], @(x) isreal(x) && all(isfinite(x(:))))
ip.addParamValue('predcf', [], @(x) isempty(x) || ...
                 isvector(x) && isreal(x) && all(isfinite(x)&x>0))
ip.addParamValue('tstind', [], @(x) isempty(x) || iscell(x) ||...
                 (isvector(x) && isreal(x) && all(isfinite(x)&x>0)))
ip.parse(gp, x, y, xt, varargin{:});
% yt=ip.Results.yt;
% z=ip.Results.z;
% zt=ip.Results.zt;
% predcf=ip.Results.predcf;
% tstind=ip.Results.tstind;

if ~strcmp(gp.lik.type, 'Coxph')
  error('Likelihood not Coxph')
end
if nargout > 3
  [Ef, Covf, lpyt] = gpla_nd_pred(gp, x, y, xt, varargin{:});
else
  [Ef, Covf] = gpla_nd_pred(gp, x, y, xt, varargin{:});
end
ntime = size(gp.lik.stime,2)-1;
Eft1 = Ef(1:ntime); Ef(1:ntime) = []; Eft2 = Ef;

end

