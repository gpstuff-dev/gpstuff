function [dic_u p_eff] = gp_dic_u(gp, x, y, varargin)
% GP_DIC The DIC statistics and effective number of parameters in a GP model
%
%  Description
%   [DIC, P_EFF] = GP_DIC_U(GP, X, Y) evaluates DIC and P_EFF from gp_dic
%   and scales DIC to same scale as waic, refpred etc.
%
%  See also
%    GP_PEFF, DEMO_MODELASSESMENT1, GP_WAIC, GP_REFPRED
ip=inputParser;
ip.FunctionName = 'GP_DIC_U';
ip.addRequired('gp',@(x) isstruct(x) || iscell(x));
ip.addRequired('x', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addRequired('y', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addOptional('focus', 'param', @(x) ismember(x,{'param','latent','all'}))
ip.addParamValue('z', [], @(x) isreal(x) && all(isfinite(x(:))))
ip.parse(gp, x, y, varargin{:});
focus=ip.Results.focus;
% pass these forward
options=struct();
if ~isempty(ip.Results.z)
  options.z=ip.Results.z;
end

if nargout > 1
  [dic p_eff] = gp_dic(gp, x, y, 'focus', focus, options);
else
  dic = gp_dic(gp, x, y, 'focus', focus, options);
end
dic_u = -dic / (2*size(x,1));

end