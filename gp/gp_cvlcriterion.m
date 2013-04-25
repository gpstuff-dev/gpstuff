function PE2 = gp_cvlcriterion(gp, x, y, varargin)
%GP_CVLCRITERION cross-validation version of L-criterion
% 
%  Description
%    PE2 = GP_CVLCRITERION(GP, X, Y, OPTIONS) returns cross-validation
%    version of L-criterion PE2 given a Gaussian process model GP,
%    training inputs X and training outputs Y.
%
%   OPTIONS is optional parameter-value pair
%      z      - optional observed quantity in triplet (x_i,y_i,z_i)
%               Some likelihoods may use this. For example, in case of 
%               Poisson likelihood we have z_i=E_i, that is, expected value 
%               for ith case. 
%
%  References
%    Marriott, J. M., Spencer, N. M. and Pettitt, A. N. (2001). A
%    Bayesian Approach to Selecting Covariates for
%    Prediction. Scandinavian Journal of Statistics 28 87–97.
%
%    Vehtari & Ojanen (2011). Bayesian preditive methods for model
%    assesment and selection. In Statistics Surveys, 6:142-228. 
%    <http://dx.doi.org/10.1214/12-SS102>
%
%  See also
%    GP_LCRITERION
%

% Copyright (c) 2011 Ville Tolvanen

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

  ip=inputParser;
  ip.FunctionName = 'GP_CVLCRITERION';
  ip=iparser(ip,'addRequired','gp',@(x) isstruct(x) || iscell(x));
  ip=iparser(ip,'addRequired','x', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))));
  ip=iparser(ip,'addRequired','y', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))));
  ip=iparser(ip,'addParamValue','z', [], @(x) isreal(x) && all(isfinite(x(:))));
  ip=iparser(ip,'parse',gp, x, y, varargin{:});
  % pass these forward
  options=struct();
  z = ip.Results.z;
  if ~isempty(ip.Results.z)
    options.zt=ip.Results.z;
    options.z=ip.Results.z;
  end
  [tn, nin] = size(x);
  if ((isstruct(gp) && isfield(gp.lik.fh, 'trcov')) || (iscell(gp) && isfield(gp{1}.lik.fh,'trcov')))
    % Gaussian likelihood
    [tmp,tmp,tmp,Ey,Vary] = gp_loopred(gp, x, y);
    PE2 = mean((Ey-y).^2 + Vary);

  else
    % Non-Gaussian likelihood
    error('cvlcriterion not sensible for non-gaussian likelihoods');
  end
  
end

