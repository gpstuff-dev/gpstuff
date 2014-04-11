function L2 = gp_lcriterion(gp, x, y, varargin)
%GP_LCRITERION  L-criterion for model selection. 
%
%  Description
%    PE2 = GP_CVLCRITERION(GP, X, Y, OPTIONS) returns L-criterion L2
%    given a Gaussian process model GP, training inputs X and training
%    outputs Y.
%
%   OPTIONS is optional parameter-value pair
%      z      - optional observed quantity in triplet (x_i,y_i,z_i)
%               Some likelihoods may use this. For example, in case of 
%               Poisson likelihood we have z_i=E_i, that is, expected value 
%               for ith case. 
%     
%  References
%    Gelfand, A. E. and Ghosh, S. K. (1998). Model Choice: A Minimum
%    Posterior Predictive Loss Approach. Biometrika 85 1–11.
%
%    Ibrahim, J. G., Chen, M.-H. and Sinha, D. (2001). Criterion-based
%    methods for Bayesian model assessment. Statistica Sinica 11
%    419–443.
%   
%    Vehtari & Ojanen (2011). Bayesian preditive methods for model
%    assesment and selection. In Statistics Surveys, 6:142-228. 
%    <http://dx.doi.org/10.1214/12-SS102>
%
%  See also
%    GP_CVLCRITERION
%     
%
% Copyright (c) 2011 Ville Tolvanen

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.


  ip=inputParser;
  ip.FunctionName = 'GP_LCRITERION';
  ip=iparser(ip,'addRequired','gp',@(x) isstruct(x) || iscell(x));
  ip=iparser(ip,'addRequired','x', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))));
  ip=iparser(ip,'addRequired','y', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))));
  ip=iparser(ip,'addParamValue','z', [], @(x) isreal(x) && all(isfinite(x(:))));
  ip=iparser(ip,'parse',gp, x, y, varargin{:});
  ip.parse(gp, x, y, varargin{:});
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
    [tmp,tmp,tmp,Ey,Vary] = gp_pred(gp, x, y, x, 'yt', y);
    L2 = sum((y-Ey).^2 + Vary);

  else
    % Non-Gaussian likelihood
    warning('L-criterion not sensible for non-gaussian likelihoods');
    [tmp,tmp,tmp,Ey,Vary] = gp_pred(gp, x, y, x, 'yt', y, options);
    L2 = sum((y-Ey).^2 + Vary);
    
  end
  
end

