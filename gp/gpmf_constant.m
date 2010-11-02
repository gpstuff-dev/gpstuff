function gpmf = gpmf_constant(varargin)
%GPMF_CONSTANT  Create a constant mean function
%
%  Description
%    GPMF = GPMF_CONSTANT('PARAM1',VALUE1,'PARAM2,VALUE2,...) 
%    creates constant mean function structure in which the named
%    parameters have the specified values. Any unspecified
%    parameters are set to default values.
%
%    GPMF = GPMF_CONSTANT(GPMF,'PARAM1',VALUE1,'PARAM2,VALUE2,...) 
%    modify a mean function structure with the named parameters
%    altered with the specified values.
%  
%    Parameters for constant mean function
%      constant          - constant value for the constant
%                          base function (default 1)
%      prior_mean        - prior mean (scalar or vector) for base
%                          functions' weight prior (default 0)
%      prior_cov         - prior covariances (scalar or vector) 
%                          for base functions' prior corresponding
%                          each selected input dimension. In 
%                          multiple dimension case prior_cov is a
%                          struct containing scalars or vectors.
%                          The covariances must all be either
%                          scalars (diagonal cov.matrix) or
%                          vectors (for non-diagonal cov.matrix)
%                          (default 100)  
% 
%  See also
%    GP_SET, GPMF_LINEAR, GPMF_SQUARED
%
  
% Copyright (c) 2010 Tuomas Nikoskinen

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.

  ip=inputParser;
  ip.FunctionName = 'GPMF_CONSTANT';
  ip.addOptional('gpmf', [], @isstruct);
  ip.addParamValue('constant',1, @(x) isvector(x) && all(x>0));
  ip.addParamValue('prior_mean',0, @(x) isvector(x));
  ip.addParamValue('prior_cov',100, @(x) isvector(x));
  ip.parse(varargin{:});
  gpmf=ip.Results.gpmf;
  
  if isempty(gpmf)
    % Initialize a mean function
    init=true;
    gpmf.type = 'gpmf_constant';
  else
    % Modify a mean function
    if ~isfield(gp,'type') && isequal(gp.type,'gpmf_constant')
      error('First argument does not seem to be a constant mean function')
    end
    init=false;
  end
    
  % Initialize parameters
  if init || ~ismember('type',ip.UsingDefaults)
    gpmf.constant = ip.Results.constant;
  end
  if init || ~ismember('prior_mean',ip.UsingDefaults)
    gpmf.p.b=ip.Results.prior_mean(:)';
  end
  if init || ~ismember('prior_mean',ip.UsingDefaults)
    gpmf.p.B=ip.Results.prior_cov(:)';
  end
  if init
    % Set the function handles to the nested functions
    gpmf.fh.geth = @gpmf_geth;
  end
  
  function h = gpmf_geth(gpmf, x)
  %GPMF_GETH  Calculate the base function values for a given input.
  %
  %  Description
  %    H = GPMF_GETH(GPMF,X) takes in a mean function data
  %    structure GPMF and inputs X. The function returns a row
  %    vector of length(X) containing the constant value which is
  %    by default 1.
    
    constant=gpmf.constant;
    h = repmat(constant,1,length(x(:,1)));
    
  end

end
