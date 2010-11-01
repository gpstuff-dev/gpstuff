function  gpmf = gpmf_squared(varargin)
%GPMF_SQUARED  Create a squared base function for the GP mean function.
%
%  Description
%    GPMF = GPMF_SQUARED('PARAM1',VALUE1,'PARAM2,VALUE2,...) 
%    creates linear mean function structure in which the named
%    parameters have the specified values. Any unspecified
%    parameters are set to default values.
%
%    GPMF = GPMF_SQUARED(GPMF,'PARAM1',VALUE1,'PARAM2,VALUE2,...) 
%    modify a mean function structure with the named parameters
%    altered with the specified values.
%  
%    Parameters for linear mean function [default]
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
%      selectedVariables - vector defining which inputs are active
% 
%  See also
%    GP_SET, GPMF_CONSTANT, GPMF_LINEAR
%
  
% Copyright (c) 2010 Tuomas Nikoskinen

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.
  

  ip=inputParser;
  ip.FunctionName = 'GPMF_SQUARED';
  ip.addOptional('gpmf', [], @isstruct);
  ip.addParamValue('selectedVariables',[], @(x) isvector(x) && all(x>0));
  ip.addParamValue('prior_mean',0, @(x) isvector(x));
  ip.addParamValue('prior_cov',100, @(x) isvector(x));
  ip.parse(varargin{:});
  gpmf=ip.Results.gpmf;
  
  if isempty(gpmf)
    % Initialize a mean function
    init=true;
    gpmf.type = 'gpmf_squared';
  else
    % Modify a mean function
    if ~isfield(gp,'type') && isequal(gp.type,'gpmf_squared')
      error('First argument does not seem to be a squared mean function')
    end
    init=false;
  end
  % Initialize parameters
  if init || ~ismember('prior_mean',ip.UsingDefaults)
    gpmf.p.b=ip.Results.prior_mean(:)';
  end
  if init || ~ismember('prior_mean',ip.UsingDefaults)
    gpmf.p.B=ip.Results.prior_cov(:)';
  end
  if init
    % Set the function handles to the nested functions
    gpmf.fh_geth = @gpmf_geth;
  end
  
  function h = gpmf_geth(gpmf, x)
  %GPMF_GETH  Calculate the base function values for given input.
  %
  %  Description
  %    H = GPMF_GETH(GPMF,X) takes in a mean function data
  %    structure GPMF and inputs X. The function returns the
  %    squared base function values H in the given input points. If
  %    selectedVariables is used the function returns only the
  %    values corresponding active inputs. The base function values
  %    are returned as a matrix in which each row corresponds to
  %    one dimension and the first row is for the smallest
  %    dimension.
    
    if ~isfield(gpmf,'selectedVariables')
      h = x'.^2;
    else
      selectedVariables=gpmf.selectedVariables;
      h=zeros(length(selectedVariables),length(x(:,1)));
      
      for i=1:length(selectedVariables)
        h(i,:)=x(:,selectedVariables(i))'.^2;
      end 
    end
    
  end

end
