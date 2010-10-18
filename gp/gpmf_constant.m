function gpmf = gpmf_constant(do,varargin)
%GPMF_CONSTANT	Create a constant base function for the GP mean function.
%
%	Description
%        GPMF = GPMF_CONSTANT('init', OPTIONS) Create and initialize
%        constant base function function for Gaussian
%        process mean function. Set the fields of GPMF
%        as  parameter-value pairs ('FIELD', VALUE) in
%        the OPTIONS. The fields that can be modified are:
%
%             'constant'            constant value for the constant
%                                   base function (default 1)
%             'prior_mean'          prior mean (scalar or vector) for base
%                                   functions' weight prior
%             'prior_cov'           prior covariances (scalar or vector) 
%                                   for base functions' prior corresponding
%                                   each selected input dimension. In 
%                                   multiple dimension case prior_cov is a
%                                   struct containing scalars or vectors.
%                                   The covariances must all be either
%                                   scalars (diagonal cov.matrix) or
%                                   vectors (for non-diagonal cov.matrix)
%             'selectedVariables'   vector defining which inputs are 
%                                   active
% 
%	See also
%       gpmf_squared, gpmf_linear
    
% Copyright (c) 2010 Tuomas Nikoskinen

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.

    ip=inputParser;
    ip.FunctionName = 'GPMF_CONSTANT';
    ip.addRequired('do', @(x) ismember(x, {'init'}));
    ip.addOptional('gpmf', [], @isstruct);
    ip.addParamValue('constant',[], @(x) isvector(x) && all(x>0));
    ip.addParamValue('prior_mean',[], @(x) isvector(x));
    ip.addParamValue('prior_cov',[], @(x) isvector(x));
    ip.parse(do, varargin{:});
    do=ip.Results.do;
    gpmf=ip.Results.gpmf;
    constant=ip.Results.constant;
    prior_mean=ip.Results.prior_mean;
    prior_cov=ip.Results.prior_cov;
    
    if isequal(do,'init')
        gpmf.type = 'gpmf_constant';
        % Initialize parameters
        if ~isempty(constant)
            gpmf.constant = constant;
        else
            gpmf.constant = 1;
        end
        if ~isempty(prior_mean)
            gpmf.p.b=prior_mean(:)';
            gpmf.p.B=prior_cov(:)';
        else
            error('No prior specified for constant base function')
        end

        % Set the function handles to the nested functions
        gpmf.fh_geth = @gpmf_geth;
    end
    
    
    function h = gpmf_geth(gpmf, x)
    %GPMF_GETH	 Calculate the base function values for given input.
    %
    %	Description
    %   H = GPMF_GETH(GPMF,X) takes in a mean function data
    %   structure GPMF and inputs X. The function returns a row
    %   vector of length(X) containing the constant value which is by
    %   default 1.
    
        constant=gpmf.constant;
        h = repmat(constant,1,length(x(:,1)));
        
    end

end
