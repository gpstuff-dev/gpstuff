function gpmf = gpmf_linear(do,varargin)
%GPMF_LINEAR	Create a linear base function for the GP mean function.
%
%	Description
%        GPMF = GPMF_LINEAR('init', OPTIONS) Create and initialize
%        linear base function function for Gaussian
%        process mean function. Set the fields of GPMF
%        as  parameter-value pairs ('FIELD', VALUE) in
%        the OPTIONS. The fields that can be modified are:
%
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
%       gpmf_squared, gpmf_costant

% Copyright (c) 2010 Tuomas Nikoskinen

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.

    ip=inputParser;
    ip.FunctionName = 'GPMF_LINEAR';
    ip.addRequired('do', @(x) ismember(x, {'init'}));
    ip.addOptional('gpmf', [], @isstruct);
    ip.addParamValue('selectedVariables',[], @(x) isvector(x) && all(x>0));
    ip.addParamValue('prior_mean',[], @(x) isvector(x));
    ip.addParamValue('prior_cov',[], @(x) isvector(x));
    ip.parse(do, varargin{:});
    do=ip.Results.do;
    gpmf=ip.Results.gpmf;
    selectedVariables=ip.Results.selectedVariables;
    prior_mean=ip.Results.prior_mean;
    prior_cov=ip.Results.prior_cov;
    
    if isequal(do,'init')
        gpmf.type = 'gpmf_linear';
        % Initialize parameters
        if ~isempty(selectedVariables)
            gpmf.selectedVariables = selectedVariables;
        end
        if ~isempty(prior_mean)
            gpmf.p.b=prior_mean(:)';
            gpmf.p.B=prior_cov(:)';
        else
            error('No prior specified for linear base function')
        end

        % Set the function handles to the nested functions
        gpmf.fh_geth = @gpmf_geth;
    end
    
    
    function h = gpmf_geth(gpmf, x)
    %GPMF_GETH	 Calculate the base function values for given input.
    %
    %	Description
    %   H = GPMF_GETH(GPMF,X) takes in a mean function data
    %   structure GPMF and inputs X. The function returns the linear base function
    %   values H in the given input points. If selectedVariables is used
    %   the function returns only the values corresponding active inputs. The base function 
    %   values are returned as a matrix in which each row corresponds to
    %   one dimension and the first row is for the smallest dimension.
    
    
    
        if ~isfield(gpmf,'selectedVariables')
            h=x';
        else
            selectedVariables=gpmf.selectedVariables;
            h=zeros(length(selectedVariables),length(x(:,1)));
            
            for i=1:length(selectedVariables)
                h(i,:)=x(:,selectedVariables(i))';
            end 
        end
        
    end

end
