function h = gpmf_constant(varargin)
%GPMF_CONSTANT	Create a constant base function for the GP mean function.
%
%	Description
%        GPMF_CONSTANT('set', constantValue) Set the constant value
%        of a constant base function.
%
%        h = GPMF_CONSTANT(x) Call the constant function with input
%        argument x, size(x)=(n,m). Returns a row vector, of size(m,n),
%        containing the constant value. If the constant value hasn't been set
%        use the default value 0.
% 
%	See also
%       gpmf_squared, gpmf_linear
    
% Copyright (c) 2007-2010 Jarno Vanhatalo

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.

persistent gpmfConstVar

if length(varargin)==2    
    if isequal(varargin{1},'set')
        gpmfConstVar=varargin{2};
    else
        error('Setting the constant variable requires the first input argument to be "set"')
    end
elseif size(varargin)==1
    [n m]=size(varargin{1});
    if isempty(gpmfConstVar)
        h=zeros(m,n);
    else
        h=repmat(gpmfConstVar,m,n);
    end
else
    error('Wrong amount on input arguments for gpmf_constant')
end


end
