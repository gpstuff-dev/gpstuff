function  h = gpmf_squared(x)
%GPMF_SQUARED	Create a squared base function for the GP mean function.
%
%	Description
%        h = GPMF_SQUARED(x) Call the squared base function with a input
%        argument column vector x. The function returns a squared input
%        vector which is transposed.
% 
%	See also
%       gpmf_constant, gpmf_linear
    
% Copyright (c) 2007-2010 Jarno Vanhatalo

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.

h = x'.^2;

end
