function h = gpmf_linear(x)
%GPMF_LINEAR	Create a linear base function for the GP mean function.
%
%	Description
%        h = GPMF_LINEAR(x) Call the linear base function with a input
%        argument column vector x. The function returns the transpose of
%        the input vector. 
% 
%	See also
%       gpmf_squared, gpmf_costant

% Copyright (c) 2007-2010 Jarno Vanhatalo

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.

h=x';

end
