function e = logunif_e(x, a)
%LOGUNIF_E  compute the prior energy term for a parameter with 
%           uniform prior in log scale. 
%
%        Description
%        E = LOGUNIF_E(X,A) takes parameter  matrix X and 
%        hyper-parameter structure A and returns a vector
%        containing energy term E.

% Copyright (c) 2008 Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

e = log(x);   % = - log(1./x)
              % where the log comes from the definition of energy 
              % as log( p(x) )