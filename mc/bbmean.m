function bbm = bbmean(x,B,w)
% BBMEAN  Bayesian bootstrap mean
%
%    Description
%    bbm = bbmean(x,B,w)
%    x MxN matrix of data
%    B number of bootstrap replicates
%    w Mx1 vector of weights 

% Copyright (c) Aki Vehtari, 1998-2004
%
% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

error('No mex-file for this architecture. See Matlab help and convert.m in ./linuxCsource or ./winCsource for help.')