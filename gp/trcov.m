function C = trcov(gpcf, x)
% TRCOV     Evaluate training covariance matrix for covariance function
%           This is a mex-function that is called from gpcf_*_trcov functions.
%
%         Description
%         K = TRCOV(GP, TX) takes in Gaussian process GP and matrix
%         TX that contains training input vectors to GP. Returns noiseless
%         covariance matrix K. Every element ij of K contains covariance 
%         between inputs i and j in TX.
%
%         [K, C] = GP_TRCOV(GP, TX) returns also the noisy
%         covariance matrix C.
%
%         For covariance function definition see manual or 
%         Neal R. M. Regression and Classification Using Gaussian 
%         Process Priors, Bayesian Statistics 6.

% Copyright (c) 2008 Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

    C = NaN;
%error('No mex-file for this architecture. See Matlab help and gp_compile.m for help.')