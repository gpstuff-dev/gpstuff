function R = cond_invgam_invgam1(A, A1, A2, X)
% COND_INVGAM_INVGAM     Sample conditional distribution from
%                        inverse gamma likelihood and prior
%
%       Description
%       R = COND_INVGAM_INVGAM(A, A1, A2, X) generates one sample
%       from the conditional distribution of A given
%       parameter structure X of lower level, structure A1 of
%       same level hyper-parameters and A2 of higher level, i.e 
%       is r~P(A|A1,A2,X). Returns one new sample R from the 
%       distribution above

% Copyright (c) 1999-2000 Aki Vehtari


% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

error('No mex-file for this architecture. See Matlab help and convert.m in ./linuxCsource or ./winCsource for help.')
