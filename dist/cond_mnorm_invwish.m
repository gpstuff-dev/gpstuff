function r = cond_mnorm_invwish(a, a1, a2, x)
% COND_MNORM_INVWISH    Sample conditional distribution from normal
%                       likelihood for multi-parameter group and
%                       inverse wishard prior.
%
%       Description
%       R = COND_MNORM_INWISH(A, A1, A2, X) generates one sample
%       from the conditional distribution of A given
%       parameter structure X of lower level, structure A1 of
%       same level hyper-parameters and A2 of higher level, i.e 
%       is r~P(A|A1,A2,X). Returns one new sample R from the 
%       distribution above.

% Copyright (c) 1999 Aki Vehtari

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.


n=size(x,1);
r=invwishrand((a2.nu*a2.s+(x-a1.mu)'*(x-a1.mu))/(a2.nu+n), a2.nu+n);
