function q = t_p(pr)
% T_P      Create Student's-t prior
%
%        Description
%        Q = T_P(PR) takes mlp networks prior hyper-parameters
%        array PR and returns a structure that specifies Student's
%        t-distribution prior for given weight or bias PR 
%        represents. T_P creates function handles to evaluate 
%        hyper-parameter error and gradient for a given set of 
%        hyper-parameters PR.

% Copyright (c) 1999-2000 Aki Vehtari

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

q.f='t';
q.fe=str2fun('t_e');
q.fg=str2fun('t_g');
%q.a.mu=0;
q.a.s=pr{1};
q.a.nu=pr{2};
if length(pr) > 2
  q.p.s.f='invgam';
  q.p.s.fe=str2fun('invgam_e');
  q.p.s.fg=str2fun('invgam_g');
  q.p.s.a.s=pr{3};
  q.p.s.a.nu=pr{4};
end
if length(pr) > 4
  q.p.nu.f='cat';
  q.p.nu.a.nus=pr{5};
end
