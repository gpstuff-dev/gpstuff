function q = laplace_p(pr)
% LAPLACE_P Create Laplace (double exponential) prior
%
%        Description
%        Q = NORM_P(PR) takes mlp networks prior hyper-parameters
%        array PR and returns a structure that specifies Laplace
%        double exponential prior distribution for a
%        given weight or bias PR represents. LAPLACE_P creates
%        function handles to evaluate hyper-parameter error and
%        gradient for a given set of hyper-parameters PR.

% Copyright (c) 1999-2003 Aki Vehtari

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

q.f='laplace';
q.fe=str2fun('laplace_e');
q.fg=str2fun('laplace_g');
q.a.s=pr{1};
if length(pr) > 1
  q.p.s.f='invgam';
  q.p.s.fe=str2fun('invgam_e');
  q.p.s.fg=str2fun('invgam_g');
  q.p.s.a.s=pr{2};
  q.p.s.a.nu=pr{3};
end
