function q = invgam_p(pr)
% INVGAM_P Create inverse-Gamma prior
%
%          Description
%          Q = INVGAM_P(PR) takes a prior array PR containing prior 
%          hyper-parameter values and returns a prior structure Q 
%          containing a prior information.
%
%          see also
%

% Copyright (c) 1999-2000 Aki Vehtari

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

q.f='invgam';
q.fe=str2fun('invgam_e');
q.fg=str2fun('invgam_g');
q.a.s=pr{1};
q.a.nu=pr{2};
q.p=[];
if length(pr) > 2
  q.p.s.f='invgam';
  q.p.s.fe=str2fun('invgam_e');
  q.p.s.fg=str2fun('invgam_g');
  q.p.s.a.s=pr{3};
  q.p.s.a.nu=pr{4};
  q.p.s.p=[];
end
