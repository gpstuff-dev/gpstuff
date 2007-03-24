function q = gamma_p(pr)
% GAMMA_P Create Gamma prior
%
%          Description
%          Q = GAMMA_P(PR) takes a prior array PR containing prior 
%          hyper-parameter values and returns a prior structure Q 
%          containing a prior information.
%
%          see also
%

% Copyright (c) 1999-2000 Aki Vehtari
% Copyright (c) 2006      Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

q.f='gamma';
q.fe=str2fun('gamma_e');
q.fg=str2fun('gamma_g');
q.a.s=pr{1};
q.a.nu=pr{2};
q.p=[];
if length(pr) > 2
  q.p.s.f='gamma';
  q.p.s.fe=str2fun('gamma_e');
  q.p.s.fg=str2fun('gamma_g');
  q.p.s.a.s=pr{3};
  q.p.s.a.nu=pr{4};
  q.p.s.p=[];
end
