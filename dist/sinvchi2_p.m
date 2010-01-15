function q = sinvchi2_p(pr)
% SINVCHI2_P Create a scaled inversi-chi-square prior
%
%          Description
%          Q = SINVCHI2_P(PR) takes a prior array PR containing prior 
%          hyper-parameter values and returns a prior structure Q 
%          containing a prior information.
%
%          Parameterisation is done by Bayesian Data Analysis,  
%          second edition, Gelman et.al 2004.
% 
%          see also
%

% Copyright (c) 1999-2000 Aki Vehtari
% Copyright (c) 2006      Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

q.f='sinvchi2';
q.fe=str2fun('sinvchi2_e');
q.fg=str2fun('sinvchi2_g');
q.a.s2=pr{1};
q.a.nu=pr{2};
q.p=[];
if length(pr) > 2
  q.p.s2.f='sinvchi2';
  q.p.s2.fe=str2fun('sinvchi2_e');
  q.p.s2.fg=str2fun('sinvchi2_g');
  q.p.s2.a.s2=pr{3};
  q.p.s2.a.nu=pr{4};
  q.p.s2.p=[];
end
