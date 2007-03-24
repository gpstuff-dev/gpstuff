function q = norm_p(pr)
% NORM_P create Gaussian (multivariate) (hierarchical) prior
%
%        Description
%        Q = NORM_P(PR) takes mlp networks prior hyper-parameters
%        array PR and returns a structure that specifies Gaussian
%        (multivariate) (hierarchical) prior distribution for a
%        given weight or bias PR represents. NORM_P creates
%        function handles to evaluate hyper-parameter error and
%        gradient for a given set of hyper-parameters PR.
%        

% Copyright (c) 1999 Aki Vehtari

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

% create function handles to compute error and gradient for the 
% weight hyper-parameters
if size(pr{1},1) == 1
  q.f='norm';
  q.fe=str2func('norm_e');
  q.fg=str2func('norm_g');
  q.a.s=pr{1};
else
  q.f='mnorm';
  q.fe=str2func('mnorm_e');
  q.fg=str2func('mnorm_g');
  q.a.s=pr{1};
end
% create function handles to compute error and gradient for the 
% first level hyperparameters in hierarchy
if length(pr) > 1
  if size(pr{1},1) == 1
    q.p.s.f='invgam';
    q.p.s.fe=str2func('invgam_e');
    q.p.s.fg=str2func('invgam_g');
    q.p.s.a.s=pr{2};
  else
    q.p.s.f='invwish';
    q.p.s.fe=str2func('invwish_e');
    q.p.s.fg=str2func('invwish_g');
    q.p.s.a.s=pr{2};
  end
  q.p.s.a.nu=pr{3};
end
% create function handles to compute error and gradient for the 
% second level hyperparamerters in hierarchy
if length(pr) > 3
  if size(pr{1},1) == 1
    q.p.s.p.s.f='invgam';
    q.p.s.p.s.fe=str2func('invgam_e');
    q.p.s.p.s.fg=str2func('invgam_g');
    q.p.s.p.s.a.s=pr{4};
  else
    q.p.s.p.s.f='invwish';
    q.p.s.p.s.fe=str2func('invwish_e');
    q.p.s.p.s.fg=str2func('invwish_g');
    q.p.s.p.s.a.s=pr{4};
  end
  q.p.s.p.s.a.nu=pr{5};
else
  % if not hirerachical model, and not mnorm-models
  % check that weight prior is not vector
  if size(pr{1},1) == 1 & size(pr{1},2) > 1
    fprintf('Multiple weight prior values without hierarchical prior.')
    error('Mismatch on prior specification.')
  end
end
