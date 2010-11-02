function p = prior_invgam(varargin)
%PRIOR_INVGAM  Inverse-gamma prior structure     
%       
%  Description
%    P = PRIOR_INVGAMMA('PARAM1', VALUE1, 'PARAM2', VALUE2, ...) 
%    creates Gamma prior structure in which the named parameters
%    have the specified values. Any unspecified parameters are set
%    to default values.
%
%    P = PRIOR_INVGAMMA(P, 'PARAM1', VALUE1, 'PARAM2', VALUE2, ...)
%    modify a prior structure with the named parameters altered
%    with the specified values.
%  
%    Parameterisation is done by Bayesian Data Analysis,  
%    second edition, Gelman et.al 2004.
%
%    Parameters for Gamma prior [default]
%      sh       - shape [4]
%      s        - scale [1]
%      sh_prior - prior for sh [prior_fixed]
%      s_prior  - prior for s [prior_fixed]
%
%  See also
%    PRIOR_*


% Copyright (c) 2000-2001,2010 Aki Vehtari
% Copyright (c) 2010 Jaakko Riihimäki

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.

  if nargin < 1
    do='init';
  elseif ischar(varargin{1})
    switch varargin{1}
      case 'init'
        do='init';varargin(1)=[];
      case 'set'
        do='set';varargin(1)=[];
      otherwise
        do='init';
    end
  elseif isstruct(varargin{1})
    do='set';
  else
    error('Unknown first argument');
  end

  switch do 
    case 'init'
      % Initialize the prior structure
      p.type = 'Invgam';
      
      % set functions
      p.fh.pak = @prior_invgam_pak;
      p.fh.unpak = @prior_invgam_unpak;
      p.fh.e = @prior_invgam_e;
      p.fh.g = @prior_invgam_g;
      p.fh.recappend = @prior_invgam_recappend;
      
      % set parameters
      p.sh = 4;
      p.s = 1;
      
      % set parameter priors
      p.p.sh = [];
      p.p.s = [];
      
      if numel(varargin) > 0 & mod(numel(varargin),2) ~=0
        error('Wrong number of arguments')
      end
      % Loop through all the parameter values that are changed
      for i=1:2:numel(varargin)-1
        switch varargin{i}
          case 'sh'
            p.sh = varargin{i+1};
          case 's'
            p.s = varargin{i+1};
          case 'sh_prior'
            p.p.sh = varargin{i+1};
          case 's_prior'
            p.p.s = varargin{i+1};                    
          otherwise
            error('Wrong parameter name!')
        end
      end

    case 'set'
      % Set the parameter values of the prior
      if numel(varargin)~=1 & mod(numel(varargin),2) ~=1
        error('Wrong number of arguments')
      end
      p = varargin{1};
      % Loop through all the parameter values that are changed
      for i=2:2:numel(varargin)-1
        switch varargin{i}
          case 'sh'
            p.sh = varargin{i+1};
          case 's'
            p.s = varargin{i+1};
          otherwise
            error('Wrong parameter name!')
        end
      end
  end

  
  function w = prior_invgam_pak(p)
    
    w = [];
    if ~isempty(p.p.sh)
      w = log(p.sh);
    end
    if ~isempty(p.p.s)
      w = [w log(p.s)];
    end
  end
  
  function [p, w] = prior_invgam_unpak(p, w)

    if ~isempty(p.p.sh)
      i1=1;
      p.sh = exp(w(i1));
      w = w(i1+1:end);
    end
    if ~isempty(p.p.s)
      i1=1;
      p.s = exp(w(i1));
      w = w(i1+1:end);
    end
  end
  
  function e = prior_invgam_e(x, p)
    
    e = sum(p.s./x + (p.sh+1).*log(x) -p.sh.*log(p.s)  + gammaln(p.sh));
    
    if ~isempty(p.p.sh)
      e = e + feval(p.p.sh.fh.e, p.sh, p.p.sh) - log(p.sh);
    end
    if ~isempty(p.p.s)
      e = e + feval(p.p.s.fh.e, p.s, p.p.s)  - log(p.s);
    end
  end
  
  function g = prior_invgam_g(x, p)
    
    g = (p.sh+1)./x - p.s./x.^2;
    
    if ~isempty(p.p.sh)
      gsh = (sum(digamma1(p.sh) - log(p.s) + log(x) ) + feval(p.p.sh.fh.g, p.sh, p.p.sh)).*p.sh - 1;
      g = [g gsh];
    end
    if ~isempty(p.p.s)
      gs = (sum(-p.sh./p.s+1./x) + feval(p.p.s.fh.g, p.s, p.p.s)).*p.s - 1;
      g = [g gs];
    end
    
  end
  
  function rec = prior_invgam_recappend(rec, ri, p)
  % The parameters are not sampled in any case.
    rec = rec;
    if ~isempty(p.p.sh)
      rec.sh(ri) = p.sh;
    end
    if ~isempty(p.p.s)
      rec.s(ri) = p.s;
    end
  end
end