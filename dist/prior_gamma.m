function p = prior_gamma(varargin)
%PRIOR_GAMMA  Gamma prior structure     
%
%  Description
%    P = PRIOR_GAMMA('PARAM1', VALUE1, 'PARAM2', VALUE2, ...) 
%    creates Gamma prior structure in which the named parameters
%    have the specified values. Any unspecified parameters are set
%    to default values.
%
%    P = PRIOR_GAMMA(P, 'PARAM1', VALUE1, 'PARAM2', VALUE2, ...)
%    modify a prior structure with the named parameters altered
%    with the specified values.
%  
%    Parameterisation is done by Bayesian Data Analysis,  
%    second edition, Gelman et.al 2004.
%
%    Parameters for Gamma prior [default]
%      sh       - shape [4]
%      is       - inverse scale [1]
%      sh_prior - prior for sh [prior_fixed]
%      is_prior - prior for is [prior_fixed]
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
      p.type = 'Gamma';
      
      % set parameters
      p.sh = 4;
      p.is = 1;
      
      % set parameter priors
      p.p.sh = [];
      p.p.is = [];
      
      if numel(varargin) > 0 & mod(numel(varargin),2) ~=0
        error('Wrong number of arguments')
      end
      % Loop through all the parameter values that are changed
      for i=1:2:numel(varargin)-1
        switch varargin{i}
          case 'sh'
            p.sh = varargin{i+1};
          case 'is'
            p.is = varargin{i+1};
          case 'sh_prior'
            p.p.sh = varargin{i+1};
          case 'is_prior'
            p.p.is = varargin{i+1};                    
          otherwise
            error('Wrong parameter name!')
        end
      end
      
      % set functions
      p.fh.pak = @prior_gamma_pak;
      p.fh.unpak = @prior_gamma_unpak;
      p.fh.e = @prior_gamma_e;
      p.fh.g = @prior_gamma_g;
      p.fh.recappend = @prior_gamma_recappend;

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
          case 'is'
            p.is = varargin{i+1};
          otherwise
            error('Wrong parameter name!')
        end
      end
  end
  
  function [w, s] = prior_gamma_pak(p)
    
    w=[];s={};
    if ~isempty(p.p.sh)
      w = log(p.sh);
      s=[s; 'log(Gamma.sh)'];
    end
    if ~isempty(p.p.is)
      w = [w log(p.is)];
      s=[s; 'log(Gamma.is)'];
    end
  end
  
  function [p, w] = prior_gamma_unpak(p, w)

    if ~isempty(p.p.sh)
      i1=1;
      p.sh = exp(w(i1));
      w = w(i1+1:end);
    end
    if ~isempty(p.p.is)
      i1=1;
      p.is = exp(w(i1));
      w = w(i1+1:end);
    end
  end
  
  function e = prior_gamma_e(x, p)
    
    e = sum(p.is.*x - (p.sh-1).*log(x) -p.sh.*log(p.is)  + gammaln(p.sh));
    
    if ~isempty(p.p.sh)
      e = e + feval(p.p.sh.fh.e, p.sh, p.p.sh) - log(p.sh);
    end
    if ~isempty(p.p.is)
      e = e + feval(p.p.is.fh.e, p.is, p.p.is)  - log(p.is);
    end
  end
  
  function g = prior_gamma_g(x, p)
    
    g = (1-p.sh)./x + p.is;
    
    if ~isempty(p.p.sh)
      gsh = (sum(digamma1(p.sh) - log(p.is) - log(x) ) + feval(p.p.sh.fh.g, p.sh, p.p.sh)).*p.sh - 1;
      g = [g gsh];
    end
    if ~isempty(p.p.is)
      gis = (sum(-p.sh./p.is+x) + feval(p.p.is.fh.g, p.is, p.p.is)).*p.is - 1;
      g = [g gis];
    end
    
  end
  
  function rec = prior_gamma_recappend(rec, ri, p)
  % The parameters are not sampled in any case.
    rec = rec;
    if ~isempty(p.p.sh)
      rec.sh(ri) = p.sh;
    end
    if ~isempty(p.p.is)
      rec.is(ri) = p.is;
    end
  end    
end