function p = prior_laplace(varargin)
%PRIOR_LAPLACE  Laplace (double exponential) prior structure     
%       
%  Description
%    P = PRIOR_LAPLACE('PARAM1', VALUE1, 'PARAM2', VALUE2, ...)
%    creates Laplace prior structure in which the named parameters
%    have the specified values. Any unspecified parameters are set
%    to default values.
%    
%    P = PRIOR_LAPLACE(P, 'PARAM1', VALUE1, 'PARAM2', VALUE2, ...)
%    modify a prior structure with the named parameters altered
%    with the specified values.
%
%    Parameters for Laplace prior [default]
%      mu       - location [0]
%      s        - scale [1]
%      mu_prior - prior for mu [prior_fixed]
%      s_prior  - prior for s  [prior_fixed]
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
      p.type = 'Laplace';
      
      % set functions
      p.fh.pak = @prior_laplace_pak;
      p.fh.unpak = @prior_laplace_unpak;
      p.fh.e = @prior_laplace_e;
      p.fh.g = @prior_laplace_g;
      p.fh.recappend = @prior_laplace_recappend;
      
      % set parameters
      p.mu = 0;
      p.s = 1;
      
      % set parameter priors
      p.p.mu = [];
      p.p.s = [];
      
      if numel(varargin) > 0 & mod(numel(varargin),2) ~=0
        error('Wrong number of arguments')
      end
      % Loop through all the parameter values that are changed
      for i=1:2:numel(varargin)-1
        switch varargin{i}
          case 'mu'
            p.mu = varargin{i+1};
          case 's'
            p.s = varargin{i+1};
          case 'mu_prior'
            p.p.mu = varargin{i+1};
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
          case 'mu'
            p.mu = varargin{i+1};
          case 's'
            p.s = varargin{i+1};
          otherwise
            error('Wrong parameter name!')
        end
      end
  end

  
  function w = prior_laplace_pak(p)
    
    w = [];
    if ~isempty(p.p.mu)
      w = p.mu;
    end
    if ~isempty(p.p.s)
      w = [w log(p.s)];
    end
  end
  
  function [p, w] = prior_laplace_unpak(p, w)
    
    if ~isempty(p.p.mu)
      i1=1;
      p.mu = w(i1);
      w = w(i1+1:end);
    end
    if ~isempty(p.p.s)
      i1=1;
      p.s = exp(w(i1));
      w = w(i1+1:end);
    end
  end
  
  function e = prior_laplace_e(x, p)
    
    e = sum(log(2*p.s) + 1./p.s.* abs(x-p.mu));
    
    if ~isempty(p.p.mu)
      e = e + feval(p.p.mu.fh.e, p.mu, p.p.mu);
    end
    if ~isempty(p.p.s)
      e = e + feval(p.p.s.fh.e, p.s, p.p.s)  - log(p.s);
    end
  end
  
  function g = prior_laplace_g(x, p)

    g = sign(x-p.mu)./p.s; 
    
    if ~isempty(p.p.mu)
      gmu = sum(-sign(x-p.mu)./p.s) + feval(p.p.mu.fh.g, p.mu, p.p.mu);
      g = [g gmu];
    end
    if ~isempty(p.p.s)
      gs = (sum( 1./p.s - 1./p.s.^2.*abs(x-p.mu)) + feval(p.p.s.fh.g, p.s, p.p.s)).*p.s - 1;
      g = [g gs];
    end
  end
  
  function rec = prior_laplace_recappend(rec, ri, p)
  % The parameters are not sampled in any case.
    rec = rec;
    if ~isempty(p.p.mu)
      rec.mu(ri) = p.mu;
    end
    if ~isempty(p.p.s)
      rec.s(ri) = p.s;
    end
  end    
end