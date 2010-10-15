function p = prior_lognormal(varargin)
%PRIOR_LOGNORMAL  Lognormal prior structure     
%       
%  Description
%    P = PRIOR_LOGNORMAL('FIELD1', VALUE1, 'FIELD2', VALUE2, ...) 
%    returns a structure that specifies lognormal prior.
%    Fields that can be set: 's2', 'mu', 's2_prior', 'mu_prior'.
%    
%    The fields in P are:
%      type         = 'Log-Normal'
%      mu           = Location (default 0)
%      s2           = Scale (default 1)
%      fh_pak       = Function handle to parameter packing routine
%      fh_unpak     = Function handle to parameter unpacking routine
%      fh_e         = Function handle to energy evaluation routine
%      fh_g         = Function handle to gradient of energy evaluation routine
%      fh_recappend = Function handle to MCMC record appending routine
%
%    P = PRIOR_LOGNORMAL('SET', P, 'FIELD1', VALUE1, 'FIELD2', VALUE2, ...)
%    Set the fields FIELD1... to the values VALUE1... in LIKELIH. 
%    Fields that can be set: 's2', 'mu', 's2_prior', 'mu_prior'.
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
      p.type = 'Log-Normal';
      
      % set functions
      p.fh_pak = @prior_lognormal_pak;
      p.fh_unpak = @prior_lognormal_unpak;
      p.fh_e = @prior_lognormal_e;
      p.fh_g = @prior_lognormal_g;
      p.fh_recappend = @prior_lognormal_recappend;
      
      % set parameters
      p.mu = 0;
      p.s2 = 1;
      
      % set parameter priors
      p.p.mu = [];
      p.p.s2 = [];
      
      if numel(varargin) > 0 & mod(numel(varargin),2) ~=0
        error('Wrong number of arguments')
      end
      % Loop through all the parameter values that are changed
      for i=1:2:numel(varargin)-1
        switch varargin{i}
          case 'mu'
            p.mu = varargin{i+1};
          case 's2'
            p.s2 = varargin{i+1};
          case 'mu_prior'
            p.p.mu = varargin{i+1};
          case 's2_prior'
            p.p.s2 = varargin{i+1};                    
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
          case 's2'
            p.s2 = varargin{i+1};
          otherwise
            error('Wrong parameter name!')
        end
      end
  end

  
  function w = prior_lognormal_pak(p)
    
    w = [];
    if ~isempty(p.p.mu)
      w = p.mu;
    end
    if ~isempty(p.p.s2)
      w = [w log(p.s2)];
    end
  end
  
  function [p, w] = prior_lognormal_unpak(p, w)

    if ~isempty(p.p.mu)
      i1=1;
      p.mu = w(i1);
      w = w(i1+1:end);
    end
    if ~isempty(p.p.s2)
      i1=1;
      p.s2 = exp(w(i1));
      w = w(i1+1:end);
    end
  end
  
  function e = prior_lognormal_e(x, p)
    
    e = 0.5*sum(log(x.^2.*p.s2*2*pi) + 1./p.s2 .* sum((log(x)-p.mu).^2,1));
    
    if ~isempty(p.p.mu)
      e = e + feval(p.p.mu.fh_e, p.mu, p.p.mu);
    end
    if ~isempty(p.p.s2)
      e = e + feval(p.p.s2.fh_e, p.s2, p.p.s2)  - log(p.s2);
    end
  end
  
  function g = prior_lognormal_g(x, p)
    
    g = (1./(x.*p.s2)).*(log(x)-p.mu+p.s2);
    
    if ~isempty(p.p.mu)
      gmu = sum(-(1./p.s2).*(log(x)-p.mu)) + feval(p.p.mu.fh_g, p.mu, p.p.mu);
      g = [g gmu];
    end
    if ~isempty(p.p.s2)
      gs2 = (sum( 0.5*(1./p.s2-1./p.s2.^2.*(log(x)-p.mu).^2 )) + feval(p.p.s2.fh_g, p.s2, p.p.s2)).*p.s2 - 1;
      g = [g gs2];
    end
  end
  
  function rec = prior_lognormal_recappend(rec, ri, p)
  % The parameters are not sampled in any case.
    rec = rec;
    if ~isempty(p.p.mu)
      rec.mu(ri) = p.mu;
    end
    if ~isempty(p.p.s2)
      rec.s2(ri) = p.s2;
    end
  end    
end