function p = prior_sqrtt(varargin)
%PRIOR_SQRTT  Student-t prior structure for the square root of the parameter
%       
%  Description
%    P = PRIOR_SQRTT('FIELD1', VALUE1, 'FIELD2', VALUE2, ...) 
%    returns a structure that specifies Student's t-distribution
%    prior for the square root of the parameter. Fields that can be
%    set: 'mu', 's2', 'nu', 'mu_prior', 's2_prior', 'nu_prior'.
%
%    Parameterisation is done as in Bayesian Data Analysis,  
%    second edition, Gelman et.al 2004.
%    
%    The fields in P are:
%      type         = 'Sqrt-Student-t'
%      mu           = Location (default 0)
%      s2           = Scale (default 1)
%      nu           = Degrees of freedom (default 4)
%      fh_pak       = Function handle to parameter packing routine
%      fh_unpak     = Function handle to parameter unpacking routine
%      fh_e         = Function handle to energy evaluation routine
%      fh_g         = Function handle to gradient of energy evaluation routine
%      fh_recappend = Function handle to MCMC record appending routine
%
%    P = PRIOR_SQRTT('SET', P, 'FIELD1', VALUE1, 'FIELD2', VALUE2, ...)
%    Set the fields FIELD1... to the values VALUE1... in LIKELIH. 
%    Fields that can be set: 'mu', 's2', 'nu', 'mu_prior', 's2_prior',
%    'nu_prior'.
%
%  See also
%    PRIOR_T, PRIOR_SQRTUNIF

% Copyright (c) 2000-2001,2010 Aki Vehtari
% Copyright (c) 2009 Jarno Vanhatalo
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
      p.type = 'Sqrt-Student-t';
      
      % set functions
      p.fh_pak = @prior_sqrtt_pak;
      p.fh_unpak = @prior_sqrtt_unpak;
      p.fh_e = @prior_sqrtt_e;
      p.fh_g = @prior_sqrtt_g;
      p.fh_recappend = @prior_sqrtt_recappend;
      
      % set parameters
      p.mu = 0;
      p.s2 = 1;
      p.nu = 4;
      
      % set parameter priors
      p.p.mu = [];
      p.p.s2 = [];
      p.p.nu = [];
      
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
          case 'nu'
            p.nu = varargin{i+1};
          case 'mu_prior'
            p.p.mu = varargin{i+1};
          case 's2_prior'
            p.p.s2 = varargin{i+1};
          case 'nu_prior'
            p.p.nu = varargin{i+1};                    
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
          case 'nu'
            p.nu = varargin{i+1};
          otherwise
            error('Wrong parameter name!')
        end
      end
  end

  
  function w = prior_sqrtt_pak(p)
    
    w = [];
    if ~isempty(p.p.mu)
      w = p.mu;
    end        
    if ~isempty(p.p.s2)
      w = [w log(p.s2)];
    end
    if ~isempty(p.p.nu)
      w = [w log(p.nu)];
    end
  end
  
  function [p, w] = prior_sqrtt_unpak(p, w)
    
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
    if ~isempty(p.p.nu)
      i1=1;
      p.nu = exp(w(i1));
      w = w(i1+1:end);
    end
  end
  
  function e = prior_sqrtt_e(x, p)
    
    e=sum(-gammaln((p.nu+1)./2) + gammaln(p.nu./2) + 0.5*log(p.nu.*pi.*p.s2) + (p.nu+1)./2.*log(1+(x-p.mu).^2./p.nu./p.s2) + 2*sqrt(x));
    
    if ~isempty(p.p.mu)
      e = e + feval(p.p.mu.fh_e, p.mu, p.p.mu);
    end
    if ~isempty(p.p.s2)
      e = e + feval(p.p.s2.fh_e, p.s2, p.p.s2) - log(p.s2);
    end
    if ~isempty(p.p.nu)
      e = e + feval(p.p.nu.fh_e, p.nu, p.p.nu) - log(p.nu);
    end
  end
  
  function g = prior_sqrtt_g(x, p)

    g=(p.nu+1).* (x-p.mu) ./ (p.nu.*p.s2 + (x-p.mu).^2) + 1/sqrt(x);
    
    if ~isempty(p.p.mu)
      gmu = sum( -(p.nu+1).* (x-p.mu) ./ (p.nu.*p.s2 + (x-p.mu).^2) ) + feval(p.p.mu.fh_g, p.mu, p.p.mu);
      g = [g gmu];
    end
    if ~isempty(p.p.s2)
      gs2 = (sum( 1./(2.*p.s2) -((p.nu + 1)*(p.mu - x)^2)/(2*p.s2*((p.mu-x)^2 + p.nu*p.s2))) + feval(p.p.s2.fh_g, p.s2, p.p.s2)).*p.s2 - 1;
      g = [g gs2];
    end
    if ~isempty(p.p.nu)
      gnu = (0.5*sum( -digamma1((p.nu+1)./2)+digamma1(p.nu./2)+1./p.nu+log(1+(x-p.mu).^2./p.nu./p.s2)-(p.nu+1)./(1+(x-p.mu).^2./p.nu./p.s2).*(x-p.mu).^2./p.s2./p.nu.^2) + feval(p.p.nu.fh_g, p.nu, p.p.nu)).*p.nu - 1;
      g = [g gnu];
    end
  end
  
  function rec = prior_sqrtt_recappend(rec, ri, p)
  % The parameters are not sampled in any case.
    rec = rec;
    if ~isempty(p.p.mu)
      rec.mu(ri) = p.mu;
    end        
    if ~isempty(p.p.s2)
      rec.s2(ri) = p.s2;
    end
    if ~isempty(p.p.nu)
      rec.nu(ri) = p.nu;
    end
  end
end
