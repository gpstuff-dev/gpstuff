function p = prior_unif(varargin)
%PRIOR_UNIF  Uniform prior structure     
%       
%  Description
%    P = PRIOR_UNIF returns a structure that specifies uniform
%    prior.
%    
%    The fields in P are:
%      type         = 'Uniform'
%      fh_pak       = Function handle to parameter packing routine
%      fh_unpak     = Function handle to parameter unpacking routine
%      fh_e         = Function handle to energy evaluation routine
%      fh_g         = Function handle to gradient of energy evaluation routine
%      fh_recappend = Function handle to MCMC record appending routine
%
%    P = PRIOR_UNIF('SET', P, 'FIELD1', VALUE1, 'FIELD2', VALUE2, ...)
%    Set the fields FIELD1... to the values VALUE1... in LIKELIH. 
%
%  See also
%    PRIOR_*

% Copyright (c) 2009 Jarno Vanhatalo
% Copyright (c) 2010 Aki Vehtari

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
      p.type = 'Uniform';
      
      % set functions
      p.fh_pak = @prior_unif_pak;
      p.fh_unpak = @prior_unif_unpak;
      p.fh_e = @prior_unif_e;
      p.fh_g = @prior_unif_g;
      p.fh_recappend = @prior_unif_recappend;
      
      % No paramaters to init
      if numel(varargin) > 0
        error('Wrong number of arguments')
      end

    case 'set'
      % No paramaters to set
      if numel(varargin)~=1
        error('Wrong number of arguments')
      end
      
      % Set the parameter values of the prior
      p = varargin{1};

  end

  
  function w = prior_unif_pak(p, w)
    w = [];
  end
  
  function [p, w] = prior_unif_unpak(p, w)
    w = w;
    p = p;
  end
  
  function e = prior_unif_e(x, p)
    e = 0;
  end
  
  function g = prior_unif_g(x, p)
    g = zeros(size(x));
  end
  
  function rec = prior_unif_recappend(rec, ri, p)
  % The parameters are not sampled in any case.
    rec = rec;
  end
  
end