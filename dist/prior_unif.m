function p = prior_unif(varargin)
%PRIOR_UNIF  Uniform prior structure     
%       
%  Description
%    P = PRIOR_UNIF creates uniform prior structure.
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
      
      % No paramaters to init
      if numel(varargin) > 0
        error('Wrong number of arguments')
      end
      
      % set functions
      p.fh.pak = @prior_unif_pak;
      p.fh.unpak = @prior_unif_unpak;
      p.fh.e = @prior_unif_e;
      p.fh.g = @prior_unif_g;
      p.fh.recappend = @prior_unif_recappend;

    case 'set'
      % No paramaters to set
      if numel(varargin)~=1
        error('Wrong number of arguments')
      end
      
      % Set the parameter values of the prior
      p = varargin{1};

  end

  
  function [w,s] = prior_unif_pak(p, w)
    w=[];s={};
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