function p = prior_sqrtunif(varargin)
%PRIOR_SQRTUNIF  Uniform prior structure for the square root of the parameter
%       
%  Description
%    P = PRIOR_SQRTUNIF creates uniform prior structure for the
%    square root of the parameter.
%    
%  See also
%    PRIOR_*
  
% Copyright (c) 2009 Jarno Vanhatalo
% Copyright (c) 2010 Jaakko Riihimäki
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
      p.type = 'Sqrt-uniform';
      
      % No paramaters to init
      if numel(varargin) > 0
        error('Wrong number of arguments')
      end
      
      % set functions
      p.fh.pak = @prior_sqrtunif_pak;
      p.fh.unpak = @prior_sqrtunif_unpak;
      p.fh.e = @prior_sqrtunif_e;
      p.fh.g = @prior_sqrtunif_g;
      p.fh.recappend = @prior_sqrtunif_recappend;

    case 'set'
      % No paramaters to set
      if numel(varargin)~=1
        error('Wrong number of arguments')
      end
      
      % Set the parameter values of the prior
      p = varargin{1};

  end

  function [w,s] = prior_sqrtunif_pak(p)
    w=[];s={};
  end
  
  function [p, w] = prior_sqrtunif_unpak(p, w)
    w = w;
    p = p;
  end
  
  function e = prior_sqrtunif_e(x, p)
    e = sum(2*sqrt(x));% = -log(1/(2*sqrt(x)))
                       % where the -log comes from the definition of 
                       % energy as -log( p(x) )
  end
  
  function g = prior_sqrtunif_g(x, p)
    g = 1./sqrt(x);
  end
  
  function rec = prior_sqrtunif_recappend(rec, ri, p)
  % The parameters are not sampled in any case.
    rec = rec;
  end
  
end
