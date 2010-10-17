function p = prior_logunif(varargin)
%PRIOR_LOGUNIF  Uniform prior structure for the logarithm of the parameter
%       
%  Description
%    P = PRIOR_LOGUNIF creates uniform prior structure for the
%    logarithm of the parameter.
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
      p.type = 'Log-uniform';
      
      % set functions
      p.fh_pak = @prior_logunif_pak;
      p.fh_unpak = @prior_logunif_unpak;
      p.fh_e = @prior_logunif_e;
      p.fh_g = @prior_logunif_g;
      p.fh_recappend = @prior_logunif_recappend;
      
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

  
  function w = prior_logunif_pak(p)
    w = [];
  end
  
  function [p, w] = prior_logunif_unpak(p, w)
    w = w;
    p = p;
  end
  
  function e = prior_logunif_e(x, p)
    e = sum(log(x));   % = - log(1./x)
                       % where the -log comes from the definition of 
                       % energy as -log( p(x) )
  end
  
  function g = prior_logunif_g(x, p)
    g = 1./x;
  end
  
  function rec = prior_logunif_recappend(rec, ri, p)
  % The parameters are not sampled in any case.
    rec = rec;
  end
  
end