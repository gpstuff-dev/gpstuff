function p = prior_sqrtunif(do, varargin)
% PRIOR_SQRTUNIF  Uniform prior structure for the square root of the parameter
%       
%       Description
%       P = PRIOR_SQRTUNIF('INIT') returns a structure that
%       specifies uniform prior for the square root of the
%       parameters.
%    
%	The fields in P are:
%           p.type         = 'sqrt-uniform'
%           p.fh_pak       = Function handle to parameter packing routine
%           p.fh_unpak     = Function handle to parameter unpacking routine
%           p.fh_e         = Function handle to energy evaluation routine
%           p.fh_g         = Function handle to gradient of energy evaluation routine
%           p.fh_recappend = Function handle to MCMC record appending routine
%
%	P = PRIOR_SQRTUNIF('SET', P, 'FIELD1', VALUE1, 'FIELD2', VALUE2, ...)
%       Set the values of fields FIELD1... to the values VALUE1... in LIKELIH. 
%
%	See also
%       PRIOR_UNIF
    
% Copyright (c) 2009 Jarno Vanhatalo
% Copyright (c) 2010 Jaakko Riihimäki, Aki Vehtari

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.

    
    if nargin < 1
        error('Not enough arguments')
    end

    % Initialize the prior structure
    if strcmp(do, 'init')
        p.type = 'sqrt-uniform';
        
        % set functions
        p.fh_pak = @prior_sqrtunif_pak;
        p.fh_unpak = @prior_sqrtunif_unpak;
        p.fh_e = @prior_sqrtunif_e;
        p.fh_g = @prior_sqrtunif_g;
        p.fh_recappend = @prior_sqrtunif_recappend;
                        
        if nargin > 1
            if mod(nargin-1,2) ~=0
                error('Wrong number of arguments')
            end
            % Loop through all the parameter values that are changed
            for i=1:2:length(varargin)-1
                switch varargin{i}
                  otherwise
                    error('Wrong parameter name!')
                end
            end
        end

    end
    
    % Set the parameter values of the prior
    if strcmp(do, 'set')
        if mod(nargin,2) ~=0
            error('Wrong number of arguments')
        end
        p = varargin{1};
        % Loop through all the parameter values that are changed
        for i=2:2:length(varargin)-1
            switch varargin{i}
              otherwise
                error('Wrong parameter name!')
            end
        end
    end

    function w = prior_sqrtunif_pak(p)
        w = [];
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
  