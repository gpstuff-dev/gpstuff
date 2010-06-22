function p = prior_gamma(do, varargin)
% PRIOR_GAMMA     Gamma prior structure     
%       
%        Description
%        P = PRIOR_GAMMA('INIT') returns a structure that specifies Gamma
%        prior.
%
%        Parameterisation is done by Bayesian Data Analysis,  
%        second edition, Gelman et.al 2004.
%
%	The fields in P are:
%           p.type         = 'Gamma'
%           p.sh           = Shape (default 4)
%           p.is           = Inverse scale (default 1)
%           p.fh_pak       = Function handle to parameter packing routine
%           p.fh_unpak     = Function handle to parameter unpacking routine
%           p.fh_e         = Function handle to energy evaluation routine
%           p.fh_g         = Function handle to gradient of energy evaluation routine
%           p.fh_recappend = Function handle to MCMC record appending routine
%
%	P = PRIOR_T('SET', P, 'FIELD1', VALUE1, 'FIELD2', VALUE2, ...)
%       Set the values of fields FIELD1... to the values VALUE1... in LIKELIH. 
%       Fields that can be set: 'sh', 'is', 'sh_prior', 'is_prior'.
%
%	See also
%       PRIOR_T, PRIOR_INVGAM, PRIOR_NORMAL, GPCF_SEXP, LIKELIH_PROBIT

    
% Copyright (c) 2000-2001 Aki Vehtari
% Copyright (c) 2010 Jaakko Riihimäki

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.

    
    if nargin < 1
        error('Not enough arguments')
    end

    % Initialize the prior structure
    if strcmp(do, 'init')
        p.type = 'Gamma';
        
        % set functions
        p.fh_pak = @prior_gamma_pak;
        p.fh_unpak = @prior_gamma_unpak;
        p.fh_e = @prior_gamma_e;
        p.fh_g = @prior_gamma_g;
        p.fh_recappend = @prior_gamma_recappend;
        
        % set parameters
        p.sh = 4;
        p.is = 1;
        
        % set parameter priors
        p.p.sh = [];
        p.p.is = [];
        
        if nargin > 1
            if mod(nargin-1,2) ~=0
                error('Wrong number of arguments')
            end
            % Loop through all the parameter values that are changed
            for i=1:2:length(varargin)-1
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
              case 'sh'
                p.sh = varargin{i+1};
              case 'is'
                p.is = varargin{i+1};
              otherwise
                error('Wrong parameter name!')
            end
        end
    end

    
    function w = prior_gamma_pak(p)
        
        w = [];
        if ~isempty(p.p.sh)
            w = log(p.sh);
        end
         if ~isempty(p.p.is)
            w = [w log(p.is)];
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
            e = e + feval(p.p.sh.fh_e, p.sh, p.p.sh) - log(p.sh);
        end
        if ~isempty(p.p.is)
            e = e + feval(p.p.is.fh_e, p.is, p.p.is)  - log(p.is);
        end
    end
    
    function g = prior_gamma_g(x, p)
       
        g = (1-p.sh)./x + p.is;
        
        if ~isempty(p.p.sh)
        	gsh = (sum(digamma1(p.sh) - log(p.is) - log(x) ) + feval(p.p.sh.fh_g, p.sh, p.p.sh)).*p.sh - 1;
            g = [g gsh];
        end
        if ~isempty(p.p.is)
            gis = (sum(-p.sh./p.is+x) + feval(p.p.is.fh_g, p.is, p.p.is)).*p.is - 1;
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