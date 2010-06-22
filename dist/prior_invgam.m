function p = prior_invgam(do, varargin)
% PRIOR_INVGAM     Inverse-gamma prior structure     
%       
%        Description
%        P = PRIOR_NORMAL('INIT') returns a structure that specifies
%        Inverse-gamma prior.
%
%        Parameterisation is done by Bayesian Data Analysis,  
%        second edition, Gelman et.al 2004.
%    
%	The fields in P are:
%           p.type         = 'Invgam'
%           p.sh           = Shape (default 4)
%           p.s            = Scale (default 1)
%           p.fh_pak       = Function handle to parameter packing routine
%           p.fh_unpak     = Function handle to parameter unpacking routine
%           p.fh_e         = Function handle to energy evaluation routine
%           p.fh_g         = Function handle to gradient of energy evaluation routine
%           p.fh_recappend = Function handle to MCMC record appending routine
%
%	P = PRIOR_T('SET', P, 'FIELD1', VALUE1, 'FIELD2', VALUE2, ...)
%       Set the values of fields FIELD1... to the values VALUE1... in LIKELIH. 
%       Fields that can be set: 'sh', 's', 'sh_prior', 's_prior'.
%
%	See also
%       PRIOR_GAMMA, PRIOR_LAPLACE, PRIOR_T, GPCF_SEXP, LIKELIH_PROBIT

    
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
        p.type = 'Invgam';
        
        % set functions
        p.fh_pak = @prior_invgam_pak;
        p.fh_unpak = @prior_invgam_unpak;
        p.fh_e = @prior_invgam_e;
        p.fh_g = @prior_invgam_g;
        p.fh_recappend = @prior_invgam_recappend;
        
        % set parameters
        p.sh = 4;
        p.s = 1;
        
        % set parameter priors
        p.p.sh = [];
        p.p.s = [];
        
        if nargin > 1
            if mod(nargin-1,2) ~=0
                error('Wrong number of arguments')
            end
            % Loop through all the parameter values that are changed
            for i=1:2:length(varargin)-1
                switch varargin{i}
                  case 'sh'
                    p.sh = varargin{i+1};
                  case 's'
                    p.s = varargin{i+1};
                  case 'sh_prior'
                    p.p.sh = varargin{i+1};
                  case 's_prior'
                    p.p.s = varargin{i+1};                    
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
              case 's'
                p.s = varargin{i+1};
              otherwise
                error('Wrong parameter name!')
            end
        end
    end

    
    function w = prior_invgam_pak(p)
        
        w = [];
        if ~isempty(p.p.sh)
            w = log(p.sh);
        end
         if ~isempty(p.p.s)
            w = [w log(p.s)];
        end
    end
    
    function [p, w] = prior_invgam_unpak(p, w)

        if ~isempty(p.p.sh)
            i1=1;
            p.sh = exp(w(i1));
            w = w(i1+1:end);
        end
        if ~isempty(p.p.s)
            i1=1;
            p.s = exp(w(i1));
            w = w(i1+1:end);
        end
    end
    
    function e = prior_invgam_e(x, p)
        
        e = sum(p.s./x + (p.sh+1).*log(x) -p.sh.*log(p.s)  + gammaln(p.sh));
        
        if ~isempty(p.p.sh)
            e = e + feval(p.p.sh.fh_e, p.sh, p.p.sh) - log(p.sh);
        end
        if ~isempty(p.p.s)
            e = e + feval(p.p.s.fh_e, p.s, p.p.s)  - log(p.s);
        end
    end
    
    function g = prior_invgam_g(x, p)
            
        g = (p.sh+1)./x - p.s./x.^2;
        
        if ~isempty(p.p.sh)
        	gsh = (sum(digamma1(p.sh) - log(p.s) + log(x) ) + feval(p.p.sh.fh_g, p.sh, p.p.sh)).*p.sh - 1;
            g = [g gsh];
        end
        if ~isempty(p.p.s)
            gs = (sum(-p.sh./p.s+1./x) + feval(p.p.s.fh_g, p.s, p.p.s)).*p.s - 1;
            g = [g gs];
        end
        
    end
    
    function rec = prior_invgam_recappend(rec, ri, p)
    % The parameters are not sampled in any case.
        rec = rec;
        if ~isempty(p.p.sh)
        	rec.sh(ri) = p.sh;
        end
        if ~isempty(p.p.s)
        	rec.s(ri) = p.s;
        end
    end
end