function p = prior_laplace(do, varargin)
% PRIOR_LAPLACE      Laplace (double exponential) prior structure     
%       
%        Description
%        P = PRIOR_LAPLACE('INIT') returns a structure that specifies
%        Laplace (double exponential) prior.
%    
%	The fields in P are:
%           p.type         = 'Laplace'
%           p.mu           = Location (default 0)
%           p.s            = Scale (default 1)
%           p.fh_pak       = Function handle to parameter packing routine
%           p.fh_unpak     = Function handle to parameter unpacking routine
%           p.fh_e         = Function handle to energy evaluation routine
%           p.fh_g         = Function handle to gradient of energy evaluation routine
%           p.fh_recappend = Function handle to MCMC record appending routine
%
%	P = PRIOR_T('SET', P, 'FIELD1', VALUE1, 'FIELD2', VALUE2, ...)
%       Set the values of fields FIELD1... to the values VALUE1... in LIKELIH. 
%       Fields that can be set: 's', 'nu', 's_prior', 'mu_prior'.
%
%	See also
%       PRIOR_GAMMA, PRIOR_T, PRIOR_LOGLOGUNIF, GPCF_SEXP, LIKELIH_PROBIT


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
        p.type = 'Laplace';
        
        % set functions
        p.fh_pak = @prior_laplace_pak;
        p.fh_unpak = @prior_laplace_unpak;
        p.fh_e = @prior_laplace_e;
        p.fh_g = @prior_laplace_g;
        p.fh_recappend = @prior_laplace_recappend;
        
        % set parameters
        p.mu = 0;
        p.s = 1;
        
        % set parameter priors
        p.p.mu = [];
        p.p.s = [];
        
        if nargin > 1
            if mod(nargin-1,2) ~=0
                error('Wrong number of arguments')
            end
            % Loop through all the parameter values that are changed
            for i=1:2:length(varargin)-1
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
            e = e + feval(p.p.mu.fh_e, p.mu, p.p.mu);
        end
        if ~isempty(p.p.s)
            e = e + feval(p.p.s.fh_e, p.s, p.p.s)  - log(p.s);
        end
    end
    
    function g = prior_laplace_g(x, p)

        g = sign(x-p.mu)./p.s; 
        
        if ~isempty(p.p.mu)
        	gmu = sum(-sign(x-p.mu)./p.s) + feval(p.p.mu.fh_g, p.mu, p.p.mu);
            g = [g gmu];
        end
        if ~isempty(p.p.s)
            gs = (sum( 1./p.s - 1./p.s.^2.*abs(x-p.mu)) + feval(p.p.s.fh_g, p.s, p.p.s)).*p.s - 1;
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