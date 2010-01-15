function p = prior_t(do, varargin)

% PRIOR_T     Student-t prior structure     
%       
%        Description
%        P = PRIOR_T('INIT') returns a structure that specifies Student's
%        t-distribution prior. 
%    
%	The fields in P are:
%           p.type         = 'Student-t'
%           p.s            = The scale (default 1)
%           p.nu           = The degrees of freedom (default 4)
%           p.fh_pak       = Function handle to parameter packing routine
%           p.fh_unpak     = Function handle to parameter unpacking routine
%           p.fh_e         = Function handle to energy evaluation routine
%           p.fh_g         = Function handle to gradient of energy evaluation routine
%           p.fh_recappend = Function handle to MCMC record appending routine
%
%	P = PRIOR_T('SET', P, 'FIELD1', VALUE1, 'FIELD2', VALUE2, ...)
%       Set the values of fields FIELD1... to the values VALUE1... in LIKELIH. 
%       Fields that can be set are 's' and 'nu'.
%
%	See also
%       PRIOR_GAMMA, GPCF_SEXP, LIKELIH_PROBIT

    
% Copyright (c) 2000-2001 Aki Vehtari
% Copyright (c) 2009 Jarno Vanhatalo
% Copyright (c) 2010 Jaakko Riihimäki

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.

    
    if nargin < 1
        error('Not enough arguments')
    end

    % Initialize the prior structure
    if strcmp(do, 'init')
        p.type = 'Student-t';
        
        % set functions
        p.fh_pak = @prior_t_pak;
        p.fh_unpak = @prior_t_unpak;
        p.fh_e = @prior_t_e;
        p.fh_g = @prior_t_g;
        p.fh_recappend = @prior_t_recappend;
        
        % set parameters
        p.s = 1;
        p.nu = 4;
        
        % set parameter priors
        p.p.s = [];
        p.p.nu = [];
        
        if nargin > 1
            if mod(nargin-1,2) ~=0
                error('Wrong number of arguments')
            end
            % Loop through all the parameter values that are changed
            for i=1:2:length(varargin)-1
                switch varargin{i}
                  case 'scale'
                    p.s = varargin{i+1};
                  case 'nu'
                    p.nu = varargin{i+1};
                  case 'scale_prior'
                    p.p.s = varargin{i+1};
                  case 'nu_prior'
                    p.p.nu = varargin{i+1};                    
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
              case 'scale'
                p.s = varargin{i+1};
              case 'nu'
                p.nu = varargin{i+1};
              otherwise
                error('Wrong parameter name!')
            end
        end
    end

    
    function w = prior_t_pak(p)
        
        w = [];
        if ~isempty(p.p.s)
            w = log(p.s);
        end
         if ~isempty(p.p.nu)
            w = [w log(p.nu)];
        end
    end
    
    function [p, w] = prior_t_unpak(p, w)

        if ~isempty(p.p.s)
            i1=1;
            p.s = exp(w(i1));
            w = w(i1+1:end);
        end
        if ~isempty(p.p.nu)
            i1=1;
            p.nu = exp(w(i1));
            w = w(i1+1:end);
        end
    end
    
    function e = prior_t_e(x, p)
        e = sum(log(1 + (x./p.s).^2 ./ p.nu)).* (p.nu+1)/2;
        
        if ~isempty(p.p.s)
            e = e + feval(p.p.s.fh_e, p.s, p.p.s) - log(p.s);
        end
        if ~isempty(p.p.nu)
            e = e + feval(p.p.nu.fh_e, p.nu, p.p.nu)  - log(p.nu);
        end
    end
    
    function g = prior_t_g(x, p)
        d=x./p.s;
        g=(p.nu+1)./p.nu .* (d./p.s) ./ (1 + (d.^2)./p.nu);
        
        if ~isempty(p.p.s)
        	gsigma = (sum(-(p.nu+1)./(p.nu.*p.s^3).*x.^2./(1+d.^2./p.nu)) + feval(p.p.s.fh_g, p.s, p.p.s)).*p.s - 1;
            g = [g gsigma];
        end
        if ~isempty(p.p.nu)
            gnu = (sum(0.5*log(1+d.^2./p.nu) - (p.nu+1)./(2*p.nu.^2).*d.^2./(1+d.^2./p.nu)) + feval(p.p.nu.fh_g, p.nu, p.p.nu)).*p.nu - 1;
            g = [g gnu];
        end
    end
    
    function rec = prior_t_recappend(rec, ri, p)
    % The parameters are not sampled in any case.
        rec = rec;
        if ~isempty(p.p.s)
        	rec.s(ri) = p.s;
        end
        if ~isempty(p.p.nu)
        	rec.nu(ri) = p.nu;
        end
    end    
end