function p = prior_sinvchi2(do, varargin)

% PRIOR_SINVCHI2     Scaled inverse-chi-square prior structure
%       
%        Description
%        P = PRIOR_SINVCHI2('INIT') returns a structure that specifies
%        Scaled inverse-chi-square prior. 
%    
%	The fields in P are:
%           p.type         = 'Sinvchi2'
%           p.s2           = The scale (default 1)
%           p.nu           = The degrees of freedom (default 4)
%           p.fh_pak       = Function handle to parameter packing routine
%           p.fh_unpak     = Function handle to parameter unpacking routine
%           p.fh_e         = Function handle to energy evaluation routine
%           p.fh_g         = Function handle to gradient of energy evaluation routine
%           p.fh_recappend = Function handle to MCMC record appending routine
%
%	P = PRIOR_SINVCHI2('SET', P, 'FIELD1', VALUE1, 'FIELD2', VALUE2, ...)
%       Set the values of fields FIELD1... to the values VALUE1... in LIKELIH. 
%       Fields that can be set are 's2' and 'nu'.
%
%	See also
%       PRIOR_GAMMA, GPCF_SEXP, LIKELIH_PROBIT

    
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
        p.type = 'Sinvchi2';
        
        % set functions
        p.fh_pak = @prior_sinvchi2_pak;
        p.fh_unpak = @prior_sinvchi2_unpak;
        p.fh_e = @prior_sinvchi2_e;
        p.fh_g = @prior_sinvchi2_g;
        p.fh_recappend = @prior_sinvchi2_recappend;
        
        % set parameters
        p.s2 = 1;
        p.nu = 4;
        
        % set parameter priors
        p.p.s2 = [];
        p.p.nu = [];
        
        if nargin > 1
            if mod(nargin-1,2) ~=0
                error('Wrong number of arguments')
            end
            % Loop through all the parameter values that are changed
            for i=1:2:length(varargin)-1
                switch varargin{i}
                  case 'scale'
                    p.s2 = varargin{i+1};
                  case 'nu'
                    p.nu = varargin{i+1};
                  case 'scale_prior'
                    p.p.s2 = varargin{i+1};
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
                p.s2 = varargin{i+1};
              case 'nu'
                p.nu = varargin{i+1};
              otherwise
                error('Wrong parameter name!')
            end
        end
    end

   
    
    function w = prior_sinvchi2_pak(p)
        
        w = [];
        if ~isempty(p.p.s2)
            w = log(p.s2);
        end
        if ~isempty(p.p.nu)
            w = [w log(p.nu)];
        end
    end
    
    function [p, w] = prior_sinvchi2_unpak(p, w)
        
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

    function e = prior_sinvchi2_e(x, p)
        e = sum((p.nu./2+1) .* log(x) + (p.s2.*p.nu./2./x) + (p.nu/2) .* log(2./(p.s2.*p.nu)) + gammaln(p.nu/2)) ;
        
        if ~isempty(p.p.s2)
            e = e + feval(p.p.s2.fh_e, p.s2, p.p.s2) - log(p.s2);
        end
        if ~isempty(p.p.nu)
            e = e + feval(p.p.nu.fh_e, p.nu, p.p.nu)  - log(p.nu);
        end
    end
    
    function g = prior_sinvchi2_g(x, p)
        g = (p.nu/2+1)./x-p.nu.*p.s2./(2*x.^2);

        if ~isempty(p.p.s2)
            gs2 = (sum(p.nu/2.*(1./x-1./p.s2)) + feval(p.p.s2.fh_g, p.s2, p.p.s2)).*p.s2 - 1; 
            g = [g gs2];
        end
        if ~isempty(p.p.nu)
            gnu = (sum(0.5*(log(x) + p.s2./x + log(2./p.s2./p.nu) - 1 + digamma1(p.nu/2))) + feval(p.p.nu.fh_g, p.nu, p.p.nu)).*p.nu - 1;
            g = [g gnu];
        end
    end
    
    function rec = prior_sinvchi2_recappend(rec, ri, p)
    % The parameters are not sampled in any case.
        rec = rec;
        if ~isempty(p.p.s2)
        	rec.s2(ri) = p.s2;
        end
        if ~isempty(p.p.nu)
        	rec.nu(ri) = p.nu;
        end
    end
    
end