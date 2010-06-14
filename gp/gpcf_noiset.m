function gpcf = gpcf_noiset(do, varargin)
%GPCF_NOISET	Create a scale mixture noise covariance function (~Student-t) 
%               for Gaussian Process.
%
%	Description
%        GPCF = GPCF_NOISET('init', 'nin', NIN, OPTIONS) Create and initialize
%        Student't noise covariance function , with a scale mixture
%        representation, for Gaussian process with input dimension NIN.
%        OPTIONS is optional parameter-value pair used as described below by
%        GPCF_NOISET('set',...
%
%        GPCF = GPCF_NOISET('SET', GPCF, OPTIONS) Set the fields of GPCF
%        as described by the parameter-value pairs ('FIELD', VALUE) in
%        the OPTIONS. The fields that can be modified are:
%
%             'noiseSigmas2'     : set the noiseSigma2
%             'U'                : set the vector U
%             'tau2'             : set tau^2
%             'alpha'            : set alpha
%             'nu'               : set nu
%             'fix_nu'           : set fix_nu to 0 or 1
%             'nu_prior'         : set the prior structure for nu. 
%                  (nu is the only parameter whose prior we can adjust)
%
%       NOTE!
%       The Student-t residual model is greated as in Gelman et. al. (2004) page 304-305:
%    
%          y-E[y] ~ N(0, alpha^2 * U), where U = diag(u_1, u_2, ..., u_n)
%             u_i ~ Inv-Chi^2(nu, tau^2)
%       
%          The degrees of freedom nu are given a 1/nu prior and they are sampled via 
%          slice sampling.
%
%	See also
%	gpcf_noiset, gpcf_noise
%
%

% Copyright (c) 1998,1999 Aki Vehtari
% Copyright (c) 2007-2010 Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

    ip=inputParser;
    ip.FunctionName = 'GPCF_NOISET';
    ip.addRequired('do', @(x) ismember(x, {'init','set'}));
    ip.addOptional('gpcf', [], @isstruct);
    ip.addParamValue('nin',[], @(x) isscalar(x) && x>0 && mod(x,1)==0);
    ip.addParamValue('noiseSigmas2',[], @(x) isvector(x) && all(x>0));
    ip.addParamValue('U',[], @isvector);
    ip.addParamValue('tau2',[], @isscalar);
    ip.addParamValue('alpha',[], @isscalar);
    ip.addParamValue('fix_nu',[], @(x) isscalar(x) && (x==0||x==1));
    ip.addParamValue('nu',[], @isscalar);
    ip.addParamValue('nu_prior',[], @(x) isstruct(x) || isempty(x));
    ip.addParamValue('censored',[], @(x) isstruct);
    ip.parse(do, varargin{:});
    do=ip.Results.do;
    gpcf=ip.Results.gpcf;
    noiseSigmas2=ip.Results.noiseSigmas2;
    U=ip.Results.U;
    tau2=ip.Results.tau2;
    alpha=ip.Results.alpha;
    fix_nu=ip.Results.fix_nu;
    nu=ip.Results.nu;
    nu_prior=ip.Results.nu_prior;
    nin=ip.Results.nin;
    censored=ip.Results.censored;

    switch do
        case 'init'
            gpcf.type = 'gpcf_noiset';            
            
            % Initialize parameters
            gpcf.ndata = nin;
            gpcf.r = zeros(nin,1);
            if isempty(U)
                gpcf.U = ones(nin,1);
            else
                if size(U,1) == gpcf.ndata
                    gpcf.U = U;
                else
                    error('the size of U has to be NINx1')
                end
            end
            if isempty(noiseSigmas2)
                gpcf.noiseSigmass2 = 0.1^2.*ones(nin,1);
            else
                if (size(noiseSigmas2,1) == gpcf.ndata && size(noiseSigmas2,2) == 1)
                        gpcf.noiseSigmas2 = noiseSigmas2;
                    else
                        error('the size of noiseSigmas2 has to be NINx1')
                end
            end
            if isempty(tau2)
                gpcf.tau2 = 0.1;
            else
                gpcf.tau2 = tau2;
            end
            if isempty(alpha)
                gpcf.alpha = 0.5;
            else
                gpcf.alpha = alpha;
            end
            if isempty(nu)
                gpcf.nu = 4;
            else
                gpcf.nu = nu;
            end
            if isempty(fix_nu)
                gpcf.fix_nu = 1;
            else
                gpcf.fix_nu=fix_nu;
            end
            if ~isempty(censored)
                gpcf.censored = censored{1};
                yy = censored{2};
                if gpcf.censored(1) >= gpcf.censored(2)
                    error('gpcf_noiset -> if censored model is used the limits have to be given in increasing order.')
                end
                
                imis1 = [];
                imis2 = [];
                if gpcf.censored(1) > -inf
                    imis1 = find(yy<=gpcf.censored(1));
                end            
                if gpcf.censored(1) < inf
                    imis2 = find(yy>=gpcf.censored(2));
                end                                
                gpcf.cy = yy([imis1 ; imis2])';
                gpcf.imis = [imis1 ; imis2];
            end

            % Initialize prior structure
            gpcf.p=[];
            gpcf.p.noiseSigmas2=[];
            if isempty(nu_prior)
                gpcf.p.nu=prior_logunif('init');
            else
                gpcf.p.nu=nu_prior;
            end

            % Set the function handles to the nested functions
            gpcf.fh_pak = @gpcf_noiset_pak;
            gpcf.fh_unpak = @gpcf_noiset_unpak;
            gpcf.fh_e = @gpcf_noiset_e;
            gpcf.fh_ghyper = @gpcf_noiset_ghyper;
            gpcf.fh_cov = @gpcf_noiset_cov;
            gpcf.fh_trcov  = @gpcf_noiset_trcov;
            gpcf.fh_trvar  = @gpcf_noiset_trvar;
            gpcf.fh_gibbs = @gpcf_noiset_gibbs;
            % gpcf.sampling_opt = 'noiset_opt';
            gpcf.fh_recappend = @gpcf_noiset_recappend;

        case 'set'
            % Set the parameter values of covariance function
            % go through all the parameter values that are changed
            if isempty(gpcf)
                error('with set option you have to provide the old covariance structure.')
            end
            if ~isempty(U)
                if size(U,1) == gpcf.ndata
                    gpcf.U = U;
                else
                    error('the size of U has to be NINx1')
                end
            end
            if ~isempty(noiseSigmas2)
                if (size(noiseSigmas2,1) == gpcf.ndata && size(noiseSigmas2,2) == 1)
                        gpcf.noiseSigmas2 = noiseSigma2;
                    else
                        error('the size of noiseSigmas2 has to be NINx1')
                end
            end
            if ~isempty(tau2)
                gpcf.tau2 = tau2;
            end
            if ~isempty(alpha)
                gpcf.alpha = alpha;
            end
            if ~isempty(nu)
                gpcf.nu = nu;
            end
            if ~isempty(fix_nu)
                gpcf.fix_nu=fix_nu;
            end
            if ~isempty(censored)
                gpcf.censored = censored{1};
                yy = censored{2};
                if gpcf.censored(1) >= gpcf.censored(2)
                    error('gpcf_noiset -> if censored model is used the limits have to be given in increasing order.')
                end
                
                imis1 = [];
                imis2 = [];
                if gpcf.censored(1) > -inf
                    imis1 = find(yy<=gpcf.censored(1));
                end            
                if gpcf.censored(1) < inf
                    imis2 = find(yy>=gpcf.censored(2));
                end                                
                gpcf.cy = yy([imis1 ; imis2])';
                gpcf.imis = [imis1 ; imis2];
            end
            if ~isempty(nu_prior)
                gpcf.p.nu=nu_prior;
            end
    end


    function w = gpcf_noiset_pak(gpcf)
        w = [];
    end


    function [gpcf, w] = gpcf_noiset_unpak(gpcf, w)


    end

    function eprior =gpcf_noiset_e(gpcf, p, t)

        eprior = 0;
    end

    function [DCff, gprior]  = gpcf_noiset_ghyper(gpcf, p, t, g, gdata, gprior, invC, varargin)
        
        DCff = [];
        gprior = [];
    end

    function C = gpcf_noiset_cov(gpcf, x1, x2)

        C = 0;
    end


    function C = gpcf_noiset_trcov(gpcf, x)
    % GP_NOISET_TRCOV     Evaluate training covariance matrix of inputs.
    %
    %         Description
    %         C = GP_NOISET_TRCOV(GP, TX) takes in covariance function of a
    %         Gaussian process GP and matrix TX that contains training
    %         input vectors. Returns covariance matrix C. Every
    %         element ij of C contains covariance between inputs i and
    %         j in TX
    %
    %         See also
    %         GPCF_NOISET_COV, GPCF_NOISET_TRVAR, GP_COV, GP_TRCOV

        [n, m] =size(x);
        n1=n+1;
        
        if n ~= gpcf.ndata
            error(['gpcf_noiset -> _trcov: The training covariance can be evaluated'... 
                  '      only for training data.                                   '])
        end
        
        C = sparse(1:n, 1:n, gpcf.noiseSigmas2, n, n);
    end

    function C = gpcf_noiset_trvar(gpcf, x)
    % GP_NOISET_TRVAR     Evaluate training variance vector
    %
    %         Description
    %         C = GP_NOISET_TRVAR(GPCF, TX) takes in covariance function 
    %         of a Gaussian process GPCF and matrix TX that contains
    %         training inputs. Returns variance vector C. Every
    %         element i of C contains variance of input i in TX
    %
    %
    %         See also
    %         GPCF_NOISET_COV, GP_COV, GP_TRCOV
        
        [n, m] =size(x);
        if n ~= gpcf.ndata
            error(['gpcf_noiset -> _trvar: The training variance can be evaluated'... 
                  '      only for training data.                                 '])
        end
        C = gpcf.noiseSigmas2;
        
    end
    
    function [gpcf, y] = gpcf_noiset_gibbs(gp, gpcf, opt, x, y)
    % GPCF_NOISET_GIBBS     Function for sampling the noiseSigmas2:s
    %
    %         Description
    %         Perform Gibbs sampling for the covariance function parameters.

            
        [n,m] = size(x);
        
        % Draw a sample of the mean of y. Its distribution is
        % f ~ N(K*inv(C)*y, K - K*inv(C)*K')
        switch gp.type
          case 'FULL'
            sampy = gp_rnd(gp, x, y, x);
          case 'FIC'
            sampy = gp_rnd(gp, x, y, x, 'tstind', 1:n);
          case {'PIC' 'PIC_BLOCK'}
            sampy = gp_rnd(gp, x, y, x, 'tstind', gp.tr_index);
        end
        % Calculate the residual
        r = y-sampy;
        
        U = gpcf.U;
        t2 = gpcf.tau2;
        alpha = gpcf.alpha;
        nu = gpcf.nu;
        rss2=alpha.^2.*U;
        
        % Perform the gibbs sampling (Gelman et.al. (2004) page 304-305)
        % Notice that 'sinvchi2rand' is parameterized as in Gelman et. al.
% $$$         U=invgamrand((nu.*t2+(r./alpha).^2)./(nu+1),nu+1);
% $$$         t2=gamrand(hmean(U),n*nu);
% $$$         alpha2=invgamrand(mean(r.^2./U),n);
% $$$         rss2=alpha2.*U;
% $$$         %nu=sls1mm(@invgam_nu_e,nu,soptnu,[],t2,U);
% $$$         nu=sls1mm(@(nu) -sum(sinvchi2_lpdf(U,nu,t2))+log(nu),nu,opt);
        
        U=sinvchi2rand(nu+1, (nu.*t2+(r./alpha).^2)./(nu+1));        
% $$$         U2=invgamrand((nu.*t2+(r./alpha).^2)./(nu+1),nu+1);
        shape = n*nu./2;                               % These are parameters...
        invscale = nu.*sum(1./U)./2;                   % used in Gelman        
        t2=gamrnd(shape, 1./invscale);                 % Notice! The matlab parameterization is different from Gelmans
% $$$         t2=gamrand(hmean(U),n*nu);
        alpha2=sinvchi2rand(n,mean(r.^2./U));
% $$$         alpha2=invgamrand(mean(r.^2./U),n);
        rss2=alpha2.*U;
        % Sample nu
        if gpcf.fix_nu == 0 && ~isempty(gpcf.p.nu)
            pp = gpcf.p.nu;            
            nu=sls1mm( @(nu) (-sum(sinvchi2_lpdf(U,nu,t2))+feval(pp.fh_e, nu, pp)) ,nu,opt ) ;
        end
        gpcf.noiseSigmas2 = rss2;
        gpcf.U = U;
        gpcf.tau2 = t2;
        gpcf.alpha = sqrt(alpha2);
        gpcf.nu = nu;
        gpcf.r = r;
        if isfield(gpcf, 'censored')   
            imis1 = [];
            imis2 = [];
            if gpcf.censored(1) > -inf
                imis1 = find(y<=gpcf.censored(1));
                y(imis1)=normrtrand(sampy(imis1),alpha2*U(imis1),gpcf.censored(1));
            end
            
            if gpcf.censored(1) < inf
                imis2 = find(y>=gpcf.censored(2));
                y(imis2)=normltrand(sampy(imis2),alpha2*U(imis2),gpcf.censored(2));
            end
            gpcf.cy = y([imis1 ; imis2]);
        end
    end

    function reccf = gpcf_noiset_recappend(reccf, ri, gpcf)
        % RECAPPEND - Record append
        %
        %          Description
        %          RECCF = GPCF_NOISET_RECAPPEND(RECCF, RI, GPCF)
        %          takes a likelihood record structure RECCF, record
        %          index RI and likelihood structure GPCF with the
        %          current MCMC samples of the hyperparameters. Returns
        %          RECCF which contains all the old samples and the
        %          current samples from GPCF .
        %
        %          See also
        %          GP_MC and GP_MC -> RECAPPEND
        
        
        % Initialize record
        if nargin == 2
            reccf.type = 'gpcf_noiset';
            gpcf.ndata = [];
            
            % Initialize parameters
            reccf.noiseSigmas2 = [];
            
            % Set the function handles
            reccf.fh_pak = @gpcf_noiset_pak;
            reccf.fh_unpak = @gpcf_noiset_unpak;
            reccf.fh_e = @gpcf_noiset_e;
            reccf.fh_g = @gpcf_noiset_g;
            reccf.fh_cov = @gpcf_noiset_cov;
            reccf.fh_trcov  = @gpcf_noiset_trcov;
            reccf.fh_trvar  = @gpcf_noiset_trvar;
            reccf.fh_gibbs = @gpcf_noiset_gibbs;
            reccf.fh_recappend = @gpcf_noiset_recappend;
            return
        end
        
        reccf.ndata = gpcf.ndata;
        gpp = gpcf.p;
        
        % record noiseSigma
        if ~isempty(gpcf.noiseSigmas2)
            reccf.noiseSigmas2(ri,:)=gpcf.noiseSigmas2;
        elseif ri==1
            reccf.noiseSigmas2=[];
        end
        if ~isempty(gpcf.nu)
            reccf.nu(ri,:)=gpcf.nu;
            reccf.U(ri,:) = gpcf.U;
            reccf.tau2(ri,:) = gpcf.tau2;
            reccf.alpha(ri,:) = gpcf.alpha;
            reccf.r(ri,:) = gpcf.r;
        elseif ri==1
            reccf.noiseSigmas2=[];
        end
        if isfield(gpcf, 'censored')
            reccf.cy(ri,:) = gpcf.cy';
        end
        
    end
end

