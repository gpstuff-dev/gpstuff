function gpcf = gpcf_neuralnetwork(do, varargin)
%GPCF_NEURALNETWORK	Create a squared exponential covariance function
%
%	Description
%        GPCF = GPCF_NEURALNETWORK('init', OPTIONS) Create and initialize
%        squared exponential covariance function for Gaussian
%        process. OPTIONS is optional parameter-value pair used as
%        described below by GPCF_NEURALNETWORK('set',...
%
%        GPCF = GPCF_NEURALNETWORK('SET', GPCF, OPTIONS) Set the fields of GPCF
%        as described by the parameter-value pairs ('FIELD', VALUE) in
%        the OPTIONS. The fields that can be modified are:
%
%             'biasSigma2'         : Magnitude (squared) for exponential 
%                                   part. (default 0.1)
%             'weightSigma2'       : Length scale for each input. This 
%                                   can be either scalar corresponding 
%                                   to an isotropic function or vector 
%                                   defining own length-scale for each 
%                                   input direction. (default 10).
%             'biasSigma2_prior'   : prior structure for magnSigma2
%             'weightSigma2_prior' : prior structure for lengthScale
%             'selectedVariables'  : vector defining which inputs are 
%                                    active
%
%       Note! If the prior structure is set to empty matrix
%       (e.g. 'biasSigma2_prior', []) then the parameter in question
%       is considered fixed and it is not handled in optimization,
%       grid integration, MCMC etc.
%
%	See also
%       gpcf_exp, gp_init, gp_e, gp_g, gp_trcov, gp_cov, gp_unpak, gp_pak
    
% Copyright (c) 2007-2009 Jarno Vanhatalo
% Copyright (c) 2009 Jaakko Riihimaki

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.

    ip=inputParser;
    ip.FunctionName = 'GPCF_NEURALNETWORK';
    ip.addRequired('do', @(x) ismember(x, {'init','set'}));
    ip.addOptional('gpcf', [], @isstruct);
    ip.addParamValue('biasSigma2',[], @(x) isscalar(x) && x>0);
    ip.addParamValue('weightSigma2',[], @(x) isvector(x) && all(x>0));
    ip.addParamValue('biasSigma2_prior',[], @(x) isstruct(x) || isempty(x));
    ip.addParamValue('weightSigma2_prior',[], @(x) isstruct(x) || isempty(x));
    ip.addParamValue('selectedVariables',[], @(x) isvector(x) && all(x>0));
    ip.parse(do, varargin{:});
    do=ip.Results.do;
    gpcf=ip.Results.gpcf;
    biasSigma2=ip.Results.biasSigma2;
    weightSigma2=ip.Results.weightSigma2;
    biasSigma2_prior=ip.Results.biasSigma2_prior;
    weightSigma2_prior=ip.Results.weightSigma2_prior;
    selectedVariables=ip.Results.selectedVariables;

    switch do
        case 'init'
            gpcf.type = 'gpcf_neuralnetwork';

            % Initialize parameters
            if isempty(weightSigma2)
                gpcf.weightSigma2 = 10;
            else
                gpcf.weightSigma2=weightSigma2;
            end
            if isempty(biasSigma2)
                gpcf.biasSigma2 = 0.1;
            else
                gpcf.biasSigma2=biasSigma2;
            end
            if ~isempty(selectedVariables)
                gpcf.selectedVariables = selectedVariables;
                if ~sum(strcmp(varargin, 'weightSigma2'))
                    gpcf.weightSigma2= repmat(10, 1, length(gpcf.selectedVariables));
                end
            end

            % Initialize prior structure
            gpcf.p=[];
            if isempty(biasSigma2_prior)
                gpcf.p.biasSigma2=prior_unif('init');
            else
                gpcf.p.biasSigma2=biasSigma2_prior;
            end
            if isempty(weightSigma2_prior)
                gpcf.p.weightSigma2=prior_unif('init');
            else
                gpcf.p.weightSigma2=weightSigma2_prior;
            end

            % Set the function handles to the nested functions
            gpcf.fh_pak = @gpcf_neuralnetwork_pak;
            gpcf.fh_unpak = @gpcf_neuralnetwork_unpak;
            gpcf.fh_e = @gpcf_neuralnetwork_e;
            gpcf.fh_ghyper = @gpcf_neuralnetwork_ghyper;
            gpcf.fh_ginput = @gpcf_neuralnetwork_ginput;
            gpcf.fh_cov = @gpcf_neuralnetwork_cov;
            gpcf.fh_trcov  = @gpcf_neuralnetwork_trcov;
            gpcf.fh_trvar  = @gpcf_neuralnetwork_trvar;
            gpcf.fh_recappend = @gpcf_neuralnetwork_recappend;

        case 'set'
            % Set the parameter values of covariance function
            % go through all the parameter values that are changed
            if isempty(gpcf)
                error('with set option you have to provide the old covariance structure.')
            end
            if ~isempty(weightSigma2);
                gpcf.weightSigma2=weightSigma2;
            end
            if ~isempty(biasSigma2);
                gpcf.biasSigma2=biasSigma2;
            end
            if ~isempty(biasSigma2_prior);
                gpcf.p.biasSigma2=biasSigma2_prior;
            end
            if ~isempty(weightSigma2_prior);
                gpcf.p.weightSigma2=weightSigma2_prior;
            end
            if ~isempty(selectedVariables)
                gpcf.selectedVariables=selectedVariables;
            end
    end


    function w = gpcf_neuralnetwork_pak(gpcf, w)
    %GPCF_NEURALNETWORK_PAK	 Combine GP covariance function hyper-parameters into one vector.
    %
    %	Description
    %   W = GPCF_NEURALNETWORK_PAK(GPCF) takes a covariance function data
    %   structure GPCF and combines the covariance function parameters
    %   and their hyperparameters into a single row vector W and takes
    %   a logarithm of the covariance function parameters.
    %
    %       w = [ log(gpcf.biasSigma2)
    %             (hyperparameters of gpcf.biasSigma2) 
    %             log(gpcf.weightSigma2(:))
    %             (hyperparameters of gpcf.weightSigma2)]'
    %	  
    %
    %	See also
    %	GPCF_NEURALNETWORK_UNPAK

        i1=0;i2=1;
        ww = []; w = [];
        
        if ~isempty(gpcf.p.biasSigma2)
            i1 = i1+1;
            w(i1) = log(gpcf.biasSigma2);
            
            % Hyperparameters of magnSigma2
            ww = feval(gpcf.p.biasSigma2.fh_pak, gpcf.p.biasSigma2);
        end        
        
        if ~isempty(gpcf.p.weightSigma2)
            w = [w log(gpcf.weightSigma2)];
            
            % Hyperparameters of lengthScale
            w = [w feval(gpcf.p.weightSigma2.fh_pak, gpcf.p.weightSigma2)];
        end
        w = [w ww];
    end
    

    function [gpcf, w] = gpcf_neuralnetwork_unpak(gpcf, w)
    %GPCF_NEURALNETWORK_UNPAK  Sets the covariance function parameters pack into the structure
    %
    %	Description
    %   [GPCF, W] = GPCF_NEURALNETWORK_UNPAK(GPCF, W) takes a covariance
    %   function data structure GPCF and a hyper-parameter vector W,
    %   and returns a covariance function data structure identical to
    %   the input, except that the covariance hyper-parameters have
    %   been set to the values in W. Deletes the values set to GPCF
    %   from W and returns the modeified W.
    %
    %   The covariance function parameters are transformed via exp
    %   before setting them into the structure.
    %
    %	See also
    %	GPCF_NEURALNETWORK_PAK
        
        gpp=gpcf.p;
        if ~isempty(gpp.biasSigma2)
            i1=1;
            gpcf.biasSigma2 = exp(w(i1));
            w = w(i1+1:end);
        end

        if ~isempty(gpp.weightSigma2)
            i2=length(gpcf.weightSigma2);
            i1=1;
            gpcf.weightSigma2 = exp(w(i1:i2));
            w = w(i2+1:end);
            
            % Hyperparameters of lengthScale
            [p, w] = feval(gpcf.p.weightSigma2.fh_unpak, gpcf.p.weightSigma2, w);
            gpcf.p.weightSigma2 = p;
        end
        
        if ~isempty(gpp.biasSigma2)
            % Hyperparameters of magnSigma2
            [p, w] = feval(gpcf.p.biasSigma2.fh_unpak, gpcf.p.biasSigma2, w);
            gpcf.p.biasSigma2 = p;
        end
    end

    function eprior =gpcf_neuralnetwork_e(gpcf, x, t)
    %GPCF_NEURALNETWORK_E     Evaluate the energy of prior of NEURALNETWORK parameters
    %
    %	Description
    %   E = GPCF_NEURALNETWORK_E(GPCF, X, T) takes a covariance function data
    %   structure GPCF together with a matrix X of input vectors and a
    %   vector T of target vectors and evaluates log p(th) x J, where
    %   th is a vector of NEURALNETWORK parameters and J is the Jacobian of
    %   transformation exp(w) = th. (Note that the parameters are log
    %   transformed, when packed.) 
    %
    %   Also the log prior of the hyperparameters of the covariance
    %   function parameters is added to E if hyper-hyperprior is
    %   defined.
    %
    %	See also
    %	GPCF_NEURALNETWORK_PAK, GPCF_NEURALNETWORK_UNPAK, GPCF_NEURALNETWORK_G, GP_E

        [n, m] =size(x);

        % Evaluate the prior contribution to the error. The parameters that
        % are sampled are from space W = log(w) where w is all the "real" samples.
        % On the other hand errors are evaluated in the W-space so we need take
        % into account also the  Jakobian of transformation W -> w = exp(W).
        % See Gelman et.all., 2004, Bayesian data Analysis, second edition, p24.
        eprior = 0;
        gpp=gpcf.p;

        if ~isempty(gpp.biasSigma2)
            eprior = feval(gpp.biasSigma2.fh_e, gpcf.biasSigma2, gpp.biasSigma2) - log(gpcf.biasSigma2);
        end
        if ~isempty(gpp.weightSigma2)
            eprior = eprior + feval(gpp.weightSigma2.fh_e, gpcf.weightSigma2, gpp.weightSigma2) - sum(log(gpcf.weightSigma2));
        end

    end

    function [DKff, gprior]  = gpcf_neuralnetwork_ghyper(gpcf, x, x2, mask)  % , t, g, gdata, gprior, varargin
    %GPCF_NEURALNETWORK_GHYPER     Evaluate gradient of covariance function and hyper-prior with 
    %                     respect to the hyperparameters.
    %
    %	Description
    %	[DKff, GPRIOR] = GPCF_NEURALNETWORK_GHYPER(GPCF, X) 
    %   takes a covariance function data structure GPCF, a matrix X of
    %   input vectors and returns DKff, the gradients of covariance
    %   matrix Kff = k(X,X) with respect to th (cell array with matrix
    %   elements), and GPRIOR = d log (p(th))/dth, where th is the
    %   vector of hyperparameters
    %
    %	[DKff, GPRIOR] = GPCF_NEURALNETWORK_GHYPER(GPCF, X, X2) 
    %   takes a covariance function data structure GPCF, a matrix X of
    %   input vectors and returns DKff, the gradients of covariance
    %   matrix Kff = k(X,X2) with respect to th (cell array with matrix
    %   elements), and GPRIOR = d log (p(th))/dth, where th is the
    %   vector of hyperparameters
    %
    %	[DKff, GPRIOR] = GPCF_NEURALNETWORK_GHYPER(GPCF, X, [], MASK) 
    %   takes a covariance function data structure GPCF, a matrix X of
    %   input vectors and returns DKff, the diagonal of gradients of
    %   covariance matrix Kff = k(X,X2) with respect to th (cell array
    %   with matrix elements), and GPRIOR = d log (p(th))/dth, where
    %   th is the vector of hyperparameters. This is needed for
    %   example with FIC sparse approximation.
    %
    %	See also
    %   GPCF_NEURALNETWORK_PAK, GPCF_NEURALNETWORK_UNPAK, GPCF_NEURALNETWORK_E, GP_G
        
        gpp=gpcf.p;
        
        if isfield(gpcf, 'selectedVariables')
           x=x(:,gpcf.selectedVariables); 
            if nargin == 3
                x2=x2(:,gpcf.selectedVariables); 
            end
        end
        
        [n, m] =size(x);
        
        i1=0;
        DKff = {};
        gprior = [];
        
        % Evaluate: DKff{1} = d Kff / d biasSigma2
        %           DKff{2} = d Kff / d weightSigma2
        % NOTE! Here we have already taken into account that the parameters are transformed
        % through log() and thus dK/dlog(p) = p * dK/dp
        
        % evaluate the gradient for training covariance
        if nargin == 2
            
            x_aug=[ones(size(x,1),1) x];
            
            if length(gpcf.weightSigma2) == 1
                % In the case of an isotropic NEURALNETWORK
                s = gpcf.weightSigma2*ones(1,m);
            else
                s = gpcf.weightSigma2;
            end
            
            S_nom=2*x_aug*diag([gpcf.biasSigma2 s])*x_aug';
            
            S_den_tmp=(2*sum(repmat([gpcf.biasSigma2 s], n, 1).*x_aug.^2,2)+1);
            S_den2=S_den_tmp*S_den_tmp';
            S_den=sqrt(S_den2);
            
            C_tmp=2/pi./sqrt(1-(S_nom./S_den).^2);
            % C(abs(C)<=eps) = 0;
            C_tmp = (C_tmp+C_tmp')./2;
            
            bnom_g=2*ones(n);
            bden_g=(0.5./S_den).*(bnom_g.*repmat(S_den_tmp',n,1)+repmat(S_den_tmp,1,n).*bnom_g);
            bg=gpcf.biasSigma2*C_tmp.*(bnom_g.*S_den-bden_g.*S_nom)./S_den2;

            ii1 = 0;
            if ~isempty(gpcf.p.biasSigma2)
                ii1 = ii1+1;
                DKff{ii1}=(bg+bg')/2;
            end
            
            if ~isempty(gpcf.p.weightSigma2)
                if length(gpcf.weightSigma2) == 1
                    wnom_g=2*x*x';
                    tmp_g=sum(2*x.^2,2);
                    wden_g=0.5./S_den.*(tmp_g*S_den_tmp'+S_den_tmp*tmp_g');
                    wg=s(1)*C_tmp.*(wnom_g.*S_den-wden_g.*S_nom)./S_den2;
                    
                    ii1 = ii1+1;
                    DKff{ii1}=(wg+wg')/2;
                else
                    for d1=1:m
                        wnom_g=2*x(:,d1)*x(:,d1)';
                        tmp_g=2*x(:,d1).^2;
                        wden_g=0.5./S_den.*(tmp_g*S_den_tmp'+S_den_tmp*tmp_g');
                        wg=s(d1)*C_tmp.*(wnom_g.*S_den-wden_g.*S_nom)./S_den2;
                        
                        ii1 = ii1+1;
                        DKff{ii1}=(wg+wg')/2;
                    end
                end
            end
            
            % Evaluate the gradient of non-symmetric covariance (e.g. K_fu)
        elseif nargin == 3
            
            if size(x,2) ~= size(x2,2)
                error('gpcf_neuralnetwork -> _ghyper: The number of columns in x and x2 has to be the same. ')
            end
            
            n2 =size(x2,1);
            
            x_aug=[ones(size(x,1),1) x];
            x_aug2=[ones(size(x2,1),1) x2];
            
            if length(gpcf.weightSigma2) == 1
                % In the case of an isotropic NEURALNETWORK
                s = gpcf.weightSigma2*ones(1,m);
            else
                s = gpcf.weightSigma2;
            end
            
            S_nom=2*x_aug*diag([gpcf.biasSigma2 s])*x_aug2';
            
            S_den_tmp1=(2*sum(repmat([gpcf.biasSigma2 s], n, 1).*x_aug.^2,2)+1);
            S_den_tmp2=(2*sum(repmat([gpcf.biasSigma2 s], n2, 1).*x_aug2.^2,2)+1);
            
            S_den2=S_den_tmp1*S_den_tmp2';
            S_den=sqrt(S_den2);
            
            C_tmp=2/pi./sqrt(1-(S_nom./S_den).^2);
            %C(abs(C)<=eps) = 0;
            
            bnom_g=2*ones(n, n2);            
            bden_g=(0.5./S_den).*(bnom_g.*repmat(S_den_tmp2',n,1)+repmat(S_den_tmp1,1,n2).*bnom_g);
            
            ii1 = 0;
            if ~isempty(gpcf.p.biasSigma2)
                ii1 = ii1 + 1;
                DKff{ii1}=gpcf.biasSigma2*C_tmp.*(bnom_g.*S_den-bden_g.*S_nom)./S_den2;
            end
            
            if ~isempty(gpcf.p.weightSigma2)
                if length(gpcf.weightSigma2) == 1
                    wnom_g=2*x*x2';
                    tmp_g1=sum(2*x.^2,2);
                    tmp_g2=sum(2*x2.^2,2);
                    wden_g=0.5./S_den.*(tmp_g1*S_den_tmp2'+S_den_tmp1*tmp_g2');
                    
                    ii1 = ii1 + 1;
                    DKff{ii1}=s(1)*C_tmp.*(wnom_g.*S_den-wden_g.*S_nom)./S_den2;
                else
                    for d1=1:m
                        wnom_g=2*x(:,d1)*x2(:,d1)';
                        tmp_g1=2*x(:,d1).^2;
                        tmp_g2=2*x2(:,d1).^2;
                        wden_g=0.5./S_den.*(tmp_g1*S_den_tmp2'+S_den_tmp1*tmp_g2');
                        
                        ii1 = ii1 + 1;
                        DKff{ii1}=s(d1)*C_tmp.*(wnom_g.*S_den-wden_g.*S_nom)./S_den2;
                    end
                end
            end

            % Evaluate: DKff{1}    = d mask(Kff,I) / d biasSigma2
            %           DKff{2...} = d mask(Kff,I) / d weightSigma2
        elseif nargin == 4
            
            x_aug=[ones(size(x,1),1) x];
            
            if length(gpcf.weightSigma2) == 1
                % In the case of an isotropic NEURALNETWORK
                s = gpcf.weightSigma2*ones(1,m);
            else
                s = gpcf.weightSigma2;
            end
            
            S_nom=2*sum(repmat([gpcf.biasSigma2 s],n,1).*x_aug.^2,2);
            
            S_den=(S_nom+1);
            S_den2=S_den.^2;
            
            C_tmp=2/pi./sqrt(1-(S_nom./S_den).^2);
            %C(abs(C)<=eps) = 0;
            
            bnom_g=2*ones(n,1);
            bden_g=(0.5./S_den).*(2*bnom_g.*S_den);
            
            ii1 = 0;
            if ~isempty(gpcf.p.biasSigma2)
                ii1 = ii1 + 1;
                DKff{ii1}=gpcf.biasSigma2*C_tmp.*(bnom_g.*S_den-bden_g.*S_nom)./S_den2;
            end
            
            if ~isempty(gpcf.p.weightSigma2)
                if length(gpcf.weightSigma2) == 1
                    wnom_g=sum(2*x.^2,2);
                    wden_g=0.5./S_den.*(2*wnom_g.*S_den);
                    
                    ii1 = ii1+1;
                    DKff{ii1}=s(1)*C_tmp.*(wnom_g.*S_den-wden_g.*S_nom)./S_den2;
                else
                    for d1=1:m
                        wnom_g=2*x(:,d1).^2;
                        wden_g=0.5./S_den.*(2*wnom_g.*S_den);
                        
                        ii1 = ii1+1;                        
                        DKff{ii1}=s(d1)*C_tmp.*(wnom_g.*S_den-wden_g.*S_nom)./S_den2;
                    end
                end
            end
        end
        if nargout > 1
            % Evaluate the gprior with respect to biasSigma2
            ggs = [];
            if ~isempty(gpcf.p.biasSigma2)
                % Evaluate the gprior with respect to magnSigma2
                i1 = 1;
                ggs = feval(gpp.biasSigma2.fh_g, gpcf.biasSigma2, gpp.biasSigma2);
                gprior = ggs(i1).*gpcf.biasSigma2 - 1;
            end
            
            if ~isempty(gpcf.p.weightSigma2)
                i1=i1+1; 
                lll = length(gpcf.weightSigma2);
                gg = feval(gpp.weightSigma2.fh_g, gpcf.weightSigma2, gpp.weightSigma2);
                gprior(i1:i1-1+lll) = gg(1:lll).*gpcf.weightSigma2 - 1;
                gprior = [gprior gg(lll+1:end)];
            end
            if length(ggs) > 1
                gprior = [gprior ggs(2:end)];
            end
        end
    end


    function DKff  = gpcf_neuralnetwork_ginput(gpcf, x, x2)
    %GPCF_NEURALNETWORK_GINPUT     Evaluate gradient of covariance function with 
    %                     respect to x.
    %
    %	Description
    %	DKff = GPCF_NEURALNETWORK_GHYPER(GPCF, X) 
    %   takes a covariance function data structure GPCF, a matrix X of
    %   input vectors and returns DKff, the gradients of covariance
    %   matrix Kff = k(X,X) with respect to X (cell array with matrix
    %   elements)
    %
    %	DKff = GPCF_NEURALNETWORK_GHYPER(GPCF, X, X2) 
    %   takes a covariance function data structure GPCF, a matrix X of
    %   input vectors and returns DKff, the gradients of covariance
    %   matrix Kff = k(X,X2) with respect to X (cell array with matrix
    %   elements).
    %
    %	See also
    %   GPCF_NEURALNETWORK_PAK, GPCF_NEURALNETWORK_UNPAK, GPCF_NEURALNETWORK_E, GP_G
        
       if isfield(gpcf, 'selectedVariables')
            x=x(:,gpcf.selectedVariables); 
            if nargin == 3
                x2=x2(:,gpcf.selectedVariables); 
            end
        end
    
        [n, m] =size(x);
        
        if nargin == 2
            
            if length(gpcf.weightSigma2) == 1
                % In the case of an isotropic NEURALNETWORK
                s = gpcf.weightSigma2*ones(1,m);
            else
                s = gpcf.weightSigma2;
            end
            
            x_aug=[ones(size(x,1),1) x];
            
            S_nom=2*x_aug*diag([gpcf.biasSigma2 s])*x_aug';
            S_den_tmp=(2*sum(repmat([gpcf.biasSigma2 s], n, 1).*x_aug.^2,2)+1);
            
            S_den2=S_den_tmp*S_den_tmp';
            S_den=sqrt(S_den2);
            
            C_tmp=2/pi./sqrt(1-(S_nom./S_den).^2);
            %C(abs(C)<=eps) = 0;
            C_tmp = (C_tmp+C_tmp')./2;
            
            ii1=0;
            for d1=1:m
                for j=1:n
                    
                    DK = zeros(n);
                    DK(j,:)=s(d1)*x(:,d1)';
                    DK = DK + DK';
                    inom_g=2*DK;
                    
                    tmp_g=zeros(n);
                    tmp_g(j,:)=2*s(d1)*2*x(j,d1)*S_den_tmp';
                    tmp_g=tmp_g+tmp_g';
                    
                    iden_g=0.5./S_den.*(tmp_g);
                    
                    ii1=ii1+1;
                    DKff{ii1}=C_tmp.*(inom_g.*S_den-iden_g.*S_nom)./S_den2;
                end
            end
            
        elseif nargin == 3
            
            if length(gpcf.weightSigma2) == 1
                % In the case of an isotropic NEURALNETWORK
                s = gpcf.weightSigma2*ones(1,m);
            else
                s = gpcf.weightSigma2;
            end
            
            n2 =size(x2,1);
            
            x_aug=[ones(size(x,1),1) x];
            x_aug2=[ones(size(x2,1),1) x2];
            
            S_nom=2*x_aug*diag([gpcf.biasSigma2 s])*x_aug2';
            
            S_den_tmp1=(2*sum(repmat([gpcf.biasSigma2 s], n, 1).*x_aug.^2,2)+1);
            S_den_tmp2=(2*sum(repmat([gpcf.biasSigma2 s], n2, 1).*x_aug2.^2,2)+1);
            
            S_den2=S_den_tmp1*S_den_tmp2';
            S_den=sqrt(S_den2);
            
            C_tmp=2/pi./sqrt(1-(S_nom./S_den).^2);
            % C(abs(C)<=eps) = 0;
            
            ii1 = 0;
            for d1=1:m
                for j = 1:n
                    
                    DK = zeros(n, n2);
                    DK(j,:)=s(d1)*x2(:,d1)';
                    inom_g=2*DK;
                    
                    tmp_g=zeros(n, n2);
                    tmp_g(j,:)=2*s(d1)*2*x(j,d1)*S_den_tmp2';
                    
                    iden_g=0.5./S_den.*(tmp_g);
                    
                    ii1=ii1+1;
                    DKff{ii1}=C_tmp.*(inom_g.*S_den-iden_g.*S_nom)./S_den2;
                end
            end
        end
    end


    function C = gpcf_neuralnetwork_cov(gpcf, x1, x2, varargin)
    % GP_NEURALNETWORK_COV     Evaluate covariance matrix between two input vectors.
    %
    %         Description         
    %         C = GP_NEURALNETWORK_COV(GP, TX, X) takes in covariance function of a
    %         Gaussian process GP and two matrixes TX and X that
    %         contain input vectors to GP. Returns covariance matrix
    %         C. Every element ij of C contains covariance between
    %         inputs i in TX and j in X.
    %
    %
    %         See also
    %         GPCF_NEURALNETWORK_TRCOV, GPCF_NEURALNETWORK_TRVAR, GP_COV, GP_TRCOV
        
        if isfield(gpcf, 'selectedVariables')
        	x1=x1(:,gpcf.selectedVariables); 
            if nargin == 3
                x2=x2(:,gpcf.selectedVariables); 
            end
        end
    
    
        if isempty(x2)
            x2=x1;
        end
        
        [n1,m1]=size(x1);
        [n2,m2]=size(x2);
        
        if m1~=m2
            error('the number of columns of X1 and X2 has to be same')
        end
        
        x_aug1=[ones(n1,1) x1];
        x_aug2=[ones(n2,1) x2];
        
        if length(gpcf.weightSigma2) == 1
            % In the case of an isotropic NEURALNETWORK
            s = gpcf.weightSigma2*ones(1,m1);
        else
            s = gpcf.weightSigma2;
        end
        
        S_nom=2*x_aug1*diag([gpcf.biasSigma2 s])*x_aug2';
        
        S_den_tmp1=(2*sum(repmat([gpcf.biasSigma2 s], n1, 1).*x_aug1.^2,2)+1);
        S_den_tmp2=(2*sum(repmat([gpcf.biasSigma2 s], n2, 1).*x_aug2.^2,2)+1);
        S_den2=S_den_tmp1*S_den_tmp2';
        
        C=2/pi*asin(S_nom./sqrt(S_den2));
        
        C(abs(C)<=eps) = 0;
    end


    function C = gpcf_neuralnetwork_trcov(gpcf, x)
    % GP_NEURALNETWORK_TRCOV     Evaluate training covariance matrix of inputs.
    %
    %         Description
    %         C = GP_NEURALNETWORK_TRCOV(GP, TX) takes in covariance function of a
    %         Gaussian process GP and matrix TX that contains training
    %         input vectors. Returns covariance matrix C. Every
    %         element ij of C contains covariance between inputs i and
    %         j in TX
    %
    %         See also
    %         GPCF_NEURALNETWORK_COV, GPCF_NEURALNETWORK_TRVAR, GP_COV, GP_TRCOV
        
        if isfield(gpcf, 'selectedVariables')
        	x=x(:,gpcf.selectedVariables); 
        end
    
        [n,m]=size(x);
        x_aug=[ones(n,1) x];
        
        if length(gpcf.weightSigma2) == 1
            % In the case of an isotropic NEURALNETWORK
            s = gpcf.weightSigma2*ones(1,m);
        else
            s = gpcf.weightSigma2;
        end
        
        S_nom=2*x_aug*diag([gpcf.biasSigma2 s])*x_aug';
        
        S_den_tmp=(2*sum(repmat([gpcf.biasSigma2 s], n, 1).*x_aug.^2,2)+1);        
        S_den2=S_den_tmp*S_den_tmp';
        
        C=2/pi*asin(S_nom./sqrt(S_den2));
        
        C(abs(C)<=eps) = 0;
        C = (C+C')./2;
        
    end

    function C = gpcf_neuralnetwork_trvar(gpcf, x)
    % GP_NEURALNETWORK_TRVAR     Evaluate training variance vector
    %
    %         Description
    %         C = GP_NEURALNETWORK_TRVAR(GPCF, TX) takes in covariance function 
    %         of a Gaussian process GPCF and matrix TX that contains
    %         training inputs. Returns variance vector C. Every
    %         element i of C contains variance of input i in TX
    %
    %
    %         See also
    %         GPCF_NEURALNETWORK_COV, GP_COV, GP_TRCOV
        
    	if isfield(gpcf, 'selectedVariables')
        	x=x(:,gpcf.selectedVariables); 
        end
    
        [n,m]=size(x);
        x_aug=[ones(n,1) x];
        
        if length(gpcf.weightSigma2) == 1
            % In the case of an isotropic NEURALNETWORK
            s = gpcf.weightSigma2*ones(1,m);
        else
            s = gpcf.weightSigma2;
        end

        s_tmp=sum(repmat([gpcf.biasSigma2 s], n, 1).*x_aug.^2,2);
        
        C=2/pi*asin(2*s_tmp./(1+2*s_tmp));
        C(C<eps)=0;
        
    end

    function reccf = gpcf_neuralnetwork_recappend(reccf, ri, gpcf)
    % RECAPPEND - Record append
    %
    %          Description
    %          RECCF = GPCF_NEURALNETWORK_RECAPPEND(RECCF, RI, GPCF)
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
            reccf.type = 'gpcf_neuralnetwork';
            reccf.nin = ri;
            reccf.nout = 1;

            % Initialize parameters
            reccf.weightSigma2= [];
            reccf.biasSigma2 = [];

            % Set the function handles
            reccf.fh_pak = @gpcf_neuralnetwork_pak;
            reccf.fh_unpak = @gpcf_neuralnetwork_unpak;
            reccf.fh_e = @gpcf_neuralnetwork_e;
            reccf.fh_g = @gpcf_neuralnetwork_g;
            reccf.fh_cov = @gpcf_neuralnetwork_cov;
            reccf.fh_trcov  = @gpcf_neuralnetwork_trcov;
            reccf.fh_trvar  = @gpcf_neuralnetwork_trvar;
            reccf.fh_recappend = @gpcf_neuralnetwork_recappend;
            reccf.p=[];
            reccf.p.weightSigma2=[];
            reccf.p.biasSigma2=[];
            if ~isempty(ri.p.weightSigma2)
                reccf.p.weightSigma2 = ri.p.weightSigma2;
            end
            if ~isempty(ri.p.biasSigma2)
                reccf.p.biasSigma2 = ri.p.biasSigma2;
            end
            
            return
        end

        gpp = gpcf.p;
        % record weightSigma2
        if ~isempty(gpcf.weightSigma2)
            reccf.weightSigma2(ri,:)=gpcf.weightSigma2;
            reccf.p.weightSigma2 = feval(gpp.weightSigma2.fh_recappend, reccf.p.weightSigma2, ri, gpcf.p.weightSigma2);
        elseif ri==1
            reccf.weightSigma2=[];
        end
        % record biasSigma2
        if ~isempty(gpcf.biasSigma2)
            reccf.biasSigma2(ri,:)=gpcf.biasSigma2;
        elseif ri==1
            reccf.biasSigma2=[];
        end
        
        if isfield(gpcf, 'selectedVariables')
        	reccf.selectedVariables = gpcf.selectedVariables;
        end
        
    end
end