function likelih = likelih_poisson(do, varargin)
%likelih_poisson	Create a Poisson likelihood structure for Gaussian Process
%
%	Description
%
%	LIKELIH = LIKELIH_POISSON('INIT', Y, YE) Create and initialize Poisson likelihood. 
%       The input argument Y contains incedence counts and YE the expected number of
%       incidences
%
%	The fields in LIKELIH are:
%	  type                     = 'likelih_poisson'
%         likelih.avgE             = YE;
%         likelih.gamlny           = gammaln(Y+1);
%         likelih.fh_pak           = function handle to pak
%         likelih.fh_unpak         = function handle to unpak
%         likelih.fh_permute       = function handle to permutation
%         likelih.fh_e             = function handle to energy of likelihood
%         likelih.fh_g             = function handle to gradient of energy
%         likelih.fh_g2            = function handle to second derivatives of energy
%         likelih.fh_g3            = function handle to third (diagonal) gradient of energy 
%         likelih.fh_tiltedMoments = function handle to evaluate tilted moments for EP
%         likelih.fh_mcmc          = function handle to MCMC sampling of latent values
%         likelih.fh_recappend     = function handle to record append
%
%	LIKELIH = LIKELIH_POISSON('SET', LIKELIH, 'FIELD1', VALUE1, 'FIELD2', VALUE2, ...)
%       Set the values of fields FIELD1... to the values VALUE1... in LIKELIH.
%
%	See also
%       LIKELIH_LOGIT, LIKELIH_PROBIT, LIKELIH_NEGBIN
%
%

% Copyright (c) 2006      Helsinki University of Technology (author) Jarno Vanhatalo
% Copyright (c) 2007-2008 Jarno Vanhatalo

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.

    if nargin < 2
        error('Not enough arguments')
    end

    % Initialize the likelihood structure
    if strcmp(do, 'init')
        y = varargin{1};
        avgE = varargin{2};
        likelih.type = 'poisson';
        
        % check the arguments
        if ~isempty(find(y<0))
            error('The incidence counts have to be greater or equal to zero y >= 0.')
        end     
        if ~isempty(find(avgE<=0))
            error('The expected counts have to be greater than zero avgE > 0.')
        end
        
        % Set parameters
        avgE = max(avgE,1e-3);
        likelih.avgE = avgE;
        likelih.gamlny = gammaln(y+1);

        % Initialize prior structure

        % Set the function handles to the nested functions
        likelih.fh_pak = @likelih_poisson_pak;
        likelih.fh_unpak = @likelih_poisson_unpak;
        likelih.fh_permute = @likelih_poisson_permute;
        likelih.fh_e = @likelih_poisson_e;
        likelih.fh_g = @likelih_poisson_g;    
        likelih.fh_g2 = @likelih_poisson_g2;
        likelih.fh_g3 = @likelih_poisson_g3;
        likelih.fh_tiltedMoments = @likelih_poisson_tiltedMoments;
        likelih.fh_mcmc = @likelih_poisson_mcmc;
        likelih.fh_recappend = @likelih_poisson_recappend;

        if length(varargin) > 2
            if mod(nargin,2) ~=1
                error('Wrong number of arguments')
            end
            % Loop through all the parameter values that are changed
            for i=2:2:length(varargin)-1
                switch varargin{i}
                  case 'avgE'
                    likelih.avgE = varargin{i+1};
                    likelih.avgE = max(likelih.avgE,1e-3);
                  case 'gamlny'
                    likelih.gamlny = varargin{i+1};
                  otherwise
                    error('Wrong parameter name!')
                end
            end
        end
    end

    % Set the parameter values of likelihood
    if strcmp(do, 'set')
        if mod(nargin,2) ~=0
            error('Wrong number of arguments')
        end
        gpcf = varargin{1};
        % Loop through all the parameter values that are changed
        for i=2:2:length(varargin)-1
            switch varargin{i}
              case 'avgE'
                likelih.avgE = varargin{i+1};
                likelih.avgE = max(likelih.avgE,1e-3);
              case 'gamlny'
                likelih.gamlny = varargin{i+1};
              otherwise
                error('Wrong parameter name!')
            end
        end
    end



    function w = likelih_poisson_pak(likelih, w)
    %LIKELIH_POISSON_PAK      Combine likelihood parameters into one vector.
    %
    %   NOT IMPLEMENTED!
    %
    %	Description
    %	W = LIKELIH_POISSON_PAK(GPCF, W) takes a likelihood data structure LIKELIH and
    %	combines the parameters into a single row vector W.
    %	  
    %
    %	See also
    %	LIKELIH_POISSON_UNPAK
        w = [];
    end


    function w = likelih_poisson_unpak(likelih, w)
    %LIKELIH_POISSON_UNPAK      Combine likelihood parameters into one vector.
    %
    %   NOT IMPLEMENTED!
    %
    %	Description
    %	W = LIKELIH_POISSON_UNPAK(GPCF, W) takes a likelihood data structure LIKELIH and
    %	combines the parameter vector W and sets the parameters in LIKELIH.
    %	  
    %
    %	See also
    %	LIKELIH_POISSON_PAK
        w = w;
    end



    function likelih = likelih_poisson_permute(likelih, p)
    %LIKELIH_POISSON_PERMUTE    A function to permute the ordering of parameters 
    %                           in likelihood structure
    %   Description
    %	LIKELIH = LIKELIH_POISSON_UNPAK(LIKELIH, P) takes a likelihood data structure
    %   LIKELIH and permutation vector P and returns LIKELIH with its parameters permuted
    %   according to P.
    %
    %   See also 
    %   GPLA_E, GPLA_G, GPEP_E, GPEP_G with CS+FIC model
        
        likelih.avgE = likelih.avgE(p,:);
        likelih.gamlny = likelih.gamlny(p,:);
    end


    function logLikelih = likelih_poisson_e(likelih, y, f)
    %LIKELIH_POISSON_E    (Likelihood) Energy function
    %
    %   Description
    %   E = LIKELIH_POISSON_E(LIKELIH, Y, F) takes a likelihood data structure
    %   LIKELIH, incedence counts Y and latent values F and returns the log likelihood.
    %
    %   See also
    %   LIKELIH_POISSON_G, LIKELIH_POISSON_G3, LIKELIH_POISSON_G2, GPLA_E
        
        lambda = likelih.avgE.*exp(f);
        gamlny = likelih.gamlny;
        logLikelih =  sum(-lambda + y.*log(lambda) - gamlny);
    end


    function deriv = likelih_poisson_g(likelih, y, f, param)
    %LIKELIH_POISSON_G    Gradient of (likelihood) energy function
    %
    %   Description
    %   G = LIKELIH_POISSON_G(LIKELIH, Y, F, PARAM) takes a likelihood data structure
    %   LIKELIH, incedence counts Y and latent values F and returns the gradient of 
    %   log likelihood with respect to PARAM. At the moment PARAM can be only 'latent'.
    %
    %   See also
    %   LIKELIH_POISSON_E, LIKELIH_POISSON_G2, LIKELIH_POISSON_G3, GPLA_E
        
        switch param
          case 'latent'
            deriv = y - likelih.avgE.*exp(f);
        end
    end


    function g2 = likelih_poisson_g2(likelih, y, f, param)
    %LIKELIH_POISSON_G2    Third gradients of (likelihood) energy function
    %
    %   Description
    %   G2 = LIKELIH_POISSON_G2(LIKELIH, Y, F, PARAM) takes a likelihood data 
    %   structure LIKELIH, incedence counts Y and latent values F and returns the 
    %   hessian of log likelihood with respect to PARAM. At the moment PARAM can 
    %   be only 'latent'. G2 is a vector with diagonal elements of the hessian 
    %   matrix (off diagonals are zero).
    %
    %   See also
    %   LIKELIH_POISSON_E, LIKELIH_POISSON_G, LIKELIH_POISSON_G3, GPLA_E

        switch param
          case 'latent'
            g2 = -likelih.avgE.*exp(f);
        end
    end    
    
    function third_grad = likelih_poisson_g3(likelih, y, f, param)
    %LIKELIH_POISSON_G3    Gradient of (likelihood) Energy function
    %
    %   Description
    %   G3 = LIKELIH_POISSON_G3(LIKELIH, Y, F, PARAM) takes a likelihood data 
    %   structure LIKELIH, incedence counts Y and latent values F and returns the 
    %   third gradients of log likelihood with respect to PARAM. At the moment PARAM can 
    %   be only 'latent'. G3 is a vector with third gradients.
    %
    %   See also
    %   LIKELIH_POISSON_E, LIKELIH_POISSON_G, LIKELIH_POISSON_G2, GPLA_E, GPLA_G
    
        switch param
          case 'latent'
            third_grad = - likelih.avgE.*exp(f);
        end
    end


    function [m_0, m_1, m_2] = likelih_poisson_tiltedMoments(likelih, y, i1, sigm2_i, myy_i)
    %LIKELIH_POISSON_TILTEDMOMENTS    Returns the moments of the tilted distribution
    %
    %   Description
    %   [M_0, M_1, M2] = LIKELIH_POISSON_TILTEDMOMENTS(LIKELIH, Y, I, S2, MYY) takes a 
    %   likelihood data structure LIKELIH, incedence counts Y, index I and cavity variance 
    %   S2 and mean MYY. Returns the zeroth moment M_0, firtst moment M_1 and second moment 
    %   M_2 of the tilted distribution
    %
    %   See also
    %   GPEP_E

        zm = @zeroth_moment;

        atol = 1e-10;
        reltol = 1e-6;
        yy = y(i1);
        gamlny = likelih.gamlny(i1);
        avgE = likelih.avgE(i1);
        
        % Set the limits for integration and integrate with quad
        % -----------------------------------------------------
        if yy > 0
            mean_app = (myy_i/sigm2_i + log(yy/avgE.^2)*avgE/yy)/(1/sigm2_i + avgE/yy);
            sigm_app = sqrt((1/sigm2_i + avgE/yy)^-1);
        else
            mean_app = myy_i;
            sigm_app = sqrt(sigm2_i);                    
        end

        lambdaconf(1) = mean_app - 6.*sigm_app; lambdaconf(2) = mean_app + 6.*sigm_app;
        test1 = zm((lambdaconf(2)+lambdaconf(1))/2)>zm(lambdaconf(1));
        test2 = zm((lambdaconf(2)+lambdaconf(1))/2)>zm(lambdaconf(2));
        testiter = 1;
        if test1 == 0 
            lambdaconf(1) = lambdaconf(1) - 3*sigm_app;
            test1 = zm((lambdaconf(2)+lambdaconf(1))/2)>zm(lambdaconf(1));
            if test1 == 0
                go=true;
                while testiter<10 & go
                    lambdaconf(1) = lambdaconf(1) - 2*sigm_app;
                    lambdaconf(2) = lambdaconf(2) - 2*sigm_app;
                    test1 = zm((lambdaconf(2)+lambdaconf(1))/2)>zm(lambdaconf(1));
                    test2 = zm((lambdaconf(2)+lambdaconf(1))/2)>zm(lambdaconf(2));
                    if test1==1&test2==1
                        go=false;
                    end
                    testiter=testiter+1;
                end
            end
            mean_app = (lambdaconf(2)+lambdaconf(1))/2;
        elseif test2 == 0
            lambdaconf(2) = lambdaconf(2) + 3*sigm_app;
            test2 = zm((lambdaconf(2)+lambdaconf(1))/2)>zm(lambdaconf(2));
            if test2 == 0
                go=true;
                while testiter<10 & go
                    lambdaconf(1) = lambdaconf(1) + 2*sigm_app;
                    lambdaconf(2) = lambdaconf(2) + 2*sigm_app;
                    test1 = zm((lambdaconf(2)+lambdaconf(1))/2)>zm(lambdaconf(1));
                    test2 = zm((lambdaconf(2)+lambdaconf(1))/2)>zm(lambdaconf(2));
                    if test1==1&test2==1
                        go=false;
                    end
                    testiter=testiter+1;
                end
            end
            mean_app = (lambdaconf(2)+lambdaconf(1))/2;
        end

        % ------------------------------------------------
% $$$         % Plot the integrands to check that integration limits are ok. Uncomment if you want to use this.
% $$$         fm = @first_moment; sm = @second_moment;
% $$$         function integrand = first_moment(f)
% $$$             lambda = avgE.*exp(f);
% $$$             integrand = exp(-lambda + yy.*log(lambda) - gamlny - 0.5 * (f-myy_i).^2./sigm2_i - log(sigm2_i)/2 - log(2*pi)/2 - log(m_0)); %
% $$$             integrand = f.*integrand; %
% $$$         end
% $$$         function integrand = second_moment(f)
% $$$             lambda = avgE.*exp(f);
% $$$             integrand = exp(log((f-m_1).^2) -lambda + yy.*log(lambda) - gamlny - 0.5 * (f-myy_i).^2./sigm2_i - log(sigm2_i)/2 - log(2*pi)/2 - log(m_0));
% $$$             %integrand = (f-m_1).^2.*integrand; %
% $$$         end
% $$$         clf; ff = [lambdaconf(1):0.01:lambdaconf(2)];
% $$$         subplot(3,1,1); plot([lambdaconf(1) lambdaconf(2)], [0 0], 'r'); hold on; plot(ff, feval(zm, ff))
% $$$         [m_0, fhncnt] = quadgk(zm, lambdaconf(1), lambdaconf(2));
% $$$         subplot(3,1,2); plot([lambdaconf(1) lambdaconf(2)], [0 0], 'r'); hold on; plot(ff, feval(fm, ff));
% $$$         [m_1, fhncnt] = quadgk(fm, lambdaconf(1), lambdaconf(2));
% $$$         subplot(3,1,3); plot([lambdaconf(1) lambdaconf(2)], [0 0], 'r'); hold on; plot(ff, feval(sm, ff));
% $$$         drawnow; S = sprintf('iter %d, y=%d, avgE=%.1f, sigm_a=%.2f, sigm2_i=%.2f', i1, yy, avgE, sigm_app, sigm2_i); title(S)
% $$$         pause
        % ------------------------------------------------
                
        % Fixed integration parameters.
        RTOL = 1.e-6;
        ATOL = 1.e-10;
        MAXINTERVALCOUNT = 650;
        % compute the moments
        [m_0, m_1, m_2] = moments(zm, lambdaconf(1), lambdaconf(2));
        m_1 = m_1/m_0;
        sigm2hati1 = m_2./m_0 - m_1.^2;
        
        % If the second central moment is less than cavity variance integrate more
        % precisely. Theoretically should be sigm2hati1 < sigm2_i
        if sigm2hati1 >= sigm2_i
            ATOL = ATOL.^2;
            RTOL = RTOL.^2;
            [m_0, m_1, m_2] = moments(zm, lambdaconf(1), lambdaconf(2));
            m_1 = m_1/m_0;
            sigm2hati1 = m_2./m_0 - m_1.^2;
            if sigm2hati1 >= sigm2_i
                error('likelih_Poisson:tiltedMoments:  sigm2hati1 >= sigm2_i');
            end
        end
        m_2 = sigm2hati1;
        
        function integrand = zeroth_moment(f)
            lambda = avgE.*exp(f);
            integrand = exp(-lambda + yy.*log(lambda) - gamlny - 0.5 * (f-myy_i).^2./sigm2_i - log(sigm2_i)/2 - log(2*pi)/2); %
        end
        
        function [q, q1, q2, errbnd] = moments(fun,a,b,varargin)
        % Evaluate the tilted moments. The fun input is a function handle to the
        % integrand of zeroth moment.

        %   The implementation is Based on "quadva" by Lawrence F. Shampine.
        %   "Vectorized Adaptive Quadrature in Matlab", Journal of Computational 
        %   and Applied Mathematics.

        % Gauss-Kronrod (7,15) pair. Use symmetry in defining nodes and weights.
            pnodes = [ ...
                0.2077849550078985; 0.4058451513773972; 0.5860872354676911; ...
                0.7415311855993944; 0.8648644233597691; 0.9491079123427585; ...
                0.9914553711208126];
            pwt = [ ...
                0.2044329400752989, 0.1903505780647854, 0.1690047266392679, ...
                0.1406532597155259, 0.1047900103222502, 0.06309209262997855, ...
                0.02293532201052922];
            pwt7 = [0,0.3818300505051189,0,0.2797053914892767,0,0.1294849661688697,0];
            NODES = [-pnodes(end:-1:1); 0; pnodes];
            WT = [pwt(end:-1:1), 0.2094821410847278, pwt];
            EWT = WT - [pwt7(end:-1:1), 0.4179591836734694, pwt7];

            % Integration interval
            tinterval = [a,b];

            % Compute the path length and split tinterval.
            minsubs = 10;   % Minimum number subintervals to start.
            absdx = abs(b-a);
            pathlen = absdx;
            if pathlen > 0
                udelta = minsubs/pathlen;
                nnew = ceil(absdx*udelta) - 1;
                idxnew = find(nnew > 0);
                nnew = nnew(idxnew);
                for j = numel(idxnew):-1:1
                    k = idxnew(j);
                    nnj = nnew(j);
                    % Calculate new points.
                    newpts = tinterval(k) + (1:nnj)./(nnj+1)*(tinterval(k+1)-tinterval(k));
                    % Insert the new points.
                    tinterval = [tinterval(1:k),newpts,tinterval(k+1:end)];
                end
            end
            % Remove useless subintervals.
            tinterval(abs(diff(tinterval))==0) = [];

            % Initialize array of subintervals of [a,b].
            subs = [tinterval(1:end-1);tinterval(2:end)];
            % Initialize partial sums.
            q_ok = 0;
            q1_ok = 0;
            q2_ok = 0;
            err_ok = 0;
            % The main loop
            while true
                % SUBS contains subintervals of [a,b] where the integral is not
                % sufficiently accurate. The first row of SUBS holds the left end
                % points and the second row, the corresponding right endpoints.
                midpt = sum(subs)/2;   % midpoints of the subintervals
                halfh = diff(subs)/2;  % half the lengths of the subintervals            
                x = bsxfun(@plus,NODES*halfh,midpt);
                x = reshape(x,1,[]);   % function f expects a row vector
                fx = fun(x);
                fx1 = fx.*x;
                fx2 = fx.*x.^2;
                
                fx = reshape(fx,numel(WT),[]);
                fx1 = reshape(fx1,numel(WT),[]);
                fx2 = reshape(fx2,numel(WT),[]);
                % Quantities for subintervals.
                qsubs = (WT*fx) .* halfh;
                errsubs = (EWT*fx) .* halfh;
                qsubs1 = (WT*fx1) .* halfh;
                qsubs2 = (WT*fx2) .* halfh;
                % Calculate current values of q and tol.
                q = sum(qsubs) + q_ok;
                q1 = sum(qsubs1) + q1_ok;
                q2 = sum(qsubs2) + q2_ok;
                tol = max(ATOL,RTOL*abs(q));
                % Locate subintervals where the approximate integrals are
                % sufficiently accurate and use them to update the partial
                % error sum.
                ndx = find(abs(errsubs) <= (2*tol/pathlen)*halfh);
                err_ok = err_ok + sum(errsubs(ndx));
                % Remove errsubs entries for subintervals with accurate
                % approximations.
                errsubs(ndx) = [];
                % The approximate error bound is constructed by adding the
                % approximate error bounds for the subintervals with accurate
                % approximations to the 1-norm of the approximate error bounds
                % for the remaining subintervals.  This guards against
                % excessive cancellation of the errors of the remaining
                % subintervals.
                errbnd = abs(err_ok) + norm(errsubs,1);
                % Check for nonfinites.
                if ~(isfinite(q) && isfinite(errbnd))
                    warning('MATLAB:quadgk:NonFiniteValue', ...
                            'Infinite or Not-a-Number value encountered.');
                    break
                end
                % Test for convergence.
                if errbnd <= tol
                    break
                end         
                % Remove subintervals with accurate approximations.
                subs(:,ndx) = [];
                if isempty(subs)
                    break
                end
                % Update the partial sum for the integral.
                q_ok = q_ok + sum(qsubs(ndx));
                q1_ok = q1_ok + sum(qsubs1(ndx));
                q2_ok = q2_ok + sum(qsubs2(ndx));
                % Split the remaining subintervals in half. Quit if splitting
                % results in too many subintervals.
                nsubs = 2*size(subs,2);
                if nsubs > MAXINTERVALCOUNT
                    warning('quadgk2:MaxIntervalCountReached', ...
                            ['Reached the limit on the maximum number of intervals in use.']);
                    break
                end
                midpt(ndx) = []; % Remove unneeded midpoints.
                subs = reshape([subs(1,:); midpt; midpt; subs(2,:)],2,[]);
            end

        end % moments
        
% $$$         % Old version of integration. Below the integration is done separately for each moment. 
% $$$         % The present implementation is 3 times faster
% $$$         [m_0, fhncnt] = quadgk(zm, lambdaconf(1), lambdaconf(2)); %,'AbsTol',atol,'RelTol',reltol
% $$$         [m_1, fhncnt] = quadgk(fm, lambdaconf(1), lambdaconf(2));
% $$$         [sigm2hati1, fhncnt] = quadgk(sm, lambdaconf(1), lambdaconf(2));
% $$$         
% $$$         % If the second central moment is less than cavity variance integrate more
% $$$         % precisely. Theoretically should be sigm2hati1 < sigm2_i
% $$$         if sigm2hati1 >= sigm2_i
% $$$             tol = atol.^2;
% $$$             reltol = reltol.^2;
% $$$             [m_0, fhncnt] = quadgk(zm, lambdaconf(1), lambdaconf(2));
% $$$             [m_1, fhncnt] = quadgk(fm, lambdaconf(1), lambdaconf(2));
% $$$             [sigm2hati1, fhncnt] = quadgk(sm, lambdaconf(1), lambdaconf(2));
% $$$         end
% $$$         m_2 = sigm2hati1;
% $$$         
% $$$         function integrand = zeroth_moment(f)
% $$$             lambda = avgE.*exp(f);
% $$$             integrand = exp(-lambda + yy.*log(lambda) - gamlny - 0.5 * (f-myy_i).^2./sigm2_i - log(sigm2_i)/2 - log(2*pi)/2); %
% $$$         end
% $$$ 
% $$$         function integrand = first_moment(f)
% $$$             lambda = avgE.*exp(f);
% $$$             integrand = exp(-lambda + yy.*log(lambda) - gamlny - 0.5 * (f-myy_i).^2./sigm2_i - log(sigm2_i)/2 - log(2*pi)/2 - log(m_0)); %
% $$$             integrand = f.*integrand; %
% $$$         end
% $$$         function integrand = second_moment(f)
% $$$             lambda = avgE.*exp(f);
% $$$             integrand = exp(log((f-m_1).^2) -lambda + yy.*log(lambda) - gamlny - 0.5 * (f-myy_i).^2./sigm2_i - log(sigm2_i)/2 - log(2*pi)/2 - log(m_0));
% $$$             %integrand = (f-m_1).^2.*integrand; %
% $$$         end

        
    end


    function [z, energ, diagn] = likelih_poisson_mcmc(z, opt, varargin)
    %LIKELIH_POISSON_MCMC        Conducts the MCMC sampling of latent values
    %
    %   Description
    %   [F, ENERG, DIAG] = LIKELIH_POISSON_MCMC(F, OPT, GP, X, Y) takes the current latent 
    %   values F, options structure OPT, Gaussian process data structure GP, inputs X and
    %   incedence counts Y. Samples new latent values and returns also energies ENERG and 
    %   diagnostics DIAG.
    %
    %   See Vanhatalo and Vehtari (2007) for details on implementation.
    %
    %   See also
    %   GP_MC

        if isfield(opt, 'rstate')
            if ~isempty(opt.rstate)
                latent_rstate = opt.latent_opt.rstate;
            end
        else
            latent_rstate = sum(100*clock);
        end

        % Initialize variables 
        gp = varargin{1};
        x = varargin{2}; 
        y = varargin{3}; 
        [n,nin] = size(x);
        switch gp.type
          case 'FULL'
            u = [];
          case 'FIC'
            u = gp.X_u;
            Lav=[];
          case 'CS+FIC'
            u = gp.X_u;
            Labl=[];
            Lp = [];            
          case {'PIC' 'PIC_BLOCK'}
            u = gp.X_u;
            ind = gp.tr_index;
            Labl=[];
            Lp = [];
        end
        n=length(y);

        J = [];
        U = [];
        iJUU = [];
        Linv=[];
        L2=[];
        iLaKfuic=[];
        mincut = -300;
        if isfield(gp.likelih,'avgE');
            E=gp.likelih.avgE(:);
        else
            E=1;
        end     

        % Transform the latent values
        switch gp.type
          case 'FULL'
            getL(z, gp, x, y);             % Evaluate the help matrix for transformation
            w = (L2\z)';                   % Rotate z towards prior
          case 'FIC'
            getL(z, gp, x, y, u);          % Evaluate the help matrix for transformation
            zs = z./Lp;                    % Rotate z towards prior
            w = zs + U*((J*U'-U')*zs);     
          case {'PIC' 'PIC_BLOCK'}
            getL(z, gp, x, y, u);          % Evaluate the help matrix for transformation
            zs=zeros(size(z));             % Rotate z towards prior
            for i=1:length(ind)
                zs(ind{i}) = Lp{i}\z(ind{i});
            end
            w = zs + U*((J*U'-U')*zs);
          case {'CS+FIC'}
            getL(z, gp, x, y, u);          % Evaluate the help matrix for transformation
            zs = Lp\z;                     % Rotate z towards prior
            w = zs + U*((J*U'-U')*zs);
          otherwise 
            error('unknown type of GP\n')
        end
        
        
        %        gradcheck(w, @lvpoisson_er, @lvpoisson_gr, gp, x, y, u, z)
        
        % Conduct the HMC sampling for the transformed latent values
        hmc2('state',latent_rstate)
        rej = 0;
        gradf = @lvpoisson_gr;
        f = @lvpoisson_er;
        for li=1:opt.repeat 
            [w, energ, diagn] = hmc2(f, w, opt, gradf, gp, x, y, u, z);
            w = w(end,:);
            if li<opt.repeat/2
                if diagn.rej
                    opt.stepadj=max(1e-5,opt.stepadj/1.4);
                else
                    opt.stepadj=min(1,opt.stepadj*1.02);
                end
            end
            rej=rej+diagn.rej/opt.repeat;
            if isfield(diagn, 'opt')
                opt=diagn.opt;
            end
        end
        w = w(end,:);
        
        % Rotate w pack to the latent value space
        w=w(:);
        switch gp.type
          case 'FULL'
            z=L2*w;
          case 'FIC'
            z = Lp.*(w + U*(iJUU*w));
          case  {'PIC' 'PIC_BLOCK'}
            w2 = w + U*(iJUU*w);
            for i=1:length(ind)
                z(ind{i}) = Lp{i}*w2(ind{i});
            end
          case  {'CS+FIC'}
            w2 = w + U*(iJUU*w);
            z = Lp*w2;            
        end
        opt.latent_rstate = hmc2('state');
        diagn.opt = opt;
        diagn.rej = rej;
        diagn.lvs = opt.stepadj;

        function [g, gdata, gprior] = lvpoisson_gr(w, gp, x, y, u, varargin)
        %LVPOISSON_G	Evaluate gradient function for transformed GP latent 
        %               values 
        %               
            
        % Force z and E to be a column vector
            w=w(:);
            
            switch gp.type
              case 'FULL'
                z = L2*w;
                z = max(z,mincut);
                gdata = exp(z).*E - y;
                %gdata = ((I+U*J*U'-U*U')*(mu-y)))'; % (  (mu-y) )';
                b=Linv*z;
                gprior=Linv'*b;
                g = (L2'*(gdata + gprior))';
              case 'FIC'
                %        w(w<eps)=0;
                z = Lp.*(w + U*(iJUU*w));
                z = max(z,mincut);
                gdata = exp(z).*E - y;
                gprior = z./Lav - iLaKfuic*(iLaKfuic'*z);
                g = gdata +gprior;
                g = Lp.*g;
                g = g + U*(iJUU*g);
                g = g';
              case {'PIC' 'PIC_BLOCK'}
                w2= w + U*(iJUU*w);
                for i=1:length(ind)
                    z(ind{i}) = Lp{i}*w2(ind{i});
                end
                z = max(z,mincut);
                gdata = exp(z).*E - y;
                gprior = zeros(size(gdata));
                for i=1:length(ind)
                    gprior(ind{i}) = Labl{i}\z(ind{i});
                end
                gprior = gprior - iLaKfuic*(iLaKfuic'*z);
                g = gdata' + gprior';
                for i=1:length(ind)
                    g(ind{i}) = g(ind{i})*Lp{i};
                end
                g = g + g*U*(iJUU);
                %g = g';
              case {'CS+FIC'}
                w2= w + U*(iJUU*w);
                z = Lp*w2;
                z = max(z,mincut);
                gdata = exp(z).*E - y;
                gprior = zeros(size(gdata));
                gprior = ldlsolve(Labl,z);
                gprior = gprior - iLaKfuic*(iLaKfuic'*z);
                g = gdata' + gprior';
                g = g*Lp;
                g = g + g*U*(iJUU);
            end
        end

        function [e, edata, eprior] = lvpoisson_er(w, gp, x, t, u, varargin)
        % LVPOISSON_E     Evaluate energy function for transformed GP latent 
        %                 values 
            
        % force z and E to be a column vector
            w=w(:);
            switch gp.type
              case 'FULL'
                z = L2*w;        
                z = max(z,mincut);
                B=Linv*z;
                eprior=.5*sum(B.^2);
              case 'FIC' 
                z = Lp.*(w + U*(iJUU*w));
                z = max(z,mincut);                
                B = z'*iLaKfuic;  % 1 x u
                eprior = 0.5*sum(z.^2./Lav)-0.5*sum(B.^2);
              case {'PIC' 'PIC_BLOCK'}
                w2= w + U*(iJUU*w);
                for i=1:length(ind)
                    z(ind{i}) = Lp{i}*w2(ind{i});
                end
                z = max(z,mincut);
                B = z'*iLaKfuic;  % 1 x u
                eprior = - 0.5*sum(B.^2);
                for i=1:length(ind)
                    eprior = eprior + 0.5*z(ind{i})'/Labl{i}*z(ind{i});
                end
              case {'CS+FIC'}
                w2= w + U*(iJUU*w);
                z = Lp*w2;
                z = max(z,mincut);
                B = z'*iLaKfuic;  % 1 x u
                eprior = - 0.5*sum(B.^2);
                eprior = eprior + 0.5*z'*ldlsolve(Labl,z);
            end
            mu = exp(z).*E;
            edata = sum(mu-t.*log(mu));
            e=edata + eprior;
        end

        function getL(w, gp, x, t, u)
        % GETL        Evaluate the transformation matrix (or matrices)
            
        % Evaluate the Lambda (La) for specific model
            switch gp.type
              case 'FULL'
                C=gp_trcov(gp, x);
                % Evaluate a approximation for posterior variance
                % Take advantage of the matrix inversion lemma
                %        L=chol(inv(inv(C) + diag(1./gp.likelih.avgE)))';
                Linv = inv(chol(C)');
                L2 = C/chol(diag(1./E) + C);  %sparse(1:n, 1:n, 1./gp.likelih.avgE)
                L2 = chol(C - L2*L2')';                    
              case 'FIC'
                [Kv_ff, Cv_ff] = gp_trvar(gp, x);  % f x 1  vector
                K_fu = gp_cov(gp, x, u);         % f x u
                K_uu = gp_trcov(gp, u);    % u x u, noiseles covariance K_uu
                K_uu = (K_uu+K_uu')/2;     % ensure the symmetry of K_uu
                Luu = chol(K_uu)';
                % Q_ff = K_fu*inv(K_uu)*K_fu'
                % Here we need only the diag(Q_ff), which is evaluated below
                b=Luu\(K_fu');       % u x f
                Qv_ff=sum(b.^2)';
                Lav = Cv_ff-Qv_ff;   % f x 1, Vector of diagonal elements
                                     % Lets scale Lav to ones(f,1) so that Qff+La -> sqrt(La)*Qff*sqrt(La)+I
                                     % and form iLaKfu
                iLaKfu = zeros(size(K_fu));  % f x u,
                for i=1:n
                    iLaKfu(i,:) = K_fu(i,:)./Lav(i);  % f x u 
                end
                c = K_uu+K_fu'*iLaKfu; 
                c = (c+c')./2;         % ensure symmetry
                c = chol(c)';   % u x u, 
                ic = inv(c);
                iLaKfuic = iLaKfu*ic';
                Lp = sqrt(1./(E + 1./Lav));
                b=b';
                for i=1:n
                    b(i,:) = iLaKfuic(i,:).*Lp(i);
                end        
                [V,S2]= eig(b'*b);
                S = sqrt(S2);
                U = b*(V/S);
                U(abs(U)<eps)=0;
                %        J = diag(sqrt(diag(S2) + 0.01^2));
                J = diag(sqrt(1-diag(S2)));   % this could be done without forming the diag matrix 
                                              % J = diag(sqrt(2/(1+diag(S))));
                iJUU = J\U'-U';
                iJUU(abs(iJUU)<eps)=0;
              case {'PIC' 'PIC_BLOCK'}
                [Kv_ff, Cv_ff] = gp_trvar(gp, x);  % f x 1  vector
                K_fu = gp_cov(gp, x, u);         % f x u
                K_uu = gp_trcov(gp, u);    % u x u, noiseles covariance K_uu
                K_uu = (K_uu+K_uu')/2;     % ensure the symmetry of K_uu
                Luu = chol(K_uu)';
                
                % Q_ff = K_fu*inv(K_uu)*K_fu'
                % Here we need only the diag(Q_ff), which is evaluated below
                B=Luu\(K_fu');       % u x f
                iLaKfu = zeros(size(K_fu));  % f x u
                for i=1:length(ind)
                    Qbl_ff = B(:,ind{i})'*B(:,ind{i});
                    [Kbl_ff, Cbl_ff] = gp_trcov(gp, x(ind{i},:));
                    Labl{i} = Cbl_ff - Qbl_ff;
                    iLaKfu(ind{i},:) = Labl{i}\K_fu(ind{i},:);    % Check if works by changing inv(Labl{i})!!!
                end
                % Lets scale Lav to ones(f,1) so that Qff+La -> sqrt(La)*Qff*sqrt(La)+I
                % and form iLaKfu
                A = K_uu+K_fu'*iLaKfu;
                A = (A+A')./2;            % Ensure symmetry
                
                % L = iLaKfu*inv(chol(A));
                iLaKfuic = iLaKfu*inv(chol(A));
                
                for i=1:length(ind)
                    Lp{i} = chol(inv(diag(E(ind{i})) + inv(Labl{i})));
                end
                b=zeros(size(B'));
                
                for i=1:length(ind)
                    b(ind{i},:) = Lp{i}*iLaKfuic(ind{i},:);
                end   
                
                [V,S2]= eig(b'*b);
                S = sqrt(S2);
                U = b*(V/S);
                U(abs(U)<eps)=0;
                %        J = diag(sqrt(diag(S2) + 0.01^2));
                J = diag(sqrt(1-diag(S2)));   % this could be done without forming the diag matrix 
                                              % J = diag(sqrt(2./(1+diag(S))));
                iJUU = J\U'-U';
                iJUU(abs(iJUU)<eps)=0;
              case 'CS+FIC'

                % Evaluate the FIC part of the prior covariance
                cf_orig = gp.cf;
                
                cf1 = {};
                cf2 = {};
                j = 1;
                k = 1;
                for i = 1:length(gp.cf)
                    if ~isfield(gp.cf{i},'cs')
                        cf1{j} = gp.cf{i};
                        j = j + 1;
                    else
                        cf2{k} = gp.cf{i};
                        k = k + 1;
                    end         
                end
                gp.cf = cf1;        
                
                [Kv_ff, Cv_ff] = gp_trvar(gp, x);  % n x 1  vector
                K_fu = gp_cov(gp, x, u);           % n x m
                K_uu = gp_trcov(gp, u);            % m x m, noiseles covariance K_uu
                K_uu = (K_uu+K_uu')/2;             % ensure the symmetry of K_uu
                Luu = chol(K_uu)';                
                B=Luu\(K_fu');                     % m x n
                
                Qv_ff=sum(B.^2)';
                Lav = Cv_ff-Qv_ff;                 % n x 1, Vector of diagonal elements
                
                % Evaluate the CS part of the prior covariance
                gp.cf = cf2;        
                K_cs = gp_trcov(gp,x);
                La = sparse(1:n,1:n,Lav,n,n) + K_cs;
                
                Labl = ldlchol(La);
                
                gp.cf = cf_orig;
                iLaKfu = ldlsolve(Labl,K_fu);

                % scale Lav to ones(f,1) so that Qff+La -> sqrt(La)*Qff*sqrt(La)+I
                A = K_uu+K_fu'*iLaKfu;
                A = (A+A')./2;                     % Ensure symmetry
                
                % L = iLaKfu*inv(chol(A));
                iLaKfuic = iLaKfu/chol(A);
                
% $$$                 Lp = chol(inv(sparse(1:n,1:n,E,n,n) + inv(La)));
                Lp = sparse(1:n,1:n,sqrt(1./(E + 1./diag(La))), n, n);

                b=zeros(size(B'));
                
                b = Lp*iLaKfuic;
                
                [V,S2]= eig(b'*b);
                S = sqrt(S2);
                U = b*(V/S);
                U(abs(U)<eps)=0;
                J = diag(sqrt(1-diag(S2)));   % this could be done without forming the diag matrix 
                                
                iJUU = J\U'-U';
                iJUU(abs(iJUU)<eps)=0;
            end
        end
    end 
    
    function reclikelih = likelih_poisson_recappend(reclikelih, ri, likelih)
    % RECAPPEND - Record append
    %          Description
    %          RECCF = GPCF_SEXP_RECAPPEND(RECCF, RI, GPCF) takes old covariance
    %          function record RECCF, record index RI, RECAPPEND returns a
    %          structure RECCF containing following record fields:
    %          lengthHyper    =
    %          lengthHyperNu  =
    %          lengthScale    =
    %          magnSigma2     =


    end
end


