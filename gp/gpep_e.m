function [e, edata, eprior, param] = gpep_e(w, gp, varargin)
%GPEP_E  Do Expectation propagation and return marginal log posterior estimate
%
%  Description
%    E = GPEP_E(W, GP, X, Y, OPTIONS) takes a GP structure GP
%    together with a matrix X of input vectors and a matrix Y of
%    target vectors, and finds the EP approximation for the
%    conditional posterior p(Y | X, th), where th is the
%    parameters. Returns the energy at th (see below). Each row of
%    X corresponds to one input vector and each row of Y
%    corresponds to one target vector.
%
%    [E, EDATA, EPRIOR] = GPEP_E(W, GP, X, Y, OPTIONS) returns also
%    the data and prior components of the total energy.
%
%    The energy is minus log posterior cost function for th:
%      E = EDATA + EPRIOR
%        = - log p(Y|X, th) - log p(th),
%      where th represents the parameters (lengthScale,
%      magnSigma2...), X is inputs and Y is observations.
%
%    OPTIONS is optional parameter-value pair
%      z - optional observed quantity in triplet (x_i,y_i,z_i)
%          Some likelihoods may use this. For example, in case of
%          Poisson likelihood we have z_i=E_i, that is, expected
%          value for ith case.
%
%  References
%
%    Rasmussen, C. E. and Williams, C. K. I. (2006). Gaussian
%    Processes for Machine Learning. The MIT Press.
%
%    van Gerven, M., Cseke, B., Oostenveld, R., and Heskes, T. (2009).
%    Bayesian source localization with the multivariate Laplace prior.
%    In Advances in Neural Information Processing Systems 22, ed.\
%    Y. Bengio, D. Schuurmans, J. Lafferty, C. K. I. Williams, and
%    A. Culotta, 1901--1909.
%
%    Pasi Jylänki, Jarno Vanhatalo and Aki Vehtari (2011). Robust
%    Gaussian process regression with a Student-t likelihood. Journal
%    of Machine Learning Research, 12(Nov):3227-3257.
%
%  See also
%    GP_SET, GP_E, GPEP_G, GPEP_PRED

%  Description 2
%    Additional properties meant only for internal use.
%
%    GP = GPEP_E('init', GP) takes a GP structure GP and
%    initializes required fields for the EP algorithm.
%
%    GPEP_E('clearcache', GP) takes a GP structure GP and cleares
%    the internal cache stored in the nested function workspace
%
%    [e, edata, eprior, site_tau, site_nu, L, La2, b, muvec_i, sigm2vec_i]
%      = GPEP_E(w, gp, x, y, options)
%    returns many useful quantities produced by EP algorithm.
%
% Copyright (c) 2007  Jaakko Riihimäki
% Copyright (c) 2007-2010  Jarno Vanhatalo
% Copyright (c) 2010 Heikki Peura
% Copyright (c) 2010-2012 Aki Vehtari
% Copyright (c) 2011 Pasi Jylänki

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

% parse inputs
ip=inputParser;
ip.FunctionName = 'GPEP_E';
ip.addRequired('w', @(x) ...
    isempty(x) || ...
    (ischar(x) && ismember(x, {'init' 'clearcache'})) || ...
    (isvector(x) && isreal(x) && all(isfinite(x))) || ...
    all(isnan(x)));
ip.addRequired('gp',@isstruct);
ip.addOptional('x', [], @(x) isnumeric(x) && isreal(x) && all(isfinite(x(:))))
ip.addOptional('y', [], @(x) isnumeric(x) && isreal(x) && all(isfinite(x(:))))
ip.addParamValue('z', [], @(x) isnumeric(x) && isreal(x) && all(isfinite(x(:))))
ip.parse(w, gp, varargin{:});
x=ip.Results.x;
y=ip.Results.y;
z=ip.Results.z;

if strcmp(w, 'init')
    % intialize cache
    ch = [];
    
    % return function handle to the nested function ep_algorithm
    % this way each gp has its own peristent memory for EP
    gp.fh.ne = @ep_algorithm;
    % set other function handles
    gp.fh.e=@gpep_e;
    gp.fh.g=@gpep_g;
    gp.fh.pred=@gpep_pred;
    gp.fh.jpred=@gpep_jpred;
    gp.fh.looe=@gpep_looe;
    gp.fh.loog=@gpep_loog;
    gp.fh.loopred=@gpep_loopred;
    e = gp;
    % remove clutter from the nested workspace
    clear w gp varargin ip x y z
elseif strcmp(w, 'clearcache')
    % clear the cache
    gp.fh.ne('clearcache');
else
    % call ep_algorithm using the function handle to the nested function
    % this way each gp has its own peristent memory for EP
    [e, edata, eprior, param] = gp.fh.ne(w, gp, x, y, z);
end

    function [e, edata, eprior, param] = ep_algorithm(w, gp, x, y, z)
        
        if strcmp(w, 'clearcache')
            ch=[];
            return
        end
        if isempty(z)
            datahash=hash_sha512([x y]);
        else
            datahash=hash_sha512([x y z]);
        end
        
        if ~isempty(ch) && all(size(w)==size(ch.w)) && all(abs(w-ch.w)<1e-8) && isequal(datahash,ch.datahash)
            % The covariance function parameters or data haven't changed
            % so we can return the energy and the site parameters that are saved
            e = ch.e;
            edata = ch.edata;
            eprior = ch.eprior;
            param.tautilde = ch.tautilde;
            param.nutilde = ch.nutilde;
            param.L = ch.L;
            param.La2 = ch.La2;
            param.b = ch.b;
            param.muvec_i = ch.muvec_i;
            param.sigm2vec_i = ch.sigm2vec_i;
            param.logZ_i = ch.logZ_i;
            param.eta = ch.eta;
        else
            
            switch gp.latent_opt.optim_method
                case 'basic-EP'
                    
                    % The parameters or data have changed since
                    % the last call for gpep_e. In this case we need to
                    % re-evaluate the EP approximation
                    gp=gp_unpak(gp, w);
                    ncf = length(gp.cf);
                    n = size(x,1);
                    
                    % EP iteration parameters
                    iter=1;
                    maxiter = gp.latent_opt.maxiter;
                    tol = gp.latent_opt.tol;
                    df = gp.latent_opt.df;
                    
                    nutilde = zeros(size(y));
                    tautilde = zeros(size(y));
                    muvec_i=zeros(size(y));
                    sigm2vec_i=zeros(size(y));
                    logZep_old=0; logZep=Inf;
                    if ~isfield(gp,'meanf')
                        mf = zeros(size(y));
                    else
                        [H,b_m,B_m]=mean_prep(gp,x,[]);
                        mf = H'*b_m;
                    end
                    
                    logM0 = zeros(n,1);
                    muhat = zeros(n,1);
                    sigm2hat = zeros(n,1);
                    
                    % =================================================
                    % First Evaluate the data contribution to the error
                    switch gp.type
                        % ============================================================
                        % FULL
                        % ============================================================
                        case 'FULL'   % A full GP
                            
                            [K,C] = gp_trcov(gp, x);
                            
                            % The EP algorithm for full support covariance function
                            if ~isfield(gp,'meanf')
                                Sigm = C;
                                meanfp=false;
                            else
                                Sigm = C + H'*B_m*H;
                                meanfp=true;
                            end
                            
                            % The EP -algorithm
                            convergence=false;
                            while iter<=maxiter && ~convergence
                                logZep_old=logZep;
                                logM0_old=logM0;
                                
                                % Choose how the site variables are
                                % initialized. If 'on', use the solution
                                % from previous round
                                if isequal(gp.latent_opt.init_prev, 'on') && iter==1 && ~isempty(ch) && all(size(w)==size(ch.w)) && all(abs(w-ch.w)<1) && isequal(datahash,ch.datahash)
                                    tautilde=ch.tautilde;
                                    nutilde=ch.nutilde;
                                else
                                    if isequal(gp.latent_opt.parallel,'on')
                                        % parallel-EP
                                        % compute marginal and cavity parameters
                                        dSigm=diag(Sigm);
                                        tau=1./dSigm-tautilde;
                                        nu = 1./dSigm.*mf-nutilde;
                                        muvec_i=nu./tau;
                                        sigm2vec_i=1./tau;
                                        
                                        % compute moments of tilted distributions
                                        
                                        [logM0, muhat, sigm2hat] = gp.lik.fh.tiltedMoments(gp.lik, y, 1:n, sigm2vec_i, muvec_i, z);
                                        if any(isnan(logM0))
                                            [e, edata, eprior, param, ch] = set_output_for_notpositivedefinite();
                                            return
                                        end
                                        % update site parameters
                                        deltatautilde=1./sigm2hat-tau-tautilde;
                                        tautilde=tautilde+df.*deltatautilde;
                                        deltanutilde=1./sigm2hat.*muhat-nu-nutilde;
                                        nutilde=nutilde+df.*deltanutilde;
                                    else
                                        % sequential-EP   =  Rasmussen and
                                        % Williams (2006) style
                                        for i1=1:n
                                            % Algorithm as in Rasmussen and Williams 2006
                                            % approximate cavity parameters
                                            Sigmi=Sigm(:,i1);
                                            Sigmii=Sigmi(i1);
                                            tau_i=1/Sigmii-tautilde(i1);
                                            nu_i = 1/Sigmii*mf(i1)-nutilde(i1);
                                            mu_i=nu_i/tau_i;
                                            sigm2_i=1/tau_i;
                                            
                                            % marginal moments
                                            [logM0(i1), muhat(i1), sigm2hat(i1)] = gp.lik.fh.tiltedMoments(gp.lik, y, i1, sigm2_i, mu_i, z);
                                            if isnan(logM0(i1))
                                                [e, edata, eprior, param, ch] = set_output_for_notpositivedefinite();
                                                return
                                            end
                                            % update site parameters
                                            deltatautilde=sigm2hat(i1)^-1-tau_i-tautilde(i1);
                                            tautilde(i1)=tautilde(i1)+df*deltatautilde;
                                            deltanutilde=sigm2hat(i1)^-1*muhat(i1)-nu_i-nutilde(i1);
                                            nutilde(i1)=nutilde(i1)+df*deltanutilde;
                                            
                                            % Update mean and variance after each site update (standard EP)
                                            ds = deltatautilde/(1+deltatautilde*Sigmii);
                                            Sigm = Sigm - ((ds*Sigmi)*Sigmi');
                                            %Sigm = Sigm - ((ds*Sigm(:,i1))*Sigm(:,i1)');
                                            % The below is how Rasmussen and Williams
                                            % (2006) do the update. The above version is
                                            % more robust.
                                            %ds = deltatautilde^-1+Sigm(i1,i1);
                                            %ds = (Sigm(:,i1)/ds)*Sigm(:,i1)';
                                            %Sigm = Sigm - ds;
                                            %Sigm=Sigm-(deltatautilde^-1+Sigm(i1,i1))^-1*(Sigm(:,i1)*Sigm(:,i1)');
                                            
                                            if ~meanfp
                                                mf=Sigm*nutilde;
                                            else
                                                mf=Sigm*(C\(H'*b_m)+nutilde);
                                            end
                                            
                                            muvec_i(i1)=mu_i;
                                            sigm2vec_i(i1)=sigm2_i;
                                        end
                                    end
                                end
                                
                                % Recompute the approximate posterior parameters
                                % parallel- and sequential-EP
                                
                                Stilde=tautilde;
                                Stildesqr=sqrt(Stilde);
                                
                                if ~meanfp % zero mean function used
                                    % NOTICE! upper triangle matrix! cf. to
                                    % line 13 in the algorithm 3.5, p. 58.
                                    
                                    %B=eye(n)+Stildesqr*C*Stildesqr;
                                    B=bsxfun(@times,bsxfun(@times,Stildesqr,C),Stildesqr');
                                    B(1:size(B,1)+1:end)=B(1:size(B,1)+1:end)+1;
                                    [L,notpositivedefinite] = chol(B,'lower');
                                    if notpositivedefinite
                                        [e, edata, eprior, param, ch] = set_output_for_notpositivedefinite();
                                        return
                                    end
                                    %V=(L\Stildesqr)*C;
                                    V=L\bsxfun(@times,Stildesqr,C);
                                    Sigm=C-V'*V;
                                    mf=Sigm*nutilde;
                                    
                                    % Compute the marginal likelihood
                                    % Direct formula (3.65):
                                    % Sigmtilde=diag(1./tautilde);
                                    % mutilde=inv(Stilde)*nutilde;
                                    %
                                    % logZep=-0.5*log(det(Sigmtilde+K))-0.5*mutilde'*inv(K+Sigmtilde)*mutilde+
                                    %         sum(log(normcdf(y.*muvec_i./sqrt(1+sigm2vec_i))))+
                                    %         0.5*sum(log(sigm2vec_i+1./tautilde))+
                                    %         sum((muvec_i-mutilde).^2./(2*(sigm2vec_i+1./tautilde)))
                                    
                                    
                                    % 4. term & 1. term
                                    term41=0.5*sum(log(1+tautilde.*sigm2vec_i))-sum(log(diag(L)));
                                    
                                    % 5. term (1/2 element) & 2. term
                                    T=1./sigm2vec_i;
                                    Cnutilde = C*nutilde;
                                    L2 = V*nutilde;
                                    term52 = nutilde'*Cnutilde - L2'*L2 - (nutilde'./(T+Stilde)')*nutilde;
                                    term52 = term52.*0.5;
                                    
                                    % 5. term (2/2 element)
                                    term5=0.5*muvec_i'.*(T./(Stilde+T))'*(Stilde.*muvec_i-2*nutilde);
                                    
                                    % 3. term
                                    term3 = sum(logM0);
                                    
                                    logZep = -(term41+term52+term5+term3);
                                    iter=iter+1;
                                    
                                else
                                    % mean function used
                                    % help variables
                                    hBh = H'*B_m*H;
                                    C_t = C + hBh;
                                    CHb  = C\H'*b_m;
                                    S   = diag(Stildesqr.^2);
                                    %B = eye(n)+Stildesqroot*C*Stildesqroot;
                                    B=bsxfun(@times,bsxfun(@times,Stildesqr,C),Stildesqr');
                                    B(1:n+1:end)=B(1:n+1:end)+1;
                                    %B_h = eye(n) + Stildesqroot*C_t*Stildesqroot;
                                    B_h=bsxfun(@times,bsxfun(@times,Stildesqr,C_t),Stildesqr');
                                    B_h(1:n+1:end)=B_h(1:n+1:end)+1;
                                    % L to return, without the hBh term
                                    [L,notpositivedefinite]=chol(B,'lower');
                                    if notpositivedefinite
                                        [e, edata, eprior, param, ch] = set_output_for_notpositivedefinite();
                                        return
                                    end
                                    % L for the calculation with mean term
                                    [L_m,notpositivedefinite]=chol(B_h,'lower');
                                    if notpositivedefinite
                                        [e, edata, eprior, param, ch] = set_output_for_notpositivedefinite();
                                        return
                                    end
                                    
                                    % Recompute the approximate posterior parameters
                                    % parallel- and sequential-EP
                                    
                                    %V=(L_m\Stildesqroot)*C_t;
                                    V=L_m\bsxfun(@times,Stildesqr,C_t);
                                    Sigm=C_t-V'*V;
                                    mf=Sigm*(CHb+nutilde);
                                    
                                    T=1./sigm2vec_i;
                                    Cnutilde = (C_t - S^-1)*(S*H'*b_m-nutilde);
                                    L2 = V*(S*H'*b_m-nutilde);
                                    
                                    Stildesqroot = diag(Stildesqr);
                                    zz   = Stildesqroot*(L'\(L\(Stildesqroot*C)));
                                    % inv(K + S^-1)*S^-1
                                    Ks  = eye(size(zz)) - zz;
                                    
                                    % 5. term (1/2 element)
                                    term5_1  = 0.5.*((nutilde'*S^-1)./(T.^-1+Stilde.^-1)')*(S^-1*nutilde);
                                    % 2. term
                                    term2    = 0.5.*((S*H'*b_m-nutilde)'*Cnutilde - L2'*L2);
                                    % 4. term
                                    term4    = 0.5*sum(log(1+tautilde.*sigm2vec_i));
                                    % 1. term
                                    term1    = -1.*sum(log(diag(L_m)));
                                    % 3. term
                                    term3    = sum(logM0);
                                    % 5. term (2/2 element)
                                    term5    = 0.5*muvec_i'.*(T./(Stilde+T))'*(Stilde.*muvec_i-2*nutilde);
                                    
                                    logZep = -(term4+term1+term5_1+term5+term2+term3);
                                    
                                    iter=iter+1;
                                    
                                end
                                convergence=max(abs(logM0_old-logM0))<tol && abs(logZep_old-logZep)<tol;
                            end
                            
                            
                            
                            La2 = B;
                            
                            edata = logZep;
                            % Set something into La2
                            % La2 = B;
                            b = 0;
                            
                        otherwise
                            error('Unknown type of Gaussian process!')
                    end
                    
                    % ==================================================
                    % Evaluate the prior contribution to the error from
                    % covariance functions and likelihood
                    % ==================================================
                    
                    % Evaluate the prior contribution to the error from covariance
                    % functions
                    eprior = 0;
                    for i=1:ncf
                        gpcf = gp.cf{i};
                        eprior = eprior - gpcf.fh.lp(gpcf);
                    end
                    
                    % Evaluate the prior contribution to the error from likelihood
                    % functions
                    if isfield(gp.lik, 'p')
                        lik = gp.lik;
                        eprior = eprior - lik.fh.lp(lik);
                    end
                    
                    
                    % The last things to do
                    if isfield(gp.latent_opt, 'display') && ismember(gp.latent_opt.display,{'final','iter'})
                        fprintf('GPEP_E: Number of iterations in EP: %d \n', iter-1)
                    end
                    
                    e = edata + eprior;
                    logZ_i = logM0(:);
                    eta = [];
                    
                    global iter_lkm
                    iter_lkm=iter;
                    
                otherwise
                    error('Unknown optim method!');
            end
            
            
            % store values to struct param
            param.L = L;
            param.nutilde = nutilde;
            param.tautilde = tautilde;
            param.La2 = La2;
            param.b = b;
            param.eta = eta;
            param.logZ_i = logZ_i;
            param.sigm2vec_i = sigm2vec_i;
            param.muvec_i = muvec_i;
            if exist('Sigm','var')
                param.Sigma=Sigm;
            else
                param.Sigma=[];
            end
            param.mf=mf;
            
            % store values to the cache
            ch=param;
            ch.w = w;
            ch.e = e;
            ch.edata = edata;
            ch.eprior = eprior;
            ch.datahash = datahash;
            
        end
    end

    function [e, edata, eprior, param, ch] = set_output_for_notpositivedefinite()
        % Instead of stopping to chol error, return NaN
        e = NaN;
        edata = NaN;
        eprior = NaN;
        param.tautilde = NaN;
        param.nutilde = NaN;
        param.L = NaN;
        param.La2 = NaN;
        param.b = NaN;
        param.muvec_i = NaN;
        param.sigm2vec_i = NaN;
        param.logZ_i = NaN;
        param.eta = NaN;
        param.mf3=NaN;
        param.La3=NaN;
        param.mf2=NaN;
        ch=param;
        ch.e = e;
        ch.edata = edata;
        ch.eprior = eprior;
        ch.datahash = NaN;
        ch.w = NaN;
    end

end

