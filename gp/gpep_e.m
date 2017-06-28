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
% Copyright (c) 2007-2017  Jarno Vanhatalo
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
    
    if isfield(gp, 'monotonic') && gp.monotonic
        [gp,x,y,z] = gp.fh.setUpDataForMonotonic(gp,x,y,z);
    end
    
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
    
    if isfield(gp.lik,'nondiagW') % non-diagonal W
      
      if ~isempty(ch) && all(size(w)==size(ch.w)) && all(abs(w-ch.w)<1e-8) && isequal(datahash,ch.datahash)
        % The covariance function parameters or data haven't changed
        % so we can return the energy and the site parameters that are saved
        e = ch.e;
        edata = ch.edata;
        eprior = ch.eprior;
        param.tautilde = ch.tautilde;
        param.nutilde = ch.nutilde;
        param.BKnu=ch.BKnu;
        param.B=ch.B;
        param.cholP=ch.cholP;
        param.invPBKnu=ch.invPBKnu;
        
      else
        
        % The parameters or data have changed since
        % the last call for gpep_e. In this case we need to
        % re-evaluate the EP approximation
        if size(x,1) ~= size(y,1)
          error('GPEP_E: x and y must contain equal amount of rows!')
        end
        [n,nout] = size(y);
        
        if isfield(gp, 'comp_cf')  % own covariance for each ouput component
          multicf = true;
          if length(gp.comp_cf) ~= nout
            error('GPEP_E: the number of component vectors in gp.comp_cf must be the same as number of outputs.')
          end
        else
          multicf = false;
        end
        
        gp=gp_unpak(gp, w);
        ncf = length(gp.cf);
        display=gp.latent_opt.display;
        
        % EP iteration parameters
        iter=1;
        maxiter = gp.latent_opt.maxiter;
        tol = gp.latent_opt.tol;
        
        % damping factors for parallel EP
        df_vec=0.5*ones(1,maxiter+1);
        df_vec(1:10)=0.85; df_vec(11:15)=0.8; df_vec(16:20)=0.7; df_vec(21:25)=0.6;
        
        Inout=eye(nout);
        
        % class numbers (instead of binary coding)
        t=(nout+1)-sum(cumsum(y'))';
        
        % =================================================
        % First Evaluate the data contribution to the error
        switch gp.type
          % ============================================================
          % FULL
          % ============================================================
          case 'FULL'
            
            % covariance matrices
            K = zeros(n,n,nout);
            if multicf
              for i1=1:nout
                % different covariance function for latent processes
                K(:,:,i1) = gp_trcov(gp, x, gp.comp_cf{i1});
              end
            else
              Ktmp=gp_trcov(gp, x);
              for i1=1:nout
                % same covariance function for latent processes
                K(:,:,i1) = Ktmp;
              end
            end
            
            % posterior mean
            mf=zeros(n*nout,1);
            
            % posterior covariance
            Sigm = zeros(nout,nout,n);
            % initialize posterior covariance
            for i1=1:nout
              Sigm(i1,i1,:)=diag(K(:,:,i1));
            end
            
            % help matrices to create the low-rank representation of tautilde
            ei=eye(nout);
            E_i=zeros(nout,nout-1,nout);
            for i1=1:nout
              cni=1:nout;
              cni(i1)=[];
              for k1=1:(nout-1)
                E_i(cni(k1),k1,i1)=1;
              end
            end
            
            % initialize low-rank representation for site parameters
            pivec=y(:);
            
            % matrices to store previous site parameters
            alphatildeprev=zeros(n,nout-1);
            betatildeprev=zeros(n,nout-1);
            
            % matrices for posterior update
            Knu=zeros(n*nout,1);
            BKnu=zeros(n,nout);
            BK=zeros(n,n,nout);
            invcholPBK=zeros(n,n,nout);
            
            tautilde = zeros(nout,nout,n);
            sigm2hatvec=zeros(nout,nout,n);
            sigm2vec_i = zeros(nout,nout,n);
            tauvec_i = zeros(nout,nout,n);
            
            % initialize site parameters
            nutilde = zeros(size(y));
            muhatvec=zeros(n*nout,1);
            muvec_i = zeros(n,nout);
            logM0 = zeros(n,1);
            
            convergence=0;
            last_iter=0;
            
            while (iter<=maxiter && ~convergence) || last_iter
              
              for i1=1:n
                
                % ep with non-diagonal site covariances
                
                %- the cavity distribution
                LSigm=chol(Sigm(:,:,i1),'lower');
                iLSigm=LSigm\Inout;
                invSigmi=iLSigm'*iLSigm;
                
                tau_i=invSigmi-tautilde(:,:,i1);
                nu_i=invSigmi*mf(i1:n:end)-nutilde(i1,:)';
                
                Ltau_i=chol(tau_i,'lower');
                iLtau_i=Ltau_i\Inout;
                sigm2_i=iLtau_i'*iLtau_i;
                mu_i=sigm2_i*nu_i;
                
                %- marginal moments
                if isfield(gp.latent_opt,'incremental') && strcmp(gp.latent_opt.incremental,'on')
                  % In the last iteration after convergence also the zero'th
                  % tilted moments are evaluated
                  if last_iter
                    [alphatilde, betatilde, muhati, sigm2hati, LM0hati] = feval(gp.lik.fh.tiltedMoments, gp.lik, y, i1, sigm2_i, mu_i, z, alphatildeprev(i1,:)', betatildeprev(i1,:)');
                  else
                    [alphatilde, betatilde, muhati, sigm2hati] = feval(gp.lik.fh.tiltedMoments, gp.lik, y, i1, sigm2_i, mu_i, z, alphatildeprev(i1,:)', betatildeprev(i1,:)');
                  end
                else
                  % In the last iteration after convergence also the zero'th
                  % tilted moments are evaluated
                  if last_iter
                    [alphatilde, betatilde, muhati, sigm2hati, LM0hati] = feval(gp.lik.fh.tiltedMoments, gp.lik, y, i1, sigm2_i, mu_i, z);
                  else
                    [alphatilde, betatilde, muhati, sigm2hati] = feval(gp.lik.fh.tiltedMoments, gp.lik, y, i1, sigm2_i, mu_i, z);
                  end
                end
                
                muhatvec(i1:n:end)=muhati;
                sigm2hatvec(:,:,i1)=sigm2hati;
                if last_iter
                  logM0(i1)=LM0hati;
                end
                
                % Update the site parameters:
                
                % damping factor
                df=df_vec(iter);
                
                % damped site precision for latents xk'*w
                alphatilde=alphatildeprev(i1,:)'+df*(alphatilde-alphatildeprev(i1,:)');
                
                % create the vector Pi to form tautilde
                ci=t(i1);
                ui=E_i(:,:,ci)*alphatilde;
                pit=ui+ei(:,ci);
                pivec(i1:n:end)=pit;
                
                % compute tautilde, Equation (12) in the paper (Riihimäki et al., 2013)
                tautilde(:,:,i1)=diag(pit)-(pit*pit')./(ones(1,nout)*pit);
                
                % nutilde (with damped nu site parameter of latent xk'*w)
                betatilde=betatildeprev(i1,:)'+df*(betatilde-betatildeprev(i1,:)');
                
                % compute nutilde, Equation (15) in the paper (Riihimäki et al., 2013)
                nutilde(i1,:)=((ones(1,nout-1)*betatilde)./(ones(1,nout)*pit))*pit-E_i(:,:,ci)*betatilde;
                
                % store tautildes and nutildes for latents xk'*w
                alphatildeprev(i1,:)=alphatilde;
                betatildeprev(i1,:)=betatilde;
                
              end
              
              %------------------------
              % posterior update for EP
              
              B=zeros(n,n,nout);
              cholA=zeros(n,n,nout);
              for k1=1:nout
                Dsq=sqrt(pivec((1:n)+n*(k1-1))); % Dsq = diag( D^(1/2) )
                A=(Dsq*Dsq').*K(:,:,k1); % = D^(1/2) K D^(1/2)
                A(1:n+1:end)=A(1:n+1:end)+1;
                cholA(:,:,k1)=chol(A,'lower');
                invcholADsq=cholA(:,:,k1)\diag(Dsq);
                B(:,:,k1)=invcholADsq'*invcholADsq;
              end
              cholP=chol(sum(B,3),'lower'); % = chol( R^T B R )
              
              % update posterior mean
              for k1=1:nout
                Knu((1:n)+n*(k1-1))=K(:,:,k1)*nutilde(:,k1);
                BKnu(:,k1)=B(:,:,k1)*Knu((1:n)+(k1-1)*n);
                BK(:,:,k1)=B(:,:,k1)*K(:,:,k1);
                invcholPBK(:,:,k1)=cholP\BK(:,:,k1);
              end
              invPBKnu=cholP'\(cholP\sum(BKnu,2));
              for k1=1:nout
                mf((1:n)+(k1-1)*n)=Knu((1:n)+(k1-1)*n)-K(:,:,k1)*(BKnu(:,k1)-B(:,:,k1)*invPBKnu);
              end
              
              % update posterior covariance
              for k1=1:nout
                Sigm(k1,k1,:)=diag(K(:,:,k1))-sum(K(:,:,k1).*BK(:,:,k1))'+sum(invcholPBK(:,:,k1).*invcholPBK(:,:,k1))';
                for j1=(k1+1):nout
                  Sigm(k1,j1,:)=sum(invcholPBK(:,:,k1).*invcholPBK(:,:,j1));
                  Sigm(j1,k1,:)=Sigm(k1,j1,:);
                end
              end
              
              if last_iter
                last_iter=0;
              else
                convergence=~(norm(max(abs(Sigm-sigm2hatvec),[],3))>tol || max(abs(mf-muhatvec))>tol);
                if convergence
                  % After convergence, do one last iteration (where also the zero'th
                  % tilted moments are evaluated)
                  last_iter=1;
                end
              end
              
              iter=iter+1;
              
            end
            
            %-------------------------
            % posterior update
            
            B=zeros(n,n,nout);
            for k1=1:nout
              Dsq=sqrt(pivec((1:n)+n*(k1-1)));
              A=(Dsq*Dsq').*K(:,:,k1);
              A(1:n+1:end)=A(1:n+1:end)+1;
              cholA(:,:,k1)=chol(A,'lower');
              invcholADsq=cholA(:,:,k1)\diag(Dsq);
              B(:,:,k1)=invcholADsq'*invcholADsq;
            end
            cholP=chol(sum(B,3),'lower');
            
            % update posterior mean
            for k1=1:nout
              Knu((1:n)+n*(k1-1))=K(:,:,k1)*nutilde(:,k1);
              BKnu(:,k1)=B(:,:,k1)*Knu((1:n)+(k1-1)*n);
              BK(:,:,k1)=B(:,:,k1)*K(:,:,k1);
              invcholPBK(:,:,k1)=cholP\BK(:,:,k1);
            end
            invPBKnu=cholP'\(cholP\sum(BKnu,2));
            
            for k1=1:nout
              mf((1:n)+(k1-1)*n)=Knu((1:n)+(k1-1)*n)-K(:,:,k1)*(BKnu(:,k1)-B(:,:,k1)*(invPBKnu));
            end
            
            % update posterior covariance
            for k1=1:nout
              Sigm(k1,k1,:)=diag(K(:,:,k1))-sum(K(:,:,k1).*BK(:,:,k1))'+sum(invcholPBK(:,:,k1).*invcholPBK(:,:,k1))';
              for j1=(k1+1):nout
                Sigm(k1,j1,:)=sum(invcholPBK(:,:,k1).*invcholPBK(:,:,j1));
                Sigm(j1,k1,:)=Sigm(k1,j1,:);
              end
            end
            
            
            %-----------------------------------------
            % Marginal likelihood approximation for EP
            
            logZcavities=0;
            logZmarginals=0;
            
            for i1=1:n
              %- update the cavity distribution
              LSigm=chol(Sigm(:,:,i1),'lower');
              iLSigm=LSigm\Inout;
              invSigmi=iLSigm'*iLSigm;
              
              tau_i=invSigmi-tautilde(:,:,i1);
              nu_i=invSigmi*mf(i1:n:end)-nutilde(i1,:)';
              
              Ltau_i=chol(tau_i,'lower');
              iLtau_i=Ltau_i\Inout;
              sigm2_i=iLtau_i'*iLtau_i;
              mu_i=sigm2_i*nu_i;
              
              muvec_i(i1,:)=mu_i;
              sigm2vec_i(:,:,i1)=sigm2_i;
              tauvec_i(:,:,i1)=tau_i;
              
              % cavity terms
              logZcavities=logZcavities+0.5*muvec_i(i1,:)*tauvec_i(:,:,i1)*muvec_i(i1,:)' - sum(log(diag(Ltau_i)));
              
              % marginal terms
              iLSigmmf=LSigm\mf(i1:n:end);
              logZmarginals=logZmarginals-0.5*(iLSigmmf'*iLSigmmf)-sum(log(diag(LSigm)));
            end
            
            logdetA=0;
            for i1=1:nout
              logdetA=logdetA+sum(log(diag(cholA(:,:,i1))));
            end
            logdetP=sum(log(diag(cholP)));
            
            logdetRDR=sum(log(sum(reshape(pivec,n,nout),2)))/2;
            
            % marginal likelihood approximation
            logZep = -(0.5*mf'*nutilde(:) - (logdetA+logdetP-logdetRDR)+ ...
              sum(logM0) + ...
              logZcavities + logZmarginals);
            
            if ismember(display, {'on', 'iter'})
              disp(['Number of EP iterations: ' num2str(iter-1) ', Maximum of EP iterations:' num2str(maxiter)])
            end
            edata = logZep;

            % *** Sparse methods not implemented for non-diagonal W ***
            
            % ============================================================
            % FIC
            % ============================================================
          case 'FIC'
            
            % ============================================================
            % PIC
            % ============================================================
          case {'PIC' 'PIC_BLOCK'}
            
            % ============================================================
            % CS+FIC
            % ============================================================
          case 'CS+FIC'
            
            % ============================================================
            % SSGP
            % ============================================================
          case 'SSGP'
            
          otherwise
            error('Unknown type of Gaussian process!')
        end
        
        % ======================================================================
        % Evaluate the prior contribution to the error from covariance functions
        % ======================================================================
        eprior = 0;
        if ~isempty(strfind(gp.infer_params, 'covariance'))
          for i=1:ncf
            gpcf = gp.cf{i};
            eprior = eprior -feval(gpcf.fh.lp, gpcf);
          end
        end
        
        % ============================================================
        % Evaluate the prior contribution to the error from Gaussian likelihood
        % ============================================================
        % Evaluate the prior contribution to the error from likelihood function
        if isfield(gp, 'lik') && isfield(gp.lik, 'p')
          lik = gp.lik;
          eprior = eprior - feval(lik.fh.lp, lik);
        end
        
        e = edata + eprior;
        
        % store values to struct param
        param.tautilde = tautilde;
        param.nutilde = nutilde;
        param.BKnu=BKnu;
        param.B=B;
        param.cholP=cholP;
        param.invPBKnu=invPBKnu;
        
        % store values to the cache
        ch.w = w;
        ch.e = e;
        ch.edata = edata;
        ch.eprior = eprior;
        ch.tautilde = tautilde;
        ch.nutilde = nutilde;
        ch.BKnu=BKnu;
        ch.B=B;
        ch.cholP=cholP;
        ch.invPBKnu=invPBKnu;
        
        ch.datahash=datahash;
      end
      
    else % isfield(gp.lik,'nondiagW') % diagonal W
      
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
        if  isfield(gp.lik, 'int_likparam')
          if ~(isfield(gp.lik,'joint_mean_magnitude') && gp.lik.joint_mean_magnitude)
            if isfield(gp.lik, 'int_likparam') && gp.lik.int_likparam
              param.mf2=ch.mf2;
              param.Sigm2=ch.La2;
            end
            if isfield(gp.lik, 'int_magnitude') && gp.lik.int_magnitude
              param.mf3=ch.mf3;
              param.La3=ch.La3;
            end
          else
            param.mf=ch.mf;
            param.Sigm=ch.Sigm;
            param.C=ch.C;
          end
        end
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
            
            if isfield(gp.lik, 'int_likparam')
            
              if ~isfield(gp.latent_opt, 'ninner1')
                ninner1=20;
              else
                ninner1=gp.latent_opt.ninner1;
              end
              if ~isfield(gp.latent_opt, 'ninner2')
                ninner2=40;
              else
                ninner2=gp.latent_opt.ninner2;
              end
              if ~isfield(gp.latent_opt, 'df2')
                df2o=0.5;
                df2=df2o;
              else
                df2o=gp.latent_opt.df2;
                df2=df2o;
              end
              logZep_old=0; logZep=Inf;
              if isfield(gp.latent_opt,'display')
                display=gp.latent_opt.display;
              else
                display='on';
                gp.latent_opt.display='on';
              end
              logM0 = zeros(n,1);
              if isfield(gp.lik, 'int_magnitude') && gp.lik.int_magnitude ...
                  && isfield(gp.lik, 'int_likparam') && gp.lik.int_likparam
                if ~isfield(gp.lik, 'joint_mean_magnitude') || ~gp.lik.joint_mean_magnitude
                  int_likparam=true;
                  ns=3;
                  int_magnitude=true;
                  gp=gp_unpak(gp, [0 w(2:end)]);
                  joint_mean_magnitude=false;
                else
                  joint_mean_magnitude=true;
                  int_likparam=true;
                  ns=4;
                  int_magnitude=true;
                  gp=gp_unpak(gp, [0 w(2:end)]);
                end
              elseif isfield(gp.lik, 'int_likparam') && gp.lik.int_likparam ...
                  && (~isfield(gp.lik, 'int_magnitude') || ~gp.lik.int_magnitude)
                int_magnitude=false;
                int_likparam=true;
                ns=2;
              elseif isfield(gp.lik, 'int_magnitude') && gp.lik.int_magnitude ...
                  && (~isfield(gp.lik, 'int_likparam') || ~gp.lik.int_likparam)
                int_magnitude=true;
                int_likparam=false;
                ns=2;
                gp=gp_unpak(gp, [0 w(2:end)]);
              else
                int_magnitude=false;
                int_likparam=false;
                ns=1;
              end
              
              muhat = zeros(n,ns);
              sigm2hat = zeros(n,ns);
              nutilde = zeros(n,ns);
              tautilde = zeros(n,ns);
              muvec_i=zeros(n,ns);
              sigm2vec_i=zeros(n,ns);
              if ~isfield(gp,'meanf')
                mf = zeros(n,ns);
              else
                [H,b_m,B_m]=mean_prep(gp,x,[]);
                mf = H'*b_m;
              end
              if isfield(gp.lik, 'param_lim')
                param_lim=gp.lik.param_lim;
              else
                param_lim=[0.01^2 2];
              end
            
            else
            
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
            
            end
            
            % =================================================
            % First Evaluate the data contribution to the error
            switch gp.type
              % ============================================================
              % FULL
              % ============================================================
              case 'FULL'   % A full GP
                
                if isfield(gp.lik, 'int_likparam')
                
%---------------% Skip intendation
%---------------% -->
                
                if (int_likparam && gp.lik.inputparam) || (int_magnitude && gp.lik.inputmagnitude) ...
                    || (isfield(gp.lik, 'int_likparam') && isfield(gp, 'comp_cf'))
                  [K,C] = gp_trcov(gp, x, gp.comp_cf{1});
                else
                  [K,C] = gp_trcov(gp, x);
                end
                if any(isnan(K(:))) || any(K(:)>1e10)
                  [e, edata, eprior, param, ch] = set_output_for_notpositivedefinite();
                  return
                end
                
                if issparse(C)
                  error('Sparse cov not implemented for lik_epgaussian.')
                end
                
                % The EP algorithm for full support covariance function
                if ~isfield(gp,'meanf')
                  Sigm = C;                      
                  meanfp=false;                    
                else
                  Sigm = C + H'*B_m*H;
                  meanfp=true;
                end
                if int_likparam
                  if ~gp.lik.inputparam                      
                    inputparam=0;
                    tauprior=0.5;
                    nuprior=log(0.1);
                    Sigm2=1/tauprior;
                    mf(:,2)=nuprior/tauprior;
                  else
                    inputparam=1;
                    if ~isfield(gp, 'comp_cf') || isempty(gp.comp_cf)
                      error('Define multiple covariance functions for latent processes using gp.comp_cf (see gp_set)');
                    end
                    C2=gp_trcov(gp, x, gp.comp_cf{2});
                    if any(isnan(C2(:))) || any(C2(:)>1e10)
                      [e, edata, eprior, param, ch] = set_output_for_notpositivedefinite();
                      return
                    end
                    Sigm2=C2;
                    %mf(:,2)=zeros(n,1);
                  end
                end
                if int_magnitude
                  if isfield(gp.lik, 'inputmagnitude') && gp.lik.inputmagnitude
                    inputmagnitude=1;
                    if exist('inputparam','var') && inputparam
                      C3=gp_trcov(gp,x,gp.comp_cf{3});
                    else
                      C3=gp_trcov(gp,x,gp.comp_cf{2});
                    end
                    if any(isnan(C3(:))) || any(C3(:)>1e10)
                      [e, edata, eprior, param, ch] = set_output_for_notpositivedefinite();
                      return
                    end
                    Sigm3=C3;
                  else
                    inputmagnitude=0;
                    nuprior_magnitude=0;
                    tauprior_magnitude=0.1;
                    Sigm3=1./tauprior_magnitude;
                  end
                end
                df0=df;
                
                if exist('joint_mean_magnitude', 'var') && joint_mean_magnitude
                  Sigm=blkdiag(Sigm,Sigm2,Sigm3);
                  C=Sigm;
                  [LC,npd]=chol(C);
                  if npd
                    [e, edata, eprior, param, ch] = set_output_for_notpositivedefinite();
                    return
                  end
                  invLC=inv(LC);
                  invC=invLC*invLC';
                  mf=[zeros(n,1) -2.*ones(n,1) zeros(n,1)];
                  % Natural parameter version
                  tautilde(:,[1 2 3])=zeros(n,3);
                  % Covariance version
%                     tautilde(:,[1 2 3])=100.*ones(n,3);
                end
                
                % The EP -algorithm
                convergence=false;
                while iter<=maxiter && ~convergence
                  logZep_old=logZep;
                  logM0_old=logM0;
                  
                  rej=0;
                  if isequal(gp.latent_opt.init_prev, 'on') && iter==1 && ~isempty(ch) && all(size(w)==size(ch.w)) && all(abs(w-ch.w)<1) && isequal(datahash,ch.datahash)
                    tautilde=ch.tautilde;
                    nutilde=ch.nutilde;
                  else
                    if isequal(gp.latent_opt.parallel,'on')
                      % parallel-EP
                      % compute marginal and cavity parameters
                      if ~int_likparam && ~int_magnitude
                        
                      else
                        if ~exist('joint_mean_magnitude', 'var')|| ~joint_mean_magnitude
                          dSigm=diag(Sigm);
                          if int_likparam
                            if ~gp.lik.inputparam
                              dSigm=[dSigm repmat(Sigm2,n,1)];
                            else
                              dSigm=[dSigm diag(Sigm2)];
                            end
                          end
                          if int_magnitude
                            if inputmagnitude
                              dSigm=[dSigm diag(Sigm3)];
                              if exist('joint_mean_magnitude', 'var') && joint_mean_magnitude
                                dSigm=[dSigm Sigm4];
                              end
                            else
                              dSigm=[dSigm repmat(Sigm3,n,1)];
                            end
                          end
                          tau=1./dSigm-tautilde;
                          nu = 1./dSigm.*mf-nutilde;
                          muvec_i=nu./tau;
                          sigm2vec_i=1./tau;
                        else
%                             if iter==1
                          for ii=1:n
                            tt=Sigm([ii ii+n ii+2*n],[ii ii+n ii+2*n]);
                            tauu=[tautilde(ii,1) 0 tautilde(ii,4); ...
                              0 tautilde(ii,2) 0; ...
                              tautilde(ii,4) 0 tautilde(ii,3)];
                            % Natural parameter version
                            tt2=inv(inv(tt)-tauu);
                            nuu=tt\mf(ii,1:3)'-nutilde(ii,1:3)';
                            % Covariance version
%                               tt2=inv(inv(tt)-inv(tauu));
%                               nuu=tt\mf(ii,1:3)'-tauu\nutilde(ii,1:3)';

                            sigm2vec_i(ii,[1 2 3 4])=[tt2(1,1) tt2(2,2) tt2(3,3) tt2(1,3)];
                            muvec_i(ii,[1 2 3])=tt2*nuu;
                          end
%                             else
%                               sigm2vec_i=ss_vec_i;
%                               muvec_i=mu_vec_i;
%                             end
%                             dSigm=diag(Sigm(n+1:2*n,n+1:2*n));
%                             tau=1./dSigm-tautilde(:,2);
%                             nu = 1./dSigm.*mf(:,2)-nutilde(:,2);
%                             muvec_i(:,2)=nu./tau;
%                             sigm2vec_i(:,2)=1./tau;
                        end
                        
                        % compute moments of tilted distributions
                        [logM0, muhat, sigm2hat] = gp.lik.fh.tiltedMoments(gp.lik, y, 1:n, sigm2vec_i, muvec_i, z);
                        if any(isnan(logM0))
                          [e, edata, eprior, param, ch] = set_output_for_notpositivedefinite();
                          return
                        end
                        
                        % update site parameters
                        if ~exist('joint_mean_magnitude', 'var')|| ~joint_mean_magnitude
                          tautilde_old=tautilde;
                          nutilde_old=nutilde;
                          deltatautilde=1./sigm2hat-tau-tautilde;
                          deltanutilde=1./sigm2hat.*muhat-nu-nutilde;
                          
                          
                          dfvec=zeros(size(deltatautilde));
                          dfvec(deltatautilde>0)=df;
                          dfvec(deltatautilde<0)=df2;
                          ij1=0;
                          if isfield(gp, 'test')
                            ij1=0;
                            if iter>5
                              if int_likparam && iter>ninner1 %&& ~int_magnitude
                                dfvec(:,2)=0;
                              end
                              if ~int_likparam && ns==2
                                if iter>ninner2
                                  dfvec(:,2)=0;
                                end
                              elseif ns==3
                                if iter>ninner2
                                  dfvec(:,2:3)=0;
                                end
                              end
                              %                               if iter>10
                              %                                 if iter>15
                              %                                   dfvec(:,2)=0;
                              %                                 else
                              % %                                   dfvec(:,1)=0;
                              %                                 end
                              %                               else
                              %                                 dfvec(:,2)=0;
                              %                               end
                            end
                            
                            %                             deltavec(:,iter)=reshape((deltatautilde(:,2:end)),2*n,1);
                            %                             if iter==10
                            %                               rec_mean=mean(deltavec(:,iter-4:end),2);
                            %                               ind_zeros=[];
                            %                             elseif iter>10
                            %                               rec_mean=rec_mean+0.2.*(deltavec(:,iter)-deltavec(:,iter-5));
                            %                             end
                            %                             if iter>10
                            %                               ind_zeros=unique([ind_zeros; find(abs(rec_mean)<1e-4)]);
                            %                               dfvec(ind_zeros+n)=0;
                            %                             end
                          end
                        else
                          tautilde_old=tautilde;
                          nutilde_old=nutilde;
                          if iter>1
                            deltanutilde_old=deltanutilde;
                            deltatautilde_old=deltatautilde;
                          end
                          dfvec=zeros(size(tautilde));
                          reji=1;
                          for ii=1:n
                            % Update site approximations:
                            % new posterior - cavity
                            s_new=[sigm2hat(ii,1) 0 sigm2hat(ii,4); ...
                              0 sigm2hat(ii,2) 0; ...
                              sigm2hat(ii,4) 0 sigm2hat(ii,3)];
                            s_cav=[sigm2vec_i(ii,1) 0 sigm2vec_i(ii,4); ...
                              0 sigm2vec_i(ii,2) 0; ...                                
                              sigm2vec_i(ii,4) 0 sigm2vec_i(ii,3)];
                            % Natural parameter version
                            s_site_new=inv(s_new)-inv(s_cav);
                            % Covariance version
%                               s_site_new=inv(inv(s_new)-inv(s_cav));

%                               ds=df0.*s_site_new + (1-df0).*[tautilde(ii,1) 0 tautilde(ii,4);0 tautilde(ii,2) 0;tautilde(ii,4) 0 tautilde(ii,3)];
%                               [~,npd]=chol(ds);
%                               while npd
%                                 df0=0.9.*df0;
%                                 ds=df0.*s_site_new + (1-df0).*[tautilde(ii,1) 0 tautilde(ii,4);0 tautilde(ii,2) 0;tautilde(ii,4) 0 tautilde(ii,3)];
%                                 [~,npd]=chol(ds);
%                                 if df0<1e-5
%                                   df0=0;
%                                   rej=rej+1;
%                                   break;
%                                 end
%                               end
%                               dfvec(ii,:)=df0.*ones(size(dfvec(ii,:)));
%                               df0=0.5;
%                               [~,npd]=chol(s_site_new);
%                               if npd
%                                 deltatautilde(ii,:) = zeros(1,4);
%                                 deltanutilde(ii,1:3) = zeros(1,3);
%                                 rej=rej+1;
% %                                 reji=[reji ii];
%                               else
                              deltatautilde(ii,:) = [s_site_new(1,1) s_site_new(2,2) s_site_new(3,3) s_site_new(1,3)] ...
                                - tautilde(ii,:);
                              % Natural parameter version
                              mu_site_new=s_new\muhat(ii,1:3)' - s_cav\muvec_i(ii,1:3)';
                              % Covariance version
%                                 mu_site_new=s_site_new*(s_new\muhat(ii,1:3)' - s_cav\muvec_i(ii,1:3)');

                              deltanutilde(ii,1:3) = mu_site_new - nutilde(ii,1:3)';
%                                 if any(isnan(deltatautilde(ii,:)))
%                                   deltatautilde(ii,:)=zeros(1,4);
%                                   deltanutilde(ii,:)=zeros(1,3);
%                                   rej=rej+1;
%                                 end
%                               end
                          end
%                             deltatautilde(:,2)=1./sigm2hat(:,2)-tau-tautilde(:,2);
%                             deltanutilde(:,2)=1./sigm2hat(:,2).*muhat(:,2)-nu-nutilde(:,2);
                          if iter>150
                            df=df/(1+floor(iter/150));
                            df2=df2/(1+floor(iter/150));
                            tol=tol*10^(-floor(iter/150));
                          end
                          dfvec(deltatautilde>0)=df;
                          dfvec(deltatautilde<0)=df2;
                          deltanutilde=[deltanutilde(:,1:3) zeros(n,1)];
%                             if iter>1
%                               deltanutilde=deltanutilde + 0.1.*deltanutilde_old;
%                               deltatautilde=deltatautilde + 0.1.*deltatautilde_old;
%                             end
%                             if iter<100
%                               dfvec(:,4)=0;
%                             end
%                             while(any(any(tautilde(:,1:3)+dfvec(:,1:3).*deltatautilde(:,1:3) < 0)))
%                               [inds,tmp]=find(tautilde(:,1:3)+dfvec(:,1:3).*deltatautilde(:,1:3) < 0);
%                               rej=length(inds)*3;
%                               dfvec(inds,:)=0.5.*dfvec(inds,:);
%                             end
                            
                          %                             inds=unique([find(abs(deltatautilde)>1e4); ...
                          %                               find(isinf(deltatautilde)); find(isnan(deltatautilde)); ...
                          %                               find(abs(deltanutilde)>1e4); ...
                          %                               find(isinf(deltanutilde)); find(isnan(deltanutilde))]);
                          %                             dfvec(inds)=0;
                          %                             rej=length(inds);
                          %                             deltatautilde(:,4)=zeros(n,1);
                        end
                        tautilde=tautilde+dfvec.*deltatautilde;
                        nutilde=nutilde+dfvec.*deltanutilde;
                        
                      end
                      
                    else
                      % sequential-EP
                      muvec_i = zeros(n,ns); sigm2vec_i = zeros(n,ns);
                      for i1=1:n
                        % Algorithm utilizing Cholesky updates
                        % This is numerically more stable but slower
                        % $$$                             % approximate cavity parameters
                        % $$$                             S11 = sum(Ls(:,i1).^2);
                        % $$$                             S1 = Ls'*Ls(:,i1);
                        % $$$                             tau_i=S11^-1-tautilde(i1);
                        % $$$                             nu_i=S11^-1*mf(i1)-nutilde(i1);
                        % $$$
                        % $$$                             mu_i=nu_i/tau_i;
                        % $$$                             sigm2_i=tau_i^-1;
                        % $$$
                        % $$$                             if sigm2_i < 0
                        % $$$                                 [ii i1]
                        % $$$                             end
                        % $$$
                        % $$$                             % marginal moments
                        % $$$                             [M0(i1), muhat, sigm2hat] = feval(gp.lik.fh.tiltedMoments, gp.lik, y, i1, sigm2_i, mu_i, z);
                        % $$$
                        % $$$                             % update site parameters
                        % $$$                             deltatautilde = sigm2hat^-1-tau_i-tautilde(i1);
                        % $$$                             tautilde(i1) = tautilde(i1)+deltatautilde;
                        % $$$                             nutilde(i1) = sigm2hat^-1*muhat-nu_i;
                        % $$$
                        % $$$                             upfact = 1./(deltatautilde^-1+S11);
                        % $$$                             if upfact > 0
                        % $$$                                 Ls = cholupdate(Ls, S1.*sqrt(upfact), '-');
                        % $$$                             else
                        % $$$                                 Ls = cholupdate(Ls, S1.*sqrt(-upfact));
                        % $$$                             end
                        % $$$                             Sigm = Ls'*Ls;
                        % $$$                             mf=Sigm*nutilde;
                        % $$$
                        % $$$                             muvec_i(i1,1)=mu_i;
                        % $$$                             sigm2vec_i(i1,1)=sigm2_i;
                        
                        % Algorithm as in Rasmussen and Williams 2006
                        % approximate cavity parameters
                        if ~int_likparam && ~int_magnitude
                          
                        else
                          % Integrate over likelihood parameter with EP
                          Sigmi=Sigm(:,i1);
                          if int_likparam
                            if ~inputparam
                              Sigmi=[Sigmi repmat(Sigm2,n,1)];
                            else
                              Sigmi=[Sigmi Sigm2(:,i1)];
                            end
                          end
                          if int_magnitude
                            if ~inputmagnitude
                              Sigmi=[Sigmi repmat(Sigm3,n,1)];
                            else
                              Sigmi=[Sigmi Sigm3(:,i1)];
                            end
                          end
                          
                          Sigmii=Sigmi(i1,:);
                          tau_i=1./Sigmii-tautilde(i1,:);
                          nu_i = 1./Sigmii.*mf(i1,:)-nutilde(i1,:);
                          mu_i=nu_i./tau_i;
                          sigm2_i=1./tau_i;
                          muvec_i(i1,:)=mu_i;
                          sigm2vec_i(i1,:)=sigm2_i;
                          
                          [logM0(i1), muhat(i1,:), sigm2hat(i1,:)] = gp.lik.fh.tiltedMoments(gp.lik, y, i1, sigm2_i, mu_i, z);
                            
                          if isnan(logM0(i1))
                            [e, edata, eprior, param, ch] = set_output_for_notpositivedefinite();
                            return
                          end
                          

                          deltatautilde=sigm2hat(i1,:).^-1-tau_i-tautilde(i1,:);
                          deltanutilde=sigm2hat(i1,:).^-1.*muhat(i1,:)-nu_i-nutilde(i1,:);
                          tautilde_old=tautilde;
                          nutilde_old=nutilde;
                          Sigm2_old=Sigm2;
                          Sigm_old=Sigm;
                          
                          % Update mean and variance after each site update (standard EP)
                          
                          % if deltautilde<0, choose df so that
                          %
                          % dlog|Sigm| = (1+df*deltatautilde(1)*Sigmii(1)) >0;
                          %
                          % Sigm2vec = diag(Sigm2) - (ds*Sigmii.^2);
                          % 
                          % Sigm2vec_i = Sigm2vec - tautildenew >0
                          %
                          % if not decrease df
                          %
                          % else
                          %
                          % Sigm2 = Sigm2 - ((ds*Sigmi(:,2))*Sigmi(:,2)');
                          
                          df2=df2o;
                          if deltatautilde(1)<0
                            dft=df2;
                            dflim1=-1/(deltatautilde(1)*Sigmii(1));
                            if dft > dflim1
                              dft=dflim1;
                            end
                          else
                            dft=df;
                          end                              
                          ds = dft.*deltatautilde(1)/(1+dft.*deltatautilde(1)*Sigmii(1));
                          sigm2vec=diag(Sigm) - (ds.*Sigmi(:,1).^2);
                          deltat=zeros(n,1);
                          deltat(i1)=deltatautilde(1);
                          while any(1./sigm2vec - (tautilde(:,1)+dft.*deltat) < 1./diag(C))
                            dft=0.5.*dft;
                            if dft<0.01
                              dft=0;
                            end
                            ds = dft.*deltatautilde(1)/(1+dft.*deltatautilde(1)*Sigmii(1));
                            sigm2vec=diag(Sigm) - (ds.*Sigmi(:,1).^2);
                            rej = rej+1;
                            if isequal(gp.latent_opt.display, 'iter')
                              fprintf('Bad cavity variances for f, increasing damping\n');
                            end
                          end
                          Sigm = Sigm - ((ds*Sigmi(:,1))*Sigmi(:,1)');
                          tautilde(i1,1)=tautilde(i1,1)+dft.*deltatautilde(1);
                          nutilde(i1,1)=nutilde(i1,1)+dft.*deltanutilde(1);
                          %                               df2=df2o;
                          %                             else
                          %                               ds = df.*deltatautilde(1)/(1+df.*deltatautilde(1)*Sigmii(1));
                          %                               Sigm = Sigm - ((ds*Sigmi(:,1))*Sigmi(:,1)');
                          %                               tautilde(i1,1)=tautilde(i1,1)+df.*deltatautilde(1);
                          %                               nutilde(i1,1)=nutilde(i1,1)+df.*deltanutilde(1);
                          %                             end
                          if int_likparam
                            if deltatautilde(2)<0
                              dft=df2;
                              dflim1=-1/(deltatautilde(2)*Sigmii(2));
                              if dft > dflim1
                                dft=dflim1;
                              end
                            else
                              dft=df;
                            end
                            ds = dft.*deltatautilde(2)/(1+dft.*deltatautilde(2)*Sigmii(2));
                            if ~gp.lik.inputparam
                              sigm2vec = Sigm2 - (ds*Sigmii(2)^2);
                            else
                              sigm2vec = diag(Sigm2) - (ds*Sigmi(:,2).^2);
                            end
                            deltat=zeros(n,1);
                            deltat(i1)=deltatautilde(2);
                            while (inputparam && any(1./sigm2vec - (tautilde_old(:,2)+dft.*deltat) < 1./diag(C2))) ...
                                || (~inputparam && any(1./sigm2vec - (tautilde_old(:,2)+dft.*deltat) < 0))
                              dft=0.5.*dft;
                              if dft<0.01
                                dft=0;
                              end
                              ds = dft.*deltatautilde(2)/(1+dft.*deltatautilde(2)*Sigmii(2));
                              if ~gp.lik.inputparam
                                sigm2vec = Sigm2 - (ds*Sigmii(2)^2);
                              else
                                sigm2vec = diag(Sigm2) - (ds*Sigmi(:,2).^2);
                              end
                              rej = rej+1;
                              if isequal(gp.latent_opt.display, 'iter')
                                fprintf('Bad cavity variances for theta, increasing damping\n');
                              end
                            end
                            if ~inputparam
                              Sigm2 = Sigm2 - (ds*Sigmii(2)^2);
                            else
                              Sigm2 = Sigm2 - (ds*Sigmi(:,2)*Sigmi(:,2)');
                            end
                            tautilde(i1,2)=tautilde(i1,2)+dft.*deltatautilde(2);
                            nutilde(i1,2)=nutilde(i1,2)+dft.*deltanutilde(2);
                          end
                          
                          if int_magnitude
                            % Integrate over magnitude
                            if deltatautilde(ns)<0
                              dft=df2;
                              dflim1=-1/(deltatautilde(ns)*Sigmii(ns));
                              if dft > dflim1
                                dft=dflim1;
                              end
                            else
                              dft=df;
                            end
                            ds = dft.*deltatautilde(ns)/(1+dft.*deltatautilde(ns)*Sigmii(ns));
                            if ~inputmagnitude
                              sigm2vec = Sigm3 - (ds*Sigmii(ns)^2);
                            else
                              sigm2vec = diag(Sigm3) - (ds*Sigmi(:,ns).^2);
                            end
                            deltat=zeros(n,1);
                            deltat(i1)=deltatautilde(ns);
                            while (inputmagnitude && any(1./sigm2vec - (tautilde_old(:,ns)+dft.*deltat) < 0)) ...
                              || (~inputmagnitude && any(1./sigm2vec - (tautilde_old(:,ns)+dft.*deltat) < 0))
                              dft=0.5.*dft;
                              if dft<0.01
                                dft=0;
                              end
                              ds = dft.*deltatautilde(ns)/(1+dft.*deltatautilde(ns)*Sigmii(ns));
                              if ~inputmagnitude
                                sigm2vec = Sigm3 - (ds*Sigmii(ns)^2);
                              else
                                sigm2vec = diag(Sigm3) - (ds*Sigmi(:,ns).^2);
                              end
                              rej = rej+1;
                              if isequal(gp.latent_opt.display, 'iter')
                                fprintf('Bad cavity variances for phi, increasing damping\n');
                              end
                            end
                            if ~inputmagnitude
                              Sigm3 = Sigm3 - (ds*Sigmii(ns)^2);
                            else
                              Sigm3 = Sigm3 - (ds*Sigmi(:,ns)*Sigmi(:,ns)');
                            end
                            tautilde(i1,ns)=tautilde(i1,ns)+dft.*deltatautilde(ns);
                            nutilde(i1,ns)=nutilde(i1,ns)+dft.*deltanutilde(ns);
                          end
%                             Sigm*(taut\([nutilde(:,1);nutilde(:,2);nutilde(:,3)]))
                          mf(:,1)=Sigm*nutilde(:,1);
                          if int_likparam
                            if ~inputparam
                              mf(:,2)=Sigm2*(sum(nutilde(:,2))+nuprior);
                            else
                              mf(:,2)=Sigm2*nutilde(:,2);
                            end
                          end
                          if int_magnitude
                            if ~inputmagnitude
                              mf(:,ns)=Sigm3*(sum(nutilde(:,ns))+nuprior_magnitude);
                            else
                              mf(:,ns)=Sigm3*nutilde(:,ns);
                            end
                          end
%                             if any(1./diag(Sigm) - tautilde(:,1) < 1./diag(C))
%                               1;
%                             end
%                             if inputparam
%                               if any((1./diag(Sigm2) - tautilde(:,2)) + eps < 1./diag(C2))
%                                 1;
%                               end
%                             end
%                             if inputmagnitude
%                               if any((1./diag(Sigm3) - tautilde(:,ns)) + eps < 1./diag(C3))
%                                 1;
%                               end
%                             end
%                             if any(exp(mf(:,2))<-5)
%                               tautilde=tautilde_old;
%                               nutilde=nutile_old;
%                               Sigm2=Sigm2_old;
%                               if isequal(gp.latent_opt.display, 'iter')
%                                 fprintf('Bad means for theta\n');                                
%                               end
%                               rej=rej+1;
%                             end
                        end
                      end
                    end
                  end
                  
                  % Recompute the approximate posterior parameters
                  % parallel- and sequential-EP
                  if any(isnan(tautilde(:,1))) || any(isinf(tautilde(:,1)))
                    indt=find(isinf(tautilde(:,1)) | isnan(tautilde(:,1)));
                    tautilde(indt,:)=tautilde_old(indt,:);
                    nutilde(indt,:)=nutilde_old(indt,:);
                  end
                  if exist('joint_mean_magnitude', 'var') && joint_mean_magnitude
                    if any(isnan(tautilde(:))) || any(isinf(tautilde(:))) || ...
                        any(isnan(nutilde(:))) || any(isinf(nutilde(:)))
                      indt=find(isinf(tautilde) | isnan(tautilde) | ...
                        isinf(nutilde) | isnan(nutilde));
                      tautilde(indt)=tautilde_old(indt);
                      nutilde(indt)=nutilde_old(indt);
                      rej=length(indt);
                    end
                    %Sigm=blkdiag(C,C2,C3);
                    taut=diag([tautilde(:,1);tautilde(:,2);tautilde(:,3)]);
                    taut=taut+diag(tautilde(:,4),2*n)+diag(tautilde(:,4),-2*n);
                    % Natural parameter version
                    tmp=invC+taut;
                    % Covariance version
                    %                       tmp=inv(C)+inv(taut);
                    Sigm=inv(tmp);
                    nuut=[nutilde(:,1);nutilde(:,2);nutilde(:,3)];
                    % Natural parameter version
                    mf=tmp\nuut;
                    % Covariance version
                    % mf=tmp\(taut\nuut);
                    mf=reshape(mf,n,3);
                      
                    [~,npd]=chol(Sigm);

                    if ~npd
                      for ii=1:n
                        tt=Sigm([ii ii+n ii+2*n],[ii ii+n ii+2*n]);
                        tauu=[tautilde(ii,1) 0 tautilde(ii,4); ...
                          0 tautilde(ii,2) 0; ...
                          tautilde(ii,4) 0 tautilde(ii,3)];
                        % Natural parameter version
                        tt2=inv(inv(tt)-tauu);
                        % Covariance version
%                           tt2=inv(inv(tt)-inv(tauu));
                        [~,npd]=chol(tt2);
                        if npd
                          ss_vec_i(ii,:)=-1*ones(1,4);
                          break;
                        else
                          % Natural parameter version
                          nuu=tt\mf(ii,1:3)'-nutilde(ii,1:3)';
                          % Covariance version
%                             nuu=tt\mf(ii,1:3)'-tauu\nutilde(ii,1:3)';
                          mu_vec_i(ii,[1 2 3])=tt2*nuu;
                          ss_vec_i(ii,:)=[tt2(1,1) tt2(2,2) tt2(3,3) tt2(1,3)];
                        end
                      end
                      if  any(any(ss_vec_i(:,1:3)<0)) && isequal(display, 'iter')
                        fprintf('Bad cavity variances, recomputing with more damping. \n');
                      end
                    else
                      if isequal(display, 'iter')
                        fprintf('Posterior covariance not positive definite, recomputing with more damping. \n');
                      end
                      ss_vec_i=-1.*ones(size(tautilde));
                    end
                    while any(any(ss_vec_i(:,1:3)<0))
                      if isequal(display, 'iter')
                        fprintf('Bad cavity variances, recomputing with more damping. \n');
                      end
                      dfvec=0.1.*dfvec;
                      tautilde=tautilde_old+dfvec.*deltatautilde;
                      nutilde=nutilde_old+dfvec.*deltanutilde;
%                         indi=find(ss_vec_i(:,1:3)<0);
%                         tautilde(indi,:)=tautilde_old(indi,:);
%                         nutilde(indi,:)=nutilde_old(indi,:);
                      taut=diag([tautilde(:,1);tautilde(:,2);tautilde(:,3)]);
                      taut=taut+diag(tautilde(:,4),2*n)+diag(tautilde(:,4),-2*n);
                      % Natural parameter version
                      tmp=invC+taut;
                      % Covariance version
                      % tmp=inv(C)+inv(taut);
                      Sigm=inv(tmp);
                      nuut=[nutilde(:,1);nutilde(:,2);nutilde(:,3)];
                      % Natural parameter version
                      mf=tmp\nuut;
                      % Covariance version
                      % mf=tmp\(taut\nuut);
                      mf=reshape(mf,n,3);                        
                      
                      for ii=1:n
                        tt=Sigm([ii ii+n ii+2*n],[ii ii+n ii+2*n]);
                        tauu=[tautilde(ii,1) 0 tautilde(ii,4); ...
                          0 tautilde(ii,2) 0; ...
                          tautilde(ii,4) 0 tautilde(ii,3)];
                        % Natural parameter version
                        tt2=inv(inv(tt)-tauu);
                        nuu=tt\mf(ii,1:3)'-nutilde(ii,1:3)';
                        % Covariance version
%                           tt2=inv(inv(tt)-inv(tauu));
%                           nuu=tt\mf(ii,1:3)'-tauu\nutilde(ii,1:3)';
                        mu_vec_i(ii,[1 2 3])=tt2*nuu;
                        ss_vec_i(ii,[1 2 3 4])=[tt2(1,1) tt2(2,2) tt2(3,3) tt2(1,3)];
                      end
                      if all(dfvec<1e-3)
                        if isequal(display, 'iter')
                          fprintf('Could not find positive cavity variances. Resetting EP algorithm with more initial damping.\n');
                        end
                        df=0.5.*df;
                        df2=0.5.*df2;
                        Sigm=C;
                        tmp=eye(size(Sigm));
                        mf=[zeros(n,1) -2.*ones(n,1) zeros(n,1)];
                        tautilde=zeros(size(tautilde));
                        nutilde=zeros(size(nutilde));
                        break;
                      end
                    end
                    if any(diag(Sigm)<0)
                      if isequal(display, 'iter')
                        fprintf('Negative posterior variances, recomputing with more damping. \n');
                      end
                      indi=find(reshape(diag(Sigm),n,3)<0);
                      tautilde(indi)=tautilde_old(indi);
                      nutilde(indi)=nutilde_old(indi);
                      taut=diag([tautilde(:,1);tautilde(:,2);tautilde(:,3)]);
                      taut=taut+diag(tautilde(:,4),2*n)+diag(tautilde(:,4),-2*n);
                      % Natural parameter version
                      tmp=invC+taut;
                      % Covariance version
%                         tmp=inv(C)+inv(taut);
                      Sigm=inv(tmp);
                    end
                    [LS,npd]=chol(Sigm);
                    if npd
                      if isequal(display, 'iter')
                        fprintf('Posterior covariance not positive definite. Resetting EP algorithm with more initial damping.\n');
                      end
                      df=0.5.*df;
                      df2=0.5.*df2;
                      if all(df<1e-4) || all(df2<1e-4)
                        if isequal(display, 'iter')
                          fprintf('Could not find positive definite posterior covariance matrix even with high damping. Returnin NaN.\n');
                        end
                        [e, edata, eprior, param, ch] = set_output_for_notpositivedefinite();
                        return
                      end
                      Sigm=C;
                      tmp=eye(size(Sigm));
                      tautilde=zeros(size(tautilde));
                      nutilde=zeros(size(nutilde));
                    end
                    nuut=[nutilde(:,1);nutilde(:,2);nutilde(:,3)];
                    % Natural parameter version
                    mf=tmp\nuut;
                    % Covariance version
%                       mf=tmp\(taut\nuut);

%                       mf=tmp\(taut\([nutilde(:,1);nutilde(:,2)./tautilde(:,2);nutilde(:,3)]));
                    mf=reshape(mf,n,3);
%                       cov=diag(Sigm(1:n,2*n+1:end));
%                       Sigm3=diag(Sigm(2*n+1:end,2*n+1:end));
%                       figure(1);plot(x,y,'.',x,(mf(:,1)-cov./Sigm3.*mf(:,3)).*exp(0.5.*mf(:,3)+1/8.*Sigm3) ...
%                         + cov./Sigm3.*exp(0.5.*mf(:,3) + 1/8.*Sigm3).*(mf(:,3)+0.5.*Sigm3), '-k', ...
%                         x(reji),(mf(reji,1)-cov(reji)./Sigm3(reji).*mf(reji,3)).*exp(0.5.*mf(reji,3)+1/8.*Sigm3(reji)) ...
%                         + cov(reji)./Sigm3(reji).*exp(0.5.*mf(reji,3) + 1/8.*Sigm3(reji)).*(mf(reji,3)+0.5.*Sigm3(reji)),'.r');
%                       figure(3);plot(x,y,'.',x,mf(:,1));
%                       figure(2);plot(x,exp(0.5.*mf(:,2)));
%                       figure(4);plot(x,exp(0.5.*mf(:,3)));
                    % Natural parameter version
%                       [LL,npd]=chol(C+inv(taut),'lower');
%                       
%                       U=taut;U(1:3*n+1:end)=0;U(U<0)=0;
%                       U(1:n,1:n)=U(1:n,2*n+1:3*n);
%                       U(2*n+1:3*n,2*n+1:3*n)=U(1:n,2*n+1:3*n);
%                       U=sqrt(U./2);
%                       V=taut;V(1:3*n+1:end)=0;V(V>0)=0;V=abs(V);
%                       V(2*n+1:3*n,2*n+1:3*n)=V(1:n,2*n+1:3*n);
%                       V(1:n,1:n)=V(1:n,2*n+1:3*n);
%                       V=sqrt(V./2);
%                       D=diag(taut-U*U'+V*V');
%                       [~,A] = evaluate_q(zeros(3*n,1), D, C, display);
%                       Laa=chol(A);
% %                       A=inv(inv(C)+diag(D));
%                       La=chol(eye(3*n)+U*A*U);
%                       %B=inv(inv(A)+U*U);
%                       Lb=chol(eye(3*n)-V*((inv(A)+U*U)\V));
%                       
%                       LS=(-sum(log(diag(Lb))) - sum(log(diag(La))) ...
%                         +sum(log(diag(Laa))));
                    % Covariance version
%                       [LL,npd]=chol(C+taut,'lower');


                    term1=0.5.*mf(:)'*(Sigm\mf(:)) + sum(log(diag(LS))) + ...
                       - sum(log(diag(LC)));                     
                    
                    term2 = sum(logM0);
                    term3 = 0;
                    term4 = 0;
                    for i=1:n
                      sigm2v=[sigm2vec_i(i,1) 0 sigm2vec_i(i,4);0 sigm2vec_i(i,2) 0; ...
                        sigm2vec_i(i,4) 0 sigm2vec_i(i,3)];
                      muv=muvec_i(i,1:3)';
                      sigm2p = Sigm([i, i+n, i+2*n], [i, i+n, i+2*n]);
                      mup = mf(i,:)';
                      term3 = term3 + 0.5.*muv'*(sigm2v\muv) + 0.5.*log(det(sigm2v)) + 3/2*log(2*pi);
                      term4 = term4 - 0.5.*mup'*(sigm2p\mup) - 0.5.*log(det(sigm2p)) - 3/2*log(2*pi);
                    end
%                       term4=-n*term1;
                    logZep=term1+term2+term3+term4;
                      
%                       logZep=sum(logM0) - n/2*log(2*pi) -sum(log(diag(chol(C)))) ...
%                         - 0.5.*log(det(taut)) + (-sum(log(diag(Lb))) - sum(log(diag(La))) ...
%                         +sum(log(diag(Laa)))) - 0.5.*nuut'*((C+inv(taut))\nuut);
                    
                    
%                       if npd
%                         if isequal(display, 'iter')
%                           fprintf('Negative definite q-distribution\n');
%                         end
%                         chol(Sigm);
%                         % Natural parameter version
%                         [U,S,V]=svd(C+inv(taut));
%                         % Covariance version
% %                         [U,S,V]=svd(C+taut);
%                         %                           logZep=sum(logM0) - n/2*log(2*pi) - 0.5.*real(log(det(C+taut))) ...
%                         %                             -0.5.*nuut'*(taut\nuut) + 0.5.*mf(:)'*(tmp*mf(:));
%                         logZep=sum(logM0) - n/2*log(2*pi) - 0.5.*sum(log(diag(S))) ...
%                           + 0.5.*nuut'*(taut\nuut) - 0.5.*mf(:)'*(tmp*mf(:));
%                       else
%                         logZep=sum(logM0) - n/2*log(2*pi) - sum(log(diag(LL))) ...
%                           + 0.5.*nuut'*(taut\nuut) - 0.5.*mf(:)'*(tmp*mf(:));
%                         %                           dif=0.5.*real(log(det(C+taut))) - sum(log(diag(LL)))
%                       end
                    logZep=-logZep;
%                     evec(iter)=logZep;
                    L=LS;
                    B=1;
%                       if isnan(logZep)
%                         [e, edata, eprior, param, ch] = set_output_for_notpositivedefinite();
%                         return
%                       end
                    
                    iter=iter+1;
                    if ismember(display, {'iter', 'on'})
                      if exist('inputparam','var') && ~inputparam
                        if ~int_magnitude || inputmagnitude
                          fprintf('iter=%1.0f, mlpd=%.2f, dpd=%.3f, e=%.2f, de=%.3f, theta=%.5f, var(theta)=%.5f, rejected updates=%.0f\n', iter, sum(logM0), abs(sum(logM0_old)-sum(logM0)),logZep, abs(logZep_old-logZep), (mf(end,2)), Sigm2, rej);
                        elseif ~int_likparam
                          fprintf('iter=%1.0f, mlpd=%.2f, dpd=%.3f, e=%.2f, de=%.3f, phi=%.5f, var(phi)=%.5f, rejected updates=%.0f\n', iter, sum(logM0), abs(sum(logM0_old)-sum(logM0)),logZep, abs(logZep_old-logZep), (mf(end,2)), Sigm3, rej);
                        else
                          fprintf('iter=%1.0f, mlpd=%.2f, dpd=%.3f, e=%.2f, de=%.3f, theta=%.5f, var(theta)=%.5f, phi=%.5f, var(phi)=%.5f, rejected updates=%.0f\n', iter, sum(logM0), abs(sum(logM0_old)-sum(logM0)),logZep, abs(logZep_old-logZep),mf(end,2), Sigm2, mf(end,3), Sigm3,rej);
                        end
                      else
                        if int_magnitude && ~inputmagnitude
                          fprintf('iter=%1.0f, mlpd=%.2f, dpd=%.3f, e=%.2f, de=%.3f, phi=%.5f, var(phi)=%.5f, rejected updates=%.0f\n', iter, sum(logM0), abs(sum(logM0_old)-sum(logM0)),logZep, abs(logZep_old-logZep), mf(end,ns), Sigm3,rej);
                        else
                          fprintf('iter=%1.0f, mlpd=%.2f, dpd=%.3f, e=%.2f, de=%.3f, rejected updates=%.0f\n', iter, sum(logM0), abs(sum(logM0_old)-sum(logM0)),logZep, abs(logZep_old-logZep),rej);
                        end
                      end
                    end
                    
                  else
                    if ~meanfp % zero mean function used
                      % NOTICE! upper triangle matrix! cf. to
                      % line 13 in the algorithm 3.5, p. 58.
                      
                      [mf(:,1), Sigm, tmp, L1t, L2t] = evaluate_q(nutilde(:,1), tautilde(:,1), C, display);
                      if isempty(L1t)
                        [e, edata, eprior, param, ch] = set_output_for_notpositivedefinite();
                        return
                      end
                      
                      if isequal(gp.latent_opt.parallel, 'on')
                        df2=df2o;
                        df=df0;
                        rej=0;
                        clear('indt');
                        % Check that cavity distributions can be computed
                        while (isempty(L2t)) %|| any(1./diag(Sigm) - tautilde(:,1) < 0)
                          %                       while (any(tautilde(:,1) < 0) || notpositivedefinite) && size(deltatautilde,1)>1
                          if any(isnan(tautilde(:,1))) || any(isinf(tautilde(:,1)))
                            indt=find(isinf(tautilde(:,1)) | isnan(tautilde(:,1)));
                            tautilde(indt,:)=tautilde_old(indt,:);
                            nutilde(indt,:)=nutilde_old(indt,:);
                          end
                          %                           indt2=find(1./diag(Sigm) - tautilde(:,1) < 0);
                          indt=find(deltatautilde(:,1)<0);
                          %                           indt=unique([indt(:), indt2(:)]);
                          %                           indt=unique(find(tautilde(:,1)<0));
                          if isequal(gp.latent_opt.display, 'iter')
                            fprintf('Bad cavity distributions for f at %.0f sites, increasing damping.\n', length(indt));
                          end
                          dfvec(indt,1)=0.5.*dfvec(indt,1);
                          %                           dfvec(indt2,1)=0.1.*dfvec(indt2,1);
                          if all(dfvec(indt,1)<0.1)
                            dfvec(indt,1)=0;
                          end
                          %                           if all(dfvec(indt2,1)<0.1)
                          %                             dfvec(indt2,1)=0;
                          %                           end
                          rej=length(indt);
                          %                         end
                          %                         df2=0.5.*df2;
                          %                         if df2<0.05
                          %                           df2=0;
                          %                         end
                          %                           tautilde(:,1)=tautilde_old(:,1)+dfvec(:,1).*deltatautilde(:,1);
                          %                           nutilde(:,1)=nutilde_old(:,1)+dfvec(:,1).*deltanutilde(:,1);
                          tautilde(indt,1)=tautilde_old(indt,1)+dfvec(indt,1).*deltatautilde(indt,1);
                          nutilde(indt,1)=nutilde_old(indt,1)+dfvec(indt,1).*deltanutilde(indt,1);
                          %                           tautilde(indt2,1)=tautilde_old(indt2,1)+dfvec(indt2,1).*deltatautilde(indt2,1);
                          %                           nutilde(indt2,1)=nutilde_old(indt2,1)+dfvec(indt2,1).*deltanutilde(indt2,1);
                          [mf(:,1), Sigm, tmp, L1t, L2t] = evaluate_q(nutilde(:,1), tautilde(:,1), C, display);
                          if (all(dfvec(indt,1)==0)) && (isempty(L2t))
                            break;
                          end
                          %                             rej=length(indt);
                          %                           else
                          %                             rej=0;
                          %                           end
                          %                           if df2==0 && any(1./diag(Sigm) - tautilde(:,1) < 0)
                          %                             % Reset algorithm, increase damping
                          %                             error('foo');
                          %                           end
                        end
                        df=df0;
                        df2=df2o;
                      end
                      if isempty(L2t) || any(1./diag(Sigm) - tautilde(:,1) < 0)
                        tautilde=zeros(n,ns);
                        nutilde=zeros(n,ns);
                        muvec_i=zeros(n,ns);
                        sigm2vec_i=ones(n,ns);
                        logM0=-1e4;
                        df0=0.8.*df0;
                        df2o=0.8.*df2o;
                        if isequal(display, 'iter')
                          fprintf('Energy is inf, resetting tilted distribution for f & increasing damping.\n');
                        end
                        if df0<0.05 || df2o<0.05
                          [e, edata, eprior, param, ch] = set_output_for_notpositivedefinite();
                          return
                        end
                        [mf(:,1), Sigm, tmp, L1t, L2t] = evaluate_q(nutilde(:,1), tautilde(:,1), C, display);
                        
                      end
                      
                      % Cseke & Heskes (2011) Marginal likelihood
                      % psi(f) + psi(f_prior)
                      term1=0.5*mf(:,1)'*(Sigm\mf(:,1)) - sum(log(diag(L1t))) ...
                        - sum(log(diag(L2t)));
                      
                      % \sum_i logZ_i
                      term2=sum(logM0);
                      % sum_i psi(muvec_i,sigm2vec_i) - psi(mu_i, sigm2_i)
                      term3=sum(0.5.*muvec_i(:,1).^2./sigm2vec_i(:,1)+0.5.*log(sigm2vec_i(:,1)) ...
                        -0.5.*mf(:,1).^2./diag(Sigm)-0.5.*log(diag(Sigm)));
                      
                      logZep=-(term1+term2+term3);
                      
                      if (isinf(logZep) || (isnan(logZep) && iter>1) || ~isreal(logZep))
                        % Reset algorithm, increase damping
                        if isequal(display, 'iter')
                          fprintf('Energy is inf, resetting tilted distributions for f & increasing damping.\n');
                        end
                        tautilde(:,1)=zeros(n,1);
                        nutilde(:,1)=zeros(n,1);
                        muvec_i(:,1)=zeros(n,1);
                        sigm2vec_i(:,1)=ones(n,1);
                        df0=0.9.*df0;
                        df2o=0.9.*df2o;
                        if df0<0.1 || df2o<0.1
                          [e, edata, eprior, param, ch] = set_output_for_notpositivedefinite();
                          return
                        end
                        [mf(:,1), Sigm, tmp, L1t, L2t] = evaluate_q(nutilde(:,1), tautilde(:,1), C, display);
                        logZep=0;
                      end
                      
                      %if exist('dfvec','var') && ~all(dfvec(:,2)==0)
                      if int_likparam
                        if ~inputparam
                          
                          Sigm2=1./(tauprior+sum(tautilde(:,2)));
                          mf(:,2)=Sigm2*(sum(nutilde(:,2)) + nuprior);
                          
                          if isequal(gp.latent_opt.parallel, 'on')
                            % Check cavity distributions
                            df2=df2o;
                            while (Sigm2 < 0) || any(1./Sigm2 - tautilde(:,2) < 0)
                              indt=find(deltatautilde(:,2)<0);
                              if isequal(display, 'iter')
                                fprintf('Bad cavity distributions for theta at %.0f sites, increasing damping.\n', length(indt));
                              end
                              df2=0.5.*df2;
                              if df2<0.05
                                df2=0;
                              end
                              tautilde(indt,2)=tautilde_old(indt,2)+df2.*deltatautilde(indt,2);
                              nutilde(indt,2)=nutilde_old(indt,2)+df2.*deltanutilde(indt,2);
                              Sigm2=(tauprior+sum(tautilde(:,2)))^-1;
                              mf(:,2)=Sigm2*(sum(nutilde(:,2))+nuprior);
                              if df2==0
                                rej=length(indt);
                              end
                            end
                            df2=df2o;
                          end
                          
                          term1=(0.5.*mf(end,2)^2/Sigm2 + 0.5.*log(Sigm2)+0.5.*log(2*pi));
                          %term2=sum(logM0);
                          term2=0;
                          term3=sum(0.5.*muvec_i(:,2).^2./sigm2vec_i(:,2) + ...
                            0.5.*log(sigm2vec_i(:,2))+0.5.*log(2*pi) - term1);
                          term4=0.5*(nuprior/tauprior)^2*tauprior - 0.5*log(tauprior) + 0.5*log(2*pi);
                          logZep = logZep - (term1+term2+term3-term4);
                          
                          if (isinf(logZep) || (isnan(logZep) && iter>1) || ~isreal(logZep))
                            % Reset algorithm, increase damping
                            if isequal(display, 'iter')
                              fprintf('Energy is inf, resetting tilted distributions for theta & increasing damping.\n');
                            end
                            tautilde(:,2)=zeros(n,1);
                            nutilde(:,2)=zeros(n,1);
                            muvec_i(:,2)=zeros(n,1);
                            sigm2vec_i(:,2)=ones(n,1);
                            Sigm2=1./(tauprior+sum(tautilde(:,2)));
                            mf(:,2)=Sigm2*(sum(nutilde(:,2)) + nuprior);
                            df0=0.9*df0;
                            df=df0;
                            df2o=0.9*df2o;
                            df2=df2o;
                          end
                          
                          
                        else
                          [mf(:,2), Sigm2, tmp, L12, L22] = evaluate_q(nutilde(:,2), tautilde(:,2), C2, display);
                          if isempty(L12)
                            [e, edata, eprior, param, ch] = set_output_for_notpositivedefinite();
                            return
                          end
                          df2=df2o;
                          % Check that cavity distributions can be computed
                          %                         while (any(tautilde(:,2) < 0) || notpositivedefinite) %&& size(deltatautilde,1)>1
                          if isequal(gp.latent_opt.parallel, 'on')
                            while (isempty(L22) || any(1./diag(Sigm2) - tautilde(:,2) < 0)) && df2>0
                              if any(isnan(tautilde(:,2))) || any(isinf(tautilde(:,2)))
                                indt=find(isinf(tautilde(:,2)) | isnan(tautilde(:,2)));
                                tautilde(indt,:)=tautilde_old(indt,:);
                              end
                              indt=unique([find(dfvec(:,2).*deltatautilde(:,2)<0); find(deltatautilde(:,2)<0)]);
                              if isequal(display, 'iter')
                                fprintf('Bad cavity distributions for theta at %.0f sites, increasing damping.\n', length(indt));
                              end
                              %                                   df2=0.5.*df2;
                              %                                   if df2<0.05
                              df2=0;
                              %                                   end
                              tautilde(indt,2)=tautilde_old(indt,2)+df2.*deltatautilde(indt,2);
                              nutilde(indt,2)=nutilde_old(indt,2)+df2.*deltanutilde(indt,2);
                              [mf(:,2),Sigm2,tmp,L12,L22]=evaluate_q(nutilde(:,2),tautilde(:,2),C2,display);
                              if df2==0
                                rej=length(indt);
                              end
                            end
                          end
                          if (isempty(L22) || any(1./diag(Sigm2) - tautilde(:,2) < 0))
                            [e, edata, eprior, param, ch] = set_output_for_notpositivedefinite();
                            return
                          end
                          df2=df2o;
                          
                          % Cseke & Heskes (2011) Marginal likelihood
                          % psi(theta) + psi(theta_prior)
                          term1=0.5*mf(:,2)'*(Sigm2\mf(:,2)) - sum(log(diag(L12))) ...
                            - sum(log(diag(L22)));
                          
                          % sum_i psi(muvec_i,sigm2vec_i) - psi(mu_i, sigm2_i)
                          term2=sum(0.5.*muvec_i(:,2).^2./sigm2vec_i(:,2)+0.5.*log(sigm2vec_i(:,2)) ...
                            -0.5.*mf(:,2).^2./diag(Sigm2)-0.5.*log(diag(Sigm2)));
                          
                          logZep=logZep-(term1+term2);
                          
                          if (isinf(logZep) || (isnan(logZep) && iter>1) || ~isreal(logZep))
                            % Reset algorithm, increase damping
                            if isequal(display, 'iter')
                              fprintf('Energy is inf, resetting tilted distributions for theta & increasing damping.\n');
                            end
                            tautilde(:,2)=zeros(n,1);
                            nutilde(:,2)=zeros(n,1);
                            muvec_i(:,2)=zeros(n,1);
                            sigm2vec_i(:,2)=ones(n,1);
                            df0=0.9.*df0;
                            df2o=0.9.*df2o;
                            [mf(:,2), Sigm2, tmp, L12 L22] = evaluate_q(nutilde(:,2), tautilde(:,2), C2, display);
                            df=df0;
                            df2=df2o;
                          end
                        end
                      end
                      %end
                      if int_magnitude
                        if ~inputmagnitude
                          Sigm3=1./(tauprior_magnitude+sum(tautilde(:,ns)));
                          mf(:,ns)=Sigm3*(sum(nutilde(:,ns)) + nuprior_magnitude);
                          
                          if isequal(gp.latent_opt.parallel, 'on')
                            % Check cavity distributions
                            df2=df2o;
                            while (Sigm3 < 0) %|| any(1./Sigm3 - tautilde(:,ns) < 0)
                              indt=find(tautilde(:,ns)<0);
                              if isequal(display, 'iter')
                                fprintf('Bad cavity distributions for phi at %.0f sites, increasing damping.\n', length(indt));
                              end
                              df2=0.5.*df2;
                              if df2<0.05
                                df2=0;
                              end
                              tautilde(indt,ns)=tautilde_old(indt,ns)+df2.*deltatautilde(indt,ns);
                              nutilde(indt,ns)=nutilde_old(indt,ns)+df2.*deltanutilde(indt,ns);
                              Sigm3=(tauprior_magnitude+sum(tautilde(:,ns)))^-1;
                              mf(:,ns)=Sigm3*(sum(nutilde(:,ns))+nuprior_magnitude);
                              if df2==0
                                rej=length(indt);
                              end
                            end
                            df2=df2o;
                          end
                          
                          term1=(0.5.*mf(end,ns)^2/Sigm3 + 0.5.*log(Sigm3)+0.5.*log(2*pi));
                          %term2=sum(logM0);
                          term2=0;
                          term3=sum(0.5.*muvec_i(:,ns).^2./sigm2vec_i(:,ns) + ...
                            0.5.*log(sigm2vec_i(:,ns))+0.5.*log(2*pi) - term1);
                          term4=0.5*(nuprior_magnitude/tauprior_magnitude)^2*tauprior_magnitude ...
                            -0.5*log(tauprior_magnitude) + 0.5*log(2*pi);
                          logZep = logZep - (term1+term2+term3-term4);
                          
                          if (isinf(logZep) || (isnan(logZep) && iter>1) || ~isreal(logZep))
                            % Reset algorithm, increase damping
                            if isequal(display, 'iter')
                              fprintf('Energy is inf, resetting tilted distributions for phi & increasing damping.\n');
                            end
                            tautilde(:,ns)=zeros(n,1);
                            nutilde(:,ns)=zeros(n,1);
                            muvec_i(:,ns)=zeros(n,1);
                            sigm2vec_i(:,ns)=ones(n,1);
                            df0=0.9.*df0;
                            df2o=0.9.*df2o;
                            Sigm3=1./(tauprior_magnitude+sum(tautilde(:,ns)));
                            mf(:,ns)=Sigm3*(sum(nutilde(:,ns)) + nuprior_magnitude);
                            df=df0;
                            df2=df2o;
                          end
                        else
                          [mf(:,ns), Sigm3, tmp, L13, L23] = evaluate_q(nutilde(:,ns), tautilde(:,ns), C3, display);
                          if isempty(L13)
                            [e, edata, eprior, param, ch] = set_output_for_notpositivedefinite();
                            return
                          end
                          df2=df2o;
                          % Check that cavity distributions can be computed
                          %                         while (any(tautilde(:,2) < 0) || notpositivedefinite) %&& size(deltatautilde,1)>1
                          if isequal(gp.latent_opt.parallel, 'on')
                            while isempty(L23) || any(1./diag(Sigm3) - tautilde(:,ns) < 0)
                              %                               if any(isnan(tautilde(:,ns))) || any(isinf(tautilde(:,ns)))
                              %                                 error('foo');
                              indt=find(isinf(tautilde(:,3)) | isnan(tautilde(:,3)));
                              tautilde(indt,:)=tautilde_old(indt,:);
                              nutilde(indt,:)=nutilde_old(indt,:);
                              %                               else
                              indt=find(tautilde(:,ns)<0);
                              indt2=find(1./diag(Sigm3) - tautilde(:,ns) < 0);
                              if isequal(display, 'iter')
                                fprintf('Bad cavity distributions for phi at %.0f sites, increasing damping.\n', length(indt));
                              end
                              df2=0.5.*df2;
                              if df2<0.05
                                df2=0;
                              end
                              tautilde(indt,ns)=tautilde_old(indt,ns)+df2.*deltatautilde(indt,ns);
                              nutilde(indt,ns)=nutilde_old(indt,ns)+df2.*deltanutilde(indt,ns);
                              tautilde(indt2,ns)=tautilde_old(indt2,ns)+df2.*deltatautilde(indt2,ns);
                              nutilde(indt2,ns)=nutilde_old(indt2,ns)+df2.*deltanutilde(indt2,ns);
                              %                               end
                              [mf(:,ns),Sigm3,tmp,L13,L23]=evaluate_q(nutilde(:,ns),tautilde(:,ns),C3,display);
                              if df2==0
                                rej=length(indt);
                                if isempty(L23) || any(1./diag(Sigm3) - tautilde(:,ns) < 0)
                                  L23=[];
                                  break;
                                end
                              end
                            end
                          end
                          if isempty(L23) && any(1./diag(Sigm3) - tautilde(:,ns) < 0)
                            if isequal(display, 'iter')
                              fprintf('Energy is inf, resetting tilted distributions for phi & increasing damping.\n');
                            end
                            tautilde(:,ns)=zeros(n,1);
                            nutilde(:,ns)=zeros(n,1);
                            muvec_i(:,ns)=zeros(n,1);
                            sigm2vec_i(:,ns)=ones(n,1);
                            df0=0.9.*df0;
                            df2o=0.9.*df2o;
                            if df0<0.01 || df2o<0.01
                              [e, edata, eprior, param, ch] = set_output_for_notpositivedefinite();
                              return
                            end
                          end
                          df2=df2o;
                          
                          % Cseke & Heskes (2011) Marginal likelihood
                          % psi(theta) + psi(theta_prior)
                          term1=0.5*mf(:,ns)'*(Sigm3\mf(:,ns)) - sum(log(diag(L13))) ...
                            - sum(log(diag(L23)));
                          
                          % sum_i psi(muvec_i,sigm2vec_i) - psi(mu_i, sigm2_i)
                          term2=sum(0.5.*muvec_i(:,ns).^2./sigm2vec_i(:,ns)+0.5.*log(sigm2vec_i(:,ns)) ...
                            -0.5.*mf(:,ns).^2./diag(Sigm3)-0.5.*log(diag(Sigm3)));
                          
                          logZep=logZep-(term1+term2);
                          
                          if (isinf(logZep) || (isnan(logZep) && iter>1) || ~isreal(logZep))
                            % Reset algorithm, increase damping
                            if isequal(display, 'iter')
                              fprintf('Energy is inf, resetting tilted distributions for phi & increasing damping.\n');
                            end
                            tautilde(:,ns)=zeros(n,1);
                            nutilde(:,ns)=zeros(n,1);
                            muvec_i(:,ns)=zeros(n,1);
                            sigm2vec_i(:,ns)=ones(n,1);
                            df0=0.9.*df0;
                            df2o=0.9.*df2o;
                            [mf(:,ns), Sigm3, tmp, L13 L23] = evaluate_q(nutilde(:,ns), tautilde(:,ns), C3, display);
                            df=df0;
                            df2=df2o;
                          end
                        end
                      end
                      
%                       evec(iter)=logZep;
%                       mlpd(iter)=sum(logM0);
%                       if ns==3 && inputmagnitude
%                         nuvec(iter,:)=[nutilde(10,:)];
%                         tauvec(iter,:)=[tautilde(10,:)];
%                         nuvec2(iter,:)=[nutilde(20,:)];
%                         tauvec2(iter,:)=[tautilde(20,:)];
%                         nuvec3(iter,:)=[nutilde(30,:)];
%                         tauvec3(iter,:)=[tautilde(30,:)];
%                       end
                      iter=iter+1;
                      if ismember(display, {'iter', 'on'})
                        if exist('inputparam','var') && ~inputparam
                          if ~int_magnitude || inputmagnitude
                            fprintf('iter=%1.0f, mlpd=%.2f, dpd=%.3f, e=%.2f, de=%.3f, theta=%.5f, var(theta)=%.5f, rejected updates=%.0f\n', iter, sum(logM0), abs(sum(logM0_old)-sum(logM0)),logZep, abs(logZep_old-logZep), (mf(end,2)), Sigm2, rej);
                          elseif ~int_likparam
                            fprintf('iter=%1.0f, mlpd=%.2f, dpd=%.3f, e=%.2f, de=%.3f, phi=%.5f, var(phi)=%.5f, rejected updates=%.0f\n', iter, sum(logM0), abs(sum(logM0_old)-sum(logM0)),logZep, abs(logZep_old-logZep), (mf(end,2)), Sigm3, rej);
                          else
                            fprintf('iter=%1.0f, mlpd=%.2f, dpd=%.3f, e=%.2f, de=%.3f, theta=%.5f, var(theta)=%.5f, phi=%.5f, var(phi)=%.5f, rejected updates=%.0f\n', iter, sum(logM0), abs(sum(logM0_old)-sum(logM0)),logZep, abs(logZep_old-logZep),mf(end,2), Sigm2, mf(end,3), Sigm3,rej);
                          end
                        else
                          if int_magnitude && ~inputmagnitude
                            fprintf('iter=%1.0f, mlpd=%.2f, dpd=%.3f, e=%.2f, de=%.3f, phi=%.5f, var(phi)=%.5f, rejected updates=%.0f\n', iter, sum(logM0), abs(sum(logM0_old)-sum(logM0)),logZep, abs(logZep_old-logZep), mf(end,ns), Sigm3,rej);
                          else
                            fprintf('iter=%1.0f, mlpd=%.2f, dpd=%.3f, e=%.2f, de=%.3f, rejected updates=%.0f\n', iter, sum(logM0), abs(sum(logM0_old)-sum(logM0)),logZep, abs(logZep_old-logZep),rej);
                          end
                        end
                      end
                      B=1;
                    else
                      %                       % mean function used
                      %                       % help variables
                      %                       hBh = H'*B_m*H;
                      %                       C_t = C + hBh;
                      %                       CHb  = C\H'*b_m;
                      %                       S   = diag(Stildesqr.^2);
                      %                       %B = eye(n)+Stildesqroot*C*Stildesqroot;
                      %                       B=bsxfun(@times,bsxfun(@times,Stildesqr,C),Stildesqr');
                      %                       B(1:n+1:end)=B(1:n+1:end)+1;
                      %                       %B_h = eye(n) + Stildesqroot*C_t*Stildesqroot;
                      %                       B_h=bsxfun(@times,bsxfun(@times,Stildesqr,C_t),Stildesqr');
                      %                       B_h(1:n+1:end)=B_h(1:n+1:end)+1;
                      %                       % L to return, without the hBh term
                      %                       [L,notpositivedefinite]=chol(B,'lower');
                      %                       if notpositivedefinite
                      %                         [e, edata, eprior, param, ch] = set_output_for_notpositivedefinite();
                      %                         return
                      %                       end
                      %                       % L for the calculation with mean term
                      %                       [L_m,notpositivedefinite]=chol(B_h,'lower');
                      %                       if notpositivedefinite
                      %                         [e, edata, eprior, param, ch] = set_output_for_notpositivedefinite();
                      %                         return
                      %                       end
                      %
                      %                       % Recompute the approximate posterior parameters
                      %                       % parallel- and sequential-EP
                      %
                      %                       %V=(L_m\Stildesqroot)*C_t;
                      %                       V=L_m\bsxfun(@times,Stildesqr,C_t);
                      %                       Sigm=C_t-V'*V;
                      %                       mf=Sigm*(CHb+nutilde);
                      %
                      %                       T=1./sigm2vec_i;
                      %                       Cnutilde = (C_t - S^-1)*(S*H'*b_m-nutilde);
                      %                       L2 = V*(S*H'*b_m-nutilde);
                      %
                      %                       Stildesqroot = diag(Stildesqr);
                      %                       zz   = Stildesqroot*(L'\(L\(Stildesqroot*C)));
                      %                       % inv(K + S^-1)*S^-1
                      %                       Ks  = eye(size(zz)) - zz;
                      %
                      %                       % 5. term (1/2 element)
                      %                       term5_1  = 0.5.*((nutilde'*S^-1)./(T.^-1+Stilde.^-1)')*(S^-1*nutilde);
                      %                       % 2. term
                      %                       term2    = 0.5.*((S*H'*b_m-nutilde)'*Cnutilde - L2'*L2);
                      %                       % 4. term
                      %                       term4    = 0.5*sum(log(1+tautilde.*sigm2vec_i));
                      %                       % 1. term
                      %                       term1    = -1.*sum(log(diag(L_m)));
                      %                       % 3. term
                      %                       term3    = sum(logM0);
                      %                       % 5. term (2/2 element)
                      %                       term5    = 0.5*muvec_i'.*(T./(Stilde+T))'*(Stilde.*muvec_i-2*nutilde);
                      %
                      %                       logZep = -(term4+term1+term5_1+term5+term2+term3);
                      %
                      %                       iter=iter+1;
                      
                    end
                  end

                  convergence=max(abs(logM0_old-logM0))<tol && abs(logZep_old-logZep)<tol;
                  
                  if (iter==maxiter || convergence) && isequal(gp.latent_opt.display, 'final')
                    if exist('inputparam','var') && ~inputparam
                      if ~int_magnitude || inputmagnitude
                        fprintf('iter=%1.0f, mlpd=%.2f, dpd=%.3f, e=%.2f, de=%.3f, theta=%.5f, var(theta)=%.5f, rejected updates=%.0f\n', iter, sum(logM0), abs(sum(logM0_old)-sum(logM0)),logZep, abs(logZep_old-logZep), (mf(end,2)), Sigm2, rej);
                      elseif ~int_likparam
                        fprintf('iter=%1.0f, mlpd=%.2f, dpd=%.3f, e=%.2f, de=%.3f, phi=%.5f, var(phi)=%.5f, rejected updates=%.0f\n', iter, sum(logM0), abs(sum(logM0_old)-sum(logM0)),logZep, abs(logZep_old-logZep), (mf(end,2)), Sigm3, rej);
                      else
                        fprintf('iter=%1.0f, mlpd=%.2f, dpd=%.3f, e=%.2f, de=%.3f, theta=%.5f, var(theta)=%.5f, phi=%.5f, var(phi)=%.5f, rejected updates=%.0f\n', iter, sum(logM0), abs(sum(logM0_old)-sum(logM0)),logZep, abs(logZep_old-logZep),mf(end,2), Sigm2, mf(end,3), Sigm3,rej);
                      end
                    else
                      if int_magnitude && ~inputmagnitude
                        fprintf('iter=%1.0f, mlpd=%.2f, dpd=%.3f, e=%.2f, de=%.3f, phi=%.5f, var(phi)=%.5f, rejected updates=%.0f\n', iter, sum(logM0), abs(sum(logM0_old)-sum(logM0)),logZep, abs(logZep_old-logZep), mf(end,ns), Sigm3,rej);
                      else
                        fprintf('iter=%1.0f, mlpd=%.2f, dpd=%.3f, e=%.2f, de=%.3f, rejected updates=%.0f\n', iter, sum(logM0), abs(sum(logM0_old)-sum(logM0)),logZep, abs(logZep_old-logZep),rej);
                      end
                    end
                  end
                  
                  if iter==maxiter && (abs((logZep_old-logZep)/logZep)>0.001)
                    warning('maxiter reached, increase maxiter or tol');
                    [e, edata, eprior, param, ch] = set_output_for_notpositivedefinite();
                    return
                  end
                end
                
%---------------% <--
%---------------% Skip intendation   
                else
%---------------% Skip intendation
%---------------% -->
                
                
                [K,C] = gp_trcov(gp, x);
                
                if ~issparse(C)
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
                        % sequential-EP
                        for i1=1:n
                          % Algorithm utilizing Cholesky updates
                          % This is numerically more stable but slower
                          % $$$                             % approximate cavity parameters
                          % $$$                             S11 = sum(Ls(:,i1).^2);
                          % $$$                             S1 = Ls'*Ls(:,i1);
                          % $$$                             tau_i=S11^-1-tautilde(i1);
                          % $$$                             nu_i=S11^-1*mf(i1)-nutilde(i1);
                          % $$$
                          % $$$                             mu_icovg=nu_i/tau_i;
                          % $$$                             sigm2_i=tau_i^-1;
                          % $$$
                          % $$$                             if sigm2_i < 0
                          % $$$                                 [ii i1]
                          % $$$                             end
                          % $$$
                          % $$$                             % marginal moments
                          % $$$                             [M0(i1), muhat, sigm2hat] = feval(gp.lik.fh.tiltedMoments, gp.lik, y, i1, sigm2_i, mu_i, z);
                          % $$$
                          % $$$                             % update site parameters
                          % $$$                             deltatautilde = sigm2hat^-1-tau_i-tautilde(i1);
                          % $$$                             tautilde(i1) = tautilde(i1)+deltatautilde;
                          % $$$                             nutilde(i1) = sigm2hat^-1*muhat-nu_i;
                          % $$$
                          % $$$                             upfact = 1./(deltatautilde^-1+S11);
                          % $$$                             if upfact > 0
                          % $$$                                 Ls = cholupdate(Ls, S1.*sqrt(upfact), '-');
                          % $$$                             else
                          % $$$                                 Ls = cholupdate(Ls, S1.*sqrt(-upfact));
                          % $$$                             end
                          % $$$                             Sigm = Ls'*Ls;
                          % $$$                             mf=Sigm*nutilde;
                          % $$$
                          % $$$                             muvec_i(i1,1)=mu_i;
                          % $$$                             sigm2vec_i(i1,1)=sigm2_i;
                          
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
                  
                else
                  % EP algorithm for compactly supported covariance function
                  % (C is a sparse matrix)
                  p = analyze(K);
                  r(p) = 1:n;
                  if ~isempty(z)
                    z = z(p,:);
                  end
                  y = y(p);
                  K = K(p,p);
                  
                  Inn = sparse(1:n,1:n,1,n,n);
                  sqrtS = sparse(1:n,1:n,0,n,n);
                  mf = zeros(size(y));
                  sigm2 = zeros(size(y));
                  dSigm=full(diag(K));
                  gamma = zeros(size(y));
                  VD = sparse(1:n,1:n,1,n,n);
                  
                  % The EP -algorithm
                  convergence=false;
                  while iter<=maxiter && ~convergence
                    logZep_old=logZep;
                    logM0_old=logM0;
                    
                    if isequal(gp.latent_opt.parallel,'on')
                      % parallel-EP
                      % approximate cavity parameters
                      sqrtSK = ssmult(sqrtS, K);
                      tttt = ldlsolve(VD,sqrtSK);
                      sigm2 = full(diag(K) - sum(sqrtSK.*tttt)');
                      mf = gamma - tttt'*sqrtS*gamma;
                      tau=1./sigm2-tautilde;
                      nu = 1./sigm2.*mf-nutilde;
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
                      tautilde=tautilde+df*deltatautilde;
                      deltanutilde=1./sigm2hat.*muhat-nu-nutilde;
                      nutilde=nutilde+df*deltanutilde;
                      gamma = gamma + sum(bsxfun(@times,K,df.*deltanutilde'),2);
                    else
                      % sequential-EP
                      muvec_i = zeros(n,1); sigm2vec_i = zeros(n,1);
                      for i1=1:n
                        % approximate cavity parameters
                        Ki1 = K(:,i1);
                        sqrtSKi1 = ssmult(sqrtS, Ki1);
                        tttt = ldlsolve(VD,sqrtSKi1);
                        sigm2(i1) = Ki1(i1) - sqrtSKi1'*tttt;
                        mf(i1) = gamma(i1) - tttt'*sqrtS*gamma;
                        
                        tau_i=sigm2(i1)^-1-tautilde(i1);
                        nu_i=sigm2(i1)^-1*mf(i1)-nutilde(i1);
                        
                        mu_i=nu_i/tau_i;
                        sigm2_i=tau_i^-1;
                        
                        % marginal moments
                        [logM0(i1), muhat(i1), sigm2hat(i1)] = gp.lik.fh.tiltedMoments(gp.lik, y, i1, sigm2_i, mu_i, z);
                        
                        % update site parameters
                        tautilde_old = tautilde(i1);
                        deltatautilde=sigm2hat(i1)^-1-tau_i-tautilde(i1);
                        tautilde(i1)=tautilde(i1)+df*deltatautilde;
                        deltanutilde=sigm2hat(i1)^-1*muhat(i1)-nu_i-nutilde(i1);
                        nutilde(i1)=nutilde(i1)+df*deltanutilde;
                        gamma = gamma + Ki1.*df*deltanutilde;
                        
                        % Update the LDL decomposition
                        sqrtS(i1,i1) = sqrt(tautilde(i1));
                        sqrtSKi1(i1) = sqrt(tautilde(i1)).*Ki1(i1);
                        D2_n = sqrtSKi1.*sqrtS(i1,i1) + Inn(:,i1);
                        
                        if tautilde_old == 0
                          VD = ldlrowupdate(i1,VD,VD(:,i1),'-');
                          VD = ldlrowupdate(i1,VD,D2_n,'+');
                        else
                          VD = ldlrowmodify(VD, D2_n, i1);
                        end
                        
                        muvec_i(i1,1)=mu_i;
                        sigm2vec_i(i1,1)=sigm2_i;
                      end
                      
                    end
                    
                    % Recompute the approximate posterior parameters
                    % parallel- and sequential-EP
                    sqrtS = sparse(1:n,1:n,sqrt(tautilde),n,n);
                    KsqrtS = ssmult(K,sqrtS);
                    B = ssmult(sqrtS,KsqrtS) + Inn;
                    [VD, notpositivedefinite] = ldlchol(B);
                    if notpositivedefinite
                      [e, edata, eprior, param, ch] = set_output_for_notpositivedefinite();
                      return
                    end
                    Knutilde = K*nutilde;
                    mf = Knutilde - KsqrtS*ldlsolve(VD,sqrtS*Knutilde);
                    
                    % Compute the marginal likelihood
                    % 4. term & 1. term
                    term41=0.5*sum(log(1+tautilde.*sigm2vec_i)) - 0.5.*sum(log(diag(VD)));
                    
                    % 5. term (1/2 element) & 2. term
                    T=1./sigm2vec_i;
                    term52 = nutilde'*mf - (nutilde'./(T+tautilde)')*nutilde;
                    term52 = term52.*0.5;
                    
                    % 5. term (2/2 element)
                    term5=0.5*muvec_i'.*(T./(tautilde+T))'*(tautilde.*muvec_i-2*nutilde);
                    
                    % 3. term
                    term3 = sum(logM0);
                    
                    logZep = -(term41+term52+term5+term3);
                    
                    iter=iter+1;
                    
                    convergence=max(abs(logM0_old-logM0))<tol && abs(logZep_old-logZep)<tol;
                    %[iter-1 max(abs(muhat-mf)./abs(mf)) max(abs(sqrt(sigm2hat)-s)./abs(s)) max(abs(logM0_old-logM0)) abs(logZep_old-logZep)]
                    %[iter-1 max(abs(muhat-mf)./abs(mf)) max(abs(logM0_old-logM0)) abs(logZep_old-logZep)]
                  end
                  % Reorder all the returned and stored values
                  
                  B = B(r,r);
                  nutilde = nutilde(r);
                  tautilde = tautilde(r);
                  muvec_i = muvec_i(r);
                  sigm2vec_i = sigm2vec_i(r);
                  logM0 = logM0(r);
                  mf = mf(r);
                  y = y(r);
                  if ~isempty(z)
                    z = z(r,:);
                  end
                  [L, notpositivedefinite] = ldlchol(B);
                  if notpositivedefinite
                    [e, edata, eprior, param, ch] = set_output_for_notpositivedefinite();
                    return
                  end
                  
                end
                La2 = B;

                
%---------------% <--
%---------------% Skip intendation   
                end
                
                edata = logZep;
                % Set something into La2
                % La2 = B;
                b = 0;
                
                % ============================================================
                % FIC
                % ============================================================
              case 'FIC'
                u = gp.X_u;
                m = size(u,1);
                
                % First evaluate needed covariance matrices
                % v defines that parameter is a vector
                [Kv_ff, Cv_ff] = gp_trvar(gp, x);  % f x 1  vector
                K_fu = gp_cov(gp, x, u);           % f x u
                K_uu = gp_trcov(gp, u);     % u x u, noiseles covariance K_uu
                K_uu = (K_uu+K_uu')./2;     % ensure the symmetry of K_uu
                [Luu, notpositivedefinite] = chol(K_uu, 'lower');
                if notpositivedefinite
                  [e, edata, eprior, param, ch] = set_output_for_notpositivedefinite();
                  return
                end
                % Evaluate the Lambda (La)
                % Q_ff = K_fu*inv(K_uu)*K_fu'
                % Here we need only the diag(Q_ff), which is evaluated below
                B=Luu\(K_fu');       % u x f
                Qv_ff=sum(B.^2)';
                Lav = Cv_ff-Qv_ff;   % f x 1, Vector of diagonal elements
                % iLaKfu = diag(iLav)*K_fu = inv(La)*K_fu
                % First some helper parameters
                iLaKfu = zeros(size(K_fu));  % f x u,
                for i=1:n
                  iLaKfu(i,:) = K_fu(i,:)./Lav(i);  % f x u
                end
                A = K_uu+K_fu'*iLaKfu;  A = (A+A')./2;     % Ensure symmetry
                [A, notpositivedefinite] = chol(A);
                if notpositivedefinite
                  [e, edata, eprior, param, ch] = set_output_for_notpositivedefinite();
                  return
                end
                L = iLaKfu/A;
                Lahat = 1./Lav;
                I = eye(size(K_uu));
                
                [R0, notpositivedefinite] = chol(inv(K_uu));
                if notpositivedefinite
                  [e, edata, eprior, param, ch] = set_output_for_notpositivedefinite();
                  return
                end
                R = R0;
                P = K_fu;
                mf = zeros(size(y));
                eta = zeros(size(y));
                gamma = zeros(size(K_uu,1),1);
                D_vec = Lav;
                Ann=0;
                
                % The EP -algorithm
                convergence=false;
                while iter<=maxiter && ~convergence
                  logZep_old=logZep;
                  logM0_old=logM0;
                  
                  if isequal(gp.latent_opt.parallel,'on')
                    % parallel-EP
                    % approximate cavity parameters
                    Ann = D_vec+sum((P*R').^2,2);
                    mf = eta + sum(bsxfun(@times,P,gamma'),2);
                    tau = 1./Ann-tautilde;
                    nu = 1./Ann.*mf-nutilde;
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
                    tautilde=tautilde+df*deltatautilde;
                    deltanutilde=1./sigm2hat.*muhat-nu-nutilde;
                    nutilde=nutilde+df*deltanutilde;
                  else
                    % sequential-EP
                    muvec_i = zeros(n,1); sigm2vec_i = zeros(n,1);
                    for i1=1:n
                      % approximate cavity parameters
                      pn = P(i1,:)';
                      Ann = D_vec(i1) + sum((R*pn).^2);
                      tau_i = Ann^-1-tautilde(i1);
                      mf(i1) = eta(i1) + pn'*gamma;
                      nu_i = Ann^-1*mf(i1)-nutilde(i1);
                      
                      mu_i=nu_i/tau_i;
                      sigm2_i=tau_i^-1;
                      
                      % marginal moments
                      [logM0(i1), muhat(i1), sigm2hat(i1)] = gp.lik.fh.tiltedMoments(gp.lik, y, i1, sigm2_i, mu_i, z);
                      
                      % update site parameters
                      deltatautilde = sigm2hat(i1)^-1-tau_i-tautilde(i1);
                      tautilde(i1) = tautilde(i1)+df*deltatautilde;
                      deltanutilde = sigm2hat(i1)^-1*muhat(i1)-nu_i - nutilde(i1);
                      nutilde(i1) = nutilde(i1)+df*deltanutilde;
                      
                      % Update the parameters
                      dn = D_vec(i1);
                      D_vec(i1) = D_vec(i1) - deltatautilde.*D_vec(i1).^2 ./ (1+deltatautilde.*D_vec(i1));
                      P(i1,:) = pn' - (deltatautilde.*dn ./ (1+deltatautilde.*dn)).*pn';
                      updfact = deltatautilde./(1 + deltatautilde.*Ann);
                      if updfact > 0
                        RtRpnU = R'*(R*pn).*sqrt(updfact);
                        R = cholupdate(R, RtRpnU, '-');
                      elseif updfact < 0
                        RtRpnU = R'*(R*pn).*sqrt(abs(updfact));
                        R = cholupdate(R, RtRpnU, '+');
                      end
                      eta(i1) = eta(i1) + (deltanutilde - deltatautilde.*eta(i1)).*dn./(1+deltatautilde.*dn);
                      gamma = gamma + (deltanutilde - deltatautilde.*mf(i1))./(1+deltatautilde.*dn) * R'*(R*pn);
                      %                            mf = eta + P*gamma;
                      
                      % Store cavity parameters
                      muvec_i(i1,1)=mu_i;
                      sigm2vec_i(i1,1)=sigm2_i;
                    end
                  end
                  
                  % Recompute the approximate posterior parameters
                  % parallel- and sequential-EP
                  temp1 = (1+Lav.*tautilde).^(-1);
                  D_vec = temp1.*Lav;
                  R0P0t = R0*K_fu';
                  temp2 = zeros(size(R0P0t));
                  %                for i2 = 1:length(temp1)
                  %                  P(i2,:) = temp1(i2).*K_fu(i2,:);
                  %                  temp2(:,i2) = R0P0t(:,i2).*tautilde(i2).*temp1(i2);
                  %                end
                  %                R = chol(inv(eye(size(R0)) + temp2*R0P0t')) * R0;
                  P=bsxfun(@times,temp1,K_fu);
                  temp2=bsxfun(@times,(tautilde.*temp1)',R0P0t);
                  temp2=temp2*R0P0t';
                  temp2(1:m+1:end)=temp2(1:m+1:end)+1;
                  R = chol(inv(temp2)) * R0;
                  eta = D_vec.*nutilde;
                  gamma = R'*(R*(P'*nutilde));
                  mf = eta + P*gamma;
                  
                  % Compute the marginal likelihood, see FULL model for
                  % details about equations
                  Lahat = 1./Lav + tautilde;
                  Lhat = bsxfun(@rdivide,L,Lahat);
                  H = I-L'*Lhat;
                  B = H\L';
                  Bhat = B./repmat(Lahat',m,1);
                  
                  % 4. term & 1. term
                  Stildesqroot=sqrt(tautilde);
                  D = Stildesqroot.*Lav.*Stildesqroot + 1;
                  SsqrtKfu = K_fu.*repmat(Stildesqroot,1,m);
                  AA = K_uu + (SsqrtKfu'./repmat(D',m,1))*SsqrtKfu; AA = (AA+AA')/2;
                  [AA, notpositivedefinite] = chol(AA,'lower');
                  if notpositivedefinite
                    [e, edata, eprior, param, ch] = set_output_for_notpositivedefinite();
                    return
                  end
                  term41 = - 0.5*sum(log(1+tautilde.*sigm2vec_i)) - sum(log(diag(Luu))) + sum(log(diag(AA))) + 0.5.*sum(log(D));
                  
                  % 5. term (1/2 element) & 2. term
                  T=1./sigm2vec_i;
                  term52 = -0.5*( (nutilde./Lahat)'*nutilde + (nutilde'*Lhat)*(Bhat*nutilde) - (nutilde./(T+tautilde))'*nutilde);
                  
                  % 5. term (2/2 element)
                  term5 = - 0.5*muvec_i'.*(T./(tautilde+T))'*(tautilde.*muvec_i-2*nutilde);
                  
                  % 3. term
                  term3 = -sum(logM0);
                  
                  logZep = term41+term52+term5+term3;
                  
                  iter=iter+1;
                  convergence=max(abs(logM0_old-logM0))<tol && abs(logZep_old-logZep)<tol;
                end
                edata = logZep;
                %L = iLaKfu;
                
                % b'  = (La + Kfu*iKuu*Kuf + 1./S)*1./S * nutilde
                %     = (S - S * (iLa - L*L' + S)^(-1) * S) * 1./S
                %     = I - S * (Lahat - L*L')^(-1)
                % L   = S*Kfu * (Lav + 1./S)^(-1) / chol(K_uu + SsqrtKfu'*(Lav + 1./S)^(-1)*SsqrtKfu)
                % La2 = D./S = Lav + 1./S,
                %
                % The way evaluations are done is numerically more stable
                % See equations (3.71) and (3.72) in Rasmussen and Williams (2006)
                b = nutilde'.*(1 - Stildesqroot./Lahat.*Stildesqroot)' - (nutilde'*Lhat)*Bhat.*tautilde';    % part of eq. (3.71)
                L = ((repmat(Stildesqroot,1,m).*SsqrtKfu)./repmat(D',m,1)')/AA';                             % part of eq. (3.72)
                La2 = 1./(Stildesqroot./D.*Stildesqroot);                                                    % part of eq. (3.72)
                D = D_vec;
                
                % ============================================================
                % PIC
                % ============================================================
              case {'PIC' 'PIC_BLOCK'}
                ind = gp.tr_index;
                u = gp.X_u;
                m = length(u);
                
                % First evaluate needed covariance matrices
                % v defines that parameter is a vector
                K_fu = gp_cov(gp, x, u);         % f x u
                
                K_uu = gp_trcov(gp, u);    % u x u, noiseles covariance K_uu
                K_uu = (K_uu+K_uu')./2;     % ensure the symmetry of K_uu
                [Luu, notpositivedefinite] = chol(K_uu, 'lower');
                if notpositivedefinite
                  [e, edata, eprior, param, ch] = set_output_for_notpositivedefinite();
                  return
                end
                % Evaluate the Lambda (La)
                % Q_ff = K_fu*inv(K_uu)*K_fu'
                % Here we need only the diag(Q_ff), which is evaluated below
                B=Luu\(K_fu');       % u x f
                
                % First some helper parameters
                iLaKfu = zeros(size(K_fu));  % f x u
                for i=1:length(ind)
                  Qbl_ff = B(:,ind{i})'*B(:,ind{i});
                  [Kbl_ff, Cbl_ff] = gp_trcov(gp, x(ind{i},:));
                  Labl{i} = Cbl_ff - Qbl_ff;
                  [Llabl, notpositivedefinite] = chol(Labl{i});
                  if notpositivedefinite
                    [e, edata, eprior, param, ch] = set_output_for_notpositivedefinite();
                    return
                  end
                  iLaKfu(ind{i},:) = Llabl\(Llabl'\K_fu(ind{i},:));
                end
                A = K_uu+K_fu'*iLaKfu;
                A = (A+A')./2;     % Ensure symmetry
                [A, notpositivedefinite] = chol(A);
                if notpositivedefinite
                  [e, edata, eprior, param, ch] = set_output_for_notpositivedefinite();
                  return
                end
                L = iLaKfu/A;
                I = eye(size(K_uu));
                
                [R0, notpositivedefinite] = chol(inv(K_uu));
                if notpositivedefinite
                  [e, edata, eprior, param, ch] = set_output_for_notpositivedefinite();
                  return
                end
                R = R0;
                P = K_fu;
                R0P0t = R0*K_fu';
                mf = zeros(size(y));
                eta = zeros(size(y));
                gamma = zeros(size(K_uu,1),1);
                D = Labl;
                Ann=0;
                
                % The EP -algorithm
                convergence=false;
                while iter<=maxiter && ~convergence
                  logZep_old=logZep;
                  logM0_old=logM0;
                  
                  if isequal(gp.latent_opt.parallel,'on')
                    % parallel-EP
                    % approximate cavity parameters
                    for bl=1:length(ind)
                      bl_ind = ind{bl};
                      Pbl=P(bl_ind,:);
                      Ann = diag(D{bl}) +sum((Pbl*R').^2,2);
                      tau(bl_ind,1) = 1./Ann-tautilde(bl_ind);
                      mf(bl_ind,1) = eta(bl_ind) + sum(bsxfun(@times,Pbl,gamma'),2);
                      nu(bl_ind,1) = 1./Ann.*mf(bl_ind)-nutilde(bl_ind);
                    end
                    muvec_i=nu./tau;
                    sigm2vec_i=1./tau;
                    % compute moments of tilted distributions
                    [logM0, muhat, sigm2hat] = gp.lik.fh.tiltedMoments(gp.lik, y, 1:n, sigm2vec_i, muvec_i, z);
                    if any(isnan(logM0))
                      [e, edata, eprior, param, ch] = set_output_for_notpositivedefinite();
                      return
                    end
                    % update site parameters
                    deltatautilde = 1./sigm2hat-tau-tautilde;
                    tautilde = tautilde+df*deltatautilde;
                    deltanutilde = 1./sigm2hat.*muhat-nu-nutilde;
                    nutilde = nutilde+df*deltanutilde;;
                  else
                    
                    muvec_i = zeros(n,1); sigm2vec_i = zeros(n,1);
                    for bl=1:length(ind)
                      bl_ind = ind{bl};
                      for in=1:length(bl_ind)
                        i1 = bl_ind(in);
                        % approximate cavity parameters
                        Dbl = D{bl}; dn = Dbl(in,in); pn = P(i1,:)';
                        Ann = dn + sum((R*pn).^2);
                        tau_i = Ann^-1-tautilde(i1);
                        mf(i1) = eta(i1) + pn'*gamma;
                        nu_i = Ann^-1*mf(i1)-nutilde(i1);
                        
                        mu_i=nu_i/tau_i;
                        sigm2_i=tau_i^-1;
                        
                        % marginal moments
                        [logM0(i1), muhat(i1), sigm2hat(i1)] = gp.lik.fh.tiltedMoments(gp.lik, y, i1, sigm2_i, mu_i, z);
                        
                        % update site parameters
                        deltatautilde = sigm2hat(i1)^-1-tau_i-tautilde(i1);
                        tautilde(i1) = tautilde(i1)+df*deltatautilde;
                        deltanutilde = sigm2hat(i1)^-1*muhat(i1)-nu_i - nutilde(i1);
                        nutilde(i1) = nutilde(i1) + df*deltanutilde;
                        
                        % Update the parameters
                        Dblin = Dbl(:,in);
                        Dbl = Dbl - deltatautilde ./ (1+deltatautilde.*dn) * Dblin*Dblin';
                        %Dbl = inv(inv(Dbl) + diag(tautilde(bl_ind)));
                        P(bl_ind,:) = P(bl_ind,:) - ((deltatautilde ./ (1+deltatautilde.*dn)).* Dblin)*pn';
                        updfact = deltatautilde./(1 + deltatautilde.*Ann);
                        if updfact > 0
                          RtRpnU = R'*(R*pn).*sqrt(updfact);
                          R = cholupdate(R, RtRpnU, '-');
                        elseif updfact < 0
                          RtRpnU = R'*(R*pn).*sqrt(abs(updfact));
                          R = cholupdate(R, RtRpnU, '+');
                        end
                        eta(bl_ind) = eta(bl_ind) + (deltanutilde - deltatautilde.*eta(i1))./(1+deltatautilde.*dn).*Dblin;
                        gamma = gamma + (deltanutilde - deltatautilde.*mf(i1))./(1+deltatautilde.*dn) * (R'*(R*pn));
                        %mf = eta + P*gamma;
                        
                        D{bl} = Dbl;
                        % Store cavity parameters
                        muvec_i(i1,1)=mu_i;
                        sigm2vec_i(i1,1)=sigm2_i;
                      end
                    end
                  end
                  
                  % Recompute the approximate posterior parameters
                  % parallel- and sequential-EP
                  temp2 = zeros(size(R0P0t));
                  
                  Stildesqroot=sqrt(tautilde);
                  for i=1:length(ind)
                    sdtautilde = diag(Stildesqroot(ind{i}));
                    Dhat = sdtautilde*Labl{i}*sdtautilde + eye(size(Labl{i}));
                    [Ldhat{i}, notpositivedefinite] = chol(Dhat);
                    if notpositivedefinite
                      [e, edata, eprior, param, ch] = set_output_for_notpositivedefinite();
                      return
                    end
                    D{i} = Labl{i} - Labl{i}*sdtautilde*(Ldhat{i}\(Ldhat{i}'\sdtautilde*Labl{i}));
                    P(ind{i},:) = D{i}*(Labl{i}\K_fu(ind{i},:));
                    
                    temp2(:,ind{i}) = R0P0t(:,ind{i})*sdtautilde/Dhat*sdtautilde;
                    eta(ind{i}) = D{i}*nutilde(ind{i});
                  end
                  R = chol(inv(eye(size(R0)) + temp2*R0P0t')) * R0;
                  gamma = R'*(R*(P'*nutilde));
                  mf = eta + P*gamma;
                  
                  % Compute the marginal likelihood, see FULL model for
                  % details about equations
                  %
                  % First some helper parameters
                  for i = 1:length(ind)
                    Lhat(ind{i},:) = D{i}*L(ind{i},:);
                  end
                  H = I-L'*Lhat;
                  B = H\L';
                  
                  % Compute the marginal likelihood, see FULL model for
                  % details about equations
                  term41 = 0; term52 = 0;
                  for i=1:length(ind)
                    Bhat(:,ind{i}) = B(:,ind{i})*D{i};
                    SsqrtKfu(ind{i},:) = bsxfun(@times,K_fu(ind{i},:),Stildesqroot(ind{i}));
                    %SsqrtKfu(ind{i},:) = gtimes(K_fu(ind{i},:),Stildesqroot(ind{i}));
                    iDSsqrtKfu(ind{i},:) = Ldhat{i}\(Ldhat{i}'\SsqrtKfu(ind{i},:));
                    term41 = term41 + sum(log(diag(Ldhat{i})));
                    term52 = term52 + nutilde(ind{i})'*(D{i}*nutilde(ind{i}));
                    
                  end
                  AA = K_uu + SsqrtKfu'*iDSsqrtKfu; AA = (AA+AA')/2;
                  [AA, notpositivedefinite] = chol(AA,'lower');
                  if notpositivedefinite
                    [e, edata, eprior, param, ch] = set_output_for_notpositivedefinite();
                    return
                  end
                  term41 = term41 - 0.5*sum(log(1+tautilde.*sigm2vec_i)) - sum(log(diag(Luu))) + sum(log(diag(AA)));
                  
                  % 5. term (1/2 element) & 2. term
                  T=1./sigm2vec_i;
                  term52 = -0.5*( term52 + (nutilde'*Lhat)*(Bhat*nutilde) - (nutilde./(T+tautilde))'*nutilde);
                  
                  % 5. term (2/2 element)
                  term5 = - 0.5*muvec_i'.*(T./(tautilde+T))'*(tautilde.*muvec_i-2*nutilde);
                  
                  % 3. term
                  term3 = -sum(logM0);
                  
                  logZep = term41+term52+term5+term3;
                  
                  iter=iter+1;
                  convergence=max(abs(logM0_old-logM0))<tol && abs(logZep_old-logZep)<tol;
                end
                edata = logZep;
                
                b = zeros(1,n);
                for i=1:length(ind)
                  b(ind{i}) = nutilde(ind{i})'*D{i};
                  La2{i} = inv(diag(Stildesqroot(ind{i}))*(Ldhat{i}\(Ldhat{i}'\diag(Stildesqroot(ind{i})))));
                end
                b = nutilde' - ((b + (nutilde'*Lhat)*Bhat).*tautilde');
                
                L = (repmat(Stildesqroot,1,m).*iDSsqrtKfu)/AA';
                
                % ============================================================
                % CS+FIC
                % ============================================================
              case 'CS+FIC'
                u = gp.X_u;
                m = length(u);
                cf_orig = gp.cf;
                
                cf1 = {};
                cf2 = {};
                j = 1;
                k = 1;
                for i = 1:ncf
                  if ~isfield(gp.cf{i},'cs')
                    cf1{j} = gp.cf{i};
                    j = j + 1;
                  else
                    cf2{k} = gp.cf{i};
                    k = k + 1;
                  end
                end
                gp.cf = cf1;
                
                % First evaluate needed covariance matrices
                % v defines that parameter is a vector
                [Kv_ff, Cv_ff] = gp_trvar(gp, x);  % f x 1  vector
                K_fu = gp_cov(gp, x, u);           % f x u
                K_uu = gp_trcov(gp, u);            % u x u, noiseles covariance K_uu
                K_uu = (K_uu+K_uu')./2;            % ensure the symmetry of K_uu
                [Luu, notpositivedefinite] = chol(K_uu, 'lower');
                if notpositivedefinite
                  [e, edata, eprior, param, ch] = set_output_for_notpositivedefinite();
                  return
                end
                
                % Evaluate the Lambda (La)
                % Q_ff = K_fu*inv(K_uu)*K_fu'
                B=Luu\(K_fu');       % u x f
                Qv_ff=sum(B.^2)';
                Lav = Cv_ff-Qv_ff;   % f x 1, Vector of diagonal elements
                
                gp.cf = cf2;
                K_cs = gp_trcov(gp,x);
                La = sparse(1:n,1:n,Lav,n,n) + K_cs;
                gp.cf = cf_orig;
                
                % clear unnecessary variables
                clear K_cs; clear Qv_ff; clear Kv_ff; clear Cv_ff; clear Lav;
                
                % Find fill reducing permutation and permute all the
                % matrices
                p = analyze(La);
                r(p) = 1:n;
                if ~isempty(z)
                  z = z(p,:);
                end
                y = y(p);
                La = La(p,p);
                K_fu = K_fu(p,:);
                
                [VD, notpositivedefinite] = ldlchol(La);
                if notpositivedefinite
                  [e, edata, eprior, param, ch] = set_output_for_notpositivedefinite();
                  return
                end
                iLaKfu = ldlsolve(VD,K_fu);
                A = K_uu+K_fu'*iLaKfu; A = (A+A')./2;     % Ensure symmetry
                [A, notpositivedefinite] = chol(A);
                if notpositivedefinite
                  [e, edata, eprior, param, ch] = set_output_for_notpositivedefinite();
                  return
                end
                L = iLaKfu/A;
                
                I = eye(size(K_uu));
                
                Inn = sparse(1:n,1:n,1,n,n);
                sqrtS = sparse(1:n,1:n,0,n,n);
                [R0, notpositivedefinite] = chol(inv(K_uu));
                if notpositivedefinite
                  [e, edata, eprior, param, ch] = set_output_for_notpositivedefinite();
                  return
                end
                R = R0;
                P = K_fu;
                R0P0t = R0*K_fu';
                mf = zeros(size(y));
                eta = zeros(size(y));
                gamma = zeros(size(K_uu,1),1);
                Ann=0;
                LasqrtS = La*sqrtS;
                [VD, notpositivedefinite] = ldlchol(Inn);
                if notpositivedefinite
                  [e, edata, eprior, param, ch] = set_output_for_notpositivedefinite();
                  return
                end
                
                % The EP -algorithm
                convergence=false;
                while iter<=maxiter && ~convergence
                  logZep_old=logZep;
                  logM0_old=logM0;
                  
                  if isequal(gp.latent_opt.parallel,'on')
                    % parallel-EP
                    % approximate cavity parameters
                    tttt = ldlsolve(VD,ssmult(sqrtS,La));
                    D_vec = full(diag(La) - sum(LasqrtS'.*tttt)');
                    Ann = D_vec+sum((P*R').^2,2);
                    mf = eta + sum(bsxfun(@times,P,gamma'),2);
                    tau = 1./Ann-tautilde;
                    nu = 1./Ann.*mf-nutilde;
                    muvec_i=nu./tau;
                    sigm2vec_i= 1./tau;
                    % compute moments of tilted distributions
                    [logM0, muhat, sigm2hat] = gp.lik.fh.tiltedMoments(gp.lik, y, 1:n, sigm2vec_i, muvec_i, z);
                    if any(isnan(logM0))
                      [e, edata, eprior, param, ch] = set_output_for_notpositivedefinite();
                      return
                    end
                    % update site parameters
                    deltatautilde=1./sigm2hat-tau-tautilde;
                    tautilde=tautilde+df*deltatautilde;
                    deltanutilde=1./sigm2hat.*muhat-nu-nutilde;
                    nutilde=nutilde+df*deltanutilde;
                  else
                    % sequential-EP
                    muvec_i = zeros(n,1); sigm2vec_i = zeros(n,1);
                    for i1=1:n
                      % approximate cavity parameters
                      tttt = ldlsolve(VD,ssmult(sqrtS,La(:,i1)));
                      Di1 =  La(:,i1) - ssmult(LasqrtS,tttt);
                      
                      dn = Di1(i1);
                      pn = P(i1,:)';
                      Ann = dn + sum((R*pn).^2);
                      tau_i = Ann^-1-tautilde(i1);
                      mf(i1) = eta(i1) + pn'*gamma;
                      nu_i = Ann^-1*mf(i1)-nutilde(i1);
                      
                      mu_i=nu_i/tau_i;
                      sigm2_i= tau_i^-1;  % 1./tau_i;  %
                      
                      % marginal moments
                      [logM0(i1), muhat(i1), sigm2hat(i1)] = gp.lik.fh.tiltedMoments(gp.lik, y, i1, sigm2_i, mu_i, z);
                      
                      % update site parameters
                      deltatautilde = sigm2hat(i1)^-1-tau_i-tautilde(i1);
                      tautilde(i1) = tautilde(i1)+df*deltatautilde;
                      deltanutilde = sigm2hat(i1)^-1*muhat(i1)-nu_i - nutilde(i1);
                      nutilde(i1) = nutilde(i1) + df*deltanutilde;
                      
                      % Update the parameters
                      P = P - ((deltatautilde ./ (1+deltatautilde.*dn)).* Di1)*pn';
                      updfact = deltatautilde./(1 + deltatautilde.*Ann);
                      if updfact > 0
                        RtRpnU = R'*(R*pn).*sqrt(updfact);
                        R = cholupdate(R, RtRpnU, '-');
                      elseif updfact < 0
                        RtRpnU = R'*(R*pn).*sqrt(abs(updfact));
                        R = cholupdate(R, RtRpnU, '+');
                      end
                      eta = eta + (deltanutilde - deltatautilde.*eta(i1))./(1+deltatautilde.*dn).*Di1;
                      gamma = gamma + (deltanutilde - deltatautilde.*mf(i1))./(1+deltatautilde.*dn) * (R'*(R*pn));
                      
                      % Store cavity parameters
                      muvec_i(i1,1)=mu_i;
                      sigm2vec_i(i1,1)=sigm2_i;
                      
                      D2_o = ssmult(sqrtS,LasqrtS(:,i1)) + Inn(:,i1);
                      sqrtS(i1,i1) = sqrt(tautilde(i1));
                      LasqrtS(:,i1) = La(:,i1).*sqrtS(i1,i1);
                      D2_n = ssmult(sqrtS,LasqrtS(:,i1)) + Inn(:,i1);
                      
                      if tautilde(i1) - deltatautilde == 0
                        VD = ldlrowupdate(i1,VD,VD(:,i1),'-');
                        VD = ldlrowupdate(i1,VD,D2_n,'+');
                      else
                        VD = ldlrowmodify(VD, D2_n, i1);
                      end
                    end
                  end
                  
                  % Recompute the approximate posterior parameters
                  % parallel- and sequential-EP
                  sqrtS = sparse(1:n,1:n,sqrt(tautilde),n,n);
                  sqrtSLa = ssmult(sqrtS,La);
                  D2 = ssmult(sqrtSLa,sqrtS) + Inn;
                  LasqrtS = ssmult(La,sqrtS);
                  [VD, notpositivedefinite] = ldlchol(D2);
                  if notpositivedefinite
                    [e, edata, eprior, param, ch] = set_output_for_notpositivedefinite();
                    return
                  end
                  
                  SsqrtKfu = sqrtS*K_fu;
                  iDSsqrtKfu = ldlsolve(VD,SsqrtKfu);
                  P = K_fu - sqrtSLa'*iDSsqrtKfu;
                  R = chol(inv( eye(size(R0)) + R0P0t*sqrtS*ldlsolve(VD,sqrtS*R0P0t'))) * R0;
                  eta = La*nutilde - sqrtSLa'*ldlsolve(VD,sqrtSLa*nutilde);
                  gamma = R'*(R*(P'*nutilde));
                  mf = eta + P*gamma;
                  
                  % Compute the marginal likelihood,
                  Lhat = La*L - sqrtSLa'*ldlsolve(VD,sqrtSLa*L);
                  H = I-L'*Lhat;
                  B = H\L';
                  Bhat = B*La - ldlsolve(VD,sqrtSLa*B')'*sqrtSLa;
                  
                  % 4. term & 1. term
                  AA = K_uu + SsqrtKfu'*iDSsqrtKfu; AA = (AA+AA')/2;
                  [AA, notpositivedefinite] = chol(AA,'lower');
                  if notpositivedefinite
                    [e, edata, eprior, param, ch] = set_output_for_notpositivedefinite();
                    return
                  end
                  term41 = - 0.5*sum(log(1+tautilde.*sigm2vec_i)) - sum(log(diag(Luu))) + sum(log(diag(AA))) + 0.5*sum(log(diag(VD)));
                  
                  % 5. term (1/2 element) & 2. term
                  T=1./sigm2vec_i;
                  term52 = -0.5*( nutilde'*(eta) + (nutilde'*Lhat)*(Bhat*nutilde) - (nutilde./(T+tautilde))'*nutilde);
                  
                  % 5. term (2/2 element)
                  term5 = - 0.5*muvec_i'.*(T./(tautilde+T))'*(tautilde.*muvec_i-2*nutilde);
                  
                  % 3. term
                  term3 = -sum(logM0);
                  
                  logZep = term41+term52+term5+term3;
                  
                  iter=iter+1;
                  convergence=max(abs(logM0_old-logM0))<tol && abs(logZep_old-logZep)<tol;
                end
                edata = logZep;
                
                % b'  = (K_fu/K_uu*K_fu' + La + diag(1./tautilde)) \ (tautilde.\nutilde)
                % L   = S*Kfu * (Lav + 1./S)^(-1) / chol(K_uu + SsqrtKfu'*(Lav + 1./S)^(-1)*SsqrtKfu)
                % La2 = D./S = Lav + 1./S,
                %
                % The way evaluations are done is numerically more stable than with inversion of S (tautilde)
                % See equations (3.71) and (3.72) in Rasmussen and Williams (2006)
                b = nutilde' - ((eta' + (nutilde'*Lhat)*Bhat).*tautilde');
                
                L = (sqrtS*iDSsqrtKfu)/AA';
                La2 = sqrtS\D2/sqrtS;
                
                % Reorder all the returned and stored values
                b = b(r);
                L = L(r,:);
                La2 = La2(r,r);
                D = La(r,r);
                nutilde = nutilde(r);
                tautilde = tautilde(r);
                logM0 = logM0(r);
                muvec_i = muvec_i(r);
                sigm2vec_i = sigm2vec_i(r);
                mf = mf(r);
                P = P(r,:);
                y = y(r);
                if ~isempty(z)
                  z = z(r,:);
                end
                % ============================================================
                % DTC,VAR
                % ============================================================
              case {'DTC' 'VAR' 'SOR'}
                % First evaluate needed covariance matrices
                % v defines that parameter is a vector
                u = gp.X_u;
                m = size(u,1);
                
                % First evaluate needed covariance matrices
                % v defines that parameter is a vector
                [Kv_ff, Cv_ff] = gp_trvar(gp, x);  % f x 1  vector
                K_fu = gp_cov(gp, x, u);           % f x u
                K_uu = gp_trcov(gp, u);     % u x u, noiseles covariance K_uu
                K_uu = (K_uu+K_uu')./2;     % ensure the symmetry of K_uu
                [Luu, notpositivedefinite] = chol(K_uu, 'lower');
                if notpositivedefinite
                  [e, edata, eprior, param, ch] = set_output_for_notpositivedefinite();
                  return
                end
                % Evaluate the Lambda (La)
                % Q_ff = K_fu*inv(K_uu)*K_fu'
                % Here we need only the diag(Q_ff), which is evaluated below
                B=Luu\(K_fu');       % u x f
                
                
                Phi = B';
                m = size(Phi,2);
                
                R = eye(m,m);
                P = Phi;
                mf = zeros(size(y));
                gamma = zeros(m,1);
                Ann=0;
                
                % The EP -algorithm
                convergence=false;
                while iter<=maxiter && ~convergence
                  logZep_old=logZep;
                  logM0_old=logM0;
                  
                  if isequal(gp.latent_opt.parallel,'on')
                    % parallel-EP
                    % approximate cavity parameters
                    Ann = sum((P*R').^2,2);
                    mf = sum(bsxfun(@times,Phi,gamma'),2);%phi'*gamma;
                    tau = 1./Ann-tautilde;
                    nu = 1./Ann.*mf-nutilde;
                    muvec_i=nu./tau;
                    sigm2vec_i= 1./tau;
                    % compute moments of tilted distributions
                    [logM0, muhat, sigm2hat] = gp.lik.fh.tiltedMoments(gp.lik, y, 1:n, sigm2vec_i, muvec_i, z);
                    if any(isnan(logM0))
                      [e, edata, eprior, param, ch] = set_output_for_notpositivedefinite();
                      return
                    end
                    % update site parameters
                    deltatautilde=1./sigm2hat-tau-tautilde;
                    tautilde=tautilde+df*deltatautilde;
                    deltanutilde=1./sigm2hat.*muhat-nu-nutilde;
                    nutilde=nutilde+df*deltanutilde;
                  else
                    % sequential-EP
                    muvec_i = zeros(n,1); sigm2vec_i = zeros(n,1);
                    for i1=1:n
                      % approximate cavity parameters
                      phi = Phi(i1,:)';
                      Ann = sum((R*phi).^2);
                      tau_i = Ann^-1-tautilde(i1);
                      mf(i1) = phi'*gamma;
                      nu_i = Ann^-1*mf(i1)-nutilde(i1);
                      
                      mu_i=nu_i/tau_i;
                      sigm2_i=tau_i^-1;
                      
                      % marginal moments
                      [logM0(i1), muhat(i1), sigm2hat(i1)] = gp.lik.fh.tiltedMoments(gp.lik, y, i1, sigm2_i, mu_i, z);
                      
                      % update site parameters
                      deltatautilde = sigm2hat(i1)^-1-tau_i-tautilde(i1);
                      tautilde(i1) = tautilde(i1)+df*deltatautilde;
                      deltanutilde = sigm2hat(i1)^-1*muhat(i1)-nu_i - nutilde(i1);
                      nutilde(i1) = nutilde(i1) + df*deltanutilde;
                      
                      % Update the parameters
                      lnn = sum((R*phi).^2);
                      updfact = deltatautilde/(1 + deltatautilde*lnn);
                      if updfact > 0
                        RtLphiU = R'*(R*phi).*sqrt(updfact);
                        R = cholupdate(R, RtLphiU, '-');
                      elseif updfact < 0
                        RtLphiU = R'*(R*phi).*sqrt(updfact);
                        R = cholupdate(R, RtLphiU, '+');
                      end
                      gamma = gamma - R'*(R*phi)*(deltatautilde*mf(i1)-deltanutilde);
                      % Store cavity parameters
                      muvec_i(i1,1)=mu_i;
                      sigm2vec_i(i1,1)=sigm2_i;
                    end
                  end
                  
                  % Recompute the approximate posterior parameters
                  % parallel- and sequential-EP
                  R = chol(inv(eye(m,m) + Phi'*(repmat(tautilde,1,m).*Phi)));
                  gamma = R'*(R*(Phi'*nutilde));
                  mf = Phi*gamma;
                  
                  % Compute the marginal likelihood, see FULL model for
                  % details about equations
                  % 4. term & 1. term
                  Stildesqroot=sqrt(tautilde);
                  SsqrtPhi = Phi.*repmat(Stildesqroot,1,m);
                  AA = eye(m,m) + SsqrtPhi'*SsqrtPhi; AA = (AA+AA')/2;
                  [AA, notpositivedefinite] = chol(AA,'lower');
                  if notpositivedefinite
                    [e, edata, eprior, param, ch] = set_output_for_notpositivedefinite();
                    return
                  end
                  term41 = - 0.5*sum(log(1+tautilde.*sigm2vec_i)) + sum(log(diag(AA)));
                  
                  % 5. term (1/2 element) & 2. term
                  T=1./sigm2vec_i;
                  bb = nutilde'*Phi;
                  bb2 = bb*SsqrtPhi';
                  bb3 = bb2*SsqrtPhi/AA';
                  term52 = -0.5*( bb*bb' - bb2*bb2' + bb3*bb3' - (nutilde./(T+tautilde))'*nutilde);
                  
                  % 5. term (2/2 element)
                  term5 = - 0.5*muvec_i'.*(T./(tautilde+T))'*(tautilde.*muvec_i-2*nutilde);
                  
                  % 3. term
                  term3 = -sum(logM0);
                  
                  logZep = term41+term52+term5+term3;
                  
                  iter=iter+1;
                  convergence=max(abs(logM0_old-logM0))<tol && abs(logZep_old-logZep)<tol;
                end
                edata = logZep;
                %L = iLaKfu;
                if strcmp(gp.type,'VAR')
                  Qv_ff = sum(B.^2)';
                  edata = edata + 0.5*sum((Kv_ff-Qv_ff).*tautilde);
                end
                
                temp = Phi*(SsqrtPhi'*(SsqrtPhi*bb'));
                %                b = Phi*bb' - temp + Phi*(SsqrtPhi'*(SsqrtPhi*(AA'\(AA\temp))));
                
                b = nutilde - bb2'.*Stildesqroot + repmat(tautilde,1,m).*Phi*(AA'\bb3');
                b = b';
                
                %                 StildeKfu = zeros(size(K_fu));  % f x u,
                %                 for i=1:n
                %                     StildeKfu(i,:) = K_fu(i,:).*tautilde(i);  % f x u
                %                 end
                %                 A = K_uu+K_fu'*StildeKfu;  A = (A+A')./2;     % Ensure symmetry
                %                 A = chol(A);
                %                 L = StildeKfu/A;
                L = repmat(tautilde,1,m).*Phi/AA';
                %L = repmat(tautilde,1,m).*K_fu/AA';
                mu=nutilde./tautilde;
                %b = nutilde - mu'*L*L'*mu;
                %b=b';
                La2 = 1./tautilde;
                D = 0;
                
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
            
            % Evaluate the prior contribution to the error from the inducing inputs
            if ~isempty(strfind(gp.infer_params, 'inducing'))
              if isfield(gp, 'p') && isfield(gp.p, 'X_u') && ~isempty(gp.p.X_u)
                if iscell(gp.p.X_u) % Own prior for each inducing input
                  for i = 1:size(gp.X_u,1)
                    pr = gp.p.X_u{i};
                    eprior = eprior - pr.fh.lp(gp.X_u(i,:), pr);
                  end
                else
                  eprior = eprior - gp.p.X_u.fh.lp(gp.X_u(:), gp.p.X_u);
                end
              end
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
          case 'robust-EP'
            
            % The parameters or data have changed since
            % the last call for gpep_e. In this case we need to
            % re-evaluate the EP approximation
            
            % preparations
            ninit=gp.latent_opt.ninit; % max number of initial parallel iterations
            maxiter=gp.latent_opt.maxiter; % max number of double-loop iterations
            max_ninner=gp.latent_opt.max_ninner; % max number of inner loop iterations in the double-loop algorithm
            tolStop=gp.latent_opt.tolStop; % converge tolerance
            tolUpdate=gp.latent_opt.tolUpdate; % tolerance for the EP site updates
            tolInner=gp.latent_opt.tolInner; % inner loop energy tolerance
            tolGrad=gp.latent_opt.tolGrad; % minimum gradient (g) decrease in the search direction, abs(g_new)<tolGrad*abs(g)
            Vc_lim=gp.latent_opt.cavity_var_lim; % limit for the cavity variance Vc, Vc < Vc_lim*diag(K)
            df0=gp.latent_opt.df; % the intial damping factor
            eta1=gp.latent_opt.eta; % the initial fraction parameter
            eta2=gp.latent_opt.eta2; % the secondary fraction parameter
            display=gp.latent_opt.display; % control the display
            
            gp=gp_unpak(gp,w);
            likelih=gp.lik;
            ncf = length(gp.cf);
            n=length(y);
            pvis=0;
            
            eta=repmat(eta1,n,1);  % the initial vector of fraction parameters
            fh_tm=@(si,m_c,V_c,eta) likelih.fh.tiltedMoments2(likelih,y,si,V_c,m_c,z,eta);
            
            switch gp.type
              case 'FULL'
                % prior covariance
                K = gp_trcov(gp, x);
                
              case 'FIC'
                % Sparse (only FIC implemented)
                u = gp.X_u;
                m = size(u,1);
                K_uu = gp_trcov(gp,u);
                K_uu = (K_uu + K_uu')./2;
                K_fu = gp_cov(gp,x,u);
                [Kv_ff, Cv_ff] = gp_trvar(gp,x);
                [Luu, notpositivedefinite] = chol(K_uu, 'lower');
                if notpositivedefinite
                  [e, edata, eprior, param, ch] = set_output_for_notpositivedefinite();
                  return
                end
                B=Luu\(K_fu');
                Qv_ff=sum(B.^2)';
                Sf = [];
                Sf2 = [];
                L2 = [];
            end
            
            % prior (zero) initialization
            [nu_q,tau_q]=deal(zeros(n,1));
            
            % initialize the q-distribution (the multivariate Gaussian posterior approximation)
            switch gp.type
              case 'FULL'
                [mf,Sf,lnZ_q]=evaluate_q(nu_q,tau_q,K,display);
                Vf = diag(Sf);
              case 'FIC'
                [mf,Vf,lnZ_q]=evaluate_q2(nu_q,tau_q,Luu, K_fu, Kv_ff, Qv_ff, display);
              otherwise
                error('Robust-EP not implemented for this type of GP!');
            end
            
            % initialize the surrogate distribution (the independent Gaussian marginal approximations)
            nu_s=mf./Vf;
            tau_s=1./Vf;
            lnZ_s=0.5*sum( (-log(tau_s) +nu_s.^2 ./tau_s)./eta ); % minus 0.5*log(2*pi)./eta
            
            % initialize r-distribution (the tilted distributions)
            [lnZ_r,lnZ_i,m_r,V_r]=evaluate_r(nu_q,tau_q,eta,fh_tm,nu_s,tau_s,display);
            
            % initial energy (lnZ_ep)
            e = lnZ_q + lnZ_r -lnZ_s;
            
            if ismember(display,{'iter'})
              fprintf('\nInitial energy: e=%.4f, hyperparameters:\n',e)
              fprintf('Cov:%s \n',sprintf(' %.2g,',gp_pak(gp,'covariance')))
              fprintf('Lik:%s \n',sprintf(' %.2g,',gp_pak(gp,'likelihood')))
            end
            
            if isfinite(e) % do not run the algorithm if the prior energy is not defined
              
              %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
              % initialize with ninit rounds of parallel EP
              %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
              
              % EP search direction
              up_mode='ep'; % choose the moment matching
              [dnu_q,dtau_q]=ep_update_dir(mf,Vf,m_r,V_r,eta,up_mode,tolUpdate);
              
              convergence=false; % convergence indicator
              df=df0; % initial damping factor
              tol_m=zeros(1,2); % absolute moment tolerances
              switch gp.type
                case 'FULL'
                  tauc_min=1./(Vc_lim*diag(K)); % minimum cavity precision
                case 'FIC'
                  tauc_min=1./(Vc_lim*Cv_ff);
              end
              % Adjust damping by setting an upper limit (Vf_mult) to the increase
              % of the marginal variance
              Vf_mult=2;
              i1=0;
              while i1<ninit
                i1=i1+1;
                
                %%%%%%%%%%%%%%%%%%%
                % the damped update
                dfi=df(ones(n,1));
                temp=(1/Vf_mult-1)./Vf;
                ii2=df*dtau_q<temp;
                if any(ii2)
                  dfi(ii2)=temp(ii2)./dtau_q(ii2);
                end
                
                % proposal site parameters
                nu_q2=nu_q+dfi.*dnu_q;
                tau_q2=tau_q+dfi.*dtau_q;
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%
                % a proposal q-distribution
                switch gp.type
                  case 'FULL'
                    [mf2,Sf2,lnZ_q2,L1,L2]=evaluate_q(nu_q2,tau_q2,K,display);
                    Vf2 = diag(Sf2);
                  case 'FIC'
                    [mf2,Vf2,lnZ_q2,L1,L2]=evaluate_q2(nu_q2,tau_q2,Luu, K_fu, Kv_ff, Qv_ff, display);
                  otherwise
                    error('Robust-EP not implemented for this type of GP!');
                end
                
                % check that the new cavity variances do not exceed the limit
                tau_s2=1./Vf2;
                pcavity=all( (tau_s2-eta.*tau_q2 )>=tauc_min);
                if isempty(L2) || ~pcavity
                  % In case of too small cavity precisions, half the step size
                  df=df*0.5;
                  if df<0.1,
                    % If mediocre damping is not sufficient, proceed to
                    % the double-loop algorithm
                    break
                  else
                    if ismember(display,{'iter'})
                      fprintf('%d, e=%.6f, dm=%.4f, dV=%.4f, increasing damping to df=%g.\n',i1,e,tol_m(1),tol_m(2),df)
                    end
                    continue
                  end
                end
                
                % a proposal surrogate distribution
                nu_s2=mf2./Vf2;
                lnZ_s2=0.5*sum( (-log(tau_s2) +nu_s2.^2 ./tau_s2)./eta );
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%
                % a proposal r-distribution
                [lnZ_r2,lnZ_i2,m_r2,V_r2,p]=evaluate_r(nu_q2,tau_q2,eta,fh_tm,nu_s2,tau_s2,display);
                
                % the new energy
                e2 = lnZ_q2 + lnZ_r2 -lnZ_s2;
                
                % check that the energy is defined and that the tilted moments are proper
                if ~all(p) || ~isfinite(e2)
                  df=df*0.5;
                  if df<0.1,
                    break
                  else
                    if ismember(display,{'iter'})
                      fprintf('%d, e=%.6f, dm=%.4f, dV=%.4f, increasing damping to df=%g.\n',i1,e,tol_m(1),tol_m(2),df)
                    end
                    continue
                  end
                end
                
                % accept the new state
                [nu_q,tau_q,mf,Vf,Sf,lnZ_q]=deal(nu_q2,tau_q2,mf2,Vf2,Sf2,lnZ_q2);
                [lnZ_r,lnZ_i,m_r,V_r,lnZ_s,nu_s,tau_s]=deal(lnZ_r2,lnZ_i2,m_r2,V_r2,lnZ_s2,nu_s2,tau_s2);
                
                % EP search direction (moment matching)
                [dnu_q,dtau_q]=ep_update_dir(mf,Vf,m_r,V_r,eta,up_mode,tolUpdate);
                
                % Check for convergence
                % the difference between the marginal moments
                %               Vf=diag(Sf);
                tol_m=[abs(mf-m_r) abs(Vf-V_r)];
                
                % measure the convergence by the moment difference
                convergence=all(tol_m(:,1)<tolStop*abs(mf)) && all(tol_m(:,2)<tolStop*abs(Vf));
                
                % measure the convergence by the change of energy
                %convergence=abs(e2-e)<tolStop;
                
                tol_m=max(tol_m);
                e=e2;
                
                if ismember(display,{'iter'})
                  fprintf('%d, e=%.6f, dm=%.4f, dV=%.4f, df=%g.\n',i1,e,tol_m(1),tol_m(2),df)
                end
                
                if convergence
                  if ismember(display,{'final','iter'})
                    fprintf('Convergence with parallel EP, iter %d, e=%.6f, dm=%.4f, dV=%.4f, df=%g.\n',i1,e,tol_m(1),tol_m(2),df)
                  end
                  break
                end
              end
            end % end of initial rounds of parallel EP
            
            if isfinite(e) && ~convergence
              %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
              % if no convergence with the parallel EP
              % start double-loop iterations
              %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
              
              up_mode=gp.latent_opt.up_mode; % update mode in double-loop iterations
              %up_mode='ep'; % choose the moment matching
              %up_mode='grad'; % choose the gradients
              df_lim=gp.latent_opt.df_lim; % step size limit (1 suitable for ep updates)
              
              tol_e=inf;  % the energy difference for measuring convergence (tol_e < tolStop)
              ninner=0;   % counter for the inner loop iterations
              df=df0;     % initial step size (damping factor)
              
              % the intial gradient in the search direction
              g = sum( (mf -m_r).*dnu_q ) +0.5*sum( (V_r +m_r.^2 -Vf -mf.^2).*dtau_q );
              
              sdir_reset=false;
              rec_sadj=[0 e g]; % record for step size adjustment
              for i1=1:maxiter
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % calculate a new proposal state
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                
                % Limit the step size separately for each site so that the cavity variances
                % do not exceed the upper limit (this will change the search direction)
                % this should not happen after step size adjustment
                ii1=tau_s-eta.*(tau_q+df*dtau_q)<tauc_min;
                if any(ii1)
                  %ii1=dtau_q>0; df1=min( ( (tau_s(ii1)-tauc_min(ii1))./eta(ii1)-tau_q(ii1) )./dtau_q(ii1)/df ,1);
                  df1=( (tau_s(ii1)-tauc_min(ii1))./eta(ii1) -tau_q(ii1) )./dtau_q(ii1)/df;
                  
                  dnu_q(ii1)=dnu_q(ii1).*df1;
                  dtau_q(ii1)=dtau_q(ii1).*df1;
                  
                  % the intial gradient in the search direction
                  g = sum( (mf -m_r).*dnu_q ) +0.5*sum( (V_r +m_r.^2 -Vf -mf.^2).*dtau_q );
                  
                  % re-init the step size adjustment record
                  rec_sadj=[0 e g];
                end
                % proposal
                nu_q2=nu_q+df*dnu_q;
                tau_q2=tau_q+df*dtau_q;
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % energy for the proposal state
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                
                % update the q-distribution
                %               [mf2,Sf2,lnZ_q2,L1,L2]=evaluate_q(nu_q2,tau_q2,K,display,K_uu, K_fu, Kv_ff, Qv_ff);
                switch gp.type
                  case 'FULL'
                    [mf2,Sf2,lnZ_q2,L1,L2]=evaluate_q(nu_q2,tau_q2,K,display);
                    Vf2 = diag(Sf2);
                  case 'FIC'
                    [mf2,Vf2,lnZ_q2,L1,L2]=evaluate_q2(nu_q2,tau_q2,Luu, K_fu, Kv_ff, Qv_ff, display);
                  otherwise
                    error('Robust-EP not implemented for this type of GP!');
                end
                
                % check cavity
                pcavity=all( (1./Vf2-eta.*tau_q2 )>=tauc_min);
                
                g2=NaN;
                if isempty(L2)
                  % the q-distribution not defined (the posterior covariance
                  % not positive definite)
                  e2=inf;
                elseif pcavity
                  % the tilted distribution
                  [lnZ_r2,lnZ_i2,m_r2,V_r2]=evaluate_r(nu_q2,tau_q2,eta,fh_tm,nu_s,tau_s,display);
                  
                  % the new energy
                  e2 = lnZ_q2 + lnZ_r2 -lnZ_s;
                  
                  % gradients in the search direction
                  g2 = sum( (mf2 -m_r2).*dnu_q ) +0.5*sum( (V_r2 +m_r2.^2 -Vf2 -mf2.^2).*dtau_q );
                  
                  if ismember(display,{'iter'})
                    % ratio of the gradients
                    fprintf('dg=%6.3f, ',min(abs(g2)/abs(g),99))
                  end
                end
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % check if the energy decreases
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                if ~isfinite(e2) || ( pcavity && g2>10*abs(g) )
                  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                  % ill-conditioned q-distribution or very large increase
                  % in the gradient
                  % => half the step size
                  df=df*0.5;
                  
                  if ismember(display,{'iter'})
                    fprintf('decreasing step size, ')
                  end
                elseif ~pcavity && ~pvis
                  % The cavity distributions resulting from the proposal distribution
                  % are not well defined, reset the site parameters by doing
                  % one parallel update with a zero initialization and continue
                  % with double loop iterations
                  
                  if ismember(display,{'iter'})
                    fprintf('re-init the posterior due to ill-conditioned cavity distributions, ')
                  end
                  
                  % Do resetting only once
                  pvis=1;
                  
                  up_mode='ep';
                  nu_q=zeros(size(y));tau_q=zeros(size(y));
                  mf=zeros(size(y));
                  switch gp.type
                    case 'FULL'
                      Sf=K;Vf=diag(K);
                    case 'FIC'
                      Vf=Cv_ff;
                  end
                  nu_s=mf./Vf;
                  tau_s=1./Vf;
                  %                 lnZ_s=0.5*sum( (-log(tau_s) +nu_s.^2 ./tau_s)./eta ); % minus 0.5*log(2*pi)./eta
                  [lnZ_r,lnZ_i,m_r,V_r]=evaluate_r(nu_q,tau_q,eta,fh_tm,nu_s,tau_s,display);
                  %                 e = lnZ_q + lnZ_r -lnZ_s;
                  [dnu_q,dtau_q]=ep_update_dir(mf,Vf,m_r,V_r,eta,up_mode,tolUpdate);
                  %nu_q=dnu_q; tau_q=dtau_q;
                  nu_q=0.9.*dnu_q; tau_q=0.9.*dtau_q;
                  
                  switch gp.type
                    case 'FULL'
                      [mf,Sf,lnZ_q]=evaluate_q(nu_q,tau_q,K,display);
                      Vf = diag(Sf);
                    case 'FIC'
                      [mf,Vf,lnZ_q]=evaluate_q2(nu_q,tau_q,Luu, K_fu, Kv_ff, Qv_ff, display);
                    otherwise
                      error('Robust-EP not implemented for this type of GP!');
                  end
                  nu_s=mf./Vf; tau_s=1./Vf;
                  lnZ_s=0.5*sum( (-log(tau_s) +nu_s.^2 ./tau_s)./eta ); % minus 0.5*log(2*pi)./eta
                  [lnZ_r,lnZ_i,m_r,V_r]=evaluate_r(nu_q,tau_q,eta,fh_tm,nu_s,tau_s,display);
                  e = lnZ_q + lnZ_r -lnZ_s;
                  [dnu_q,dtau_q]=ep_update_dir(mf,Vf,m_r,V_r,eta,up_mode,tolUpdate);
                  
                  df=0.8;
                  
                  g = sum( (mf -m_r).*dnu_q ) +0.5*sum( (V_r +m_r.^2 -Vf -mf.^2).*dtau_q );
                  rec_sadj=[0 e g];
                  
                elseif size(rec_sadj,1)<=1 && ( e2>e || abs(g2)>abs(g)*tolGrad )
                  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                  % no decrease in energy or the new gradient exceeds the
                  % pre-defined limit
                  % => adjust the step size
                  
                  if ismember(display,{'iter'})
                    fprintf('adjusting step size,  ')
                  end
                  
                  % update the record for step size adjustment
                  ii1=find(df>rec_sadj(:,1),1,'last');
                  ii2=find(df<rec_sadj(:,1),1,'first');
                  rec_sadj=[rec_sadj(1:ii1,:); df e2 g2; rec_sadj(ii2:end,:)];
                  
                  df_new=0;
                  if size(rec_sadj,1)>1
                    if exist('csape','file')==2
                      if g2>0
                        % adjust the step size with spline interpolation
                        pp=csape(rec_sadj(:,1)',[rec_sadj(1,3) rec_sadj(:,2)' rec_sadj(end,3)],[1 1]);
                        [tmp,df_new]=fnmin(pp,[0 df]);
                        
                      elseif isfinite(g2)
                        % extrapolate with Hessian end-conditions
                        H=(rec_sadj(end,3)-rec_sadj(end-1,3))/(rec_sadj(end,1)-rec_sadj(end-1,1));
                        pp=csape(rec_sadj(:,1)',[rec_sadj(1,3) rec_sadj(:,2)' H],[1 2]);
                        % extrapolate at most by 100% at a time
                        [tmp,df_new]=fnmin(pp,[df df*1.5]);
                      end
                    else
                      % if curvefit toolbox does not exist, use a simple Hessian
                      % approximation
                      [tmp,ind]=sort(rec_sadj(:,2),'ascend');
                      ind=ind(1:2);
                      
                      H=(rec_sadj(ind(1),3)-rec_sadj(ind(2),3))/(rec_sadj(ind(1),1)-rec_sadj(ind(2),1));
                      df_new=rec_sadj(ind(1),1) -rec_sadj(ind(1),3)/H;
                      if g2>0
                        % interpolate
                        df_new=max(min(df_new,df),0);
                      else
                        % extrapolate at most 100%
                        df_new=max(min(df_new,1.5*df),df);
                      end
                    end
                    df_new=min(df_new,df_lim);
                  end
                  
                  if df_new==0
                    % the spline approxmation fails or no record of the previous gradients
                    if g2>0
                      df=df*0.9; % too long step since the gradient is positive
                    else
                      df=df*1.1; % too short step since the gradient is negative
                    end
                  else
                    df=df_new;
                  end
                  % prevent too small cavity-variances after the step-size adjustment
                  ii1=dtau_q>0;
                  if any(ii1)
                    df_max=min( ( (tau_s(ii1)-tauc_min(ii1)-1e-8)./eta(ii1) -tau_q(ii1) )./dtau_q(ii1) );
                    df=min(df,df_max);
                  end
                  
                elseif e2>e+tolInner || (abs(g2)>abs(g)*tolGrad && strcmp(up_mode,'ep'))
                  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                  % No decrease in energy despite the step size adjustments.
                  % In some difficult cases the EP search direction may not
                  % result in decrease of the energy or the gradient
                  % despite of the step size adjustment. One reason for this
                  % may be the parallel EP search direction
                  % => try the negative gradient as the search direction
                  %
                  % or if the problem persists
                  % => try resetting the search direction
                  
                  if abs(g2)>abs(g)*tolGrad && strcmp(up_mode,'ep')
                    % try switching to gradient based updates
                    up_mode='grad';
                    df_lim=1e3;
                    df=0.1;
                    if ismember(display,{'iter'})
                      fprintf('switch to gradient updates, ')
                    end
                  elseif ~sdir_reset
                    if ismember(display,{'iter'})
                      fprintf('reset the search direction, ')
                    end
                    sdir_reset=true;
                  elseif g2<0 && abs(g2)<abs(g) && e2>e
                    if ismember(display,{'final','iter'})
                      fprintf('Unable to continue: gradients of the inner-loop objective are inconsistent\n')
                    end
                    break;
                  else
                    df=df*0.1;
                  end
                  
                  % the new search direction
                  [dnu_q,dtau_q]=ep_update_dir(mf,Vf,m_r,V_r,eta,up_mode,tolUpdate);
                  
                  % the initial gradient in the search direction
                  g = sum( (mf -m_r).*dnu_q ) +0.5*sum( (V_r +m_r.^2 -Vf -mf.^2).*dtau_q );
                  
                  % re-init the step size adjustment record
                  rec_sadj=[0 e g];
                else
                  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                  % decrease of energy => accept the new state
                  
                  dInner=abs(e-e2); % the inner loop energy change
                  
                  % accept the new site parameters (nu_q,tau_q)
                  [mf,Vf,Sf,nu_q,tau_q,lnZ_q]=deal(mf2,Vf2,Sf2,nu_q2,tau_q2,lnZ_q2);
                  
                  % accept also the new tilted distributions
                  [lnZ_r,lnZ_i,m_r,V_r,e]=deal(lnZ_r2,lnZ_i2,m_r2,V_r2,e2);
                  
                  % check that the new cavity variances are positive and not too large
                  tau_s2=1./Vf;
                  pcavity=all( (tau_s2-eta.*tau_q )>=tauc_min);
                  supdate=false;
                  if pcavity && (dInner<tolInner || ninner>=max_ninner)
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    % try to update the surrogate distribution on the condition that
                    % - the cavity variances are positive and not too large
                    % - the new tilted moments are proper
                    % - sufficient tolerance or the maximum number of inner
                    %   loop updates is exceeded
                    
                    % update the surrogate distribution
                    nu_s2=mf.*tau_s2;
                    lnZ_s2=0.5*sum( (-log(tau_s2) +nu_s2.^2 ./tau_s2)./eta );
                    
                    % update the tilted distribution
                    [lnZ_r2,lnZ_i2,m_r2,V_r2]=evaluate_r(nu_q,tau_q,eta,fh_tm,nu_s2,tau_s2,display);
                    
                    % evaluate the new energy
                    e2 = lnZ_q + lnZ_r2 -lnZ_s2;
                    
                    if isfinite(e2)
                      % a successful surrogate update
                      supdate=true;
                      ninner=0; % reset the inner loop iteration counter
                      
                      % update the convergence criteria
                      tol_e=abs(e2-e);
                      
                      % accept the new state
                      [lnZ_r,lnZ_i,m_r,V_r,lnZ_s,nu_s,tau_s,e]=deal(lnZ_r2,lnZ_i2,m_r2,V_r2,lnZ_s2,nu_s2,tau_s2,e2);
                      
                      if ismember(display,{'iter'})
                        fprintf('surrogate update,     ')
                      end
                    else
                      % Improper tilted moments even though the cavity variances are
                      % positive. This is an indication of numerically unstable
                      % tilted moment integrations but fractional updates usually help
                      % => try switching to fractional updates
                      pcavity=false;
                      
                      if ismember(display,{'iter'})
                        fprintf('surrogate update failed, ')
                      end
                    end
                  end
                  
                  if all(eta==eta1) && ~pcavity && (dInner<tolInner || ninner>=max_ninner)
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    % If the inner loop moments (within tolerance) are matched
                    % but the new cavity variances are negative or the tilted moment
                    % integrations fail after the surrogate update
                    % => switch to fractional EP.
                    %
                    % This is a rare situation and most likely the
                    % hyperparameters are such that the approximating family
                    % is not flexible enough, i.e., the hyperparameters are
                    % unsuitable for the data.
                    %
                    % One can also try to reduce the lower limit for the
                    % cavity precisions tauc_min=1./(Vc_lim*diag(K)), i.e.
                    % increase the maximum cavity variance Vc_lim.
                    
                    % try switching to fractional updates
                    eta=repmat(eta2,n,1);
                    
                    % correct the surrogate normalization accordingly
                    % the surrogate distribution is not updated
                    lnZ_s2=0.5*sum( (-log(tau_s) +nu_s.^2 ./tau_s)./eta );
                    
                    % update the tilted distribution
                    [lnZ_r2,lnZ_i2,m_r2,V_r2]=evaluate_r(nu_q,tau_q,eta,fh_tm,nu_s,tau_s,display);
                    
                    % evaluate the new energy
                    e2 = lnZ_q + lnZ_r2 -lnZ_s2;
                    
                    if isfinite(e2)
                      % successful switch to fractional energy
                      supdate=true;
                      pcavity=true;
                      ninner=0; % reset the inner loop iteration counter
                      
                      % accept the new state
                      [lnZ_r,lnZ_i,m_r,V_r,lnZ_s,e]=deal(lnZ_r2,lnZ_i2,m_r2,V_r2,lnZ_s2,e2);
                      
                      % start with ep search direction
                      up_mode='ep';
                      df_lim=0.9;
                      df=0.1;
                      if ismember(display,{'iter'})
                        fprintf('switching to fractional EP, ')
                      end
                    else
                      % Improper tilted moments even with fractional updates
                      % This is very unlikely to happen because decreasing the
                      % fraction parameter (eta2<eta1) stabilizes the
                      % tilted moment integrations
                      
                      % revert back to the previous fraction parameter
                      eta=repmat(eta1,n,1);
                      
                      if ismember(display,{'final','iter'})
                        fprintf('Unable to switch to the fractional EP, check that eta2<eta1\n')
                      end
                      break;
                    end
                  end
                  
                  if all(eta==eta2) && ~pcavity && (dInner<tolInner || ninner>=10)
                    % Surrogate updates do not result into positive cavity variances
                    % even with fractional updates with eta2 => terminate iterations
                    if ismember(display,{'final','iter'})
                      fprintf('surrogate update failed with fractional updates, try decreasing eta2\n')
                    end
                    break
                  end
                  
                  if ~supdate
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    % no successful surrogate update, no sufficient tolerance,
                    % or the maximum number of inner loop updates is not yet exceeded
                    % => continue with the same surrogate distribution
                    
                    ninner=ninner+1; % increase inner loop iteration counter
                    if ismember(display,{'iter'})
                      fprintf('inner-loop update,    ')
                    end
                  end
                  
                  % the new search direction
                  [dnu_q,dtau_q]=ep_update_dir(mf,Vf,m_r,V_r,eta,up_mode,tolUpdate);
                  
                  % the initial gradient in the search direction
                  g = sum( (mf -m_r).*dnu_q ) +0.5*sum( (V_r +m_r.^2 -Vf -mf.^2).*dtau_q );
                  
                  % re-init step size adjustment record
                  rec_sadj=[0 e g];
                end
                
                if ismember(display,{'iter'})
                  % maximum difference of the marginal moments
                  tol_m=[max(abs(mf-m_r)) max(abs(Vf-V_r))];
                  fprintf('%d, e=%.6f, dm=%.4f, dV=%.4f, df=%6f, eta=%.2f\n',i1,e,tol_m(1),tol_m(2),df,eta(1))
                end
                
                %%%%%%%%%%%%%%%%%%%%%%%
                % check for convergence
                convergence = tol_e<=tolStop;
                if convergence
                  if ismember(display,{'final','iter'})
                    % maximum difference of the marginal moments
                    tol_m=[max(abs(mf-m_r)) max(abs(Vf-V_r))];
                    fprintf('Convergence, iter %d, e=%.6f, dm=%.4f, dV=%.4f, df=%6f, eta=%.2f\n',i1,e,tol_m(1),tol_m(2),df,eta(1))
                  end
                  break
                end
              end % end of the double-loop updates
            end
            
            % the current energy is not finite or no convergence
            if ~isfinite(e)
              fprintf('GPEP_E: Initial energy not defined, check the hyperparameters\n')
            elseif ~convergence
              fprintf('GPEP_E: No convergence, %d iter, e=%.6f, dm=%.4f, dV=%.4f, df=%6f, eta=%.2f\n',i1,e,tol_m(1),tol_m(2),df,eta(1))
              fprintf('GPEP_E: Check the hyperparameters, increase maxiter and/or max_ninner, or decrease tolInner\n')
            end
            edata=-e; % the data contribution to the marginal posterior density
            
            % =====================================================================================
            % Evaluate the prior contribution to the error from covariance functions and likelihood
            % =====================================================================================
            
            % Evaluate the prior contribution to the error from covariance functions
            eprior = 0;
            for i=1:ncf
              gpcf = gp.cf{i};
              eprior = eprior - gpcf.fh.lp(gpcf);
              %         eprior = eprior - feval(gpcf.fh.lp, gpcf, x, y);
            end
            
            % Evaluate the prior contribution to the error from likelihood functions
            if isfield(gp, 'lik') && isfield(gp.lik, 'p')
              likelih = gp.lik;
              eprior = eprior - likelih.fh.lp(likelih);
            end
            
            % Evaluate the prior contribution to the error from the inducing inputs
            if ~isempty(strfind(gp.infer_params, 'inducing'))
              if isfield(gp, 'p') && isfield(gp.p, 'X_u') && ~isempty(gp.p.X_u)
                if iscell(gp.p.X_u) % Own prior for each inducing input
                  for i = 1:size(gp.X_u,1)
                    pr = gp.p.X_u{i};
                    eprior = eprior - pr.fh.lp(gp.X_u(i,:), pr);
                  end
                else
                  eprior = eprior - gp.p.X_u.fh.lp(gp.X_u(:), gp.p.X_u);
                end
              end
            end
            
            % the total energy
            e = edata + eprior;
            
            sigm2vec_i = 1./(tau_s-eta.*tau_q);     % vector of cavity variances
            muvec_i = (nu_s-eta.*nu_q).*sigm2vec_i; % vector of cavity means
            logZ_i = lnZ_i; % vector of tilted normalization factors
            
            
            % check that the posterior covariance is positive definite and
            % calculate its Cholesky decomposition
            switch gp.type
              case 'FULL'
                [L, notpositivedefinite] = chol(Sf);
                b = [];
                La2 = [];
                if notpositivedefinite || ~isfinite(e)
                  [e, edata, eprior, param, ch] = set_output_for_notpositivedefinite();
                  return
                end
              case 'FIC'
                La2 = Luu;
                L = L2;
                b = Kv_ff - Qv_ff;
            end            
            nutilde = nu_q;
            tautilde = tau_q;
            
          otherwise
            error('Unknown optim method!');
        end
             
        if exist('joint_mean_magnitude','var') && joint_mean_magnitude
          param.mf=mf;
          param.Sigm=Sigm;
          param.C=C;
          La2=1;
          b=1;
          eta=1;
          logZ_i=logM0;
        else
          if (isfield(gp.lik, 'int_likparam') && gp.lik.int_likparam) || ...
              (isfield(gp.lik, 'int_magnitude') && gp.lik.int_magnitude)
            [L, notpositivedefinite]=chol(Sigm);
            if notpositivedefinite || ~isfinite(e)
              [e, edata, eprior, param, ch] = set_output_for_notpositivedefinite();
              return
            end
          end
          if (isfield(gp.lik, 'int_likparam') && gp.lik.int_likparam)
            [La2, notpositivedefinite]=chol(Sigm2);
            if notpositivedefinite || ~isfinite(e)
              [e, edata, eprior, param, ch] = set_output_for_notpositivedefinite();
              return
            end
            if ~gp.lik.inputparam
              param.mf2=mf(1,2);
            else
              param.mf2=mf(:,2);
            end
          end
          if isfield(gp.lik, 'int_magnitude') && gp.lik.int_magnitude
            if ~inputmagnitude
              param.mf3=mf(1,ns);
              param.La3=sqrt(Sigm3);
            else
              param.mf3=mf(:,ns);
              [param.La3, notpositivedefinite]=chol(Sigm3);
              if notpositivedefinite || ~isfinite(e)
                [e, edata, eprior, param, ch] = set_output_for_notpositivedefinite();
                return
              end
            end
          end
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

function [m_q,S_q,lnZ_q,L1,L2]=evaluate_q(nu_q,tau_q,K,display)

% function for determining the parameters of the q-distribution
% when site variances tau_q may be negative
%
% q(f) = N(f|0,K)*exp( -0.5*f'*diag(tau_q)*f + nu_q'*f )/Z_q = N(f|m_q,S_q)
%
% S_q = inv(inv(K)+diag(tau_q))
% m_q = S_q*nu_q;
%
% det(eye(n)+K*diag(tau_q))) = det(L1)^2 * det(L2)^2
% where L1 and L2 are upper triangular
%
% see Expectation consistent approximate inference (Opper & Winther, 2005)

n=length(nu_q);
ii1=find(tau_q>0); n1=length(ii1); W1=sqrt(tau_q(ii1));
ii2=find(tau_q<0); n2=length(ii2); W2=sqrt(abs(tau_q(ii2)));

L=zeros(n);
S_q=K;
if ~isempty(ii1)
  % Cholesky decomposition for the positive sites
  L1=(W1*W1').*K(ii1,ii1);
  L1(1:n1+1:end)=L1(1:n1+1:end)+1;
  [L1, notpositivedefinite]=chol(L1);
  if notpositivedefinite
    L1=[];L2=[];lnZ_q=NaN;m_q=NaN;S_q=NaN;
    return
  end
  
  L(:,ii1) = bsxfun(@times,K(:,ii1),W1')/L1;
  
  S_q=S_q-L(:,ii1)*L(:,ii1)';
else
  L1=1;
end

if ~isempty(ii2)
  % Cholesky decomposition for the negative sites
  V=bsxfun(@times,K(ii2,ii1),W1')/L1;
  L2=(W2*W2').*(V*V'-K(ii2,ii2));
  L2(1:n2+1:end)=L2(1:n2+1:end)+1;
  
  [L2,pd]=chol(L2);
  if pd==0
    L(:,ii2)=bsxfun(@times,K(:,ii2),W2')/L2 -L(:,ii1)*(bsxfun(@times,V,W2)'/L2);
    S_q=S_q+L(:,ii2)*L(:,ii2)';
  else
    L2=[];
    if ismember(display,{'iter'})
      fprintf('Negative definite q-distribution.\n')
    end
  end
  
else
  L2=1;
end
%V_q=diag(S_q);
m_q=S_q*nu_q;

% log normalization
lnZ_q = -sum(log(diag(L1))) -sum(log(diag(L2))) +0.5*sum(m_q.*nu_q);

end

function [m_q,S_q,lnZ_q,L1,L2]=evaluate_q2(nu_q,tau_q,LK_uu, K_fu, Kv_ff, Qv_ff, display)

% function for determining the parameters of the q-distribution
% when site variances tau_q may be negative
%
% q(f) = N(f|0,K)*exp( -0.5*f'*diag(tau_q)*f + nu_q'*f )/Z_q = N(f|m_q,S_q)
%
% S_q = inv(inv(K)+diag(tau_q)) where K is sparse approximation for prior
%       covariance
% m_q = S_q*nu_q;
%
% det(eye(n)+K*diag(tau_q))) = det(L1)^2 * det(L2)^2
% where L1 and L2 are upper triangular
%
% see Expectation consistent approximate inference (Opper & Winther, 2005)

n=length(nu_q);

S_q = Kv_ff;
m_q = nu_q;
D = Kv_ff - Qv_ff;
L1 = sqrt(1 + D.*tau_q);
L = [];
if any(~isreal(L1))
  if ismember(display,{'iter'})
    fprintf('Negative definite q-distribution.\n')
  end
else
  U = K_fu;
  WDtilde = tau_q./(1+tau_q.*D);
  
  % Evaluate diagonal of S_q
  
  ii1=find(WDtilde>0); n1=length(ii1); W1=sqrt(WDtilde(ii1)); % WS^-1
  ii2=find(WDtilde<0); n2=length(ii2); W2=sqrt(abs(WDtilde(ii2))); % WS^-1
  if ~isempty(ii2) || ~isempty(ii1)
    if ~isempty(ii1)
      UWS(:,ii1) = bsxfun(@times, U(ii1,:)', W1');
    end
    
    if ~isempty(ii2)
      UWS(:,ii2) = bsxfun(@times, U(ii2,:)', W2');
    end
    [L, p] = chol(LK_uu*LK_uu' + UWS(:,ii1)*UWS(:,ii1)' - UWS(:,ii2)*UWS(:,ii2)', 'lower');
    if p~=0
      L=[];
      if ismember(display,{'iter'})
        fprintf('Negative definite q-distribution.\n')
      end
    else
      
      S = 1 + D.*tau_q;
      %               S_q = diag(D./S) + diag(1./S)*U*inv(L*L')*U'*diag(1./S);
      S_q = D./S + sum((bsxfun(@times, 1./S, U)/L').^2,2);
      m_q = D.*nu_q./S + (U*(L'\(L\(U'*(nu_q./S)))))./S;
    end
  else
  end
  %   end
  
end

% log normalization
L2 = L;
lnZ_q = -0.5*sum(log(L1.^2)) - sum(log(diag(L))) + sum(log(diag(LK_uu))) +0.5*sum(m_q.*nu_q);

end

function [lnZ_r,lnZ_i,m_r,V_r,p]=evaluate_r(nu_q,tau_q,eta,fh_tm,nu_s,tau_s,display)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function for determining the parameters of the r-distribution
% (the product of the tilted distributions)
%
% r(f) = exp(-lnZ_r) * prod_i p(y(i)|f(i)) * exp( -0.5*f(i)^2 tau_r(i) + nu_r(i)*f(i) )
%      ~ prod_i N(f(i)|m_r(i),V_r(i))
%
% tau_r = tau_s - tau_q
% nu_r = nu_s - nu_q
%
% lnZ_i(i) = log int p(y(i)|f(i)) * N(f(i)|nu_r(i)/tau_r(i),1/tau_r(i)) df(i)
%
% see Expectation consistent approximate inference (Opper & Winther, 2005)

n=length(nu_q);
[lnZ_i,m_r,V_r,nu_r,tau_r]=deal(zeros(n,1));
p=false(n,1);
for si=1:n
  % cavity distribution
  tau_r_si=tau_s(si)-eta(si)*tau_q(si);
  if tau_r_si<=0
    %     if ismember(display,{'iter'})
    %       %fprintf('Negative cavity precision at site %d\n',si)
    %     end
    continue
  end
  nu_r_si=nu_s(si)-eta(si)*nu_q(si);
  
  % tilted moments
  [lnZ_si,m_r_si,V_r_si] = fh_tm(si, nu_r_si/tau_r_si, 1/tau_r_si, eta(si));
  
  if ~isfinite(lnZ_si) || V_r_si<=0
    %     if ismember(display,{'iter'})
    %       fprintf('Improper normalization or tilted variance at site %d\n',si)
    %     end
    continue
  end
  
  % store the new parameters
  [nu_r(si),tau_r(si),lnZ_i(si),m_r(si),V_r(si)]=deal(nu_r_si,tau_r_si,lnZ_si,m_r_si,V_r_si);
  
  p(si)=true;
end

lnZ_r=sum(lnZ_i./eta) +0.5*sum((-log(tau_r) +nu_r.^2 ./tau_r)./eta);
end

function [dnu_q,dtau_q]=ep_update_dir(m_q,V_q,m_r,V_r,eta,up_mode,tolUpdate)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% update direction for double-loop EP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% V_q=diag(S_q);
switch up_mode
  case 'ep'
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % site updates by moment matching
    
    [dnu_q,dtau_q]=deal(zeros(size(m_q)));
    
    %ind_up=V_r>0 & max(abs(V_r-V_q),abs(m_r-m_q))>tolUpdate;
    ind_up=V_r>0 & (abs(V_r-V_q) > tolUpdate*abs(V_q) | abs(m_r-m_q) > tolUpdate*abs(m_q));
    
    dnu_q(ind_up) = ( m_r(ind_up)./V_r(ind_up) - m_q(ind_up)./V_q(ind_up) ) ./ eta(ind_up);
    dtau_q(ind_up) = ( 1./V_r(ind_up) - 1./V_q(ind_up) )./ eta(ind_up);
    
  case 'grad'
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % gradient descend
    % Not used at the moment!
    
    % evaluate the gradients wrt nu_q and tau_q
    gnu_q = m_q - m_r;
    gtau_q = 0.5*(V_r + m_r.^2 - V_q - m_q.^2);
    
    % the search direction
    dnu_q=-gnu_q;
    dtau_q=-gtau_q;
end

end
