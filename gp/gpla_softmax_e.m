function [e, edata, eprior, f, L, a, E, M, p] = gpla_softmax_e(w, gp, varargin)
%GPLA_SOFTMAX_E  Do Laplace approximation and return marginal log posterior
%                estimate for softmax likelihood
%
%  Description
%    E = GPLA_SOFTMAX_E(W, GP, X, Y, OPTIONS) takes a GP data
%    structure GP together with a matrix X of input vectors and a
%    matrix Y of target vectors, and finds the Laplace
%    approximation for the conditional posterior p(Y | X, th),
%    where th is the hyperparameters. Returns the energy at th (see
%    below). Each row of X corresponds to one input vector and each
%    row of Y corresponds to one target vector.
%
%    [E, EDATA, EPRIOR] = GPLA_SOFTMAX_E(W, GP, X, Y, OPTIONS)
%    returns also the data and prior components of the total
%    energy.
%
%    The energy is minus log posterior cost function for th:
%      E = EDATA + EPRIOR 
%        = - log p(Y|X, th) - log p(th),
%      where th represents the hyperparameters (lengthScale,
%      magnSigma2...), X is inputs and Y is observations.
%
%    OPTIONS is optional parameter-value pair
%      z - optional observed quantity in triplet (x_i,y_i,z_i)
%          Some likelihoods may use this. For example, in case of
%          Poisson likelihood we have z_i=E_i, that is, expected
%          value for ith case.
%
%  See also
%    GP_SET, GP_E, GPLA_SOFTMAX_G, LA_SOFTMAX_PRED

%  Description 2
%    Additional properties meant only for internal use.
%  
%    GP = GPLA_SOFTMAX_E('init', GP, X, Y, OPTIONS) takes a GP data
%    structure GP together with a matrix X of input vectors and a
%    matrix Y of target vectors, and initializes required fiels for
%    the Laplace approximation.
%
%    [e, edata, eprior, f, L, a, La2, p]
%       = gpla_softmax_e(w, gp, x, y, varargin)
%    returns many useful quantities produced by EP algorithm.
%
% The Newton's method is implemented as described in
% Rasmussen and Williams (2006).

% Copyright (c) 2010 Jaakko Riihimäki, Pasi Jylänki, 
%                    Jarno Vanhatalo, Aki Vehtari

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.

  ip=inputParser;
  ip.FunctionName = 'GPLA_SOFTMAX_E';
  ip.addRequired('w', @(x) ...
                 (ischar(x) && strcmp(w, 'init')) || ...
                 isvector(x) && isreal(x) && all(isfinite(x)));
  ip.addRequired('gp',@isstruct);
  ip.addOptional('x', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
  ip.addOptional('y', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
  ip.addParamValue('z', [], @(x) isreal(x) && all(isfinite(x(:))))
  ip.parse(w, gp, varargin{:});
  x=ip.Results.x;
  y=ip.Results.y;
  z=ip.Results.z;
  
  if strcmp(w, 'init')
    % intialize saved values
    w0 = NaN;
    e0=[];
    edata0= inf;
    eprior0=[];
    W = zeros(size(y(:)));
    W0 = W;
    f0 = zeros(size(y(:)));
    L0 = [];
    E0=[];
    M0=[];
    
    f = zeros(size(y(:)));
    n0 = size(x,1);
    %La20 = [];
    a0 = 0;
    p0 = [];
    datahash0=0;

    % return function handle to the nested function ep_algorithm
    % this way each gp has its own peristent memory for EP
    gp.fh.e = @laplace_algorithm;
    e = gp;
  else
    % call laplace_algorithm using the function handle to the nested function
    % this way each gp has its own peristent memory for Laplace
    [e, edata, eprior, f, L, a, E, M, p] = feval(gp.fh.e, w, gp, x, y, z);
  end

  function [e, edata, eprior, f, L, a, E, M, p] = laplace_algorithm(w, gp, x, y, z)
  % code for the Laplace algorithm
    
  % check whether saved values can be used
    datahash=hash_sha512([x y]);
    if all(size(w)==size(w0)) && all(abs(w-w0)<1e-8) && isequal(datahash,datahash0)
      % The covariance function parameters or data haven't changed
      % so we can return the energy and the site parameters that are saved
      e = e0;
      edata = edata0;
      eprior = eprior0;
      f = f0;
      L = L0;
      %La2 = La20;
      E = E0;
      M = M0;
      W = W0;
      a = a0;
      p = p0;
    else
      % The hyperparameters or data have changed since
      % the last call for gpla_e. In this case we need to
      % re-evaluate the Laplace approximation
      gp=gp_unpak(gp, w);
      ncf = length(gp.cf);
      n = length(x);
      p = [];

      % Initialize latent values
      % zero seems to be a robust choice (Jarno)
      f = zeros(size(y(:)));

      % =================================================
      % First Evaluate the data contribution to the error
      switch gp.type
        % ============================================================
        % FULL
        % ============================================================
        case 'FULL'
          K = gp_trcov(gp, x);

          % If K is sparse, permute all the inputs so that evaluations are more efficient
          if issparse(K)         % Check if compact support covariance is used
            p = analyze(K);
            y = y(p);
            K = K(p,p);
            if ~isempty(z)
              z = z(p,:);
            end
            LD = ldlchol(K);
          else
            % LD = chol(K);
          end
          
          switch gp.latent_opt.optim_method
            % --------------------------------------------------------------------------------
            % find the posterior mode of latent variables by fminunc large scale method
            %                   case 'fminunc_large'
            %                     if ~isfield(gp.latent_opt, 'fminunc_opt')
            %                         opt=optimset('GradObj','on');
            %                         opt=optimset(opt,'Hessian','on');
            %                         if issparse(K)
            %                             fhm = @(W, f, varargin) (ldlsolve(LD,f) + repmat(W,1,size(f,2)).*f);  % W*f; %
            %                         else
            %                             fhm = @(W, f, varargin) (LD\(LD'\f) + repmat(W,1,size(f,2)).*f);  % W*f; %
            %                         end                            
            %                         opt=optimset(opt,'HessMult', fhm);
            %                         opt=optimset(opt,'TolX', 1e-12);
            %                         opt=optimset(opt,'TolFun', 1e-12);
            %                         opt=optimset(opt,'LargeScale', 'on');
            %                         opt=optimset(opt,'Display', 'off'); % 'iter'
            %                     else
            %                         opt = gp.latent_opt.fminunc_opt;
            %                     end
            %                
            %                     if issparse(K)
            %                         fe = @(f, varargin) (0.5*f*(ldlsolve(LD,f')) - feval(gp.lik.fh.ll, gp.lik, y, f', z));
            %                         fg = @(f, varargin) (ldlsolve(LD,f') - feval(gp.lik.fh.llg, gp.lik, y, f', 'latent', z))';
            %                         fh = @(f, varargin) (-feval(gp.lik.fh.llg2, gp.lik, y, f', 'latent', z)); %inv(K) + diag(g2(f', gp.lik)) ; %
            %                     else
            %                         fe = @(f, varargin) (0.5*f*(LD\(LD'\f')) - feval(gp.lik.fh.ll, gp.lik, y, f', z));
            %                         fg = @(f, varargin) (LD\(LD'\f') - feval(gp.lik.fh.llg, gp.lik, y, f', 'latent', z))';
            %                         fh = @(f, varargin) (-feval(gp.lik.fh.llg2, gp.lik, y, f', 'latent', z)); %inv(K) + diag(g2(f', gp.lik)) ; %
            %                     end
            %                     
            %                     mydeal = @(varargin)varargin{1:nargout};
            %                     [f,fval,exitflag,output] = fminunc(@(ww) mydeal(fe(ww), fg(ww), fh(ww)), f', opt);
            %                     f = f';
            % 
            %                     if issparse(K)
            %                         a = ldlsolve(LD,f);
            %                     else
            %                         a = LD\(LD'\f);
            %                     end
            % --------------------------------------------------------------------------------
            % find the posterior mode of latent variables by Newton method
            case 'newton'
              tol = 1e-12;
              a = f;
              
              nout=size(y,2);
              f2=reshape(f,n,nout);
              
              %W = -feval(gp.lik.fh.llg2, gp.lik, y, f2, 'latent', z);
              %dlp = feval(gp.lik.fh.llg, gp.lik, y, f2, 'latent', z);
              %lp_new = feval(gp.lik.fh.ll, gp.lik, y, f2, z);
              
              lp_new = feval(gp.lik.fh.ll, gp.lik, y, f2, z);
              lp_old = -Inf;
              
              Kbb=zeros(n*nout,1);
              c=zeros(n*nout,1);
              ERMMRc=zeros(n*nout,1);
              pipif=zeros(n*nout,1);
              E=zeros(n,n,nout);
              L=zeros(n,n,nout);
              
              while lp_new - lp_old > tol
                lp_old = lp_new; a_old = a; 
                
                % softmax:
                expf2 = exp(f2);
                pi2 = expf2./(sum(expf2, 2)*ones(1,nout));
                pi_vec=pi2(:);
                
                E_tmp=zeros(n);
                
                for i1=1:nout
                  Dc=sqrt(pi2(:,i1));
                  Lc=(Dc*Dc').*K;
                  Lc(1:n+1:end)=Lc(1:n+1:end)+1;
                  Lc=chol(Lc);
                  L(:,:,i1)=Lc;
                  
                  Ec=Lc'\diag(Dc);
                  Ec=Ec'*Ec;
                  E_tmp=E_tmp+Ec;
                  E(:,:,i1)=Ec;
                end
                
                M=chol(E_tmp);
                
                pif=sum(pi2.*f2,2);
                
                for i1=1:nout
                  pipif((1:n)+(i1-1)*n)=pi2(:,i1).*pif;
                end
                
                b = pi_vec.*f-pipif+y(:)-pi_vec;
                % b = -feval(gp.lik.fh.llg2, gp.lik, y, f2, 'latent', z)*f + ...
                %       feval(gp.lik.fh.llg, gp.lik, y, f2, 'latent', z);
                
                for i1=1:nout
                  Kbb((1:n)+(i1-1)*n)=K*b((1:n)+(i1-1)*n);
                  c((1:n)+(i1-1)*n)=E(:,:,i1)*Kbb((1:n)+(i1-1)*n);
                end
                
                Rc=sum(reshape(c, n, nout),2);
                MMRc=M\(M'\Rc);
                for i1=1:nout
                  ERMMRc((1:n)+(i1-1)*n) = E(:,:,i1)*MMRc;
                end
                a=b-c+ERMMRc;
                
                for i1=1:nout
                  f((1:n)+(i1-1)*n)=K*a((1:n)+(i1-1)*n);
                end
                f2=reshape(f,n,nout);
                
                lp_new = -a'*f/2 + feval(gp.lik.fh.ll, gp.lik, y, f2, z);
                
                i = 0;
                while i < 10 && lp_new < lp_old  || isnan(sum(f))
                  % reduce step size by half
                  a = (a_old+a)/2;                                  
                  
                  for i1=1:nout
                    f((1:n)+(i1-1)*n)=K*a((1:n)+(i1-1)*n);
                  end
                  f2=reshape(f,n,nout);
                  
                  lp_new = -a'*f/2 + feval(gp.lik.fh.ll, gp.lik, y, f2, z);
                  i = i+1;
                end 
              end
              % --------------------------------------------------------------------------------
              % find the posterior mode of latent variables by stabilized Newton method.
              % This is implemented as suggested by Hannes Nickisch (personal communication)
            case 'stabilized-newton'
              %                     % Gaussian initialization
              %                     %   sigma=gp.lik.sigma;
              %                     %   W = ones(n,1)./sigma.^2;
              %                     %   sW = sqrt(W);
              %                     %   %B = eye(n) + siV*siV'.*K;
              %                     %   L=bsxfun(@times,bsxfun(@times,sW,K),sW');
              %                     %   L(1:n+1:end)=L(1:n+1:end)+1;
              %                     %   L = chol(L,'lower');
              %                     %   a=sW.*(L'\(L\(sW.*y)));
              %                     %   f = K*a;
              %                     
              %                     % initialize to observations
              %                     %f=y;
              %                     
              %                     nu=gp.lik.nu;
              %                     sigma2=gp.lik.sigma2;
              %                     Wmax=(nu+1)/nu/sigma2;
              %                     Wlim=0;
              %                     
              %                     tol = 1e-10;
              %                     W = -feval(gp.lik.fh.llg2, gp.lik, y, f, 'latent');
              %                     dlp = feval(gp.lik.fh.llg, gp.lik, y, f, 'latent');
              %                     lp = -(f'*(K\f))/2 +feval(gp.lik.fh.ll, gp.lik, y, f);
              %                     lp_old = -Inf;
              %                     f_old = f+1;
              %                     ge = Inf; %max(abs(a-dlp));
              %                     
              %                     i1=0;
              %                     % begin Newton's iterations
              %                     while lp - lp_old > tol || max(abs(f-f_old)) > tol
              %                       i1=i1+1;
              %                       
              %                       W = -feval(gp.lik.fh.llg2, gp.lik, y, f, 'latent');
              %                       dlp = feval(gp.lik.fh.llg, gp.lik, y, f, 'latent');
              %                       
              %                       W(W<Wlim)=Wlim;
              %                       sW = sqrt(W);
              %                       if issparse(K)
              %                         sW = sparse(1:n, 1:n, sW, n, n);
              %                         L = ldlchol( speye(n)+sW*K*sW );
              %                       else
              %                         %L = chol(eye(n)+sW*sW'.*K); % L'*L=B=eye(n)+sW*K*sW
              %                         L=bsxfun(@times,bsxfun(@times,sW,K),sW');
              %                         L(1:n+1:end)=L(1:n+1:end)+1;
              %                         L = chol(L);
              %                       end
              %                       %L = chol(eye(n)+sW*sW'.*K); % L'*L=B=eye(n)+sW*K*sW
              %                       b = W.*f+dlp;
              %                         if issparse(K)
              %                             a = b - sW*ldlsolve(L,sW*(K*b));
              %                         else
              %                             a = b - sW.*(L\(L'\(sW.*(K*b))));
              %                         end
              %                       
              %                       f_new = K*a;
              %                       lp_new = -(a'*f_new)/2 + feval(gp.lik.fh.ll, gp.lik, y, f_new);
              %                       ge_new=max(abs(a-dlp));
              %                       
              %                       d=lp_new-lp;
              %                       if (d<-1e-6 || (abs(d)<1e-6 && ge_new>ge) )  && Wlim<Wmax*0.5
              %                         %fprintf('%3d, p(f)=%.12f, max|a-g|=%.12f, %.3f \n',i1,lp,ge,Wlim)
              %                         Wlim=Wlim+Wmax*0.05; %Wmax*0.01
              %                       else
              %                         Wlim=0;
              %                         
              %                         ge=ge_new;
              %                         lp_old = lp;
              %                         lp = lp_new;
              %                         f_old = f;
              %                         f = f_new;
              %                         %fprintf('%3d, p(f)=%.12f, max|a-g|=%.12f, %.3f \n',i1,lp,ge,Wlim)
              %                         
              %                       end
              %                       
              %                       if Wlim>Wmax*0.5 || i1>5000
              %                         %fprintf('\n%3d, p(f)=%.12f, max|a-g|=%.12f, %.3f \n',i1,lp,ge,Wlim)
              %                         break
              %                       end
              %                       
              %                     end
              % --------------------------------------------------------------------------------
              % find the posterior mode of latent variables with likelihood specific algorithm
              % For example, with Student-t likelihood this mean EM-algorithm which is coded in the
              % likelih_t file.
            case 'lik_specific'
              [f, a] = feval(gp.lik.fh.optimizef, gp, y, K);
            otherwise 
              error('gpla_e: Unknown optimization method ! ')
          end
          
          % evaluate the approximate log marginal likelihood
          
          expf2 = exp(f2);
          pi2 = expf2./(sum(expf2, 2)*ones(1,nout));
          pi_vec=pi2(:);
          E_tmp=zeros(n);
          
          for i1=1:nout
            Dc=sqrt(pi2(:,i1));
            Lc=(Dc*Dc').*K;
            Lc(1:n+1:end)=Lc(1:n+1:end)+1;
            Lc=chol(Lc);
            L(:,:,i1)=Lc;
            
            Ec=Lc'\diag(Dc);
            Ec=Ec'*Ec;
            E_tmp=E_tmp+Ec;
            E(:,:,i1)=Ec;
          end
          M=chol(E_tmp);
          
          det_diag=0;
          det_mat_sum = zeros(n);
          for i1 = 1:nout
            det_diag=det_diag+sum(log(diag(L(:,:,i1))));
            Kpi=K*diag(pi_vec((1:n)+(i1-1)*n));
            
            det_mat_sum=det_mat_sum+E(:,:,i1)*Kpi;
          end
          
          %det_mat_sum=(det_mat_sum+det_mat_sum')./2;
          det_mat=chol(eye(n)-det_mat_sum);
          det_term=sum(log(diag(det_mat)))+det_diag;
          
          
          logZ = a'*f/2 - feval(gp.lik.fh.ll, gp.lik, y, f2, z) + det_term;
          edata = logZ;
          
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
      for i=1:ncf
        gpcf = gp.cf{i};
        eprior = eprior + feval(gpcf.fh.e, gpcf, x, y);
      end

      % Evaluate the prior contribution to the error from likelihood function
      if isfield(gp, 'lik') && isfield(gp.lik, 'p')
        lik = gp.lik;
        eprior = eprior + feval(lik.fh.eprior, lik);
      end

      e = edata + eprior;
      
      w0 = w;
      e0 = e;
      edata0 = edata;
      eprior0 = eprior;
      f0 = f;
      L0 = L;
      M0 = M;
      E0 = E;
      W0 = W;
      n0 = size(x,1);
      %La20 = La2;
      a0 = a;
      p0=p;
    end
    
    assert(isreal(edata))
    assert(isreal(eprior))

  end
end
