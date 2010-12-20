function [e, edata, eprior, f, L, a, E, M, p] = gpla_mo_e(w, gp, varargin)
%GPLA_SOFTMAX_E  Do Laplace approximation and return marginal log posterior
%                estimate for multioutput likelihood
%
%  Description
%    E = GPLA_SOFTMAX_E(W, GP, X, Y, OPTIONS) takes a GP data
%    structure GP together with a matrix X of input vectors and a
%    matrix Y of target vectors, and finds the Laplace
%    approximation for the conditional posterior p(Y | X, th),
%    where th is the parameters. Returns the energy at th (see
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
%      where th represents the parameters (lengthScale,
%      magnSigma2...), X is inputs and Y is observations.
%
%    OPTIONS is optional parameter-value pair
%      z - optional observed quantity in triplet (x_i,y_i,z_i)
%          Some likelihoods may use this. For example, in case of
%          Poisson likelihood we have z_i=E_i, that is, expected
%          value for ith case.
%
%  See also
%    GP_SET, GP_E, GPLA_SOFTMAX_G, GPLA_SOFTMAX_PRED

%  Description 2
%    Additional properties meant only for internal use.
%  
%    GP = GPLA_SOFTMAX_E('init', GP) takes a GP structure GP and
%    initializes required fields for the Laplace approximation.
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
  ip.FunctionName = 'GPLA_MO_E';
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
      % The parameters or data have changed since
      % the last call for gpla_e. In this case we need to
      % re-evaluate the Laplace approximation
      if size(x,1) ~= size(y,1)
          error('GPLA_MO_E: x and y must contain equal amount of rows!')
      end
      [n,nout] = size(y);
      p = [];
      
      if isfield(gp, 'comp_cf')  % own covariance for each ouput component
          multicf = true;
          if length(gp.comp_cf) ~= nout
              error('GPLA_MO_E: the number of component vectors in gp.comp_cf must be the same as number of outputs.')
          end
      else
          multicf = false;
      end
      
      gp=gp_unpak(gp, w);
      ncf = length(gp.cf);
      
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
              K = zeros(n,n,nout);
              if multicf
                  for i1=1:nout
                      K(:,:,i1) = gp_trcov(gp, x, gp.comp_cf{i1});
                  end
              else
                  for i1=1:nout
                      K(:,:,i1) = gp_trcov(gp, x);
                  end
              end

          switch gp.latent_opt.optim_method
            case 'newton'
              tol = 1e-12;
              a = f;
              
              f2=reshape(f,n,nout);
                           
              lp_new = feval(gp.lik.fh.ll, gp.lik, y, f2, z);
              lp_old = -Inf;
              
              c=zeros(n*nout,1);
              ERMMRc=zeros(n*nout,1);
              pipif=zeros(n*nout,1);
              E=zeros(n,n,nout);
              L=zeros(n,n,nout);
              RER = zeros(n,n,nout);
              
              while lp_new - lp_old > tol
                lp_old = lp_new; a_old = a; 
                
                llg = feval(gp.lik.fh.llg,gp.lik, y, f2, 'latent', z);
                [pi2_vec, pi2_mat] = feval(gp.lik.fh.llg2, gp.lik, y, f2, 'latent', z);
                pi2 = reshape(pi2_vec,size(y));
                
                R = repmat(1./pi2_vec,1,n).*pi2_mat;
                for i1=1:nout
                  Dc=sqrt(pi2(:,i1));
                  Lc=(Dc*Dc').*K(:,:,i1);
                  Lc(1:n+1:end)=Lc(1:n+1:end)+1;
                  Lc=chol(Lc);
                  L(:,:,i1)=Lc;
                  
                  Ec=Lc'\diag(Dc);
                  Ec=Ec'*Ec;
                  E(:,:,i1)=Ec;
                  RER(:,:,i1) = R((1:n)+(i1-1)*n,:)'*Ec*R((1:n)+(i1-1)*n,:);
                end
                M=chol(sum(RER,3));
                
                b = pi2_vec.*f - pi2_mat*(pi2_mat'*f) + llg;                
                for i1=1:nout
                  c((1:n)+(i1-1)*n)=E(:,:,i1)*(K(:,:,i1)*b((1:n)+(i1-1)*n));
                end
               
                RMMRc=R*(M\(M'\(R'*c)));
                for i1=1:nout
                  ERMMRc((1:n)+(i1-1)*n) = E(:,:,i1)*RMMRc((1:n)+(i1-1)*n,:);
                end
                a=b-c+ERMMRc;
                
                for i1=1:nout
                  f((1:n)+(i1-1)*n)=K(:,:,i1)*a((1:n)+(i1-1)*n);
                end
                f2=reshape(f,n,nout);
                
                lp_new = -a'*f/2 + feval(gp.lik.fh.ll, gp.lik, y, f2, z);
                
                i = 0;
                while i < 10 && lp_new < lp_old  || isnan(sum(f))
                  % reduce step size by half
                  a = (a_old+a)/2;                                  
                  
                  for i1=1:nout
                    f((1:n)+(i1-1)*n)=K(:,:,i1)*a((1:n)+(i1-1)*n);
                  end
                  f2=reshape(f,n,nout);
                  
                  lp_new = -a'*f/2 + feval(gp.lik.fh.ll, gp.lik, y, f2, z);
                  i = i+1;
                end 
              end
            otherwise 
              error('gpla_e: Unknown optimization method ! ')
          end
           
          [pi2_vec, pi2_mat] = feval(gp.lik.fh.llg2, gp.lik, y, f2, 'latent', z);
          pi2 = reshape(pi2_vec,size(y));
          
          zc=0;
          Detn=0;
          R = repmat(1./pi2_vec,1,n).*pi2_mat;
          for i1=1:nout
              Dc=sqrt( pi2(:,i1) );
              Lc=(Dc*Dc').*K(:,:,i1);
              Lc(1:n+1:end)=Lc(1:n+1:end)+1;
              Lc=chol(Lc);
              L(:,:,i1)=Lc;
              
              pi2i = pi2_mat((1:n)+(i1-1)*n,:);
              pipi = pi2i'/diag(Dc);
              Detn = Detn + pipi*(Lc\(Lc'\diag(Dc)))*K(:,:,i1)*pi2i;
              zc = zc + sum(log(diag(Lc)));
              
              Ec=Lc'\diag(Dc);
              Ec=Ec'*Ec;
              E(:,:,i1)=Ec;
              RER(:,:,i1) = R((1:n)+(i1-1)*n,:)'*Ec*R((1:n)+(i1-1)*n,:);
          end
          M=chol(sum(RER,3));
          
          zc = zc + sum(log(diag(chol( eye(size(K(:,:,i1))) - Detn))));
          
          logZ = a'*f/2 - feval(gp.lik.fh.ll, gp.lik, y, f2, z) + zc;
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
