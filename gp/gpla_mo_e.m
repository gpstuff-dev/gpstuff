function [e, edata, eprior, f, L, a, E, M, p] = gpla_mo_e(w, gp, varargin)
%GPLA_MO_E  Do Laplace approximation and return marginal log posterior
%                estimate for multioutput likelihood
%
%  Description
%    E = GPLA_MO_E(W, GP, X, Y, OPTIONS) takes a GP data
%    structure GP together with a matrix X of input vectors and a
%    matrix Y of target vectors, and finds the Laplace
%    approximation for the conditional posterior p(Y | X, th),
%    where th is the parameters. Returns the energy at th (see
%    below). Each row of X corresponds to one input vector and each
%    row of Y corresponds to one target vector.
%
%    [E, EDATA, EPRIOR] = GPLA_MO_E(W, GP, X, Y, OPTIONS)
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
%    GP = GPLA_MO_E('init', GP) takes a GP structure GP and
%    initializes required fields for the Laplace approximation.
%
%    GP = GPLA_MO_E('clearcache', GP) takes a GP structure GP and
%    cleares the internal cache stored in the nested function workspace
%
%    [e, edata, eprior, f, L, a, La2, p]
%       = gpla_mo_e(w, gp, x, y, varargin)
%    returns many useful quantities produced by EP algorithm.
%
% The Newton's method is implemented as described in
% Rasmussen and Williams (2006).

% Copyright (c) 2010 Jaakko Riihim�ki, Pasi Jyl�nki, 
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
  ip.addOptional('x', @(x) ~isempty(x) && isnumeric(x) && isreal(x) && all(isfinite(x(:))))
  ip.addOptional('y', @(x) ~isempty(x) && isnumeric(x) && isreal(x) && all(isfinite(x(:))))
  ip.addParamValue('z', [], @(x) isnumeric(x) && isreal(x) && all(isfinite(x(:))))
  ip.parse(w, gp, varargin{:});
  x=ip.Results.x;
  y=ip.Results.y;
  z=ip.Results.z;
  
  if strcmp(w, 'init')
    % initialize cache
    ch = [];

    % return function handle to the nested function ep_algorithm
    % this way each gp has its own peristent memory for EP
    gp.fh.e = @laplace_algorithm;
    e = gp;
    % remove clutter from the nested workspace
    clear w gp varargin ip x y z
  elseif strcmp(w, 'clearcache')
    % clear the cache
    gp.fh.e('clearcache');
  else
    % call laplace_algorithm using the function handle to the nested function
    % this way each gp has its own peristent memory for Laplace
    [e, edata, eprior, f, L, a, E, M, p] = gp.fh.e(w, gp, x, y, z);
  end

  function [e, edata, eprior, f, L, a, E, M, p] = laplace_algorithm(w, gp, x, y, z)
      
  if strcmp(w, 'clearcache')
	ch=[];
    return
  end
  % code for the Laplace algorithm

  % check whether saved values can be used
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
      f = ch.f;
      L = ch.L;
      E = ch.E;
      M = ch.M;
      a = ch.a;
      p = ch.p;
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
                           
              lp_new = gp.lik.fh.ll(gp.lik, y, f2, z);
              lp_old = -Inf;
              
              c=zeros(n*nout,1);
              ERMMRc=zeros(n*nout,1);
              pipif=zeros(n*nout,1);
              E=zeros(n,n,nout);
              L=zeros(n,n,nout);
              RER = zeros(n,n,nout);
              
              while lp_new - lp_old > tol
                lp_old = lp_new; a_old = a; 
                
                llg = gp.lik.fh.llg(gp.lik, y, f2, 'latent', z);
                [pi2_vec, pi2_mat] = gp.lik.fh.llg2(gp.lik, y, f2, 'latent', z);
                pi2 = reshape(pi2_vec,size(y));
                
                R = repmat(1./pi2_vec,1,n).*pi2_mat;
                for i1=1:nout
                  Dc=sqrt(pi2(:,i1));
                  Lc=(Dc*Dc').*K(:,:,i1);
                  Lc(1:n+1:end)=Lc(1:n+1:end)+1;
                  [Lc,notpositivedefinite]=chol(Lc);
                  if notpositivedefinite
                    [edata,e,eprior,f,L,a,E,M,p,ch] = set_output_for_notpositivedefinite();
                    return
                  end
                  L(:,:,i1)=Lc;
                  
                  Ec=Lc'\diag(Dc);
                  Ec=Ec'*Ec;
                  E(:,:,i1)=Ec;
                  RER(:,:,i1) = R((1:n)+(i1-1)*n,:)'*Ec*R((1:n)+(i1-1)*n,:);
                end
                [M, notpositivedefinite]=chol(sum(RER,3));
                if notpositivedefinite
                  [edata,e,eprior,f,L,a,E,M,p,ch] = set_output_for_notpositivedefinite();
                  return
                end
                
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
                
                lp_new = -a'*f/2 + gp.lik.fh.ll(gp.lik, y, f2, z);
                
                i = 0;
                while i < 10 && lp_new < lp_old  || isnan(sum(f))
                  % reduce step size by half
                  a = (a_old+a)/2;                                  
                  
                  for i1=1:nout
                    f((1:n)+(i1-1)*n)=K(:,:,i1)*a((1:n)+(i1-1)*n);
                  end
                  f2=reshape(f,n,nout);
                  
                  lp_new = -a'*f/2 + gp.lik.fh.ll(gp.lik, y, f2, z);
                  i = i+1;
                end 
              end
            otherwise 
              error('gpla_e: Unknown optimization method ! ')
          end
           
          [pi2_vec, pi2_mat] = gp.lik.fh.llg2(gp.lik, y, f2, 'latent', z);
          pi2 = reshape(pi2_vec,size(y));
          
          zc=0;
          Detn=0;
          R = repmat(1./pi2_vec,1,n).*pi2_mat;
          for i1=1:nout
              Dc=sqrt( pi2(:,i1) );
              Lc=(Dc*Dc').*K(:,:,i1);
              Lc(1:n+1:end)=Lc(1:n+1:end)+1;
              [Lc, notpositivedefinite]=chol(Lc);
              if notpositivedefinite
                [edata,e,eprior,f,L,a,E,M,p,ch] = set_output_for_notpositivedefinite();
                return
              end
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
          [M, notpositivedefinite]=chol(sum(RER,3));
          if notpositivedefinite
            [edata,e,eprior,f,L,a,E,M,p,ch] = set_output_for_notpositivedefinite();
            return
          end
          
          zc = zc + sum(log(diag(chol( eye(size(K(:,:,i1))) - Detn))));
          
          logZ = a'*f/2 - gp.lik.fh.ll(gp.lik, y, f2, z) + zc;
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
              eprior = eprior - gpcf.fh.lp(gpcf);
          end
      end

      % ============================================================
      % Evaluate the prior contribution to the error from Gaussian likelihood
      % ============================================================
      % Evaluate the prior contribution to the error from likelihood function
      if isfield(gp, 'lik') && isfield(gp.lik, 'p')
        lik = gp.lik;
        eprior = eprior - lik.fh.lp(lik);
      end

      e = edata + eprior;
      
      % store values to the cache
      ch.w = w;
      ch.e = e;
      ch.edata = edata;
      ch.eprior = eprior;
      ch.f = f;
      ch.L = L;
      ch.M = M;
      ch.E = E;
%       ch.W = W;
      ch.n = size(x,1);
      %La20 = La2;
      ch.a = a;
      ch.p=p;
      ch.datahash=datahash;
    end
    
    assert(isreal(edata))
    assert(isreal(eprior))
  end

function [edata,e,eprior,f,L,a,E,M,p,ch] = set_output_for_notpositivedefinite()
  % Instead of stopping to chol error, return NaN
  edata=NaN;
  e=NaN;
  eprior=NaN;
  f=NaN;
  L=NaN;
  a=NaN;
%   La2=NaN;
  E = NaN;
  M = NaN;
  p=NaN;
  ch.w = w;
  ch.e = e;
  ch.edata = edata;
  ch.eprior = eprior;
  ch.f = f;
  ch.L = L;
  ch.M = M;
  ch.E = E;
  ch.n = size(x,1);
  ch.La2 = La2;
  ch.a = a;
  ch.p=p;
  ch.datahash=datahash;
end
end

