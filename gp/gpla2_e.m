function [e, edata, eprior, f, L, a, E, M, p] = gpla2_e(w, gp, varargin)
%GPLA_ND_E  Do Laplace approximation and return marginal log posterior
%                estimate for multioutput likelihood
%
%  Description
%    E = GPLA_ND_E(W, GP, X, Y, OPTIONS) takes a GP data
%    structure GP together with a matrix X of input vectors and a
%    matrix Y of target vectors, and finds the Laplace
%    approximation for the conditional posterior p(Y | X, th),
%    where th is the parameters. Returns the energy at th (see
%    below). Each row of X corresponds to one input vector and each
%    row of Y corresponds to one target vector.
%
%    [E, EDATA, EPRIOR] = GPLA_ND_E(W, GP, X, Y, OPTIONS)
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
%    GP_SET, GP_E, GPLA_ND_G, GPLA_ND_PRED

%  Description 2
%    Additional properties meant only for internal use.
%
%    GP = GPLA_ND_E('init', GP) takes a GP structure GP and
%    initializes required fields for the Laplace approximation.
%
%    GP = GPLA_ND_E('clearcache', GP) takes a GP structure GP and
%    cleares the internal cache stored in the nested function workspace
%
%    [e, edata, eprior, f, L, a, La2, p]
%       = gpla_nd_e(w, gp, x, y, varargin)
%    returns many useful quantities produced by EP algorithm.
%
% The Newton's method is implemented as described in
% Rasmussen and Williams (2006).

% Copyright (c) 2010 Jaakko Riihim�ki, Pasi Jyl�nki,
%                    Jarno Vanhatalo, Aki Vehtari

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

% parse inputs
ip=inputParser;
ip.FunctionName = 'GPLA_ND_E';
ip.addRequired('w', @(x) ...
  isempty(x) || ...
  (ischar(x) && strcmp(w, 'init')) || ...
  isvector(x) && isreal(x) && all(isfinite(x)) ...
  || all(isnan(x)));
ip.addRequired('gp',@isstruct);
ip.addOptional('x', @(x) ~isempty(x) && isnumeric(x) && isreal(x) && all(isfinite(x(:))))
ip.addOptional('y', @(x) ~isempty(x) && isnumeric(x) && isreal(x) && all(isfinite(x(:))))
ip.addParamValue('z', [], @(x) isnumeric(x) && isreal(x) && all(isfinite(x(:))))
ip.parse(w, gp, varargin{:});
x=ip.Results.x;
y=ip.Results.y;
z=ip.Results.z;

if strcmp(w, 'init')
  % Initialize cache
  ch = [];
  
  % return function handle to the nested function laplace_algorithm
  % this way each gp has its own peristent memory for EP
  gp.fh.ne = @laplace_algorithm;
  % set other function handles
  gp.fh.e=@gpla2_e;
  gp.fh.g=@gpla2_g;
  gp.fh.pred=@gpla2_pred;
  gp.fh.jpred=@gpla2_jpred;
  gp.fh.looe=@gpla2_looe;
  gp.fh.loog=@gpla2_loog;
  e = gp;
  % remove clutter from the nested workspace
  clear w gp varargin ip x y z
elseif strcmp(w, 'clearcache')
  % clear the cache
  gp.fh.ne('clearcache');
else
  % call laplace_algorithm using the function handle to the nested function
  % this way each gp has its own peristent memory for Laplace
  [e, edata, eprior, f, L, a, E, M, p] = gp.fh.ne(w, gp, x, y, z);
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
        error('GPLA_ND_E: x and y must contain equal amount of rows!')
      end
      [n,nout] = size(y);
      p = [];
      
      if isfield(gp, 'comp_cf')  % own covariance for each ouput component
        multicf = true;
        if length(gp.comp_cf) ~= nout && nout > 1
          error('GPLA_ND_E: the number of component vectors in gp.comp_cf must be the same as number of outputs.')
        end
      else
        multicf = false;
      end
      
      gp=gp_unpak(gp, w);
      ncf = length(gp.cf);
      maxiter = gp.latent_opt.maxiter;
      
      % =================================================
      % First Evaluate the data contribution to the error
      switch gp.type
        % ============================================================
        % FULL
        % ============================================================
        case 'FULL'
          
          switch gp.lik.type
            
            case {'LGP', 'LGPC'}
              
              nl=n;
              
              % Initialize latent values
              % zero seems to be a robust choice (Jarno)
              % with mean functions, initialize to mean function values
              if ~isfield(gp,'meanf')
                f = zeros(sum(nl),1);
              else
                [H,b_m,B_m]=mean_prep(gp,x,[]);
                Hb_m=H'*b_m;
                f = Hb_m;
              end
              
              if isfield(gp.latent_opt, 'kron') && gp.latent_opt.kron==1
                gptmp=gp; gptmp.jitterSigma2=0;
                % Use Kronecker product kron(Ka,Kb) instead of K
                Ka = gp_trcov(gptmp, unique(x(:,1)));
                % fix the magnitude sigma to 1 for Kb matrix
                wtmp=gp_pak(gptmp); wtmp(1)=0; gptmp=gp_unpak(gptmp,wtmp);
                Kb = gp_trcov(gptmp, unique(x(:,2)));
                clear gptmp
                n1=size(Ka,1);
                n2=size(Kb,1);
                
                [Va,Da]=eig(Ka); [Vb,Db]=eig(Kb);
                % eigenvalues of K matrix
                Dtmp=kron(diag(Da),diag(Db));
                [sDtmp,istmp]=sort(Dtmp,'descend');
                
                % Form the low-rank approximation.  Exclude eigenvalues
                % smaller than gp.latent_opt.eig_tol or take
                % gp.latent_opt.eig_prct*n eigenvalues at most.
                nlr=min([sum(sDtmp>gp.latent_opt.eig_tol) round(gp.latent_opt.eig_prct*n)]);
                sDtmp=sDtmp+gp.jitterSigma2;
                
                itmp1=meshgrid(1:n1,1:n2);
                itmp2=meshgrid(1:n2,1:n1)';
                ind=[itmp1(:) itmp2(:)];
                
                % included eigenvalues
                Dlr=sDtmp(1:nlr);
                % included eigenvectors
                Vlr=zeros(n,nlr);
                for i1=1:nlr
                  Vlr(:,i1)=kron(Va(:,ind(istmp(i1),1)),Vb(:,ind(istmp(i1),2)));
                end
                %L=[];
                
                % diag(K)-diag(Vlr*diag(Dlr)*Vlr')
                Lb=gp_trvar(gp,x)-sum(bsxfun(@times,Vlr.*Vlr,Dlr'),2);
                if isfield(gp,'meanf')
                  Dt=[Dlr; diag(B_m)];
                  Vt=[Vlr H'];
                else
                  Dt=Dlr;
                  Vt=Vlr;
                end
                Dtsq=sqrt(Dt);
                
              elseif isfield(gp.latent_opt, 'fft') && gp.latent_opt.fft==1
                % unique values from covariance matrix
                K1 = gp_cov(gp, x(1,:), x);
                K1(1)=K1(1)+gp.jitterSigma2;
                if size(x,2)==1
                  % form circulant matrix to avoid border effects
                  Kcirc=[K1 0 K1(end:-1:2)];
                  fftKcirc = fft(Kcirc);
                elseif size(x,2)==2
                  n1=gp.latent_opt.gridn(1);
                  n2=gp.latent_opt.gridn(2);
                  Ktmp=reshape(K1,n2,n1);
                  % form circulant matrix to avoid border effects
                  Ktmp=[Ktmp; zeros(1,n1); flipud(Ktmp(2:end,:))];
                  fftKcirc=fft2([Ktmp zeros(2*n2,1) fliplr(Ktmp(:,2:end))]);
                else
                  error('FFT speed-up implemented only for 1D and 2D cases.')
                end
              else
                K = gp_trcov(gp, x);
              end
              
              % Mean function contribution to K
              if isfield(gp,'meanf')
                if isfield(gp.latent_opt, 'kron') && gp.latent_opt.kron==1
                  % only zero mean function implemented for Kronecker
                  % approximation
                  iKHb_m=zeros(n,1);
                elseif isfield(gp.latent_opt, 'fft') && gp.latent_opt.fft==1
                  % only zero mean function implemented for FFT speed-up
                  iKHb_m=zeros(n,1);
                else
                  K=K+H'*B_m*H;
                  iKHb_m=K\Hb_m;
                end
              end
              
              
              % Main Newton algorithm
              
              tol = 1e-12;
              a = f;
              if isfield(gp,'meanf')
                a = a-Hb_m;
              end
              
              % a vector to form the second gradient
              g2 = gp.lik.fh.llg2(gp.lik, y, f, 'latent', z);
              g2sq=sqrt(g2);
              
              ny=sum(y); % total number of observations
              
              dlp = gp.lik.fh.llg(gp.lik, y, f, 'latent', z);
              lp_new = gp.lik.fh.ll(gp.lik, y, f, z);
              lp_old = -Inf;
              
              iter=0;
              while abs(lp_new - lp_old) > tol && iter < maxiter
                iter = iter + 1;
                lp_old = lp_new; a_old = a;
                
                
                if ~isfield(gp,'meanf')
                  if strcmpi(gp.lik.type,'LGPC')
                    n1=gp.lik.gridn(1); n2=gp.lik.gridn(2);
                    b=zeros(n,1);
                    ny2=sum(reshape(y,fliplr(gp.lik.gridn)));
                    for k1=1:n1
                      b((1:n2)+(k1-1)*n2) = ny2(k1)*(g2((1:n2)+(k1-1)*n2).*f((1:n2)+(k1-1)*n2)-g2((1:n2)+(k1-1)*n2)*(g2((1:n2)+(k1-1)*n2)'*f((1:n2)+(k1-1)*n2)))+dlp((1:n2)+(k1-1)*n2);
                    end
                  else
                    b = ny*(g2.*f-g2*(g2'*f))+dlp;
                    %b = W.*f+dlp;
                  end
                else
                  if strcmpi(gp.lik.type,'LGPC')
                    n1=gp.lik.gridn(1); n2=gp.lik.gridn(2);
                    b=zeros(n,1);
                    ny2=sum(reshape(y,fliplr(gp.lik.gridn)));
                    for k1=1:n1
                      b((1:n2)+(k1-1)*n2) = ny2(k1)*(g2((1:n2)+(k1-1)*n2).*f((1:n2)+(k1-1)*n2)-g2((1:n2)+(k1-1)*n2)*(g2((1:n2)+(k1-1)*n2)'*f((1:n2)+(k1-1)*n2)))+iKHb_m((1:n2)+(k1-1)*n2)+dlp((1:n2)+(k1-1)*n2);
                    end
                  else
                    b = ny*(g2.*f-g2*(g2'*f))+iKHb_m+dlp;
                    %b = W.*f+K\(H'*b_m)+dlp;
                  end
                end
                
                if isfield(gp.latent_opt, 'kron') && gp.latent_opt.kron==1
                  
                  % Use Kronecker product structure in matrix vector
                  % multiplications
                  %-
                  % q=Kb*reshape(b,n2,n1)*Ka;
                  % Kg=q(:);
                  % Kg=Kg+gp.jitterSigma2*b;
                  %-
                  % OR use reduced-rank approximation for K
                  %-
                  Kg=Lb.*b+Vlr*(Dlr.*(Vlr'*b));
                  %-
                  
                  if isfield(gp,'meanf')
                    Kg=Kg+H'*(B_m*(H*b));
                  end
                  
                  % % Use Kronecker product structure
                  %-
                  % v=sqrt(ny)*(g2sq.*Kg-(g2*(g2'*Kg))./g2sq);
                  % % fast matrix vector multiplication with
                  % % Kronecker product for matrix inversion
                  % if isfield(gp,'meanf')
                  %   [iSg,~]=pcg(@(z) mvm_kron(g2,ny,Ka,Kb,H,B_m,gp.jitterSigma2,z), v, gp.latent_opt.pcg_tol);
                  % else
                  %   [iSg,~]=pcg(@(z) mvm_kron(g2,ny,Ka,Kb,[],[],gp.jitterSigma2,z), v, gp.latent_opt.pcg_tol);
                  % end
                  % a=b-sqrt(ny)*(g2sq.*iSg  - g2*(g2'*(iSg./g2sq)));
                  %-
                  
                  % use reduced-rank approximation for K
                  %-
                  Zt=1./(1+ny*g2.*Lb);
                  Ztsq=sqrt(Zt);
                  Ltmp=bsxfun(@times,Ztsq.*sqrt(ny).*g2sq,bsxfun(@times,Vt,sqrt(Dt)'));
                  Ltmp=Ltmp'*Ltmp;
                  Ltmp(1:(size(Dt,1)+1):end)=Ltmp(1:(size(Dt,1)+1):end)+1;
                  L=chol(Ltmp,'lower');
                  
                  EKg=ny*g2.*(Zt.*Kg)-sqrt(ny)*g2sq.*(Zt.*(sqrt(ny)*g2sq.*(Vt*(Dtsq.*(L'\(L\(Dtsq.*(Vt'*(sqrt(ny)*g2sq.*(Zt.*(sqrt(ny)*g2sq.*Kg)))))))))));
                  E1=ny*g2.*(Zt.*ones(n,1))-sqrt(ny)*g2sq.*(Zt.*(sqrt(ny)*g2sq.*(Vt*(Dtsq.*(L'\(L\(Dtsq.*(Vt'*(sqrt(ny)*g2sq.*(Zt.*(sqrt(ny)*g2sq.*ones(n,1))))))))))));
                  a=b-(EKg-E1*((E1'*Kg)./(ones(1,n)*E1)));
                  %-
                  
                elseif isfield(gp.latent_opt, 'fft') && gp.latent_opt.fft==1
                  
                  % use FFT speed-up in matrix vector multiplications
                  if size(x,2)==1
                    gge=zeros(2*n,1);
                    gge(1:n)=b;
                    q=ifft(fftKcirc.*fft(gge'));
                    Kg=q(1:n)';
                  elseif size(x,2)==2
                    gge=zeros(2*n2,2*n1);
                    gge(1:n2,1:n1)=reshape(b,n2,n1);
                    
                    q=ifft2(fftKcirc.*fft2(gge));
                    q=q(1:n2,1:n1);
                    Kg=q(:);
                  else
                    error('FFT speed-up implemented only for 1D and 2D cases.')
                  end
                  
                  if isfield(gp,'meanf')
                    Kg=Kg+H'*(B_m*(H*b));
                  end
                  v=sqrt(ny)*(g2sq.*Kg-(g2*(g2'*Kg))./g2sq);
                  
                  if isfield(gp,'meanf')
                    % fast matrix vector multiplication with fft for matrix inversion
                    [iSg,~]=pcg(@(z) mvm_fft(g2,ny,fftKcirc,H,B_m,z), v, gp.latent_opt.pcg_tol);
                  else
                    [iSg,~]=pcg(@(z) mvm_fft(g2,ny,fftKcirc,[],[],z), v, gp.latent_opt.pcg_tol);
                  end
                  a=b-sqrt(ny)*(g2sq.*iSg  - g2*(g2'*(iSg./g2sq)));
                  
                else
                  if strcmpi(gp.lik.type,'LGPC')
                    R=zeros(n);
                    RKR=K;
                    for k1=1:n1
                      R((1:n2)+(k1-1)*n2,(1:n2)+(k1-1)*n2)=sqrt(ny2(k1))*(diag(g2sq((1:n2)+(k1-1)*n2))-g2((1:n2)+(k1-1)*n2)*g2sq((1:n2)+(k1-1)*n2)');
                      RKR(:,(1:n2)+(k1-1)*n2)=RKR(:,(1:n2)+(k1-1)*n2)*R((1:n2)+(k1-1)*n2,(1:n2)+(k1-1)*n2);
                    end
                    for k1=1:n1
                      RKR((1:n2)+(k1-1)*n2,:)=R((1:n2)+(k1-1)*n2,(1:n2)+(k1-1)*n2)'*RKR((1:n2)+(k1-1)*n2,:);
                    end
                    %RKR=R'*K*R;
                    RKR(1:(n+1):end)=RKR(1:(n+1):end)+1;
                    [L,notpositivedefinite] = chol(RKR,'lower');
                    
                    if notpositivedefinite
                      [edata,e,eprior,f,L,a,E,M,p,ch] = set_output_for_notpositivedefinite();
                      return
                    end
                    
                    Kb=K*b;
                    RCb=R'*Kb;
                    
                    iRCb=L'\(L\RCb);
                    a=b-R*iRCb;
                  else
                    %R=-g2*g2sq'; R(1:(n+1):end)=R(1:(n+1):end)+g2sq';
                    KR=bsxfun(@times,K,g2sq')-(K*g2)*g2sq';
                    RKR=ny*(bsxfun(@times,g2sq,KR)-g2sq*(g2'*KR));
                    RKR(1:(n+1):end)=RKR(1:(n+1):end)+1;
                    [L,notpositivedefinite] = chol(RKR,'lower');
                    
                    if notpositivedefinite
                      [edata,e,eprior,f,L,a,E,M,p,ch] = set_output_for_notpositivedefinite();
                      return
                    end
                    
                    Kb=K*b;
                    RCb=g2sq.*Kb-g2sq*(g2'*Kb);
                    iRCb=L'\(L\RCb);
                    a=b-ny*(g2sq.*iRCb-g2*(g2sq'*iRCb));
                  end
                end
                
                
                if isfield(gp.latent_opt, 'kron') && gp.latent_opt.kron==1
                  
                  % % Use Kronecker product structure
                  %-
                  % f2=Kb*reshape(a,n2,n1)*Ka;
                  % f=f2(:);
                  % f=f+gp.jitterSigma2*a;
                  %-
                  % use reduced-rank approximation for K
                  %-
                  f=Lb.*a+Vlr*(Dlr.*(Vlr'*a));
                  %-
                  
                  if isfield(gp,'meanf')
                    f=f+H'*(B_m*(H*a));
                  end
                elseif isfield(gp.latent_opt, 'fft') && gp.latent_opt.fft==1
                  if size(x,2)==1
                    a2=zeros(2*n,1);
                    a2(1:n)=a;
                    f2=ifft(fftKcirc.*fft(a2'));
                    f=f2(1:n)';
                    if isfield(gp,'meanf')
                      f=f+H'*(B_m*(H*a));
                    end
                  elseif size(x,2)==2
                    a2=zeros(2*n2,2*n1);
                    a2(1:n2,1:n1)=reshape(a,n2,n1);
                    
                    f2=ifft2(fftKcirc.*fft2(a2));
                    f2=f2(1:n2,1:n1);
                    f=f2(:);
                    if isfield(gp,'meanf')
                      f=f+H'*(B_m*(H*a));
                    end
                  else
                    error('FFT speed-up implemented only for 1D and 2D cases.')
                  end
                else
                  f = K*a;
                end
                
                lp = gp.lik.fh.ll(gp.lik, y, f, z);
                if ~isfield(gp,'meanf')
                  lp_new = -a'*f/2 + lp;
                else
                  %lp_new = -(f-H'*b_m)'*(a-K\(H'*b_m))/2 + lp; %f^=f-H'*b_m,
                  lp_new = -(f-Hb_m)'*(a-iKHb_m)/2 + lp; %f^=f-Hb_m,
                end
                i = 0;
                while i < 10 && lp_new < lp_old && ~isnan(sum(f))
                  % reduce step size by half
                  a = (a_old+a)/2;
                  
                  if isfield(gp.latent_opt, 'kron') && gp.latent_opt.kron==1
                    % % Use Kronecker product structure
                    %-
                    % f2=Kb*reshape(a,n2,n1)*Ka;
                    % f=f2(:);
                    % f=f+gp.jitterSigma2*a;
                    %-
                    % use reduced-rank approximation for K
                    %-
                    f=Lb.*a+Vlr*(Dlr.*(Vlr'*a));
                    %-
                    
                    if isfield(gp,'meanf')
                      f=f+H'*(B_m*(H*a));
                    end
                  elseif isfield(gp.latent_opt, 'fft') && gp.latent_opt.fft==1
                    if size(x,2)==1
                      a2=zeros(2*n,1);
                      a2(1:n)=a;
                      f2=ifft(fftKcirc.*fft(a2'));
                      f=f2(1:n)';
                      
                      if isfield(gp,'meanf')
                        f=f+H'*(B_m*(H*a));
                      end
                    elseif size(x,2)==2
                      a2=zeros(2*n2,2*n1);
                      a2(1:n2,1:n1)=reshape(a,n2,n1);
                      f2=ifft2(fftKcirc.*fft2(a2));
                      f2=f2(1:n2,1:n1);
                      f=f2(:);
                      
                      if isfield(gp,'meanf')
                        f=f+H'*(B_m*(H*a));
                      end
                    else
                      error('FFT speed-up implemented only for 1D and 2D cases.')
                    end
                  else
                    f = K*a;
                  end
                  
                  lp = gp.lik.fh.ll(gp.lik, y, f, z);
                  if ~isfield(gp,'meanf')
                    lp_new = -a'*f/2 + lp;
                  else
                    %lp_new = -(f-H'*b_m)'*(a-K\(H'*b_m))/2 + lp;
                    lp_new = -(f-Hb_m)'*(a-iKHb_m)/2 + lp;
                  end
                  i = i+1;
                end
                
                g2 = gp.lik.fh.llg2(gp.lik, y, f, 'latent', z);
                g2sq=sqrt(g2);
                
                dlp = gp.lik.fh.llg(gp.lik, y, f, 'latent', z);
              end
              
              % evaluate the approximate log marginal likelihood
              g2 = gp.lik.fh.llg2(gp.lik, y, f, 'latent', z);
              g2sq=sqrt(g2);
              
              if ~isfield(gp,'meanf')
                logZ = 0.5 *f'*a - gp.lik.fh.ll(gp.lik, y, f, z);
              else
                % logZ = 0.5 *((f-H'*b_m)'*(a-K\(H'*b_m))) - gp.lik.fh.ll(gp.lik, y, f, z);
                logZ = 0.5 *((f-Hb_m)'*(a-iKHb_m)) - gp.lik.fh.ll(gp.lik, y, f, z);
              end
              
              
              if isfield(gp.latent_opt, 'kron') && gp.latent_opt.kron==1
                % % Use Kronecker product structure
                %-
                % tmp=bsxfun(@times,Lb.^(-1/2),bsxfun(@times,Vt,sqrt(Dt)'));
                % tmp=tmp'*tmp;
                % tmp(1:size(tmp,1)+1:end)=tmp(1:size(tmp,1)+1:end)+1;
                % logZa=sum(log(diag(chol(tmp,'lower'))));
                %
                % Lbt=ny*(g2)+1./Lb;
                %
                % St=[diag(1./Dt)+Vt'*bsxfun(@times,1./Lb,Vt) zeros(size(Dt,1),1); ...
                %   zeros(1,size(Dt,1)) 1];
                % Pt=[bsxfun(@times,1./Lb,Vt) sqrt(ny)*g2];
                %
                % logZb=sum(log(diag(chol(St,'lower'))));
                %
                % Ptt=bsxfun(@times,1./sqrt(Lbt),Pt);
                % logZc=sum(log(diag(chol(St-Ptt'*Ptt,'lower'))));
                %
                % edata = logZ + logZa - logZb + logZc + 0.5*sum(log(Lb)) + 0.5*sum(log(Lbt));
                %-
                
                % use reduced-rank approximation for K
                %-
                Zt=1./(1+ny*g2.*Lb);
                Ztsq=sqrt(Zt);
                Ltmp=bsxfun(@times,Ztsq.*sqrt(ny).*g2sq,bsxfun(@times,Vt,sqrt(Dt)'));
                Ltmp=Ltmp'*Ltmp;
                Ltmp(1:(size(Dt,1)+1):end)=Ltmp(1:(size(Dt,1)+1):end)+1;
                L=chol(Ltmp,'lower');
                
                LTtmp=L\( Dtsq.*(Vt'*( (g2sq.*sqrt(ny)).*((1./(1+ny*g2.*Lb)).* (sqrt(ny)*g2sq) ) )) );
                edata = logZ + sum(log(diag(L)))+0.5*sum(log(1+ny*g2.*Lb)) ...
                  -0.5*log(ny) + 0.5*log(sum(((g2*ny)./(ny*g2.*Lb+1)))-LTtmp'*LTtmp);
                %-
              elseif isfield(gp.latent_opt, 'fft') && gp.latent_opt.fft==1
                
                K = gp_trcov(gp, x);
                if isfield(gp,'meanf')
                  K=K+H'*B_m*H;
                end
                
                % exact determinant
                KR=bsxfun(@times,K,g2sq')-(K*g2)*g2sq';
                RKR=ny*(bsxfun(@times,g2sq,KR)-g2sq*(g2'*KR));
                RKR(1:(n+1):end)=RKR(1:(n+1):end)+1;
                [L,notpositivedefinite] = chol(RKR,'lower');
                if notpositivedefinite
                  [edata,e,eprior,f,L,a,E,M,p,ch] = set_output_for_notpositivedefinite();
                  return
                end
                edata = logZ + sum(log(diag(L)));
                
                % % determinant approximated using only the largest eigenvalues
                % opts.issym = 1;
                % Deig=eigs(@(z) mvm_fft(g2, ny, fftKcirc, H, B_m, z),n,round(n*0.05),'lm',opts);
                % edata = logZ + 0.5*sum(log(Deig));
                % L=[];
              else
                
                if strcmpi(gp.lik.type,'LGPC')
                  R=zeros(n);
                  RKR=K;
                  for k1=1:n1
                    R((1:n2)+(k1-1)*n2,(1:n2)+(k1-1)*n2)=sqrt(ny2(k1))*(diag(g2sq((1:n2)+(k1-1)*n2))-g2((1:n2)+(k1-1)*n2)*g2sq((1:n2)+(k1-1)*n2)');
                    RKR(:,(1:n2)+(k1-1)*n2)=RKR(:,(1:n2)+(k1-1)*n2)*R((1:n2)+(k1-1)*n2,(1:n2)+(k1-1)*n2);
                  end
                  for k1=1:n1
                    RKR((1:n2)+(k1-1)*n2,:)=R((1:n2)+(k1-1)*n2,(1:n2)+(k1-1)*n2)'*RKR((1:n2)+(k1-1)*n2,:);
                  end
                  %RKR=R'*K*R;
                else
                  KR=bsxfun(@times,K,g2sq')-(K*g2)*g2sq';
                  RKR=ny*(bsxfun(@times,g2sq,KR)-g2sq*(g2'*KR));
                end
                RKR(1:(n+1):end)=RKR(1:(n+1):end)+1;
                [L,notpositivedefinite] = chol(RKR,'lower');
                if notpositivedefinite
                  [edata,e,eprior,f,L,a,E,M,p,ch] = set_output_for_notpositivedefinite();
                  return
                end
                edata = logZ + sum(log(diag(L)));
              end
              
              M=[];
              E=[];
              
            case {'Softmax', 'Multinom'}
              
              
              % Initialize latent values
              % zero seems to be a robust choice (Jarno)
              f = zeros(size(y(:)));
              
              K = zeros(n,n,nout);
              if multicf
                for i1=1:nout
                  K(:,:,i1) = gp_trcov(gp, x, gp.comp_cf{i1});
                end
              else
                Ktmp=gp_trcov(gp, x);
                for i1=1:nout
                  K(:,:,i1) = Ktmp;
                end
              end
              
              % Main newton algorithm, see Rasmussen & Williams (2006),
              % p. 50
              
              tol = 1e-12;
              a = f;
              
              f2=reshape(f,n,nout);
              
              % lp_new = log(p(y|f))
              lp_new = gp.lik.fh.ll(gp.lik, y, f2, z);
              lp_old = -Inf;
              
              c=zeros(n*nout,1);
              ERMMRc=zeros(n*nout,1);
              E=zeros(n,n,nout);
              L=zeros(n,n,nout);
              RER = zeros(n,n,nout);
              
              while lp_new - lp_old > tol
                lp_old = lp_new; a_old = a;
                
                % llg = d(log(p(y|f)))/df
                llg = gp.lik.fh.llg(gp.lik, y, f2, 'latent', z);
                % Second derivatives
                [pi2_vec, pi2_mat] = gp.lik.fh.llg2(gp.lik, y, f2, 'latent', z);
                % W = -diag(pi2_vec) + pi2_mat*pi2_mat'
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
              
            otherwise
              
              if ~isfield(gp, 'comp_cf') || isempty(gp.comp_cf)
                error('Define multiple covariance functions for latent processes using gp.comp_cf (see gp_set)');
              end
              
              if isfield(gp.lik,'xtime')
                xtime=gp.lik.xtime;
                ntime = size(xtime,1);
                nl=[ntime n];
              else
                nl=repmat(n,1,length(gp.comp_cf));
              end
              nlp=length(nl); % number of latent processes
              
              % Initialize latent values
              % zero seems to be a robust choice (Jarno)
              % with mean functions, initialize to mean function values
              if ~isfield(gp,'meanf')
                f = zeros(sum(nl),1);
                if isequal(gp.lik.type, 'Inputdependentnoise')
                  % Inputdependent-noise needs initialization to mean
                  Kf = gp_trcov(gp,x,gp.comp_cf{1});
                  f(1:n) = Kf*((Kf+gp.lik.sigma2.*eye(n))\y);
                end
              else
                [H,b_m,B_m]=mean_prep(gp,x,[]);
                Hb_m=H'*b_m;
                f = Hb_m;
              end
              
              % K is block-diagonal covariance matrix where blocks
              % correspond to latent processes
              K = zeros(sum(nl));
              if isfield(gp.lik,'xtime')
                K(1:ntime,1:ntime)=gp_trcov(gp, xtime, gp.comp_cf{1});
                K((1:n)+ntime,(1:n)+ntime) = gp_trcov(gp, x, gp.comp_cf{2});
              else
                for i1=1:nlp
                  K((1:n)+(i1-1)*n,(1:n)+(i1-1)*n) = gp_trcov(gp, x, gp.comp_cf{i1});
                end
              end
              
              % Mean function contribution to K
              if isfield(gp,'meanf')
                K=K+H'*B_m*H;
                iKHb_m=K\Hb_m;
              end
              
              % Main Newton algorithm, see Rasmussen & Williams (2006),
              % p. 46
              
              tol = 1e-12;
              a = f;
              if isfield(gp,'meanf')
                a = a-Hb_m;
              end
              
              % Second derivatives of log-likelihood
              if isfield(gp.lik,'xtime')
                [llg2diag, llg2mat] = gp.lik.fh.llg2(gp.lik, y, f, 'latent', z);
                % W = [diag(Wdiag(1:ntime)) Wmat; Wmat' diag(Wdiag(ntime+1:end)]
                Wdiag=-llg2diag; Wmat=-llg2mat;
                W=[];
              else
                Wvec = -gp.lik.fh.llg2(gp.lik, y, f, 'latent', z);
                % W = [diag(Wvec(1:n,1)) diag(Wvec(1:n,2)) diag(Wvec(n+1:end,1)) diag(Wvec(n+1:end,2))]
                Wdiag=[Wvec(1:nl(1),1); Wvec(nl(1)+(1:nl(2)),2)];
              end
              % dlp = d(log(p(y|f)))/df
              dlp = gp.lik.fh.llg(gp.lik, y, f, 'latent', z);
              % lp_new = log(p(y|f))
              lp_new = gp.lik.fh.ll(gp.lik, y, f, z);
              lp_old = -Inf;
              
              WK=zeros(sum(nl));
              
              iter=0;
              
              while (abs(lp_new - lp_old) > tol && iter < maxiter)
                iter = iter + 1;
                lp_old = lp_new; a_old = a;
                
                
                % b = W*f - d(log(p(y|f)))/df
                if isfield(gp.lik,'xtime')
                  b=Wdiag.*f+[Wmat*f((ntime+1):end); Wmat'*f(1:ntime)]+dlp;
                else
                  b = sum(Wvec.*repmat(reshape(f,n,nlp),nlp,1),2)+dlp;
                end
                
                WK(1:nl(1),1:nl(1))=bsxfun(@times, Wdiag(1:nl(1)),K(1:nl(1),1:nl(1)));
                WK(nl(1)+(1:nl(2)),nl(1)+(1:nl(2)))=bsxfun(@times, Wdiag(nl(1)+(1:nl(2))),K(nl(1)+(1:nl(2)),nl(1)+(1:nl(2))));
                if isfield(gp.lik,'xtime')
                  WK(1:nl(1),nl(1)+(1:nl(2)))=Wmat*K(nl(1)+(1:nl(2)),nl(1)+(1:nl(2)));
                  WK(nl(1)+(1:nl(2)),1:nl(1))=Wmat'*K(1:nl(1),1:nl(1));
                else
                  WK(1:nl(1),nl(1)+(1:nl(2)))=bsxfun(@times, Wvec(1:nl(1),2),K(nl(1)+(1:nl(2)),nl(1)+(1:nl(2))));
                  WK(nl(1)+(1:nl(2)),1:nl(1))=bsxfun(@times, Wvec(nl(1)+(1:nl(2)),1),K(1:nl(1),1:nl(1)));
                end

                % B = I + WK
                B=WK;
                B(1:sum(nl)+1:end) = B(1:sum(nl)+1:end) + 1;
                [ll,uu]=lu(B);
                
                % a = inv(I+WK)*(W*f - d(log(p(y|f)))/df)
                a=uu\(ll\b);
%                 a=B\b;
                
                f = K*a;                
                
                lp = gp.lik.fh.ll(gp.lik, y, f, z);
                if ~isfield(gp,'meanf')
                  lp_new = -a'*f/2 + lp;
                else
                  %lp_new = -(f-H'*b_m)'*(a-K\(H'*b_m))/2 + lp; %f^=f-H'*b_m,
                  lp_new = -(f-Hb_m)'*(a-iKHb_m)/2 + lp; %f^=f-Hb_m,
                end
                i = 0;
                while i < 10 && lp_new < lp_old && ~isnan(sum(f))
                  % reduce step size by half
                  a = (a_old+a)/2;
                  f = K*a;
                  lp = gp.lik.fh.ll(gp.lik, y, f, z);
                  if ~isfield(gp,'meanf')
                    lp_new = -a'*f/2 + lp;
                  else
                    %lp_new = -(f-H'*b_m)'*(a-K\(H'*b_m))/2 + lp;
                    lp_new = -(f-Hb_m)'*(a-iKHb_m)/2 + lp;
                  end
                  i = i+1;
                end
                
                if isfield(gp.lik,'xtime')
                  [llg2diag, llg2mat] = gp.lik.fh.llg2(gp.lik, y, f, 'latent', z);
                  Wdiag=-llg2diag; Wmat=-llg2mat;
                else
                  Wvec = -gp.lik.fh.llg2(gp.lik, y, f, 'latent', z);
                  Wdiag=[Wvec(1:nl(1),1); Wvec(nl(1)+(1:nl(2)),2)];
                end
                dlp = gp.lik.fh.llg(gp.lik, y, f, 'latent', z);
              end
              
              % evaluate the approximate log marginal likelihood

              if isfield(gp.lik,'xtime')
                [llg2diag, llg2mat] = gp.lik.fh.llg2(gp.lik, y, f, 'latent', z);
                Wdiag=-llg2diag; Wmat=-llg2mat;
              else
                Wvec = -gp.lik.fh.llg2(gp.lik, y, f, 'latent', z);
                Wdiag=[Wvec(1:nl(1),1); Wvec(nl(1)+(1:nl(2)),2)];
              end
              
              if ~isfield(gp,'meanf')
                logZ = 0.5 *f'*a - gp.lik.fh.ll(gp.lik, y, f, z);
              else
                % logZ = 0.5 *((f-H'*b_m)'*(a-K\(H'*b_m))) - gp.lik.fh.ll(gp.lik, y, f, z);
                logZ = 0.5 *((f-Hb_m)'*(a-iKHb_m)) - gp.lik.fh.ll(gp.lik, y, f, z);
              end
              
              WK(1:nl(1),1:nl(1))=bsxfun(@times, Wdiag(1:nl(1)),K(1:nl(1),1:nl(1)));
              WK(nl(1)+(1:nl(2)),nl(1)+(1:nl(2)))=bsxfun(@times, Wdiag(nl(1)+(1:nl(2))),K(nl(1)+(1:nl(2)),nl(1)+(1:nl(2))));
              if isfield(gp.lik,'xtime')
                WK(1:ntime,ntime+(1:n))=Wmat*K(nl(1)+(1:nl(2)),nl(1)+(1:nl(2)));
                WK(nl(1)+(1:nl(2)),1:nl(1))=Wmat'*K(1:nl(1),1:nl(1));
              else
                WK(1:nl(1),nl(1)+(1:nl(2)))=bsxfun(@times, Wvec(1:nl(1),2),K(nl(1)+(1:nl(2)),nl(1)+(1:nl(2))));
                WK(nl(1)+(1:nl(2)),1:nl(1))=bsxfun(@times, Wvec(nl(1)+(1:nl(2)),1),K(1:nl(2),1:nl(2)));
              end

              % B = I + WK
              B=WK;
              B(1:sum(nl)+1:end) = B(1:sum(nl)+1:end) + 1;
              
              [Ll,Lu]=lu(B);
              edata = logZ + 0.5*det(Ll)*prod(sign(diag(Lu))).*sum(log(abs(diag(Lu))));
            
              % Return help parameters for gradient and prediction
              % calculations
              L=B;
              E=Ll;
              M=Lu;
          end
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
      ch.n = size(x,1);
      ch.a = a;
      ch.p=p;
      ch.datahash=datahash;
      
    end
  end

  function [edata,e,eprior,f,L,a,E,M,p,ch] = set_output_for_notpositivedefinite()
    % Instead of stopping to chol error, return NaN
    edata=NaN;
    e=NaN;
    eprior=NaN;
    f=NaN;
    L=NaN;
    a=NaN;
    E = NaN;
    M = NaN;
    p=NaN;
    datahash = NaN;
    w = NaN;
    ch.w = w;
    ch.e = e;
    ch.edata = edata;
    ch.eprior = eprior;
    ch.f = f;
    ch.L = L;
    ch.M = M;
    ch.E = E;
    ch.a = a;
    ch.p=p;
    ch.datahash=datahash;
  end
end

