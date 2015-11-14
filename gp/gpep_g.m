function [g, gdata, gprior] = gpep_g(w, gp, x, y, varargin)
%GPEP_G  Evaluate gradient of EP's marginal log posterior estimate (GPEP_E)
%
%  Description
%    G = GPEP_G(W, GP, X, Y, OPTIONS) takes a full GP parameter
%    vector W, structure GP, a matrix X of input vectors and a
%    matrix Y of target vectors, and evaluates the gradient G of
%    EP's marginal log posterior estimate (gpep_e). Each row of X
%    corresponds to one input vector and each row of Y corresponds
%    to one target vector.
%
%    [G, GDATA, GPRIOR] = GPEP_G(GP, X, Y, OPTIONS) also returns
%    separately the data and prior contributions to the gradient.
%
%    OPTIONS is optional parameter-value pair
%      z - optional observed quantity in triplet (x_i,y_i,z_i)
%          Some likelihoods may use this. For example, in case of
%          Poisson likelihood we have z_i=E_i, that is, expected
%          value for ith case.
%
%  See also
%    GP_SET, GP_G, GPEP_E, GPEP_PRED
%
% Copyright (c) 2007-2010  Jarno Vanhatalo
% Copyright (c) 2010       Heikki Peura
% Copyright (c) 2010,2013  Aki Vehtari

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

  ip=inputParser;
  ip.FunctionName = 'GPEP_G';
  ip.addRequired('w', @(x) isvector(x) && isreal(x) && all(isfinite(x)));
  ip.addRequired('gp',@isstruct);
  ip.addRequired('x', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
  ip.addRequired('y', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
  ip.addParamValue('z', [], @(x) isreal(x) && all(isfinite(x(:))))
  ip.addOptional('method', '1', @(x) ismember(x,{'1','2'}))
  ip.parse(w, gp, x, y, varargin{:});
  z=ip.Results.z;
  method = ip.Results.method;

  gp=gp_unpak(gp, w);       % unpak the parameters
  [tmp,tmp,hier]=gp_pak(gp);% Hierarchy of the parameters
  ncf = length(gp.cf);

  g = [];
  gdata = [];
  gprior = [];
  gdata_inducing = [];
  gprior_inducing = [];
  sigm2_i=[];

  if isfield(gp.lik, 'int_magnitude') && gp.lik.int_magnitude
    int_magnitude=true;
    gp=gp_unpak(gp, [0 w(2:end)]);
  else
    int_magnitude=false;
  end
  if isfield(gp.lik, 'int_likparam') && gp.lik.int_likparam
    int_likparam=true;
  else
    int_likparam=false;
  end
  
  if (int_likparam && gp.lik.inputparam) || (int_magnitude && gp.lik.inputmagnitude) ...
        || (isfield(gp.lik, 'int_likparam') && isfield(gp, 'comp_cf'))
    ncf=length(gp.comp_cf{1});
  end
  
  if (isfield(gp,'savememory') && gp.savememory) || int_magnitude
    savememory=1;
  else
    savememory=0;
  end
  [n,nout] = size(y);

  if isfield(gp.lik, 'nondiagW')
    if isfield(gp, 'comp_cf')  % own covariance for each ouput component
      multicf = true;
      if length(gp.comp_cf) ~= nout
        error('GPEP_G: the number of component vectors in gp.comp_cf must be the same as number of outputs.')
      end
    else
      multicf = false;
    end

    % First Evaluate the data contribution to the error
    switch gp.type
      % ============================================================
      % FULL
      % ============================================================
      case 'FULL'   % A full GP
                    % Calculate covariance matrix and the site parameters
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

        [e, edata, eprior, param]= gpep_e(w, gp, x, y, 'z', z);
        tautilde=param.tautilde;
        nutilde=param.nutilde;
        BKnu=param.BKnu;
        B=param.B;
        cholP=param.cholP;
        invPBKnu=param.invPBKnu;

        M=zeros(n,n,nout);

        % full ep with non-diagonal site covariances
        b=zeros(n*nout,1);
        for k1=1:nout
          b((1:n)+(k1-1)*n)=nutilde(:,k1)-(BKnu(:,k1)-B(:,:,k1)*invPBKnu);
        end

        invcholPB=zeros(n,n,nout);
        for k1=1:nout
          invcholPB(:,:,k1)=cholP\B(:,:,k1);
        end

        for k1=1:nout
          M(:,:,k1)=B(:,:,k1)-invcholPB(:,:,k1)'*invcholPB(:,:,k1);
        end

        % =================================================================
        % Gradient with respect to covariance function parameters
        if ~isempty(strfind(gp.infer_params, 'covariance'))
          % Evaluate the gradients from covariance functions
          for i=1:ncf
            i1=0;
            if ~isempty(gprior)
              i1 = length(gprior);
            end

            % Gradients from covariance functions
            gpcf = gp.cf{i};
            DKff = feval(gpcf.fh.cfg, gpcf, x);
            gprior_cf = -feval(gpcf.fh.lpg, gpcf);

            % check in which components the covariance function is present
            do = false(nout,1);
            if multicf
              for z1=1:nout
                if any(gp.comp_cf{z1}==i)
                  do(z1) = true;
                end
              end
            else
              do = true(nout,1);
            end

            for i2 = 1:length(DKff)
              i1 = i1+1;

              Cdl=0;
              DKffb=zeros(n*nout,1);
              for z1=1:nout
                if do(z1)
                  DKffb((1:n)+(z1-1)*n)=DKff{i2}*b((1:n)+(z1-1)*n);
                  Cdl = Cdl + sum(sum( M(:,:,z1) .* DKff{i2} ));
                end
              end
              Bdl = b'*DKffb;

              gdata(i1)=0.5.*(Cdl - Bdl);
              gprior(i1) = gprior_cf(i2);
            end

            % Set the gradients of hyperparameter
            if length(gprior_cf) > length(DKff)
              for i2=length(DKff)+1:length(gprior_cf)
                i1 = i1+1;
                gdata(i1) = 0;
                gprior(i1) = gprior_cf(i2);
              end
            end
          end
        end

        % =================================================================
        % Gradient with respect to likelihood function parameters
        if ~isempty(strfind(gp.infer_params, 'likelihood')) && isfield(gp.lik.fh, 'siteDeriv')
          [Ef, Varf] = gpep_pred(gp, x, y, x, 'z', z);

          sigm2_i = (Varf.^-1 - tautilde).^-1;
          mu_i = sigm2_i.*(Ef./Varf - nutilde);

          gdata_lik = 0;
          lik = gp.lik;
          for k1 = 1:length(y)
            gdata_lik = gdata_lik - feval(lik.fh.siteDeriv, lik, y, k1, sigm2_i(k1), mu_i(k1), z);
          end
          % evaluate prior contribution for the gradient
          if isfield(gp.lik, 'p')
            g_logPrior = -feval(lik.fh.lpg, lik);
          else
            g_logPrior = zeros(size(gdata_lik));
          end
          % set the gradients into vectors that will be returned
          gdata = [gdata gdata_lik];
          gprior = [gprior g_logPrior];
          i1 = length(gdata);
        end

      case {'FIC'}
        % ============================================================
        % FIC
        % ============================================================

      case {'PIC' 'PIC_BLOCK'}
        % ============================================================
        % PIC
        % ============================================================

      case {'CS+FIC'}
        % ============================================================
        % CS+FIC
        % ============================================================

      case {'DTC' 'SOR'}
        % ============================================================
        % DTC/SOR
        % ============================================================

      case 'VAR'
        % ============================================================
        % VAR

    end

    g = gdata + gprior;

  else

    % First Evaluate the data contribution to the error
    switch gp.type
      % ============================================================
      % FULL
      % ============================================================
      case 'FULL'   % A full GP
                    % Calculate covariance matrix and the site parameters

        if isfield(gp.lik, 'int_magnitude')
        
        if (int_likparam && gp.lik.inputparam) || (int_magnitude && gp.lik.inputmagnitude) ...
                    || (isfield(gp.lik, 'int_likparam') && isfield(gp, 'comp_cf'))
          [K, C] = gp_trcov(gp,x,gp.comp_cf{1});
        else
          [K, C] = gp_trcov(gp,x);
        end

        [e, edata, eprior, p] = gpep_e(w, gp, x, y, 'z', z);
        [tautildee, nutildee, mu_ii, sigm2_ii, Z_i, eta] = ...
            deal(p.tautilde, p.nutilde, p.muvec_i, p.sigm2vec_i, p.logZ_i, p.eta);
        tautilde=tautildee(:,1);
        nutilde=nutildee(:,1);
        mu_i=mu_ii(:,1);
        sigm2_i=sigm2_ii(:,1);
        
        if issparse(C)
          % If compact support covariance functions are used
          % the covariance matrix will be sparse
          %[e, edata, eprior, tautilde, nutilde, LD, tmp, tmp, mu_i, sigm2_i, Z_i, eta] = gpep_e(w, gp, x, y, 'z', z);
          LD=p.L;

          Stildesqroot = sparse(1:n,1:n,sqrt(tautilde),n,n);

          b = nutilde - Stildesqroot*ldlsolve(LD,Stildesqroot*(C*nutilde));
          % evaluate the sparse inverse
          invC = spinv(LD,1);
          invC = Stildesqroot*invC*Stildesqroot;
        else
          %[e, edata, eprior, tautilde, nutilde, L, tmp, tmp, mu_i, sigm2_i, Z_i, eta] = gpep_e(w, gp, x, y, 'z', z);
          L=p.L;

          if ~int_likparam && ~int_magnitude && all(tautilde > 0) && ~isequal(gp.latent_opt.optim_method, 'robust-EP')
            % This is the usual case where likelihood is log concave
            % for example, Poisson and probit
            % Stildesqroot=diag(sqrt(tautilde));
            % logZep; nutilde; tautilde;
            % b=nutilde-Stildesqroot*(L'\(L\(Stildesqroot*(C*nutilde))));
            % invC = Stildesqroot*(L'\(L\Stildesqroot));
            Stildesqroot=sqrt(tautilde);
            temp=L\diag(Stildesqroot);
            invC = temp'*temp;
            b=nutilde-Stildesqroot.*(L'\(L\(Stildesqroot.*(C*nutilde))));
          elseif isfield(gp.lik, 'int_likparam') || isequal(gp.latent_opt.optim_method, 'robust-EP')

            A=bsxfun(@times,tautilde,L');   % Sf = L'*L;
            b=nutilde-A*(L*nutilde);        % (eye(n)-diag(tautilde)*Sf)\nutilde
            A=-A*A';                         % = -diag(tautilde)*Sf*diag(tautilde)
            A(1:n+1:end)=A(1:n+1:end)+tautilde'; % diag(tautilde)-diag(tautilde)*Sf*diag(tautilde)
            invC = A;

          else
            % We might end up here if the likelihood is not log concave
            % For example Student-t likelihood.
            % NOTE! This does not work reliably yet
            S = diag(tautilde);
            b = nutilde - tautilde.*(L'*L*(nutilde));
            invC = S*L';
            invC = S - invC*invC';
          end

        end

        if ~all(isfinite(e));
          % instead of stopping to error, return NaN
          g=NaN;
          gdata = NaN;
          gprior = NaN;
          return;
        end

        % =================================================================
        % Gradient with respect to covariance function parameters
        if ~isempty(strfind(gp.infer_params, 'covariance'))
          % Evaluate the gradients from covariance functions
          for i=1:ncf
            i1=0;
            if ~isempty(gprior)
              i1 = length(gprior);
            end
            gpcf = gp.cf{i};
            if savememory
              np=gpcf.fh.cfg(gpcf,[],[],[],0);
              if int_magnitude && (~isfield(gp,'comp_cf') || ...
                  (isfield(gp,'comp_cf') && sum(gp.comp_cf{1}==i)>0))
                nps=2:np;
                i1=i1+1;
              else
                nps=1:np;
              end
            else
              DKffc = gpcf.fh.cfg(gpcf, x);
              nps=1:length(DKffc);                
            end
            gprior_cf = -gpcf.fh.lpg(gpcf);

            if ~isfield(gp,'meanf')
              for i2 = nps
                if savememory
                  DKff=gpcf.fh.cfg(gpcf,x,[],[],i2);
                else
                  DKff=DKffc{i2};
                end
                i1 = i1+1;
                Bdl = b'*(DKff*b);
                Cdl = sum(sum(invC.*DKff)); % help arguments for lengthScale
                gdata(i1)=0.5.*(Cdl - Bdl);
                gprior(i1) = gprior_cf(i2);
              end
            else
              i1=0;
              Stildesqroot=diag(sqrt(tautilde));
              invKs=eye(size(C))-Stildesqroot*(L'\(L\(Stildesqroot*C)));
              [dMNM trA]=mean_gf(gp,x,C,invKs,DKff,Stildesqroot,nutilde,'EP');
              for i2 = 1:np
                i1=i1+1;
                if savememory
                  DKff=gpcf.fh.cfg(gpcf,x,[],[],i2);
                else
                  DKff=DKffc{i2};
                end
                trK=sum(sum(invC.*DKff));
                gdata(i2)=0.5*(-1*dMNM + trK + trA{i2});
                gprior(i1) = gprior_cf(i2);
              end
            end

            % Set the gradients of hyperparameter
            if length(gprior_cf) > nps(end)
              for i2=np+1:length(gprior_cf)
                i1 = i1+1;
                gdata(i1) = 0;
                gprior(i1) = gprior_cf(i2);
              end
            end
          end
        end
        if int_magnitude
          gdata(1)=0; gprior(1)=0;
        end
        if int_likparam && gp.lik.inputparam
          [K, C] = gp_trcov(gp,x, gp.comp_cf{2});

          ncf2=length(gp.comp_cf{2});
            
          tautilde=tautildee(:,2);
          nutilde=nutildee(:,2);
          mu_i=mu_ii(:,2);
          sigm2_i=sigm2_ii(:,2);
          L=p.La2;

%           Stildesqroot=sqrt(tautilde);
%           temp=L\diag(Stildesqroot);
%           invC = temp'*temp;
%           b=nutilde-Stildesqroot.*(L'\(L\(Stildesqroot.*(C*nutilde))));
          
          A=bsxfun(@times,tautilde,L');   % Sf = L'*L;
          b=nutilde-A*(L*nutilde);        % (eye(n)-diag(tautilde)*Sf)\nutilde
          A=-A*A';                         % = -diag(tautilde)*Sf*diag(tautilde)
          A(1:n+1:end)=A(1:n+1:end)+tautilde'; % diag(tautilde)-diag(tautilde)*Sf*diag(tautilde)
          invC = A;
          
          
          % =================================================================
          % Gradient with respect to covariance function parameters
          if ~isempty(strfind(gp.infer_params, 'covariance'))
            % Evaluate the gradients from covariance functions
            for i=(1:ncf2)+ncf
              gpcf = gp.cf{i};
              if savememory
                np=gpcf.fh.cfg(gpcf,[],[],[],0);
              else
                DKffc = gpcf.fh.cfg(gpcf, x);
                np=length(DKffc);
              end
              gprior_cf = -gpcf.fh.lpg(gpcf);
              
              if ~isfield(gp,'meanf')
                for i2 = 1:np
                  if savememory
                    DKff=gpcf.fh.cfg(gpcf,x,[],[],i2);
                  else
                    DKff=DKffc{i2};
                  end
                  i1 = i1+1;
                  Bdl = b'*(DKff*b);
                  Cdl = sum(sum(invC.*DKff)); % help arguments for lengthScale
                  gdata(i1)=0.5.*(Cdl - Bdl);
                  gprior(i1) = gprior_cf(i2);
                end
              else
                i1=0;
                Stildesqroot=diag(sqrt(tautilde));
                invKs=eye(size(C))-Stildesqroot*(L'\(L\(Stildesqroot*C)));
                [dMNM trA]=mean_gf(gp,x,C,invKs,DKff,Stildesqroot,nutilde,'EP');
                for i2 = 1:np
                  i1=i1+1;
                  if savememory
                    DKff=gpcf.fh.cfg(gpcf,x,[],[],i2);
                  else
                    DKff=DKffc{i2};
                  end
                  trK=sum(sum(invC.*DKff));
                  gdata(i2)=0.5*(-1*dMNM + trK + trA{i2});
                  gprior(i1) = gprior_cf(i2);
                end
              end
              
              % Set the gradients of hyperparameter
              if length(gprior_cf) > np
                for i2=np+1:length(gprior_cf)
                  i1 = i1+1;
                  gdata(i1) = 0;
                  gprior(i1) = gprior_cf(i2);
                end
              end
            end
          end
        end
        
        if int_magnitude && gp.lik.inputmagnitude
          [K, C] = gp_trcov(gp,x, gp.comp_cf{end});
          
          ncf2=length(gp.comp_cf{end});
          
          tautilde=tautildee(:,end);
          nutilde=nutildee(:,end);
          mu_i=mu_ii(:,end);
          sigm2_i=sigm2_ii(:,end);
          L=p.La3;
          
          %           Stildesqroot=sqrt(tautilde);
          %           temp=L\diag(Stildesqroot);
          %           invC = temp'*temp;
          %           b=nutilde-Stildesqroot.*(L'\(L\(Stildesqroot.*(C*nutilde))));
          
          A=bsxfun(@times,tautilde,L');   % Sf = L'*L;
          b=nutilde-A*(L*nutilde);        % (eye(n)-diag(tautilde)*Sf)\nutilde
          A=-A*A';                         % = -diag(tautilde)*Sf*diag(tautilde)
          A(1:n+1:end)=A(1:n+1:end)+tautilde'; % diag(tautilde)-diag(tautilde)*Sf*diag(tautilde)
          invC = A;
          
          
          % =================================================================
          % Gradient with respect to covariance function parameters
          if ~isempty(strfind(gp.infer_params, 'covariance'))
            % Evaluate the gradients from covariance functions
            for i=length(gp.cf)-ncf2+1:length(gp.cf)
              gpcf = gp.cf{i};
              if savememory
                np=gpcf.fh.cfg(gpcf,[],[],[],0);
              else
                DKffc = gpcf.fh.cfg(gpcf, x);
                np=length(DKffc);
              end
              gprior_cf = -gpcf.fh.lpg(gpcf);
              
              if ~isfield(gp,'meanf')
                for i2 = 1:np
                  if savememory
                    DKff=gpcf.fh.cfg(gpcf,x,[],[],i2);
                  else
                    DKff=DKffc{i2};
                  end
                  i1 = i1+1;
                  Bdl = b'*(DKff*b);
                  Cdl = sum(sum(invC.*DKff)); % help arguments for lengthScale
                  gdata(i1)=0.5.*(Cdl - Bdl);
                  gprior(i1) = gprior_cf(i2);
                end
              else
                i1=0;
                Stildesqroot=diag(sqrt(tautilde));
                invKs=eye(size(C))-Stildesqroot*(L'\(L\(Stildesqroot*C)));
                [dMNM trA]=mean_gf(gp,x,C,invKs,DKff,Stildesqroot,nutilde,'EP');
                for i2 = 1:np
                  i1=i1+1;
                  if savememory
                    DKff=gpcf.fh.cfg(gpcf,x,[],[],i2);
                  else
                    DKff=DKffc{i2};
                  end
                  trK=sum(sum(invC.*DKff));
                  gdata(i2)=0.5*(-1*dMNM + trK + trA{i2});
                  gprior(i1) = gprior_cf(i2);
                end
              end
              
              % Set the gradients of hyperparameter
              if length(gprior_cf) > np
                for i2=np+1:length(gprior_cf)
                  i1 = i1+1;
                  gdata(i1) = 0;
                  gprior(i1) = gprior_cf(i2);
                end
              end
            end
          end
        end
        
        else
        
        [K, C] = gp_trcov(gp,x);
        [e, edata, eprior, p] = gpep_e(w, gp, x, y, 'z', z);
        [tautilde, nutilde, mu_i, sigm2_i, Z_i, eta] = ...
            deal(p.tautilde, p.nutilde, p.muvec_i, p.sigm2vec_i, p.logZ_i, p.eta);
        L=p.L;
        if isfield(gp, 'lik_mono')
          x2=x;
          x=gp.xv;
          [K,C]=gp_dtrcov(gp,x2,x);
          n=size(K,1);
          C=K;
          L=p.La2;
          n2=size(y,1);
%           L=chol(p.Sigma);
        end
        if issparse(C)
          % If compact support covariance functions are used
          % the covariance matrix will be sparse
          %[e, edata, eprior, tautilde, nutilde, LD, tmp, tmp, mu_i, sigm2_i, Z_i, eta] = gpep_e(w, gp, x, y, 'z', z);
          LD=p.L;

          Stildesqroot = sparse(1:n,1:n,sqrt(tautilde),n,n);

          b = nutilde - Stildesqroot*ldlsolve(LD,Stildesqroot*(C*nutilde));
          % evaluate the sparse inverse
          invC = spinv(LD,1);
          invC = Stildesqroot*invC*Stildesqroot;
        else
          %[e, edata, eprior, tautilde, nutilde, L, tmp, tmp, mu_i, sigm2_i, Z_i, eta] = gpep_e(w, gp, x, y, 'z', z);

          if all(tautilde > 0) && ~(isequal(gp.latent_opt.optim_method, 'robust-EP') ...
            || (isfield(gp, 'lik_mono')))  %&& isequal(gp.lik_mono.type, 'Gaussian')))
            % This is the usual case where likelihood is log concave
            % for example, Poisson and probit
            % Stildesqroot=diag(sqrt(tautilde));
            % logZep; nutilde; tautilde;
            % b=nutilde-Stildesqroot*(L'\(L\(Stildesqroot*(C*nutilde))));
            % invC = Stildesqroot*(L'\(L\Stildesqroot));
            Stildesqroot=sqrt(tautilde);
            temp=L\diag(Stildesqroot);
            invC = temp'*temp;
            b=nutilde-Stildesqroot.*(L'\(L\(Stildesqroot.*(C*nutilde))));
          elseif isequal(gp.latent_opt.optim_method, 'robust-EP') || (isfield(gp, 'lik_mono')) ...
              %&& isequal(gp.lik_mono.type, 'Gaussian'))

            A=bsxfun(@times,tautilde,L');   % Sf = L'*L;
            b=nutilde-A*(L*nutilde);        % (eye(n)-diag(tautilde)*Sf)\nutilde
            A=-A*A';                         % = -diag(tautilde)*Sf*diag(tautilde)
            A(1:n+1:end)=A(1:n+1:end)+tautilde'; % diag(tautilde)-diag(tautilde)*Sf*diag(tautilde)
            invC = A;

          else
            % We might end up here if the likelihood is not log concave
            % For example Student-t likelihood.
            % NOTE! This does not work reliably yet
            S = diag(tautilde);
            b = nutilde - tautilde.*(L'*L*(nutilde));
            invC = S*L';
            invC = S - invC*invC';
          end

        end

        if ~all(isfinite(e));
          % instead of stopping to error, return NaN
          g=NaN;
          gdata = NaN;
          gprior = NaN;
          return;
        end

        % =================================================================
        % Gradient with respect to covariance function parameters
        if ~isempty(strfind(gp.infer_params, 'covariance'))
          % Evaluate the gradients from covariance functions
          i1=0;
          for i=1:ncf
            gpcf = gp.cf{i};
            if isfield(gp, 'lik_mono')
              savememory=0;
              [n, m]=size(x);
              DKffa = gpcf.fh.cfg(gpcf, x2);
              if ~isempty(DKffa)
                DKdf = gpcf.fh.cfdg(gpcf, x, x2);
                DKdd = gpcf.fh.cfdg2(gpcf, x);
                % Select monotonic dimensions
                inds=[];
                nvd=abs(gp.nvd);
                if isfield(gpcf,'selectedVariables')
                    [~,nvd]=ismember(nvd,gpcf.selectedVariables);
                    nvd=nvd(logical(nvd));
                end
                for idd=1:length(nvd)
                  inds=[inds size(x,1)*(nvd(idd)-1)+1:size(x,1)*nvd(idd)];
                end
                for ijj=1:length(DKffa)
                  DKdf{ijj}=DKdf{ijj}(inds,:);
                  DKdd{ijj}=DKdd{ijj}(inds,inds);
                end
                
                DKffc{1}=[DKffa{1} DKdf{1}';DKdf{1} DKdd{1}];
                for i2=2:length(DKffa)
                  DKffc{i2}=[DKffa{i2} DKdf{i2}';DKdf{i2} DKdd{i2}];
                end
              end
              np=length(DKffa);
            else
              if savememory
                np=gpcf.fh.cfg(gpcf,[],[],[],0);
              else
                DKffc = gpcf.fh.cfg(gpcf, x);
                np=length(DKffc);
              end
            end
            gprior_cf = -gpcf.fh.lpg(gpcf);

            if ~isfield(gp,'meanf')
              for i2 = 1:np
                if savememory
                  DKff=gpcf.fh.cfg(gpcf,x,[],[],i2);
                else
                  DKff=DKffc{i2};
                end
                i1 = i1+1;
                Bdl = b'*(DKff*b);
                Cdl = sum(sum(invC.*DKff)); % help arguments for lengthScale
                gdata(i1)=0.5.*(Cdl - Bdl);
              end
            else
              Stildesqroot=diag(sqrt(tautilde));
              invKs=eye(size(C))-Stildesqroot*(L'\(L\(Stildesqroot*C)));
              if ~savememory
                [dMNMc trAc]=mean_gf(gp,x,C,invKs,DKffc,Stildesqroot,nutilde,'EP');
              end
              for i2 = 1:np
                i1=i1+1;
                if savememory
                  DKff=gpcf.fh.cfg(gpcf,x,[],[],i2);
                  [dMNM trA]=mean_gf(gp,x,C,invKs,{DKff},Stildesqroot,nutilde,'EP');
                  trA=trA{1};
                  dMNM=dMNM{1};
                else
                  DKff=DKffc{i2};
                  trA=trAc{i2};
                  dMNM=dMNMc{i2};
                end
                trK=sum(sum(invC.*DKff));
                gdata(i1)=0.5*(-1*dMNM + trK + trA);
              end
            end

            gprior = [gprior gprior_cf];
          end
        end
        
        end
        
      case {'FIC'}
        % ============================================================
        % FIC
        % ============================================================
        u = gp.X_u;
        DKuu_u = 0;
        DKuf_u = 0;

        %[e, edata, eprior, tautilde, nutilde, L, La, b, mu_i, sigm2_i, Z_i, eta] = gpep_e(w, gp, x, y, 'z', z);
        [e, edata, eprior, p] = gpep_e(w, gp, x, y, 'z', z);
        [tautilde, nutilde, L, La, b, mu_i, sigm2_i, Z_i, eta] = ...
            deal(p.tautilde, p.nutilde, p.L, p.La2, p.b, p.muvec_i, p.sigm2vec_i, p.logZ_i, p.eta);

        K_fu = gp_cov(gp, x, u);         % f x u
        K_uu = gp_trcov(gp, u);          % u x u, noiseles covariance K_uu
        K_uu = (K_uu+K_uu')./2;          % ensure the symmetry of K_uu
        iKuuKuf = K_uu\K_fu';

        LL = sum(L.*L,2);

        if isequal(gp.latent_opt.optim_method, 'robust-EP')
          % Help parameters for Robust-EP
          S = 1+tautilde.*b;
          WiS = tautilde./S;
        end

        if ~all(isfinite(e));
          % instead of stopping to error, return NaN
          g=NaN;
          gdata = NaN;
          gprior = NaN;
          return;
        end

        % =================================================================
        % Gradient with respect to covariance function parameters
        if ~isempty(strfind(gp.infer_params, 'covariance'))
          i1=0;
          for i=1:ncf
            gpcf = gp.cf{i};
            % Get the gradients of the covariance matrices
            % and gprior from gpcf_* structures
            gpcf = gp.cf{i};
            if savememory
              % If savememory option is used, just get the number of
              % hyperparameters and calculate gradients later
              np=gpcf.fh.cfg(gpcf,[],[],[],0);
            else
              DKffc = gpcf.fh.cfg(gpcf, x, [], 1);
              DKuuc = gpcf.fh.cfg(gpcf, u);
              DKufc = gpcf.fh.cfg(gpcf, u, x);
              np=length(DKuuc);
            end
            gprior_cf = -gpcf.fh.lpg(gpcf);

            if ~isequal(gp.latent_opt.optim_method, 'robust-EP')

              for i2 = 1:np
                i1 = i1+1;
                if savememory
                  DKff=gpcf.fh.cfg(gpcf,x,[],1,i2);
                  DKuu=gpcf.fh.cfg(gpcf,u,[],[],i2);
                  DKuf=gpcf.fh.cfg(gpcf,u,x,[],i2);
                else
                  DKff=DKffc{i2};
                  DKuu=DKuuc{i2};
                  DKuf=DKufc{i2};
                end
                KfuiKuuKuu = iKuuKuf'*DKuu;
                gdata(i1) = -0.5.*((2*b*DKuf'-(b*KfuiKuuKuu))*(iKuuKuf*b') + 2.*sum(sum(L'.*(L'*DKuf'*iKuuKuf))) - ...
                                   sum(sum(L'.*((L'*KfuiKuuKuu)*iKuuKuf))));

                gdata(i1) = gdata(i1) - 0.5.*(b.*DKff')*b';
                gdata(i1) = gdata(i1) + 0.5.*(2.*b.*sum(DKuf'.*iKuuKuf',2)'*b'- b.*sum(KfuiKuuKuu.*iKuuKuf',2)'*b');
                gdata(i1) = gdata(i1) + 0.5.*(sum(DKff./La) - sum(LL.*DKff));
                gdata(i1) = gdata(i1) + 0.5.*(2.*sum(LL.*sum(DKuf'.*iKuuKuf',2)) - sum(LL.*sum(KfuiKuuKuu.*iKuuKuf',2)));

              end

            else
              % Robust-EP
              for i2 = 1:np
                i1 = i1+1;

                if savememory
                  DKff=gpcf.fh.cfg(gpcf,x,[],1,i2);
                  DKuu=gpcf.fh.cfg(gpcf,u,[],[],i2);
                  DKuf=gpcf.fh.cfg(gpcf,u,x,[],i2);
                else
                  DKff=DKffc{i2};
                  DKuu=DKuuc{i2};
                  DKuf=DKufc{i2};
                end
                % Evaluate derivative of log(det(I+Ktilde*W)) where Ktilde is
                % FIC sparse approximation of covariance with respect to
                % hyperparameters

                Dd = DKff - 2.*sum(DKuf.*iKuuKuf)' - sum((-iKuuKuf'*DKuu)'.*iKuuKuf)'; % d(diag(Kff - Qff)) / dth
                DS = Dd.*tautilde;
                gdata(i1) = -0.5.*sum(DS./S);
                DTtilde = DKuu + DKuf*bsxfun(@times, WiS, K_fu) - K_fu'*bsxfun(@times, WiS.*DS./S, K_fu) + ...
                          K_fu'*bsxfun(@times, WiS, DKuf');
                gdata(i1) = gdata(i1) - 0.5.*sum(sum(inv(L).*(L\DTtilde)));
                gdata(i1) = gdata(i1) - 0.5.*sum(sum(-inv(La).*(La\DKuu)));
                iSKfuiL = bsxfun(@times, 1./S, K_fu/L');

                % Evaluate derivative of quadratic term
                % nutilde'*sigma^-1*nutilde with respect to hyperparameters

                nud=(nutilde'*iSKfuiL)/L;
                nuDpcovnu = sum(nutilde.^2.*(-DS.*b./S.^2 + Dd./S)) + 2*(nutilde./S)'*(DKuf'*nud') - ...
                    2*((nutilde.*DS./S)'*iSKfuiL)*(iSKfuiL'*nutilde) - nud*DTtilde*nud'; % nutilde'* d(sigma^-1)/dth *nutilde
                gdata(i1) = gdata(i1) + 0.5*nuDpcovnu;
                gdata(i1) = -gdata(i1);
              end

            end
            gprior = [gprior gprior_cf];
          end

        end

        % =================================================================
        % Gradient with respect to inducing inputs

        if ~isempty(strfind(gp.infer_params, 'inducing'))
          if isfield(gp.p, 'X_u') && ~isempty(gp.p.X_u)
            m = size(gp.X_u,2);
            st=0;

            gdata_inducing(st+1:st+length(gp.X_u(:))) = 0;
            i1 = st+1;
            gprior_inducing=[];
            if iscell(gp.p.X_u) % Own prior for each inducing input
              for i = 1:size(gp.X_u,1)
                pr = gp.p.X_u{i};
                gprior_inducing =[gprior_inducing -pr.fh.lpg(gp.X_u(i,:), pr)];
              end
            else % One prior for all inducing inputs
              gprior_inducing = -gp.p.X_u.fh.lpg(gp.X_u(:)', gp.p.X_u);
            end

            for i=1:ncf
              i1=st;

              gpcf = gp.cf{i};
              if savememory
                % If savememory option is used, just get the number of
                % covariates in X and calculate gradients later
                np=gpcf.fh.ginput(gpcf,u,[],0);
              else
                DKuu = gpcf.fh.ginput(gpcf, u);
                DKuf = gpcf.fh.ginput(gpcf, u, x);
                np=1;
              end

              if ~isequal(gp.latent_opt.optim_method, 'robust-EP')
                for i3 = 1:np
                  if savememory
                    DKuu=gpcf.fh.ginput(gpcf,u,[],i3);
                    DKuf=gpcf.fh.ginput(gpcf,u,x,i3);
                  end
                  for i2 = 1:length(DKuu)
                    if savememory
                      i1 = st + (i2-1)*np+i3;
                      %i1 = st + 2*i2 - (i3==1);
                    else
                      i1 = i1+1;
                    end

                    KfuiKuuKuu = iKuuKuf'*DKuu{i2};

                    gdata_inducing(i1) = gdata_inducing(i1) - 0.5.*((2*b*DKuf{i2}'-(b*KfuiKuuKuu))*(iKuuKuf*b') + ...
                                                                    2.*sum(sum(L'.*(L'*DKuf{i2}'*iKuuKuf))) - sum(sum(L'.*((L'*KfuiKuuKuu)*iKuuKuf))));
                    gdata_inducing(i1) = gdata_inducing(i1) + 0.5.*(2.*b.*sum(DKuf{i2}'.*iKuuKuf',2)'*b'- b.*sum(KfuiKuuKuu.*iKuuKuf',2)'*b');
                    gdata_inducing(i1) = gdata_inducing(i1) + 0.5.*(2.*sum(LL.*sum(DKuf{i2}'.*iKuuKuf',2)) - ...
                                                                    sum(LL.*sum(KfuiKuuKuu.*iKuuKuf',2)));
                  end
                end
              else
                for i3 = 1:np
                  if savememory
                    DKuu=gpcf.fh.ginput(gpcf,u,[],i3);
                    DKuf=gpcf.fh.ginput(gpcf,u,x,i3);
                  end
                  % Robust-EP
                  for i2 = 1:length(DKuu)
                    if savememory
                      i1 = st + (i2-1)*np+i3;
                      %i1 = st + 2*i2 - (i3==1);
                    else
                      i1 = i1+1;
                    end
                    Dd = -2.*sum(DKuf{i2}.*iKuuKuf)' - sum((-iKuuKuf'*DKuu{i2})'.*iKuuKuf)';
                    DS = Dd.*tautilde;
                    gdata_inducing(i1) = -0.5.*sum(DS./S);
                    DTtilde = DKuu{i2} + DKuf{i2}*bsxfun(@times, WiS, K_fu) - K_fu'*bsxfun(@times, WiS.*DS./S, K_fu) + ...
                              K_fu'*bsxfun(@times, WiS, DKuf{i2}');
                    gdata_inducing(i1) = gdata_inducing(i1) - 0.5.*sum(sum(inv(L).*(L\DTtilde)));
                    gdata_inducing(i1) = gdata_inducing(i1) - 0.5.*sum(sum(-La^-1.*(La\DKuu{i2})));
                    iSKfuiL = bsxfun(@times, 1./S, K_fu/L');
                    nud=(nutilde'*iSKfuiL)/L;
                    nuDpcovnu = sum(nutilde.^2.*(-DS.*b./S.^2 + Dd./S)) + 2*(nutilde./S)'*(DKuf{i2}'*nud') - 2*((nutilde.*DS./S)'*iSKfuiL)*(iSKfuiL'*nutilde) - nud*DTtilde*nud';
                    gdata_inducing(i1) = gdata_inducing(i1) + 0.5*nuDpcovnu;
                    gdata_inducing(i1) = -gdata_inducing(i1);
                  end
                end
              end
            end
          end
        end

      case {'PIC' 'PIC_BLOCK'}
        % ============================================================
        % PIC
        % ============================================================
        u = gp.X_u;
        ind = gp.tr_index;
        DKuu_u = 0;
        DKuf_u = 0;

        %[e, edata, eprior, tautilde, nutilde, L, La, b] = gpep_e(w, gp, x, y, 'z', z);
        [e, edata, eprior, p] = gpep_e(w, gp, x, y, 'z', z);
        [tautilde, nutilde, L, La, b, eta] = deal(p.tautilde, p.nutilde, p.L, p.La2, p.b, p.eta);

        K_fu = gp_cov(gp, x, u);         % f x u
        K_uu = gp_trcov(gp, u);          % u x u, noiseles covariance K_uu
        K_uu = (K_uu+K_uu')./2;          % ensure the symmetry of K_uu
        iKuuKuf = K_uu\K_fu';

        % =================================================================
        % Gradient with respect to covariance function parameters
        if ~isempty(strfind(gp.infer_params, 'covariance'))

          % Evaluate the gradients from covariance functions
          i1=0;
          for i=1:ncf
            % Get the gradients of the covariance matrices
            % and gprior from gpcf_* structures
            gpcf = gp.cf{i};
            if savememory
              % If savememory option is used, just get the number of
              % hyperparameters and calculate gradients later
              np=gpcf.fh.cfg(gpcf,[],[],[],0);
            else
              DKuuc = gpcf.fh.cfg(gpcf, u);
              DKufc = gpcf.fh.cfg(gpcf, u, x);
              for kk = 1:length(ind)
                DKffc{kk} = gpcf.fh.cfg(gpcf, x(ind{kk},:));
              end
              np=length(DKuuc);
            end
            gprior_cf = -gpcf.fh.lpg(gpcf);

            for i2 = 1:np
              i1 = i1+1;
              if savememory
                DKuu=gpcf.fh.cfg(gpcf,u,[],[],i2);
                DKuf=gpcf.fh.cfg(gpcf,u,x,[],i2);
              else
                DKuu=DKuuc{i2};
                DKuf=DKufc{i2};
              end

              KfuiKuuKuu = iKuuKuf'*DKuu;
              %            H = (2*K_uf'- KfuiKuuKuu)*iKuuKuf;
              % Here we evaluate  gdata = -0.5.* (b*H*b' + trace(L*L'H)
              gdata(i1) = -0.5.*((2*b*DKuf'-(b*KfuiKuuKuu))*(iKuuKuf*b') + 2.*sum(sum(L'.*(L'*DKuf'*iKuuKuf))) - ...
                                 sum(sum(L'.*((L'*KfuiKuuKuu)*iKuuKuf))));

              for kk=1:length(ind)
                if savememory
                  DKff=gpcf.fh.cfg(gpcf, x(ind{kk},:),[],[],i2);
                else
                  DKff=DKffc{kk}{i2};
                end
                gdata(i1) = gdata(i1) ...
                    + 0.5.*(-b(ind{kk})*DKff*b(ind{kk})' ...
                            + 2.*b(ind{kk})*DKuf(:,ind{kk})'*iKuuKuf(:,ind{kk})*b(ind{kk})'- ...
                            b(ind{kk})*KfuiKuuKuu(ind{kk},:)*iKuuKuf(:,ind{kk})*b(ind{kk})' ...
                            + trace(La{kk}\DKff)...
                            - trace(L(ind{kk},:)*(L(ind{kk},:)'*DKff)) ...
                            + 2.*sum(sum(L(ind{kk},:)'.*(L(ind{kk},:)'*DKuf(:,ind{kk})'*iKuuKuf(:,ind{kk})))) - ...
                            sum(sum(L(ind{kk},:)'.*((L(ind{kk},:)'*KfuiKuuKuu(ind{kk},:))*iKuuKuf(:,ind{kk})))));
              end
            end
            gprior=[gprior gprior_cf];
          end

        end

        % =================================================================
        % Gradient with respect to inducing inputs

        if ~isempty(strfind(gp.infer_params, 'inducing'))
          if isfield(gp.p, 'X_u') && ~isempty(gp.p.X_u)
            m = size(gp.X_u,2);

            st=0;
            if ~isempty(gprior_inducing)
              st = length(gprior_inducing);
            end
            gdata_inducing(st+1:st+length(gp.X_u(:))) = 0;

            i1 = st+1;
            gprior_inducing=[];
            if iscell(gp.p.X_u) % Own prior for each inducing input
              for i = 1:size(gp.X_u,1)
                pr = gp.p.X_u{i};
                gprior_inducing =[gprior_inducing -pr.fh.lpg(gp.X_u(i,:), pr)];
              end
            else % One prior for all inducing inputs
              gprior_inducing = -gp.p.X_u.fh.lpg(gp.X_u(:)', gp.p.X_u);
            end

            % Loop over the  covariance functions
            for i=1:ncf
              i1=st;
              gpcf = gp.cf{i};
              if savememory
                % If savememory option is used, just get the number of
                % covariates in X and calculate gradients later
                np=gpcf.fh.ginput(gpcf,u,[],0);
              else
                DKuu = gpcf.fh.ginput(gpcf, u);
                DKuf = gpcf.fh.ginput(gpcf, u, x);
                np=1;
              end

              for i3 = 1:np
                if savememory
                  DKuu=gpcf.fh.ginput(gpcf,u,[],i3);
                  DKuf=gpcf.fh.ginput(gpcf,u,x,i3);
                end
                for i2 = 1:length(DKuu)
                  if savememory
                    i1 = st + (i2-1)*np+i3;
                    %i1 = st + 2*i2 - (i3==1);
                  else
                    i1 = i1+1;
                  end
                  KfuiKuuDKuu_u = iKuuKuf'*DKuu{i2};

                  gdata_inducing(i1) = gdata_inducing(i1) - 0.5.*((2*b*DKuf{i2}'-(b*KfuiKuuDKuu_u))*(iKuuKuf*b') + 2.*sum(sum(L'.*((L'*DKuf{i2}')*iKuuKuf))) - ...
                                                                  sum(sum(L'.*((L'*KfuiKuuDKuu_u)*iKuuKuf))));

                  for kk=1:length(ind)
                    gdata_inducing(i1) = gdata_inducing(i1) + 0.5.*(2.*b(ind{kk})*DKuf{i2}(:,ind{kk})'*iKuuKuf(:,ind{kk})*b(ind{kk})'- ...
                                                                    b(ind{kk})*KfuiKuuDKuu_u(ind{kk},:)*iKuuKuf(:,ind{kk})*b(ind{kk})' ...
                                                                    + 2.*sum(sum(L(ind{kk},:)'.*(L(ind{kk},:)'*DKuf{i2}(:,ind{kk})'*iKuuKuf(:,ind{kk})))) - ...
                                                                    sum(sum(L(ind{kk},:)'.*((L(ind{kk},:)'*KfuiKuuDKuu_u(ind{kk},:))*iKuuKuf(:,ind{kk})))));
                  end
                end
              end
            end
          end
        end


      case {'CS+FIC'}
        % ============================================================
        % CS+FIC
        % ============================================================
        u = gp.X_u;
        DKuu_u = 0;
        DKuf_u = 0;

        %[e, edata, eprior, tautilde, nutilde, L, La, b] = gpep_e(w, gp, x, y, 'z', z);
        [e, edata, eprior, p] = gpep_e(w, gp, x, y, 'z', z);
        [tautilde, nutilde, L, La, b, eta] = deal(p.tautilde, p.nutilde, p.L, p.La2, p.b, p.eta);

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
        K_fu = gp_cov(gp, x, u);         % f x u
        K_uu = gp_trcov(gp, u);    % u x u, noiseles covariance K_uu
        K_uu = (K_uu+K_uu')./2;     % ensure the symmetry of K_uu
        iKuuKuf = K_uu\K_fu';
        gp.cf = cf_orig;

        LD = ldlchol(La);
        siLa = spinv(LD,1);
        idiagLa = diag(siLa);
        LL = sum(L.*L,2);

        % =================================================================
        % Gradient with respect to covariance function parameters
        if ~isempty(strfind(gp.infer_params, 'covariance'))
          i1=0;
          for i=1:ncf
            gpcf = gp.cf{i};

            % Evaluate the gradient for FIC covariance functions
            if ~isfield(gpcf,'cs')
              % Get the gradients of the covariance matrices
              % and gprior from gpcf_* structures
              if savememory
                % If savememory option is used, just get the number of
                % hyperparameters and calculate gradients later
                np=gpcf.fh.cfg(gpcf,[],[],[],0);
              else
                DKffc = gpcf.fh.cfg(gpcf, x, [], 1);
                DKuuc = gpcf.fh.cfg(gpcf, u);
                DKufc = gpcf.fh.cfg(gpcf, u, x);
                np=length(DKuuc);
              end
              gprior_cf = -gpcf.fh.lpg(gpcf);

              for i2 = 1:np
                i1 = i1+1;
                if savememory
                  DKff=gpcf.fh.cfg(gpcf,x,[],1,i2);
                  DKuu=gpcf.fh.cfg(gpcf,u,[],[],i2);
                  DKuf=gpcf.fh.cfg(gpcf,u,x,[],i2);
                else
                  DKff=DKffc{i2};
                  DKuu=DKuuc{i2};
                  DKuf=DKufc{i2};
                end

                KfuiKuuKuu = iKuuKuf'*DKuu;
                gdata(i1) = -0.5.*((2*b*DKuf'-(b*KfuiKuuKuu))*(iKuuKuf*b') + 2.*sum(sum(L'.*(L'*DKuf'*iKuuKuf))) - ...
                                   sum(sum(L'.*((L'*KfuiKuuKuu)*iKuuKuf))));

                gdata(i1) = gdata(i1) - 0.5.*(b.*DKff')*b';
                gdata(i1) = gdata(i1) + 0.5.*(2.*b.*sum(DKuf'.*iKuuKuf',2)'*b'- b.*sum(KfuiKuuKuu.*iKuuKuf',2)'*b');
                gdata(i1) = gdata(i1) + 0.5.*sum(idiagLa.*DKff - LL.*DKff);   % corrected
                gdata(i1) = gdata(i1) + 0.5.*(2.*sum(LL.*sum(DKuf'.*iKuuKuf',2)) - sum(LL.*sum(KfuiKuuKuu.*iKuuKuf',2)));

                %gdata(i1) = gdata(i1) + 0.5.*sum(sum(La\((2.*K_uf') - KfuiKuuKuu).*iKuuKuf',2));
                gdata(i1) = gdata(i1) + 0.5.*sum(sum(ldlsolve(LD,2.*DKuf' - KfuiKuuKuu).*iKuuKuf',2));
                gdata(i1) = gdata(i1) - 0.5.*( idiagLa'*(sum((2.*DKuf' - KfuiKuuKuu).*iKuuKuf',2)) ); % corrected
%                 gprior(i1) = gprior_cf(i2);
              end

              % Evaluate the gradient for compact support covariance functions
            else
              % Get the gradients of the covariance matrices
              % and gprior from gpcf_* structures
              if savememory
                % If savememory option is used, just get the number of
                % hyperparameters and calculate gradients later
                np=gpcf.fh.cfg(gpcf,[],[],[],0);
              else
                DKffc = gpcf.fh.cfg(gpcf, x);
                np=length(DKffc);
              end
              gprior_cf = -gpcf.fh.lpg(gpcf);
              for i2 = 1:np
                if savememory
                  DKff=gpcf.fh.cfg(gpcf,x,[],[],i2);
                else
                  DKff=DKffc{i2};
                end
                i1 = i1+1;
                gdata(i1) = 0.5*(sum(sum(siLa.*DKff',2)) - sum(sum(L.*(L'*DKff')')) - b*DKff*b');
              end
            end
            gprior = [gprior gprior_cf];
          end

        end

        % =================================================================
        % Gradient with respect to inducing inputs

        if ~isempty(strfind(gp.infer_params, 'inducing'))
          if isfield(gp.p, 'X_u') && ~isempty(gp.p.X_u)
            m = size(gp.X_u,2);
            st=0;
            if ~isempty(gprior_inducing)
              st = length(gprior_inducing);
            end

            gdata_inducing(st+1:st+length(gp.X_u(:))) = 0;
            i1 = st+1;
            gprior_inducing=[];
            if iscell(gp.p.X_u) % Own prior for each inducing input
              for i = 1:size(gp.X_u,1)
                pr = gp.p.X_u{i};
                gprior_inducing =[gprior_inducing -pr.fh.lpg(gp.X_u(i,:), pr)];
              end
            else % One prior for all inducing inputs
              gprior_inducing = -gp.p.X_u.fh.lpg(gp.X_u(:)', gp.p.X_u);
            end

            for i=1:ncf
              i1=st;

              gpcf = gp.cf{i};
              if ~isfield(gpcf,'cs')
                if savememory
                  % If savememory option is used, just get the number of
                  % covariates in X and calculate gradients later
                  np=gpcf.fh.ginput(gpcf,u,[],0);
                else
                  DKuu = gpcf.fh.ginput(gpcf, u);
                  DKuf = gpcf.fh.ginput(gpcf, u, x);
                  np=1;
                end

                for i3 = 1:np
                  if savememory
                    DKuu=gpcf.fh.ginput(gpcf,u,[],i3);
                    DKuf=gpcf.fh.ginput(gpcf,u,x,i3);
                  end
                  for i2 = 1:length(DKuu)
                    if savememory
                      i1 = st + (i2-1)*np+i3;
                      %i1 = st + 2*i2 - (i3==1);
                    else
                      i1 = i1+1;
                    end
                    KfuiKuuKuu = iKuuKuf'*DKuu{i2};

                    gdata_inducing(i1) = gdata_inducing(i1) - 0.5.*((2*b*DKuf{i2}'-(b*KfuiKuuKuu))*(iKuuKuf*b') + ...
                                                                    2.*sum(sum(L'.*(L'*DKuf{i2}'*iKuuKuf))) - sum(sum(L'.*((L'*KfuiKuuKuu)*iKuuKuf))));
                    gdata_inducing(i1) = gdata_inducing(i1) + 0.5.*(2.*b.*sum(DKuf{i2}'.*iKuuKuf',2)'*b'- b.*sum(KfuiKuuKuu.*iKuuKuf',2)'*b');
                    gdata_inducing(i1) = gdata_inducing(i1) + 0.5.*(2.*sum(LL.*sum(DKuf{i2}'.*iKuuKuf',2)) - ...
                                                                    sum(LL.*sum(KfuiKuuKuu.*iKuuKuf',2)));

                    gdata_inducing(i1) = gdata_inducing(i1) + 0.5.*sum(sum(ldlsolve(LD,(2.*DKuf{i2}') - KfuiKuuKuu).*iKuuKuf',2));
                    gdata_inducing(i1) = gdata_inducing(i1) - 0.5.*( idiagLa'*(sum((2.*DKuf{i2}' - KfuiKuuKuu).*iKuuKuf',2)) ); % corrected
%                     gprior_inducing(i1) = gprior_inducing_ind(i2);
                  end
                end
              end
            end
          end
        end

      case {'DTC' 'SOR'}
        % ============================================================
        % DTC/SOR
        % ============================================================
        u = gp.X_u;
        DKuu_u = 0;
        DKuf_u = 0;

        %[e, edata, eprior, tautilde, nutilde, L, La, b] = gpep_e(w, gp, x, y, 'z', z);
        [e, edata, eprior, p] = gpep_e(w, gp, x, y, 'z', z);
        [tautilde, nutilde, L, La, b, eta] = deal(p.tautilde, p.nutilde, p.L, p.La2, p.b, p.eta);

        K_fu = gp_cov(gp, x, u);         % f x u
        K_uu = gp_trcov(gp, u);          % u x u, noiseles covariance K_uu
        K_uu = (K_uu+K_uu')./2;          % ensure the symmetry of K_uu
        iKuuKuf = K_uu\K_fu';

        LL = sum(L.*L,2);
        iLav=1./La;

        % =================================================================
        % Gradient with respect to covariance function parameters
        if ~isempty(strfind(gp.infer_params, 'covariance'))
          i1=0;
          for i=1:ncf

            gpcf = gp.cf{i};
            % Get the gradients of the covariance matrices
            % and gprior from gpcf_* structures
            gpcf = gp.cf{i};
            if savememory
              % If savememory option is used, just get the number of
              % hyperparameters and calculate gradients later
              np=gpcf.fh.cfg(gpcf,[],[],[],0);
            else
              %             DKffc = gpcf.fh.cfg(gpcf, x, [], 1);
              DKuuc = gpcf.fh.cfg(gpcf, u);
              DKufc = gpcf.fh.cfg(gpcf, u, x);
              np=length(DKuuc);
            end
            gprior_cf = -gpcf.fh.lpg(gpcf);

            for i2 = 1:np
              i1 = i1+1;
              if savememory
                %               DKff=gpcf.fh.cfg(gpcf,x,[],1,i2);
                DKuu=gpcf.fh.cfg(gpcf,u,[],[],i2);
                DKuf=gpcf.fh.cfg(gpcf,u,x,[],i2);
              else
                %               DKff=DKffc{i2};
                DKuu=DKuuc{i2};
                DKuf=DKufc{i2};
              end

              KfuiKuuKuu = iKuuKuf'*DKuu;
              gdata(i1) = -0.5.*((2*b*DKuf'-(b*KfuiKuuKuu))*(iKuuKuf*b'));
              gdata(i1) = gdata(i1) + 0.5.*(2.*(sum(iLav'*sum(DKuf'.*iKuuKuf',2))-sum(sum(L'.*(L'*DKuf'*iKuuKuf))))...
                                            - sum(iLav'*sum(KfuiKuuKuu.*iKuuKuf',2))+ sum(sum(L'.*((L'*KfuiKuuKuu)*iKuuKuf))));
            end
            gprior = [gprior gprior_cf];
          end

        end

        % =================================================================
        % Gradient with respect to inducing inputs
        if ~isempty(strfind(gp.infer_params, 'inducing'))
          if isfield(gp.p, 'X_u') && ~isempty(gp.p.X_u)
            m = size(gp.X_u,2);
            st=0;

            gdata_inducing(st+1:st+length(gp.X_u(:))) = 0;
            i1 = st+1;
            gprior_inducing=[];
            if iscell(gp.p.X_u) % Own prior for each inducing input
              for i = 1:size(gp.X_u,1)
                pr = gp.p.X_u{i};
                gprior_inducing =[gprior_inducing -pr.fh.lpg(gp.X_u(i,:), pr)];
              end
            else % One prior for all inducing inputs
              gprior_inducing = -gp.p.X_u.fh.lpg(gp.X_u(:)', gp.p.X_u);
            end

            for i=1:ncf
              i1=st;

              gpcf = gp.cf{i};
              if savememory
                % If savememory option is used, just get the number of
                % covariates in X and calculate gradients later
                np=gpcf.fh.ginput(gpcf,u,[],0);
              else
                DKuu = gpcf.fh.ginput(gpcf, u);
                DKuf = gpcf.fh.ginput(gpcf, u, x);
                np=1;
              end

              for i3 = 1:np
                if savememory
                  DKuu=gpcf.fh.ginput(gpcf,u,[],i3);
                  DKuf=gpcf.fh.ginput(gpcf,u,x,i3);
                end
                for i2 = 1:length(DKuu)
                  if savememory
                    i1 = st + (i2-1)*np+i3;
                    %i1 = st + 2*i2 - (i3==1);
                  else
                    i1 = i1+1;
                  end

                  KfuiKuuKuu = iKuuKuf'*DKuu{i2};
                  gdata_inducing(i1) = gdata_inducing(i1) - 0.5.*((2*b*DKuf{i2}'-(b*KfuiKuuKuu))*(iKuuKuf*b'));
                  gdata_inducing(i1) = gdata_inducing(i1) + 0.5.*(2.*(sum(iLav'*sum(DKuf{i2}'.*iKuuKuf',2))-sum(sum(L'.*(L'*DKuf{i2}'*iKuuKuf))))...
                                                                  - sum(iLav'*sum(KfuiKuuKuu.*iKuuKuf',2))+ sum(sum(L'.*((L'*KfuiKuuKuu)*iKuuKuf))));

                  if strcmp(gp.type, 'VAR')
                    gdata_inducing(i1) = gdata_inducing(i1) + 0.5.*(0-2.*sum(iLav'*sum(DKuf{i2}'.*iKuuKuf',2)) + ...
                                                                    sum(iLav'*sum(KfuiKuuKuu.*iKuuKuf',2)));
                  end
                end
              end
            end
          end
        end


      case 'VAR'
        % ============================================================
        % VAR
        % ============================================================
        % NOTE! Not properly implemented as no analytical result has been
        % derived. Not suitable for large data sets.
        epsilon = 1.0e-6;

        func = fcnchk(@gpep_e, 3);
        %grad = fcnchk(grad, length(varargin));

        % Treat
        nparams = length(w);
        deltaf = zeros(1, nparams);
        step = zeros(1, nparams);
        for i = 1:nparams
          % Move a small way in the ith coordinate of w
          step(i) = 1.0;
          func = fcnchk(func, 3);
          fplus = func(w+epsilon.*step, gp,x,y,'z',z);
          fminus = func(w-epsilon.*step, gp,x,y,'z',z);
          %   fplus  = feval('linef_test', epsilon, func, w, step, varargin{:});
          %   fminus = feval('linef_test', -epsilon, func, w, step, varargin{:});
          % Use central difference formula for approximation
          deltaf(i) = 0.5*(fplus - fminus)/epsilon;
          step(i) = 0.0;
        end
        % gradient = feval(grad, w, varargin{:});
        % fprintf(1, 'Checking gradient ...\n\n');
        % fprintf(1, '   analytic   diffs     delta\n\n');
        % disp([gradient', deltaf', gradient' - deltaf'])

        %delta = gradient' - deltaf';
        gdata=deltaf;

        %gdata=numgrad_test(gp_pak(gp), @gpep_e, gp, x, y);
        gprior=0;

    end

    if ~strcmp(gp.type,'VAR')
      
      if isfield(gp.lik, 'int_likparam')
        
        % =================================================================
        % Gradient with respect to likelihood function parameters
        if ~isempty(strfind(gp.infer_params, 'likelihood')) && isfield(gp.lik.fh, 'siteDeriv') ...
            && ~isempty(gp.lik.fh.pak(gp.lik))

          if isempty(sigm2_i)
            sigm2_i=p.sigm2vec_i;
            mu_i=p.muvec_i;
          end

          gdata_lik = 0;
          lik = gp.lik;
          if int_magnitude
            sigm2_i=p.sigm2vec_i;
            mu_i=p.muvec_i;
          end
          for k1 = 1:length(y)
            if isempty(eta)
              gdata_lik = gdata_lik - lik.fh.siteDeriv(lik, y, k1, sigm2_i(k1,:), mu_i(k1,:), z);
            else
              gdata_lik = gdata_lik - lik.fh.siteDeriv2(lik, y, k1, sigm2_i(k1), mu_i(k1), z, eta(k1), Z_i(k1));
            end
          end

          % evaluate prior contribution for the gradient
          if isfield(gp.lik, 'p')
            gprior_lik = -lik.fh.lpg(lik);
          else
            gprior_lik = zeros(size(gdata_lik));
          end

          % set the gradients into vectors that will be returned
          gdata = [gdata gdata_lik];
          gprior = [gprior gprior_lik];
        end
        
      else
        
        % =================================================================
        % Gradient with respect to likelihood function parameters
        if ~isempty(strfind(gp.infer_params, 'likelihood')) && (isfield(gp.lik.fh, 'siteDeriv') ...
            || (isfield(gp, 'lik_mono') && isfield(gp.lik.fh, 'siteDeriv')))

          if isempty(sigm2_i)
            sigm2_i=p.sigm2vec_i;
            mu_i=p.muvec_i;
          end

          gdata_lik = 0;       
          lik = gp.lik; 
          gprior_lik = -lik.fh.lpg(lik);   % evaluate prior contribution for the gradient
          if ~isempty(gprior_lik)          % calculate gradient wrt likelihood parameters only if they have prior

              for k1 = 1:length(y)
                  if isempty(eta)
                      gdata_lik = gdata_lik - lik.fh.siteDeriv(lik, y, k1, sigm2_i(k1), mu_i(k1), z);
                  else
                      gdata_lik = gdata_lik - lik.fh.siteDeriv2(lik, y, k1, sigm2_i(k1), mu_i(k1), z, eta(k1), Z_i(k1));
                  end
              end

              % set the gradients into vectors that will be returned
              gdata = [gdata gdata_lik];
              gprior = [gprior gprior_lik];
          end
        end
        
      end

    end

    if isfield(gp,'lik_mono') && isequal(gp.lik.type, 'Gaussian')
      % Monotonic GP with Gaussian likelihood
      s2=gp.lik.sigma2;
%       DCff = blkdiag(s2.*eye(n2), zeros(70));
%       gdata_lik = -(-0.5.*trace((C+diag(1./tautilde))\DCff) ...
%         + 0.5.*(nutilde./tautilde)'*((C+diag(1./tautilde))\(DCff*((C+diag(1./tautilde))\(nutilde./tautilde)))));
      Sigma=p.La2'*p.La2;
      mf=Sigma*nutilde;
      gdata_lik = -sum((-s2+diag(Sigma(1:n2,1:n2))+(mf(1:n2)-y).^2)./(2*s2.^2)).*s2;
%       gdata_lik = gdata_lik - n2./2;% + sum(y.^2./(2*s2));
      lik=gp.lik_mono;
      if isfield(gp.lik, 'p')  && ~isempty(gp.lik.p.sigma2)
        gprior_lik = -gp.lik.fh.lpg(gp.lik);
      else
        gprior_lik = zeros(size(gdata_lik));
      end
      
      % set the gradients into vectors that will be returned
      gdata = [gdata gdata_lik];
      gprior = [gprior gprior_lik];
    end

    % add gradient with respect to inducing inputs (computed in gp.type sepcific way)
    gdata = [gdata gdata_inducing];
    gprior = [gprior gprior_inducing];

    % If ther parameters of the model (covariance function parameters,
    % likelihood function parameters, inducing inputs) have additional
    % hyperparameters that are not fixed, set the gradients in correct order
    if length(gprior) > length(gdata)
      %gdata(gdata==0)=[];
      tmp=gdata;
      gdata = zeros(size(gprior));
      % Set the gradients to right place
      if any(hier==0)
        gdata([hier(1:find(hier==0,1)-1)==1 ...  % Covariance function
          hier(find(hier==0,1):find(hier==0,1)+length(gprior_lik)-1)==0 ... % Likelihood function
          hier(find(hier==0,1)+length(gprior_lik):end)==1]) = tmp;  % Inducing inputs
      else
        gdata(hier==1)=tmp;
      end
    end
    
    % total gradient
    g = gdata + gprior;

  end

end