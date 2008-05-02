function likelih = likelih_poisson(do, varargin)
%likelih_poisson	Create a Poisson likelihood structure for Gaussian Process
%
%	Description
%
%	likelih = LIKELIH_POISSON('INIT', NIN) Create and initialize squared exponential
%       covariance function fo Gaussian process
%
%	The fields and (default values) in GPCF_SEXP are:
%	  type           = 'likelih_poisson'
%
%	likelih = LIKELIH_POISSON('SET', LIKELH, 'FIELD1', VALUE1, 'FIELD2', VALUE2, ...)
%       Set the values of fields FIELD1... to the values VALUE1... in LIKELIH.
%
%	See also
%
%
%

% Copyright (c) 2006-2007 Jarno Vanhatalo

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.

    if nargin < 2
        error('Not enough arguments')
    end

    % Initialize the covariance function
    if strcmp(do, 'init')
        y = varargin{1};
        avgE = varargin{2};
        likelih.type = 'poisson';
        
        % Set parameters
        likelih.avgE = avgE;
        likelih.gamlny = gammaln(y+1);

        % Initialize prior structure

        % Set the function handles to the nested functions
        likelih.fh_pak = @likelih_poisson_pak;
        likelih.fh_unpak = @likelih_poisson_unpak;
        likelih.fh_permute = @likelih_poisson_permute;
        likelih.fh_e = @likelih_poisson_e;
        likelih.fh_g = @likelih_poisson_g;    
        likelih.fh_hessian = @likelih_poisson_hessian;
        likelih.fh_g3 = @likelih_poisson_g3;
        likelih.fh_tiltedMoments = @likelih_poisson_tiltedMoments;
        likelih.fh_mcmc = @likelih_poisson_mcmc;

        if length(varargin) > 2
            if mod(nargin,2) ~=1
                error('Wrong number of arguments')
            end
            % Loop through all the parameter values that are changed
            for i=2:2:length(varargin)-1
                switch varargin{i}
                  case 'avgE'
                    likelih.avgE = varargin{i+1};
                  case 'gamlny'
                    likelih.gamlny = varargin{i+1};
                  otherwise
                    error('Wrong parameter name!')
                end
            end
        end
    end

    % Set the parameter values of covariance function
    if strcmp(do, 'set')
        if mod(nargin,2) ~=1
            error('Wrong number of arguments')
        end
        gpcf = varargin{1};
        % Loop through all the parameter values that are changed
        for i=2:2:length(varargin)-1
            switch varargin{i}
              case 'avgE'
                likelih.avgE = varargin{i+1};
              case 'gamlny'
                likelih.gamlny = varargin{i+1};
              otherwise
                error('Wrong parameter name!')
            end
        end
    end



    function w = likelih_poisson_pak(likelih, w)
    %
    %
    %   
    end


    function w = likelih_poisson_unpak(likelih, w)
    %
    %
    %    
    end



    function likelih = likelih_poisson_permute(likelih, p)
    %
    %
    %
        likelih.avgE = likelih.avgE(p,:);
        likelih.gamlny = likelih.gamlny(p,:);
    end


    function logLikelih = likelih_poisson_e(likelih, y, f)
    %
    %
    %
        lambda = likelih.avgE.*exp(f);
        gamlny = likelih.gamlny;
        logLikelih =  sum(-lambda + y.*log(lambda) - gamlny);
    end


    function deriv = likelih_poisson_g(likelih, y, f, param)
        switch param
          case 'latent'
            deriv = y - likelih.avgE.*exp(f);
        end
    end


    function hessian = likelih_poisson_hessian(likelih, y, f, param)
    %
    %
    %
        switch param
          case 'latent'
            hessian = -likelih.avgE.*exp(f);
        end
    end    
    
    function third_grad = likelih_poisson_g3(likelih, y, f, param)
    %
    %
    %
        switch param
          case 'latent'
            third_grad = - likelih.avgE.*exp(f);
        end
    end


    function [m_0, m_1, m_2] = likelih_poisson_tiltedMoments(likelih, y, i1, sigm2_i, myy_i)
    %
    %
    %
        zm = @zeroth_moment;
        fm = @first_moment;
        sm = @second_moment;

        atol = 1e-10;
        reltol = 1e-6;
        yy = y(i1);
        gamlny = likelih.gamlny(i1);
        avgE = likelih.avgE(i1);
        
        % Set the limits for integration and integrate with quad
        % -----------------------------------------------------
        if yy > 0
            mean_app = (myy_i/sigm2_i + log(yy/avgE).*yy)/(1/sigm2_i + yy);
            sigm_app = sqrt((1/sigm2_i + avgE)^-1);
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
        
        [m_0, fhncnt] = quadgk(zm, lambdaconf(1), lambdaconf(2)); %,'AbsTol',atol,'RelTol',reltol
        [m_1, fhncnt] = quadgk(fm, lambdaconf(1), lambdaconf(2));
        [sigm2hati1, fhncnt] = quadgk(sm, lambdaconf(1), lambdaconf(2));
        if sigm2hati1 >= sigm2_i
            tol = atol.^2;
            reltol = reltol.^2;
            [m_0, fhncnt] = quadgk(zm, lambdaconf(1), lambdaconf(2));
            [m_1, fhncnt] = quadgk(fm, lambdaconf(1), lambdaconf(2));
            [sigm2hati1, fhncnt] = quadgk(sm, lambdaconf(1), lambdaconf(2));
            %                    [M, fhncnt] = quadl_4moms(moms, lambdaconf(1), lambdaconf(2), tol, false);
        end
        m_2 = sigm2hati1;
        
        function integrand = zeroth_moment(f)
            lambda = avgE.*exp(f);
            integrand = exp(-lambda + yy.*log(lambda) - gamlny - 0.5 * (f-myy_i).^2./sigm2_i - log(sigm2_i)/2 - log(2*pi)/2); %
        end

        function integrand = first_moment(f)
            lambda = avgE.*exp(f);
            integrand = exp(-lambda + yy.*log(lambda) - gamlny - 0.5 * (f-myy_i).^2./sigm2_i - log(sigm2_i)/2 - log(2*pi)/2 - log(m_0)); %
            integrand = f.*integrand; %
        end
        function integrand = second_moment(f)
            lambda = avgE.*exp(f);
            integrand = exp(log((f-m_1).^2) -lambda + yy.*log(lambda) - gamlny - 0.5 * (f-myy_i).^2./sigm2_i - log(sigm2_i)/2 - log(2*pi)/2 - log(m_0));
            %integrand = (f-m_1).^2.*integrand; %
        end
        function integrand = moments(f)
            lambda = avgE.*exp(f);
            temp = exp(-lambda + yy.*log(lambda) - gamlny - 0.5 * (f-myy_i).^2./sigm2_i - log(sigm2_i)/2 - log(2*pi)/2); %
            integrand(3,:) =  temp;
            integrand(2,:) = f.*temp; %
            integrand(1,:) = f.^2.*temp; %
        end
        
    end


    function [z, energ, diagn] = likelih_poisson_mcmc(z, opt, varargin)
    %
    %
    %
        if isfield(opt, 'rstate')
            if ~isempty(opt.rstate)
                latent_rstate = opt.latent_opt.rstate;
            end
        else
            latent_rstate = sum(100*clock);
        end

        % Set the variables 
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
          case {'PIC_BLOCK','CS+PIC'}
            u = gp.X_u;
            ind = gp.tr_index;
            Labl=[];
            Lp = [];
          case 'PIC_BAND'
            u = gp.X_u;
            ind = gp.tr_index;
            nzmax = size(ind,1);
            Labl= sparse([],[],[],n,n,0);
            Lp = sparse([],[],[],n,n,0); 
        end
        n=length(y);

        J = [];
        U = [];
        iJUU = [];
        Linv=[];
        L2=[];
        iLaKfuic=[];
        mincut = -300;
        if isfield(gp,'avgE');
            E=gp.avgE(:);
        else
            E=1;
        end     

        % Evaluate the help matrices for covariance matrix
        switch gp.type
          case 'FULL'
            getL(z, gp, x, y);
            % Rotate z towards prior
            w = (L2\z)';    
          case 'FIC'
            getL(z, gp, x, y, u);
            % Rotate z towards prior as w = (L\z)';
            % Here we take an advantage of the fact that L = chol(diag(Lav)+b'b)
            % See cholrankup.m for general case of solving a Ax=b system
    % $$$     Uz = U'*z;
    % $$$     w = z + U*inv(J)*Uz - U*Uz;
            zs = z./Lp;
            w = zs + U*((J*U'-U')*zs);
          case 'PIC_BLOCK'
            getL(z, gp, x, y, u);
            zs=zeros(size(z));
            for i=1:length(ind)
                zs(ind{i}) = Lp{i}\z(ind{i});
            end
            w = zs + U*((J*U'-U')*zs);
          case {'CS+FIC'}
            getL(z, gp, x, y, u);
            %zs = Lp\z;
            zs = Lp*z;
            w = zs + U*((J*U'-U')*zs);        
          otherwise 
            error('unknown type of GP\n')
        end
        
        
        %gradcheck(w', @lvpoisson_er, @lvpoisson_gr, gp, x, y, u, z)
        
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
        % Rotate w to z
        w=w(:);
        switch gp.type
          case 'FULL'
            z=L2*w;
          case 'FIC'
    % $$$     z = Lp.*(w + U*inv(JUU*w));
            z = Lp.*(w + U*(iJUU*w));
          case  'PIC_BLOCK'
            w2 = w + U*(iJUU*w);
            for i=1:length(ind)
                z(ind{i}) = Lp{i}*w2(ind{i});
            end
          case  {'CS+FIC'}
            w2 = w + U*(iJUU*w);
            %        z = Lp*w2;
            z = Lp\w2;
        end
        opt.latent_rstate = hmc2('state');
        diagn.opt = opt;
        diagn.rej = rej;
        diagn.lvs = opt.stepadj;

        function [g, gdata, gprior] = lvpoisson_gr(w, gp, x, y, u, varargin)
        %LVPOISSON_G	Evaluate gradient function for GP latent values with
        %               Poisson likelihood
            
        % Force z and E to be a column vector
            w=w(:);
            
            switch gp.type
              case 'FULL'
                z = L2*w;
                z = max(z,mincut);
                gdata = exp(z).*E - y;
                %gdata = ((I+U*J*U'-U*U')*(mu-y)))'; % (  (mu-y) )';
        % $$$         gprior = w';                   % make the gradient a row vector
                b=Linv*z;
                gprior=Linv'*b;  %dsymvr takes advantage of the symmetry of Cinv
        % $$$         gprior = (dsymvr(Cinv,z))';   % make the gradient a row vector
                g = (L2'*(gdata +gprior))';        
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
              case 'PIC_BLOCK'
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
                %            z = Lp*w2;
                z = Lp\w2;
                z = max(z,mincut);
                gdata = exp(z).*E - y;
                gprior = zeros(size(gdata));
                gprior = Labl\z;
                gprior = gprior - iLaKfuic*(iLaKfuic'*z);
                g = gdata' + gprior';
                %            g = g*Lp;
                g = g/Lp;
                g = g + g*U*(iJUU);
            end
        end

        function [e, edata, eprior] = lvpoisson_er(w, gp, x, t, u, varargin)
        %function [e, edata, eprior] = gp_e(w, gp, x, t, varargin)
        % LVPOISSON_E     Minus log likelihood function for spatial modelling.
        %
        %       E = LVPOISSON_E(X, GP, T, Z) takes.... and returns minus log from 
            
        % The field gp.avgE (if given) contains the information about averige
        % expected number of cases at certain location. The target, t, is 
        % distributed as t ~ poisson(avgE*exp(z))
            
        % force z and E to be a column vector

            w=w(:);

            switch gp.type
              case 'FULL'
                z = L2*w;        
                z = max(z,mincut);
                B=Linv*z;
                eprior=.5*sum(B.^2);
              case 'FIC' 
                %        w(w<eps)=0;
                z = Lp.*(w + U*(iJUU*w));
                z = max(z,mincut);
                % eprior = 0.5*z'*inv(La)*z-0.5*z'*(inv(La)*K_fu*inv(K_uu+Kuf*inv(La)*K_fu)*K_fu'*inv(La))*z;
                B = z'*iLaKfuic;  % 1 x u
                eprior = 0.5*sum(z.^2./Lav)-0.5*sum(B.^2);
              case 'PIC_BLOCK'
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
                %            z = Lp*w2;
                z = Lp\w2;
                z = max(z,mincut);
                B = z'*iLaKfuic;  % 1 x u
                eprior = - 0.5*sum(B.^2);
                eprior = eprior + 0.5*z'/Labl*z;
            end
            mu = exp(z).*E;
            edata = sum(mu-t.*log(mu));
            %        eprior = .5*sum(w.^2);
            e=edata + eprior;
        end

        function getL(w, gp, x, t, u)
        % Evaluate the cholesky decomposition if needed
            if nargin < 5
                C=gp_trcov(gp, x);
                % Evaluate a approximation for posterior variance
                % Take advantage of the matrix inversion lemma
                %        L=chol(inv(inv(C) + diag(1./gp.avgE)))';
                Linv = inv(chol(C)');
                L2 = C/chol(diag(1./gp.avgE) + C);  %sparse(1:n, 1:n, 1./gp.avgE)
                L2 = chol(C - L2*L2')';
            else        
                % Evaluate the Lambda (La) for specific model
                switch gp.type
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
                    Lp = sqrt(1./(gp.avgE + 1./Lav));
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
                                                  % J = diag(sqrt(2./(1+diag(S))));
                    iJUU = J\U'-U';
                    iJUU(abs(iJUU)<eps)=0;
                  case 'PIC_BLOCK'
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
                        Lp{i} = chol(inv(diag(gp.avgE(ind{i})) + inv(Labl{i})));
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
                    % Q_ff = K_fu*inv(K_uu)*K_fu'
                    % Here we need only the diag(Q_ff), which is evaluated below
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

                    [Kv_ff, Cv_ff] = gp_trvar(gp, x);  % f x 1  vector
                    K_fu = gp_cov(gp, x, u);         % f x u
                    K_uu = gp_trcov(gp, u);    % u x u, noiseles covariance K_uu
                    K_uu = (K_uu+K_uu')/2;     % ensure the symmetry of K_uu
                    Luu = chol(K_uu)';                
                    B=Luu\(K_fu');       % u x f

                    Qv_ff=sum(B.^2)';
                    Lav = Cv_ff-Qv_ff;   % f x 1, Vector of diagonal elements
                    
                    gp.cf = cf2;        
                    K_cs = gp_trcov(gp,x);
                    Labl = sparse(1:n,1:n,Lav,n,n) + K_cs;
                    gp.cf = cf_orig;
                    iLaKfu = Labl\K_fu;
                    % Lets scale Lav to ones(f,1) so that Qff+La -> sqrt(La)*Qff*sqrt(La)+I
                    % and form iLaKfu
                    A = K_uu+K_fu'*iLaKfu;
                    A = (A+A')./2;            % Ensure symmetry
                    
                    % L = iLaKfu*inv(chol(A));
                    iLaKfuic = iLaKfu*inv(chol(A));
                    
                    %Lp = chol(inv(sparse(1:n,1:n,gp.avgE,n,n) + inv(Labl)));
                    %Lp = inv(chol(sparse(1:n,1:n,gp.avgE,n,n) + inv(Labl))');
                    Lp = inv(Labl);
                    Lp = sparse(1:n,1:n,gp.avgE,n,n) + Lp;
                    Lp = chol(Lp)';
                    %                Lp = inv(Lp);


                    b=zeros(size(B'));
                    
                    %                b = Lp*iLaKfuic;
                    b = Lp\iLaKfuic;
                    
                    [V,S2]= eig(b'*b);
                    S = sqrt(S2);
                    U = b*(V/S);
                    U(abs(U)<eps)=0;
                    %        J = diag(sqrt(diag(S2) + 0.01^2));
                    J = diag(sqrt(1-diag(S2)));   % this could be done without forming the diag matrix 
                                                  % J = diag(sqrt(2./(1+diag(S))));
                    iJUU = J\U'-U';
                    iJUU(abs(iJUU)<eps)=0;
                end
            end
        end
    end 
end


