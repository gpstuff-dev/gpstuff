function [z, energ, diagn] = latent_hmcr(z, opt, varargin)
% LATENT_HMC2     HMC sampler for latent values.
%
%
%
%

% Copyright (c) 2006-2007      Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.


% Set the state of HMC samler
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
      case 'PIC_BLOCK'
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
      case 'PIC_BAND'
        getL(z, gp, x, y, u);
        %zs = Lp\z;
        zs = Lp*z;
        w = zs + U*((J*U'-U')*zs);
    end
    
    
    %    gradcheck(w', @lvpoisson_er, @lvpoisson_gr, gp, x, y, u, z)
    
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
      case  'PIC_BAND'
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
          case 'PIC_BAND'
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
          case 'PIC_BAND'
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
            [Kv_ff, Cv_ff] = gp_trvar(gp, x);  % f x 1  vector
            K_fu = gp_cov(gp, x, u);         % f x u
            K_uu = gp_trcov(gp, u);    % u x u, noiseles covariance K_uu
            K_uu = (K_uu+K_uu')/2;     % ensure the symmetry of K_uu
            Luu = chol(K_uu)';
            % Evaluate the Lambda (La) for specific model
            switch gp.type
              case 'FIC'
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
% $$$                     Lp{i} = chol(diag(gp.avgE(ind{i})) + inv(Labl{i}))';
% $$$                     Lp{i} = inv(Lp{i});
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
              case 'PIC_BAND'
                % Q_ff = K_fu*inv(K_uu)*K_fu'
                % Here we need only the diag(Q_ff), which is evaluated below
                B=Luu\(K_fu');       % u x f

                [I,J]=find(tril(sparse(ind(:,1),ind(:,2),1,n,n),-1));
% $$$                 for i = 1:length(J)
% $$$                     q_ff(i) = B(:,I(i))'*B(:,J(i));
% $$$                 end
                q_ff = sum(B(:,I).*B(:,J));
                q_ff = sparse(I,J,q_ff,n,n);
                c_ff = gp_covvec(gp, x(I,:), x(J,:))';
                c_ff = sparse(I,J,c_ff,n,n);
                [Kv_ff, Cv_ff] = gp_trvar(gp,x);
                Labl = c_ff + c_ff' - q_ff - q_ff' + sparse(1:n,1:n, Cv_ff-sum(B.^2,1)',n,n);

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
