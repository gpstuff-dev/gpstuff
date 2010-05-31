function [f, energ, diagn] = scaled_hmc(f, opt, gp, x, y, z)
%scaled_hmc        A scaled hybric Monte Carlo samping for latent values
%
%   Description
%   [F, ENERG, DIAG] = SCALED_HMC(F, OPT, GP, X, Y) takes the current
%   latent values F, options structure OPT, Gaussian process data
%   structure GP, inputs X and outputs Y. Samples new latent values
%   and returns also energies ENERG and diagnostics DIAG. The latent
%   values are sampled from their conditional posterior p(f|y,th).
%
%   The latent values are whitened with an approximate posterior
%   covariance before the sampling. This reduces the autocorrelation
%   and speeds up the mixing of the sampler. See Vanhatalo and Vehtari
%   (2007) for details on implementation.
%
%   The options structure needs to be similar than with Hybrid Monte
%   Carlo, hmc2. See hmc2 and hmc2_opt for details.
%
%   See also
%   GP_MC, HMC2, HMC2_OPT
 
% Copyright (c) 2006 Aki Vehtari
% Copyright (c) 2006-2010 Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

    % Check if an old sampler state is provided
    if isfield(opt, 'rstate')
        if ~isempty(opt.rstate)
            latent_rstate = opt.rstate;
        end
    else
        latent_rstate = sum(100*clock);
    end
    

    % Initialize variables
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

    % Transform the latent values
    switch gp.type
      case 'FULL'
        getL(f, gp, x, y, z);             % Evaluate the help matrix for transformation
        w = (L2\f)';                   % Rotate f towards prior
      case 'FIC'
        getL(f, gp, x, y, u, z);          % Evaluate the help matrix for transformation
        fs = f./Lp;                    % Rotate f towards prior
        w = fs + U*((J*U'-U')*fs);     
      case {'PIC' 'PIC_BLOCK'}
        getL(f, gp, x, y, u, z);          % Evaluate the help matrix for transformation
        fs=zeros(size(f));             % Rotate f towards prior
        for i=1:length(ind)
            fs(ind{i}) = Lp{i}\f(ind{i});
        end
        w = fs + U*((J*U'-U')*fs);
      case {'CS+FIC'}
        getL(f, gp, x, y, u, z);          % Evaluate the help matrix for transformation
        fs = Lp\f;                        % Rotate f towards prior
        w = fs + U*((J*U'-U')*fs);
      otherwise 
        error('unknown type of GP\n')
    end
    
    
    %gradcheck(w, @f_e, @f_g, gp, x, y, u, f);
    
    % Conduct the HMC sampling for the transformed latent values
    hmc2('state',latent_rstate)
    rej = 0;
    for li=1:opt.repeat 
        [w, energ, diagn] = hmc2(@f_e, w, opt, @f_g, gp, x, y, u, z);
        w = w(end,:);
        % Find an optimal scaling during the first half of repeat
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
        f=L2*w;
      case 'FIC'
        f = Lp.*(w + U*(iJUU*w));
      case  {'PIC' 'PIC_BLOCK'}
        w2 = w + U*(iJUU*w);
        for i=1:length(ind)
            f(ind{i}) = Lp{i}*w2(ind{i});
        end
      case  {'CS+FIC'}
        w2 = w + U*(iJUU*w);
        f = Lp*w2;            
    end
    % Record the options
    opt.rstate = hmc2('state');
    diagn.opt = opt;
    diagn.rej = rej;
    diagn.lvs = opt.stepadj;

    function [g, gdata, gprior] = f_g(w, gp, x, y, u, z)
    %F_G     Evaluate gradient function for transformed GP latent values 
    %               
        
    % Force f and E to be a column vector
        w=w(:);
        
        switch gp.type
          case 'FULL'
            f = L2*w;
            f = max(f,mincut);
            gdata = - feval(gp.likelih.fh_g, gp.likelih, y, f, 'latent', z);
            b=Linv*f;
            gprior=Linv'*b;
            g = (L2'*(gdata + gprior))';
          case 'FIC'
            %        w(w<eps)=0;
            f = Lp.*(w + U*(iJUU*w));
            gdata = - feval(gp.likelih.fh_g, gp.likelih, y, f, 'latent', z);
            f = max(f,mincut);
            gprior = f./Lav - iLaKfuic*(iLaKfuic'*f);
            g = gdata +gprior;
            g = Lp.*g;
            g = g + U*(iJUU*g);
            g = g';
          case {'PIC' 'PIC_BLOCK'}
            w2= w + U*(iJUU*w);
            for i=1:length(ind)
                f(ind{i}) = Lp{i}*w2(ind{i});
            end
            f = max(f,mincut);
            gdata = - feval(gp.likelih.fh_g, gp.likelih, y, f, 'latent', z);
            gprior = zeros(size(gdata));
            for i=1:length(ind)
                gprior(ind{i}) = Labl{i}\f(ind{i});
            end
            gprior = gprior - iLaKfuic*(iLaKfuic'*f);
            g = gdata' + gprior';
            for i=1:length(ind)
                g(ind{i}) = g(ind{i})*Lp{i};
            end
            g = g + g*U*(iJUU);
            %g = g';
          case {'CS+FIC'}
            w2= w + U*(iJUU*w);
            f = Lp*w2;
            f = max(f,mincut);
            gdata = - feval(gp.likelih.fh_g, gp.likelih, y, f, 'latent', z);
            gprior = zeros(size(gdata));
            gprior = ldlsolve(Labl,f);
            gprior = gprior - iLaKfuic*(iLaKfuic'*f);
            g = gdata' + gprior';
            g = g*Lp;
            g = g + g*U*(iJUU);
        end
    end

    function [e, edata, eprior] = f_e(w, gp, x, y, u, z)
    % F_E     Evaluate energy function for transformed GP latent values 
        
    % force f and E to be a column vector
        w=w(:);
        switch gp.type
          case 'FULL'
            f = L2*w;        
            f = max(f,mincut);
            B=Linv*f;
            eprior=.5*sum(B.^2);
          case 'FIC' 
            f = Lp.*(w + U*(iJUU*w));
            f = max(f,mincut);                
            B = f'*iLaKfuic;  % 1 x u
            eprior = 0.5*sum(f.^2./Lav)-0.5*sum(B.^2);
          case {'PIC' 'PIC_BLOCK'}
            w2= w + U*(iJUU*w);
            for i=1:length(ind)
                f(ind{i}) = Lp{i}*w2(ind{i});
            end
            f = max(f,mincut);
            B = f'*iLaKfuic;  % 1 x u
            eprior = - 0.5*sum(B.^2);
            for i=1:length(ind)
                eprior = eprior + 0.5*f(ind{i})'/Labl{i}*f(ind{i});
            end
          case {'CS+FIC'}
            w2= w + U*(iJUU*w);
            f = Lp*w2;
            f = max(f,mincut);
            B = f'*iLaKfuic;  % 1 x u
            eprior = - 0.5*sum(B.^2);
            eprior = eprior + 0.5*f'*ldlsolve(Labl,f);
        end
        edata =  - feval(gp.likelih.fh_e, gp.likelih, y, f, z);
        e=edata + eprior;
    end

    function getL(w, gp, x, y, u, z)
    % GETL        Evaluate the transformation matrix (or matrices)
        
    % Evaluate the Lambda (La) for specific model
        E = -feval(gp.likelih.fh_g2, gp.likelih, y, zeros(size(y)), 'latent', z);
        switch gp.type
          case 'FULL'
            C=gp_trcov(gp, x);
            % Evaluate a approximation for posterior variance
            % Take advantage of the matrix inversion lemma
            %        L=chol(inv(inv(C) + diag(E)))';
            Linv = inv(chol(C)');
            L2 = C/chol(diag(1./E) + C);
            L2 = chol(C - L2*L2')';                    
          case 'FIC'
            [Kv_ff, Cv_ff] = gp_trvar(gp, x);  % f x 1  vector
            K_fu = gp_cov(gp, x, u);           % f x u
            K_uu = gp_trcov(gp, u);            % u x u, noiseles covariance K_uu
            K_uu = (K_uu+K_uu')/2;             % ensure the symmetry of K_uu
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
            K_fu = gp_cov(gp, x, u);           % f x u
            K_uu = gp_trcov(gp, u);            % u x u, noiseles covariance K_uu
            K_uu = (K_uu+K_uu')/2;             % ensure the symmetry of K_uu
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
