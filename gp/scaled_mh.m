function [f, energ, diagn] = scaled_mh(f, opt, gp, x, y, z)
%scaled_mh        A scaled Metropolis Hastings samping for latent values
%
%   Description

%   [F, ENERG, DIAG] = SCALED_MH(F, OPT, GP, X, Y) takes the current
%   latent values F, options structure OPT, Gaussian process data
%   structure GP, inputs X and outputs Y. Samples new latent values
%   and returns also energies ENERG and diagnostics DIAG. The latent
%   values are sampled from their conditional posterior p(f|y,th).
%
%   The latent values are whitened with the prior covariance before
%   the sampling. This reduces the autocorrelation and speeds up the
%   mixing of the sampler. See (Neal, 1993) for details on
%   implementation.
%
%   The options structure should include the following fields:
%      opt.repeat              : The number MH-steps before 
%                                returning single sample
%      opt.sample_latent_scale : scale for the MH-step
%
%
%   See also
%   GP_MC
 
% Copyright (c) 1999 Aki Vehtari
% Copyright (c) 2006-2010 Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

    maxcut = -log(eps);
    mincut = -log(1/realmin - 1);
    lvs=opt.sample_latent_scale;
    a = max(min(f, maxcut),mincut);
    
    switch gp.type
      case {'FULL'}
  
        [K,C]=gp_trcov(gp, x);
        L=chol(C)';
        n=length(y);
        e = -feval(gp.likelih.fh_e, gp.likelih, y, f, z);
        
        % Adaptive control algorithm to find such a value for lvs 
        % that the rejection rate of Metropolis is optimal. 
        slrej = 0;
        for li=1:100
            ft=sqrt(1-lvs.^2).*f+lvs.*L*randn(n,1);
            at = max(min(ft, maxcut),mincut);
            ed = -feval(gp.likelih.fh_e, gp.likelih, y, ft, z);
            a=e-ed;
            if exp(a) > rand(1)
                f=ft;
                e=ed;
                lvs=min(1,lvs*1.1);
            else
                lvs=max(1e-8,lvs/1.05);
            end
        end
        opt.sample_latent_scale=lvs;
        % Do the actual sampling 
        for li=1:(opt.repeat)
            ft=sqrt(1-lvs.^2).*f+lvs.*L*randn(n,1);
            at = max(min(ft, maxcut),mincut);
            ed = -feval(gp.likelih.fh_e, gp.likelih, y, ft, z);
            a=e-ed;
            if exp(a) > rand(1)
                f=ft;
                e=ed;
            else
                slrej=slrej+1;
            end
        end
        diagn.rej = slrej/opt.repeat;
        diagn.lvs = lvs;
        diagn.opt=opt;
        energ=[];
        f = f';
      
      case 'FIC'
        u = gp.X_u;
        m = size(u,1);
        % Turn the inducing vector on right direction
        if size(u,2) ~= size(x,2)
            u=u';
        end
        % Calculate some help matrices
        [Kv_ff, Cv_ff] = gp_trvar(gp, x);
        K_fu = gp_cov(gp, x, u);
        K_uu = gp_trcov(gp, u);
        Luu = chol(K_uu)';

        % Evaluate the Lambda (La) 
        B=Luu\(K_fu');
        Qv_ff=sum(B.^2)';
        Lav = Cv_ff-Qv_ff;
        sLav = sqrt(Lav);
       
        n=length(y);
        e = -feval(gp.likelih.fh_e, gp.likelih, y, f, z);

        % Adaptive control algorithm to find such a value for lvs 
        % so that the rejection rate of Metropolis is optimal. 
        slrej = 0;
        for li=1:100
            ft=sqrt(1-lvs.^2).*f + lvs.*(sLav.*randn(n,1) + B'*randn(m,1));
            at = max(min(ft, maxcut),mincut);
            ed = -feval(gp.likelih.fh_e, gp.likelih, y, ft, z);
            a=e-ed;
            if exp(a) > rand(1)
                f=ft;
                e=ed;
                lvs=min(1,lvs*1.1);
            else
                lvs=max(1e-8,lvs/1.05);
            end
        end
        opt.sample_latent_scale=lvs;
        % Do the actual sampling 
        for li=1:(opt.repeat)
            ft=sqrt(1-lvs.^2).*f + lvs.*(sLav.*randn(n,1) + B'*randn(m,1));
            at = max(min(ft, maxcut),mincut);
            ed = -feval(gp.likelih.fh_e, gp.likelih, y, ft, z);
            a=e-ed;
            if exp(a) > rand(1)
                f=ft;
                e=ed;
            else
                slrej=slrej+1;
            end
        end
        diagn.rej = slrej/opt.repeat;
        diagn.lvs = lvs;
        diagn.opt=opt;
        energ=[];
        f = f';        
        
      case 'PIC'
        u = gp.X_u;
        m = size(u,1);
        ind = gp.tr_index;
        if size(u,2) ~= size(x,2)
            u=u';
        end
        
        % Calculate some help matrices
        [Kv_ff, Cv_ff] = gp_trvar(gp, x);  % 1 x f  vector
        K_fu = gp_cov(gp, x, u);         % f x u
        K_uu = gp_trcov(gp, u);    % u x u, noiseles covariance K_uu
        Luu = chol(K_uu)';
    
        % Evaluate the Lambda (La) for specific model
        % Q_ff = K_fu*inv(K_uu)*K_fu'
        % Here we need only the diag(Q_ff), which is evaluated below
        B=Luu\K_fu';
        iLaKfu = zeros(size(K_fu));  % f x u
        for i=1:length(ind)
            Qbl_ff = B(:,ind{i})'*B(:,ind{i});
            [Kbl_ff, Cbl_ff] = gp_trcov(gp, x(ind{i},:));
            La{i} = Cbl_ff - Qbl_ff;
            CLa{i} = chol(La{i})' ;
        end
        
        n=length(y);
        e = -feval(gp.likelih.fh_e, gp.likelih, y, f, z);

        % Adaptive control algorithm to find such a value for lvs 
        % so that the rejection rate of Metropolis is optimal. 
        slrej = 0;
        for li=1:100
            sampf = randn(size(f));
            for i=1:length(ind)
                sampf(ind{i},:) = CLa{i}*sampf(ind{i},:);
            end
            ft=sqrt(1-lvs.^2).*f + lvs.*(sampf + B'*randn(m,1));
            at = max(min(ft, maxcut),mincut);
            ed = -feval(gp.likelih.fh_e, gp.likelih, y, ft, z);
            a=e-ed;
            if exp(a) > rand(1)
                f=ft;
                e=ed;
                lvs=min(1,lvs*1.1);
            else
                lvs=max(1e-8,lvs/1.05);
            end
        end
        opt.sample_latent_scale=lvs;
        % Do the actual sampling 
        for li=1:(opt.repeat)
            sampf = randn(size(f));
            for i=1:length(ind)
                sampf(ind{i},:) = CLa{i}*sampf(ind{i},:);
            end
            ft=sqrt(1-lvs.^2).*f + lvs.*(sampf + B'*randn(m,1));
            at = max(min(ft, maxcut),mincut);
            ed = -feval(gp.likelih.fh_e, gp.likelih, y, ft, z);
            a=e-ed;
            if exp(a) > rand(1)
                f=ft;
                e=ed;
            else
                slrej=slrej+1;
            end
        end
        diagn.rej = slrej/opt.repeat;
        diagn.lvs = lvs;
        diagn.opt=opt;
        energ=[];
        f = f';        
        
      case 'CS+FIC'
        u = gp.X_u;
        cf_orig = gp.cf;
        ncf = length(gp.cf);
        n = size(x,1); m = size(u,1);

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
        
        % First evaluate the needed covariance matrices
        % if they are not in the memory
        % v defines that parameter is a vector
        [Kv_ff, Cv_ff] = gp_trvar(gp, x);  % 1 x f  vector
        K_fu = gp_cov(gp, x, u);         % f x u
        K_uu = gp_trcov(gp, u);          % u x u, noiseles covariance K_uu
        K_uu = (K_uu+K_uu')./2;          % ensure the symmetry of K_uu
        Luu = chol(K_uu)';

        B=Luu\(K_fu');
        Qv_ff=sum(B.^2)';
        Lav = Cv_ff-Qv_ff;   % 1 x f, Vector of diagonal elements        
        gp.cf = cf2;
        K_cs = gp_trcov(gp,x);
        La = sparse(1:n,1:n,Lav,n,n) + K_cs;
        gp.cf = cf_orig;
        
        LD = ldlchol(La);
        sLa = chol(La)';
        
        n=length(y);
        e = -feval(gp.likelih.fh_e, gp.likelih, y, f, z);

        % Adaptive control algorithm to find such a value for lvs 
        % so that the rejection rate of Metropolis is optimal. 
        slrej = 0;
        for li=1:100
            ft=sqrt(1-lvs.^2).*f + lvs.*(sLa*randn(n,1) + B'*randn(m,1));
            at = max(min(ft, maxcut),mincut);
            ed = -feval(gp.likelih.fh_e, gp.likelih, y, ft, z);
            a=e-ed;
            if exp(a) > rand(1)
                f=ft;
                e=ed;
                lvs=min(1,lvs*1.1);
            else
                lvs=max(1e-8,lvs/1.05);
            end
        end
        opt.sample_latent_scale=lvs;
        % Do the actual sampling 
        for li=1:(opt.repeat)
            ft=sqrt(1-lvs.^2).*f + lvs.*(sLa*randn(n,1) + B'*randn(m,1));
            at = max(min(ft, maxcut),mincut);
            ed = -feval(gp.likelih.fh_e, gp.likelih, y, ft, z);
            a=e-ed;
            if exp(a) > rand(1)
                f=ft;
                e=ed;
            else
                slrej=slrej+1;
            end
        end
        diagn.rej = slrej/opt.repeat;
        diagn.lvs = lvs;
        diagn.opt=opt;
        energ=[];
        f = f';
        
    end
end