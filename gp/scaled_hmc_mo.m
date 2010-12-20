function [f, energ, diagn] = scaled_hmc_mo(f, opt, gp, x, y, z)
%SCALED_HMC  A scaled hybrid Monte Carlo sampling for latent values
%
%   Description
%    [F, ENERG, DIAG] = SCALED_HMC(F, OPT, GP, X, Y) takes the
%    current latent values F, options structure OPT, Gaussian
%    process structure GP, inputs X and outputs Y. Samples new
%    latent values and returns also energies ENERG and diagnostics
%    DIAG. The latent values are sampled from their conditional
%    posterior p(f|y,th).
%
%    The latent values are whitened with an approximate posterior
%    covariance before the sampling. This reduces the
%    autocorrelation and speeds up the mixing of the sampler. See
%    Vanhatalo and Vehtari (2007) for details on implementation.
%
%    The options structure needs to be similar to hybrid
%    Monte Carlo, HMC2. See HMC2 and HMC2_OPT for details.
%
%  See also
%    GP_MC, HMC2, HMC2_OPT
 
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
    [n,nout] = size(y);

    Linv=[];
    L2=[];
    mincut = -300;

    % Transform the latent values
    getL(f, gp, x, y, z);             % Evaluate the help matrix for transformation
    w = (L2\f)';                   % Rotate f towards prior
    
    %gradcheck(w, @f_e, @f_g, gp, x, y, f);
    
    % Conduct the HMC sampling for the transformed latent values
    hmc2('state',latent_rstate)
    rej = 0;
    for li=1:opt.repeat 
        [w, energ, diagn] = hmc2(@f_e, w, opt, @f_g, gp, x, y, z);
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
    f=L2*w;

    % Record the options
    opt.rstate = hmc2('state');
    diagn.opt = opt;
    diagn.rej = rej;
    diagn.lvs = opt.stepadj;

    function [g, gdata, gprior] = f_g(w, gp, x, y, z)
    %F_G     Evaluate gradient function for transformed GP latent values 
    %               
        
    % Force f and E to be a column vector
        w=w(:);
        
        f = L2*w;
        f = max(f,mincut);
        f2 = reshape(f,n,nout);
        gdata = - feval(gp.lik.fh.llg, gp.lik, y, f2, 'latent', z);
        
        b=Linv*f;
        gprior=Linv'*b;
        
        g = (L2'*(gdata + gprior))';
    end

    function [e, edata, eprior] = f_e(w, gp, x, y, z)
    % F_E     Evaluate energy function for transformed GP latent values 
        
    % force f and E to be a column vector
        w=w(:);
        f = L2*w;
        f = max(f,mincut);
        
        B=Linv*f;
        
        eprior=.5*sum(B.^2);
            
        f2 = reshape(f,n,nout);
        edata =  - feval(gp.lik.fh.ll, gp.lik, y, f2, z);
        e=edata + eprior;
    end

    function getL(w, gp, x, y, z)
    % GETL        Evaluate the transformation matrix (or matrices)
        
    % Evaluate the Lambda (La) for specific model
    
       [pi2_vec, pi2_mat] = feval(gp.lik.fh.llg2, gp.lik, y, zeros(n,nout), 'latent', z);
       pi2 = reshape(pi2_vec,size(y));
       
       if isfield(gp, 'comp_cf')  % own covariance for each ouput component
          multicf = true;
          if length(gp.comp_cf) ~= nout
              error('GPLA_MO_E: the number of component vectors in gp.comp_cf must be the same as number of outputs.')
          end
       else
          multicf = false;
       end
       
       % Evaluate the blocks of the covariance matrix
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

       R = repmat(1./pi2_vec,1,n).*pi2_mat;
       RE = zeros(n,n*nout);
       for i1=1:nout
           Dc=sqrt( pi2(:,i1) );
           Lc=(Dc*Dc').*K(:,:,i1);
           Lc(1:n+1:end)=Lc(1:n+1:end)+1;
           Lc=chol(Lc);
           L(:,:,i1)=Lc;
           
           Ec=Lc'\diag(Dc);
           Ec=Ec'*Ec;
           E(:,:,i1)=Ec;
           RER(:,:,i1) = R((1:n)+(i1-1)*n,:)'*Ec*R((1:n)+(i1-1)*n,:);
           RE(:,(1:n)+(i1-1)*n) = R((1:n)+(i1-1)*n,:)'*E(:,:,i1);
       end
       M=chol(sum(RER,3));
              
       % from this on conduct with full matrices
       % NOTE! Speed up should be considered in the future
       C = zeros(n*nout,n*nout);
       EE = zeros(n*nout,n*nout);
       for i1 = 1:nout
           C((1:n)+(i1-1)*n,(1:n)+(i1-1)*n) = K(:,:,i1);
           EE((1:n)+(i1-1)*n,(1:n)+(i1-1)*n) = E(:,:,i1);
       end
       Cpost = C - C*(EE-RE'*(M\(M'\RE)))*C;
       
        % Evaluate a approximation for posterior variance
        % Take advantage of the matrix inversion lemma
        %        L=chol(inv(inv(C) + diag(E)))';
        
        Linv = inv(chol(C)');
              
        %L2 = C/chol(diag(1./E) + C);
        %L2 = chol(C - L2*L2')';
        L2 = chol(Cpost)';
    end
end
