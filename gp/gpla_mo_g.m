function [g, gdata, gprior] = gpla_mo_g(w, gp, x, y, varargin)
%GPLA_SOFTMAX_G   Evaluate gradient of Laplace approximation's marginal
%         log posterior estimate for softmax likelihood (GPLA_SOFTMAX_E)
%
%  Description
%    G = GPLA_SOFTMAX_G(W, GP, X, Y, OPTIONS) takes a full GP
%    hyper-parameter vector W, structure GP a matrix X of
%    input vectors and a matrix Y of target vectors, and evaluates
%    the gradient G of EP's marginal log posterior estimate . Each
%    row of X corresponds to one input vector and each row of Y
%    corresponds to one target vector.
%
%    [G, GDATA, GPRIOR] = GPLA_SOFTMAX_G(W, GP, X, Y, OPTIONS) also
%    returns the data and prior contributions to the gradient.
%
%    OPTIONS is optional parameter-value pair
%      z - optional observed quantity in triplet (x_i,y_i,z_i)
%          Some likelihoods may use this. For example, in case of
%          Poisson likelihood we have z_i=E_i, that is, expected
%          value for ith case.
%
%  See also
%    GPLA_SOFTMAX_E, GPLA_E, GPLA_SOFTMAX_PRED

% Copyright (c) 2010 Jaakko Riihim�ki, Pasi Jyl�nki

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.

ip=inputParser;
ip.FunctionName = 'GPLA_MO_G';
ip.addRequired('w', @(x) isvector(x) && isreal(x) && all(isfinite(x)));
ip.addRequired('gp',@isstruct);
ip.addRequired('x', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addRequired('y', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addParamValue('z', [], @(x) isreal(x) && all(isfinite(x(:))))
ip.parse(w, gp, x, y, varargin{:});
z=ip.Results.z;

gp = gp_unpak(gp, w);       % unpak the parameters
ncf = length(gp.cf);
n=size(x,1);
nout=size(y,2);

g = [];
gdata = [];
gprior = [];

% First Evaluate the data contribution to the error
switch gp.type
    % ============================================================
    % FULL
    % ============================================================
    case 'FULL'   % A full GP
        
        if isfield(gp, 'comp_cf')  % own covariance for each ouput component
            multicf = true;
            if length(gp.comp_cf) ~= nout
                error('GPLA_MO_E: the number of component vectors in gp.comp_cf must be the same as number of outputs.')
            end
        else
            multicf = false;
        end
        
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
        
        [e, edata, eprior, f, L, a, E, M, p] = gpla_mo_e(gp_pak(gp), gp, x, y, 'z', z);
        
        % softmax
        f2=reshape(f,n,nout);
        
        llg = feval(gp.lik.fh.llg,gp.lik, y, f2, 'latent', z);
        [pi2_vec, pi2_mat] = feval(gp.lik.fh.llg2, gp.lik, y, f2, 'latent', z);
        R = repmat(1./pi2_vec,1,n).*pi2_mat;
        RE = zeros(n,n*nout);
        for i1=1:nout
            RE(:,(1:n)+(i1-1)*n) = R((1:n)+(i1-1)*n,:)'*E(:,:,i1);
        end       
        
        inv_iWK=zeros(n,n,nout);
                
        % Matrices for computing the derivative of determinant term w.r.t. f
        A=zeros(nout, nout, n);
        Minv=M\(M'\eye(n));
        Minv=(Minv+Minv')./2;
        for cc1=1:nout
            EMinv=RE(:,(1:n)+(cc1-1)*n)'*Minv;
            KEMinv=K(:,:,cc1)*EMinv;
            for cc2=1:nout
                if cc2>=cc1
                    if cc1==cc2
                        EMtmp = - EMinv*RE(:,(1:n)+(cc2-1)*n);
                        EMtmp = EMtmp + E(:,:,cc1);
                        inv_iWK(:,:,cc1) = EMtmp;
                        A(cc1,cc1,:) = diag(K(:,:,cc1))-sum((K(:,:,cc1)*EMtmp).*K(:,:,cc1),2);
                    else
                        EMtmp = - KEMinv*RE(:,(1:n)+(cc2-1)*n);
                        A(cc1,cc2,:) = -sum(EMtmp.*K(:,:,cc2),2);
                        A(cc2,cc1,:) = -sum(EMtmp.*K(:,:,cc2),2);
                    end
                end
            end
        end
        
        % Derivative of determinant term w.r.t. f
        s2=zeros(n*nout,1);
        dw_mat = feval(gp.lik.fh.llg3, gp.lik, y, f, 'latent', z);
        for cc3=1:nout
            for ii1=1:n
                s2(ii1+(cc3-1)*n) = -0.5*trace(A(:,:,ii1)*dw_mat(:,:,cc3,ii1));
            end
        end
        
        % Loop over the covariance functions
        for i=1:ncf
            DKllg=zeros(size(a));
            EDKllg=zeros(size(a));
            DKffba=zeros(n*nout,1);
            
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
            
            i1=0;
            if ~isempty(gprior)
                i1 = length(gprior);
            end
            
            % Gradients from covariance functions
            gpcf = gp.cf{i};
            DKff = feval(gpcf.fh.cfg, gpcf, x);
            gprior_cf = -feval(gpcf.fh.lpg, gpcf);
            
            for i2 = 1:length(DKff)
                i1 = i1+1;
                DKffb=DKff{i2};
                
                % Derivative of explicit terms
                trace_sum_tmp=0;
                for z1=1:nout
                    if do(z1)
                        DKffba((1:n)+(z1-1)*n)=DKffb*a((1:n)+(z1-1)*n);
                        trace_sum_tmp = trace_sum_tmp + sum(sum( inv_iWK(:,:,z1) .* DKffb ));
                    end
                end
                s1 = 0.5 * a'*DKffba - 0.5.*trace_sum_tmp;
                                
                % Derivative of f w.r.t. theta
                for z1=1:nout
                    if do(z1)
                        DKllg((1:n)+(z1-1)*n)=DKffb*llg((1:n)+(z1-1)*n);
                        EDKllg((1:n)+(z1-1)*n)=E(:,:,z1)*DKllg((1:n)+(z1-1)*n);
                    end
                end
                s3 = EDKllg - RE'*(M\(M'\(RE*DKllg)));
                for z1=1:nout
                    s3((1:n)+(z1-1)*n)=K(:,:,z1)*s3((1:n)+(z1-1)*n);
                end
                s3 = DKllg - s3;

                gdata(i1) = -(s1 + s2'*s3);
                gprior(i1) = gprior_cf(i2);
                
            end
            
            % Set the gradients of hyper-hyperparameter
            if length(gprior_cf) > length(DKff)
                for i2=length(DKff)+1:length(gprior_cf)
                    i1 = i1+1;
                    gdata(i1) = 0;
                    gprior(i1) = gprior_cf(i2);
                end
            end
        end
                
%         % =================================================================
%         % Gradient with respect to likelihood function parameters
%         if ~isempty(strfind(gp.infer_params, 'likelihood')) ...
%             && ~isempty(gp.lik.fh.pak(gp.lik))
%             
%             gdata_likelih = 0;
%             lik = gp.lik;
%             
%             g_logPrior = feval(lik.fh.gprior, lik);
%             if ~isempty(g_logPrior)
%             
%                 DW_sigma = feval(lik.fh.llg3, lik, y, f, 'latent2+hyper', z);
%                 DL_sigma = feval(lik.fh.llg, lik, y, f, 'hyper', z);
%                 b = K * feval(lik.fh.llg2, lik, y, f, 'latent+hyper', z);
%                 s3 = b - K*(R*b);
%                 nl= size(DW_sigma,2);
%                 
%                 gdata_lik = - DL_sigma - 0.5.*sum(repmat(C2,1,nl).*DW_sigma) - s2'*s3;
% 
%                 % set the gradients into vectors that will be returned
%                 gdata = [gdata gdata_lik];
%                 gprior = [gprior g_logPrior];
%                 i1 = length(g_logPrior);
%                 i2 = length(gdata_lik);
%                 if i1  > i2
%                     gdata = [gdata zeros(1,i1-i2)];
%                 end
%             end
%         end
        
        g = gdata + gprior;

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
       
    end
    
    assert(isreal(gdata))
    assert(isreal(gprior))
end
