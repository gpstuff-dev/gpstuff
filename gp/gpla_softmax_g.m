function [g, gdata, gprior] = gpla_softmax_g(w, gp, x, y, varargin)
%GPLA_G   Evaluate gradient of Laplace approximation's marginal
%         log posterior estimate for softmax likelihood (GPLA_SOFTMAX_E)
%
%     Description
%	G = GPLA_SOFTMAX_G(W, GP, X, Y, OPTIONS) takes a full GP hyper-parameter
%        vector W, data structure GP a matrix X of input vectors
%        and a matrix Y of target vectors, and evaluates the
%        gradient G of EP's marginal log posterior estimate . Each
%        row of X corresponds to one input vector and each row of Y
%        corresponds to one target vector.
%
%	[G, GDATA, GPRIOR] = GPLA_SOFTMAX_G(W, GP, X, Y, OPTIONS) also returns
%        the data and prior contributions to the gradient.
%
%     OPTIONS is optional parameter-value pair
%       'z'    is optional observed quantity in triplet (x_i,y_i,z_i)
%              Some likelihoods may use this. For example, in case of
%              Poisson likelihood we have z_i=E_i, that is, expected
%              value for ith case.
%
%	See also
%       GPLA_SOFTMAX_E, GPLA_E, LA_PRED

% Copyright (c) 2010 Jaakko Riihimäki, Pasi Jylänki

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.

ip=inputParser;
ip.FunctionName = 'GPLA_SOFTMAX_G';
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
        
        K = gp_trcov(gp,x);
        
        [e, edata, eprior, f, L, a, E, M, p] = gpla_softmax_e(gp_pak(gp), gp, x, y, 'z', z);
        
        % softmax
        f2=reshape(f,n,nout);
        expf2 = exp(f2);
        pi2 = expf2./(sum(expf2, 2)*ones(1,nout));
        pi_vec=pi2(:);
        
        DKBy_pi=zeros(n*nout,1);
        DKffba=zeros(n*nout,1);
        inv_iWK=zeros(n,n,nout);
        EDKBy=zeros(n*nout,1);
        EME=zeros(n*nout,1);
        E_diff=zeros(n*nout,1);
        
        % Matrices for computing the derivative of determinant term w.r.t. f
        A=zeros(nout, nout, n);
        Minv=M\(M'\eye(n));
        Minv=(Minv+Minv')./2;
        for cc1=1:nout
            EMinv=E(:,:,cc1)*Minv;
            KEMinv=K*EMinv;
            for cc2=1:nout
                if cc2>=cc1
                    if cc1==cc2
                        EMtmp= - EMinv*E(:,:,cc2);
                        EMtmp=EMtmp + E(:,:,cc1);
                        inv_iWK(:,:,cc1)=EMtmp;
                        A(cc1,cc1,:)=diag(K)-sum((K*EMtmp).*K,2);
                    else
                        EMtmp= - KEMinv*E(:,:,cc2);
                        A(cc1,cc2,:)=-sum(EMtmp.*K,2);
                        A(cc2,cc1,:)=-sum(EMtmp.*K,2);
                    end
                end
            end
        end
        
        s2=zeros(n*nout,1);
        
        for cc3=1:nout
            for ii1=1:n
                
                dw_mat=zeros(nout,nout);
                pic=pi_vec(ii1:n:(nout*n));
                
                for cc1=1:nout
                    for cc2=1:nout
                        
                        % softmax third derivatives
                        cc_sum_tmp=0;
                        if cc1==cc2 && cc1==cc3 && cc2==cc3
                            cc_sum_tmp=cc_sum_tmp+pic(cc1);
                        end
                        if cc1==cc2
                            cc_sum_tmp=cc_sum_tmp-pic(cc1)*pic(cc3);
                        end
                        if cc2==cc3
                            cc_sum_tmp=cc_sum_tmp-pic(cc1)*pic(cc2);
                        end
                        if cc1==cc3
                            cc_sum_tmp=cc_sum_tmp-pic(cc1)*pic(cc2);
                        end
                        cc_sum_tmp=cc_sum_tmp+2*pic(cc1)*pic(cc2)*pic(cc3);
                        
                        dw_mat(cc1,cc2)=cc_sum_tmp;
                    end
                end
                % Derivative of determinant term w.r.t. f
                s2(ii1+(cc3-1)*n)=-0.5*trace(A(:,:,ii1)*dw_mat);
            end
        end
        
        
        for i=1:ncf
            
            i1=0;
            if ~isempty(gprior)
                i1 = length(gprior);
            end
            
            % Gradients from covariance functions
            gpcf = gp.cf{i};
            [DKff, gprior_cf] = feval(gpcf.fh_ghyper, gpcf, x);
            
            for i2 = 1:length(DKff)
                i1 = i1+1;
                DKffb=DKff{i2};
                
                trace_sum_tmp=0;
                for z1=1:nout
                    DKffba((1:n)+(z1-1)*n)=DKffb*a((1:n)+(z1-1)*n);
                    trace_sum_tmp=trace_sum_tmp+sum(sum( inv_iWK(:,:,z1) .* DKffb ));
                end
                % Derivative of explicit terms
                s1 = 0.5 * a'*DKffba - 0.5*trace_sum_tmp;
                
                y_pi=y(:)-pi_vec;
                for z1=1:nout
                    DKBy_pi((1:n)+(z1-1)*n)=DKffb*y_pi((1:n)+(z1-1)*n);
                    EDKBy((1:n)+(z1-1)*n)=E(:,:,z1)*DKBy_pi((1:n)+(z1-1)*n);
                end
                
                EDKBys=sum(reshape(EDKBy, n, nout),2);
                MMRc=Minv*EDKBys;
                
                for z1=1:nout
                    EME((1:n)+(z1-1)*n) = E(:,:,z1)*MMRc;
                end
                EDK_EME=EDKBy-EME;
                
                for z1=1:nout
                    E_diff((1:n)+(z1-1)*n)=K*EDK_EME((1:n)+(z1-1)*n);
                end
                
                % Derivative of f w.r.t. theta
                s3=DKBy_pi-E_diff;
                
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
%             && ~isempty(gp.likelih.fh_pak(gp.likelih))
%             
%             gdata_likelih = 0;
%             likelih = gp.likelih;
%             
%             g_logPrior = feval(likelih.fh_priorg, likelih);
%             if ~isempty(g_logPrior)
%             
%                 DW_sigma = feval(likelih.fh_g3, likelih, y, f, 'latent2+hyper', z);
%                 DL_sigma = feval(likelih.fh_g, likelih, y, f, 'hyper', z);
%                 b = K * feval(likelih.fh_g2, likelih, y, f, 'latent+hyper', z);
%                 s3 = b - K*(R*b);
%                 nl= size(DW_sigma,2);
%                 
%                 gdata_likelih = - DL_sigma - 0.5.*sum(repmat(C2,1,nl).*DW_sigma) - s2'*s3;
% 
%                 % set the gradients into vectors that will be returned
%                 gdata = [gdata gdata_likelih];
%                 gprior = [gprior g_logPrior];
%                 i1 = length(g_logPrior);
%                 i2 = length(gdata_likelih);
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
