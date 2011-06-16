function [Ef, Varf, lpyt, Ey, Vary] = gpla_mo_pred(gp, x, y, xt, varargin)
%function [Ef, Varf, Ey, Vary, Pyt] = gpla_multinom_pred(gp, x, y, xt, varargin)
%GPLA_MO_PRED Predictions with Gaussian Process Laplace
%                approximation with multinom likelihood
%
%  Description
%    [EFT, VARFT] = GPLA_MO_PRED(GP, X, Y, XT, OPTIONS) takes
%    a GP structure GP together with a matrix XT of input vectors,
%    matrix X of training inputs and vector Y of training targets,
%    and evaluates the predictive distribution at inputs XT. Returns
%    a posterior mean EFT and variance VARFT of latent variables.
%
%    [EF, VARF, LPYT] = GPLA_MO_PRED(GP, X, Y, XT, 'yt', YT, ...)
%    returns also logarithm of the predictive density PYT of the observations YT
%    at input locations XT. This can be used for example in the
%    cross-validation. Here Y has to be vector.
%
%    [EF, VARF, LPYT, EYT, VARYT] = GPLA_MO_PRED(GP, X, Y, XT, 'yt', YT, ...)
%    returns also the posterior predictive mean EYT and variance VARYT.
%
%     OPTIONS is optional parameter-value pair
%       predcf - is index vector telling which covariance functions are 
%                used for prediction. Default is all (1:gpcfn). See 
%                additional information below.
%       tstind - is a vector/cell array defining, which rows of X belong 
%                to which training block in *IC type sparse models. Deafult 
%                is []. In case of PIC, a cell array containing index 
%                vectors specifying the blocking structure for test data.
%                IN FIC and CS+FIC a vector of length n that points out the 
%                test inputs that are also in the training set (if none,
%                set TSTIND = []).
%       yt     - is optional observed yt in test points
%       z      - optional observed quantity in triplet (x_i,y_i,z_i)
%                Some likelihoods may use this. For example, in
%                case of Poisson likelihood we have z_i=E_i, that
%                is, expected value for ith case.
%       zt     - optional observed quantity in triplet (xt_i,yt_i,zt_i)
%                Some likelihoods may use this. For example, in
%                case of Poisson likelihood we have zt_i=Et_i, that
%                is, expected value for ith case.
%
%  See also
%    GPLA_MO_E, GPLA_MO_G, GP_PRED, DEMO_MULTICLASS
%
% Copyright (c) 2010 Jaakko Riihim�ki

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

  ip=inputParser;
  ip.FunctionName = 'GPLA_MO_PRED';
  ip.addRequired('gp', @isstruct);
  ip.addRequired('x', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
  ip.addRequired('y', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
  ip.addRequired('xt', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
  ip.addParamValue('yt', [], @(x) isreal(x) && all(isfinite(x(:))))
  ip.addParamValue('z', [], @(x) isreal(x) && all(isfinite(x(:))))
  ip.addParamValue('zt', [], @(x) isreal(x) && all(isfinite(x(:))))
  ip.addParamValue('predcf', [], @(x) isempty(x) || iscell(x) && isvector(x))
  ip.addParamValue('tstind', [], @(x) isempty(x) || iscell(x) ||...
                 (isvector(x) && isreal(x) && all(isfinite(x)&x>0)))
  ip.parse(gp, x, y, xt, varargin{:});
  yt=ip.Results.yt;
  z=ip.Results.z;
  zt=ip.Results.zt;
  predcf=ip.Results.predcf;
  tstind=ip.Results.tstind;

  Ey=[];
  Vary=[];
  
    [tn, nout] = size(y);
    
    switch gp.type
        % ============================================================
        % FULL
        % ============================================================
      case 'FULL'
        [e, edata, eprior, f, L, a, E, M, p] = gpla_mo_e(gp_pak(gp), gp, x, y, 'z', z);
        
        if isfield(gp, 'comp_cf')  % own covariance for each ouput component
            multicf = true;
            if length(gp.comp_cf) ~= nout
                error('GPLA_MO_E: the number of component vectors in gp.comp_cf must be the same as number of outputs.')
            end
            if ~isempty(predcf)
                if ~iscell(predcf) || length(predcf)~=nout
                    error(['GPLA_MO_PRED: if own covariance for each output component is used,'...
                           'predcf has to be cell array and contain nout (vector) elements.   '])
                end
            else
                predcf = gp.comp_cf;
            end
        else
            multicf = false;
            for i1=1:nout
                predcf2{i1} = predcf;
            end
            predcf=predcf2;
        end
      
        ntest=size(xt,1);
        K_nf = zeros(ntest,tn,nout);
        if multicf
            for i1=1:nout
                K_nf(:,:,i1) = gp_cov(gp,xt,x,predcf{i1});
            end
        else
            for i1=1:nout
                K_nf(:,:,i1) = gp_cov(gp,xt,x,predcf{i1});
            end
        end
        
        nout=size(y,2);
        f2=reshape(f,tn,nout);
                
        llg_vec = gp.lik.fh.llg(gp.lik, y, f2, 'latent', z);
        llg = reshape(llg_vec,size(y));
                   
        %mu_star = K_nf*reshape(a,tn,nout);
        a=reshape(a,size(y));
        for i1 = 1:nout
         %   Ef(:,i1) = K_nf(:,:,i1)*llg(:,i1);
            Ef(:,i1) = K_nf(:,:,i1)*a(:,i1);
        end
        
        if nargout > 1
            [pi2_vec, pi2_mat] = gp.lik.fh.llg2(gp.lik, y, f2, 'latent', z);
            Varf=zeros(nout, nout, ntest);
            
            R=(repmat(1./pi2_vec,1,tn).*pi2_mat);
            for i1=1:nout
                b=E(:,:,i1)*K_nf(:,:,i1)';
                c_cav = R((1:tn)+(i1-1)*tn,:)*(M\(M'\(R((1:tn)+(i1-1)*tn,:)'*b)));
                
                for j1=1:nout
                    c=E(:,:,j1)*c_cav;
                    Varf(i1,j1,:)=sum(c.*K_nf(:,:,j1)');
                end
                
                kstarstar = gp_trvar(gp,xt,predcf{i1});
                Varf(i1,i1,:) = squeeze(Varf(i1,i1,:)) + kstarstar - sum(b.*K_nf(:,:,i1)')';
            end
        end
        % ============================================================
        % FIC
        % ============================================================    
      case 'FIC'        % Predictions with FIC sparse approximation for GP
        % ============================================================
        % PIC
        % ============================================================
      case {'PIC' 'PIC_BLOCK'}        % Predictions with PIC sparse approximation for GP
        % ============================================================
        % CS+FIC
        % ============================================================
      case 'CS+FIC'        % Predictions with CS+FIC sparse approximation for GP
    end
    
    % ============================================================
    % Evaluate also the predictive mean and variance of new observation(s)
    % ============================================================
   if nargout > 2 && isempty(yt)
       error('yt has to be provided to get lpyt.')
   end
   if nargout > 3
       [lpyt, Ey, Vary] = gp.lik.fh.predy(gp.lik, Ef, Varf, [], zt);
   elseif nargout > 2
       lpyt = gp.lik.fh.predy(gp.lik, Ef, Varf, yt, zt);
   end
end