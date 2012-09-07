function lik = lik_softmax(varargin)
%LIK_SOFTMAX    Create a softmax likelihood structure 
%
%  Description
%    LIK = LIK_SOFTMAX creates Softmax likelihood for multi-class
%    classification problem. The observed class label with C
%    classes is given as 1xC vector where C-1 entries are 0 and the
%    observed class label is 1.
%
%  See also
%    GP_SET, LIK_*

% Copyright (c) 2010 Jaakko Riihim�ki, Pasi Jyl�nki
% Copyright (c) 2010 Aki Vehtari

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

  ip=inputParser;
  ip.FunctionName = 'LIK_SOFTMAX2';
  ip.addOptional('lik', [], @isstruct);
  ip.parse(varargin{:});
  lik=ip.Results.lik;

  if isempty(lik)
    init=true;
    lik.type = 'Softmax';
    lik.nondiagW=true;
  else
    if ~isfield(lik,'type') && ~isequal(lik.type,'Softmax')
      error('First argument does not seem to be a valid likelihood function structure')
    end
    init=false;
  end

  if init
    % Set the function handles to the subfunctions
    lik.fh.pak = @lik_softmax_pak;
    lik.fh.unpak = @lik_softmax_unpak;
    lik.fh.ll = @lik_softmax_ll;
    lik.fh.llg = @lik_softmax_llg;    
    lik.fh.llg2 = @lik_softmax_llg2;
    lik.fh.llg3 = @lik_softmax_llg3;
    lik.fh.tiltedMoments = @lik_softmax_tiltedMoments;
    lik.fh.predy = @lik_softmax_predy;
    lik.fh.recappend = @lik_softmax_recappend;
  end
  
end

function [w,s] = lik_softmax_pak(lik)
%LIK_LOGIT_PAK  Combine likelihood parameters into one vector.
%
%  Description 
%    W = LIK_LOGIT_PAK(LIK) takes a likelihood structure LIK and
%    returns an empty verctor W. If Logit likelihood had
%    parameters this would combine them into a single row vector
%    W (see e.g. lik_negbin).
%     
%
%  See also
%    LIK_NEGBIN_UNPAK, GP_PAK
  
  w = []; s = {};
end


function [lik, w] = lik_softmax_unpak(lik, w)
%LIK_LOGIT_UNPAK  Extract likelihood parameters from the vector.
%
%  Description
%    W = LIK_LOGIT_UNPAK(W, LIK) Doesn't do anything.
% 
%    If Logit likelihood had parameters this would extracts them
%    parameters from the vector W to the LIK structure.
%     
%
%  See also
%    LIK_LOGIT_PAK, GP_UNPAK

  lik=lik;
  w=w;
end


function ll = lik_softmax_ll(lik, y, f2, z)
%LIK_LOGIT_LL  Log likelihood
%
%  Description
%    LL = LIK_LOGIT_LL(LIK, Y, F) takes a likelihood structure
%    LIK, class labels Y (NxC matrix), and latent values F (NxC
%    matrix). Returns the log likelihood, log p(y|f,z).
%
%  See also
%    LIK_LOGIT_LLG, LIK_LOGIT_LLG3, LIK_LOGIT_LLG2, GPLA_E

  if ~isempty(find(y~=1 & y~=0))
    error('lik_softmax: The class labels have to be {0,1}')
  end
  
  % softmax:
  ll = y(:)'*f2(:) - sum(log(sum(exp(f2),2)));
  
end


function llg = lik_softmax_llg(lik, y, f2, param, z)
%LIK_LOGIT_LLG    Gradient of the log likelihood
%
%  Description
%    LLG = LIK_LOGIT_LLG(LIK, Y, F, PARAM) takes a likelihood
%    structure LIK, class labels Y, and latent values F. Returns
%    the gradient of the log likelihood with respect to PARAM. At
%    the moment PARAM can be 'param' or 'latent'.
%
%  See also
%    LIK_LOGIT_LL, LIK_LOGIT_LLG2, LIK_LOGIT_LLG3, GPLA_E
  
  if ~isempty(find(y~=1 & y~=0))
    error('lik_softmax: The class labels have to be {0,1}')
  end

  expf2 = exp(f2);
  pi2 = expf2./(sum(expf2, 2)*ones(1,size(y,2)));
  pi_vec=pi2(:);
  llg = y(:)-pi_vec;
  
end


function [pi_vec, pi_mat] = lik_softmax_llg2(lik, y, f2, param, z)
%LIK_LOGIT_LLG2  Second gradients of the log likelihood
%
%  Description        
%    LLG2 = LIK_LOGIT_LLG2(LIK, Y, F, PARAM) takes a likelihood
%    structure LIK, class labels Y, and latent values F. Returns
%    the Hessian of the log likelihood with respect to PARAM. At
%    the moment PARAM can be only 'latent'. LLG2 is a vector with
%    diagonal elements of the Hessian matrix (off diagonals are
%    zero).
%
%  See also
%    LIK_LOGIT_LL, LIK_LOGIT_LLG, LIK_LOGIT_LLG3, GPLA_E

% softmax:    
  expf2 = exp(f2);
  pi2 = expf2./(sum(expf2, 2)*ones(1,size(y,2)));
  pi_vec=pi2(:);
  [n,nout]=size(y);
  pi_mat=zeros(nout*n, n);
  for i1=1:nout
    pi_mat((1+(i1-1)*n):(nout*n+1):end)=pi2(:,i1);
  end
  %     D=diag(pi_vec);
  %     llg2=-D+pi_mat*pi_mat';
  
end    

function dw_mat = lik_softmax_llg3(lik, y, f, param, z)
%LIK_LOGIT_LLG3  Third gradients of the log likelihood
%
%  Description
%    LLG3 = LIK_LOGIT_LLG3(LIK, Y, F, PARAM) takes a likelihood
%    structure LIK, class labels Y, and latent values F and
%    returns the third gradients of the log likelihood with
%    respect to PARAM. At the moment PARAM can be only 'latent'. 
%    LLG3 is a vector with third gradients.
%
%  See also
%    LIK_LOGIT_LL, LIK_LOGIT_LLG, LIK_LOGIT_LLG2, GPLA_E, GPLA_G
  
  if ~isempty(find(y~=1 & y~=0))
    error('lik_softmax: The class labels have to be {0,1}')
  end
  
  [n,nout] = size(y);
  f2 = reshape(f,n,nout);
  
  expf2 = exp(f2);
  pi2 = expf2./(sum(expf2, 2)*ones(1,nout));
  pi_vec=pi2(:);
  
  dw_mat=zeros(nout,nout,nout,n);
  
  for cc3=1:nout
    for ii1=1:n
      
      pic=pi_vec(ii1:n:(nout*n));
      for cc1=1:nout
        for cc2=1:nout
          
          % multinom third derivatives
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
          
          dw_mat(cc1,cc2,cc3,ii1)=cc_sum_tmp;
        end
      end
    end
  end
end

function [logM_0, m_1, sigm2hati1] = lik_softmax_tiltedMoments(lik, y, i1, sigm2_i, myy_i, z)
end

function [lpy, Ey, Vary] = lik_softmax_predy(lik, Ef, Varf, yt, zt)
%LIK_SOFTMAX_PREDY  Returns the predictive mean, variance and density of
%y
%
%  Description         
%    LPY = LIK_SOFTMAX_PREDY(LIK, EF, VARF YT, ZT)
%    Returns logarithm of the predictive density PY of YT, that is 
%        p(yt | y, zt) = \int p(yt | f, zt) p(f|y) df.
%    This requires also the succes counts YT, numbers of trials ZT.
%
%    [EY, VARY] = LIK_SOFTMAX_PREDY(LIK, EF, VARF) takes a
%    likelihood structure LIK, posterior mean EF and posterior
%    Variance VARF of the latent variable and returns the
%    posterior predictive mean EY and variance VARY of the
%    observations related to the latent variables
%        
%
%  See also 
%    GPEP_PRED, GPLA_PRED, GPMC_PRED
  
  
  if ~isempty(find(yt~=1 & yt~=0))
    error('lik_softmax: The class labels have to be {0,1}')
  end
  
  S=10000;
  [ntest,nout]=size(yt);
  pi=zeros(ntest,nout);
  lpy=zeros(ntest,nout);
  [notused,notused,c] =size(Varf);
  if c>1
    mcmc=false;
  else
    mcmc=true;
  end
  for i1=1:ntest
    if mcmc
      Sigm_tmp = diag(Varf(i1,:));
    else
      Sigm_tmp=(Varf(:,:,i1)'+Varf(:,:,i1))./2;
    end
    f_star=mvnrnd(Ef(i1,:), Sigm_tmp, S);
    
    tmp = exp(f_star);
    tmp = tmp./(sum(tmp, 2)*ones(1,size(tmp,2)));
    pi(i1,:)=mean(tmp);
    ytmp = repmat(yt(i1,:),S,1);
    lpy(i1,:) = log(mean(tmp.^(ytmp).*(1-tmp).^(1-ytmp)));
  end
  if nargout > 1
    Ey = 2*pi-1;
    Vary = 1-(2*pi-1).^2;
  end
end

function reclik = lik_softmax_recappend(reclik, ri, lik)
%RECAPPEND  Append the parameters to the record
%
%  Description 
%    RECLIK = GPCF_LOGIT_RECAPPEND(RECLIK, RI, LIK) takes a
%    likelihood record structure RECLIK, record index RI and
%    likelihood structure LIK with the current MCMC samples of
%    the parameters. Returns RECLIK which contains all the old
%    samples and the current samples from LIK.
% 
%  See also
%    GP_MC

  if nargin == 2
    reclik.type = 'softmax';

    % Set the function handles
    reclik.fh.pak = @lik_softmax_pak;
    reclik.fh.unpak = @lik_softmax_unpak;
    reclik.fh.ll = @lik_softmax_ll;
    reclik.fh.llg = @lik_softmax_llg;    
    reclik.fh.llg2 = @lik_softmax_llg2;
    reclik.fh.llg3 = @lik_softmax_llg3;
    reclik.fh.tiltedMoments = @lik_softmax_tiltedMoments;
    reclik.fh.predy = @lik_softmax_predy;
    reclik.fh.recappend = @lik_softmax_recappend;
  end
end
