function lik = lik_softmax(varargin)
%LIK_SOFTMAX    Create a softmax likelihood structure 
%
%  Description
%    LIK = LIK_SOFTMAX creates Softmax likelihood for multi-class
%    classification problem. The observed class label with C
%    classes is given as 1xC vector where C-1 entries are 0 and the
%    observed class label is 1.
%
%       See also
%       GP_SET, LIK_*

% Copyright (c) 2010 Jaakko Riihimäki, Pasi Jylänki
% Copyright (c) 2010 Aki Vehtari

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.

% allow use with or without init and set options
  if nargin<1
    do='init';
  elseif ischar(varargin{1})
    switch varargin{1}
      case 'init'
        do='init';varargin(1)=[];
      case 'set'
        do='set';varargin(1)=[];
      otherwise
        do='init';
    end
  elseif isstruct(varargin{1})
    do='set';
  else
    error('Unknown first argument');
  end

  switch do
    case 'init'
      % Initialize the likelihood structure
      lik.type = 'Softmax';
      
      % Set the function handles to the nested functions
      lik.fh_pak = @lik_softmax_pak;
      lik.fh_unpak = @lik_softmax_unpak;
      lik.fh_ll = @lik_softmax_ll;
      lik.fh_llg = @lik_softmax_llg;    
      lik.fh_llg2 = @lik_softmax_llg2;
      lik.fh_llg3 = @lik_softmax_llg3;
      lik.fh_tiltedMoments = @lik_softmax_tiltedMoments;
      lik.fh_predy = @lik_softmax_predy;
      lik.fh_recappend = @lik_softmax_recappend;

      % No paramaters to init
      if numel(varargin) > 0
        error('Wrong number of arguments')
      end

    case 'set'
      % No paramaters to set
      if numel(varargin)~=1
        error('Wrong number of arguments')
      end
      
      % Pass the likelihood
      lik = varargin{1};

  end


  function w = lik_softmax_pak(lik)
  %LIK_LOGIT_PAK    Combine likelihood parameters into one vector.
  %
  %   Description 
  %   W = LIK_LOGIT_PAK(LIK) takes a likelihood data
  %   structure LIK and returns an empty verctor W. If Logit
  %   likelihood had hyperparameters this would combine them into a
  %   single row vector W (see e.g. lik_negbin).
  %     
  %
  %   See also
  %   LIK_NEGBIN_UNPAK, GP_PAK
    
    w = [];
  end


  function [lik, w] = lik_softmax_unpak(w, lik)
  %LIK_LOGIT_UNPAK  Extract likelihood parameters from the vector.
  %
  %   Description
  %   W = LIK_LOGIT_UNPAK(W, LIK) Doesn't do anything.
  % 
  %   If Logit likelihood had hyperparameters this would extracts
  %   them parameters from the vector W to the LIK structure.
  %     
  %
  %   See also
  %   LIK_LOGIT_PAK, GP_UNPAK

    lik=lik;
    w=w;
  end


  function logLik = lik_softmax_ll(lik, y, f2, z)
  %LIK_LOGIT_LL    Log likelihood
  %
  %   Description
  %   E = LIK_LOGIT_LL(LIK, Y, F) takes a likelihood
  %   data structure LIK, class labels Y (NxC matrix), and latent values
  %   F (NxC matrix). Returns the log likelihood, log p(y|f,z).
  %
  %   See also
  %   LIK_LOGIT_LLG, LIK_LOGIT_LLG3, LIK_LOGIT_LLG2, GPLA_E

    if ~isempty(find(y~=1 & y~=0))
      error('lik_softmax: The class labels have to be {0,1}')
    end
    
    % softmax:
    logLik = y(:)'*f2(:) - sum(log(sum(exp(f2),2)));
    
  end


  function deriv = lik_softmax_llg(lik, y, f2, param, z)
  %LIK_LOGIT_LLG    Gradient of log likelihood (energy)
  %
  %   Description
  %   G = LIK_LOGIT_LLG(LIK, Y, F, PARAM) takes a likelihood
  %   data structure LIK, class labels Y, and latent values
  %   F. Returns the gradient of log likelihood with respect to
  %   PARAM. At the moment PARAM can be 'hyper' or 'latent'.
  %
  %   See also
  %   LIK_LOGIT_LL, LIK_LOGIT_LLG2, LIK_LOGIT_LLG3, GPLA_E
    
    if ~isempty(find(y~=1 & y~=0))
      error('lik_softmax: The class labels have to be {0,1}')
    end

    expf2 = exp(f2);
    pi2 = expf2./(sum(expf2, 2)*ones(1,size(y,2)));
    pi_vec=pi2(:);
    deriv = y(:)-pi_vec;
  end


  function g2 = lik_softmax_llg2(lik, y, f2, param, z)
  %LIK_LOGIT_LLG2  Second gradients of log likelihood (energy)
  %
  %   Description        
  %   G2 = LIK_LOGIT_LLG2(LIK, Y, F, PARAM) takes a likelihood
  %   data structure LIK, class labels Y, and latent values
  %   F. Returns the hessian of log likelihood with respect to
  %   PARAM. At the moment PARAM can be only 'latent'. G2 is a
  %   vector with diagonal elements of the hessian matrix (off
  %   diagonals are zero).
  %
  %   See also
  %   LIK_LOGIT_LL, LIK_LOGIT_LLG, LIK_LOGIT_LLG3, GPLA_E

  % softmax:    
    expf2 = exp(f2);
    pi2 = expf2./(sum(expf2, 2)*ones(1,size(y,2)));
    pi_vec=pi2(:);
    [n,nout]=size(y);
    pi_mat=zeros(nout*n, n);
    for i1=1:nout
      pi_mat((1+(i1-1)*n):(nout*n+1):end)=pi2(:,i1);
    end
    D=diag(pi_vec);
    g2=-D+pi_mat*pi_mat';
    
  end    
  
  function third_grad = lik_softmax_llg3(lik, y, f, param, z)
  %LIK_LOGIT_LLG3  Third gradients of log likelihood (energy)
  %
  %   Description
  %   G3 = LIK_LOGIT_LLG3(LIK, Y, F, PARAM) takes a likelihood 
  %   data structure LIK, class labels Y, and latent values F
  %   and returns the third gradients of log likelihood with respect
  %   to PARAM. At the moment PARAM can be only 'latent'. G3 is a
  %   vector with third gradients.
  %
  %   See also
  %   LIK_LOGIT_LL, LIK_LOGIT_LLG, LIK_LOGIT_LLG2, GPLA_E, GPLA_G
    
    if ~isempty(find(y~=1 & y~=0))
      error('lik_softmax: The class labels have to be {0,1}')
    end
    
  end


  function [m_0, m_1, sigm2hati1] = lik_softmax_tiltedMoments(lik, y, i1, sigm2_i, myy_i, z)
  end
  
  function [Ey, Vary, py] = lik_softmax_predy(lik, Ef, Varf, y, z)
  end


  function reclik = lik_softmax_recappend(reclik, ri, lik)
  % RECAPPEND  Append the parameters to the record
  %
  %          Description 
  %          RECLIK = GPCF_LOGIT_RECAPPEND(RECLIK, RI, LIK)
  %          takes a likelihood record structure RECLIK, record
  %          index RI and likelihood structure LIK with the
  %          current MCMC samples of the hyperparameters. Returns
  %          RECLIK which contains all the old samples and the
  %          current samples from LIK.
  % 
  %  See also:
  %  gp_mc

    if nargin == 2
      reclik.type = 'softmax';

      % Set the function handles
      reclik.fh_pak = @lik_softmax_pak;
      reclik.fh_unpak = @lik_softmax_unpak;
      reclik.fh_ll = @lik_softmax_ll;
      reclik.fh_llg = @lik_softmax_llg;    
      reclik.fh_llg2 = @lik_softmax_llg2;
      reclik.fh_llg3 = @lik_softmax_llg3;
      reclik.fh_tiltedMoments = @lik_softmax_tiltedMoments;
      reclik.fh_predy = @lik_softmax_predy;
      reclik.fh_recappend = @lik_softmax_recappend;
      return
    end
    
  end
end
