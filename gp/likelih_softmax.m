function likelih = likelih_softmax(varargin)
%LIKELIH_SOFTMAX    Create a softmax likelihood structure 
%
%     Description
%       LIKELIH = LIKELIH_SOFTMAX Create and initialize softmax
%       likelihood for multi-class classification problem. The
%       observed class label with C classes is given as 1xC vector
%       where C-1 entries are 0 and the observed class label is 1.
%
%       The fields in LIKELIH are:
%         type             = 'Logit'
%         fh_pak           = function handle to pak
%         fh_unpak         = function handle to unpak
%         fh_ll            = function handle to the log likelihood
%         fh_llg           = function handle to the gradient of 
%                            the log likelihood
%         fh_llg2          = function handle to the second gradient
%                            of the log likelihood
%         fh_llg3          = function handle to the third gradient  
%                            of the log likelihood
%         fh_tiltedMoments = function handle to evaluate posterior
%                            moments for EP
%         fh_predy         = function handle to evaluate predictive 
%                            density of y
%         fh_recappend     = function handle to append the record
%
%       See also
%       LIKELIH_LOGIT, LIKELIH_PROBIT, LIKELIH_NEGBIN

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
      likelih.type = 'Softmax';
      
      % Set the function handles to the nested functions
      likelih.fh_pak = @likelih_softmax_pak;
      likelih.fh_unpak = @likelih_softmax_unpak;
      likelih.fh_ll = @likelih_softmax_ll;
      likelih.fh_llg = @likelih_softmax_llg;    
      likelih.fh_llg2 = @likelih_softmax_llg2;
      likelih.fh_llg3 = @likelih_softmax_llg3;
      likelih.fh_tiltedMoments = @likelih_softmax_tiltedMoments;
      likelih.fh_predy = @likelih_softmax_predy;
      likelih.fh_recappend = @likelih_softmax_recappend;

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
      likelih = varargin{1};

  end


  function w = likelih_softmax_pak(likelih)
  %LIKELIH_LOGIT_PAK    Combine likelihood parameters into one vector.
  %
  %   Description 
  %   W = LIKELIH_LOGIT_PAK(LIKELIH) takes a likelihood data
  %   structure LIKELIH and returns an empty verctor W. If Logit
  %   likelihood had hyperparameters this would combine them into a
  %   single row vector W (see e.g. likelih_negbin).
  %     
  %
  %   See also
  %   LIKELIH_NEGBIN_UNPAK, GP_PAK
    
    w = [];
  end


  function [likelih, w] = likelih_softmax_unpak(w, likelih)
  %LIKELIH_LOGIT_UNPAK  Extract likelihood parameters from the vector.
  %
  %   Description
  %   W = LIKELIH_LOGIT_UNPAK(W, LIKELIH) Doesn't do anything.
  % 
  %   If Logit likelihood had hyperparameters this would extracts
  %   them parameters from the vector W to the LIKELIH structure.
  %     
  %
  %   See also
  %   LIKELIH_LOGIT_PAK, GP_UNPAK

    likelih=likelih;
    w=w;
  end


  function logLikelih = likelih_softmax_ll(likelih, y, f2, z)
  %LIKELIH_LOGIT_LL    Log likelihood
  %
  %   Description
  %   E = LIKELIH_LOGIT_LL(LIKELIH, Y, F) takes a likelihood
  %   data structure LIKELIH, class labels Y (NxC matrix), and latent values
  %   F (NxC matrix). Returns the log likelihood, log p(y|f,z).
  %
  %   See also
  %   LIKELIH_LOGIT_LLG, LIKELIH_LOGIT_LLG3, LIKELIH_LOGIT_LLG2, GPLA_E

    if ~isempty(find(y~=1 & y~=0))
      error('likelih_softmax: The class labels have to be {0,1}')
    end
    
    % softmax:
    logLikelih = y(:)'*f2(:) - sum(log(sum(exp(f2),2)));
    
  end


  function deriv = likelih_softmax_llg(likelih, y, f2, param, z)
  %LIKELIH_LOGIT_LLG    Gradient of log likelihood (energy)
  %
  %   Description
  %   G = LIKELIH_LOGIT_LLG(LIKELIH, Y, F, PARAM) takes a likelihood
  %   data structure LIKELIH, class labels Y, and latent values
  %   F. Returns the gradient of log likelihood with respect to
  %   PARAM. At the moment PARAM can be 'hyper' or 'latent'.
  %
  %   See also
  %   LIKELIH_LOGIT_LL, LIKELIH_LOGIT_LLG2, LIKELIH_LOGIT_LLG3, GPLA_E
    
    if ~isempty(find(y~=1 & y~=0))
      error('likelih_softmax: The class labels have to be {0,1}')
    end

    expf2 = exp(f2);
    pi2 = expf2./(sum(expf2, 2)*ones(1,size(y,2)));
    pi_vec=pi2(:);
    deriv = y(:)-pi_vec;
  end


  function g2 = likelih_softmax_llg2(likelih, y, f2, param, z)
  %LIKELIH_LOGIT_LLG2  Second gradients of log likelihood (energy)
  %
  %   Description        
  %   G2 = LIKELIH_LOGIT_LLG2(LIKELIH, Y, F, PARAM) takes a likelihood
  %   data structure LIKELIH, class labels Y, and latent values
  %   F. Returns the hessian of log likelihood with respect to
  %   PARAM. At the moment PARAM can be only 'latent'. G2 is a
  %   vector with diagonal elements of the hessian matrix (off
  %   diagonals are zero).
  %
  %   See also
  %   LIKELIH_LOGIT_LL, LIKELIH_LOGIT_LLG, LIKELIH_LOGIT_LLG3, GPLA_E

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
  
  function third_grad = likelih_softmax_llg3(likelih, y, f, param, z)
  %LIKELIH_LOGIT_LLG3  Third gradients of log likelihood (energy)
  %
  %   Description
  %   G3 = LIKELIH_LOGIT_LLG3(LIKELIH, Y, F, PARAM) takes a likelihood 
  %   data structure LIKELIH, class labels Y, and latent values F
  %   and returns the third gradients of log likelihood with respect
  %   to PARAM. At the moment PARAM can be only 'latent'. G3 is a
  %   vector with third gradients.
  %
  %   See also
  %   LIKELIH_LOGIT_LL, LIKELIH_LOGIT_LLG, LIKELIH_LOGIT_LLG2, GPLA_E, GPLA_G
    
    if ~isempty(find(y~=1 & y~=0))
      error('likelih_softmax: The class labels have to be {0,1}')
    end
    
  end


  function [m_0, m_1, sigm2hati1] = likelih_softmax_tiltedMoments(likelih, y, i1, sigm2_i, myy_i, z)
  end
  
  function [Ey, Vary, py] = likelih_softmax_predy(likelih, Ef, Varf, y, z)
  end


  function reclikelih = likelih_softmax_recappend(reclikelih, ri, likelih)
  % RECAPPEND  Append the parameters to the record
  %
  %          Description 
  %          RECLIKELIH = GPCF_LOGIT_RECAPPEND(RECLIKELIH, RI, LIKELIH)
  %          takes a likelihood record structure RECLIKELIH, record
  %          index RI and likelihood structure LIKELIH with the
  %          current MCMC samples of the hyperparameters. Returns
  %          RECLIKELIH which contains all the old samples and the
  %          current samples from LIKELIH.
  % 
  %  See also:
  %  gp_mc

    if nargin == 2
      reclikelih.type = 'softmax';

      % Set the function handles
      reclikelih.fh_pak = @likelih_softmax_pak;
      reclikelih.fh_unpak = @likelih_softmax_unpak;
      reclikelih.fh_ll = @likelih_softmax_ll;
      reclikelih.fh_llg = @likelih_softmax_llg;    
      reclikelih.fh_llg2 = @likelih_softmax_llg2;
      reclikelih.fh_llg3 = @likelih_softmax_llg3;
      reclikelih.fh_tiltedMoments = @likelih_softmax_tiltedMoments;
      reclikelih.fh_predy = @likelih_softmax_predy;
      reclikelih.fh_recappend = @likelih_softmax_recappend;
      return
    end
    
  end
end
