function lik = lik_probit(varargin)
%LIK_PROBIT   Create a Probit likelihood structure 
%
%  Description
%    LIK = LIK_PROBIT creates Probit likelihood for classification
%    problem with class labels {-1,1}.
%  
%    The likelihood is defined as follows:
%                  __ n
%      p(y|f, z) = || i=1 normcdf(y_i * f_i)
%    
%      where f is the latent value vector.
%
%       See also
%       GP_SET, LIK_*
%

% Copyright (c) 2007      Jaakko Riihimäki
% Copyright (c) 2007-2010 Jarno Vanhatalo
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
      lik.type = 'Probit';
      
      % Set the function handles to the nested functions
      lik.fh.pak = @lik_probit_pak;
      lik.fh.unpak = @lik_probit_unpak;
      lik.fh.ll = @lik_probit_ll;
      lik.fh.llg = @lik_probit_llg;    
      lik.fh.llg2 = @lik_probit_llg2;
      lik.fh.llg3 = @lik_probit_llg3;
      lik.fh.tiltedMoments = @lik_probit_tiltedMoments;
      lik.fh.predy = @lik_probit_predy;
      lik.fh.recappend = @lik_probit_recappend;

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

  function [w,s] = lik_probit_pak(lik)
  %LIK_PROBIT_PAK    Combine likelihood parameters into one vector.
  %
  %     Description 
  %   W = LIK_PROBIT_PAK(LIK) takes a likelihood data
  %   structure LIK and returns an empty verctor W. If Probit
  %   likelihood had hyperparameters this would combine them into a
  %   single row vector W (see e.g. lik_negbin).
  %       
  %
  %     See also
  %     LIK_NEGBIN_UNPAK, GP_PAK

    w = []; s = {};
  end


  function [lik, w] = lik_probit_unpak(w, lik)
  %LIK_PROBIT_UNPAK  Extract likelihood parameters from the vector.
  %
  %     Description
  %   W = LIK_PROBIT_UNPAK(W, LIK) Doesn't do anything.
  % 
  %   If Probit likelihood had hyperparameters this would extracts
  %   them parameters from the vector W to the LIK structure.
  %       
  %
  %     See also
  %     LIK_PROBIT_PAK, GP_UNPAK

    
    lik=lik;
    w=[];
  end

  function logLik = lik_probit_ll(lik, y, f, z)
  %LIK_PROBIT_LL    Log likelihood
  %
  %   Description
  %   E = LIK_PROBIT_LL(LIK, Y, F) takes a likelihood
  %   data structure LIK, class labels Y, and latent values
  %   F. Returns the log likelihood, log p(y|f,z).
  %
  %   See also
  %   LIK_PROBIT_LLG, LIK_PROBIT_LLG3, LIK_PROBIT_LLG2, GPLA_E

    if ~isempty(find(abs(y)~=1))
      error('lik_probit: The class labels have to be {-1,1}')
    end

    logLik = sum(log(normcdf(y.*f)));
  end


  function deriv = lik_probit_llg(lik, y, f, param, z)
  %LIK_PROBIT_LLG    Gradient of log likelihood (energy)
  %
  %   Description
  %   G = LIK_PROBIT_LLG(LIK, Y, F, PARAM) takes a likelihood
  %   data structure LIK, class labels Y, and latent values
  %   F. Returns the gradient of log likelihood with respect to
  %   PARAM. At the moment PARAM can be 'hyper' or 'latent'.
  %
  %   See also
  %   LIK_PROBIT_LL, LIK_PROBIT_LLG2, LIK_PROBIT_LLG3, GPLA_E

    if ~isempty(find(abs(y)~=1))
      error('lik_probit: The class labels have to be {-1,1}')
    end
    
    switch param
      case 'latent'
        deriv = y.*normpdf(f)./normcdf(y.*f);
    end
  end


  function g2 = lik_probit_llg2(lik, y, f, param, z)
  %LIK_PROBIT_LLG2  Second gradients of log likelihood (energy)
  %
  %   Description        
  %   G2 = LIK_PROBIT_LLG2(LIK, Y, F, PARAM) takes a likelihood
  %   data structure LIK, class labels Y, and latent values
  %   F. Returns the hessian of log likelihood with respect to
  %   PARAM. At the moment PARAM can be only 'latent'. G2 is a
  %   vector with diagonal elements of the hessian matrix (off
  %   diagonals are zero).
  %
  %   See also
  %   LIK_PROBIT_LL, LIK_PROBIT_LLG, LIK_PROBIT_LLG3, GPLA_E

    
    if ~isempty(find(abs(y)~=1))
      error('lik_probit: The class labels have to be {-1,1}')
    end
    
    switch param
      case 'latent'
        z = y.*f;
        g2 = -(normpdf(f)./normcdf(z)).^2 - z.*normpdf(f)./normcdf(z);
    end
  end
  
  function thir_grad = lik_probit_llg3(lik, y, f, param, z)
  %LIK_PROBIT_LLG3  Third gradients of log likelihood (energy)
  %
  %   Description
  %   G3 = LIK_PROBIT_LLG3(LIK, Y, F, PARAM) takes a likelihood 
  %   data structure LIK, class labels Y, and latent values F
  %   and returns the third gradients of log likelihood with respect
  %   to PARAM. At the moment PARAM can be only 'latent'. G3 is a
  %   vector with third gradients.
  %
  %   See also
  %   LIK_PROBIT_LL, LIK_PROBIT_LLG, LIK_PROBIT_LLG2, GPLA_E, GPLA_G

    if ~isempty(find(abs(y)~=1))
      error('lik_probit: The class labels have to be {-1,1}')
    end
    
    switch param
      case 'latent'
        z2 = normpdf(f)./normcdf(y.*f);
        thir_grad = 2.*y.*z2.^3 + 3.*f.*z2.^2 - z2.*(y-y.*f.^2);
    end
  end
  

  function [m_0, m_1, m_2] = lik_probit_tiltedMoments(lik, y, i1, sigm2_i, myy_i, z)
  %LIK_PROBIT_TILTEDMOMENTS    Returns the marginal moments for EP algorithm
  %
  %   Description
  %   [M_0, M_1, M2] = LIK_PROBIT_TILTEDMOMENTS(LIK, Y, I, S2, MYY) 
  %   takes a likelihood data structure LIK, class labels Y,
  %   index I and cavity variance S2 and mean MYY. Returns the
  %   zeroth moment M_0, mean M_1 and variance M_2 of the posterior
  %   marginal (see Rasmussen and Williams (2006): Gaussian
  %   processes for Machine Learning, page 55).
  %
  %   See also
  %   GPEP_E
    
    if ~isempty(find(abs(y)~=1))
      error('lik_probit: The class labels have to be {-1,1}')
    end
    
    m_0 = normcdf(y(i1).*myy_i./sqrt(1+sigm2_i));
    zi=y(i1)*myy_i/sqrt(1+sigm2_i);
    normp_zi = normpdf(zi);
    normc_zi = normcdf(zi);
    muhati1=myy_i+(y(i1)*sigm2_i*normp_zi)/(normc_zi*sqrt(1+sigm2_i));
    sigm2hati1=sigm2_i-(sigm2_i^2*normp_zi)/((1+sigm2_i)*normc_zi)*(zi+normp_zi/normc_zi);
    m_1 = muhati1;
    m_2 = sigm2hati1;
  end

  function [Ey, Vary, py] = lik_probit_predy(lik, Ef, Varf, yt, zt)
  %LIK_PROBIT_PREDY    Returns the predictive mean, variance and density of y
  %
  %   Description         
  %   [EY, VARY] = LIK_PROBIT_PREDY(LIK, EF, VARF)
  %   takes a likelihood data structure LIK, posterior mean EF
  %   and posterior Variance VARF of the latent variable and returns
  %   the posterior predictive mean EY and variance VARY of the
  %   observations related to the latent variables
  %        
  %   [Ey, Vary, PY] = LIK_PROBIT_PREDY(LIK, EF, VARF, YT)
  %   Returns also the predictive density of YT, that is 
  %        p(yt | y) = \int p(yt | f) p(f|y) df.
  %   This requires also the class labels YT.
  %
  %   See also 
  %   ep_pred, la_pred, mc_pred

    
    if ~isempty(find(abs(yt)~=1))
      error('lik_probit: The class labels have to be {-1,1}')
    end

    py1 = normcdf(Ef./sqrt(1+Varf));
    Ey = 2*py1 - 1;

    Vary = 1-Ey.^2;
    
    if nargout > 2
      py = normcdf(Ef.*yt./sqrt(1+Varf));    % Probability p(y_new)
    end
  end

  function reclik = lik_probit_recappend(reclik, ri, lik)
  % RECAPPEND  Append the parameters to the record
  %
  %          Description 
  %          RECLIK = GPCF_PROBIT_RECAPPEND(RECLIK, RI, LIK)
  %          takes a likelihood record structure RECLIK, record
  %          index RI and likelihood structure LIK with the
  %          current MCMC samples of the hyperparameters. Returns
  %          RECLIK which contains all the old samples and the
  %          current samples from LIK.
  % 
  %  See also:
  %  gp_mc

    if nargin == 2
      reclik.type = 'Probit';

      % Set the function handles
      reclik.fh.pak = @lik_probit_pak;
      reclik.fh.unpak = @lik_probit_unpak;
      reclik.fh.ll = @lik_probit_ll;
      reclik.fh.llg = @lik_probit_llg;    
      reclik.fh.llg2 = @lik_probit_llg2;
      reclik.fh.llg3 = @lik_probit_llg3;
      reclik.fh.tiltedMoments = @lik_probit_tiltedMoments;
      reclik.fh.predy = @lik_probit_predy;
      reclik.fh.recappend = @lik_probit_recappend;
      return
    end

  end

end


