function p = prior_corrunif(varargin)
% * PRIOR_R  Correlation prior structure     
%       
% * Description:
% 
%  - REFERENCE: prior for correlation matrix
%    Barnard, J.; McCulloch, R. & li Meng, X. 
%    Modelling covariance matrices in terms of standart deviations and correlations
%    with applications to shrinkage. Statistical Sinica, 2000
%
%  - P = PRIOR_CORRUNIF('PARAM1', VALUE1, ...) 
%    creates the prior structure in which the
%    named parameters have the specified values. Any unspecified
%    parameters are set to default values.
%
%  - P = PRIOR_CORRUNIF(P, 'PARAM1', VALUE1, ...)
%    modify a prior structure with the named parameters altered
%    with the specified values.
%
%  - Parameters for correlation prior [default]
%     nu       - degree of freedom [15]
%     prior_nu - prior for nu [prior_fixed]
%
%  - some inverse-Wishart properties
%    if W ~ InvWish_d (v, A) then E[W] = A/(v-d-1)
%    nu > = d
%    d = number os species (dimension of the square matrix)
%    the construction for this prior assumes A = I.
%  
%  - dimension of the correlation vector: (numberClass^2 - numberClass)/2
%
% * See also
%    PRIOR_*
%
% Copyright (c) 2000-2001,2010 Aki Vehtari
% Copyright (c) 2009,2015 Jarno Vanhatalo
% Copyright (c) 2010 Jaakko Riihimäki
% ------------ 2015 Marcelo Hartmann 

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

  ip = inputParser;
  ip.FunctionName = 'PRIOR_CORRUNIF';
  ip.addOptional('p', [], @isstruct);
  ip.addParamValue('nu', 15, @(x) isscalar(x) && x > 0);
  ip.addParamValue('prior_nu', [], @(x) isstruct(x) || isempty(x));
  ip.addParamValue('numberClass', 2, @(x) mod(x, 1) == 0 && x > 1);
  ip.addParamValue('aValue', 1, @(x) isreal(x) &&  x > 0);
  ip.parse(varargin{:});
  p = ip.Results.p;
  
  if isempty(p)
    init = true;
    p.type = 'CORRUNIF';
    
  else
    if ~isfield(p, 'type') && ~isequal(p.type, 'CORRUNIF')
      error('First argument does not seem to be a valid prior structure')
    end
    
    init = false;
  end

  % transformation to the real line (aValue stretches or squeeze the real line).
  p.aValue = ip.Results.aValue;
  
  % Initialize parameters
  % check the condition nu > numberSpecies
  if init || ~ismember('nu', ip.UsingDefaults)
      p.nu = ip.Results.nu;
  end
  
  if init || ~ismember('numberClass', ip.UsingDefaults)
      p.numberClass = ip.Results.numberClass;
      p.vectSize = (p.numberClass^2 - p.numberClass)/2;
  end
  
  if p.nu < (p.numberClass - 1)
      error('Degrees of freedom (nu) should be greater than number of classes')
  end
  
  % Initialize prior structure
  if init
      p.p = [];
  end
  if init || ~ismember('nu_prior', ip.UsingDefaults)
      p.p.nu = ip.Results.prior_nu;
  end
  
  if init
    % set functions
    p.fh.pak = @prior_corrunif_pak;
    p.fh.unpak = @prior_corrunif_unpak;
  % p.fh.RealToRho = @prior_corrunif_RealToRho;
    p.fh.lp = @prior_corrunif_lp;
    p.fh.lpg = @prior_corrunif_lpg;
    p.fh.recappend = @prior_corrunif_recappend;
  end

end

function [w, s, h] = prior_corrunif_pak(p)
% This is a mandatory subfunction used for example 
% in energy and gradient computations.
  
  w = [];  s = {};  h = [];
  
  if ~isempty(p.p.nu)
      w = [w p.nu];
      s = [s; 'R.nu'];
      h = [h 1];
  end
  
end

function [p, w] = prior_corrunif_unpak(p, w)
% This is a mandatory subfunction used for example 
% in energy and gradient computations.

  if ~isempty(p.p.nu)
      i1 = 1;
      p.nu = w(i1);
      w = w(i1 + 1:end);
  end
  
end

function lp = prior_corrunif_lp(x, p)
% This is a mandatory subfunction used for example 
% in energy computations.

  % Evaluating log-prior(R)
  % correlation vector
  rho = x';  
  
  % create entries
  seq = 1:p.vectSize;
  i = ceil(0.5 + 0.5 * sqrt(1 + 8*(seq)));
  j = seq - (i - 2).*(i - 1)/2;
  ind1 = (j - 1) * p.numberClass + i;
  ind2 = (i - 1) * p.numberClass + j;
  
  % build correlation matrix
  R = eye(p.numberClass);  

  % filling elements in lower and upper part
  R([ind1, ind2]) = [rho; rho];
  detR = det(R);
  
  if eps < detR && ~any(abs(rho) > 1)    
      % cholesk decompostion
      L = chol(R, 'lower');
      
      % parameters of the distribution
      a = 0.5*(p.nu - 1)*(p.numberClass - 1) - 1;
      b = - p.nu/2;
      
      % building principal submatrices and evaluating log determinant
      sDetlogSub = 0;
      for k = 1:p.numberClass
          A = R;
          A(:, k) = []; A(k, :) = [];
          Lsub = chol(A, 'lower');
          sDetlogSub = sDetlogSub + sum(log(diag(Lsub)));
      end
      
      % evaluating unnormalized log-prior
      lp = 2 * (a*sum(log(diag(L))) + b*sDetlogSub);
      
  else
      lp = -Inf;
      
  end

% adding log-hyperprior(nu)
if ~isempty(p.p.nu)
    lp = lp + p.p.nu.fh.lp(p.nu, p.p.nu) + log(p.nu);
end

end

function lpg = prior_corrunif_lpg(x, p)
% This is a mandatory subfunction used for example 
% in gradient computations.
 
 % taking the correlation vector
 rho = x';
 
 % creating entries
 seq = 1:p.vectSize;
 i = ceil(0.5 + 0.5 * sqrt(1 + 8*(seq)));
 j = seq - (i - 2).*(i - 1)/2;
 ind1 = (j - 1) * p.numberClass + i;
 ind2 = (i - 1) * p.numberClass + j;
  
 % building corr matrix
 R = eye(p.numberClass); 
 
 % filling elements in lower and upper part
 R([ind1, ind2]) = [rho; rho];
 
 % all(eig(R) > 0)
  if ~any(abs(rho) > 1)
     
     % grad vector
     lpg = ones(1, p.vectSize);
     
     % cholesk decompostion
     % L = chol(R, 'lower');
          
     % parameters of the distribution
     a = 0.5*(p.nu - 1)*(p.numberClass - 1) - 1;
     b = - p.nu/2;
     
     % building principal submatrices and evaluating log determinant
     invR = inv(R);
     
     % COULD WE USE COFACTOR MATRIX ?
     % Id1 = eye(p.numberClass-1);
     aux = 0;
     
     for j = 2:p.numberClass
         for i = 1:(j-1)
             sumtrDer = 0;
             sumtrDer = sumtrDer + 2*a*invR(j, i);
             for k = 1:p.numberClass                 
                 if k == i || k == j
                     % sumtrDer = sumtrDer + 0;
                     continue
                 else
                     A = R;
                     A(:, k) = []; A(k, :) = [];
                     % L = chol(A, 'lower');
                     invAk = inv(A);                      
                     if k < j && (i == 1 || k > i)
                         m = j-1;
                         sumtrDer = sumtrDer + 2*b*invAk(m, i);
                     elseif k < j && k < i 
                         l = i-1; 
                         m = j-1;
                         sumtrDer = sumtrDer + 2*b*invAk(m, l);
                     else
                         sumtrDer = sumtrDer + 2*b*invAk(j, i);
                     end
                 end
             end
             aux = aux + 1;
             lpg(aux) = sumtrDer;
         end
     end
     
 else
     lpg = repmat(-Inf, 1, p.vectSize);
 end
 
 if ~isempty(p.p.nu)
     lpgnu = p.p.nu.fh.lpg(p.nu, p.p.nu).*p.nu + 1;
     lpg = [lpg lpgnu];
  end
end

function rec = prior_corrunif_recappend(rec, ri, p)
% This subfunction is needed when using MCMC sampling (gp_mc).
% The parameters are not sampled in any case.

rec = rec;
if ~isempty(p.p.nu)
    rec.nu(ri,:) = p.nu;
end

end
