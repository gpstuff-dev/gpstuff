function lik = lik_liks(varargin)
%  LIKS creates a likelihood which is composed by many
%  different likelihoods.
%
%  Description:
%    LIK = LIKS(LIKELIHOODS, VALUE1, ...) 
%    creates a likelihood structure with many different likelihoods. if
%    there is any unspecified parameters you get an error
%  
%    The likelihood is the product of the likelihoods (independent
%    observations given the latent process),
%
%               __ k    __ n_j
%      p(y|f) = || j=1  || i=1 L(y_ij|f_ij, th_j)
%
%      f = (f1, f2, ..., fk)
%      y = (y1, y2, ..., yk)
%
%  See also
%    GP_SET, LIK_*, PRIOR_*
%
% Copyright (c) 2011 Jaakko Riihimäki
% Copyright (c) 2011 Aki Vehtari
% Copyright (c) 2012 Ville Tolvanen
% ───────────── 2015 Marcelo Hartmann

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

  ip = inputParser;
  ip.FunctionName = 'LIKS';
  ip.addOptional('lik', [], @isstruct);
  ip.addParamValue('likelihoods', {}, @(x) ~isempty(x) && iscell(x));
  ip.addParamValue('classVariables', [], @(x) ~isempty(x) && isvector(x));
  ip.parse(varargin{:});
  lik = ip.Results.lik;
  
  if isempty(lik)
      init = true;
      lik.type = 'Liks';
  else
      if ~isfield(lik, 'type') || ~isequal(lik.type, 'Liks')
          error('First argument does not seem to be a valid likelihood function structure')
      end
      init = false;
  end
  
  % Initialize likelihoods
  if init || ~ismember('likelihoods', ip.UsingDefaults)
      lik.liks = ip.Results.likelihoods;
  end
  
  % number of likelihood functions
  lik.nliks = length(lik.liks);
  if lik.nliks < 2
      error('use one likelihood structure')
      
  else 
      % for observations
      lik.classVariables = ip.Results.classVariables;
      if issorted(lik.classVariables)       
          if (length(unique(lik.classVariables)) == lik.nliks && lik.classVariables(end) == lik.nliks);
          else
              error('something is wrong with the number of likelihoods and given class variables');
          end
      else
         error('you need to give the class variable increasing downwards');
      end
      
  end
  
  % Initialize prior structure
  % Even if there is no prior at all 
  % see line 2065 in the file gpla_e.m, and you may see the light !!!
  if init
      lik.p = [];
  end
  
  if init
      % Set the function handles to the subfunctions
      lik.fh.pak = @liks_pak;
      lik.fh.unpak = @liks_unpak;
      lik.fh.lp = @liks_lp;
      lik.fh.lpg = @liks_lpg;
      lik.fh.ll = @liks_ll;
      lik.fh.llg = @liks_llg;    
      lik.fh.llg2 = @liks_llg2;
      lik.fh.llg3 = @liks_llg3;
      lik.fh.tiltedMoments = @liks_tiltedMoments;
      lik.fh.siteDeriv = @liks_siteDeriv;
      lik.fh.predy = @liks_predy;
      lik.fh.recappend = @liks_recappend;
  end
    

end


function [w, s, h] = liks_pak(lik)
% LIKS_PAK  Combines each likelihood parameters into one vector.
%
%  Description:
%    W = LIKS_PAK(LIK) takes a likelihood structure LIK and
%    combines the parameters into a single row vector W. This is a 
%    mandatory subfunction used for example in energy and gradient 
%    computations.
%     
%   See also
%   LIKS_UNPAK, GP_PAK

 w = []; s = {}; h = [];

 for j = 1:lik.nliks
     lik_j = lik.liks{j};
     [wj, sj, hj] = lik_j.fh.pak(lik_j);
     w = [w wj];
     s = [s; sj];
     h = [h hj];  
     
 end
 
end


function [lik, w] = liks_unpak(lik, w)
% LIKS_UNPAK  Extracts each likelihood parameters from the vector.
%
%  Description: 
%    [LIK, W] = LIKS_UNPAK(W, LIK) takes each likelihood
%    structure inside LIK and extracts the parameters from the vector W
%    to the whole LIK structure. This is a mandatory subfunction used 
%    for example in energy and gradient computations.
%     
%   See also:
%   LIKS_PAK, GP_UNPAK

% Assignment is inverse parameter transformation 

 for j = 1:lik.nliks
     lik_j = lik.liks{j};
     [lik_j, w] = lik_j.fh.unpak(lik_j, w);    
     lik.liks{j} = lik_j;
     
 end
 
end


function lp = liks_lp(lik, varargin)
% LIKS_LP  log(prior) of each the likelihood parameters
%
%  Description:
%    LP = LIKS_LP(LIK) takes the a likelihood structure LIK and
%    returns log(p(th)), where th collects the parameters. This 
%    subfunction is needed when there are likelihood parameters.
%
%  See also:
%    LIKS_LLG, LIKS_LLG2, LIKS_LLG3, GPLA_E
  
 % If there is prior for the likelihood parameters
 lp = 0;

 for j = 1:lik.nliks
     if ~isempty(lik.liks{j}.fh.pak(lik.liks{j}))
         lp = lp + lik.liks{j}.fh.lp(lik.liks{j});
     end
 end
 
end


function lpg = liks_lpg(lik)
% LIKS_LPG  dlog(prior)/dth of each the likelihood parameters th
%
%  Description:
%    E = LIKS_LPG(LIK) takes a likelihood structure LIK and
%    returns d log(p(th))/dth for each likelihood function,
%    where th collects the parameters.
%    This subfunction is needed when there are likelihood parameters.
%
%  See also:
%    LIKS_LLG, LIKS_LG3, LIKS_LLG2, GPLA_G

 lpg = [];
 
 for j = 1:lik.nliks 
     if ~isempty(lik.liks{j}.fh.pak(lik.liks{j}))
         lpg = [lpg lik.liks{j}.fh.lpg(lik.liks{j})];
     end
 end
 
end


function ll = liks_ll(lik, y, ff, z)
% LIKS_LL log-likelihood
%
%  Description:
%    LL = LIKS_LL(LIK, Y, F) takes a likelihood structure LIK. 
%    Returns the log-likelihood, sum_{i=1}^{k} (log p_i(y|f)).
%    This subfunction is needed when using Laplace approximation 
%    or MCMC for inference with non-Gaussian likelihoods. This 
%    subfunction is also used in information criteria (DIC, WAIC) 
%    computations.
%
%  See also:
%    LIK_LLG, LIK_LLG3, LIK_LLG2, GPLA_E
 
 n = size(y, 1);
 if n ~= numel(lik.classVariables)
     error('lengths of y and class variables are different')
 end
 
 f = ff(:);
 
 u = diff([0 lik.classVariables(:)' lik.classVariables(end)+1]) .* (1:(n+1));
 u = u(u ~= 0);
 
 ll = 0;
 
 for j = 1:lik.nliks
     ind = (u(j):u(j+1)-1);
     yj = y(ind);
     fj = f(ind);
     zj = z(ind);
     ll = ll + lik.liks{j}.fh.ll(lik.liks{j}, yj, fj, zj);
     
 end
 
end


function llg = liks_llg(lik, y, ff, param, z)
% LIKS_LLG  Gradient of the log-likelihood
%
%  Description:
%    LLG = LIKS_LLG(LIK, Y, F, PARAM) takes a likelihood
%    structure LIK, Returns the gradient of the log likelihood
%    with respect to PARAM for each likelihood function. At the moment PARAM
%    can be 'param' or 'latent'. This subfunction is needed when using 
%    Laplace approximation or MCMC for inference with non-Gaussian
%    likelihoods.
%
%  See also:
%    LIKS_LL, LIKS_LLG2, LIKS_LLG3, GPLA_E
  
 n = size(y, 1);
 if n ~= numel(lik.classVariables)
     error('lengths of y and class variables are different')
 end
 
 f = ff(:);
 
 u = diff([0 lik.classVariables(:)' lik.classVariables(end)+1]) .* (1:(n + 1));
 u = u(u ~= 0);
 
 llg = [];
 
 for j = 1:lik.nliks
     ind = (u(j):u(j+1)-1);
     yj = y(ind); 
     fj = f(ind); 
     zj = z(ind);
     
     switch param
         case 'param'
             if ~isempty(lik.liks{j}.fh.pak(lik.liks{j}))
                 llg = [llg lik.liks{j}.fh.llg(lik.liks{j}, yj, fj, param, zj)];
                 
             end
             
         case 'latent'
             llg = [llg; lik.liks{j}.fh.llg(lik.liks{j}, yj, fj, param, zj)];
             
     end
     
 end
end


function llg2 = liks_llg2(lik, y, ff, param, z)
% LIKS_LLG2  Second gradients of the log-likelihood
%
%  Description        
%    LLG2 = LIKS_LLG2(LIK, Y, F, PARAM) takes a likelihood
%    structure LIK, Returns the hessian of the log likelihood
%    with respect to PARAM. At the moment PARAM can be only
%    'latent'. LLG2 is a vector with diagonal elements of the
%    Hessian matrix (off diagonals are zero). This subfunction 
%    is needed when using Laplace approximation or EP for 
%    inference with non-Gaussian likelihoods.
%
%  See also
%    LIKS_LL, LIKS_LLG, LIKS_LLG3, GPLA_E

 n = size(y, 1);
 if n ~= numel(lik.classVariables)
     error('lengths of y and class variables are different')
 end
 
 f = ff(:);
  
 u = diff([0 lik.classVariables(:)' lik.classVariables(end)+1]) .* (1 : (n + 1));
 u = u(u ~= 0);
 
 llg2 = [];
 
 nlikpar = length(lik.fh.pak(lik));
 z0 = zeros(n, nlikpar);
 aux(1) = 0;
  
 for j = 1:lik.nliks
     ind = (u(j):u(j+1)-1);
     yj = y(ind);
     fj = f(ind);
     
     if ~isempty(z)
         zj = z(ind);
     else
         zj = [];
     end

     switch param
         case 'param'
             
         case 'latent'
             llg2 = [llg2; lik.liks{j}.fh.llg2(lik.liks{j}, yj, fj, param, zj)];
             
         case 'latent+param'
             if ~isempty(lik.liks{j}.fh.pak(lik.liks{j}))
                 % take the column vectors
                 llg2_tmp = lik.liks{j}.fh.llg2(lik.liks{j}, yj, fj, param, zj);
                 
                 % auxiliar indexes
                 aux(end + 1) = aux(end) + size(lik.liks{j}.fh.pak(lik.liks{j}), 2);
                 
                 % auxiliar matrices for derivatives w.r.t parameters in
                 % the specific likelihood j
                 z0(ind, (aux(end - 1) + 1) : aux(end)) = llg2_tmp;
                 llg2 = [llg2 z0(:, aux(end - 1) + 1 : aux(end))];
                 z0(ind, (aux(end - 1) + 1) : aux(end)) = 0;
                 
             end 
     end
     
 end
end    


function llg3 = liks_llg3(lik, y, ff, param, z)
% LIKS_LLG3  Third gradients of the log likelihood
%
%  Description:
%    LLG3 = LIKS_LLG3(LIK, Y, F, PARAM) takes a likelihood
%    structure LIK, returns the third gradients of the log
%    likelihood with respect to PARAM. At the moment PARAM can be
%    only 'latent'. LLG3 is a vector with third gradients. This 
%    subfunction is needed when using Laplace approximation for 
%    inference with non-Gaussian likelihoods.
%
%  See also:
%    LIKS_LL, LIKS_LLG, LIKS_LLG2, GPLA_E, GPLA_G

 n = size(y, 1);
 if n ~= numel(lik.classVariables)
     error('lengths of y and class variables are different')
 end
 
 f = ff(:);
  
 u = diff([0 lik.classVariables(:)' lik.classVariables(end)+1]) .* (1:(n+1));
 u = u(u ~= 0);
 
 llg3 = [];

 % auxiliar matrix for derivatives of parameters w.r.t. many likelihoods
 nlikpar = length(lik.fh.pak(lik));
 z0 = zeros(n, nlikpar);
 aux(1) = 0;
      
 for j = 1:lik.nliks
     switch param
         case 'param'
             
         case 'latent'
             ind = (u(j):u(j+1)-1);
             yj = y(ind); 
             fj = f(ind);
             zj = z(ind);
             
             llg3 = [llg3; lik.liks{j}.fh.llg3(lik.liks{j}, yj, fj, param, zj)];
             
         case 'latent2+param'
             if ~isempty(lik.liks{j}.fh.pak(lik.liks{j}))
                 % take indexes and respective observations for specific likelihood
                 ind = (u(j):u(j+1)-1);
                 yj = y(ind); 
                 fj = f(ind);
                 zj = z(ind);
                 
                 % take the column vectors
                 llg3_tmp = lik.liks{j}.fh.llg3(lik.liks{j}, yj, fj, param, zj);
                 
                 % auxiliar indexes
                 aux(end + 1) = aux(end) + size(lik.liks{j}.fh.pak(lik.liks{j}), 2);
                 
                 % auxiliar matrices for derivatives w.r.t parameters in
                 % the specific likelihood j
                 z0(ind, (aux(end - 1) + 1) : aux(end)) = llg3_tmp;
                 llg3 = [llg3 z0(:, aux(end - 1) + 1 : aux(end))];
                 z0(ind, (aux(end - 1) + 1) : aux(end)) = 0;

             end
             
     end
     
 end
end

function [logM_0, m_1, sigm2hati1] = liks_tiltedMoments(lik, y, i1, sigm2_i, myy_i, z)
% LIKS_TILTEDMOMENTS  Returns the marginal moments for EP algorithm
%
%  Description
%    [M_0, M_1, M2] = LIKS_TILTEDMOMENTS(LIK, Y, I, S2,
%    MYY) takes a likelihood structure LIKS, the observation Y, index 
%    I, cavity variance S2 and mean MYY. Returns the zeroth
%    moment M_0, mean M_1 and variance M_2 of the posterior
%    marginal (see Rasmussen and Williams (2006): Gaussian
%    processes for Machine Learning, page 55). This subfunction 
%    is needed when using EP for inference with non-Gaussian 
%    likelihoods.
%
%  See also
%    GPEP_E

 n = size(y, 1);
 if n ~= numel(lik.classVariables)
     error('length of y and class variables are different')
 end
  
 u = diff([0 lik.classVariables(:)' lik.classVariables(end)+1]) .* (1:(n + 1));
 u = u(u ~= 0);
 sizeObs = diff(u);
 
 logM_0 = zeros(n, 1);
 m_1 = zeros(n, 1);
 sigm2hati1 = zeros(n, 1);

 for j = 1:lik.nliks
     ind = (u(j) : u(j + 1) - 1);
     yj = y(ind); 
     sigm2_ij = sigm2_i(ind);
     myy_ij = myy_i(ind);
     zj = z(ind);
     
     [logM_0(ind), m_1(ind), sigm2hati1(ind)] = ...
     lik.liks{j}.fh.tiltedMoments(lik.liks{j}, yj, 1:sizeObs(j), sigm2_ij, myy_ij, zj);
 end
end

function [g_i] = liks_siteDeriv(lik, y, i1, sigm2_i, myy_i, z)
% LIKS_SITEDERIV  Evaluate the expectation of the gradient
%                 of the log likelihood term with respect
%                 to the multiple-likelihood parameters for EP 
%
%  Description [M_0, M_1, M2] =
%    LIKS_SITEDERIV(LIK, Y, I, S2, MYY, Z) takes a
%    likelihood structure LIKS, incedence counts Y, expected
%    counts Z, index I and cavity variance S2 and mean MYY. 
%    Returns E_f [d log p(y_i|f_i) /d a], where a is the
%    likelihood parameter and the expectation is over the
%    marginal posterior. This is done for all likelihoods.
%    This term is needed when evaluating the
%    gradients of the marginal likelihood estimate Z_EP with
%    respect to the likelihood parameters (see Seeger (2008):
%    Expectation propagation for exponential families). This 
%    subfunction is needed when using EP for inference with 
%    non-Gaussian likelihoods and there are likelihood parameters.
%
%  See also
%    GPEP_G

 n = size(y, 1);
 if n ~= numel(lik.classVariables)
     error('lengths of y and class variables are different')
 end
  
 u = diff([0 lik.classVariables(:)' lik.classVariables(end)+1]) .* (1:(n + 1));
 u = u(u ~= 0);
 
 nlikpar = length(lik.fh.pak(lik));
 g_i = zeros(1, nlikpar);
 aux = 0;
 
 for j = 1:lik.nliks
     if ~isempty(lik.liks{j}.fh.pak(lik.liks{j}))
         % auxiliar indexes for paramters
         aux = aux + 1;
         
         % indexes for that specific likelihood
         ind = (u(j) : u(j + 1) - 1);

         if any(ind == i1)
             i1j = i1;
             
             % yj = y(ind); 
             % zj = z(ind);
             
             sigm2_ij = sigm2_i;
             myy_ij = myy_i;
             
             % if some specific likelihood has more than one parameter this will not work
             g_i(1, aux) = ...
             lik.liks{j}.fh.siteDeriv(lik.liks{j}, y, i1j, sigm2_ij, myy_ij, z);
         end
     end
         
 end
end

function [lpy, Ey, Vary] = liks_predy(lik, Ef, Varf, yt, zt)
% LIKS_PREDY  Returns the predictive mean, variance and density of y
%
%  Description:  
%    LPY = LIKS_PREDY(LIK, EF, VARF YT, ZT)
%    Returns logarithm of the predictive density PY of YT, that is 
%    p(yt | zt) = \int p(yt | f, zt) p(f|y) df.
%    This requires the observations YT.
%    This subfunction is needed when computing posterior predictive 
%    distributions for future observations.
%
%    [LPY, EY, VARY] = LIKS_PREDY(LIK, EF, VARF) takes a
%    likelihood structure LIK, posterior mean EF and posterior
%    Variance VARF of the latent variable and returns the
%    posterior predictive mean EY and variance VARY of the
%    observations related to the latent variables. This subfunction
%    is needed when computing posterior predictive distributions for 
%    future observations.
%        
%
%  See also:
%    GPLA_PRED, GPEP_PRED, GPMC_PRED1

% if numel(zt) == 0
%     zt = ones(size(Ef));
% end
% 
% n = size(Ef, 1);
% 
% % take class variables for prediction or loocv
% if ~isfield(lik, 'classVariablesPred')
%     classVariables = lik.classVariables;
% else
%     classVariables = lik.classVariablesPred;
% end

if nargout > 1
    flag = 1;
    
    if size(zt, 2) ~= 2
        error(['if you want to predict with many likelihoods you need' ...
            'to pass the class variables in the second column of zt']);
    else
        classVariables = zt(:, 2);
    end
    
    % actual zt ...
    zt = zt(:, 1);
    
else
    flag = 0;
    classVariables = lik.classVariables;
    
end

% number of values to predict;
n = size(Ef, 1);

% check some conditions
if ~issorted(classVariables) 
    error('you need to give the class variable increasing downwards');
end

if max(classVariables) > lik.nliks
    error('more classes than the number of classes you are trying to model');
end

% getting the information of the classes in the data
indClass = unique(classVariables)'; 
nind = size(indClass, 2);

% getting the positions of observations in each vector 
u = find(diff([-inf classVariables' inf]));

% log-density
lpy = zeros(n, 1);

if flag
    Ey = zeros(n, 1);
    Vary = zeros(n, 1);
    
    for j = 1:nind
        ind = (u(j) : u(j + 1) - 1);
        likj = lik.liks{indClass(j)};
        
        if numel(yt) ~= 0;
           [lpy(ind), Ey(ind), Vary(ind)] = ...
           likj.fh.predy(likj, Ef(ind), Varf(ind), yt(ind), zt(ind));
           
        else
           [~, Ey(ind), Vary(ind)] = likj.fh.predy(likj, Ef(ind), Varf(ind), yt, zt(ind));
            
        end
    end
    
else
    for j = 1:nind
        ind = (u(j) : u(j + 1) - 1);
        likj = lik.liks{indClass(j)};
        
        lpy(ind) = likj.fh.predy(likj, Ef(ind), Varf(ind), yt(ind), zt(ind));
    end
end

end


function reclik = liks_recappend(reclik, ri, lik)
% RECAPPEND Append the parameters to the record
%
%  Description:
%    RECLIK = LIKS_RECAPPEND(RECLIK, RI, LIK) takes a
%    likelihood record structure RECLIK, record index RI and
%    likelihood structure LIK with the current MCMC samples of
%    the parameters. Returns RECLIK which contains all the old
%    samples and the current samples from LIK. This subfunction
%    is needed when using MCMC sampling (gp_mc).
% 
%  See also:
%    GP_MC

 if nargin == 2
     % Initialize the record
     reclik.type = 'Liks';

     % Initialize parameters 
     nliks = length(ri.liks);
     for i = 1:nliks
         lik_i = ri.liks{i};
         reclik.liks{i} = lik_i.fh.recappend([], ri.liks{i});
     end
     
     % Set the function handles
     reclik.fh.pak = @liks_pak;
     reclik.fh.unpak = @liks_unpak;
     reclik.fh.lp = @liks_lp;  
     reclik.fh.lpg = @liks_lpg;
     reclik.fh.ll = @liks_ll;
     reclik.fh.llg = @liks_llg;    
     reclik.fh.llg2 = @liks_llg2;
     reclik.fh.llg3 = @liks_llg3;
     reclik.fh.tiltedMoments = @liks_tiltedMoments;
     reclik.fh.siteDeriv = @liks_siteDeriv;
     reclik.fh.predy = @liks_predy;
     reclik.fh.recappend = @liks_recappend;  
     
     if isfield(ri, 'classVariables') 
         reclik.classVariables = ri.classVariables;
         reclik.nliks = ri.nliks;
     end
      
 else
     % Append to the record
     % Loop over all of the likelihood functions
     nliks = length(lik.liks);
     for i = 1:nliks;
         lik_i = lik.liks{i};
         reclik.liks{i} = lik_i.fh.recappend(reclik.liks{i}, ri, lik_i);
     end
     
 end
end

