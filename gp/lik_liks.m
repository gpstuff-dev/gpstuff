function lik = lik_liks(varargin)
%  LIKS creates a likelihood structure which is composed by many
%       different likelihoods.
%
%  Description:
%    LIK_LIK = LIK_LIKS(LIKELIHOODS, VALUE1, ...) creates a likelihood
%    structure that allows alternative likelihoods for latent variables.
%    The full likelihood is the product of independent likelihoods
%    (independent observations given the latent process), 
%
%               __k     __ 
%      p(y|f) = ||      ||   L_j(y_i|f_i, th_j)
%                 j=1   i \in I_j
%
%      f = (f1' f2' ... fk')'
%      y = (y1' y2' ... yk')'
%
%    Here j is the index for different likelihoods and I_j is the index
%    vector telling for which observations likelihood j is used.
%
%    When using the Liks likelihood you must give the vector/matrix z as 
%    an extra parameter to each function that requires also y. The matrix z
%    must include the covariates for independent likelihoods (see e.g.
%    LIK_POISSON) and a column telling the index of likelihood attached to
%    respective observation. For example: 
%      lik = lik_liks('likelihoods', {lik1 lik2}, 'classVariables', 2);
%      gp = gp_set('lik', lik, 'cf', k, 'latent_method', 'EP');
%      z = [1 1 1 2 2 2]'; 
%      gp_optim(gp, x, y, 'z', z)
%    Here z is a vector telling that the first 3 observations are related
%    to likelihood 1 and the last 3 to likelihood 2.
%
%    Parameters for Liks likelihood are [default]
%      likelihoods      - array of likelyhood structures [ {} ]
%      classVariables   - a scalar telling which column of matrix z defines 
%                         the likelihood indices
%
%    For the demonstration of the use of lik_liks see DEMO_MULTIVARIATEGP.
%
%  See also
%    GP_SET, LIK_*, PRIOR_*
%
% Copyright (c) 2011 Jaakko Riihimäki
% Copyright (c) 2011 Aki Vehtari
% Copyright (c) 2012 Ville Tolvanen
% Copyright (c) 2015-2017 Jarno Vanhatalo
% ------------- 2015-2017 Marcelo Hartmann
% Copyright (c) 2017 Ville Tolvanen

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

  ip = inputParser;
  ip.FunctionName = 'LIK_LIKS';
  ip.addOptional('lik', [], @isstruct);
  ip.addParamValue('likelihoods', {}, @(x) ~isempty(x) && iscell(x));
  ip.addParamValue('classVariables', [], @(x) ~isempty(x) && isscalar(x));
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
      % column of z where the classVariables can be found
      lik.classVariables = ip.Results.classVariables;
  end
  
  % Initialize prior structure
  % Even if there is no prior at all 
  % see line 2065 in the file gpla_e.m, and you may see the light !!!
  if init
      lik.p = [];
  end
  
  if init
      % Set the function handles to the subfunctions
      lik.fh.pak = @lik_liks_pak;
      lik.fh.unpak = @lik_liks_unpak;
      lik.fh.lp = @lik_liks_lp;
      lik.fh.lpg = @lik_liks_lpg;
      lik.fh.ll = @lik_liks_ll;
      lik.fh.llg = @lik_liks_llg;    
      lik.fh.llg2 = @lik_liks_llg2;
      lik.fh.llg3 = @lik_liks_llg3;
      lik.fh.tiltedMoments = @lik_liks_tiltedMoments;
      lik.fh.siteDeriv = @lik_liks_siteDeriv;
      lik.fh.predy = @lik_liks_predy;
      lik.fh.invlink = @lik_liks_invlink;
      lik.fh.recappend = @lik_liks_recappend;
  end
    

end


function [w, s, h] = lik_liks_pak(lik)
% LIK_LIKS_PAK  Combines each likelihood parameters into one vector.
%
%  Description:
%    W = LIK_LIKS_PAK(LIK) takes a likelihood structure LIK and
%    combines the parameters into a single row vector W. This is a 
%    mandatory subfunction used for example in energy and gradient 
%    computations.
%     
%   See also
%   LIK_LIKS_UNPAK, GP_PAK

 w = []; s = {}; h = [];

 for j = 1:lik.nliks
     lik_j = lik.liks{j};
     [wj, sj, hj] = lik_j.fh.pak(lik_j);
     w = [w wj];
     s = [s; sj];
     h = [h hj];  
     
 end
 
end


function [lik, w] = lik_liks_unpak(lik, w)
% LIK_LIKS_UNPAK  Extracts each likelihood parameters from the vector.
%
%  Description: 
%    [LIK, W] = LIK_LIKS_UNPAK(W, LIK) takes each likelihood
%    structure inside LIK and extracts the parameters from the vector W
%    to the whole LIK structure. This is a mandatory subfunction used 
%    for example in energy and gradient computations.
%     
%   See also:
%   LIK_LIKS_PAK, GP_UNPAK

% Assignment is inverse parameter transformation 

 for j = 1:lik.nliks
     lik_j = lik.liks{j};
     [lik_j, w] = lik_j.fh.unpak(lik_j, w);    
     lik.liks{j} = lik_j;
     
 end
 
end


function lp = lik_liks_lp(lik, varargin)
% LIK_LIKS_LP  log(prior) of each the likelihood parameters
%
%  Description:
%    LP = LIK_LIKS_LP(LIK) takes the a likelihood structure LIK and
%    returns log(p(th)), where th collects the parameters. This 
%    subfunction is needed when there are likelihood parameters.
%
%  See also:
%    LIK_LIKS_LLG, LIK_LIKS_LLG2, LIK_LIKS_LLG3, GPLA_E
  
 % If there is prior for the likelihood parameters
 lp = 0;

 for j = 1:lik.nliks
     if ~isempty(lik.liks{j}.fh.pak(lik.liks{j}))
         lp = lp + lik.liks{j}.fh.lp(lik.liks{j});
     end
 end
 
end


function lpg = lik_liks_lpg(lik)
% LIK_LIKS_LPG  dlog(prior)/dth of each the likelihood parameters th
%
%  Description:
%    E = LIK_LIKS_LPG(LIK) takes a likelihood structure LIK and
%    returns d log(p(th))/dth for each likelihood function,
%    where th collects the parameters.
%    This subfunction is needed when there are likelihood parameters.
%
%  See also:
%    LIK_LIKS_LLG, LIK_LIKS_LG3, LIK_LIKS_LLG2, GPLA_G

 lpg = [];
 
 for j = 1:lik.nliks 
     if ~isempty(lik.liks{j}.fh.pak(lik.liks{j}))
         lpg = [lpg lik.liks{j}.fh.lpg(lik.liks{j})];
     end
 end
 
end


function ll = lik_liks_ll(lik, y, ff, z)
% LIK_LIKS_LL log-likelihood
%
%  Description:
%    LL = LIK_LIKS_LL(LIK, Y, F) takes a likelihood structure LIK. 
%    Returns the log-likelihood, sum_{i=1}^{k} (log p_i(y|f)).
%    This subfunction is needed when using Laplace approximation 
%    or MCMC for inference with non-Gaussian likelihoods. This 
%    subfunction is also used in information criteria (DIC, WAIC) 
%    computations.
%
%  See also:
%    LIK_LLG, LIK_LLG3, LIK_LLG2, GPLA_E
 
 n  = size(y, 1);
 indClass = 1:size(z,2)==lik.classVariables; 
 zi = z(:, indClass);
 z  = z(:, ~indClass);
 
 indj  = unique(zi); 
 nind = numel(indj);
 
 if n ~= numel(zi)
     error('row-length of y and z are different')
 end
 
 f = ff(:);
 ll = 0; 
 for j = 1:nind
     ind = zi==indj(j);
     likj = lik.liks{indj(j)};
     
     yj = y(ind);
     fj = f(ind);
     if isempty(z)
         zj = z;
     else
         zj = z(ind);
     end
     ll = ll + likj.fh.ll(likj, yj, fj, zj);
     
 end
 
end


function llg = lik_liks_llg(lik, y, ff, param, z)
% LIK_LIKS_LLG  Gradient of the log-likelihood
%
%  Description:
%    LLG = LIK_LIKS_LLG(LIK, Y, F, PARAM) takes a likelihood
%    structure LIK, Returns the gradient of the log likelihood
%    with respect to PARAM for each likelihood function. At the moment PARAM
%    can be 'param' or 'latent'. This subfunction is needed when using 
%    Laplace approximation or MCMC for inference with non-Gaussian
%    likelihoods.
%
%  See also:
%    LIK_LIKS_LL, LIK_LIKS_LLG2, LIK_LIKS_LLG3, GPLA_E
  
 n  = size(y, 1); 
 indClass = 1:size(z,2)==lik.classVariables; 
 zi = z(:, indClass);
 z  = z(:, ~indClass);
 
 
 indj  = unique(zi); 
 nind = numel(indj);
 
 if n ~= numel(zi)
     error('row-length of y and z are different')
 end
 
 f = ff(:);
 
 switch param
     case 'param'
         llg = [];
     case 'latent'
         llg=zeros(size(f));
 end
 for j = 1:nind
     ind = zi==indj(j);
     likj = lik.liks{indj(j)};
     
     yj = y(ind); 
     fj = f(ind); 
     if isempty(z)
         zj = z;
     else
         zj = z(ind);
     end
     
     switch param
         case 'param'
             if ~isempty(lik.liks{j}.fh.pak(likj))
                 llg = [llg likj.fh.llg(likj, yj, fj, param, zj)];
             end
         case 'latent'
             llg(ind) = likj.fh.llg(likj, yj, fj, param, zj);
     end
 end

end


function llg2 = lik_liks_llg2(lik, y, ff, param, z)
% LIK_LIKS_LLG2  Second gradients of the log-likelihood
%
%  Description        
%    LLG2 = LIK_LIKS_LLG2(LIK, Y, F, PARAM) takes a likelihood
%    structure LIK, Returns the hessian of the log likelihood
%    with respect to PARAM. At the moment PARAM can be only
%    'latent'. LLG2 is a vector with diagonal elements of the
%    Hessian matrix (off diagonals are zero). This subfunction 
%    is needed when using Laplace approximation or EP for 
%    inference with non-Gaussian likelihoods.
%
%  See also
%    LIK_LIKS_LL, LIK_LIKS_LLG, LIK_LIKS_LLG3, GPLA_E

 n  = size(y, 1); 
 indClass = 1:size(z,2)==lik.classVariables; 
 zi = z(:, indClass);
 z  = z(:, ~indClass);
 
 
 indj = unique(zi); 
 nind = numel(indj);
 
 if n ~= numel(zi)
     error('row-length of y and z are different')
 end
 
 f = ff(:);

 nlikpar = length(lik.fh.pak(lik));
 z0 = zeros(n, nlikpar);
 aux(1) = 0;

 switch param
     case 'latent'
         llg2=zeros(size(f));
     case 'latent+param'
         llg2 = [];
 end
 for j = 1:nind
     ind = zi==indj(j);
     likj = lik.liks{indj(j)};
     
     yj = y(ind);
     fj = f(ind);
     if isempty(z)
         zj = z;
     else
         zj = z(ind);
     end
     
     switch param
         case 'param'
             
         case 'latent'
             llg2(ind) = likj.fh.llg2(likj, yj, fj, param, zj);
             
         case 'latent+param'
             if ~isempty(likj.fh.pak(likj))
                 % take the column vectors
                 llg2_tmp = likj.fh.llg2(likj, yj, fj, param, zj);
                 
                 % auxiliar indexes
                 aux(end + 1) = aux(end) + size(likj.fh.pak(likj), 2);
                 
                 % auxiliar matrices for derivatives w.r.t. parameters in
                 % the specific likelihood j
                 z0(ind, (aux(end - 1) + 1) : aux(end)) = llg2_tmp;
                 llg2 = [llg2 z0(:, aux(end - 1) + 1 : aux(end))];
                 z0(ind, (aux(end - 1) + 1) : aux(end)) = 0;
                 
             end 
     end
     
 end
 
end    


function llg3 = lik_liks_llg3(lik, y, ff, param, z)
% LIK_LIKS_LLG3  Third gradients of the log likelihood
%
%  Description:
%    LLG3 = LIK_LIKS_LLG3(LIK, Y, F, PARAM) takes a likelihood
%    structure LIK, returns the third gradients of the log
%    likelihood with respect to PARAM. At the moment PARAM can be
%    only 'latent'. LLG3 is a vector with third gradients. This 
%    subfunction is needed when using Laplace approximation for 
%    inference with non-Gaussian likelihoods.
%
%  See also:
%    LIK_LIKS_LL, LIK_LIKS_LLG, LIK_LIKS_LLG2, GPLA_E, GPLA_G

 n  = size(y, 1); 
 indClass = 1:size(z,2)==lik.classVariables; 
 zi = z(:, indClass);
 z  = z(:, ~indClass);
 
 
 indj  = unique(zi); 
 nind = numel(indj);
 
 if n ~= numel(zi)
     error('row-length of y and z are different')
 end
 
 f = ff(:);
  
 
 switch param
     case 'latent'
         llg3=zeros(size(f));
     case 'latent2+param'
         llg3 = [];
 end
 % auxiliar matrix for derivatives of parameters w.r.t. many likelihoods
 nlikpar = length(lik.fh.pak(lik));
 z0 = zeros(n, nlikpar);
 aux(1) = 0;
      
 for j = 1:nind
     switch param
         case 'param'
             
         case 'latent'
             ind = zi==indj(j);
             likj = lik.liks{indj(j)};
             
             yj = y(ind); 
             fj = f(ind);
             if isempty(z)
                 zj = z;
             else
                 zj = z(ind);
             end
             
             llg3(ind) = likj.fh.llg3(likj, yj, fj, param, zj);
             
         case 'latent2+param'
             if ~isempty(lik.liks{indj(j)}.fh.pak(lik.liks{indj(j)}))
                 % take indexes and respective observations for specific likelihood
                 ind = zi==indj(j);
                 likj = lik.liks{indj(j)};
                 
                 yj = y(ind); 
                 fj = f(ind);
                 if isempty(z)
                     zj = z;
                 else
                     zj = z(ind);
                 end
                 
                 % take the column vectors
                 llg3_tmp = likj.fh.llg3(likj, yj, fj, param, zj);
                 
                 % auxiliar indexes
                 aux(end + 1) = aux(end) + size(likj.fh.pak(likj), 2);
                 
                 % auxiliar matrices for derivatives w.r.t parameters in
                 % the specific likelihood j
                 z0(ind, (aux(end - 1) + 1) : aux(end)) = llg3_tmp;
                 llg3 = [llg3 z0(:, aux(end - 1) + 1 : aux(end))];
                 z0(ind, (aux(end - 1) + 1) : aux(end)) = 0;

             end
             
     end
     
 end
 
end


function [logM_0, m_1, sigm2hati1] = lik_liks_tiltedMoments(lik, y, i1, sigm2_i, myy_i, z)
% LIK_LIKS_TILTEDMOMENTS  Returns the marginal moments for EP algorithm
%
%  Description
%    [M_0, M_1, M2] = LIK_LIKS_TILTEDMOMENTS(LIK, Y, I, S2,
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

 n  = size(y, 1); 
 indClass = 1:size(z,2)==lik.classVariables; 
 zi = z(:, indClass);
 z  = z(:, ~indClass);
 
 
 indj = unique(zi); 
 nind = numel(indj);
 
 if n ~= numel(zi)
     error('row-length of y and z are different')
 end
  
 logM_0 = zeros(n, 1);
 m_1 = zeros(n, 1);
 sigm2hati1 = zeros(n, 1);
 
 for j = 1:nind
     likj = lik.liks{indj(j)};
     ind = zi==indj(j);
     if numel(sigm2_i)>1
         yj = y(ind);
         if isempty(z)
             zj = z;
         else
             zj = z(ind);
         end
         sigm2_ij = sigm2_i(ind);
         myy_ij = myy_i(ind);
         [logM_0(ind), m_1(ind), sigm2hati1(ind)] = ...
             likj.fh.tiltedMoments(likj, yj, 1:length(yj), sigm2_ij, myy_ij, zj);
     else
         if any(find(ind)==i1)
             [logM_0, m_1, sigm2hati1] = ...
                 likj.fh.tiltedMoments(likj, y, i1, sigm2_i, myy_i, z);
         end
     end

 end
 
end


function [g_i] = lik_liks_siteDeriv(lik, y, i1, sigm2_i, myy_i, z)
% LIK_LIKS_SITEDERIV  Evaluate the expectation of the gradient
%                 of the log likelihood term with respect
%                 to the multiple-likelihood parameters for EP 
%
%  Description [M_0, M_1, M2] =
%    LIK_LIKS_SITEDERIV(LIK, Y, I, S2, MYY, Z) takes a
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

 n  = size(y, 1); 
 indClass = 1:size(z,2)==lik.classVariables; 
 zi = z(:, indClass);
 z  = z(:, ~indClass);
  
 indj = unique(zi); 
 nind = numel(indj);
 
 if n ~= numel(zi)
     error('row-length of y and z are different')
 end
 
 nlikpar = length(lik.fh.pak(lik));
 g_i = zeros(1, nlikpar);
 aux = 0;
 
 for j = 1:nind
     if ~isempty(lik.liks{indj(j)}.fh.pak(lik.liks{indj(j)}))
         % auxiliar indexes for paramters
         aux = aux + 1;
         
         % indexes for that specific likelihood
         ind = find(zi==indj(j));
         likj = lik.liks{indj(j)};

         if any(ind == i1)
             % !!!! if some specific likelihood has more than one parameter this will not work
             g_i(1, aux) = likj.fh.siteDeriv(likj, y, i1, sigm2_i, myy_i, z);
         end
     end
         
 end
 
end


function [lpy, Ey, Vary] = lik_liks_predy(lik, Ef, Varf, yt, zt)
% LIK_LIKS_PREDY  Returns the predictive mean, variance and density of y
%
%  Description:  
%    LPY = LIK_LIKS_PREDY(LIK, EF, VARF YT, ZT)
%    Returns logarithm of the predictive density PY of YT, that is 
%    p(yt | zt) = \int p(yt | f, zt) p(f|y) df.
%    This requires the observations YT.
%    This subfunction is needed when computing posterior predictive 
%    distributions for future observations.
%
%    [LPY, EY, VARY] = LIK_LIKS_PREDY(LIK, EF, VARF) takes a
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

% number of values to predict;
n = size(Ef, 1);
indClass = 1:size(zt,2)==lik.classVariables;
zi = zt(:, indClass);
zt  = zt(:, ~indClass);
 

% check some conditions
if ~issorted(zi) 
    error('you need to give the class variable increasing downwards');
end

if max(zi) > lik.nliks
    error('more classes than the number of classes you are trying to model');
end

% getting the information of the classes in the data
indj = unique(zi); 
nind = numel(indj);


% log-density
lpy = zeros(n, 1);
if nargout > 1
    Ey = zeros(n, 1);
    Vary = zeros(n, 1);
    
    for j = 1:nind
        ind = zi==indj(j);
        likj = lik.liks{indj(j)};
        
        if numel(yt) ~= 0
           [lpy(ind), Ey(ind), Vary(ind)] = ...
           likj.fh.predy(likj, Ef(ind), Varf(ind), yt(ind), zt(ind));
        else
           [~, Ey(ind), Vary(ind)] = likj.fh.predy(likj, Ef(ind), Varf(ind), yt, zt(ind));
        end
    end
else
    for j = 1:nind
        ind = zi==indj(j);
        likj = lik.liks{indj(j)};
        
        if isempty(zt)
            lpy(ind) = likj.fh.predy(likj, Ef(ind), Varf(ind), yt(ind), []);
        else
            lpy(ind) = likj.fh.predy(likj, Ef(ind), Varf(ind), yt(ind), zt(ind));
        end
    end
end

end

function mu = lik_liks_invlink(lik, f, z)
%LIK_LIKS_INVLINK  Returns values of inverse link function
%             
%  Description 
%    P = LIK_LIKS_INVLINK(LIK, F) takes a likelihood structure LIK and
%    latent values F and returns the values MU of inverse link function.
%    This subfunction is needed when using gp_predprctmu. 
%
%     See also
%     LIK_POISSON_LL, LIK_POISSON_PREDY
  
 n  = size(f, 1); 
 indClass = 1:size(z,2)==lik.classVariables; 
 zi = z(:, indClass);
 z  = z(:, ~indClass);
 
 indj = unique(zi); 
 nind = numel(indj);
 
 if n ~= size(zi,1)
     error('row-length of f and z are different')
 end
 
 for j = 1:nind
     likj = lik.liks{indj(j)};
     ind = zi==indj(j);
     fj = f(ind,:);
     if isempty(z)
         zj = z;
     else
         zj = z(ind);
     end
     
     mu(ind,:) = likj.fh.invlink(likj, fj, zj);
 end


end


function reclik = lik_liks_recappend(reclik, ri, lik)
% RECAPPEND Append the parameters to the record
%
%  Description:
%    RECLIK = LIK_LIKS_RECAPPEND(RECLIK, RI, LIK) takes a
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
     reclik.fh.pak = @lik_liks_pak;
     reclik.fh.unpak = @lik_liks_unpak;
     reclik.fh.lp = @lik_liks_lp;  
     reclik.fh.lpg = @lik_liks_lpg;
     reclik.fh.ll = @lik_liks_ll;
     reclik.fh.llg = @lik_liks_llg;    
     reclik.fh.llg2 = @lik_liks_llg2;
     reclik.fh.llg3 = @lik_liks_llg3;
     reclik.fh.tiltedMoments = @lik_liks_tiltedMoments;
     reclik.fh.siteDeriv = @lik_liks_siteDeriv;
     reclik.fh.predy = @lik_liks_predy;
     reclik.fh.recappend = @lik_liks_recappend;  
     
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