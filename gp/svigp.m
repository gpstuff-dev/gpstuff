function [gp, diagnosis] = svigp(gp, x, y, varargin)
%SVIGP  Stochastic variational inference for GP
%
%  Description
%    GP = SVIGP(GP, X, Y, OPTIONS) optimises the variational, likelihood
%    and covariance function parameters of a sparse SVI GP model
%    given matrix X of training inputs and vector Y of training targets.
%    The model follows the description in Hensman, Fusi and Lawrence
%    (2013). Supported likelihood functions are gaussian (for regression)
%    and probit (for classification).
%
%    [GP, DIAGNOSIS] = SVIGP(GP, X, Y, OPTIONS) also returns values
%    monitored during the iteration. The structure DIAGNOSIS contains
%    the energy in the field e, the likelihood and covariance function
%    parameters in the field w, and the mean log predictive density (if xt
%    and yt is provided) in the field mlpd. The first dimension corresponds
%    to the main iteration in all these arrays. The second dimension
%    corresponds to the minibatch iteration in e and w. The third dimension
%    corresponds to the different parameters in w.
%
%    OPTIONS is optional parameter-value pair
%      xt               - test inputs. Used for monitoring the mean log
%                         predictive density, N.B. the monitoring needs
%                         both xt and yt.
%      yt               - observed yt in test points. Used for monitoring
%                         the mean log predictive density, N.B. the
%                         monitoring needs both xt and yt.
%      z                - optional observed quantity in triplet
%                         (x_i,y_i,z_i) Some likelihoods may use this. For
%                         example, in case of Poisson likelihood we have
%                         z_i=E_i, that is, expected value for ith case.
%      zt               - optional observed quantity in triplet
%                         (xt_i,yt_i,zt_i) Some likelihoods may use this.
%                         For example, in case of Poisson likelihood we
%                         have z_i=E_i, that is, the expected value for the
%                         ith case. N.B. used only in the monitoring of the
%                         mean log predictive density.
%      X_u              - inducing inputs. If omitted, kmeans clustering is
%                         applied and the resulting cluster centroids are
%                         selected. The number of inducing variables is
%                         then controlled by the parameter nu (see below).
%      nu               - the number of inducing variables if X_u is
%                         omitted. Can be given as an absolute value or
%                         relative to the input size. The default is
%                         min(floor(0.1*n),1500), where n is the number
%                         of training inputs.
%      m                - initial mean of the inducing variables. The
%                         default is zero vector.
%      S                - initial covariance of the inducing variables. The
%                         default is 0.1 times identity matrix.
%      n_minibatch      - absolute or relative size of the minibatches
%                         (relative to the number of the training inputs).
%                         Does not have to be divisible with the number of
%                         training inputs. The default is 0.1 (relative).
%      maxiter          - the maximum number of iterations (default 5000).
%                         Providing 0 initialises the gp structure
%                         parameters without optimisation.
%      momentum         - momentum term for the covariance function
%                         parameters (default 0.9)
%      mu1              - initial step size of the variational parameters
%                         (default 0.01).
%      mu2              - initial step size of the likelihood and
%                         covariance function parameters (default 1e-5)
%      tol              - tolerance of energy for the convergence
%                         (default 1e-6).
%      step_size        - function handle for the step size as a function
%                         of the iteration. The default function is f(i) =
%                         1/(1+i/n_minibatch), where n_minibatch is the
%                         size of the minibatches.
%      lik_sigma2       - likelihood variance in the case of non-gaussian
%                         likelihood (default 0.1).
%      lik_sigma2_prior - prior for the likelihood variance in the case of
%                         non-gaussian likelihood. The default is 
%                         prior_loggaussian('mu',-2, 's2', 0.5).
%      display          - Control the amount of diagnostic verbosity.
%                         'off' displays nothing, 'final' display the
%                         final output, and 'iter' displays output at
%                         each iteration. Alternatively by providing a
%                         scalar value, fewer iterations can be displayed,
%                         e.g. value 10 displays only every tenth
%                         value (default behaviour).
%
%  See also
%    GP_SET, GPSVI_PRED, GPSVI_PREDGRAD, GPSVI_E, GPSVI_G, DEMO_SVI*
%
%  References:
%    Hensman, J., Fusi, N. and Lawrence, N. D. (2013). Gaussian processes
%    for big data. arXiv preprint arXiv:1309.6835.

% Copyright (c) 2014 Ville Tolvanen
% Copyright (c) 2014 Tuomas Sivula

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.


ip=inputParser;
ip.FunctionName = 'SVIGP';
ip.addRequired('gp',@isstruct);
ip.addRequired('x', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addRequired('y', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addParamValue('xt', [], @(x) isnumeric(x) && isreal(x) && all(isfinite(x(:))))
ip.addParamValue('yt', [], @(x) isnumeric(x) && isreal(x) && all(isfinite(x(:))))
ip.addParamValue('z', [], @(x) isnumeric(x) && isreal(x) && all(isfinite(x(:))))
ip.addParamValue('zt', [], @(x) isnumeric(x) && isreal(x) && all(isfinite(x(:))))
ip.addParamValue('X_u', [], @(x) isnumeric(x) && isreal(x) && all(isfinite(x(:))))
ip.addParamValue('nu', [], @(x) isreal(x) && isscalar(x) && x > 0)
ip.addParamValue('m', [], @(x) isnumeric(x) && isreal(x) && all(isfinite(x(:))))
ip.addParamValue('S', [], @(x) isnumeric(x) && isreal(x) && all(isfinite(x(:))))
ip.addParamValue('n_minibatch', 0.1, @(x) isreal(x) && isscalar(x) && x > 0)
ip.addParamValue('maxiter', 2000, @(x) isreal(x) && isscalar(x) && x >= 0)
ip.addParamValue('momentum', 0.9, @(x) isreal(x) && isscalar(x) && x > 0)
ip.addParamValue('mu1', 0.01, @(x) isreal(x) && isscalar(x) && x > 0)
ip.addParamValue('mu2', 1e-5, @(x) isreal(x) && isscalar(x) && x > 0)
ip.addParamValue('tol', 1e-6, @(x) isreal(x) && isscalar(x))
ip.addParamValue('step_size', [], @(x) isa(x,'function_handle'))
ip.addParamValue('lik_sigma2',0.1, @(x) isscalar(x) && x>=0);
ip.addParamValue('lik_sigma2_prior',prior_loggaussian('mu',-2, 's2', 0.5), ...
  @(x) isstruct(x) || isempty(x));
ip.addParamValue('display', 10, @(x) ismember(x,{'final','iter','off'}) ...
  || (isreal(x) && isscalar(x) && x > 1) )

ip.parse(gp, x,y,varargin{:});
x=ip.Results.x;
y=ip.Results.y;
z=ip.Results.z;
xt=ip.Results.xt;
yt=ip.Results.yt;
zt=ip.Results.zt;
X_u=ip.Results.X_u;
nu=ip.Results.nu;
m=ip.Results.m;
S=ip.Results.S;
n_minibatch=ip.Results.n_minibatch;
momentum=ip.Results.momentum;
mu1=ip.Results.mu1;
mu2=ip.Results.mu2;
step_size=ip.Results.step_size;
maxiter=ip.Results.maxiter;
tol=ip.Results.tol;
lik_sigma2 = ip.Results.lik_sigma2;
lik_sigma2_prior = ip.Results.lik_sigma2_prior;
display = ip.Results.display;

% Initialise the diagnosis output structure
diagnosis = struct();

% Check if latent method SVI has been set
if ~isfield(gp, 'latent_method') || ~isequal(gp.latent_method, 'SVI')
  gp=gp_set(gp, 'latent_method', 'SVI');
end

% Check if latent method is not gaussian or probit
if ~strcmp(gp.lik.type, 'Gaussian') && ~strcmp(gp.lik.type, 'Probit')
  error('Supported likelihoods for SVIGP are gaussian and probit.')
end

% Process parameters
if xor(isempty(xt), isempty(yt))
  warning('Need both xt and yt for monitoring mean log predictive density.');
end
n=size(x,1);

if n_minibatch < 1
  n_minibatch = max(floor(n_minibatch*n),1);
end
if n_minibatch > n
  n_minibatch = max(floor(0.1*n),1);
  warning('Too many minibatches, using floor(0.1*n) = %d instead.', ...
    n_minibatch)
end

if isempty(step_size)
  step_size=@(iter) 1/(1+iter./n_minibatch);
end

% Handle the inducing inputs
if ~ismember('X_u',ip.UsingDefaults)
  gp.X_u = X_u;
end
if isempty(gp.X_u)
  % Assign X_u by clustering
  if isempty(nu)
    nu = min(max(floor(0.1.*n),1),1500);
  elseif nu < 1
    nu = max(floor(nu.*n),1);
  end
  fprintf('Assign inducing inputs by clustering\n')
  Sw=warning('off','stats:kmeans:EmptyCluster');
  [~,X_u] = kmeans(x, nu,'Start','uniform',...
    'EmptyAction','singleton');
  warning(Sw);
  gp.X_u = X_u;
end
gp.nind = size(gp.X_u,1);

% Handle the rest of the parameters
if ~ismember('m',ip.UsingDefaults)
  if length(m) == gp.nind
    gp.m = m;
  else
    error('The size of m does not match with X_u')
  end
elseif ~isfield(gp, 'm') || length(gp.m) ~= gp.nind
  gp.m = zeros(gp.nind,1);
end

if ~ismember('S',ip.UsingDefaults)
  if ismatrix(S) && all(size(S) == gp.nind)
    gp.S = S;
  else
    error('The size of S does not match with X_u')
  end
elseif ~isfield(gp, 'S') || any(size(gp.S) ~= gp.nind)
  gp.S = 0.1*eye(gp.nind);
  % gp.S = gp_trcov(gp,gp.X_u);
end
gp.t1=gp.S\gp.m;
gp.t2=-0.5.*inv(gp.S);

% Handle the likelihood variance
if ~isfield(gp.lik, 'sigma2')
  gp.lik.sigma2 = lik_sigma2;
  gp.lik.p.sigma2 = lik_sigma2_prior;
  gp.lik.fh.lp = @lik_lp;
  gp.lik.fh.lpg = @lik_lpg;
end

% Return prematurely if only initialising
if maxiter == 0
  return
end

% Parameters
w = gp_pak(gp);
w0 =w;
nh1 = numel(gp.t1) + numel(gp.t2);
nh2 = length(w)-nh1;

% Initial step-size vector
mu0 = mu1.*ones(size(w));
mu0(end-nh2+1:end)=mu2;

% Size of minibatches
nb=n_minibatch;
% Number of minibatches
nbb=ceil(n/nb);

% Monitored values
if nargout > 1
  e_all = zeros(maxiter,nbb);
  w_all = zeros(maxiter,nbb,nh2);
  if ~isempty(xt) && ~isempty(yt)
    mlpd_all = zeros(maxiter,1);
    rmse_all = zeros(maxiter,1);
  end
end
g_old=zeros(size(w));
momentum=momentum.*ones(size(w));
etot_old=Inf;

% Preprocess conditions for iteration
disp_iter = strcmp(display, 'iter');
disp_i = 0;
disp_count = display;

% Iterate until convergence or maxiter
converged = 0;
try_fix_mu2 = 0;
for iter = 1:maxiter
  
  % Divide the data into minibatches
  inds = cell(nbb,1);
  ind = randperm(n);
  for i=1:nbb-1
    inds{i} = ind((i-1)*nb+1:i*nb);
  end
  inds{nbb} = ind((nbb-1)*nb+1:end);
  
  % Compute the step-size
  mu = mu0.*step_size(iter);
  
  % Iterate all the minibatches
  etot = 0;
  broken = 0;
  for i=1:nbb
    if isempty(z)
      zi = [];
    else
      zi = z(inds{i},:);
    end
    gp.data_prop=length(inds{i})./n;
    [e,~,~,param] = gpsvi_e(w,gp,x(inds{i},:),y(inds{i},:), 'z', zi);
    etot = etot+e;
    g = gpsvi_g(w,gp,x(inds{i},:),y(inds{i},:), 'z', zi, ...
      'gpsvi_e_param', param);
    g = mu.*g + momentum.*g_old;
    g_old = g;
    w = w+g;
    if nargout > 1
      e_all(iter,i) = e;
      w_all(iter,i,:) = w(end-nh2+1:end);
    end
    if ~isnan(etot) ...
        && all(~isinf(exp(w(end-nh2+1:end)))) ...
        && all(exp(w(end-nh2+1:end))~=0) ...
        && ~any(isnan(g)) ...
        && ~isnan(gpsvi_e(w+g,gp,x(inds{i},:),y(inds{i},:), 'z', zi))
      gp = gp_unpak(gp,w);
    elseif ~try_fix_mu2
      fprintf('Bad parameter values, decreasing mu2.\n');
      try_fix_mu2 = 1;
      mu2 = 0.1*mu2;
      mu0(end-nh2+1:end)=mu2;
      w = w0;
      g_old = zeros(1,nh1+nh2);
      broken = 1;
      break
    else
      fprintf('Bad parameter values, decreasing step-size and momentum.\n');
      mu0 = 0.1.*mu0;
      momentum = 0.5*momentum;
      w = w0;
      g_old = zeros(1,nh1+nh2);
      broken = 1;
      break
    end
    gpsvi_e('clearcache',gp);
  end
  if broken
    continue
  end
  etot=etot/nbb;
  
  % Analyse and print
  if isscalar(display)
    if disp_count == display
      disp_i = 1;
      disp_count = 1;
    else
      disp_i = 0;
      disp_count = disp_count +1;
    end
  end
  if ~isempty(xt) && ~isempty(yt) ...
      && ( disp_iter || nargout > 1 || disp_i)
    [Eft,~,lpyt] = gpsvi_pred(gp,x,y,xt,'yt',yt, 'z', zi, 'zt', zt);
    lpyt = mean(lpyt);
    if nargout > 1
      mlpd_all(iter) = lpyt;
      rmse = sqrt(mean((yt-Eft).^2));
      rmse_all(iter) = rmse;
    end
    if disp_iter || disp_i
      fprintf('iter=%d/%d, e=%.3f, mlpd=%.4g, rmse=%.4g, de=%.5g\n', ...
        iter, maxiter, etot, lpyt, rmse, abs(etot-etot_old));
    end
  elseif disp_iter || disp_i
    fprintf('iter=%d/%d, e=%.3f, de=%.5g\n', ...
      iter, maxiter, etot, abs(etot-etot_old));
  end
  
  % Check for convergence
  if abs(etot-etot_old)<tol
    converged = 1;
    if strcmp(display, 'iter')
      fprintf('Energy converged\n')
    end
    break
  end
  etot_old=etot;
  
end

% Display results
if ~converged
  fprintf('Iteration limit %d reached while optimising the parameters\n', ...
    maxiter)
elseif ~strcmp(display, 'off')
  fprintf('Tolerance reached in %d iterations\n', iter)
end
if strcmp(display, 'final') || isscalar(display)
  if ~isempty(xt) && ~isempty(yt) && ( nargout > 1 || disp_i)
    fprintf(['Final values:\n' ...
      'e=%.3f, mlpd=%.4g, rmse=%.4g\n'], etot, lpyt, rmse);
  elseif ~isempty(xt) && ~isempty(yt)
    [Eft,~,lpyt] = gpsvi_pred(gp,x,y,xt,'yt',yt, 'z', z, 'zt', zt);
    fprintf(['Final values:\n' ...
      'e=%.3f, mlpd=%.4g, rmse=%.4g\n'], ...
      etot, mean(lpyt)), sqrt(mean((yt-Eft).^2));
  else
    fprintf('Final energy: %.3f\n', etot);
  end
end

% Save the monitored values
if nargout > 1
  diagnosis.e = e_all(1:iter,:);
  diagnosis.w = w_all(1:iter,:,:);
  if ~isempty(xt) && ~isempty(yt)
    diagnosis.mlpd = mlpd_all(1:iter);
    diagnosis.rmse = rmse_all(1:iter);
  end
end

end


function lp = lik_lp(lik, varargin)
%LIK_LP  log(prior) of the likelihood parameters
%
%  Description
%    LP = LIK_LP(LIK) takes a likelihood structure LIK and
%    returns log(p(th)), where th collects the parameters. This
%    subfunction is needed if there are likelihood parameters.
%    Added for non-gaussian likelihoods in SVIGP.
%
%  See also
%    LIK_*, SVIGP


% If prior for sigma2sion parameter, add its contribution
lp=0;
if ~isempty(lik.p.sigma2)
  lp = lik.p.sigma2.fh.lp(lik.sigma2, lik.p.sigma2) +log(lik.sigma2);
end
end


function lpg = lik_lpg(lik)
%LIK_LPG  d log(prior)/dth of the likelihood parameters
%
%  Description
%    E = LIK_NEGBIN_LPG(LIK) takes a likelihood structure LIK and
%    returns d log(p(th))/dth, where th collects the parameters.
%    This subfunction is needed if there are likelihood parameters.
%    Added for non-gaussian likelihoods in SVIGP.
%
%  See also
%    LIK_*, SVIGP

lpg=[];
if ~isempty(lik.p.sigma2)
  % Evaluate the gprior with respect to sigma2
  ggs = lik.p.sigma2.fh.lpg(lik.sigma2, lik.p.sigma2);
  lpg = ggs(1).*lik.sigma2 + 1;
  if length(ggs) > 1
    lpg = [lpg ggs(2:end)];
  end
end
end

