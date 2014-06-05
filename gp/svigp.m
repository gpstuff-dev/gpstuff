function [gp, diagnosis] = svigp(gp, x, y, varargin)
%SVIGP Summary of this function goes here
%   Detailed explanation goes here
ip=inputParser;
ip.FunctionName = 'SVIGP';
ip.addRequired('gp',@isstruct);
ip.addRequired('x', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addRequired('y', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addOptional('xt', [], @(x) isnumeric(x) && isreal(x) && all(isfinite(x(:))))
ip.addParamValue('yt', [], @(x) isnumeric(x) && isreal(x) && all(isfinite(x(:))))
ip.addParamValue('z', [], @(x) isnumeric(x) && isreal(x) && all(isfinite(x(:))))
ip.addParamValue('zt', [], @(x) isnumeric(x) && isreal(x) && all(isfinite(x(:))))
ip.addParamValue('X_u', [], @(x) isnumeric(x) && isreal(x) && all(isfinite(x(:))))
ip.addParamValue('nu', [], @(x) isreal(x) && isscalar(x))
ip.addParamValue('m', [], @(x) isnumeric(x) && isreal(x) && all(isfinite(x(:))))
ip.addParamValue('S', [], @(x) isnumeric(x) && isreal(x) && all(isfinite(x(:))))
ip.addParamValue('n_minibatch', [], @(x) isreal(x) && isscalar(x))
ip.addParamValue('minibatch_frac', 0.1, @(x) isreal(x) && isscalar(x) && 0<x && x<1)
ip.addParamValue('maxiter', 5000, @(x) isreal(x) && isscalar(x))
ip.addParamValue('momentum', 0.9, @(x) isreal(x) && isscalar(x))
ip.addParamValue('mu1', 0.01, @(x) isreal(x) && isscalar(x))
ip.addParamValue('mu2', 1e-5, @(x) isreal(x) && isscalar(x))
ip.addParamValue('tol', 1e-6, @(x) isreal(x) && isscalar(x))
ip.addParamValue('step_size', [], @(x) isa(x,'function_handle'))
ip.addParamValue('display', 'on', @(x) ismember(x,{'on', 'off'}))

ip.parse(gp, x,y,varargin{:});
x=ip.Results.x;
y=ip.Results.y;
z=ip.Results.z;
xt=ip.Results.xt;
yt=ip.Results.yt;
X_u=ip.Results.X_u;
nu=ip.Results.nu;
m=ip.Results.m;
S=ip.Results.S;
n_minibatch=ip.Results.n_minibatch;
minibatch_frac=ip.Results.minibatch_frac;
momentum=ip.Results.momentum;
mu1=ip.Results.mu1;
mu2=ip.Results.mu2;
step_size=ip.Results.step_size;
maxiter=ip.Results.maxiter;
tol=ip.Results.tol;
display = strcmp(ip.Results.display, 'on');

% Check if latent method SVI has been set
if ~isfield(gp, 'latent_method') || ~isequal(gp.latent_method, 'SVI')
  gp=gp_set(gp, 'latent_method', 'SVI');
end

% Process parameters
if xor(isempty(xt), isempty(yt))
  warning('Need both xt and yt for monitoring mean log predictive density.');
end
n=size(x,1);
if isempty(n_minibatch)
  n_minibatch=floor(minibatch_frac.*n);
end
if n_minibatch > n
  n_minibatch=floor(minibatch_frac.*n);
  warning('Too many minibatches, using %.2f*n = %d instead.', ...
    minibatch_frac, n_minibatch)
end
if isempty(step_size)
  step_size=@(iter) 1/(1+iter./n_minibatch);
end

% Handle the inducing variables
if ~ismember('X_u',ip.UsingDefaults)
  gp.X_u = X_u;
  gp.nind = size(X_u,1);
end
if isempty(gp.X_u)
  % Assign X_u by clustering
  if isempty(nu)
    nu=min(floor(0.1.*n),1500);
  end
  fprintf('Assign inducing variables by clustering\n')
  Sw=warning('off','stats:kmeans:EmptyCluster');
  [~,X_u] = kmeans(x, nu,'Start','uniform',...
    'EmptyAction','singleton');
  warning(Sw);
  gp.X_u = X_u;
  gp.nind = size(X_u,1);
end

% Handle the rest of the parameters parameters
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

if ~isfield(gp.lik, 'sigma2')
  gp.lik.sigma2=0.1;
end
gp.t1=gp.S\gp.m;
gp.t2=-0.5.*inv(gp.S);


% Hyperparameters
w=gp_pak(gp);
w0=w;
nh1=numel(gp.t1)+numel(gp.t2);
nh2=length(w)-nh1;

% Initial step-size vector
mu0=mu1.*ones(size(w));
mu0(end-nh2+1:end)=mu2;
% mu0(end-2:end)=1e-4;

% Size of minibatches
nb=n_minibatch;
% Number of minibatches
nbb=ceil(n/nb);

% Monitored values
if nargout > 1
  e_all=zeros(maxiter,nbb);
  w_all=zeros(maxiter,nbb,nh2);
end

g_old=zeros(size(w));
momentum=momentum.*ones(size(w));
etot_old=Inf;

% Iterate until convergence or maxiter
converged = 0;
for iter=1:maxiter
  
  % Divide the data into minibatches
  inds = cell(nbb,1);
  ind = randperm(n);
  for i=1:nbb-1
    inds{i} = ind((i-1)*nb+1:i*nb);
  end
  inds{nbb} = ind((nbb-1)*nb+1:end);
  
  % Compute the step-size
  mu=mu0.*step_size(iter);
  
  etot = 0;
  for i=1:nbb
    gp.data_prop=length(inds{i})./n;
    e = gpsvi_e(w,gp,x(inds{i},:),y(inds{i},:), 'z', z);
    if nargout > 1
      e_all(iter,i) = e;
    end
    etot = etot+e;
    g=gpsvi_g(w,gp,x(inds{i},:),y(inds{i},:));
    g=mu.*g + momentum.*g_old;
    g_old=g;
    if ~isnan(etot) ...
        && all(~isinf(exp(w(end-2:end)+g(end-2:end)))) ...
        && all(exp(w(end-2:end)+g(end-2:end))~=0) ...
        && ~any(isnan(g)) ...
        && ~isnan(gpsvi_e(w+g,gp,x(inds{i},:),y(inds{i},:), 'z', z))
      w=w+g;
      if nargout > 1
        w_all(iter,i,:) = w(end-nh2+1:end);
      end
      gp=gp_unpak(gp,w);
    else
      fprintf('Bad parameter values, decreasing step-size and momentum.\n');
      mu0=0.1.*mu0;
      momentum=0.5*momentum;
      w=w0;
      g_old=zeros(1,nh1+nh2);
      break
    end
    gpsvi_e('clearcache',gp);
  end
  
  % Check for convergence
  etot=etot/nbb;
  if display
    if ~isempty(xt) && ~isempty(yt)
      [~,~,lpyt]=gpsvi_pred(gp,x,y,xt,'yt',yt);
      fprintf('iter=%d / %d, e=%.3f, mlpd=%.3f, de=%.5f\n', ...
        iter, maxiter, etot, mean(lpyt), abs(etot-etot_old));
    else
      fprintf('iter=%d / %d, e=%.3f, de=%.5f\n', ...
        iter, maxiter, etot, abs(etot-etot_old));
    end
  end
  if abs(etot-etot_old)<tol
    converged = 1;
    if display
      fprintf('Energy converged\n')
    end
    break
  end
  etot_old=etot;
  
end

if ~converged
  fprintf('Iteration limit reached while optimising the parameters\n')
end

if nargout > 1
  diagnosis.e = e_all;
  diagnosis.w = w_all;
end

end
