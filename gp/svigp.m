function [gp, diag] = svigp(gp, x, y, varargin)
%SVIGP Summary of this function goes here
%   Detailed explanation goes here
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
ip.addParamValue('nu', [], @(x) isreal(x) && isscalar(x))
ip.addParamValue('m', [], @(x) isnumeric(x) && isreal(x) && all(isfinite(x(:))))
ip.addParamValue('S', [], @(x) isnumeric(x) && isreal(x) && all(isfinite(x(:))))
ip.addParamValue('n_minibatch', [], @(x) isreal(x) && isscalar(x))
ip.addParamValue('maxiter', [], @(x) isreal(x) && isscalar(x))
ip.addParamValue('momentum', 0.9, @(x) isreal(x) && isscalar(x))
ip.addParamValue('mu1', 0.01, @(x) isreal(x) && isscalar(x))
ip.addParamValue('mu2', 1e-5, @(x) isreal(x) && isscalar(x))
ip.addParamValue('tol', 1e-6, @(x) isreal(x) && isscalar(x))
% ip.addParamValue('optimf', @fminscg, @(x) isa(x,'function_handle'))
ip.addParamValue('step_size', [], @(x) isa(x,'function_handle'))
ip.addParamValue('opt', [], @isstruct)
ip.addParamValue('optimize', 'off', @(x) ismember(x, {'on', 'off'}));
ip.addParamValue('nvd', [], @(x) isreal(x));
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
momentum=ip.Results.momentum;
mu1=ip.Results.mu1;
mu2=ip.Results.mu2;
step_size=ip.Results.step_size;
maxiter=ip.Results.maxiter;
tol=ip.Results.tol;

if xor(isempty(xt), isempty(yt))
  warning('Need both xt and yt for monitoring mean log predictive density.');
end

[n,nin]=size(x);
if isempty(nu)
  nu=floor(0.1.*n);
end
if isempty(n_minibatch)
  n_minibatch=floor(0.1.*n);
end
if isempty(step_size)
  step_size=@(iter) 1/(1+iter./n_minibatch);
end
if isempty(X_u) && ~isfield(gp, 'X_u')
  Sw=warning('off','stats:kmeans:EmptyCluster');
  [~,X_u] = kmeans(x, nu,'Start','uniform',...
    'EmptyAction','singleton');
  warning(Sw);
end
% gp=gpsvi_e('init',gp);
if isempty(m)
  m=zeros(size(X_u,1),1);
end
if isempty(S)
  S=gp_trcov(gp,X_u);
end
if ~isfield(gp, 'latent_method') || ~isequal(gp.latent_method, 'SVI')
  gp=gpsvi_e('init',gp);
  gp.latent_method='SVI';
%   gp=gp_set(gp, 'latent_method', 'SVI');
end
if ~isfield(gp, 'X_u')
  gp.X_u=X_u;
end
if ~isfield(gp, 'm')
  gp.m=m;
end
if ~isfield(gp, 'S')
  gp.S=S;
end
if ~isfield(gp.lik, 'sigma2')
  gp.lik.sigma2=0.1;
end
% gp.X_u=X_u;
% gp.type='SVIGP';
gp.t1=gp.S\gp.m;
% gp.m=m;
% gp.S=S;
gp.t2=-0.5.*inv(gp.S);
w=gp_pak(gp);
w0=w;
nh1=numel(gp.t1)+numel(gp.t2);
nh2=length(w)-nh1;

% Initial step-size vector
mu0=mu1.*ones(size(w0));
mu0(end-nh2+1:end)=mu2;
% mu0(end-2:end)=1e-4;
if isempty(maxiter)
  maxiter=5000;
end

% Divide the data to minibatches
% Size of minibatches
nb=n_minibatch;
gp.data_prop=nb./n;
% Number of minibatches
nbb=ceil(n/nb);
ind=1:n;
for i=1:nbb-1
  randi=randsample(n-(i-1)*nb,nb);
  inds{i}=ind(randi);
  ind(randi)=[];
end
inds{nbb}=ind;

e=zeros(nbb,maxiter);
g_old=zeros(size(w0));
momentum=momentum.*ones(size(w0));
wa=[];ga=[];
etot_old=Inf;

% Iterate until convergence or maxiter
for iter=1:maxiter
  % Compute the step-size
  mu=mu0.*step_size(iter);
%   mu=mu0./(1+iter/nb);
  ivec=1:nbb;
  ivec=ivec(randsample(length(ivec),length(ivec)));
  for i=ivec
    gp.data_prop=length(inds{i})./n;
    e(i,iter)=gpsvi_e(w,gp,x(inds{i},:),y(inds{i},:));
    if ~isnan(e(i,iter))
      g=gpsvi_g(w,gp,x(inds{i},:),y(inds{i},:));
      g=mu.*g + momentum.*g_old;
      g_old=g;
      ga(end+1,:)=[g(end-2:end)];
      if all(~isinf(exp(w(end-2:end)+g(end-2:end)))) && all(exp(w(end-2:end)+g(end-2:end))~=0) ...
          && ~isnan(gpsvi_e(w+g,gp,x(inds{i},:),y(inds{i},:)))
        w=w+g;
        wa(end+1,:)=w(end-nh2+1:end);
        gp=gp_unpak(gp,w);
        gpsvi_e('clearcache',gp);
      else
        e(i,iter)=randn;
        gpsvi_e('clearcache',gp);
        fprintf('Bad parameter values, decresing step-size.\n');
        mu0=0.5.*mu0;
      end
    end
  end
  gpsvi_e('clearcache',gp);
  if ~isempty(xt) && ~isempty(yt)
    [Ef,Varf,lpyt,Ey,Vary]=gpsvi_pred(gp,x,y,xt,'yt',yt);
  else
    lpyt=[];
  end
  etot=sum(e(:,iter))/nbb;
  if abs(etot-etot_old)<tol
    fprintf('Energy converged\n');
    if ~isempty(lpyt)
      fprintf('iter=%d / %d, minibatch=%d / %d, e=%.3f, mlpd=%.3f\n', ...
        iter, maxiter, i, nbb, etot,mean(lpyt));
    else
      fprintf('iter=%d / %d, minibatch=%d / %d, e=%.3f, mlpd=%.3f\n', ...
        iter, maxiter, i, nbb, etot);
    end
    break;
  end
  
  if ~isempty(lpyt)
    fprintf('iter=%d / %d, minibatch=%d / %d, e=%.3f, mlpd=%.3f, de=%.5f\n', ...
      iter, maxiter, i, nbb, etot,mean(lpyt), abs(etot-etot_old));
  else
    fprintf('iter=%d / %d, minibatch=%d / %d, e=%.3f, de=%.5f\n', ...
      iter, maxiter, i, nbb, etot, abs(etot-etot_old));
  end
  etot_old=etot;
end
diag.e=e;
diag.w=wa;

end

