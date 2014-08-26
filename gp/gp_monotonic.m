function gp = gp_monotonic(gp, varargin)
%GP_MONOTONIC Converts GP structure to monotonic GP and optimizes the
%             hyperparameters
%
%  Description
%    GP = GP_MONOTONIC(GP, X, Y, OPTIONS) takes a GP structure GP
%    together with a matrix X of input vectors and a matrix Y of
%    target vectors and converts the GP structure to monotonic GP,
%    where the latent function is assumed to be monotonic w.r.t
%    input-dimensions.  In addition, this function can optimize the
%    hyperparameters of the GP (optimize='on'). Monotonicity is forced
%    by GP prior by adding virtual observations XV and YV where the
%    function derivative is assumed to be either positive (YV=1,
%    default) or negative (YV=-1). This function adds virtual
%    observations from the true observations two at a time until the
%    monotonicity is satisfied in every observation point. If GP
%    doesn't have field XV, the virtual observations are initialized
%    randomly to 25% of the true observed inputs. Return monotonic GP
%    structure GP with optimized hyperparameters.
%
%    OPTIONS is optional parameter-value pair
%      z        - optional observed quantity in triplet (x_i,y_i,z_i)
%                 Some likelihoods may use this. For example, in case of
%                 Poisson likelihood we have z_i=E_i, that is, expected
%                 value for ith case.
%      nv       - Number of virtual observations to be used at initialization.
%                 Default value is 0.25*n where n is the number of observations.
%                 The virtual observations are randomly selected from the
%                 true observed inputs.
%      opt      - Options structure for optimizer.
%      optimf   - function handle for an optimization function, which is
%                 assumed to have similar input and output arguments
%                 as usual fmin*-functions. Default is @fminscg.
%      optimize - Option whether to optimize GP parameters. Default = 'off'. 
%      nvd      - Dimensions for which the latent functions is assumed to
%                 be monotonic. Use negative elements for monotonically
%                 decreasing and positive elements for monotonically
%                 increasing dimensions. Default 1:size(X,2), i.e.
%                 monotonically for all covariate dimensions.
%
%  See also
%    GP_SET
%
%  Reference
%    Riihimäki and Vehtari (2010). Gaussian processes with
%    monotonicity information.  Journal of Machine Learning Research:
%    Workshop and Conference Proceedings, 9:645-652.
%
% Copyright (c) 2014 Ville Tolvanen

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

% parse inputs
ip=inputParser;
ip.FunctionName = 'GP_MONOTONIC';
ip.addRequired('gp',@isstruct);
ip.addOptional('x', [], @(x) isnumeric(x) && isreal(x) && all(isfinite(x(:))))
ip.addOptional('y', [], @(x) isnumeric(x) && isreal(x) && all(isfinite(x(:))))
ip.addParamValue('z', [], @(x) isnumeric(x) && isreal(x) && all(isfinite(x(:))))
ip.addParamValue('nv', [], @(x) isreal(x) && isscalar(x))
ip.addParamValue('xv', [], @(x) isnumeric(x) && isreal(x) && all(isfinite(x(:))))
ip.addParamValue('optimf', @fminscg, @(x) isa(x,'function_handle'))
ip.addParamValue('opt', [], @isstruct)
ip.addParamValue('optimize', 'off', @(x) ismember(x, {'on', 'off'}));
ip.addParamValue('nvd', [], @(x) isreal(x));
ip.parse(gp, varargin{:});
x=ip.Results.x;
y=ip.Results.y;
z=ip.Results.z;
nv=ip.Results.nv;
opt=ip.Results.opt;
optimf=ip.Results.optimf;
optimize=ip.Results.optimize;
nvd=ip.Results.nvd;
% Check appropriate fields in GP structure and modify if necessary to make
% proper monotonic GP structure
if ~isfield(gp, 'lik_mono') || ~ismember(gp.lik_mono.type, {'Probit', 'Logit'}) 
  gp.lik_mono=lik_probit();
end
gp.derivobs=1;
% Set the virtual observations, here we use 25% of the observations as
% virtual points at initialization
if isempty(nv)
  frac=0.25;
  nv=floor(frac.*size(x,1));
end
if ~isempty(nvd)
  gp.nvd=nvd;
else
  if isfield(gp, 'nvd') && ~ismember('nvd',ip.UsingDefaults(:)) 
  else
    if ~isfield(gp, 'nvd')
      gp.nvd=1:size(x,2);
    end
  end
end
nvd=length(gp.nvd);
if ~isfield(gp, 'xv')
  S=warning('off','stats:kmeans:EmptyCluster');
  [tmp,xv]=kmeans(x, nv, 'Start','uniform', ...
    'EmptyAction', 'singleton');
  warning(S);
  gp.xv=xv;
end
xv=gp.xv;
if isempty(opt) || ~isfield(opt, 'TolX')
  % No options structure given or not a proper options structure
  opt=optimset('TolX',1e-4,'TolFun',1e-4,'Display','iter');
end
gp=gp_set(gp,'latent_method','EP');
gp.latent_opt.init_prev='off';
gp.latent_opt.maxiter=100;
gpep_e('clearcache',gp);
if isequal(optimize, 'on')
  % Optimize the parameters
  gp=gp_optim(gp,x,y,'opt',opt, 'z', z, 'optimf', optimf);
end
n=size(x,1);
[tmp,itst]=cvit(size(x,1),10);
% Predict gradients at the training points 
Ef=zeros(size(x,1),nvd);
for i=1:10
  % Predict in blocks to save memory
  Ef(itst{i},:)=gpep_predgrad(gp,x,y,x(itst{i},:),'z',z);
end
% Check if monotonicity is satisfied
yv=round(gp.nvd./abs(gp.nvd));
while any(any(bsxfun(@times,Ef, yv)<0))
  % Monotonicity not satisfied, add 2 "most wrong" predictions, for each 
  % dimension, from the observation set to the virual observations.
  fprintf('Latent function not monotonic, adding virtual observations.\n');
  gp.lik_mono.nu=1e-6;
  for j=1:nvd
    [~,ind(:,j)]=sort(Ef(:,j).*yv(j),'ascend');
  end
  ind=ind(1:2,:);
  inds=unique(ind(:));
  clear ind;
  fprintf('Added %d virtual observations.\n', length(inds));
  xv=[xv;x(inds,:)];
  gp.xv=xv;
  gpep_e('clearcache',gp);
  if isequal(optimize, 'on')
    gp=gp_optim(gp,x,y,'opt',opt,'z',z, 'optimf', optimf);
  end
  % Predict gradients at the training points
  %Ef=gpep_predgrad(gp,x,y,x,'z',z);
  for i=1:10
    % Predict in blocks to save memory
    Ef(itst{i},:)=gpep_predgrad(gp,x,y,x(itst{i},:),'z',z);
  end
end

end

