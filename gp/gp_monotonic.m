function gp = gp_monotonic(gp, varargin)
%GP_MONOTONIC Converts GP structure to monotonic GP and optimizes the
%             hyperparameters
%
%  Description
%    GP = GP_MONOTONIC(GP, X, Y, OPTIONS) takes a GP structure GP
%    together with a matrix X of input vectors and a matrix Y of
%    target vectors and converts the GP structure to monotonic GP, where
%    the latent function is assumed to be monotonic w.r.t input-dimensions.
%    In addition, this function can optimize the hyperparameters of the
%    GP (optimize='on'). Monotonicity is forced by GP prior by adding virtual observations 
%    XV and YV where the function derivative is assumed to be either
%    positive (YV=1, default) or negative (YV=-1). This function adds
%    virtual observations from the true observations two at a time until the
%    monotonicity is satisfied in every observation point. If GP doesn't
%    have field XV, the virtual observations are initialized randomly to
%    25% of the true observed inputs. Return monotonic GP structure GP with
%    optimized hyperparameters.
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
%                 be monotonic. Default is all the dimensions.
%      dir      - Whether the latent function is assumed to be
%                 monotonically increasing (dir=1, default) or decreasing
%                 (dir=-1).
%
%  See also
%    GP_SET
%
%  References
%     
%    Riihimäki (2009). Gaussian processes with monotonicity information. 
%    International Conference on Artificial Intelligence and Statistics 
%    (AISTATS), 2010, pp. 645-652.

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
ip.addParamValue('optimf', @fminscg, @(x) isa(x,'function_handle'))
ip.addParamValue('opt', [], @isstruct)
ip.addParamValue('optimize', 'off', @(x) ismember(x, {'on', 'off'}));
ip.addParamValue('nvd', [], @(x) isreal(x) && all(x>0));
ip.addParamValue('dir', 1, @(x) isscalar(x) && sum(x==[-1 1])==1);
ip.parse(gp, varargin{:});
x=ip.Results.x;
y=ip.Results.y;
z=ip.Results.z;
nv=ip.Results.nv;
opt=ip.Results.opt;
optimf=ip.Results.optimf;
optimize=ip.Results.optimize;
nvd=ip.Results.nvd;
dir=ip.Results.dir;
% Check appropriate fields in GP structure and modify if necessary to make
% proper monotonic GP structure
if ~isfield(gp, 'lik2') || ~isequal(gp.lik.type, 'Probit')
  lik=gp.lik;
  gp=gp_set(gp,'lik', lik_probit());
  gp.lik2=lik;
end
gp.derivobs=1;
% Set the virtual observations, here we use 25% of the observations as
% virtual points at initialization
if isempty(nv)
  frac=0.25;
  nv=floor(frac.*size(x,1));
end
if ~isempty(dir)
  gp.yv=dir;
else
  if ~isfield(gp, 'yv')
    % Set the latent function as increasing
    gp.yv=1;
  end
end
if ~isempty(nvd)
  gp.nvd=nvd;
  nvd=length(nvd);
else
  if isfield(gp, 'nvd') && ~ismember('nvd',ip.UsingDefaults(:)) 
    gp=rmfield(gp, 'nvd');
    nvd=size(x,2);
  else
    nvd=length(gp.nvd);
  end
end
if ~isfield(gp, 'xv')
  gp.xv=x(randsample(size(x,1),nv),:);
  xv=gp.xv;
end
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
Ef=gp_pred(gp,x,y,x, 'z', z);
Ef=Ef(size(x,1)+1:end);
Ef=reshape(Ef,n,nvd);
% Check whether monotonicity is satisfied
while any(Ef(:).*gp.yv<0)
  % Monotonicity not satisfied, add 2 "most wrong" predictions from the
  % observation set to the virual observations.
  fprintf('Latent function not monotonic, adding virtual observations.\n');
  gp.lik.nu=1e-6;
  for j=1:nvd
    [~,ind(:,j)]=sort(Ef(:,j).*gp.yv,'ascend');
  end
  ind=ind(1:2,:);
  inds=unique(ind(:));
  fprintf('Added %d virtual observations.\n', length(inds));
  xv=[xv;x(inds,:)];
  gp.xv=xv;
  gpep_e('clearcache',gp);
  if isequal(optimize, 'on')
    gp=gp_optim(gp,x,y,'opt',opt,'z',z, 'optimf', optimf);
  end
  Ef=gp_pred(gp,x,y,x,'z',z);
  Ef=Ef(size(x,1)+1:end);
  Ef=reshape(Ef,n,nvd);
end

end

