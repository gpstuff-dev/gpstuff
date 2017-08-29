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
%    sampling from X or using K-means with K=floor(N/4), where N is
%    the number of true observations. Return monotonic GP structure GP
%    with optimized hyperparameters.
%
%    Note! The monotonic GP constructed by this function is a special case
%          for GPstuff models that follows (Riihimäki and Vehtari, 2010).
%          The monotonicity constraints can be set also manually as
%          described in "demo_derivatives". The difference is that the
%          monotonic GP structure returned by this function contains the
%          virtual observations in whereas in the manual approach
%          ("demo_derivatives") they are included into data. 
%
%    OPTIONS is optional parameter-value pair
%      z        - Optional observed quantity in triplet (x_i,y_i,z_i)
%                 Some likelihoods may use this. For example, in case of
%                 Poisson likelihood we have z_i=E_i, that is, expected
%                 value for ith case.
%      nv       - Number of virtual observations to be used at initialization.
%                 Default value is floor(n/4) where n is the number of observations.
%      init     - The method used to initialise the locations of virtual
%                 observations. 'sample' (default) which selects nv locations from X
%                 or 'kmeans' which uses K-means algorithm with K=nv.
%      nvd      - Dimensions for which the latent functions is assumed to
%                 be monotonic. Use negative elements for monotonically
%                 decreasing and positive elements for monotonically
%                 increasing dimensions. Default 1:size(X,2), i.e.
%                 monotonically for all covariate dimensions.
%      nu       - The strictness of the monotonicity information, with a 
%                 smaller values corresponding to the more strict information. 
%                 Default is 1e-6.
%      force    - Boolean value indicating whether the monotonicity is
%                 forced by adding virtual observations until the function
%                 becomes monotonic at the training points. Default = true
%      display  - true or false, indicating whether to display some
%                 information when relevant. Default = true.
%      optimize - Option whether to optimize GP parameters. Default = 'off'. 
%      opt      - Options structure for optimizer.
%      optimf   - Function handle for an optimization function, which is
%                 assumed to have similar input and output arguments
%                 as usual fmin*-functions. Default is @fminscg.
%
%  See also
%    GP_SET, DEMO_MONOTONIC
%
%  Reference
%    Riihimäki and Vehtari (2010). Gaussian processes with
%    monotonicity information.  Journal of Machine Learning Research:
%    Workshop and Conference Proceedings, 9:645-652.
%
% Copyright (c) 2014 Ville Tolvanen
% Copyright (c) 2015 Aki Vehtari
% Copyright (c) 2017 Jarno Vanhatalo

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
ip.addParamValue('nu', 1e-6, @(x) isreal(x) && isscalar(x) && (x>0))
ip.addParamValue('nv', [], @(x) isreal(x) && isscalar(x))
ip.addParamValue('init', 'sample', @(x) ismember(x, {'sample', 'kmeans'}));
ip.addParamValue('xv', [], @(x) isnumeric(x) && isreal(x) && all(isfinite(x(:))))
ip.addParamValue('force', true, @(x) islogical(x));
ip.addParamValue('display', true, @(x) islogical(x));
ip.addParamValue('optimf', @fminscg, @(x) isa(x,'function_handle'))
ip.addParamValue('opt', [], @isstruct)
ip.addParamValue('optimize', 'off', @(x) ismember(x, {'on', 'off'}));
ip.addParamValue('nvd', [], @(x) isreal(x));
ip.parse(gp, varargin{:});
x=ip.Results.x;
y=ip.Results.y;
z=ip.Results.z;
nu=ip.Results.nu;
nv=ip.Results.nv;
init=ip.Results.init;
force = ip.Results.force;
display = ip.Results.display;
opt=ip.Results.opt;
optimf=ip.Results.optimf;
optimize=ip.Results.optimize;
nvd=ip.Results.nvd;

% Check appropriate fields in GP structure and modify if necessary to make
% proper monotonic GP structure
if strcmp(gp.lik.type, 'lik_liks')
    error('gp_monotonic.m has not yet been implemented for many likelihoods (lik_liks)')
end
if isfield(gp,'deriv')
    error('gp_monotonic.m has not yet been implemented for models with derivative \n observations (other than the virtual observations used to code monotonicity)')
end
lik = lik_liks('likelihoods', {gp.lik, lik_probit('nu', nu)},'classVariables', size(z,2)+1);
gp = gp_set(gp, 'lik', lik, 'deriv', size(x,2)+1);
gp.monotonic = true;
gp.fh.setUpDataForMonotonic = @setUpDataForMonotonic;

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
  switch init
    case 'sample'
      rpii=randperm(size(x,1));
      gp.xv=x(rpii(1:nv),:);
    case 'kmeans'
      S=warning('off','stats:kmeans:EmptyCluster');
      [tmp,xv]=kmeans(x, nv, 'Start','uniform', ...
                      'EmptyAction', 'singleton');
      warning(S);
  end
end
xv=gp.xv;
% xt = [x zeros(size(x,1),1)];
% yt = y;
% for i1=1:nvd
%     xt = [xt ; xv abs(gp.nvd(i1))*ones(size(xv,1),1)];
%     yt = [yt ; gp.nvd(i1)./abs(gp.nvd(i1)).*ones(size(xv,1),1)];
% end
% zt = [z ones(size(x,1),1) ; z 2*ones(nvd*size(xv,1),1)];
if isempty(opt) || ~isfield(opt, 'TolX')
  % No options structure given or not a proper options structure
  opt=optimset('TolX',1e-4,'TolFun',1e-4,'Display','iter');
end
if ~isfield(gp,'latent_method') || ~strcmpi(gp.latent_method,'EP')
    if display
        fprintf('Switching the latent method to EP.\n');
    end
    gp=gp_set(gp,'latent_method','EP', 'jitterSigma2', 1e-6);
end
gp.latent_opt.init_prev='off';
gp.latent_opt.maxiter=100;
gpep_e('clearcache',gp);
if isequal(optimize, 'on')
  % Optimize the parameters
  %gradcheck(gp_pak(gp), @gp_e, @gp_g, gp, xt, yt, 'z', zt);
  %gp=gp_optim(gp,xt,yt,'opt',opt, 'z', zt, 'optimf', optimf);
  gp=gp_optim(gp,x,y,'opt',opt, 'z', z, 'optimf', optimf);
end



if force
    % Predict gradients at the training points 
    n=size(x,1);
    nblocks = 10;
    [tmp,itst]=cvit(size(x,1),nblocks);
    Ef=zeros(size(x,1),nvd);
    for i=1:nblocks
      if length(itst{i})>0
        % Predict in blocks to save memory
        tmp=ones(length(itst{i}),1)*abs(gp.nvd);
        xtest = [repmat(x(itst{i},:),nvd,1) tmp(:)];
        %Ef(itst{i},:)=gpep_predgrad(gp,xt,yt,xtest,'z',z);
        %Ef(itst{i},:)=reshape(gp_pred(gp,xt,yt,xtest,'z',zt),length(itst{i}),nvd);
        Ef(itst{i},:)=reshape(gp_pred(gp,x,y,xtest,'z',z),length(itst{i}),nvd);
      end
    end
    % Check if monotonicity is satisfied
    yv=round(gp.nvd./abs(gp.nvd));
    while any(any(bsxfun(@times,Ef, yv)<-nu))
      % Monotonicity not satisfied, add 2 "most wrong" predictions, for each 
      % dimension, from the observation set to the virtual observations.
      if display
          fprintf('Latent function not monotonic, adding virtual observations.\n');
      end
      for j=1:nvd
        [~,ind(:,j)]=sort(Ef(:,j).*yv(j),'ascend');
      end
      ind=ind(1:2,:);
      inds=unique(ind(:));
      clear ind;
      if display
        fprintf('Added %d virtual observations.\n', length(inds));
      end
      xv=[xv;x(inds,:)];
      gp.xv=xv;
%       xt = [x zeros(size(x,1),1)];
%       yt = y;
%       for i1=1:nvd
%           xt = [xt ; xv abs(gp.nvd(i1))*ones(size(xv,1),1)];
%           yt = [yt ; gp.nvd(i1)./abs(gp.nvd(i1)).*ones(size(xv,1),1)];
%       end
%       zt = [z ones(size(x,1),1) ; z 2*ones(nvd*size(xv,1),1)];
      gpep_e('clearcache',gp);
      if isequal(optimize, 'on')
        %gp=gp_optim(gp,xt,yt,'opt',opt,'z',zt, 'optimf', optimf);
        gp=gp_optim(gp,x,y,'opt',opt,'z',z, 'optimf', optimf);
      end
      % Predict gradients at the training points
      for i=1:nblocks
          if length(itst{i})>0
              % Predict in blocks to save memory
              tmp=ones(length(itst{i}),1)*abs(gp.nvd);
              xtest = [repmat(x(itst{i},:),nvd,1) tmp(:)];
              %Ef(itst{i},:)=gpep_predgrad(gp,xt,yt,xtest,'z',z);
              %Ef(itst{i},:)=reshape(gp_pred(gp,xt,yt,xtest,'z',zt),length(itst{i}),nvd);
              Ef(itst{i},:)=reshape(gp_pred(gp,x,y,xtest,'z',z),length(itst{i}),nvd);
          end
      end
    end
end
% gp.monotonic = true;

end

function [gp,x,y,z,xtt,ztt] = setUpDataForMonotonic(gp,x,y,z,xtt,ztt)

xv=gp.xv;
nvd = length(gp.nvd);
xt = [x zeros(size(x,1),1)];
yt = y;
for i1=1:nvd
    xt = [xt ; xv abs(gp.nvd(i1))*ones(size(xv,1),1)];
    yt = [yt ; gp.nvd(i1)./abs(gp.nvd(i1)).*ones(size(xv,1),1)];
end
if ~isempty(z)
    zt = [z ones(size(x,1),1) ; zeros(size(xv,1),size(z,2)) 2*ones(nvd*size(xv,1),1)];
else
    zt = [ones(size(x,1),1) ; 2*ones(nvd*size(xv,1),1)];
end
x=xt;
y=yt;
z=zt;
gp.monotonic = false;
if nargin>4
    if size(xtt,2)~=size(x,2)
        xtt = [xtt zeros(size(xtt,1),1)];
    end
else
    xtt=[];
end
if nargin>5
    if size(ztt,2)<gp.lik.classVariables
        ztt = [ztt ones(size(ztt,1),1)];
    end
else
    ztt=[];
end


end

