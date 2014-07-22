function [Eft, Varft, lpyt, Eyt, Varyt] = gpsvi_predgrad(gp,x,y,varargin)
%GPSVI_PREDGRAD  Make predictions with SVI GP
%
%  Description
%    [EFT, VARFT] = GPSVI_PREDGRAD(GP, X, Y, XT, OPTIONS)
%    takes a GP structure together with matrix X of training inputs and
%    vector Y of training targets, and evaluates the predictive
%    distribution at test inputs XT. Returns a posterior mean EFT and
%    variance VARFT of latent variables. Each row of X corresponds to one
%    input vector and each row of Y corresponds to one output vector.
%
%    [EFT, VARFT, LPYT] = GPSVI_PREDGRAD(GP, X, Y, XT, 'yt', YT, OPTIONS)
%    returns also logarithm of the predictive density LPYT of the
%    observations YT at test input locations XT. This can be used
%    for example in the cross-validation. Here Y has to be a vector.
% 
%    [EFT, VARFT, LPYT, EYT, VARYT] = GPSVI_PREDGRAD(GP, X, Y, XT, OPTIONS)
%    returns also the posterior predictive mean EYT and variance VARYT.
%
%    [EF, VARF, LPY, EY, VARY] = GPSVI_PREDGRAD(GP, X, Y, OPTIONS)
%    evaluates the predictive distribution at training inputs X
%    and logarithm of the predictive density LPY of the training
%    observations Y.
%
%    OPTIONS is optional parameter-value pair
%      predcf - an index vector telling which covariance functions are 
%               used for prediction. Default is all (1:gpcfn). 
%               See additional information below.
%      yt     - optional observed yt in test points (see below)
%      z      - optional observed quantity in triplet (x_i,y_i,z_i)
%               Some likelihoods may use this. For example, in case of 
%               Poisson likelihood we have z_i=E_i, that is, expected value 
%               for ith case. 
%      zt     - optional observed quantity in triplet (xt_i,yt_i,zt_i)
%               Some likelihoods may use this. For example, in case of 
%               Poisson likelihood we have z_i=E_i, that is, the expected 
%               value for the ith case. 
%
%  See also
%    GPSVI_PRED, SVIGP, DEMO_SVI*
%

% Copyright (c) 2014 Ville Tolvanen

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

ip=inputParser;
ip.FunctionName = 'GP_PRED';
ip.addRequired('gp',@(x) isstruct(x) || iscell(x));
ip.addRequired('x', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addRequired('y', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addOptional('xt', [], @(x) isempty(x) || (isreal(x) && all(isfinite(x(:)))))
ip.addParamValue('yt', [], @(x) isreal(x) && all(isfinite(x(:))))
ip.addParamValue('zt', [], @(x) isreal(x) && all(isfinite(x(:))))
ip.addParamValue('z', [], @(x) isreal(x) && all(isfinite(x(:))))
ip.addParamValue('predcf', [], @(x) isempty(x) || ...
                 isvector(x) && isreal(x) && all(isfinite(x)&x>=0))
if numel(varargin)==0 || isnumeric(varargin{1})
  % inputParser should handle this, but it doesn't
  ip.parse(gp, x, y, varargin{:});
else
  ip.parse(gp, x, y, [], varargin{:});
end
xt=ip.Results.xt;
yt=ip.Results.yt;
zt=ip.Results.zt;
z=ip.Results.z;
predcf=ip.Results.predcf;
if isempty(xt)
  xt=x;
  if isempty(yt)
    yt=y;
  end
  if isempty(zt)
    zt=z;
  end
end

tn = size(x,1);
if nargout > 2 && isempty(yt)
  lpyt=[];
end

%     [tmp,tmp,tmp,param]=gpsvi_e(gp_pak(gp),gp,x,y);

% Check if the variational parameters has been set
if ~isfield(gp, 'm') || ~isfield(gp, 'S')
  error('Variational parameters has not been set. Call SVIGP first.')
end

u = gp.X_u;
m=gp.m;
S=gp.S;
if size(u,2) ~= size(x,2)
  % Turn the inducing vector on right direction
  u=u';
end
% Calculate some help matrices
K_uu = gp_trcov(gp, u);     % u x u, noiseles covariance K_uu
K_nu = gp_dcov2(gp, u, [], xt, xt);
K_nu = (K_nu(:,size(xt,1)+1:end))';
%     K_nu = gp_cov(gp,xt,u);       % n x u
[Luu, notpositivedefinite] = chol(K_uu,'lower');
if notpositivedefinite
  Eft=NaN; Varft=NaN; lpyt=NaN; Eyt=NaN; Varyt=NaN;
  return
end

Eft = K_nu*(Luu'\(Luu\m));

if nargout > 1
  [Knn_v, Cnn_v] = gp_trvar(gp,xt,predcf);
  B2=Luu\(K_nu');
  B3=K_uu\(K_nu');

  Varft = Knn_v - sum(B2.*B2)' + sum(B3.*(S*B3))';
end
s2=gp.lik.sigma2;


if nargout > 2
  if isequal(gp.lik.type, 'Gaussian')
    Eyt = Eft;
    Varyt = Varft + Cnn_v - Knn_v;
    if ~isempty(yt)
      lpyt = norm_lpdf(yt, Eyt, sqrt(Varyt));
    end
  else
    if nargout>3
      [lpyt, Eyt, Varyt] = gp.lik.fh.predy(gp.lik, Eft, Varft+s2, yt, zt);
    else
      lpyt = gp.lik.fh.predy(gp.lik, Eft, Varft+s2, yt, zt);
    end
  end
end
end

function [C, Cinv] = gp_dcov2(gp, x1, xv1, x2, xv2, predcf)
% Replacement for GP_DCOV where virtual inputs are given as arguments.

% Split the training data for normal latent input and gradient inputs
x12=x1;
%x11=gp.xv;
x11=xv1;
x3=xv2;

% Derivative observations
[n,m]=size(x1);
[n4,m4]=size(x2);
ncf=length(gp.cf);
if isfield(gp, 'nvd')
  % Only specific dimensions
  ii1=abs(gp.nvd);
else
  % All dimensions
  ii1=1:m;
end
for i1=1:ncf
  gpcf = gp.cf{i1};    
  if m==1
    Gset2 = gpcf.fh.ginput4(gpcf, x3, x12);
    Kff = gpcf.fh.cov(gpcf, x12, x2);
    if ~isempty(x11)
      Gset1 = gpcf.fh.ginput4(gpcf, x11, x2);
      Kdd = gpcf.fh.ginput2(gpcf, x11, x3);
      Kdf=Gset1{1};
    else
      Kdd={[]};
      Kdf=[];
    end

    Kfd=Gset2{1};
    C = [Kff Kfd'; Kdf Kdd{1}];
    
    % Input dimension is >1
  else
    [n,m]=size(x11);
    [n2,m2]=size(x3);
    
    Kff = gpcf.fh.cov(gpcf, x12, x2);
    Gset2 = gpcf.fh.ginput4(gpcf, x3, x12);
    
    %Gather matrices from Gset (d k(x1,x2) /d x1)
    Kfd22=cat(2,Gset2{ii1});
    Kdf22=cat(1,Gset2{ii1})';
     
    if ~isempty(x11)
      Gset1 = gpcf.fh.ginput4(gpcf, x11,x2);
      Kfd=cat(2,Gset1{ii1});
      Kdf=cat(1,Gset1{ii1});
      % both x derivatives, same dimension (to diagonal blocks)
      D = gpcf.fh.ginput2(gpcf, x11, x3);
      % both x derivatives, different dimension (non-diagonal blocks)
      Kdf2 = gpcf.fh.ginput3(gpcf, x11 ,x3);
      
      % Now build up Kdd m*n x m*n2 matrix, which contains all the
      % both partial derivative" -matrices
      
      % Add the diagonal matrices
      Kdd=blkdiag(D{1:m});
      % Add the non-diagonal matrices to Kdd
      ii3=0;
      for j=0:m-2
        for i=1+j:m-1
          ii3=ii3+1;
          Kdd(i*n+1:(i+1)*n,j*n2+1:j*n2+n2) = Kdf2{ii3};
          Kdd(j*n+1:j*n+n,i*n2+1:(i+1)*n2) = Kdf2{ii3};
        end
      end
      if isfield(gp, 'nvd')
        % Collect the correct gradient dimensions, i.e. select the blocks
        % that correspond to the input dimensions for which we want the
        % gradients to be monotonic
        Kddtmp=[];
        for ii2=1:length(ii1)
          for ii3=ii2:length(ii1)
            Kddtmp((ii2-1)*n+1:ii2*n, (ii3-1)*n2+1:ii3*n2) = ...
              Kdd((ii1(ii2)-1)*n+1:ii1(ii2)*n,(ii1(ii3)-1)*n2+1:ii1(ii3)*n2);
            if ii2~=ii3
              Kddtmp((ii3-1)*n+1:ii3*n, (ii2-1)*n2+1:ii2*n2) = ...
                Kdd((ii1(ii3)-1)*n+1:ii1(ii3)*n,(ii1(ii2)-1)*n2+1:ii1(ii2)*n2);
            end
          end
        end
        Kdd=Kddtmp;
      end
    else
      Kdf=[];
      Kdd=[];
    end
    
    % Gather all the matrices into one final matrix K which is the
    % training covariance matrix
    C = [Kff Kdf22; Kdf Kdd];
    %   C = [Kff; Kdf];
  end
  if i1==1
    CC=C;
  else
    CC=CC+C;
  end
end
C=CC;
end
