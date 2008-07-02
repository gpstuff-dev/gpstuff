function w = gp_pak(gp, param)
%GP_PAK	 Combine GP hyper-parameters into one vector.
%
%	Description
%	W = GP_PAK(GP, PARAM) takes a Gaussian Process data structure GP and
%	string PARAM defining, which parameters are packed and combines the 
%       parameters into a single row vector W.
%
%       The possiple values for PARAM are
%       'hyper'          = pack only hyperparameters
%                          W = log([hyper-params of gp.cf, hyper-params of gp.noise])
%       'indicing'       = pack only iducing inputs
%                          W = gp.X_u(:)
%       'likelih'        = pack only parameters of likelihood
%       'hyper+inducing' = pack hyperparameters and induging inputs
%                          W = [log(hyper-params of gp.cf, hyper-params of gp.noise), gp.X_u(:)];
%       'hyper+likelih'  = pack hyperparameters and parameters of likelihood
%                          W = [log(hyper-params of gp.cf, hyper-params of gp.noise), parameters of likelihood];
%       'all'            = pack all parameters in one vector
%
%	See also
%	GP_UNPAK
%

% Copyright (c) 2007-2008 Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

switch param
  case 'hyper'
    w = pak_hyper(gp);
  case 'inducing'    
    w = pak_inducing(gp);
  case 'likelih'
    w = feval(gp.likelih.fh_pak, gp.likelih);
  case 'hyper+inducing'
    w = pak_hyper(gp);
    w = [w pak_inducing(gp)];
  case 'hyper+likelih'
    w = pak_hyper(gp);
    w = [w feval(gp.likelih.fh_pak, gp.likelih)];
  case 'all'
    w = pak_hyper(gp);
    w = [w pak_inducing(gp)];
    w = [w feval(gp.likelih.fh_pak, gp.likelih)];
  otherwise
    error('Unknown parameter to take the gradient with respect to! \n')
end


% Pak the hyperparameters
function w = pak_hyper(gp)
w=[];
ncf = length(gp.cf);

for i=1:ncf
    gpcf = gp.cf{i};
    w = feval(gpcf.fh_pak, gpcf, w);
end

if isfield(gp, 'noise')
    nn = length(gp.noise);
    for i=1:nn
        noise = gp.noise{i};
        w = feval(noise.fh_pak, noise, w);
    end
end
w = log(w);

% Pak the inducing inputs
function w = pak_inducing(gp)
w = gp.X_u(:)';
