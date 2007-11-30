function w = gp_pak(gp, param)
%GP_PAK	 Combine GP hyper-parameters into one vector.
%
%	Description
%	W = GP_PAK(GP) takes a Gaussian Process data structure GP and
%	combines the hyper-parameters into a single row vector W.
%
%	The ordering of the parameters in HP is defined by
%	  hp = [hyper-params of gp.cf{1}, hyper-params of gp.cf{2}, ...];
%
%	See also
%	GP_UNPAK
%

% Copyright (c) 2006      Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

switch param
  case 'hyper'
    w = pak_hyper(gp);
  case 'inducing'    
    w = pak_inducing(gp);
  case 'all'
    w = pak_hyper(gp);
    w = [w pak_inducing(gp)];
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
