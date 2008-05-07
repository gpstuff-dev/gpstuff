function gp = gp_unpak(gp, w, param)
%GP_UNPAK  Separate GP hyper-parameter vector into components. 
%
%	Description
%	GP = GP_UNPAK(GP, W) takes an Gaussian Process data structure GP
%	and  a hyper-parameter vector W, and returns a Gaussian Process data
%	structure  identical to the input model, except that the covariance
%	hyper-parameters has been set to the of W.
%
%	See also
%	GP_PAK
%

% Copyright (c) 2006      Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

switch param
  case 'hyper'
    gp = unpak_hyper(w, gp);
  case 'inducing'    
    gp = unpak_inducing(w, gp);
  case 'likelih'    
    lik = feval(gp.likelih.fh_unpak, w, gp.likelih);
    gp.likelih = lik;
  case 'hyper+inducing'
    w1 = w(1:end-length(gp.X_u(:)));
    gp = unpak_hyper(w1, gp);
    w2 = w(length(w1)+1:end);
    gp = unpak_inducing(w2, gp);
  case 'hyper+likelih'
    w1 = w(1:length(gp_pak(gp,'hyper')));
    gp = unpak_hyper(w1, gp);
    w2 = w(length(w1)+1:end);
    lik = feval(gp.likelih.fh_unpak, w2, gp.likelih);
    gp.likelih = lik;
  case 'all'
    w1 = w(1:length(gp_pak(gp,'hyper')));
    gp = unpak_hyper(w1, gp);
    w2 = w(length(w1)+1:length(w1)+length(gp.X_u(:)));
    gp = unpak_inducing(w2, gp);
    w3 = w(length(w1)+length(w2)+1:end);
    lik = feval(gp.likelih.fh_unpak, w3, gp);
    gp.likelih = lik;
  otherwise
    error('Unknown parameter to take the gradient with respect to! \n')
end


% Function for unpacking the hyperparameters
function gp = unpak_hyper(w, gp)

w(w<-10)=-10;
w(w>10)=10;
w=exp(w);
    
w1 = w;
ncf = length(gp.cf);

for i=1:ncf
    gpcf = gp.cf{i};
    [gpcf, w1] = feval(gpcf.fh_unpak, gpcf, w1);
    gp.cf{i} = gpcf;
end

if isfield(gp, 'noise')
    nn = length(gp.noise);
    for i=1:nn
        noise = gp.noise{i};
        [noise, w1] = feval(noise.fh_unpak, noise, w1);
        gp.noise{i} = noise;
    end
end



% Function for unpacking the inducing inputs
function gp = unpak_inducing(w, gp)
gp.X_u = reshape(w, size(gp.X_u));

