function gp = gp_unpak(gp, w)
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
