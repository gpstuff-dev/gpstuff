function [e, g] = gp_looeg(w, gp, x, y, varargin)
%GP_LOOEG  Evaluate the mean negative log leave-one-out predictive density
%
%  Description
%    [E, G] = GP_LOOEG(W, GP, X, Y, OPTIONS) takes a Gaussian
%    process structure GP together with a matrix X of input vectors
%    and a matrix Y of targets, and evaluates the energy function E
%    and its gradient G (only with Gaussian likelihood). Each row
%    of X corresponds to one input vector and each row of Y
%    corresponds to one target vector.
%
%    The energy is the mean negative log
%    leave-one-out predictive density
%       LOOE  = - 1/n sum log p(Y_i | X, Y_{\i}, th)
%    where th represents the parameters (lengthScale,
%    magnSigma2...), X is inputs and Y is observations.
%
%    For non-Gaussian models EP leave-one-out is used for energy
%    (see GPEP_LOOE), but no gradients are yet implemented, and
%    thus gradient-free optimization function has to be used.
%
%    OPTIONS is optional parameter-value pair
%      z - optional observed quantity in triplet (x_i,y_i,z_i)
%          Some likelihoods may use this. For example, in case of
%          Poisson likelihood we have z_i=E_i, that is, expected
%          value for ith case.
%
%  See also
%    GP_LOOE, GP_LOOG, GPEP_LOOE
%

% Copyright (c) 2010 Aki Vehtari
  
% Single function for some optimization routines, no need for mydeal...
if isfield(gp.lik.fh,'trcov')
  % Gaussian likelihood
  e=gp_looe(w, gp, x, y);
  if nargout>1
    g=gp_loog(w, gp, x, y);
  end
else
  % non-Gaussian likelihood
  switch gp.latent_method
    case 'Laplace'
      e=gpla_looe(w, gp, x, y, varargin{:});
    case 'EP'
      e=gpep_looe(w, gp, x, y, varargin{:});
  end
  if nargout>1
    error('Laplace and EP leave-one-out do not have gradients yet, use gradient-free optimization.')
  end
end
