function [e, g] = gp_eg(w, gp, x, y, varargin)
%GP_EG  Evaluate the energy function (un-normalized negative marginal
%       log posterior) ans its gradient in case of Gaussian
%       observation model
%
%     Description
%	[E, G] = GP_EG(W, GP, X, Y, OPTIONS) takes a Gaussian process
%        structure GP together with a matrix X of input vectors and
%        a matrix Y of targets, and evaluates the energy function E
%        and its gradient G. Each row of X corresponds to one input
%        vector and each row of Y corresponds to one target vector.
%
%       The energy is minus log posterior cost function:
%            E = EDATA + EPRIOR 
%              = - log p(Y|X, th) - log p(th),
%       where th represents the hyperparameters (lengthScale, magnSigma2...),
%       X is inputs and Y is observations (regression) or latent values
%       (non-Gaussian likelihood).
%
%     OPTIONS is optional parameter-value pair
%       No applicable options
%
%	See also
%	GP_E, GP_G
%

% Copyright (c) 2010 Aki Vehtari
  
% Single function for some optimization routines, no need for mydeal...
e=gp_e(w, gp, x, y, varargin{:});
g=gp_g(w, gp, x, y, varargin{:});
