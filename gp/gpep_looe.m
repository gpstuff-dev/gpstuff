function eloo = gpep_looe(w, gp, x, y, varargin)
% GPEP_LOOE Evaluate the mean negative log leave-one-out predictive 
%           density, assuming Gaussian observation model.
%
%   Description
%     LOOE = GPEP_LOOE(W, GP, X, Y, PARAM) takes a hyper-parameter vector
%     W, Gaussian process structure GP, a matrix X of input vectors and
%     a matrix Y of targets, and evaluates the mean negative log 
%     leave-one-out predictive density
%       LOOE  = - 1/n sum log p(Y_i | X, Y_{\i}, th)
%     where th represents the hyperparameters (lengthScale, magnSigma2...), 
%     X is inputs and Y is observations. 
%
%     EP-Leave-one-out is approximated by leaving-out site-term and
%     using cavity distribution as leave-one-out posterior for
%     the ith latent value.
%
%	See also
%	GP_LOOE, EP_LOORPED, GPEP_E
%

% Copyright (c) 2010 Aki Vehtari

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.

gp=gp_unpak(gp, w);
ncf = length(gp.cf);
n=length(x);

[Ef, Varf, Ey, Vary, Py] = ep_loopred(gp, x, y, varargin{:});
eloo=-mean(log(Py));

end
