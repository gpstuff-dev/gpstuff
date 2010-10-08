function [Ef, Varf, Ey, Vary, Py] = ep_loopred(gp, x, y, varargin)
%EP_LOOPRED  Leave-one-out predictions with Gaussian Process EP approximation
%
%     Description

%	[EF, VARF, EY, VARY, PYT] = EP_LOOPRED(GP, X, Y, OPTIONS) takes
%        a GP data structure GP together with a matrix XT of input
%        vectors, matrix X of training inputs and vector Y of
%        training targets, and evaluates the leave-one-out
%        predictive distribution at inputs X. Returns a posterior
%        mean EF and variance VARF of latent variables and the
%        posterior predictive mean EY and variance VARY of
%        observations at input locations X.
%
%        Leave-one-out is approximated by leaving-out site-term and
%        using cavity distribution as leave-one-out posterior for
%        the ith latent value. Since the ith likelihood has influenced
%        other site terms through the prior, this estimate can be
%        over-optimistic.
%
%     OPTIONS is optional parameter-value pair
%       'z'      is optional observed quantity in triplet (x_i,y_i,z_i)
%                Some likelihoods may use this. For example, in case of 
%                Poisson likelihood we have z_i=E_i, that is, expected value 
%                for ith case. 
%
%	See also
%	GPEP_E, GPEP_G, GP_PRED, DEMO_SPATIAL, DEMO_CLASSIFIC
  
% Copyright (c) 2010  Aki Vehtari

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

  ip=inputParser;
  ip.FunctionName = 'EP_LOOPRED';
  ip.addRequired('gp', @isstruct);
  ip.addRequired('x', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
  ip.addRequired('y', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
  ip.addParamValue('z', [], @(x) isreal(x) && all(isfinite(x(:))))
  ip.parse(gp, x, y, varargin{:});
  z=ip.Results.z;

  [e, edata, eprior, tautilde, nutilde, L, La, b, muvec_i, sigm2vec_i] = gpep_e(gp_pak(gp), gp, x, y, 'z', z);

  Ef=muvec_i;
  Varf=sigm2vec_i;
  n=length(y);
  if isempty(z)
    if nargout > 4
      for cvi=1:n
        [Ey(cvi,1), Vary(cvi,1), Py(cvi,1)] = feval(gp.likelih.fh_predy, gp.likelih, muvec_i(cvi), sigm2vec_i(cvi), y(cvi));
      end
    elseif nargout > 2
      for cvi=1:n
        [Ey(cvi,1), Vary(cvi,1)] = feval(gp.likelih.fh_predy, gp.likelih, muvec_i(cvi), sigm2vec_i(cvi));
      end
    end
  else
    if nargout > 4
      for cvi=1:n
        [Ey(cvi,1), Vary(cvi,1), Py(cvi,1)] = feval(gp.likelih.fh_predy, gp.likelih, muvec_i(cvi), sigm2vec_i(cvi), y(cvi), z(cvi));
      end
    elseif nargout > 2
      for cvi=1:n
        [Ey(cvi,1), Vary(cvi,1)] = feval(gp.likelih.fh_predy, gp.likelih, muvec_i(cvi), sigm2vec_i(cvi), [], z(cvi));
      end
    end
  end
end
