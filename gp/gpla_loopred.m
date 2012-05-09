function [Eft, Varft, lpyt, Eyt, Varyt] = gpla_loopred(gp, x, y, varargin)
%GPLA_LOOPRED  Leave-one-out predictions with Laplace approximation
%
%  Description
%    [EFT, VARFT, LPYT, EYT, VARYT] = GPLA_LOOPRED(GP, X, Y, OPTIONS)
%    takes a Gaussian process structure GP together with a matrix X
%    of training inputs and vector Y of training targets, and
%    evaluates the leave-one-out predictive distribution at inputs
%    X and returns means EFT and variances VARFT of latent
%    variables, the logarithm of the predictive densities PYT, and
%    the predictive means EYT and variances VARYT of observations
%    at input locations X.
%
%    OPTIONS is optional parameter-value pair
%      z      - optional observed quantity in triplet (x_i,y_i,z_i)
%               Some likelihoods may use this. For example, in case of 
%               Poisson likelihood we have z_i=E_i, that is, expected value 
%               for ith case. 
%
%    Laplace leave-one-out is approximated in linear response style
%    by expressing the solutions for LOO problem in terms of
%    solution for the full problem. The computationally cheap
%    solution can be obtained by making the assumption that the
%    difference between these two solution is small such that their
%    difference may be treated as an Taylor expansion truncated at
%    first order (Winther et al 2012, in progress).
%
%  See also
%    GP_LOOPRED, GP_PRED
  
% Copyright (c) 2011-2012  Aki Vehtari, Ville Tolvanen

% This software is distributed under the GNU General Public 
% License (version 3 or later); please refer to the file 
% License.txt, included with the software, for details.

  ip=inputParser;
  ip.FunctionName = 'GPLA_LOOPRED';
  ip.addRequired('gp', @(x) isstruct(x));
  ip.addRequired('x', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
  ip.addRequired('y', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
  ip.addParamValue('z', [], @(x) isreal(x) && all(isfinite(x(:))))
  ip.addParamValue('method', 'lrs', @(x) ismember(x, {'lrs' 'cavity' 'inla'}))
  ip.parse(gp, x, y, varargin{:});
  z=ip.Results.z;
  method = ip.Results.method;
  [tn,nin] = size(x);
  
  % latent posterior
  [f, sigm2ii, lp] = gpla_pred(gp, x, y, 'z', z);
  
  switch method

    case 'inla'
      % Leonhard Held and Birgit Schrödle and Håvard Rue (2010)
      % Posterior and Cross-validatory Predictive Checks: A
      % Comparison of MCMC and INLA. In (eds) Thomas Kneib and
      % Gerhard Tutz, Statistical Modelling and Regression
      % Structures, pp. 91-110. Springer.
      Eft = zeros(tn,1);
      Varft = zeros(tn,1);
      lpyt = zeros(tn,1);
      minf = f-6.*sqrt(sigm2ii);
      maxf = f+6.*sqrt(sigm2ii);
      for i=1:tn
        if isempty(z)
          z1 = [];
        else
          z1 = z(i);
        end
        [m0, m1, m2] = quad_moments(@(x) norm_pdf(x, f(i), sqrt(sigm2ii(i)))./llvec(gp.lik,y(i),x,z1), minf(i), maxf(i));
        Eft(i) = m1;
        Varft(i) = m2-Eft(i)^2;
        lpyt(i) = -log(m0);
      end

      if nargout>3
        [~,Eyt,Varyt] = gp.lik.fh.predy(gp.lik, Eft, Varft, y, z);
      end
      if sum((abs(lpyt)./abs(lp) > 5) == 1) > 0.1*tn;
        warning('Very bad predictive densities, gpla_loopred might not be reliable, check results!');
      end
  
    case 'cavity'
      % using EP equations

      % "site parameters"
      W        = -gp.lik.fh.llg2(gp.lik, y, f, 'latent', z);
      deriv    = gp.lik.fh.llg(gp.lik, y, f, 'latent', z);
      sigm2_t  = 1./W;
      mu_t     = f + sigm2_t.*deriv;
      
      % "cavity parameters"
      sigma2_i = 1./(1./sigm2ii-1./sigm2_t);
      myy_i    = sigma2_i.*(f./sigm2ii-mu_t./sigm2_t);
      % check if cavity varianes are negative
      ii=find(sigma2_i<0);
      if ~isempty(ii)
        warning('gpla_loopred: some cavity variances are negative');
        sigma2_i(ii) = sigm2ii(ii);
        myy_i(ii) = f(ii);
      end
      
      % leave-one-out predictions
      Eft=myy_i;
      Varft=sigma2_i;

      if nargout==3
        lpyt = gp.lik.fh.predy(gp.lik, Eft, Varft, y, z);
      elseif nargout>3
        [lpyt,Eyt,Varyt] = gp.lik.fh.predy(gp.lik, Eft, Varft, y, z);
      end
      
    case 'lrs'
      % Manfred Opper and Ole Winther (2000). Gaussian Processes for
      % Classification: Mean-Field Algorithms. In Neural
      % Computation, 12(11):2655-2684.
      %
      % Ole Winther et al (2012). Work in progress.

      K = gp_trcov(gp,x);
      deriv = gp.lik.fh.llg(gp.lik, y, f, 'latent', z);
      La = 1./-gp.lik.fh.llg2(gp.lik, y, f, 'latent', z);
      % really large values don't contribute, but make variance
      % computation unstable. 2e15 approx 1/(2*eps)
      La = min(La,2e15);
      Varft=1./diag(inv(K+diag(La)))-La;
      % check if cavity varianes are negative
      ii=find(Varft<0);
      if ~isempty(ii)
        warning('gpla_loopred: some LOO latent variances are negative');
        Varft(ii) = 0;
      end
      Eft=f-Varft.*deriv;

      if nargout==3
        lpyt = gp.lik.fh.predy(gp.lik, Eft, Varft, y, z);
      elseif nargout>3
        [lpyt,Eyt,Varyt] = gp.lik.fh.predy(gp.lik, Eft, Varft, y, z);
      end
  
  end

end

function expll = llvec(gplik, y, f, z)
  for i=1:size(f,2)
    expll(i) = exp(gplik.fh.ll(gplik, y, f(i), z));
  end
end
