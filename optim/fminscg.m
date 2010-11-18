function [x, fval, exitflag, output, grad] = fminscg(fun, x, opt)
%FMINSCG  Scaled conjugate gradient optimization
%
%  Description
%    X = FMINSCG(FUN, X0) starts at X0 and attempts to find a local
%    minimizer X of the function FUN. FUN accepts input X and
%    returns a scalar function value F and its scalar or vector
%    gradient G evaluated at X. X0 can be a scalar or vector
%
%    X = FMINSCG(FUN, X0, OPTIONS) minimizes with the default
%    optimization parameters replaced by values in the structure
%    OPTIONS, an argument created with the OPTIMSET function. See
%    OPTIMSET for details. Used options are Display, TolX, TolFun,
%    DerivativeCheck and MaxIter.
%
%    [X,FVAL] = FMINSCG(FUN,X0,...) returns the value of the objective 
%    function FUN at the solution X.
%
%    [X,FVAL,EXITFLAG] = FMINSCG(FUN,X0,...) returns an EXITFLAG that 
%    describes the exit condition of FMINSCG. Possible values of EXITFLAG 
%    and the corresponding exit conditions are listed below. See the
%    documentation for a complete description.
%      1  Magnitude of gradient small enough. 
%      2  Change in X too small.
%      3  Change in objective function too small.
%      0  Too many function evaluations or iterations.
%
%    [X,FVAL,EXITFLAG,OUTPUT] = FMINSCG(FUN,X0,...) returns a
%    structure OUTPUT with the number of iterations taken in
%    OUTPUT.iterations, the number of function evaluations in
%    OUTPUT.funcCount, the algorithm used in OUTPUT.algorithm,
%    function values for each iteration in OUTPUT.f and X for each
%    iteration in OUTPUT.x.
%
%    [X,FVAL,EXITFLAG,OUTPUT,GRAD] = FMINSCG(FUN,X0,...) returns the value 
%    of the gradient of FUN at the solution X.
%
%  Options
%    Display
%      - 'off' displays no output
%      - 'iter' displays output at each iteration, and gives the
%               default exit message.
%      - 'notify' displays output only if the function does not
%                 converge, and gives the default exit message.
%      - 'final' (default) displays just the final output, and
%                 gives the default exit message.
%    TolFun
%      Termination tolerance on the function value, a positive
%      scalar. The default is 1e-6.
%    TolX
%      Termination tolerance on x, a positive scalar. The default
%      value is 1e-6.
%    DerivativeCheck
%      Compare user-supplied derivatives (gradient of objective) to
%      finite-differencing derivatives. The choices are 'on' or the
%      default 'off'.
%    MaxIter
%      Maximum number of iterations allowed, a positive integer. 
%      The default value is 400.
%
%  See also OPTIMSET

% Copyright (c) 1996,1997 Christopher M Bishop, Ian T Nabney
% Copyright (c) 2005,2010 Aki Vehtari

% Set empty omptions to default values
defaultopt = struct( ...
    'DerivativeCheck','off', ...   
    'Display','final', ...
    'MaxIter',400, ...
    'TolFun',1e-6, ...
    'TolX',1e-6); 

% If just 'defaults' passed in, return the default options in X
if nargin==1 && nargout <= 1 && isequal(fun,'defaults')
   x = defaultopt;
   return
end

if nargin < 3, opt=[]; end 

switch optimget(opt,'Display',defaultopt,'fast');
  case 'off'
    display=0;
  case 'notify'
    display=1;
  case 'final'
    display=2;
  case 'iter'
    display=3;
  otherwise
    display=2;
end
maxiter = optimget(opt,'MaxIter',defaultopt,'fast');
tolfun = optimget(opt,'TolFun',defaultopt,'fast');
tolx = optimget(opt,'TolX',defaultopt,'fast');

nparams = length(x);

%  Check gradients
if isequal(optimget(opt,'DerivativeCheck',defaultopt,'fast'),'on');
  derivativecheck(x, fun);
end

sigma0 = 1.0e-4;
iter = 0;
[fold,gradold] = feval(fun, x); % Initial function value and gradient
gradnew = gradold;
d = - gradnew;                  % Initial search direction.
success = 1;                    % Force calculation of directional derivs.
nsuccess = 0;                   % nsuccess counts number of successes.
lambda = 1.0;                   % Initial scale parameter.
lambdamin = 1.0e-15; 
lambdamax = 1.0e100;
j = 1;                          % j counts number of iterations.
if nargout >= 4
  output.f(j, :) = fold;
  output.x(j, :) = x;
  output.algorithm='fminscg';
  output.funcCount=1;
end

% Main optimization loop.
while (j <= maxiter)

  % Calculate first and second directional derivatives.
  if (success == 1)
    mu = d*gradnew';
    if (mu >= 0)
      d = - gradnew;
      mu = d*gradnew';
    end
    kappa = d*d';
    if kappa < eps
      if (display >= 2)
        disp('Gradient smaller than eps');
      end
      exitflag=1;
      return
    end
    sigma = sigma0/sqrt(kappa);
    xplus = x + sigma*d;
    [~,gplus] = feval(fun, xplus);
    if nargin>4
      output.funcCount=output.funcCount+1;
    end
    gamma = (d*(gplus' - gradnew'))/sigma;
  end

  % Increase effective curvature and evaluate step size alpha.
  delta = gamma + lambda*kappa;
  if (delta <= 0)
    delta = lambda*kappa;
    lambda = lambda - gamma/kappa;
  end
  alpha = - mu/delta;
  
  % Calculate the comparison ratio.
  xnew = x + alpha*d;
  fnew = feval(fun, xnew);
  if nargin>4
    output.funcCount=output.funcCount+1;
  end
  iter = iter + 1;
  Delta = 2*(fnew - fold)/(alpha*mu);
  if (Delta  >= 0)
    success = 1;
    nsuccess = nsuccess + 1;
    x = xnew;
    fnow = fnew;
  else
    success = 0;
    fnow = fold;
  end

  if nargout >= 4
    % Store relevant variables
    output.f(j) = fnow;               % Current function value
    output.x(j,:) = x;      % Current position
  end    
  if display >= 3
    fprintf(1, 'Iter %4d  f(x) %10.5f  Scale %1.1e\n', j, fnow, lambda);
  end
  
  if (success == 1)
    
    % Test for termination
    if max(abs(alpha*d)) < tolx
      if (display >= 2)
        disp('TolX reached')
      end
      if nargin <5
        fval=fnew;
      else
        [fval,grad]=feval(fun, x);
        if nargin>4
          output.funcCount=output.funcCount+1;
        end
      end
      exitflag=2;
      return
      
    elseif max(abs(fnew-fold)) < tolfun
      if (display >= 2)
        disp('TolFun reached')
      end
      if nargin <5
        fval=fnew;
      else
        [fval,grad]=feval(fun, x);
        if nargin>4
          output.funcCount=output.funcCount+1;
        end
      end
      exitflag=3;
      return

    else
      % Update variables for new position
      fold = fnew;
      gradold = gradnew;
      [fval,gradnew] = feval(fun, x);
      if nargin>4
        output.funcCount=output.funcCount+1;
      end
      % If the gradient is zero then we are done.
      if (gradnew*gradnew' < eps)
        if (display >= 2)
          disp('Gradient smaller than eps');
        end
        grad=gradnew;
        exitflag=1;
        return
      end
    end
  end

  % Adjust lambda according to comparison ratio.
  if (Delta < 0.25)
    lambda = min(4.0*lambda, lambdamax);
  end
  if (Delta > 0.75)
    lambda = max(0.5*lambda, lambdamin);
  end

  % Update search direction using Polak-Ribiere formula, or re-start 
  % in direction of negative gradient after nparams steps.
  if (nsuccess == nparams)
    d = -gradnew;
    nsuccess = 0;
  else
    if (success == 1)
      beta = (gradold - gradnew)*gradnew'/(mu);
      d = beta*d - gradnew;
    end
  end
  j = j + 1;
  output.iterations=j;
end

% If we get here, then we haven't terminated in the given number of 
% iterations.
exitflag=0;
if (display >= 1)
  disp('Warning: Maximum number of iterations has been exceeded');
end
grad=gradnew;
fval=fnew;
