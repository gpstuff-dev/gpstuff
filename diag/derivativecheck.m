function delta  = derivativecheck(w, fun)
%DERIVATIVECHECK Compare user-supplied derivatives to
%                finite-differencing derivatives.
%
%  Description
%    This function is intended as a utility to check whether a
%    gradient calculation has been correctly implemented for a
%    given function. 
%
%    DERIVATIVECHECK(X, FUN) checks how accurate the user-supplied
%    derivatives of the function FUN are at X. FUN accepts input X
%    and returns a scalar function value F and its scalar or vector
%    gradient G evaluated at X A central difference formula with
%    step size 1.0e-6 is used, and the results for both gradient
%    function and finite difference approximation are printed.
%
%    DELTA=DERIVATIVECHECK(X, FUN) returns the delta between the
%    user-supllied and finite difference derivatives.
%

% Copyright (c) Christopher M Bishop, Ian T Nabney (1996, 1997)
% Copyright (c) 2010 Aki Vehtari

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

% Reasonable value for step size
epsilon = 1.0e-5;

% Treat
nparams = length(w);
deltaf = zeros(1, nparams);
step0 = zeros(1, nparams);

for i = 1:nparams
  % Move a small way in the ith coordinate of w
  step = step0;
  step(i) = 1;
  fplus  = fun(w+epsilon.*step);
  fminus = fun(w-epsilon.*step);
  % Use central difference formula for approximation
  deltaf(i) = 0.5*(fplus - fminus)/epsilon;
end
[~,gradient] = fun(w);
fprintf(1, 'Checking gradient ...\n\n');
fprintf(1, '   analytic   diffs     delta\n\n');
disp([gradient', deltaf', gradient' - deltaf'])
delta = gradient' - deltaf';
