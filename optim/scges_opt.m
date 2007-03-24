function opt = scges_opt(opt)
%SCGES_OPT Default options for scaled conjugate gradient optimization
%
%   Default options for SCGES
%
%    opt=scges2_opt;
%      return default otions
%    opt=scges2_opt(scgesopt);
%      fill empty options with default values
%
%   The options and defaults are
%   display (0)
%     -1 to not display anything
%     0 to display just diagnostic messages
%     n positive integer to show also the function values and
%     validation values every nth iteration
%   checkgrad (0)
%     1 to check the user defined gradient function
%   maxiter (1000)
%     Maximum number of iterations
%   tolfun (1e-6)
%     termination tolerance on the function value
%   tolx (1e-6)
%     termination tolerance on X
%   maxfail (20)
%     maximum number of iterations validation is allowed to fail
%     if negative do not use early stopping
%
%	See also
%	SCGES

%	Copyright (c) Aki Vehtari (1998-2005)

if nargin < 1
  opt=[];
end

if ~isfield(opt,'display')
  opt.display=0;
end
if ~isfield(opt,'checkgrad')
  opt.checkgrad=0;
end
if ~isfield(opt,'maxiter') | opt.maxiter < 1
  opt.maxiter=100;
end
if ~isfield(opt,'tolfun') | opt.tolfun < 0
  opt.tolfun=1e-6;
end
if ~isfield(opt,'tolx') | opt.tolx < 0
  opt.tolx=1e-6;
end
if ~isfield(opt,'maxfail') 
  opt.maxfail=20;
end
