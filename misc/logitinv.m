function u = logitinv(v)
%LOGITINV Inverse of the logit transformation
%
%  Description
%    U = LOGITINV(V) computes the inverse of the logit
%    transformation of U
%
%      U = 1./(1 + EXP(V))
%  
%  See also LOGIT

% Copyright (c) 2011 Aki Vehtari

maxcut = -log(eps);
mincut = -log(1/realmin - 1);
u = 1 ./ (1 + exp(-max(min(v,maxcut),mincut)));
