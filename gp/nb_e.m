% Function for evaluating the energy of negative binomial
% likelihood with respect to dispersion parameter r.
%
% Description to be added later.
%
% Copyright (c) Jouni Hartikainen, 2008
function e = nb_e(r, gp, x, z, y, param, varargin)    
    E=gp.avgE(:);
    mu = exp(z).*E;
    e = sum(gammaln(r)-y.*(log(mu)-log(r+mu))-gammaln(r+y)-r.*(log(r)-log(r+mu)));
    