function [Ef, Varf, S] = ep_post(gp, x, y, ns)
%EP_POST	Posterior distribution from Gaussian Process EP
%
%	Description
%   Ef = EP_POST(GP, X, Y) takes a gp data structure GP together with a
%	    matrix X of input vectors and vector Y of targets, and returns the EP
%       solution for the latent value posterior Ef = mean(f|Y,X). Each row of X
%       corresponds to one input vector and each row of Y corresponds to one output
%       vector.
%
%	[Ef, Varf] = EP_POST(GP, X, Y, X) returns also posterior marginal
%       variances
%
%   [S] = EP_POST(GP, X, Y, X, NS) returns NS sample vectors of latent
%       values from their posterior
%
%   BUGS: the sparse sampling is not actually sparse.
%
%	See also
%	GP, GP_PAK, GP_UNPAK, EP_PRED
%
% Copyright (c) 2008 Jarno Vanhatalo

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.

switch gp.type
    case 'FULL'

        error('The function is not implemented for FULL GP yet! \n')

    case 'FIC'

        [e, edata, eprior, site_tau, site_nu, L, La2, b, D, R, P] = gpep_e(gp_pak(gp, 'hyper'), gp, x, y, 'hyper');


        eta = D.*site_nu;
        gamma = R'*(R*(P'*site_nu));
        Ef = eta + P*gamma;
        Ef = b';
        Varf = D + sum((P*R').^2,2);
        if nargin > 3
            eta = D.*site_nu;
            gamma = R'*(R*(P'*site_nu));
            Ef = eta + P*gamma;
            S = (repmat(Ef,1,ns) + chol((diag(D)+P*R'*R*P'))'*randn(length(y),ns))';
        end
    case 'PIC_BLOCK'
        ind = gp.tr_index;

        [e, edata, eprior, site_tau, site_nu, L, La2, b, D, R, P] = gpep_e(gp_pak(gp, 'hyper'), gp, x, y, 'hyper');

        if nargin == 3
            eta = zeros(size(site_nu));
            Varf = zeros(size(site_nu));
            for i=1:length(ind)
                eta(ind{i}) = D{i}*site_nu(ind{i});
                Varf(ind{i}) = diag(D{i});
            end
            gamma = R'*(R*(P'*site_nu));
            Ef = eta + P*gamma;
            if nargout == 2
                Varf = Varf + sum((P*R').^2,2);
            end
        elseif nargin > 3
            for i=1:length(ind)
                eta(ind{i}) = D{i}*site_nu(ind{i});
                Sigma(ind{i}, ind{i}) = D{i};
            end
            gamma = R'*(R*(P'*site_nu));
            Ef = eta + P*gamma;
            S = (repmat(Ef,1,ns) + chol((Sigma+P*R'*R*P'))'*randn(length(y),ns))';
        end

end
end