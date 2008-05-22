function [Ef, Varf, S] = la_post(gp, tx, ty, ns, param)
%LA_POST	Posterior distribution from Gaussian Process Laplace
%           approximation
%
%	Description
%   Ef = LA_POST(GP, X, Y) takes a gp data structure GP together with a
%	    matrix X of input vectors and vector Y of targets, and returns the
%	    Laplace approximation for the latent value posterior Ef = mean(f|Y,X).
%       Each row of X corresponds to one input vector and each row of Y
%       corresponds to one output vector.
%
%	[Ef, Varf] = LA_POST(GP, TX, TY, X) returns also posterior marginal
%       variances
%
%   [S] = LA_POST(GP, TX, TY, X, NS) returns NS sample vectors of latent
%       values from their posterior
%
%   BUGS: the sparse sampling is not actually sparse.
%
%	See also
%	GP, GP_PAK, GP_UNPAK, LA_PRED
%
% Copyright (c) 2008 Jarno Vanhatalo

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.

[tn, tnin] = size(tx);

switch gp.type
    case 'FULL'

        error('The function is not implemented for FULL GP yet! \n')

    case 'FIC'
        m = gp.nind;
        [e, edata, eprior, f, L, La2, b, W] = gpla_e(gp_pak(gp, param), gp, tx, ty, param);


        Ef = f;
        Lahat = 1./(1./La2 + W);
        A = eye(m,m) - (L'.*repmat(Lahat',m,1))*L; A = (A+A')./2;
        L2 = repmat(Lahat,1,m).*L/chol(A);
        Varf = Lahat + sum(L2.^2,2);
        if nargin > 3
            Lahat = 1./(1./La2 + W);
            A = eye(m,m) - (L'.*repmat(Lahat',m,1))*L; A = (A+A')./2;
            L2 = repmat(Lahat,1,m).*L/chol(A);

            S = (repmat(Ef,1,ns) + chol(diag(Lahat)+L2*L2')'*randn(length(ty),ns))';
        end
    case 'PIC_BLOCK'
        ind = gp.tr_index;
        m = gp.nind;

        [e, edata, eprior, f, L, La2, b, W] = gpla_e(gp_pak(gp, param), gp, tx, ty, param);

        if nargin == 3
            Ef = f;
            if nargout == 2
                B = zeros(size(L));
                Varf = zeros(size(ty));
                for i = 1:length(ind)
                    Lahat = inv(inv(La2{i}) + diag(W(ind{i})));
                    B(ind{i},:) = Lahat*L(ind{i},:);
                    Varf(ind{i}) = diag(Lahat);
                end
                A = eye(m,m) - L'*B; A = (A+A')./2;
                L2 = B/chol(A);
                Varf = Varf + sum(B.^2,2);
            end
        elseif nargin > 3
            Ef = f;
            B = zeros(size(L'));
            for i = 1:length(ind)
                Lahat = inv(inv(La2{i}) + diag(W(ind{i})));
                B = L'*Lahat;
                Sigma(ind{i}, ind{i}) = Lahat;
            end
            A = eye(m,m) - B*L; A = (A+A')./2;
            L2 = L/A;

            S = repmat(Ef,1,ns) + chol((Sigma+L2*B))'*randn(legth(ty),ns);
        end
    case 'CS+FIC'
        m = gp.nind;
        [e, edata, eprior, f, L, La2, b, W] = gpla_e(gp_pak(gp, 'hyper'), gp, tx, ty, 'hyper');

        if nargin == 3
            Ef = f;
            if nargout == 2
                W = sparse(1:n,1:n,W);
                La = W + W*La2*W;
                B1 = La\L;
                B2 = (L'*La2)/W;
                A = eye(m,m) - B2*B1; A = (A+A')./2;
                L2 = B1/A;
                Varf = Lahat + sum(B1.*B2',2);
            end
        elseif nargin > 3
            Ef = f;
            Lahat = 1./(1./La2 + W);
            A = eye(m,m) - (L'.*repmat(Lahat',1,m))*L; A = (A+A')./2;
            L2 = iLaKfu/chol(A);

            S = repmat(Ef,1,ns) + chol(diag(Lahat)+L2*L2')'*randn(legth(ty),ns);
        end
end
end