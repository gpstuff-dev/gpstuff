function [Ef, Varf] = la_post(gp, tx, ty, ns)
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
        [e, edata, eprior, f, L, La2, b, W] = gpla_e(gp_pak(gp, 'hyper'), gp, xx, yy, 'hyper');


        if nargin == 3
            Ef = f;
            if nargout == 2
                Lahat = 1./(1./La2 + W); 
                A = eye(m,m) - (L'.*repmat(Lahat',1,m))*L; A = (A+A')./2;
                L2 = iLaKfu/chol(A);
                Varf = Lahat + sum(L2.^2,1);
            end
        elseif nargin > 3
            Ef = f;
            Lahat = 1./(1./La2 + W); 
            A = eye(m,m) - (L'.*repmat(Lahat',1,m))*L; A = (A+A')./2;
            L2 = iLaKfu/chol(A);
                
            S = (repmat(Ef,1,ns) + chol(diag(Lahat)+L2*L2')'*randn(legth(ty),ns);
        end
    case 'PIC_BLOCK'
        ind = gp.tr_index;
        m = gp.nind;
        
        [e, edata, eprior, f, L, La2, b, W] = gpla_e(gp_pak(gp, 'hyper'), gp, xx, yy, 'hyper');

        if nargin == 3
            Ef = f;
            if nargout == 2
                B = zeros(size(L'));
                for i = 1:length(ind)
                    Lahat = inv(inv(La2{i}) + diag(W(ind{i}))); 
                    B = L'*Lahat;
                    Varf(ind{i}) = diag(Lahat);
                end
                A = eye(m,m) - (B*L; A = (A+A')./2;
                L2 = iLaKfu/chol(A);
                Varf = Varf + sum(L2.^2,1);
            end
        elseif nargin > 3
            Ef = f;
            B = zeros(size(L'));
            for i = 1:length(ind)
                Lahat = inv(inv(La2{i}) + diag(W(ind{i}))); 
                B = L'*Lahat;
                Sigma(ind{i}, ind{i}) = Lahat;
            end
            A = eye(m,m) - (B*L; A = (A+A')./2;
            L2 = iLaKfu/chol(A);
            
            S = (repmat(Ef,1,ns) + chol((Sigma+L2*L2'))'*randn(legth(ty),ns);
        end
end
end