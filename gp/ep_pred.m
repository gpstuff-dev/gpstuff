function [Ef, Varf, p1] = ep_pred(gp, tx, ty, x, varargin)
%EP_PRED	Predictions with Gaussian Process EP
%
%	Description
%	Y = EP_PRED(GP, TX, TY, X) takes a gp data structure GP together with a
%	matrix X of input vectors, Matrix TX of training inputs and vector TY of 
%       training targets, and forward propagates the inputs through the gp to generate 
%       a matrix Y of (noiseless) output vectors (mean(Y|X)). Each row of X 
%       corresponds to one input vector and each row of Y corresponds to one output 
%       vector.
%
%	Y = EP_PRED(GP, TX, TY, X, U) in case of sparse model takes also inducing 
%       points U.
%
%	[Y, VarY] = EP_PRED(GP, TX, TY, X) returns also the variances of Y 
%       (1xn vector).
%
%       BUGS: - only 1 output allowed
%
%	See also
%	GP, GP_PAK, GP_UNPAK
%
% Copyright (c) 2006 Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

    [tn, tnin] = size(tx);
    
    switch gp.type
      case 'FULL'
        [K, C]=gp_trcov(gp,tx);
        
        tautilde = gp.site_tau';
        nutilde = gp.site_nu';
        sqrttautilde = sqrt(tautilde);
        Stildesqroot=diag(sqrttautilde);
        
        B=eye(tn)+Stildesqroot*C*Stildesqroot;
        L=chol(B);
        z=Stildesqroot*(L'\(L\(Stildesqroot*(C*nutilde))));
        
        kstarstar=gp_trvar(gp, x);

        ntest=size(x,1);
        pistar=zeros(ntest,1);
        
        K_nf=gp_cov(gp,x,tx);
        apu = L'\Stildesqroot;
        for i1=1:ntest
            % Compute covariance between observations
            Ef(i1,1)=K_nf(i1,:)*(nutilde-z);
            v=apu*(K_nf(i1,:)');
            Varf(i1,1)=kstarstar(i1)-v'*v;
            p1(i1,1)=normcdf(Ef(i1,1)/sqrt(1+Varf(i1))); % Probability p(y_new=1)
        end
        
      case 'FIC'
      
      case 'PIC_BLOCK'
        
    end
end
