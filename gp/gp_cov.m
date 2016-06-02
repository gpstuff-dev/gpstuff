function [C, Cinv] = gp_cov(gp, x1, x2, predcf)
%GP_COV  Evaluate covariance matrix between two input vectors.
%
%  Description
%    C = GPCOV(GP, TX, X, PREDCF) takes in Gaussian process GP and
%    two matrixes TX and X that contain input vectors to GP.
%    Returns covariance matrix C. Every element ij of C contains
%    covariance between inputs i in TX and j in X. PREDCF is an
%    optional array specifying the indexes of covariance functions,
%    which are used for forming the matrix. If empty or not given,
%    the matrix is formed with all functions.
%
% Copyright (c) 2007-2010, 2016 Jarno Vanhatalo
% Copyright (c) 2010 Tuomas Nikoskinen

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

% Are gradient observations available; derivobs=1->yes, derivobs=0->no
Cinv=[];
%if ~(isfield(gp,'derivobs') && gp.derivobs)
ncf = length(gp.cf);
[n,m]=size(x1);
[n2,m2]=size(x2);

C = sparse(0);
if nargin < 4 || isempty(predcf)
    predcf = 1:ncf;
end
for i=1:length(predcf)
    gpcf = gp.cf{predcf(i)};
    if isfield(gp.lik, 'int_magnitude') && gp.lik.int_magnitude
        if ~isfield(gp,'comp_cf') || (isfield(gp,'comp_cf') && sum(gp.comp_cf{1}==predcf(i)))
            gpcf.magnSigma2=1;
        end
    end
    % derivative observations in use
    if isfield(gp,'derivobs') && gp.derivobs
        if m==1
            Kff = gpcf.fh.cov(gpcf, x1, x2);
            Kdf = gpcf.fh.ginput4(gpcf, x1, x2);
            Kdf=Kdf{1};
            Kfd = gpcf.fh.ginput4(gpcf, x2, x1); 
            Kfd=Kfd{1}';     % Notice, Kfd is calculated with lower left 
                             % parts. Hence, we need a transpose             
            Kdd = gpcf.fh.ginput2(gpcf, x1, x2);
            
            C = C + [Kff Kfd; Kdf Kdd{1}];
            
        else % Input dimension is >1            
            % the block of covariance matrix
            Kff = gpcf.fh.cov(gpcf, x1, x2);
            % the blocks on the left side, below Kff 
            Kdf = gpcf.fh.ginput4(gpcf, x1, x2);
            Kdf=cat(1,Kdf{1:m});
            % the blocks on the right side, next to Kff 
            Kfd = gpcf.fh.ginput4(gpcf, x2, x1);
            Kfd=cat(1,Kfd{1:m})';   % Notice, Kfd is calculated with lower 
                                    % left parts. Hence, we need a transpose 
            % the diagonal blocks of double derivatives
            D = gpcf.fh.ginput2(gpcf, x1, x2);
            % the off diagonal blocks of double derivatives on the
            % upper right corner. See e.g. gpcf_squared -> ginput3
            Kdf12 = gpcf.fh.ginput3(gpcf, x1 ,x2);
            Kdf21 = gpcf.fh.ginput3(gpcf, x2, x1);
            
            % Now build up Kdd m*n x m*n2 matrix, which contains all the
            % both partial derivative" -matrices
            
            % Add the diagonal matrices
            Kdd=blkdiag(D{1:m});
            % Add the non-diagonal matrices to Kdd
            ii3=0;
            for j=0:m-2
                for i=1+j:m-1
                    ii3=ii3+1;
                    Kdd(i*n+1:(i+1)*n,j*n2+1:j*n2+n2) = Kdf21{ii3}';  % down left 
                    % Notice, Kdf12 is calculated with upper right parts.
                    % Hence we need transpose above
                    Kdd(j*n+1:j*n+n,i*n2+1:(i+1)*n2) = Kdf12{ii3};    % up right
                end
            end
            
            % Gather all the matrices into one final matrix K which is the
            % training covariance matrix
            C = C + [Kff Kfd; Kdf Kdd];
        end
    else
        C = C + gpcf.fh.cov(gpcf, x1, x2);
    end
end
end


% % Below we provide example code to check the covariance matrices related
% % derivobs='on' case:
%
% mu1=[-1.5 -2.5]; Sigma1=[1 0.3; 0.3 1];
% mu2=[2 3];Sigma2=[3 0.5; 0.5 4];
% mu3=[0 0];Sigma3=[100 0; 0 100];
% fx = @(x) -log( (mvnpdf(x(:,1:2),mu1,Sigma1) + 0.3*mvnpdf(x(:,1:2),mu2,Sigma2)).*mvnpdf(x(:,1:2),mu3,Sigma3)) ./15 -1 + 0.1*x(:,3).^2;
% dfx = @(x) [(-1./( (mvnpdf(x(:,1:2),mu1,Sigma1) + 0.3*mvnpdf(x(:,1:2),mu2,Sigma2)).*mvnpdf(x(:,1:2),mu3,Sigma3))/15.*...
%     ( ( -mvnpdf(x(:,1:2),mu1,Sigma1).*(x(:,1:2)-mu1)/Sigma1 - 0.3*mvnpdf(x(:,1:2),mu2,Sigma2).*(x(:,1:2)-mu2)/Sigma2 ).*mvnpdf(x(:,1:2),mu3,Sigma3) -...
%      (mvnpdf(x(:,1:2),mu1,Sigma1) + 0.3*mvnpdf(x(:,1:2),mu2,Sigma2)).*mvnpdf(x(:,1:2),mu3,Sigma3).*(x(:,1:2)-mu3)/Sigma3  )) 0.2*x(:,3)];
% 
% x = 5-rand(2,3);
% y = fx(x);yg=[];
% for i1=1:size(x,1)
%     yg(i1,:)=dfx(x(i1,:));
% end
% 
% %cfl2 = gpcf_squared('coeffSigma2', [0.01 0.01 0.02 0.03 0.04 0.015], 'coeffSigma2_prior', prior_sqrtt(), 'interactions', 'on');
% cfl2 = gpcf_squared('coeffSigma2', 0.001, 'coeffSigma2_prior', prior_sqrtt(), 'interactions', 'on');
% gp = gp_set('cf', {cfl2}, 'derivobs', 'on', 'jitterSigma2',0); %cfc,cfl,,cfse
% 
% K = gp_trcov(gp,x);
% K2 = gp_cov(gp,x,x);
% min(min(K-K2))        % should return zero
% max(max(K-K2))        % should return zero
% gradcheck(x(:)',@teste,@testg,gp,x,'dC/dx1');
% gradcheck(x(:)',@teste,@testg,gp,x,'d2C/dx1dx2');
% 
% gradcheck(gp_pak(gp),@teste,@testg,gp,x,'dC/dx1_hyper');
% gradcheck(gp_pak(gp),@teste,@testg,gp,x,'d2C/dx1dx2_hyper');
% 
% gradcheck(gp_pak(gp),@gp_e,@gp_g,gp,x,[y;yg(:)]);
%
% --- clip ---
% 
% function e = teste(x,gp,x2,blockToTest)
% 
% [n,m]=size(x2);
% if isempty(strfind(blockToTest,'hyper'))
%     x = reshape(x,size(x2));
%     C = gp_cov(gp,x2,x);
%     % Notice, in gp_cov, C is calculate with lower left parts and upper right
%     % parts are filled in accordingly. Hence, the x that changes has to go to
%     % right hand side.
%     switch blockToTest
%         case 'dC/dx1'
%             e = sum(sum(C(1:n,1:n)));
%         case 'd2C/dx1dx2'
%             e = sum(sum(C(n+1:end,1:n)));
%     end
% else
%     gp = gp_unpak(gp,x);
%     C = gp_cov(gp,x2,x2);
%     switch blockToTest
%         case 'dC/dx1_hyper'
%             e = sum(sum(C(n+1:end,1:n)));
%         case 'd2C/dx1dx2_hyper'
%             e = sum(sum(C(n+1:end,n+1:end)));
%             %e = sum(sum(C(3:4,3:4)))+sum(sum(C(5:6,5:6)))+sum(sum(C(7:8,7:8)));
%     end
% end
%     
% end
% 
% 
% function g = testg(x,gp,x2,blockToTest)
% 
% [n,m]=size(x2);
% if isempty(strfind(blockToTest,'hyper'))
%     x = reshape(x,size(x2));
%     C = gp_cov(gp,x,x2);
%     % Notice, in gp_cov, C is calculate with lower left parts and upper right
%     % parts are filled in accordingly. Hence, the x that changes has to go to
%     % right hand side.
%     switch blockToTest
%         case 'dC/dx1'
%             g = C(1:n,n+1:end);
%             g = sum(g,1);
%         case 'd2C/dx1dx2'
%             g = C(n+1:end,n+1:end);
%             g = sum(g,1);
%     end
% else
%     if length(gp.cf) > 1
%         error('you need to test one covariance function at time')
%     end
%     gp = gp_unpak(gp,x);
%     gpcf = gp.cf{1};
%     DKffa = gpcf.fh.cfg(gpcf, x2);
%     DKdf = gpcf.fh.cfdg(gpcf, x2);
%     DKdd = gpcf.fh.cfdg2(gpcf, x2);
%     
%     np=length(DKffa);
%     for inp=1:np
%         C=[DKffa{inp} DKdf{inp}';DKdf{inp} DKdd{inp}];
%         switch blockToTest
%             case 'dC/dx1_hyper'
%                 g(inp) = sum(sum(C(n+1:end,1:n)));
%             case 'd2C/dx1dx2_hyper'
%                 g(inp) = sum(sum(C(n+1:end,n+1:end)));
%                 %g(inp) = sum(sum(C(3:4,3:4)))+sum(sum(C(5:6,5:6)))+sum(sum(C(7:8,7:8)));
%         end        
%     end
% end
% 
% end















