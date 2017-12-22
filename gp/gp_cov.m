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
% Copyright (c) 2007-2017 Jarno Vanhatalo
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

C = sparse(n,n2);
if nargin < 4 || isempty(predcf)
    predcf = 1:ncf;
end
% derivative observations in use
if isfield(gp,'deriv') && gp.deriv  % derivative observations in use
    ind_Ddim = x1(:,gp.deriv);
    ind_Ddim_derivs = ind_Ddim(ind_Ddim>0);
    uDdim = unique(ind_Ddim_derivs);
    ind_Ddim2 = x2(:,gp.deriv);
    ind_Ddim_derivs2 = ind_Ddim2(ind_Ddim2>0);
    uDdim2 = unique(ind_Ddim_derivs2);
    x1 = x1(:,setdiff(1:m,gp.deriv));   % Take only the non-index columns
    x2 = x2(:,setdiff(1:m,gp.deriv));   % Take only the non-index columns
    if any(strcmp(gp.type,{'FIC' 'PIC' 'PIC_BLOCK' 'CS+FIC' 'VAR' 'DTC' 'SOR'}))
        error('derivative observations have not been implemented for sparse GPs')
    end
end
% Loop through the covariance functions
for i=1:length(predcf)
    gpcf = gp.cf{predcf(i)};
    if isfield(gp.lik, 'int_magnitude') && gp.lik.int_magnitude
        if ~isfield(gp,'comp_cf') || (isfield(gp,'comp_cf') && sum(gp.comp_cf{1}==predcf(i)))
            gpcf.magnSigma2=1;
        end
    end
    % derivative observations in use
    if isfield(gp,'deriv') && gp.deriv
        
        if (~isfield(gpcf, 'selectedVariables') || any(ismember(gpcf.selectedVariables,uDdim)) || any(ismember(gpcf.selectedVariables,uDdim2)))
            % !!! Note. The check whether to calculate derivative
            % matrices for a covariance function could/should be made
            % nicer
            Ktemp = sparse(n,n2);
            % the block of covariance matrix
            if sum(ind_Ddim==0)>0 &&  sum(ind_Ddim2==0)>0
                % non-derivative observation non-derivative prediction
                Ktemp(ind_Ddim==0,ind_Ddim2==0) = gpcf.fh.cov(gpcf, x1(ind_Ddim==0,:), x2(ind_Ddim2==0,:));
            end
            % non-derivative observation, derivative prediction
            if sum(ind_Ddim==0)>0
                for u2 = 1:length(uDdim2)
                    Kdf = gpcf.fh.ginput4(gpcf, x2(ind_Ddim2==uDdim2(u2),:), x1(ind_Ddim==0,:), uDdim2(u2));
                    Ktemp(ind_Ddim==0,ind_Ddim2==uDdim2(u2)) = Kdf{1}';
                end
            end
            % Derivative observation non-derivative prediction
            if sum(ind_Ddim2==0)>0                
                for u1 = 1:length(uDdim)
                    Kdf = gpcf.fh.ginput4(gpcf, x1(ind_Ddim==uDdim(u1),:), x2(ind_Ddim2==0,:), uDdim(u1));
                    Ktemp(ind_Ddim==uDdim(u1),ind_Ddim2==0) = Kdf{1};
                end
            end
            % Derivative observation, derivative prediction
            for u1 = 1:length(uDdim)
                for u2=1:length(uDdim2)
                    if uDdim(u1) == uDdim2(u2)
                        Kdf2 = gpcf.fh.ginput2(gpcf, x1(ind_Ddim==uDdim(u1),:) ,x2(ind_Ddim2==uDdim2(u2),:), uDdim(u1));
                    else
                        Kdf2 = gpcf.fh.ginput3(gpcf, x1(ind_Ddim==uDdim(u1),:) ,x2(ind_Ddim2==uDdim2(u2),:), uDdim(u1), uDdim2(u2));
                    end
                    Ktemp(ind_Ddim==uDdim(u1),ind_Ddim2==uDdim2(u2)) = Kdf2{1};
                end
                
            end
        else
            Ktemp = zeros(n,n2);
            Ktemp(ind_Ddim==0,ind_Ddim2==0) = gpcf.fh.cov(gpcf, x1(ind_Ddim==0,:),x2(ind_Ddim2==0,:));
        end
        C = C+ Ktemp;
    else
        % Regular GP without derivative observations
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
