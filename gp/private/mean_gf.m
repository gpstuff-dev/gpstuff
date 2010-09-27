function [dMNM trA dyKy dyCy trAv] = mean_gf(gp,x,Ky,invKy,DKff,Stildesqroot,y,latent_method)
% MEAN_GF       Calculates help terms needed in gradient calculation with mean function
%
%     Description
%	[dMNM trA dyKy dyCy trAv] = MEAN_GF(gp,x,C,invC,DKff,Stildesqroot,y,latent_method) takes in
%                               following variables:
%
%        gp      - a gp data structure
%        x       - training inputs
%        Ky       - gaussian: K(x,x) + sigmaI
%        invKy    - gaussian: inv(Ky), EP: inv(Ky + S^-1)*S^-1
%        DKff    - d Ky / d th
%        Stildesqroot    - with EP, sqrt( diag(tautilde) )
%        y       - targets,gaussian: noisy latent values, EP: S*mutilde
%        latent_method   - which latent method in use ('gaussian','EP','laplace')


%        Returns a help terms dMNM and trA with non-vague prior and
%        dyKy, dyCy and trAv with vague prior. See (Rasmussen and Williams 2006)
%        and GPstuff doc for further explaining

%        dMNM = d M'*inv(N)*M / d th
%        trA  = d log|A| / dth
%        dyKy = d y'*Ky*y/ d th
%        dyCy = d y'*C*y / d th
%        trAv = d log|Av|/ d th


        dMNM = cell(1,length(DKff));
        trA  = cell(1,length(DKff));
        dyKy = cell(1,length(DKff));
        dyCy = cell(1,length(DKff));
        trAv = cell(1,length(DKff));
        Hapu = cell(1,length(gp.mean.meanFuncs));
        
        % Gather the basis functions' values in one matrix H 
        for i=1:length(gp.mean.meanFuncs)
            Hapu{i}=feval(gp.mean.meanFuncs{i},x);
        end
        H = cat(1,Hapu{1:end});
        
        % prior assumption for weights, w ~ N(b,B) 
        % b_m = prior mean for weights, B_m prior covariance matrix for weights
        b_m = gp.mean.p.b';            
        Bvec = gp.mean.p.B;
        B_m = reshape(Bvec,sqrt(length(Bvec)),sqrt(length(Bvec)));
        
        % help arguments
        HinvC = H*invKy;           
        N = Ky + H'*B_m*H;                                   
        
        % is prior for weights of mean functions vague
        if gp.mean.p.vague==0   % non-vague prior
            
            % help arguments that don't depend on DKff; non-vague p
            if isequal(latent_method,'gaussian')
                % help arguments with gaussian latent method
                M = H'*b_m-y;
                invN = N\eye(size(N));
                HKH = HinvC*H';
                A = B_m\eye(size(B_m)) + HKH;
                invAt=A\eye(size(A));
                invAt=invAt';
                
                % Calculate the arguments which are to be returned
                for i2 = 1:length(DKff)
                    dA = -1*HinvC*DKff{i2}*HinvC';                  % d A / d th
                    trA{i2} = sum(invAt(:).*dA(:));                 % d log(|A|) / dth 
                    dMNM{i2} = M'*(invN*DKff{i2}*invN*M);           % d M'*N*M / d th   
                end
                
                
            elseif isequal(latent_method,'EP')
                % help arguments with EP latent method
                S=Stildesqroot.^2;
                M = S*H'*b_m-y;                                     % M is now (S*H'b_m - S*mutilde)
                HKH = HinvC*S*H';                                   % inv(Ky + S^-1)*S^-1*S*H'
                A = B_m\eye(size(B_m)) + HKH;
                invAt=A\eye(size(A));
                invAt=invAt';
                B_h = eye(size(N)) + Stildesqroot*N*Stildesqroot;
                L_m=chol(B_h,'lower');
                zz=Stildesqroot*(L_m'\(L_m\(Stildesqroot*N)));
                invN = eye(size(zz))-zz;                            % inv(Ky + H'*B_m*H + S^-1)*S^-1
                
                % Calculate the arguments which are to be returned
                for i2=1:length(DKff)
                    dA = -1*HinvC*S*DKff{i2}*(invKy*S*H');          % d A / d th
                    trA{i2} = sum(invAt(:).*dA(:));                 % d log(|A|) / dth
                    dMNM{i2} = M'*(S^(-1)*invN*S*DKff{i2}*invN*M);  % with EP the d M'*N*M / d th
                end
                
                
            elseif isequal(latent_method,'laplace') 
                error('latent method = laplace not implemented yet')
            else
                error('no correct latent method specified')
            end


        else  % vague prior
            
            if isequal(latent_method,'gaussian')
                % help arguments that don't depend on DKff; vague p
                HKH = HinvC*H';
                A     = HKH;
                AH    = A\H;
                invAt = A\eye(size(A));
                invAt = invAt';
                G     = H'*AH*invKy*y;
                b     = invKy*y;

                for i2 = 1:length(DKff)
                    % help arguments that depend on DKff; vague p
                    dyKy{i2} = b'*(DKff{i2}*b);            % d y'*Ky⁻*y / d th
                    dA  = -1*HinvC*DKff{i2}*HinvC';        % d A / d th  
                    trAv{i2} = sum(invAt(:).*dA(:));       % d log(|A|)/dth = trace(inv(A) * dA/dth)
                    P   = invKy*DKff{i2}*invKy;

                    dyCy1 = y'*P*G;           
                    dyCy3 = -G'*P*G;
                    dyCy{i2} = 2*dyCy1 + dyCy3;          % d y'*C*y /d th
                end
            else
                error('vague prior only for gaussian latent method at the moment')
            end
        end
        
        
end