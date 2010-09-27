function [RB RAR] = mean_predf(gp,x,xt,K_nf,L,Ksy,latent_method,S)
% MEAN_PREDF       Calculates help terms needed in prediction with mean function
%
%     Description
%	[RB, RAR] = MEAN_PREDF(gp,x,xt,K_nf,L,Ksy,latent_method,S) takes in
%                following variables:
%        gp   - a gp data structure
%        x    - training inputs
%        xt   - test inputs
%        K_nf - covariance matrix K(x,xt)
%        L    - gaussian: chol (K(x,x) + sigmaI).EP: inv(K(x,x) + S^-1)*S^-1 
%        S    - with EP, diag(tautilde) = Stilde
%        Ksy  - gaussian: L'\(L\y) = inv(C)*y    EP: inv(K + S^-1)*S^-1*nutilde  
%        latent_method - which latent method in use ('gaussian','EP','laplace')


%        Returns a help term RB to the posterior mean EF and help term RAR 
%        for posterior variance VARF of latent variables. 
%        From (Rasmussen and Williams 2006) page 28,  
%
%        RB  = R'*Beta = R'*(inv(B1)*B2) 
%        RAR = R'*inv(B1)*R

    % Gather the basis functions' values in one matrix H for training inputs
    % and Hs for test inputs
    Hapu = cell(1,length(gp.mean.meanFuncs));
    Hapu2 = cell(1,length(gp.mean.meanFuncs));
    for i=1:length(gp.mean.meanFuncs)
        Hapu{i}=feval(gp.mean.meanFuncs{i},x);
        Hapu2{i}=feval(gp.mean.meanFuncs{i},xt);
    end
    H = cat(1,Hapu{1:end});
    Hs= cat(1,Hapu2{1:end});
    
    % prior assumption for weights, w ~ N(b,B) 
    % b = prior mean for weights, B prior covariance matrix for weights
    b = gp.mean.p.b';            
    Bvec = gp.mean.p.B;
    B = reshape(Bvec,sqrt(length(Bvec)),sqrt(length(Bvec)));

    if isequal(latent_method,'gaussian')
        KsK = L'\(L\K_nf);                       % inv(C)*K(x,xt)
        KsH = L'\(L\H');                         % inv(C)*H)
    elseif isequal(latent_method,'EP')
        KsK = L*(S*K_nf);                        % inv(K + S^-1)*S^-1*(S*K(x,xt)) 
        KsH = L*(S*H');                          % inv(K + S^-1)*S^-1*(S*H')
    elseif isequal(latent_method,'laplace') 
        error('latent method = laplace not implemented yet')
    else
        error('no correct latent method specified')
    end
        
    R = Hs - H*KsK;

    if gp.mean.p.vague==0         % non-vague prior
        invB = B\eye(size(B));
        B1 = invB + H*KsH;
        B2 = H*Ksy + invB*b;
    else                          % vague prior
        B1 = H*KsH;
        B2 = H*Ksy;
    end
    
    Beta = B1\B2;
    invAR=B1\R;
    RARapu=R'.*invAR';          % Calculate only the necessary.
    
    
    RB  = R'*Beta;                % For predictive mean
    RAR = sum(RARapu,2);          % For predictive variance
    
end