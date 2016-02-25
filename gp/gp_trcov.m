function [K, C] = gp_trcov(gp, x1, predcf)
%GP_TRCOV  Evaluate training covariance matrix (gp_cov + noise covariance). 
%
%  Description
%    K = GP_TRCOV(GP, TX, PREDCF) takes in Gaussian process GP and
%    matrix TX that contains training input vectors to GP. Returns
%    (noiseless) covariance matrix K for latent values, which is
%    formed as a sum of the covariance matrices from covariance
%    functions in gp.cf array. Every element ij of K contains
%    covariance between inputs i and j in TX. PREDCF is an array
%    specifying the indexes of covariance functions, which are used
%    for forming the matrix. If not given, the matrix is formed
%    with all functions.
%
%    [K, C] = GP_TRCOV(GP, TX, PREDCF) returns also the (noisy)
%    covariance matrix C for observations y, which is sum of K and
%    diagonal term, for example, from Gaussian noise.
%
%  See also
%    GP_SET, GPCF_*
%
% Copyright (c) 2006-2010 Jarno Vanhatalo
% Copyright (c) 2010 Tuomas Nikoskinen

% This software is distributed under the GNU General Public 
% License (version 3 or later); please refer to the file 
% License.txt, included with the software, for details.

% no covariance functions?
if length(gp.cf)==0 || (nargin>2 && ~isempty(predcf) && predcf(1)==0) ...
        || isfield(gp, 'lik_mono')
    K=[];
    C=[];
    if nargout>1 && isfield(gp.lik.fh,'trcov')
        C=sparse(0);
        % Add Gaussian noise to the covariance
        C = C + gp.lik.fh.trcov(gp.lik, x1);
        if ~isempty(gp.jitterSigma2)
            C=C+gp.jitterSigma2;
        end
    end
    return
end

switch gp.type
    case {'FULL' 'FIC' 'PIC' 'PIC_BLOCK' 'CS+FIC' 'VAR' 'DTC' 'SOR'}
        [n,m]=size(x1);
        ncf = length(gp.cf);
        
        % Evaluate the covariance without noise
        K = sparse(0);
        if isfield(gp,'derivobs') && gp.derivobs  % derivative observations in use
            if any(strcmp(gp.type,{'FIC' 'PIC' 'PIC_BLOCK' 'CS+FIC' 'VAR' 'DTC' 'SOR'}))
                error('derivative observations have not been implemented for sparse GPs')
            end
        end
        % check whether predcf is used
        if nargin < 3 || isempty(predcf)
            predcf = 1:ncf;
        end
        % loop through covariance functions
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
                    Kff = gpcf.fh.trcov(gpcf, x1);
                    Gset = gpcf.fh.ginput4(gpcf, x1);
                    D = gpcf.fh.ginput2(gpcf, x1, x1);
                    
                    Kdf=Gset{1};
                    Kfd = Kdf';
                    Kdd=D{1};
                    
                    % Add all the matrices into a one K matrix
                    K = K + [Kff Kfd; Kdf Kdd];
                else
                    Kff = gpcf.fh.trcov(gpcf, x1);
                    G= gpcf.fh.ginput4(gpcf, x1);
                    D= gpcf.fh.ginput2(gpcf, x1, x1);
                    Kdf2 = gpcf.fh.ginput3(gpcf, x1 ,x1);
                    
                    Kdf=cat(1,G{1:m});
                    
                    % Now build up Kdd m*n x m*n matrix, which contains all the
                    % both partial derivative" -matrices
                    Kdd=blkdiag(D{1:m});
                    
                    % Gather non-diagonal matrices to Kddnodi
                    if m==2
                        Kddnodi=[zeros(n,n) Kdf2{1};Kdf2{1} zeros(n,n)];
                    else
                        t1=1;
                        Kddnodi=zeros(m*n,m*n);
                        for im=1:m-1
                            aa=zeros(m-1,m);
                            t2=t1+m-2-(im-1);
                            aa(m-1,im)=1;
                            k=kron(aa,cat(1,zeros((im)*n,n),Kdf2{t1:t2}));
                            k(1:n*m,:)=[];
                            k=k+k';
                            Kddnodi = Kddnodi + k;
                            t1=t2+1;
                        end
                    end
                    % Sum the diag + no diag matrices
                    Kdd=Kdd+Kddnodi;
                    Kfd=Kdf';
                    
                    % Gather all the matrices into one final matrix K which is the
                    % training covariance matrix
                    K = K + [Kff Kfd; Kdf Kdd];
                end
            else
                % Regular GP without derivative observations
                K = K + gpcf.fh.trcov(gpcf, x1);
            end
        end
        n = size(K,1);
        n1 = n+1;
        if ~isempty(gp.jitterSigma2)
            if issparse(K)
                K = K + sparse(1:n,1:n,gp.jitterSigma2,n,n);
            else
                K(1:n1:end)=K(1:n1:end) + gp.jitterSigma2;
            end
        end
        if nargout>1
            C=K;
            if isfield(gp.lik.fh,'trcov')
                % Add Gaussian noise to the covariance
                if isfield(gp,'derivobs') && gp.derivobs  % derivative observations in use
                    % same noise for obs and grad obs
                    C = C + gp.lik.fh.trcov(gp.lik, repmat(x1,m+1,1));
                else
                    C = C + gp.lik.fh.trcov(gp.lik, x1);
                end
                
            end
        end
        
end

